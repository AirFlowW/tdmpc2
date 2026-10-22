import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(cfg.device)
		self._capacity = cfg.buffer_size if cfg.get('pretrain_data_path') else min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		self._valid_episodes = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free = torch.cuda.mem_get_info(self._device)[0] if self._device.type == 'cuda' else 0
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = self.cfg.get('buffer_device', 'auto')
		if storage_device == 'auto':
			storage_device = str(self._device) if self._device.type == 'cuda' and 2.5*total_bytes < mem_free else 'cpu'
		if torch.device(storage_device).type not in {'cpu', 'cuda'}:
			raise ValueError('Replay storage must be cpu or cuda; MPS learners sample from CPU replay')
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._valid_episodes += num_new_eps
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		if len(td) < self.cfg.horizon + 1:
			return self._num_eps
		self._valid_episodes += 1
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		# MPS async copies can outlive temporary CPU batches and read freed storage.
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(
			self._device, non_blocking=self._device.type == 'cuda')
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	@property
	def ready(self):
		return self._valid_episodes > 0

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

	def chronological_indices(self):
		"""Physical indices of occupied rows, oldest first (including ring fragments)."""
		if not hasattr(self, '_buffer'):
			return torch.empty(0, dtype=torch.long)
		length = len(self._buffer)
		start = self._buffer.writer.state_dict()['_cursor'] if length == self.capacity else 0
		return (torch.arange(length) + start) % self.capacity

	def save_dataset(self, path, **kwargs):
		from common.replay_dataset import save_dataset
		return save_dataset(self, path, **kwargs)

	def load_dataset(self, path):
		"""Load all rows without discarding short fragments or changing field values."""
		from common.replay_dataset import iter_dataset_chunks, read_manifest
		if self.cfg.task != 'dot-avoidance':
			raise ValueError('Replay datasets currently support dot-avoidance only')
		if hasattr(self, '_buffer') and len(self._buffer):
			raise ValueError('Replay datasets must be loaded into an empty buffer')
		manifest = read_manifest(path, self.cfg, self.capacity)
		for td in iter_dataset_chunks(path, manifest):
			if not hasattr(self, '_buffer'):
				self._buffer = self._init(td)
			self._buffer.extend(td)
		self._num_eps = max(s['episode'] for s in manifest['segments']) + 1
		self._valid_episodes = sum(s['length'] >= self.cfg.horizon + 1 for s in manifest['segments'])
		return manifest

	def valid_window_starts(self):
		"""Enumerate every horizon+1 window in chronological coordinates."""
		from common.replay_dataset import episode_segments
		indices = self.chronological_indices()
		if not len(indices):
			return np.empty(0, dtype=np.int64)
		episodes = self._buffer.storage.get(slice(None))['episode'][indices].cpu().numpy()
		starts = [np.arange(s['start'], s['start'] + s['length'] - self.cfg.horizon, dtype=np.int64)
				  for s in episode_segments(episodes) if s['length'] > self.cfg.horizon]
		return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)

	def iter_epoch_batches(self, starts, rng):
		"""Visit each start once, padding only the final batch; yield (batch, padding)."""
		if not len(starts):
			raise ValueError('Replay dataset has no valid training windows')
		order = rng.permutation(starts)
		physical = self.chronological_indices()
		for offset in range(0, len(order), self.cfg.batch_size):
			batch_starts = order[offset:offset + self.cfg.batch_size]
			padding = self.cfg.batch_size - len(batch_starts)
			if padding:
				batch_starts = np.concatenate([batch_starts, rng.choice(starts, size=padding)])
			indices = torch.from_numpy(batch_starts[:, None] + np.arange(self.cfg.horizon + 1))
			td = self._buffer[physical[indices.reshape(-1)]].view(-1, self.cfg.horizon + 1).permute(1, 0)
			yield self._prepare_batch(td), padding
