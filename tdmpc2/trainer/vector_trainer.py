"""Parallel dot-world collection with one learner and trajectory-safe replay."""

from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from hydra.utils import to_absolute_path
from tqdm import tqdm

from common.device import mark_step
from environments.evaluation import TDMPCPolicy, evaluate_conditions, training_metrics
from trainer.base import Trainer


def transition(obs, action=None, reward=float('nan'), terminated=float('nan')):
    """Incoming action/reward belong to the observation in the same row."""
    return TensorDict(dict(
        obs=torch.as_tensor(np.array(obs, copy=True)).unsqueeze(0),
        action=(torch.full((2,), float('nan')) if action is None else torch.as_tensor(action, dtype=torch.float32).clone()).unsqueeze(0),
        reward=torch.tensor([reward], dtype=torch.float32),
        terminated=torch.tensor([terminated], dtype=torch.float32),
    ), batch_size=[1])


class VectorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = self._updates = 0
        self._start = perf_counter()
        self._collection_time = self._planning_time = self._update_time = 0.0
        self._last_metrics = {}
        self._action_rng = np.random.default_rng(self.cfg.seed + 100_000)
        self._tds = []
        self._pretrained = False
        self._pretrain_updates = 0
        self._pretrain_seconds = 0.0

    def metrics(self):
        elapsed = perf_counter() - self._start
        return dict(step=self._step, episode=self.buffer.num_eps, updates=self._updates,
                    pretrain_updates=self._pretrain_updates, pretrain_seconds=self._pretrain_seconds,
                    elapsed_time=elapsed, steps_per_second=self._step / max(elapsed, 1e-9),
                    collection_seconds=self._collection_time, planning_seconds=self._planning_time,
                    update_seconds=self._update_time,
                    collection_steps_per_second=self._step / max(self._collection_time, 1e-9))

    def eval(self):
        results = evaluate_conditions(
            self.cfg.dot, self.cfg.obs, self.cfg.frame_stack, TDMPCPolicy(self.agent),
            self.cfg.eval_episodes, self.cfg.seed + 1_000_000,
            video=self.logger.video, video_step=self._step)
        return training_metrics(results, self.cfg.dot.get('partial_obs', False))

    def _flush(self):
        for rows in self._tds:
            if len(rows) > 1:
                # Partial episodes retain terminated=False and their actual last observation.
                self.buffer.add(torch.cat(rows))
        self._tds = []

    def _pretrain(self):
        if not self.cfg.get('pretrain_data_path'):
            return
        epochs = self.cfg.get('pretrain_epochs', 1)
        if type(epochs) is not int or epochs < 1:
            raise ValueError('pretrain_epochs must be a positive integer')
        path = to_absolute_path(str(Path(self.cfg.pretrain_data_path).expanduser()))
        start = perf_counter()
        manifest = self.buffer.load_dataset(path)
        starts = self.buffer.valid_window_starts()
        print(f'Pretraining on {manifest["rows"]:,} replay rows, {len(starts):,} windows '
              f'for {epochs} epochs from {path}')
        rng = np.random.default_rng(self.cfg.seed + 200_000)
        windows = padding_total = 0
        for epoch in range(1, epochs + 1):
            batches = self.buffer.iter_epoch_batches(starts, rng)
            total = (len(starts) + self.cfg.batch_size - 1) // self.cfg.batch_size
            for batch_index, (batch, padding) in enumerate(tqdm(
                    batches, total=total, desc=f'Pretraining epoch {epoch}/{epochs}'), 1):
                metrics = self.agent.update_batch(batch)
                self._pretrain_updates += 1
                windows += self.cfg.batch_size - padding
                padding_total += padding
                self._pretrain_seconds = perf_counter() - start
                if self._pretrain_updates % 100 == 0 or batch_index == total:
                    self.logger.log(dict(metrics, epoch=epoch, iteration=self._pretrain_updates,
                                         updates=self._pretrain_updates, valid_windows=len(starts),
                                         windows=windows, padding=padding_total,
                                         elapsed_time=self._pretrain_seconds), 'pretrain')
        self._pretrained = True
        self.logger.save_agent(self.agent, identifier='pretrained')

    def train(self):
        cfg = self.cfg
        if cfg.steps < 1 or cfg.horizon < 1 or cfg.eval_freq < 1 or cfg.eval_episodes < 1:
            raise ValueError('steps, horizon, eval_freq, and eval_episodes must be positive')
        if cfg.dot['max_episode_steps'] < cfg.horizon:
            raise ValueError('Episode length must be at least the replay horizon')
        if cfg.buffer_size < cfg.horizon + 1 or (not cfg.get('pretrain_data_path') and cfg.steps < cfg.horizon + 1):
            raise ValueError('steps and buffer_size must hold at least horizon+1 replay rows')
        if not np.isfinite(cfg.updates_per_transition) or cfg.updates_per_transition < 0:
            raise ValueError('updates_per_transition must be finite and nonnegative')
        if cfg.seed_steps < 0:
            raise ValueError('seed_steps must be nonnegative')
        next_eval, credit = cfg.eval_freq, 0.0
        try:
            self._pretrain()
            obs, _ = self.env.reset(seed=[cfg.seed + i for i in range(cfg.num_envs)])
            self._tds = [[transition(o)] for o in obs]
            reset_mask = np.ones(cfg.num_envs, dtype=bool)
            self.logger.log(dict(self.eval(), **self.metrics()), 'eval')
            while self._step < cfg.steps:
                learning = (self._pretrained or self._step >= cfg.seed_steps) and self.buffer.ready
                start = perf_counter()
                mark_step(self.agent.device)
                actions = (self.agent.act_batch(obs, reset_mask).numpy() if learning else
                           self._action_rng.uniform(-1, 1, (cfg.num_envs, 2)).astype(np.float32))
                self._planning_time += perf_counter() - start
                start = perf_counter()
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                self._collection_time += perf_counter() - start
                dones = terminated | truncated
                self._step += cfg.num_envs
                finished = []
                for i in range(cfg.num_envs):
                    final_obs = infos['final_observation'][i] if dones[i] else next_obs[i]
                    self._tds[i].append(transition(final_obs, actions[i], rewards[i], terminated[i]))
                    if dones[i]:
                        info = infos['final_info'][i]
                        self.buffer.add(torch.cat(self._tds[i]))
                        finished.append(dict(episode_reward=sum(float(row['reward'][0]) for row in self._tds[i][1:]),
                                             collision_step_rate=info['collision_steps'] / info['elapsed_steps'],
                                             collision_free_episode_fraction=info['success'],
                                             episode_success=info['success'], episode_length=info['elapsed_steps']))
                        self._tds[i] = [transition(next_obs[i])]
                if learning:
                    credit += cfg.num_envs * cfg.updates_per_transition
                    while credit >= 1:
                        start = perf_counter()
                        self._last_metrics = self.agent.update(self.buffer)
                        self._update_time += perf_counter() - start
                        self._updates += 1
                        credit -= 1
                if finished:
                    summary = {key: np.mean([ep[key] for ep in finished]) for key in finished[0]}
                    self.logger.log(dict(self._last_metrics, **summary, **self.metrics()), 'train')
                # Entering learned control from warmup also needs fresh planner histories.
                reset_mask = dones if learning else np.ones(cfg.num_envs, dtype=bool)
                obs = next_obs
                if self._step >= next_eval:
                    self.logger.log(dict(self.eval(), **self.metrics()), 'eval')
                    self.logger.save_agent(self.agent, identifier=str(self._step))
                    next_eval = (self._step // cfg.eval_freq + 1) * cfg.eval_freq
            self._flush()
            self.logger.log(dict(self._last_metrics, **self.metrics()), 'train')
            self.logger.log(dict(self.eval(), **self.metrics()), 'eval')
            try:
                if cfg.get('save_replay_buffer', False):
                    path = cfg.work_dir / 'replay'
                    snapshot = self.buffer.save_dataset(path)
                    print(f'Saved {snapshot["rows"]:,} replay rows to {path}')
            finally:
                # An export failure must not prevent the final model from being saved.
                self.logger.finish(self.agent)
        finally:
            self._flush()
            self.env.close()
