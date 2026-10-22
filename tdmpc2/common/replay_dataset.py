"""Portable, lossless replay datasets: JSON metadata and bounded NumPy shards.

Only data arrays are serialized, never TorchRL storage, samplers, or Python objects.
RGB shards store an initial stack per contiguous episode fragment and new frames.
"""

import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import BadZipFile

import numpy as np
import torch
from tensordict import TensorDict


FORMAT = 'tdmpc2-replay'
VERSION = 1
CHUNK_BYTES = 64 * 1024 * 1024
FIELDS = ('obs', 'action', 'reward', 'terminated', 'episode')


def episode_segments(episodes):
    """Return chronological fragments, including fragments too short to sample."""
    starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    if not len(episodes):
        return []
    ends = np.r_[starts[1:], len(episodes)]
    return [dict(start=int(start), length=int(end - start), episode=int(episodes[start]))
            for start, end in zip(starts, ends)]


def _encode(arrays, rgb):
    obs = arrays['obs']
    if not rgb:
        return arrays, 'raw'
    segments = episode_segments(arrays['episode'])
    if any(not np.array_equal(obs[s['start'] + 1:s['start'] + s['length'], :-3],
                              obs[s['start']:s['start'] + s['length'] - 1, 3:])
           for s in segments):
        return arrays, 'raw'
    encoded = {key: value for key, value in arrays.items() if key != 'obs'}
    encoded['obs_initial'] = obs[[s['start'] for s in segments]]
    encoded['obs_frames'] = obs[:, -3:]
    return encoded, 'frame_stack'


def save_dataset(buffer, path, *, chunk_bytes=CHUNK_BYTES):
    """Publish a complete snapshot atomically; existing datasets are never replaced."""
    if buffer.cfg.task != 'dot-avoidance':
        raise ValueError('Replay datasets currently support dot-avoidance only')
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    path = Path(path).expanduser().absolute()
    if path.exists():
        raise FileExistsError(f'Replay dataset already exists: {path}; use a new exp_name')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{path.name}-', dir=path.parent))
    cfg = dataclasses.asdict(buffer.cfg) if dataclasses.is_dataclass(buffer.cfg) else vars(buffer.cfg)
    manifest = dict(format=FORMAT, version=VERSION, rows=0, fields={}, segments=[], shards=[],
                    num_episodes_added=buffer.num_eps,
                    source_config=json.loads(json.dumps(cfg, default=str)))
    try:
        indices = buffer.chronological_indices()
        if len(indices):
            first = buffer._buffer[indices[:1]].select(*FIELDS)
            manifest['fields'] = {
                key: dict(dtype=value.cpu().numpy().dtype.str, shape=list(value.shape[1:]))
                for key, value in first.items()
            }
            row_bytes = sum(value.numel() * value.element_size() for value in first.values())
            rows_per_chunk = max(1, chunk_bytes // row_bytes)
            # Episode IDs are small; observations are copied only one shard at a time.
            episodes = buffer._buffer.storage.get(slice(None))['episode'][indices].cpu().numpy()
            manifest['segments'] = episode_segments(episodes)
            manifest['rows'] = len(indices)
            del first, episodes
            for start in range(0, len(indices), rows_per_chunk):
                end = min(start + rows_per_chunk, len(indices))
                td = buffer._buffer[indices[start:end]].select(*FIELDS).to('cpu')
                arrays = {key: value.numpy() for key, value in td.items()}
                encoded, encoding = _encode(arrays, buffer.cfg.obs == 'rgb')
                filename = f'part-{len(manifest["shards"]):05d}.npz'
                np.savez_compressed(staging / filename, **encoded)
                manifest['shards'].append(dict(file=filename, start=start, rows=end - start,
                                               observation_encoding=encoding))
                del td, arrays, encoded
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # On the same filesystem, rename exposes either the complete directory or nothing.
        os.rename(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def read_manifest(path, cfg, capacity):
    """Validate metadata and compatibility before allocating replay storage."""
    path = Path(path).expanduser()
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['format'] != FORMAT or manifest['version'] != VERSION:
            raise ValueError('Unsupported replay dataset format or version')
        rows = manifest['rows']
        if type(rows) is not int or rows < 0:
            raise ValueError('Invalid replay row count')
        if rows > capacity:
            raise ValueError(f'Replay dataset has {rows} rows, but capacity is {capacity}; '
                             f'set buffer_size>={rows}')
        if rows == 0:
            raise ValueError('Replay dataset is empty; no valid training windows')
        source = manifest['source_config']
        if source['task'] != cfg.task or source['obs'] != cfg.obs:
            raise ValueError('Replay task or observation mode is incompatible with this run')
        if cfg.obs == 'rgb' and source['frame_stack'] != cfg.frame_stack:
            raise ValueError('Replay frame_stack is incompatible with this run')
        expected = dict(obs=(cfg.obs_shape[cfg.obs], 'uint8' if cfg.obs == 'rgb' else 'float32'),
                        action=([cfg.action_dim], 'float32'), reward=([], 'float32'),
                        terminated=([], 'float32'), episode=([], 'int64'))
        if set(manifest['fields']) != set(FIELDS):
            raise ValueError('Invalid replay fields')
        for key, (shape, dtype) in expected.items():
            spec = manifest['fields'][key]
            if spec['shape'] != list(shape) or np.dtype(spec['dtype']) != np.dtype(dtype):
                raise ValueError(f'Incompatible replay field: {key}')
        offset, names = 0, set()
        for shard in manifest['shards']:
            name = shard['file']
            if (not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz')
                    or name in names or not (path / name).is_file()):
                raise ValueError(f'Invalid or missing replay shard: {name}')
            names.add(name)
            if (type(shard['rows']) is not int or shard['rows'] < 1
                    or shard['start'] != offset
                    or shard['observation_encoding'] not in {'raw', 'frame_stack'}
                    or (shard['observation_encoding'] == 'frame_stack' and cfg.obs != 'rgb')):
                raise ValueError('Invalid replay shard metadata')
            offset += shard['rows']
        if offset != rows:
            raise ValueError('Replay shards do not match the total row count')
        offset, episode_ids = 0, set()
        for segment in manifest['segments']:
            if (segment['start'] != offset or type(segment['length']) is not int
                    or segment['length'] < 1 or type(segment['episode']) is not int
                    or segment['episode'] < 0 or segment['episode'] in episode_ids):
                raise ValueError('Invalid replay episode segments')
            episode_ids.add(segment['episode'])
            offset += segment['length']
        if offset != rows:
            raise ValueError('Replay episode segments do not match the total row count')
        if not any(s['length'] >= cfg.horizon + 1 for s in manifest['segments']):
            raise ValueError('Replay dataset has no valid training windows for this horizon')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Cannot load replay dataset {path}: {exc}') from exc
    if source.get('dot') != cfg.dot:
        # Training suppresses Python warnings globally, so report transfer settings explicitly.
        print('Replay source environment differs from the current dot settings; '
              'pretraining will use the stored observations and rewards.')
    return manifest


def iter_dataset_chunks(path, manifest):
    """Decode and validate a shard at a time; no executable pickle data is accepted."""
    segment_index = 0
    for shard in manifest['shards']:
        filename = Path(path).expanduser() / shard['file']
        try:
            with np.load(filename, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            count = shard['rows']
            if shard['observation_encoding'] == 'frame_stack':
                if set(arrays) != (set(FIELDS) - {'obs'}) | {'obs_initial', 'obs_frames'}:
                    raise ValueError('Invalid frame-stack arrays')
                episodes = arrays['episode']
                if episodes.shape != (count,):
                    raise ValueError('Invalid episode array')
                segments = episode_segments(episodes)
                initial, frames = arrays.pop('obs_initial'), arrays.pop('obs_frames')
                shape = manifest['fields']['obs']['shape']
                dtype = np.dtype(manifest['fields']['obs']['dtype'])
                if (initial.shape != (len(segments), *shape)
                        or frames.shape != (count, 3, *shape[1:])
                        or initial.dtype != dtype or frames.dtype != dtype):
                    raise ValueError('Invalid encoded observations')
                obs = np.empty((count, *shape), dtype=dtype)
                for i, segment in enumerate(segments):
                    start, length = segment['start'], segment['length']
                    obs[start] = initial[i]
                    if not np.array_equal(initial[i, -3:], frames[start]):
                        raise ValueError('Inconsistent initial observation')
                    for row in range(start + 1, start + length):
                        obs[row, :-3] = obs[row - 1, 3:]
                        obs[row, -3:] = frames[row]
                arrays['obs'] = obs
            if set(arrays) != set(FIELDS):
                raise ValueError('Invalid replay arrays')
            for key, spec in manifest['fields'].items():
                value = arrays[key]
                if value.shape != (count, *spec['shape']) or value.dtype != np.dtype(spec['dtype']):
                    raise ValueError(f'Invalid shape or dtype for {key}')
            begin, end = shard['start'], shard['start'] + count
            segments = manifest['segments']
            while segments[segment_index]['start'] + segments[segment_index]['length'] <= begin:
                segment_index += 1
            index = segment_index
            while index < len(segments) and segments[index]['start'] < end:
                segment = segments[index]
                left, right = max(begin, segment['start']), min(end, segment['start'] + segment['length'])
                if left < right and not np.all(arrays['episode'][left - begin:right - begin] == segment['episode']):
                    raise ValueError('Replay episode IDs do not match the manifest')
                index += 1
            yield TensorDict({key: torch.from_numpy(value) for key, value in arrays.items()},
                             batch_size=[count])
        except (OSError, KeyError, TypeError, ValueError, EOFError, BadZipFile) as exc:
            raise ValueError(f'Invalid replay shard {filename}: {exc}') from exc
