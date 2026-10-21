"""Parallel dot-world collection with one learner and trajectory-safe replay."""

from time import perf_counter

import numpy as np
import torch
from tensordict import TensorDict

from common.device import mark_step
from environments.tdmpc2 import make_dot_env
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
        self._eval_env = make_dot_env(self.cfg.dot, self.cfg.obs, self.cfg.frame_stack)
        self._action_rng = np.random.default_rng(self.cfg.seed + 100_000)
        self._tds = []

    def metrics(self):
        elapsed = perf_counter() - self._start
        return dict(step=self._step, episode=self.buffer.num_eps, updates=self._updates,
                    elapsed_time=elapsed, steps_per_second=self._step / max(elapsed, 1e-9),
                    collection_seconds=self._collection_time, planning_seconds=self._planning_time,
                    update_seconds=self._update_time,
                    collection_steps_per_second=self._step / max(self._collection_time, 1e-9))

    def eval(self):
        results, rates, successes = [], [], []
        for episode in range(self.cfg.eval_episodes):
            # Repeat the same held-out episodes at every evaluation.
            obs, info = self._eval_env.reset(seed=self.cfg.seed + 1_000_000 + episode)
            reward_sum = 0
            if self.logger.video:
                self.logger.video.init(self._eval_env, enabled=episode == 0)
            for t in range(self.cfg.episode_length):
                mark_step(self.agent.device)
                action = self.agent.act_batch(obs[None], [t == 0], eval_mode=True)[0].numpy()
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                reward_sum += reward
                if self.logger.video:
                    self.logger.video.record(self._eval_env)
                if terminated or truncated:
                    break
            results.append(reward_sum)
            rates.append(info['collision_steps'] / info['elapsed_steps'])
            successes.append(info['success'])
            if self.logger.video and episode == 0:
                self.logger.video.save(self._step)
        return dict(episode_reward=np.mean(results), collision_step_rate=np.mean(rates),
                    collision_free_episode_fraction=np.mean(successes), episode_success=np.mean(successes))

    def _flush(self):
        for rows in self._tds:
            if len(rows) > 1:
                # Partial episodes retain terminated=False and their actual last observation.
                self.buffer.add(torch.cat(rows))
        self._tds = []

    def train(self):
        cfg = self.cfg
        if cfg.steps < 1 or cfg.horizon < 1 or cfg.eval_freq < 1 or cfg.eval_episodes < 1:
            raise ValueError('steps, horizon, eval_freq, and eval_episodes must be positive')
        if cfg.dot['max_episode_steps'] < cfg.horizon:
            raise ValueError('Episode length must be at least the replay horizon')
        if cfg.buffer_size < cfg.horizon + 1 or cfg.steps < cfg.horizon + 1:
            raise ValueError('steps and buffer_size must hold at least horizon+1 replay rows')
        if not np.isfinite(cfg.updates_per_transition) or cfg.updates_per_transition < 0:
            raise ValueError('updates_per_transition must be finite and nonnegative')
        if cfg.seed_steps < 0:
            raise ValueError('seed_steps must be nonnegative')
        next_eval, credit = cfg.eval_freq, 0.0
        try:
            obs, _ = self.env.reset(seed=[cfg.seed + i for i in range(cfg.num_envs)])
            self._tds = [[transition(o)] for o in obs]
            reset_mask = np.ones(cfg.num_envs, dtype=bool)
            self.logger.log(dict(self.eval(), **self.metrics()), 'eval')
            while self._step < cfg.steps:
                learning = self._step >= cfg.seed_steps and self.buffer.ready
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
            self.logger.finish(self.agent)
        finally:
            self._flush()
            self.env.close()
            self._eval_env.close()
