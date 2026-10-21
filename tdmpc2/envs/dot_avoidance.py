"""Adapt the root Gymnasium environment to TD-MPC2's scalar API."""

import gymnasium as gym
from environments.tdmpc2 import configure, make_dot_env


class DotAdapter(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._initial_seed = seed

    @property
    def max_episode_steps(self):
        return self.env.unwrapped.max_episode_steps

    def reset(self):
        obs, _ = self.env.reset(seed=self._initial_seed)
        self._initial_seed = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["terminated"] = terminated
        return obs, reward, terminated or truncated, info


def make_env(cfg):
    env = make_dot_env(cfg.dot, cfg.obs, cfg.frame_stack)
    configure(cfg, env)
    return DotAdapter(env, cfg.seed)
