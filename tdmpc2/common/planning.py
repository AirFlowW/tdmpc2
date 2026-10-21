"""Batched MPPI: every environment has independent candidates and warm starts."""

import torch
from common import math


@torch.no_grad()
def batched_plan(agent, obs, previous_mean, reset_mask, eval_mode):
    cfg = agent.cfg
    batch, horizon, samples, actions_dim = len(obs), cfg.horizon, cfg.num_samples, cfg.action_dim
    z = agent.model.encode(obs, None)
    candidates = torch.empty(horizon, batch, samples, actions_dim, device=z.device)
    if cfg.num_pi_trajs:
        policy_z = z[:, None, :].expand(-1, cfg.num_pi_trajs, -1).reshape(-1, z.shape[-1])
        for t in range(horizon):
            policy_action, _ = agent.model.pi(policy_z, None)
            candidates[t, :, :cfg.num_pi_trajs] = policy_action.reshape(batch, cfg.num_pi_trajs, actions_dim)
            if t + 1 < horizon:
                policy_z = agent.model.next(policy_z, policy_action, None)

    mean = torch.zeros(horizon, batch, actions_dim, device=z.device)
    mean[:-1] = previous_mean[1:] * (~reset_mask)[None, :, None]
    std = torch.full_like(mean, cfg.max_std)
    rollout_z = z[:, None, :].expand(-1, samples, -1).reshape(-1, z.shape[-1])
    for _ in range(cfg.iterations):
        noise = torch.randn(horizon, batch, samples - cfg.num_pi_trajs, actions_dim, device=z.device)
        candidates[:, :, cfg.num_pi_trajs:] = (mean[:, :, None] + std[:, :, None] * noise).clamp(-1, 1)
        values = agent._estimate_value(
            rollout_z, candidates.reshape(horizon, batch * samples, actions_dim), None
        ).reshape(batch, samples).nan_to_num(0)
        elite_values, indices = values.topk(cfg.num_elites, dim=1)
        elite_actions = candidates.gather(
            2, indices[None, :, :, None].expand(horizon, -1, -1, actions_dim))
        scores = torch.softmax(cfg.temperature * (elite_values - elite_values.max(1, keepdim=True).values), dim=1)
        weights = scores[None, :, :, None]
        mean = (weights * elite_actions).sum(dim=2)
        std = (weights * (elite_actions - mean[:, :, None]).square()).sum(dim=2).sqrt()
        std = std.clamp(cfg.min_std, cfg.max_std)

    selected = math.gumbel_softmax_sample(scores, dim=-1)
    action = elite_actions[0].gather(1, selected[:, None, None].expand(-1, 1, actions_dim)).squeeze(1)
    if not eval_mode:
        action = action + std[0] * torch.randn_like(action)
    return action.clamp(-1, 1), mean
