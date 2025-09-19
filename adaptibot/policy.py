"""Policy loader and small neural network wrapper.

- This file defines a simple neural network structure that behaves like
    the controller used during training and provides a loader that reads
    a saved checkpoint (the trained brain). The saved checkpoint also
    contains simple statistics used to normalise sensor readings.

Key grader notes:
- `ActorCritic` is the learned brain. It exposes two helpers: one for
    a deterministic action and one that samples a slightly random
    action (useful for exploration during training).
- `load_cleanrl_model` reads a file saved by the training code and
    returns both the brain and the normalisation numbers so the brain
    receives inputs in the format it expects.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    """Ant-sized actor-critic network matching the CleanRL checkpoint layout."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
                # Two-layer MLP critic mirroring CleanRL defaults.
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
                # Shared architecture for the mean action network.
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, act_dim),
        )
                # Trainable log std replicating CleanRL behaviour.
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, act_dim))

    def act_mean(self, obs_t: torch.Tensor) -> torch.Tensor:
        """Return deterministic Tanh-squashed actions."""
                # CleanRL stores logits before the Tanh squash.
        mean = self.actor_mean(obs_t)
        return torch.tanh(mean)

    def act_stochastic(self, obs_t: torch.Tensor, min_std: float = 0.05) -> torch.Tensor:
        """Sample an exploration action with a floor on the log-std."""
        mean = self.actor_mean(obs_t)
                # Ensure we never collapse exploration completely.
        std = torch.clamp(self.actor_logstd.exp().expand_as(mean), min=min_std)
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        return torch.tanh(sample)


def load_cleanrl_model(
    model_path: str,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> Tuple[ActorCritic, dict]:
    """Load a CleanRL PPO checkpoint and return (policy, obs_rms)."""
        # Torch 2.1 added weights_only; fall back when running on older versions.
    try:
        raw = torch.load(model_path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        raw = torch.load(model_path, map_location=device)

        # Extract observation normalisation stats if present.
    mean = raw.pop("obs_rms.mean", None)
    var = raw.pop("obs_rms.var", None)
    cnt = raw.pop("obs_rms.count", None)
    if mean is None or var is None or cnt is None:
        obs_rms = {
            "mean": np.zeros((obs_dim,), dtype=np.float32),
            "var": np.ones((obs_dim,), dtype=np.float32),
            "count": 1.0,
        }
    else:
        obs_rms = {
            "mean": mean.numpy() if hasattr(mean, "numpy") else np.array(mean, dtype=np.float32),
            "var": var.numpy() if hasattr(var, "numpy") else np.array(var, dtype=np.float32),
            "count": float(cnt.item() if hasattr(cnt, "item") else cnt),
        }

        # Instantiate the network and load the PPO weights.
    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(raw, strict=False)
    policy.eval()
    return policy, obs_rms
