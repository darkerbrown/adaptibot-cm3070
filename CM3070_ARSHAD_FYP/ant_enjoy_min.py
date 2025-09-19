# ant_enjoy_min.py
# Minimal, reliable CleanRL PPO enjoy for Ant-v4 (Gymnasium+MuJoCo)
# Proves the model loads, normalizes obs, and MOVES.


"""Tiny verification script for the trained policy.

- This script loads a saved policy and runs it in the Ant environment for
    a short number of steps to confirm the policy causes the robot to move.
- It includes a minimal observation normaliser so the policy sees inputs
    in the same format as during training. Use this script to sanity-check
    a model file before running longer experiments.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

class ActorCritic(torch.nn.Module):
    """Minimal actor-critic network mirroring the CleanRL Ant architecture."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
                # Mirror the CleanRL initialization so checkpoints load verbatim.
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, act_dim)
        )
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, act_dim))

    def act_stochastic(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean = self.actor_mean(obs_t)
                # Keep a minimum std so sampling never collapses.
        std  = self.actor_logstd.exp().expand_as(mean)
        std  = torch.clamp(std, min=0.05)      # guarantee nonzero std
        dist = torch.distributions.Normal(mean, std)
        a    = torch.tanh(dist.rsample())      # [-1,1] like CleanRL enjoy
        return a

# Lightweight obs normalizer to mimic VecNormalize from training.
class VecNormalize:
    """Tiny observation normalizer using stored running-moments."""
    def __init__(self, env, obs_rms):
        self.env = env
        self.mean = obs_rms["mean"]
        self.var  = obs_rms["var"]
    def _norm(self, obs): return (obs - self.mean) / np.sqrt(self.var + 1e-8)
    def reset(self, **kw):
        # Normalise the very first observation as well.
        obs, info = self.env.reset(**kw)
        return self._norm(obs), info
    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        return self._norm(obs), r, t, tr, info

def run(model_path, steps, gui):
    """Load a PPO policy and roll out Ant-v4 to confirm locomotion locally."""
        # Create the base Ant environment; use human viewer when GUI flag set.
    env = gym.make("Ant-v4", render_mode="human" if gui else None)

    device = torch.device("cpu")
        # Load CleanRL checkpoint (contains network weights + obs stats).
    raw = torch.load(model_path, map_location=device)

    obs_rms = {
        "mean": raw.pop("obs_rms.mean").numpy(),
        "var":  raw.pop("obs_rms.var").numpy(),
        "count": raw.pop("obs_rms.count").item(),
    }
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    pi = ActorCritic(obs_dim, act_dim)
    pi.load_state_dict(raw, strict=True)
    pi.eval()

    wrapped = VecNormalize(env, obs_rms)

    obs, _ = wrapped.reset(seed=0)
    total_xy_disp = 0.0
    # Track planar displacement as a sanity metric.
    last_xy = env.unwrapped.data.qpos[0:2].copy()

    # Roll out the policy for a fixed number of evaluation steps.
    for i in range(steps):
        with torch.no_grad():
            a = pi.act_stochastic(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        act = a.squeeze(0).numpy().astype(np.float32)
        # scale to env bounds just in case
        # Respect environment action bounds just in case.
        hi = getattr(env.action_space, "high", None)
        if hi is not None:
            act = np.clip(act * hi, -hi, hi).astype(np.float32)

        obs, r, term, trunc, _ = wrapped.step(act)
        xy = env.unwrapped.data.qpos[0:2]
        total_xy_disp += float(np.linalg.norm(xy - last_xy))
        last_xy = xy.copy()
        if term or trunc:
            break

    env.close()
        # Report whether the Ant actually moved meaningfully.
    print(f"[sanity] steps={i+1} total_xy_disp={total_xy_disp:.2f} (should be > 2.0 if actually moving)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    run(args.model, args.steps, args.gui)
