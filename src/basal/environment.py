import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np


class MemoryRetrievalEnv(gym.Env):
    def __init__(self, vsa_store, query_vectors: list[torch.Tensor],
                 correct_indices: list[int], dim: int = 1000):
        super().__init__()
        self.store = vsa_store
        self.queries = query_vectors
        self.correct = correct_indices
        self.dim = dim
        self.action_space = spaces.Discrete(len(correct_indices) + 1)
        self.observation_space = spaces.Box(-1, 1, shape=(dim,), dtype=np.float32)
        self._idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return self.queries[0].cpu().numpy(), {}

    def step(self, action: int):
        correct = self.correct[self._idx]
        reward = 1.0 if action == correct else -0.1
        self._idx += 1
        terminated = self._idx >= len(self.queries)
        truncated = False
        if not terminated:
            obs = self.queries[self._idx].cpu().numpy()
        else:
            obs = np.zeros(self.dim, dtype=np.float32)
        return obs, reward, terminated, truncated, {}
