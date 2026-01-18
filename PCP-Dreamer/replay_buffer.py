import random
from collections import defaultdict
import numpy as np
import torch
from typing import Optional, Dict

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: list = []
        self.current_ep: Dict[str, list] = defaultdict(list)
        self._total_steps = 0

    def add(self, obs, action, reward, next_obs, done, terminal):
        self.current_ep["obs"].append(obs.astype(np.float32))
        self.current_ep["action"].append(action.astype(np.float32))
        self.current_ep["reward"].append(np.float32(reward))
        self.current_ep["terminal"].append(np.float32(terminal))
        self._total_steps += 1

        if done:
            self.current_ep["obs"].append(next_obs.astype(np.float32))
            final_ep = {k: np.array(v) for k, v in self.current_ep.items()}
            self.episodes.append(final_ep)
            self.current_ep = defaultdict(list)

            while self._total_steps > self.capacity and len(self.episodes) > 1:
                rem = self.episodes.pop(0)
                self._total_steps -= len(rem["reward"])

    def sample(self, batch_size: int, seq_len: int) -> Optional[Dict[str, torch.Tensor]]:
        valid = [ep for ep in self.episodes if len(ep["reward"]) >= seq_len]
        if not valid:
            return None

        batch = defaultdict(list)
        for _ in range(batch_size):
            ep = random.choice(valid)
            start = random.randint(0, len(ep["reward"]) - seq_len)

            batch["obs"].append(ep["obs"][start:start + seq_len + 1])
            batch["action"].append(ep["action"][start:start + seq_len])
            batch["reward"].append(ep["reward"][start:start + seq_len])
            batch["terminal"].append(ep["terminal"][start:start + seq_len])

        return {k: torch.from_numpy(np.stack(v)) for k, v in batch.items()}

    def __len__(self):
        return self._total_steps