"""Step 3: Store transitions for experience replay."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    """A bounded memory that samples past actions for less correlated updates."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError('Replay-buffer capacity must be positive.')
        self.memory: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self.memory.append(Transition(
            state=state.copy(),
            action=action,
            reward=float(reward),
            next_state=next_state.copy(),
            done=bool(done),
            next_action_mask=next_action_mask.copy(),
        ))

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.memory):
            raise ValueError('Cannot sample more transitions than the buffer contains.')
        return self.rng.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)
