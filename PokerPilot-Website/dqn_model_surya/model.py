"""Step 3: Define the online/target Q-networks and legal-action selection."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """A small multilayer perceptron that estimates one Q-value per action."""

    def __init__(self, state_size: int, action_count: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Replace illegal action values before argmax or next-state maximization."""

    action_mask = action_mask.to(dtype=torch.bool, device=q_values.device)
    if q_values.shape != action_mask.shape:
        raise ValueError('Q-values and legal-action masks must have matching shapes.')
    if not torch.all(action_mask.any(dim=-1)):
        raise ValueError('A non-terminal state must have at least one legal action.')
    return q_values.masked_fill(~action_mask, torch.finfo(q_values.dtype).min)


def choose_greedy_action(
    network: QNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[int, list[float | None]]:
    """Choose the highest-valued legal action and return its displayed Q-values."""

    if action_mask.ndim != 1 or action_mask.size != network.layers[-1].out_features:
        raise ValueError('The action mask has the wrong shape.')
    if not action_mask.any():
        raise ValueError('No action is legal in this state.')

    network.eval()
    with torch.no_grad():
        q_values = network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0]
        masked = masked_q_values(
            q_values,
            torch.as_tensor(action_mask, dtype=torch.bool),
        )
    chosen = int(masked.argmax().item())
    displayed = [
        float(value) if legal else None
        for value, legal in zip(q_values.tolist(), action_mask.tolist(), strict=True)
    ]
    return chosen, displayed


@lru_cache(maxsize=2)
def load_policy(path: str, modified_ns: int) -> tuple[QNetwork, dict]:
    """Load and cache one CPU inference network for the website advisor."""

    del modified_ns
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    state_size = checkpoint.get('state_size')
    action_count = checkpoint.get('action_count')
    if not isinstance(state_size, int) or not isinstance(action_count, int):
        raise ValueError('The DQN checkpoint is missing its model dimensions.')

    network = QNetwork(state_size, action_count)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    return network, checkpoint
