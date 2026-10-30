"""Deep Q-Network: network, replay buffer, epsilon-greedy policy, learner.

* ``QNetwork``: MLP with an optional dueling head, one Q-value per discrete
  action (``actions.py``).
* Illegal actions are masked everywhere: action selection, the max in the
  bootstrap target, and the conservative penalty.
* Targets use double DQN: the online network picks the next action, a
  slowly updated (Polyak) target network evaluates it.
* ``cql_alpha > 0`` adds the conservative Q-learning penalty
  ``logsumexp_legal Q(s, .) - Q(s, a_logged)`` used for the offline stage:
  the dataset only shows the logged action in each state, so without it the
  max over never-tried actions is pure extrapolation and overestimates.
* Rewards are in big blinds; ``reward_scale`` divides them inside the
  learner so Q-values stay O(1). ``q_values`` returns big blinds.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .actions import NUM_ACTIONS
from .agents import Agent, Observation

NEG = -1e9


@dataclass
class DQNConfig:
    state_dim: int
    num_actions: int = NUM_ACTIONS
    hidden: tuple[int, ...] = (256, 256)
    dueling: bool = True
    lr: float = 1e-4
    gamma: float = 1.0  # episodes are single hands; chips are not discounted
    batch_size: int = 256
    tau: float = 0.005  # Polyak rate for the target network
    reward_scale: float = 20.0
    huber_delta: float = 5.0  # in scaled units (= 100 bb)
    grad_clip: float = 10.0
    cql_alpha: float = 0.0
    weight_decay: float = 0.0


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden=(256, 256), dueling=True) -> None:
        super().__init__()
        layers = []
        width = state_dim

        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size

        self.body = nn.Sequential(*layers)
        self.dueling = dueling

        if dueling:
            self.value = nn.Linear(width, 1)
            self.advantage = nn.Linear(width, num_actions)
        else:
            self.head = nn.Linear(width, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)

        if not self.dueling:
            return self.head(h)

        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


def masked(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, q, torch.full_like(q, NEG))


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done, legal(s), legal(s'))."""

    def __init__(self, capacity: int, state_dim: int, num_actions: int = NUM_ACTIONS,
                 seed: int = 0) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.masks = np.zeros((capacity, num_actions), dtype=bool)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, next_state, done, mask, next_mask) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.masks[i] = mask

        if done:
            self.next_states[i] = 0.0
            self.next_masks[i] = False
        else:
            self.next_states[i] = next_state
            self.next_masks[i] = next_mask

        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_many(self, data) -> None:
        for i in range(len(data.actions)):
            self.add(data.states[i], data.actions[i], data.rewards[i], data.next_states[i],
                     data.dones[i], data.masks[i], data.next_masks[i])

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            'states': self.states[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'next_states': self.next_states[idx],
            'dones': self.dones[idx],
            'masks': self.masks[idx],
            'next_masks': self.next_masks[idx],
        }


class DQNLearner:
    def __init__(self, config: DQNConfig, seed: int = 0, device: str = 'cpu') -> None:
        torch.manual_seed(seed)
        self.config = config
        self.device = torch.device(device)
        self.online = QNetwork(config.state_dim, config.num_actions, config.hidden,
                               config.dueling).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr,
                                          weight_decay=config.weight_decay)
        self.updates = 0

    # ---------------------------------------------------------- acting

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Q-values in big blinds for a batch (or a single) state."""

        x = torch.as_tensor(np.atleast_2d(states), dtype=torch.float32, device=self.device)
        return self.online(x).cpu().numpy() * self.config.reward_scale

    def greedy(self, state: np.ndarray, mask: np.ndarray) -> int:
        q = self.q_values(state)[0]
        q[~mask] = -np.inf
        return int(np.argmax(q))

    # --------------------------------------------------------- learning

    def _tensors(self, batch: dict) -> dict[str, torch.Tensor]:
        out = {}

        for key, value in batch.items():
            dtype = {'actions': torch.int64, 'dones': torch.bool, 'masks': torch.bool,
                     'next_masks': torch.bool}.get(key, torch.float32)
            out[key] = torch.as_tensor(value, dtype=dtype, device=self.device)

        return out

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        c = self.config
        b = self._tensors(batch)
        q_all = self.online(b['states'])
        q = q_all.gather(1, b['actions'][:, None]).squeeze(1)

        with torch.no_grad():
            live = ~b['dones'] & b['next_masks'].any(dim=1)
            next_online = masked(self.online(b['next_states']), b['next_masks'])
            best = next_online.argmax(dim=1, keepdim=True)
            next_q = self.target(b['next_states']).gather(1, best).squeeze(1)
            next_q = torch.where(live, next_q, torch.zeros_like(next_q))
            target = b['rewards'] / c.reward_scale + c.gamma * next_q

        td = F.huber_loss(q, target, delta=c.huber_delta)
        loss = td
        info = {'td_loss': td.item(), 'q_mean': q.detach().mean().item() * c.reward_scale}

        if c.cql_alpha > 0:
            logsumexp = torch.logsumexp(masked(q_all, b['masks']), dim=1)
            gap = (logsumexp - q).mean()
            loss = loss + c.cql_alpha * gap
            info['cql_gap'] = gap.item()

        info['loss'] = loss.item()
        return loss, info

    def update(self, batch: dict) -> dict:
        self.online.train()
        loss, info = self.loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.updates += 1

        with torch.no_grad():
            for p, tp in zip(self.online.parameters(), self.target.parameters()):
                tp.mul_(1 - self.config.tau).add_(p, alpha=self.config.tau)

        return info

    @torch.no_grad()
    def evaluate_loss(self, data, batch_size: int = 4096) -> dict:
        """Mean TD loss etc. over a full TransitionSet (no gradient)."""

        self.online.eval()
        totals = {}
        n = len(data.actions)

        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            batch = {
                'states': data.states[sl], 'actions': data.actions[sl],
                'rewards': data.rewards[sl], 'next_states': data.next_states[sl],
                'dones': data.dones[sl], 'masks': data.masks[sl],
                'next_masks': data.next_masks[sl],
            }
            _, info = self.loss(batch)
            weight = len(batch['actions'])

            for key, value in info.items():
                totals[key] = totals.get(key, 0.0) + value * weight

        return {key: value / n for key, value in totals.items()}

    # ------------------------------------------------------ checkpoints

    def save(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': asdict(self.config),
            'online': self.online.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'updates': self.updates,
            'extra': extra or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = 'cpu', **overrides) -> 'DQNLearner':
        data = torch.load(path, map_location=device, weights_only=False)
        config = data['config']
        config['hidden'] = tuple(config['hidden'])
        config.update(overrides)
        learner = cls(DQNConfig(**config), device=device)
        learner.online.load_state_dict(data['online'])
        learner.target.load_state_dict(data['target'])

        if not overrides:
            learner.optimizer.load_state_dict(data['optimizer'])

        learner.updates = data['updates']
        learner.extra = data.get('extra', {})
        return learner


class EpsilonSchedule:
    def __init__(self, start: float, end: float, steps: int) -> None:
        self.start, self.end, self.steps = start, end, max(steps, 1)

    def __call__(self, step: int) -> float:
        frac = min(step / self.steps, 1.0)
        return self.start + frac * (self.end - self.start)


class DQNAgent(Agent):
    """Plays with a learner's Q-values: epsilon-greedy over legal actions
    (``epsilon=0`` for evaluation)."""

    name = 'dqn'

    def __init__(self, learner: DQNLearner, epsilon: float = 0.0,
                 rng: random.Random | None = None) -> None:
        self.learner = learner
        self.epsilon = epsilon
        self.rng = rng or random.Random()

    def act(self, obs: Observation) -> int:
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return self.rng.choice(list(np.flatnonzero(obs.mask)))

        return self.learner.greedy(obs.vector(), obs.mask)
