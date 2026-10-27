"""Step 4: Train the DQN by playing against dataset-derived bot styles."""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dqn_model_surya.env import ACTION_COUNT, ACTION_NAMES, STATE_SIZE, PokerDQNEnv
from dqn_model_surya.model import QNetwork, masked_q_values
from dqn_model_surya.replay_buffer import ReplayBuffer


def epsilon_at_step(step: int, start: float, end: float, decay_steps: int) -> float:
    """Epsilon decay lowers the random-action chance as the agent gains experience."""

    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)


def choose_action(
    online_network: QNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Epsilon-greedy selection samples or maximizes over legal actions only."""

    legal_actions = np.flatnonzero(action_mask)
    if legal_actions.size == 0:
        raise ValueError('Cannot choose an action when none are legal.')
    if rng.random() < epsilon:
        return int(rng.choice(legal_actions.tolist()))

    online_network.eval()
    with torch.no_grad():
        q_values = online_network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        legal_q_values = masked_q_values(
            q_values,
            torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0),
        )
    return int(legal_q_values.argmax(dim=1).item())


def optimize_model(
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> float:
    """Apply the masked Bellman target: r + gamma * max legal Q(next_state, action)."""

    batch = replay_buffer.sample(batch_size)
    states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32)
    actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in batch]),
        dtype=torch.float32,
    )
    dones = torch.as_tensor([item.done for item in batch], dtype=torch.bool)
    next_masks = torch.as_tensor(
        np.stack([item.next_action_mask for item in batch]),
        dtype=torch.bool,
    )
    if torch.any(~dones & ~next_masks.any(dim=1)):
        raise ValueError('A continuing transition has no legal next action.')

    online_network.train()
    current_q = online_network(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_network(next_states)
        masked_next_q = next_q.masked_fill(
            ~next_masks,
            torch.finfo(next_q.dtype).min,
        )
        max_next_q = masked_next_q.max(dim=1).values
        max_next_q = torch.where(dones, torch.zeros_like(max_next_q), max_next_q)
        targets = rewards + gamma * max_next_q

    loss = nn.functional.smooth_l1_loss(current_q, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=int, default=20_000, help='Training hand episodes.')
    parser.add_argument('--seed', type=int, default=1407)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--replay-capacity', type=int, default=50_000)
    parser.add_argument('--warmup', type=int, default=1_000)
    parser.add_argument('--target-update', type=int, default=500, help='Steps between target-network copies.')
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay-steps', type=int, default=30_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path(__file__).resolve().parent / 'checkpoints' / 'pokerpilot_dqn.pt',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hands <= 0 or args.batch_size <= 0 or args.target_update <= 0:
        raise ValueError('Hands, batch size, and target-update interval must be positive.')
    if args.replay_capacity < args.batch_size or args.warmup < args.batch_size:
        raise ValueError('Replay capacity and warmup must be at least the batch size.')
    if args.epsilon_decay_steps <= 0 or not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        raise ValueError('Epsilon settings must satisfy 0 <= end <= start <= 1.')
    if not 0 <= args.gamma <= 1:
        raise ValueError('Gamma must be between 0 and 1.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    env = PokerDQNEnv(seed=args.seed, style_split='train')
    online_network = QNetwork(STATE_SIZE, ACTION_COUNT)
    target_network = QNetwork(STATE_SIZE, ACTION_COUNT)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.0003)
    replay_buffer = ReplayBuffer(args.replay_capacity, args.seed)
    action_rng = random.Random(args.seed)
    recent_rewards: deque[float] = deque(maxlen=1_000)
    steps = 0
    last_loss: float | None = None

    for episode in range(1, args.hands + 1):
        state, action_mask, _ = env.reset()
        done = False
        while not done:
            epsilon = epsilon_at_step(
                steps,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay_steps,
            )
            action = choose_action(
                online_network,
                state,
                action_mask,
                epsilon,
                action_rng,
            )
            next_state, next_mask, reward, done, _ = env.step(action)
            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
                next_mask,
            )
            state, action_mask = next_state, next_mask
            steps += 1

            if len(replay_buffer) >= args.warmup:
                last_loss = optimize_model(
                    online_network,
                    target_network,
                    optimizer,
                    replay_buffer,
                    args.batch_size,
                    args.gamma,
                )
            if steps % args.target_update == 0:
                target_network.load_state_dict(online_network.state_dict())

            if done:
                recent_rewards.append(reward)

        if episode % 1_000 == 0 or episode == args.hands:
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            print(
                f'hands={episode} steps={steps} epsilon={epsilon:.3f} '
                f'last_loss={last_loss!r} recent_bb_per_hand={avg_reward:.3f}',
                flush=True,
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': online_network.state_dict(),
        'state_size': STATE_SIZE,
        'action_count': ACTION_COUNT,
        'action_names': list(ACTION_NAMES),
        'seed': args.seed,
        'training_hands': args.hands,
        'gamma': args.gamma,
    }, args.checkpoint)
    print(f'Saved DQN checkpoint to {args.checkpoint}')


if __name__ == '__main__':
    main()
