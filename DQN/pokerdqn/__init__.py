"""Deep Q-Network experiments for PokerPilot (No-Limit Texas Hold'em).

Modules:
    cards     card parsing, a fast 5-7 card evaluator, hand features, equity
    actions   the discrete action space, legality masks and bet sizing
    features  the state vector shared by the dataset and the simulator
    dataset   loading the PokerPilot CSVs and building offline transitions
    stats     opponent-tendency tracking (VPIP, PFR, aggression, ...)
    env       a pokerkit-based multi-player simulator with one learning seat
    agents    random, rule-based and dataset-calibrated opponents/baselines
    dqn       Q-network, replay buffer and the (double, optionally
              conservative) DQN learner
    evaluation  head-to-head evaluation helpers
"""
