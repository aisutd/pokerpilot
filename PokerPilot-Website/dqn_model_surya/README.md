# Surya's PokerPilot DQN

This project trains a **Deep Q-Network (DQN)**, a neural network that estimates
the long-term value of each action, by letting it play PokerPilot's existing
6-max engine. This is online reinforcement learning: the agent learns from game
experience it generates rather than from decision labels. The logged decision
CSVs are not treated as transitions because they do not record the next state
after an action.

## Steps 1-2: environment, state, actions, and reward

[`env.py`](./env.py) wraps `engine.game.Table` without changing engine code. One
learning episode is one hand, and the agent keeps the table's stacks and anonymous
opponent tendencies when the next hand begins.

The state contains the agent's two hole cards, the public board, street and relative
position, pot/call/stack values, and averaged public opponent statistics. It never
reads opponents' hidden cards or includes names or player IDs. **A legal-action
mask** is a True/False list that prevents illegal choices; the six actions are
fold, check/call, minimum raise, 2.5x raise, pot raise, and all-in raise. Raise
sizing buckets are clamped to the engine's legal raise-to range.
Numeric features are normalized or bounded to keep the network's inputs on a
consistent scale.
Opponent tendencies are visible public statistics that update during play, so
the state can reflect changes in observed behavior without identifying a player.

The terminal reward is the agent's net result for the hand in big blinds (BB).
Intermediate rewards are zero. The environment samples opponent styles from the
training split of `Dataset/poker_hands_by_player.csv`; it uses style labels only,
not decision actions, names, or player IDs. The test split is reserved for
evaluation. The CSV's actual labels are Rock, Tight-Passive, Tight-Aggressive,
Balanced Reg, Loose-Passive, Loose-Aggressive, and Maniac.
For the requested poker terminology, Rock is the Nit proxy, Tight-Aggressive is
TAG, Loose-Aggressive is LAG, and Loose-Passive is the Calling Station proxy.

## Step 3: DQN

[`model.py`](./model.py) defines a Q-network that predicts one value per action and
a separate **target network**, a delayed copy used to stabilize learning. A
Q-value is the estimated discounted reward for taking an action in a state. The
**replay buffer** stores transitions and samples earlier examples for training.
An **epsilon-greedy policy** randomly explores with probability epsilon and
otherwise chooses the highest-valued legal action; **epsilon decay** gradually
lowers that exploration probability.

The masked Bellman target is
`q(s,a) = r + gamma * max Q(s', a')`: `r` is the hand reward, `gamma` discounts
future returns, and the maximum is taken only over legal actions in the next
state. Terminal hands do not bootstrap from a next-state Q-value.

## Steps 4-5: training and evaluation

From `PokerPilot-Website/` (Python 3.11+):

```bash
python -m pip install -r requirements.txt
python -m dqn_model_surya.train --hands 20000 --seed 1407
python -m dqn_model_surya.evaluate --hands 10000 --seed 2507
```

Training uses deterministic Python, NumPy, PyTorch, engine, and bot seeds. The
evaluation command refuses fewer than 10,000 hands, uses unseen engine seeds and
the dataset's test-split style mix, and reports net BB/100 plus a 95% bootstrap
confidence interval over independent 100-hand evaluation blocks (resampling
blocks estimates how uncertain the measured rate is). It also reports the
profitable, break-even, and losing hand percentages with block-bootstrap
confidence intervals. These are outcome rates, not decision accuracy: the
dataset contains no ground-truth labels for the correct action in a poker
decision. Training does not consume the test split.

To improve performance, compare checkpoints by held-out BB/100 and its
confidence interval, not by hand win percentage alone: a small number of large
losses can outweigh many small winning hands. A longer run such as 200,000
training hands is a reasonable experiment, but more hands do not guarantee
improvement. Use a separate validation setup for tuning, and reserve the test
split for a final unbiased comparison.

The website's `/api/advice` endpoint and **Ask AI Advisor** button use
`checkpoints/pokerpilot_dqn.pt` after training. The existing WebSocket chat relay
is unchanged and continues to carry player-to-player chat.

## Gemini textual explanations

After the DQN selects a legal action, the website can ask Gemini to explain the
recommendation in plain English. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and
restart Flask. The optional `GEMINI_MODEL` variable defaults to
`gemini-2.5-flash`. If the API key is missing or Gemini is unavailable, the
advisor returns a short built-in fallback instead of failing.

The browser displays the action and textual explanation; it no longer presents
DQN's mathematical Q-value to the player.

To test the live integration, install dependencies, export the key in the same
terminal that starts Flask, and run the website:

```bash
python -m pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python app.py
```

Start a game in the browser, choose **XGBoost + Gemini** or **DQN + Gemini**
in the advisor selector, then click **Ask AI Advisor** on your turn. Verify that
a plain-language explanation appears. The XGBoost version uses
`pokerpilot_model/action_model.json` plus `sizing_model.json` to choose the
action and amount, while Gemini only explains that result. The DQN version uses
its trained checkpoint instead. Set `GEMINI_MODEL` only if you want a
model other than `gemini-2.5-flash`. Do not commit the key. Without the key or
if Gemini fails, the endpoint still returns its built-in fallback explanation.

## Step 6: comments and learning notes

Each Python module labels its implementation step in the module comment, and
docstrings explain the important choices without assuming prior reinforcement
learning experience. The Q-value formula and the terms used by the code are
explained above.
