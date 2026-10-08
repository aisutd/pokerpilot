# Poker Pilot

Play 6-max No-Limit Texas Hold'em in your browser against five AI players.
All game rules (blinds, betting order, legal bet sizes, side pots, showdowns
and hand ranking) are handled by
[pokerkit](https://github.com/uoftcprg/pokerkit).

## Run it

Requires Python 3.11+ (pokerkit needs 3.11 or newer).

**Windows:** double-click `start.bat`. It sets everything up the first time,
starts the server and opens <http://127.0.0.1:5000>. Close the black window
to stop it. Or manually:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open <http://127.0.0.1:5000>. The server also listens on your local
network, so phones on the same Wi-Fi can open `http://<your computer's IP>:5000`.

Run the tests with `python -m unittest tests.test_game` (from this folder,
inside the virtual environment).

## How to play

1. Pick **Easy**, **Medium** or **Hard** in the lobby.
2. You sit at a six-seat table with five AI players: MathNinja,
   CardShark88, RiverRat, LuckyLlama and TiltMaster. Everyone starts with
   $200; blinds are $1/$2. The dealer button moves clockwise every hand.
3. **Fold / Call / Raise** with the buttons or the keys **F / C / R**. Size
   bets with the slider, the amount box or the presets (Min, 1/2 Pot,
   3/4 Pot, Pot, All In). The arrow on the Raise button hides or shows the
   presets. Press **N** for the next hand.
4. Players who run out of chips leave the table. You win by taking every
   chip; the game ends if you go broke. Reloading the page resumes the game.

The numbers under each player are: % of hands played voluntarily / % of
hands raised before the flop / hands won. **History** shows the hand log.

## Difficulty

Difficulty sets how *skilled* all five AI players are. Each player also has
their own personality (TiltMaster is aggressive and bluffs a lot, RiverRat
calls too much, MathNinja is tight, and so on).

| Level  | How the AI plays |
|--------|------------------|
| Easy   | Misreads its hand strength, limps and calls far too often, rarely raises or bluffs. |
| Medium | Plays sound starting hands, bets good hands, respects pot odds, bluffs occasionally. |
| Hard   | Position-aware ranges, 3-bets, continuation bets, semi-bluffs, and estimates your likely hand range from how you have played. |

AI players see only their own cards, the board and the betting. They never
see your hole cards.

## Project layout

```
app.py            Flask server and JSON API
engine/game.py    6-max table logic on top of pokerkit's State
engine/bots.py    AI players (skill levels and personalities)
static/           Lobby, loading screen and table (HTML/CSS/JS, no build step)
tests/            Engine tests (python -m unittest)
start.bat         One-click launcher for Windows
```

### API

Amounts are integer cents.

| Method | Path          | Body |
|--------|---------------|------|
| POST   | `/api/new`    | `{"difficulty": "easy\|medium\|hard"}` |
| GET    | `/api/state`  | none |
| POST   | `/api/action` | `{"action": "fold\|check_call\|raise", "amount": 4500}` |
| POST   | `/api/next`   | none (deals the next hand) |

Each POST returns `{"frames": [...], "view": {...}}`: one frame per AI
action so the page can animate them, then the final state.

Games are kept in server memory and are lost when the server restarts. For
a deployment, set `POKERPILOT_SECRET` so session cookies survive restarts,
and run the app behind a production WSGI server such as waitress.
