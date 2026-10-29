# Poker Pilot

Play 6-max No-Limit Texas Hold'em in your browser against five AI players.
All game rules (blinds, betting order, legal bet sizes, side pots, showdowns
and hand ranking) are handled by
[pokerkit](https://github.com/uoftcprg/pokerkit).

## Run it

Requires Python 3.11+ (pokerkit needs 3.11 or newer), Node.js 20.19+ and npm.

**Windows:** double-click `start.bat`. It sets everything up the first time,
builds the chat UI, starts the WebSocket chat service and website, then opens
<http://127.0.0.1:5000>. Or manually:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd ..\client
npm install
npm run build
cd ..\chat-api
npm install
npm start
```

In a second terminal:

```powershell
cd PokerPilot-Website
.venv\Scripts\python app.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../client
npm install
npm run build
cd ../chat-api
npm install
npm start
```

In a second terminal:

```bash
cd PokerPilot-Website
.venv/bin/python app.py
```

Then open <http://127.0.0.1:5000>. The server also listens on your local
network, so phones on the same Wi-Fi can open `http://<your computer's IP>:5000`.
The chat API listens on port 3000. Set `VITE_CHAT_WS_URL` before building the
client if the WebSocket server is hosted at a different URL.

Run the poker tests with `python -m unittest tests.test_game tests.test_chat_room`
(from this folder, inside the virtual environment), and the chat relay test
with `npm test --prefix ../chat-api`.

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
The draggable **Chat with fellow Pokermates** window opens in the lower-right
corner while a game is active. It uses the React chat UI and WebSocket API
from the `websockets` branch, including emoji selection. Each game gets a
private chat room; chat messages are only broadcast to clients connected to
that game's room.

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
../client/        React chat UI (built into client/dist for Flask to serve)
../chat-api/      Node.js WebSocket chat relay (port 3000)
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

## Global leaderboard

The narrow sidebar shows real players across all games ranked by **total net
winnings** (wins add and losses subtract), plus rounds participated. AI opponents
are excluded. Ties share a rank. Results update after completed hands; unfinished
hands are not counted. Starting a new game does not reset totals. Other players'
results refresh every ten seconds. The final-results dialog shows the same global
rankings.

Choose a display name in the lobby. The signed browser session identifies the
player for one year and survives server restarts. These are browser profiles,
not password-protected accounts: another device or clearing cookies creates a
new profile. Names are unique, but a name alone cannot recover an old profile.

SQLite stores players and an immutable round ledger in
`instance/leaderboard.sqlite3` (excluded from Git). Each settled round has a unique
(game, round, player) key to prevent duplicate credit on retries or reloads.
Balances come from the poker engine, never a browser-supplied score. Amounts are
stored as integer cents. Back up the database to preserve results. Set
`POKERPILOT_DB` to override its path and `POKERPILOT_SECRET` to override the signing
key (otherwise a persistent random key is stored in the database). Restarting
still clears in-progress games; completed rounds remain saved.

This board is shared by users of this server. Public internet access requires
hosting this application with persistent database storage. Multiple independent
servers need a shared database rather than separate SQLite files.

API: `GET/POST /api/profile` reads/renames the current browser profile;
`GET /api/leaderboard` returns global standings. Game API responses also include
`global_leaderboard`. The existing per-table engine snapshot remains available
internally for settled game results.

Tests: `python -m unittest discover -s tests -v` (uses temporary databases).

### Global and this-game rankings

The sidebar has **Global** and **This game** buttons. Global aggregates real
players across games; This game ranks all six table players, including bots, by
net profit since that game started. Both count completed rounds. This game resets
when a new game starts; global totals remain saved.

### Leaderboard page

Open `/leaderboard` from **View leaderboards** on the game dashboard. The page
shows global rankings and current-game standings, refreshes every ten seconds,
and links back to the dashboard. It uses the same existing leaderboard APIs and
database as the sidebar. A game is not required to view global standings.
