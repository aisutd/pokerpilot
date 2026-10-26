# PokerPilot

PokerPilot is a browser Texas Hold'em app with persistent local accounts, play
against Easy/Medium/Hard bots, and private online rooms for 2–6 human players.
PokerKit remains the sole rules engine. Chips are fictional; the app has no
money, deposits, or payment features.

## Architecture

- **FastAPI** (`backend/main.py`) is the primary app server for authentication,
  AI games, private rooms, and multiplayer WebSockets.
- **SQLite** stores users and revocable sessions in
  `backend/pokerpilot.sqlite3` by default. The schema is isolated in
  `backend/auth.py` so the storage can later be replaced.
- **PokerKit** and the existing engine/bots in `PokerPilot-Website/engine/`
  implement betting, hand resolution, Easy/Medium/Hard bot play, and the
  multi-human table. The original Flask server is preserved as a legacy entry
  point; it is not used by the FastAPI workflow.
- **Frontend:** the existing six-seat HTML/CSS/JavaScript table is reused for
  both modes, with login, signup, and room controls added. `client/` and
  `chat-api/` remain the separate React chat experiment; they are not required
  to run the poker app.

Rooms and active games are in memory, so run one Uvicorn worker. Account data
survives restarts; active poker games and rooms do not.

## Run locally (macOS/Linux)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The first request initializes SQLite. FastAPI
docs are at <http://127.0.0.1:8000/docs>.

Optional environment settings:

```bash
export POKERPILOT_DATABASE="$PWD/backend/pokerpilot.sqlite3"
export POKERPILOT_COOKIE_SECURE=0  # set to 1 behind HTTPS
```

Do not commit local database files. For a local secure-cookie test, use an
HTTPS development proxy and set `POKERPILOT_COOKIE_SECURE=1`.

## Accounts and sessions

Register with a unique username and email and a password of at least 12
characters. Passwords are stored as salted scrypt hashes, never as plaintext.
Login accepts a username or email. Sessions use random opaque tokens stored
hashed in SQLite and sent only in an HttpOnly, SameSite=Lax cookie. Mutating
requests require the CSRF cookie value in `X-CSRF-Token`; requests with a
cross-origin `Origin` are rejected. Login failures are rate-limited per
username/IP in each server process.

## Play against AI

Sign in, choose Easy, Medium, or Hard, then use the table controls. The app
reuses the existing five bots, game animations, hand history, and PokerKit
rules. Your current AI game resumes after a page refresh while the server is
running. Your opponent cards remain hidden until the existing showdown rules
reveal them.

## Play online with friends

1. Sign in on two or more independent browsers.
2. One player selects **Create Room**, then shares the six-character code.
3. Friends enter the code and select **Join Room**.
4. The host starts after at least two humans join. The existing table UI is
   reused; the first seat in each player's private view is that player.
5. Players act from their own browser. The backend validates turn and bet
   legality, applies actions to the shared PokerKit state, and broadcasts each
   player's personalized view. Opponents' hole cards are not included.

The WebSocket endpoint is `/ws/rooms/{room_code}`. It authenticates from the
same session cookie and checks the browser origin and room membership. Clients
send `{"type":"player_action","action":"check_call","version":N}` or
`{"type":"player_action","action":"raise","amount":AMOUNT,"version":N}`;
amounts are integer cents. Only the authenticated seat can act. The server
broadcasts `player_action` and a personalized `game_state` with a sequence
version. Reconnecting to the same account and room restores the current view.
When a disconnected player is up, the server checks for free or folds facing
a bet so a hand cannot remain blocked.

Room creation and joining are also available at `POST /api/rooms` and
`POST /api/rooms/join`. A room has at most six players; bots do not occupy
multiplayer seats. The host controls the first hand and subsequent hands.

## Tests

```bash
source .venv/bin/activate
pytest backend/tests
PYTHONPATH=PokerPilot-Website python -m unittest discover -s PokerPilot-Website/tests
```

The FastAPI tests cover account registration, duplicate rejection, login,
logout, protected gameplay, AI game startup, room creation/joining, WebSocket
turn checks, and per-player private-card views. PokerKit engine tests cover
legacy bot modes plus a complete two-human hand and private-card separation.

## Other existing components

- `PokerPilot-Website/app.py` is the preserved legacy Flask AI server. For its
  old interface, follow `PokerPilot-Website/README.md` and install that
  directory's requirements.
- `client/` is the React chat UI and `chat-api/` is its Node WebSocket relay.
  This chat relay is not an authoritative game server and is not started by
  the FastAPI workflow.
- `pokerpilot_model/` contains the separate offline XGBoost decision model and
  its generated Markdown evaluation reports. It is not used as a bot in this
  phase; the existing difficulty-based bots remain active.

## Phase 1 boundaries

This local build assumes one FastAPI worker because rooms, sockets, and active
tables are process memory. It is suitable for development and integration
testing, not multi-worker or public production deployment. Add shared room
state, persistent game recovery, TLS/reverse-proxy configuration, and
distributed rate limiting before hosting public rooms. Active players may
reconnect after a disconnect. Explicit leave is refused during a live hand;
logout/disconnect retains the seat and automatically checks for free or folds
when that seat is next to act. This avoids silently removing a player from a
shared hand.
