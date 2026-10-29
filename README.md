# PokerPilot

PokerPilot includes a single-player poker app, a multiplayer room app, shared
table chat, and model/advisor experiments. The game apps use PokerKit. Accounts
and sessions are stored by the server in the existing Supabase PostgreSQL
schema `pokerpilot`.

## Supabase authentication setup

The backend connects with `DATABASE_URL`; the connection string and database
password must stay on the server. Copy `.env.example` to `.env`, then set the
connection string from the Supabase project's **Connect** panel. Use a direct
or session-pooler PostgreSQL URL with TLS enabled. Do not commit `.env`.

Authentication reuses `pokerpilot.users` and `pokerpilot.sessions`; it does not
create a second account system or duplicate tables. The expected columns are:

- `users`: `id`, `username`, `email`, `password_hash`, `created_at`
- `sessions`: `token_hash`, `user_id`, `expires_at`

Username and email must each have a case-insensitive unique index. First check
for conflicting existing names/emails with these queries, then apply
[`backend/migrations/001_auth_uniqueness.sql`](backend/migrations/001_auth_uniqueness.sql)
in the Supabase SQL editor. The migration adds indexes to the existing table;
it does not create tables or copy user data. The auth health endpoint checks
the schema without changing it:

```sql
SELECT lower(username), count(*) FROM pokerpilot.users GROUP BY 1 HAVING count(*) > 1;
SELECT lower(email), count(*) FROM pokerpilot.users GROUP BY 1 HAVING count(*) > 1;
```

- `http://127.0.0.1:5000/api/auth/health`
- `http://127.0.0.1:5001/api/auth/health`

The backend hashes passwords with salted scrypt, stores only a hash of each
random session token, and expires sessions after seven days. Both game apps
share the same database-backed session cookie. Set
`POKERPILOT_COOKIE_SECURE=1` when serving over HTTPS.

## Run the poker apps on macOS/Linux

Python 3.11+ and Node.js 20+ are needed. From the repository root:

```bash
cp .env.example .env
# Edit .env and set DATABASE_URL from Supabase.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r PokerPilot-Website/requirements.txt
python -m pip install -r Multiplayer-Mode/requirements.txt
python -m pip install -r backend/requirements.txt
```

Start each app in its own terminal from the repository root:

```bash
python PokerPilot-Website/app.py
```

The single-player lobby is at <http://127.0.0.1:5000>. Sign in or create an
account; successful authentication returns to the lobby. Existing AI model
selection and the DQN advisor remain available as before.

For multiplayer table rooms and the existing chat UI, build the chat frontend
and start the chat relay:

```bash
npm ci --prefix Multiplayer-Mode/client
npm run build --prefix Multiplayer-Mode/client
npm ci --prefix chat-api
npm start --prefix chat-api
```

In another terminal:

```bash
python Multiplayer-Mode/app.py
```

The multiplayer lobby is at <http://127.0.0.1:5001>. Room membership is tied
to the signed-in account. WebSocket connections check the account session and
room membership; each player's private cards remain visible only to that
player and to players whose cards are revealed at showdown.

Both apps expose the shared login/signup screen at `/auth`. The UI returns the
user to the current lobby (including a room invite link) after login.

## Playing cards

The table apps share the scalable SVG deck at `Assets/Cards/deck.svg`, served
at `/cards/deck.svg`. It contains all 52 rank/suit faces and a PokerPilot card
back. Empty slots, face-down cards, opponent-card privacy, deal animation, and
winning-card highlights continue to use the existing table rendering logic.

## FastAPI model service and Docker

The existing FastAPI service remains the model/coach API; it is separate from
the Flask game apps. Its `/api/recommend` endpoint and XGBoost artifacts are
unchanged. To run the existing Docker setup:

```bash
cp backend/.env.example backend/.env
docker compose up -d --build
```

The model API is at <http://localhost:8000> and the sample frontend is at
<http://localhost:3000>. These containers do not start the two PokerKit game
apps; run those with the Python commands above.

## Tests

Install the Python dependencies above, then run:

```bash
python -m pytest backend/tests
OMP_NUM_THREADS=1 PYTHONPATH=PokerPilot-Website:. python -m pytest PokerPilot-Website/tests
PYTHONPATH=Multiplayer-Mode:. python -m pytest Multiplayer-Mode/tests
npm test --prefix chat-api
```

The auth tests exercise password hashing, parameterized SQL calls, account and
session API behavior, duplicate handling, rate limits, CSRF checks, and the
two-app login gates with a mocked data layer. Run the auth health endpoint
against your Supabase project to verify the real schema and connection.
