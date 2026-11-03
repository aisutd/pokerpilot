# Poker Pilot Online

Multiplayer version of Poker Pilot: up to ten people play No-Limit Texas
Hold'em in the same room, in real time over WebSockets. It uses the
same table UI and AI players as `PokerPilot-Website`; rules are handled by
[pokerkit](https://github.com/uoftcprg/pokerkit).

## Run it

Needs Python 3.11+ and, for the table chat, Node.js 20.19+ with npm.

Double-click `start.bat`. It sets everything up the first time, builds the
chat box, starts the chat server in a minimized window and opens the site.
Or manually:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm ci --prefix client
npm run build --prefix client
npm ci --prefix chat-api
npm start --prefix chat-api          # chat server, port 3000 (leave it running)
```

and in a second terminal:

```powershell
.venv\Scripts\python app.py
```

Then open <http://127.0.0.1:5001>. It uses port 5001 so it can run beside
the single-player site on 5000. Set `PORT` to change it.

## Playing with others

1. Enter your name and click **Create room**. You get a five-letter code.
2. Friends on the same Wi-Fi open the invite link shown in the room
   (`http://<your PC's IP>:5001/?room=CODE`) or type the code. If their
   browser can't connect, allow Python through Windows Firewall for private
   networks.
3. The host picks the opponents:
   - **Players only (PvP)**: humans only, at least two players. The host
     sets the table size (2 to 10 seats) with **Seats − / +**. The same
     control is in the table's top bar during the game. Seats can be added
     at any time and go to whoever is waiting. An empty seat can be removed
     between hands.
   - **Fill seats with AI (PvE)**: a 6-seat table where AI players (Easy /
     Medium / Hard) take every empty seat, so you can also play alone or
     with one friend.
4. The host clicks **Start game**.

**Testing alone:** open the invite link in a second browser tab. Each tab
is a separate player. A reload keeps your seat.

### Table rules

- Everyone starts with $200; blinds are $1/$2.
- You have 30 seconds per decision. When time runs out you check if that
  is free, otherwise you fold. Players who disconnect or leave get 4 seconds.
- The next hand starts 6 seconds after the last one ends.
- People who join mid-game sit down at the next hand. In PvE they take an
  AI player's seat (preferring one that is out of chips). If the table is
  full they wait for a seat.
- Out of chips? Click **Rebuy $200**. If only one player has chips left,
  the game pauses until someone rebuys, someone joins, or the host resets
  everyone's chips.
- If the host leaves, the next player becomes host.

### Table chat

Players in a room can talk in the **Chat with fellow Pokermates** window in
the lower-right corner, headed with the room code (Poker game: CODE). Drag it by its header; **−** / **+** folds it away.
It starts folded in the waiting room and on small screens. Your room name is
filled in; click **Join Chat**.

The chat box is the team's (`client/` and `chat-api/`, from the
`websockets` / `PokerEngine-frontend` branches of `aisutd/pokerpilot`),
unchanged. Each room gets a private chat id, sent only to that room's
members, so other rooms can't read or post in it. The chat server only
relays messages and keeps no history, so a reloaded page starts with an
empty chat. The page reaches the chat server through `/chat-ws` on the game
server, so friends only need port 5001 (or the ngrok link below).

### Playing over the internet (ngrok)

While `app.py` runs it keeps an ngrok tunnel to port 5001 open: it starts
ngrok if needed, and restarts it or reopens the tunnel if it goes away
(install ngrok and run `ngrok config add-authtoken <token>` once). The
server prints the public link, and room invite links use it. Opening the
site on this PC (`127.0.0.1`, `localhost`) or by its Wi-Fi address
redirects to the public link while the tunnel is up. ngrok keeps
running after the server stops, so restarting the server keeps the same
link; the link changes only when ngrok itself restarts. Visitors see
ngrok's one-time warning page first; they click **Visit Site**.

### Art

The pixel art comes from the team repo (`aisutd/pokerpilot`, `main`):
player portraits from `Assets/Profiles/PlayerImages/P1`-`P10`, and chips,
the button icons and the table from `chip_table_images`. The table screen
is laid out like the mock-up in that folder: on wide screens the whole
screen is the table art, with crowned name plates, cards standing above
them, your plate bottom left with your cards in a lit box in the middle,
CHECK / FOLD / CALL / BET-RAISE along the bottom, and MENU, SETTINGS,
SOUND and the chat button bottom right. BET / RAISE opens the bet sizes;
press it again to bet. Phones held upright get the same pieces stacked.

Everyone at a table gets their own portrait. Each AI player has a fixed
one, and people get one that nobody else in the room has. Bets and the pot
show chips coloured by amount (white $1, red $5, blue $10, green $25, black
$100, purple $500, yellow $1,000). Settings (sound, naming your hand, bet
size shortcuts, player stats) are remembered in the browser.

The originals have a checkerboard painted in where they should be
transparent, so `tools/make_art.py` cuts the portraits out and slices the
chip sheet into `static/art`. To rebuild it, copy those two folders from
the team repo into one folder, then run (needs Pillow):

```bash
python tools/make_art.py <folder with Assets/ and chip_table_images/> static/art
```

## Project layout

```
app.py            Flask server: room HTTP API, the /ws WebSocket, /chat/ and /chat-ws
engine/room.py    Rooms: seating, turn clocks, AI turns, per-player views
engine/bots.py    AI players (copied from PokerPilot-Website)
static/           Lobby, waiting room and table (no build step)
static/art/       Portraits, chips, felt and icon (built by tools/make_art.py)
client/           Chat box UI (React + Vite; built into client/dist)
chat-api/         Chat server: Node.js WebSocket relay on port 3000
tests/            Room tests:  python -W error::UserWarning -m unittest tests.test_room
                  Chat relay:  npm test --prefix chat-api
tools/make_art.py Builds static/art from the team's art (see Art above)
```

### Protocol

`POST /api/rooms {"name"}` creates a room and `POST /api/rooms/join
{"name", "code"}` joins one. Both return `{"code", "token"}`. The page then
opens `ws://host/ws?code=...&token=...` and sends JSON commands:

| Message | Who |
|---------|-----|
| `{"type": "settings", "mode": "pvp\|pve", "difficulty": "easy\|medium\|hard"}` | host, before start |
| `{"type": "seats", "count": 4}` | host, PvP only (2-10) |
| `{"type": "start"}` | host |
| `{"type": "action", "action": "fold\|check_call\|raise", "amount": 4500}` | player to act (amount in cents) |
| `{"type": "rebuy"}` | a player out of chips |
| `{"type": "restart"}` | host, while paused |
| `{"type": "leave"}` | anyone |

After every change, the server sends each connection
`{"type": "state", "room", "you", "game"}`. The view is personal: you see
only your own hole cards (plus cards shown at showdown), and the seats are
rotated so you sit at the bottom. Errors arrive as `{"type": "error"}`.

Rooms live in server memory. Restarting the server closes them all. Rooms
close when everyone leaves, or after 30 minutes with nobody connected.
