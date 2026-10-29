"""Poker Pilot Online: 6-max No-Limit Hold'em rooms over WebSockets. Friends
join a room by its code; in PvE rooms AI players fill the empty seats.

Run with:  python app.py   then open http://127.0.0.1:5001
(other devices on the same Wi-Fi: http://<this PC's IP>:5001)
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import simple_websocket
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_sock import ConnectionClosed, Sock

from engine.room import GameError, Room

STATIC_DIR = Path(__file__).parent / 'static'
CHAT_BUILD_DIR = Path(__file__).parent / 'client' / 'dist'  # npm run build --prefix client
PORT = int(os.environ.get('PORT', 5001))
CHAT_API_URL = os.environ.get('CHAT_API_URL', 'ws://127.0.0.1:3000/')
NGROK_API = 'http://127.0.0.1:4040/api/tunnels'  # the ngrok agent's local API
TUNNEL_CHECK_SECONDS = 5
MAX_ROOMS = 200
IDLE_SECONDS = 30 * 60  # rooms with nobody connected are closed after this
CODE_LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # no I or O, they look like 1 and 0
CODE_LENGTH = 5

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
sock = Sock(app)

rooms: dict[str, Room] = {}
rooms_lock = threading.Lock()


def prune() -> None:
    """Close empty rooms and rooms nobody has been connected to for a while."""

    now = time.time()

    for code, room in list(rooms.items()):
        idle = room.idle_since is not None and now - room.idle_since > IDLE_SECONDS

        if room.empty or idle:
            room.close()
            del rooms[code]


def find_room(code: object) -> Room:
    room = rooms.get(code.strip().upper()) if isinstance(code, str) else None

    if room is None:
        raise GameError('No room with that code. Check it and try again.')

    return room


def lan_address() -> str | None:
    """This computer's address on the local network, for sharing links."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('10.255.255.255', 1))  # no packet is sent
            return probe.getsockname()[0]
    except OSError:
        return None


def ngrok_tunnels() -> list[dict] | None:
    """The running ngrok agent's tunnels, or None if no agent is running."""

    try:
        with urlopen(NGROK_API, timeout=2) as response:
            return json.load(response)['tunnels']
    except (OSError, ValueError, KeyError):
        return None


def public_url() -> str | None:
    """This server's public ngrok link, if a tunnel to it is open."""

    for tunnel in ngrok_tunnels() or []:
        if tunnel.get('proto') == 'https' and tunnel['config']['addr'].endswith(f':{PORT}'):
            return tunnel['public_url']

    return None


def keep_tunnel() -> None:
    """Keep an ngrok tunnel to this server open the whole time it runs, so
    friends anywhere can join. Starts ngrok, and starts it again or reopens
    the tunnel if it goes away. An existing tunnel is reused, so the link
    survives server restarts."""

    ngrok = shutil.which('ngrok')

    if ngrok is None:
        print('ngrok is not installed, so there is no public link (see README).', flush=True)
        return

    process = None
    announced = None

    while True:
        url = public_url()

        if url is None:
            try:
                if ngrok_tunnels() is not None:
                    body = json.dumps({'name': f'pokerpilot-{PORT}', 'proto': 'http', 'addr': str(PORT)})
                    urlopen(Request(NGROK_API, body.encode(), {'Content-Type': 'application/json'}), timeout=5).close()
                elif process is None or process.poll() is not None:
                    process = subprocess.Popen(
                        [ngrok, 'http', str(PORT), '--log', 'false'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    )
            except OSError as error:
                print(f'ngrok: {error}', flush=True)
        elif url != announced:
            print(f'Public link (ngrok): {url}', flush=True)
            announced = url

        time.sleep(TUNNEL_CHECK_SECONDS)


@app.errorhandler(GameError)
def handle_game_error(error: GameError):
    return jsonify({'error': str(error)}), 400


@app.get('/')
def index():
    # Opened on this PC or the Wi-Fi address: move to the public ngrok link,
    # so the address bar always shows the link friends can use. Requests
    # that came through ngrok carry X-Forwarded-For and are served as usual.
    local = request.host.rsplit(':', 1)[0] in ('localhost', '127.0.0.1', lan_address())

    if local and 'X-Forwarded-For' not in request.headers and (url := public_url()):
        return redirect(url + request.full_path.rstrip('?'))

    return send_from_directory(STATIC_DIR, 'index.html')


# The team's chat box (client/, built with Vite) talks to the chat-api relay
# on port 3000 through /chat-ws below; the page embeds it with the room's
# private chat id.
@app.get('/chat')
@app.get('/chat/')
def chat_index():
    return send_from_directory(CHAT_BUILD_DIR, 'index.html')


@app.get('/chat/<path:filename>')
def chat_asset(filename: str):
    return send_from_directory(CHAT_BUILD_DIR, filename)


@sock.route('/chat-ws')
def chat_relay(ws):
    """Pass chat traffic through to chat-api, so the chat works over the same
    address and port as the game (LAN, or a tunnel such as ngrok)."""

    try:
        upstream = simple_websocket.Client.connect(f"{CHAT_API_URL}?room={quote(request.args.get('room', ''))}")
    except (OSError, simple_websocket.ConnectionError):
        return

    def forward_replies():
        try:
            while True:
                ws.send(upstream.receive())
        except ConnectionClosed:
            ws.close()

    threading.Thread(target=forward_replies, daemon=True).start()

    try:
        while True:
            upstream.send(ws.receive())
    except ConnectionClosed:
        pass
    finally:
        upstream.close()


@app.get('/api/info')
def info():
    address = lan_address()

    return jsonify({'lan_url': f'http://{address}:{PORT}' if address else None, 'public_url': public_url()})


@app.post('/api/rooms')
def create_room():
    payload = request.get_json(silent=True) or {}

    with rooms_lock:
        prune()

        if len(rooms) >= MAX_ROOMS:
            raise GameError('The server is full. Try again later.')

        code = ''.join(secrets.choice(CODE_LETTERS) for _ in range(CODE_LENGTH))

        while code in rooms:
            code = ''.join(secrets.choice(CODE_LETTERS) for _ in range(CODE_LENGTH))

        room = Room(code)
        token = room.join(payload.get('name'))
        rooms[code] = room

    return jsonify({'code': code, 'token': token})


@app.post('/api/rooms/join')
def join_room():
    payload = request.get_json(silent=True) or {}

    with rooms_lock:
        room = find_room(payload.get('code'))

    return jsonify({'code': room.code, 'token': room.join(payload.get('name'))})


@sock.route('/ws')
def play(ws):
    """One connection per browser tab. The client sends commands; the room
    pushes the whole state to everyone after every change."""

    token = request.args.get('token', '')

    with rooms_lock:
        room = rooms.get(request.args.get('code', '').upper())

    if room is None or token not in room.players or room.players[token].left:
        ws.send(json.dumps({'type': 'gone', 'message': 'That room has closed.'}))
        return

    room.connect(token, ws)

    try:
        while True:
            try:
                message = json.loads(ws.receive())
                kind = message.get('type')

                if kind == 'action':
                    room.act(token, message.get('action'), message.get('amount'))
                elif kind == 'start':
                    room.start_game(token)
                elif kind == 'settings':
                    room.update_settings(token, message.get('mode'), message.get('difficulty'))
                elif kind == 'seats':
                    room.resize(token, message.get('count'))
                elif kind == 'rebuy':
                    room.rebuy(token)
                elif kind == 'restart':
                    room.restart(token)
                elif kind == 'leave':
                    room.leave(token)
                    ws.send(json.dumps({'type': 'left'}))
                    break
                else:
                    raise GameError('Unknown message.')
            except (GameError, ValueError, AttributeError, TypeError) as error:
                message = str(error) if isinstance(error, GameError) else 'Bad message.'
                ws.send(json.dumps({'type': 'error', 'message': message}))
    except ConnectionClosed:
        pass
    finally:
        room.disconnect(ws)

        if room.empty:
            with rooms_lock:
                if rooms.get(room.code) is room:
                    room.close()
                    del rooms[room.code]


if __name__ == '__main__':
    threading.Thread(target=keep_tunnel, daemon=True).start()
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=PORT, threaded=True)
