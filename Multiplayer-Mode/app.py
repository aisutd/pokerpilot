"""Poker Pilot Online: 6-max No-Limit Hold'em rooms over WebSockets. Friends
join a room by its code; in PvE rooms AI players fill the empty seats.

Run with:  python app.py   then open http://127.0.0.1:5001
(other devices on the same Wi-Fi: http://<this PC's IP>:5001)
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_sock import ConnectionClosed, Sock

from engine.room import GameError, Room

STATIC_DIR = Path(__file__).parent / 'static'
CHAT_BUILD_DIR = Path(__file__).parent / 'client' / 'dist'  # npm run build --prefix client
PORT = int(os.environ.get('PORT', 5001))
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


@app.errorhandler(GameError)
def handle_game_error(error: GameError):
    return jsonify({'error': str(error)}), 400


@app.get('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


# The team's chat box (client/, built with Vite) talks to the chat-api relay
# on port 3000; the page embeds it with the room's private chat id.
@app.get('/chat')
@app.get('/chat/')
def chat_index():
    return send_from_directory(CHAT_BUILD_DIR, 'index.html')


@app.get('/chat/<path:filename>')
def chat_asset(filename: str):
    return send_from_directory(CHAT_BUILD_DIR, filename)


@app.get('/api/info')
def info():
    address = lan_address()

    return jsonify({'lan_url': f'http://{address}:{PORT}' if address else None})


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
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=PORT, threaded=True)
