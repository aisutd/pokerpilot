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
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from flask import Flask, g, jsonify, redirect, request, send_from_directory, url_for
from flask_sock import ConnectionClosed, Sock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / '.env')
from backend.auth_routes import (  # noqa: E402
    SESSION_COOKIE, add_logout_hook, get_user, register_auth, require_auth, require_csrf,
)

from engine.room import GameError, Room

STATIC_DIR = Path(__file__).parent / 'static'
CARD_ASSET_DIR = ROOT_DIR / 'Assets' / 'Cards'
CHAT_BUILD_DIR = Path(__file__).parent / 'client' / 'dist'  # npm run build --prefix client
PORT = int(os.environ.get('PORT', 5001))
MAX_ROOMS = 200
IDLE_SECONDS = 30 * 60  # rooms with nobody connected are closed after this
CODE_LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # no I or O, they look like 1 and 0
CODE_LENGTH = 5

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
app.secret_key = os.environ.get('POKERPILOT_SECRET') or secrets.token_hex(32)
app.config['SESSION_COOKIE_NAME'] = 'pokerpilot_multiplayer_session'
sock = Sock(app)
register_auth(app)

rooms: dict[str, Room] = {}
rooms_lock = threading.Lock()
WEBSITE_DIR = ROOT_DIR / 'PokerPilot-Website'
ADVISOR_CHECKPOINT = WEBSITE_DIR / 'dqn_model_surya' / 'checkpoints' / 'pokerpilot_dqn.pt'
sys.path.insert(0, str(WEBSITE_DIR))


def multiplayer_advice(room: Room, seat: int) -> dict:
    if room.actor_seat != seat:
        raise GameError('The AI advisor is available only on your turn.')
    if not ADVISOR_CHECKPOINT.is_file():
        raise GameError('No trained DQN checkpoint is installed. Train the agent first.')

    try:
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        from dqn_model_surya.env import (
            ACTION_NAMES,
            action_to_engine_action,
            legal_action_mask,
            observation_from_table,
        )
        from dqn_model_surya.model import choose_greedy_action, load_policy
    except ImportError as error:
        raise GameError('The DQN advisor dependencies are not installed.') from error

    network, metadata = load_policy(
        str(ADVISOR_CHECKPOINT),
        ADVISOR_CHECKPOINT.stat().st_mtime_ns,
    )
    state_vector = observation_from_table(room, seat)
    if (
        metadata.get('state_size') != len(state_vector)
        or metadata.get('action_count') != len(ACTION_NAMES)
        or metadata.get('action_names') != list(ACTION_NAMES)
    ):
        raise GameError('The trained DQN checkpoint uses incompatible state/actions.')

    mask = legal_action_mask(room, seat)
    chosen, q_values = choose_greedy_action(network, state_vector, mask)
    action, amount = action_to_engine_action(room, chosen, seat)
    label = (
        'Fold'
        if action == 'fold'
        else 'Check / Call'
        if action == 'check_call'
        else f'Raise to ${amount / 100:,.2f}'
    )
    return {
        'action': action,
        'amount': amount,
        'label': label,
        'q_value_bb': q_values[chosen],
    }


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


def player_belongs_to_user(player, user) -> bool:
    return player is not None and str(player.account_id) == str(user['id'])


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
    try:
        if get_user() is None:
            target = request.full_path.rstrip('?')
            return redirect(url_for('pokerpilot_auth.auth_page', next=target if target.startswith('/') else '/'))
    except Exception:
        app.logger.exception('Unable to check the account session.')
        return jsonify({'detail': 'Authentication storage is unavailable.'}), 503
    return send_from_directory(STATIC_DIR, 'index.html')


@app.get('/cards/deck.svg')
def card_deck_asset():
    return send_from_directory(CARD_ASSET_DIR, 'deck.svg', mimetype='image/svg+xml')


# The team's chat box (client/, built with Vite) talks to the chat-api relay
# on port 3000; the page embeds it with the room's private chat id.
@app.get('/chat')
@app.get('/chat/')
@require_auth
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
@require_auth
@require_csrf
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
        token = room.join(payload.get('name') or g.auth_user['username'], account_id=g.auth_user['id'])
        rooms[code] = room

    return jsonify({'code': code, 'token': token})


@app.post('/api/rooms/join')
@require_auth
@require_csrf
def join_room():
    payload = request.get_json(silent=True) or {}

    with rooms_lock:
        room = find_room(payload.get('code'))

    return jsonify({'code': room.code, 'token': room.join(
        payload.get('name') or g.auth_user['username'], account_id=g.auth_user['id']
    )})


@sock.route('/ws')
def play(ws):
    """One connection per browser tab. The client sends commands; the room
    pushes the whole state to everyone after every change."""

    token = request.args.get('token', '')
    origin = request.headers.get('Origin')
    if origin and urlsplit(origin).netloc.lower() != request.host.lower():
        ws.close(1008, 'Cross-origin WebSocket rejected')
        return

    try:
        user = get_user(request.cookies.get(SESSION_COOKIE))
    except Exception:
        app.logger.exception('WebSocket authentication storage is unavailable.')
        ws.close(1011, 'Authentication storage unavailable')
        return
    if user is None:
        ws.close(1008, 'Authentication required')
        return

    with rooms_lock:
        room = rooms.get(request.args.get('code', '').upper())

    player = room.players.get(token) if room is not None else None
    if room is None or player is None or player.left or not player_belongs_to_user(player, user):
        ws.send(json.dumps({'type': 'gone', 'message': 'That room has closed.'}))
        return

    room.connect(token, ws)

    try:
        while True:
            try:
                try:
                    current = get_user(request.cookies.get(SESSION_COOKIE))
                except Exception:
                    current = None
                if current is None or not player_belongs_to_user(player, current):
                    ws.close(1008, 'Authentication expired')
                    break
                raw_message = ws.receive(timeout=15)
                if raw_message is None:
                    continue
                message = json.loads(raw_message)
                kind = message.get('type')

                if kind == 'action':
                    room.act(token, message.get('action'), message.get('amount'))
                elif kind == 'advice':
                    ws.send(json.dumps({
                        'type': 'advice',
                        **multiplayer_advice(room, player.seat),
                    }))
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


def close_logged_out_user(user):
    target = str(user['id'])
    with rooms_lock:
        connections = []
        for room in rooms.values():
            with room.lock:
                for ws, player_token in list(room.sockets.items()):
                    player = room.players.get(player_token)
                    if player is not None and str(player.account_id) == target:
                        connections.append((room, ws))
    for room, ws in connections:
        room.disconnect(ws)
        try:
            ws.close(1008, 'Signed out')
        except Exception:
            pass


add_logout_hook(app, close_logged_out_user)


if __name__ == '__main__':
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=PORT, threaded=True)
