"""Poker Pilot: play 6-max No-Limit Hold'em against five AI players, powered by pokerkit.

Run with:  python app.py   then open http://127.0.0.1:5000
(other devices on the same Wi-Fi: http://<this PC's IP>:5000)
"""

from __future__ import annotations

import math
import os
import secrets
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, session

from engine.bots import DIFFICULTIES
from engine.game import GameError, Table

STATIC_DIR = Path(__file__).parent / 'static'
MAX_MATCHES = 500
IDLE_SECONDS = 6 * 60 * 60

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
app.secret_key = os.environ.get('POKERPILOT_SECRET') or secrets.token_hex(32)

matches: dict[str, tuple[Table, float]] = {}
matches_lock = threading.Lock()


def prune() -> None:
    """Drop idle matches, then the oldest ones if there are too many."""

    now = time.time()

    for key, (_, seen) in list(matches.items()):
        if now - seen > IDLE_SECONDS:
            del matches[key]

    while len(matches) >= MAX_MATCHES:
        oldest = min(matches, key=lambda key: matches[key][1])
        del matches[oldest]


def current_match() -> Table:
    match_id = session.get('match_id')

    with matches_lock:
        entry = matches.get(match_id) if match_id else None

        if entry is None:
            raise GameError('No game in progress. Pick a difficulty to start.')

        matches[match_id] = (entry[0], time.time())

    return entry[0]


@app.errorhandler(GameError)
def handle_game_error(error: GameError):
    return jsonify({'error': str(error)}), 400


@app.get('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.post('/api/new')
def new_match():
    payload = request.get_json(silent=True) or {}
    difficulty = payload.get('difficulty')

    if difficulty not in DIFFICULTIES:
        raise GameError(f'Difficulty must be one of: {", ".join(DIFFICULTIES)}.')

    match = Table(difficulty)
    match_id = secrets.token_urlsafe(16)

    with matches_lock:
        prune()
        matches[match_id] = (match, time.time())

    session['match_id'] = match_id

    with match.lock:
        return jsonify(match.response())


@app.get('/api/state')
def state():
    if session.get('match_id') not in matches:
        return jsonify(None)  # no match yet; the client stays in the lobby

    match = current_match()

    with match.lock:
        return jsonify({'frames': [], 'view': match.view()})


@app.post('/api/action')
def action():
    payload = request.get_json(silent=True) or {}
    kind = payload.get('action')
    amount = payload.get('amount')

    if kind == 'raise':
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise GameError('A raise needs a numeric amount.')

        if not math.isfinite(amount) or amount != int(amount):
            raise GameError('Bet sizes must be whole cents.')

        amount = int(amount)
    else:
        amount = None

    match = current_match()

    with match.lock:
        match.human_action(kind, amount)

        return jsonify(match.response())


@app.post('/api/next')
def next_hand():
    match = current_match()

    with match.lock:
        match.next_hand()

        return jsonify(match.response())


if __name__ == '__main__':
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', 5000)), threaded=True)
