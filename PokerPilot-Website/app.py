"""Poker Pilot: play 6-max No-Limit Hold'em against five AI players, powered by pokerkit.

Run with:  python app.py   then open http://127.0.0.1:5000
(other devices on the same Wi-Fi: http://<this PC's IP>:5000)
"""

from __future__ import annotations

import math
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from datetime import timedelta

from flask import Flask, jsonify, request, send_from_directory, session

from engine.bots import DIFFICULTIES
from engine.game import GameError, Table
from leaderboard_store import LeaderboardStore

STATIC_DIR = Path(__file__).parent / 'static'
CHAT_BUILD_DIR = Path(__file__).resolve().parents[1] / 'client' / 'dist'
MAX_MATCHES = 500
IDLE_SECONDS = 6 * 60 * 60

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
store = LeaderboardStore(os.environ.get('POKERPILOT_DB', str(Path(__file__).parent / 'instance' / 'leaderboard.sqlite3')))
app.config['LEADERBOARD_STORE'] = store
app.secret_key = os.environ.get('POKERPILOT_SECRET') or store.session_secret()
app.config.update(PERMANENT_SESSION_LIFETIME=timedelta(days=365),
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

matches: dict[str, tuple[Table, float]] = {}
matches_lock = threading.Lock()
match_chat_rooms: dict[str, str] = {}


def player_profile():
    database = app.config['LEADERBOARD_STORE']
    profile = database.profile(session.get('player_id'))
    if profile is None:
        profile = database.create_player()
        session['player_id'] = profile['id']
    session.permanent = True
    return profile


def persist_results(match):
    app.config['LEADERBOARD_STORE'].save_rounds(
        match.player_id, match.leaderboard.game_id, match.settled_rounds)


def match_response(match, frames=True):
    persist_results(match)
    response = match.response() if frames else {'frames': [], 'view': match.view()}
    response['global_leaderboard'] = app.config['LEADERBOARD_STORE'].standings(match.player_id)
    return jsonify(response)


@app.route('/api/profile', methods=['GET', 'POST'])
def profile_endpoint():
    profile = player_profile()
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        try:
            profile = app.config['LEADERBOARD_STORE'].rename_player(profile['id'], payload.get('name'))
        except ValueError as error:
            raise GameError(str(error)) from error
    return jsonify(profile)


@app.get('/api/leaderboard')
def global_leaderboard():
    return jsonify(app.config['LEADERBOARD_STORE'].standings(session.get('player_id')))


def prune() -> None:
    """Drop idle matches, then the oldest ones if there are too many."""

    now = time.time()

    for key, (_, seen) in list(matches.items()):
        if now - seen > IDLE_SECONDS:
            del matches[key]
            match_chat_rooms.pop(key, None)

    while len(matches) >= MAX_MATCHES:
        oldest = min(matches, key=lambda key: matches[key][1])
        del matches[oldest]
        match_chat_rooms.pop(oldest, None)


def current_match() -> Table:
    match_id = session.get('match_id')

    with matches_lock:
        entry = matches.get(match_id) if match_id else None

        if entry is None:
            raise GameError('No game in progress. Pick a difficulty to start.')

        matches[match_id] = (entry[0], time.time())

    return entry[0]


@app.errorhandler(sqlite3.Error)
def handle_database_error(error):
    app.logger.error('Could not save or read leaderboard results: %s', error)
    return jsonify({'error': 'Leaderboard storage is temporarily unavailable. Please retry.'}), 503


@app.errorhandler(GameError)
def handle_game_error(error: GameError):
    return jsonify({'error': str(error)}), 400


@app.get('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.get('/leaderboard')
@app.get('/leaderboard/')
def leaderboard_page():
    return send_from_directory(STATIC_DIR, 'leaderboard.html')


@app.get('/chat')
@app.get('/chat/')
def chat_index():
    return send_from_directory(CHAT_BUILD_DIR, 'index.html')


@app.get('/chat/<path:filename>')
def chat_asset(filename: str):
    return send_from_directory(CHAT_BUILD_DIR, filename)


@app.get('/api/chat-room')
def chat_room():
    match_id = session.get('match_id')

    with matches_lock:
        entry = matches.get(match_id) if match_id else None
        if entry is None:
            return jsonify({'error': 'Start a game before opening table chat.'}), 400

        matches[match_id] = (entry[0], time.time())
        room_id = match_chat_rooms.get(match_id)
        if room_id is None:
            room_id = secrets.token_urlsafe(24)
            match_chat_rooms[match_id] = room_id

    return jsonify({'room': room_id})


@app.post('/api/new')
def new_match():
    payload = request.get_json(silent=True) or {}
    difficulty = payload.get('difficulty')

    if difficulty not in DIFFICULTIES:
        raise GameError(f'Difficulty must be one of: {", ".join(DIFFICULTIES)}.')

    profile = player_profile()
    match = Table(difficulty)
    match.player_id = profile['id']
    match_id = secrets.token_urlsafe(16)

    with matches_lock:
        prune()
        matches[match_id] = (match, time.time())
        match_chat_rooms[match_id] = secrets.token_urlsafe(24)

    # A replacement game closes the previous board before starting fresh.
    previous_id = session.get('match_id')
    previous = matches.get(previous_id)
    if previous is not None:
        with previous[0].lock:
            persist_results(previous[0])
            previous[0].finish_session()
    session['match_id'] = match_id

    with match.lock:
        return match_response(match)


@app.get('/api/state')
def state():
    if session.get('match_id') not in matches:
        return jsonify(None)  # no match yet; the client stays in the lobby

    match = current_match()

    with match.lock:
        return match_response(match, frames=False)


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

        return match_response(match)


@app.post('/api/quit')
def quit_match():
    match = current_match()
    with match.lock:
        persist_results(match)
        match.finish_session()
        return jsonify({'leaderboard': match.leaderboard.snapshot()})


@app.post('/api/next')
def next_hand():
    match = current_match()

    with match.lock:
        match.next_hand()

        return match_response(match)


if __name__ == '__main__':
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', 5000)), threaded=True)
