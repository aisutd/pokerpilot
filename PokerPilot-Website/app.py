"""Poker Pilot: play 6-max No-Limit Hold'em against five AI players, powered by pokerkit.

Run with:  python app.py   then open http://127.0.0.1:5000
(other devices on the same Wi-Fi: http://<this PC's IP>:5000)
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sys
import threading
import time
from pathlib import Path

# The DQN and XGBoost opponent use separate OpenMP runtimes. A single worker
# avoids native thread-runtime crashes when a process uses both model stacks.
os.environ.setdefault('OMP_NUM_THREADS', '1')

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / '.env')
load_dotenv(ROOT_DIR / 'backend' / '.env')
from backend.auth_routes import get_user, register_auth, require_auth, require_csrf  # noqa: E402

from engine.bots import DIFFICULTIES
from engine.game import HUMAN_SEAT, GameError, Table
from engine.opponent_models import OPPONENT_MODELS, list_models

STATIC_DIR = Path(__file__).parent / 'static'
CARD_ASSET_DIR = ROOT_DIR / 'Assets' / 'Cards'
CHAT_BUILD_DIR = Path(__file__).resolve().parents[1] / 'client' / 'dist'
ADVISOR_CHECKPOINT = Path(__file__).parent / 'dqn_model_surya' / 'checkpoints' / 'pokerpilot_dqn.pt'
MAX_MATCHES = 500
IDLE_SECONDS = 6 * 60 * 60

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
app.secret_key = os.environ.get('POKERPILOT_SECRET') or secrets.token_hex(32)
app.config['SESSION_COOKIE_NAME'] = 'pokerpilot_singleplayer_session'
register_auth(app)

matches: dict[str, tuple[Table, float]] = {}
matches_lock = threading.Lock()
match_chat_rooms: dict[str, str] = {}


def gemini_explanation(view: dict, action: str, label: str) -> str:
    """Turn the numerical poker recommendation into a concise plain-language explanation."""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        return f"The advisor recommends {label.lower()}. Review your position and the betting action before choosing."

    try:
        from google import genai

        prompt = f"""
You are a poker coach helping a beginner. Explain the recommended action in plain English.
Use 2-4 short sentences. Do not use equations, percentages, mathematical notation, or DQN/Q-values.
Mention the most important table factors naturally, but do not claim certainty about hidden cards.
Recommended action: {label}
Table state (JSON):
{json.dumps(view, ensure_ascii=False)[:6000]}
"""
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
            contents=prompt,
        )
        text = getattr(response, 'text', None)
        return text.strip() if text and text.strip() else f"The advisor recommends {label.lower()}."
    except Exception:
        app.logger.exception('Gemini explanation generation failed.')
        return f"The advisor recommends {label.lower()}. Review your position and the betting action before choosing."


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


@app.errorhandler(GameError)
def handle_game_error(error: GameError):
    return jsonify({'error': str(error)}), 400


@app.get('/')
def index():
    try:
        if get_user() is None:
            return redirect(url_for('pokerpilot_auth.auth_page', next='/'))
    except Exception:
        app.logger.exception('Unable to check the account session.')
        return jsonify({'detail': 'Authentication storage is unavailable.'}), 503
    return send_from_directory(STATIC_DIR, 'index.html')


@app.get('/cards/deck.svg')
def card_deck_asset():
    return send_from_directory(CARD_ASSET_DIR, 'deck.svg', mimetype='image/svg+xml')


@app.get('/chat')
@app.get('/chat/')
@require_auth
def chat_index():
    return send_from_directory(CHAT_BUILD_DIR, 'index.html')


@app.get('/chat/<path:filename>')
def chat_asset(filename: str):
    return send_from_directory(CHAT_BUILD_DIR, filename)


@app.get('/api/chat-room')
@require_auth
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


@app.get('/api/models')
@require_auth
def models():
    """AI models the player can pick for the opponents, and which are ready."""

    return jsonify({'models': list_models()})


@app.post('/api/new')
@require_auth
@require_csrf
def new_match():
    payload = request.get_json(silent=True) or {}
    difficulty = payload.get('difficulty')
    model = payload.get('model', 'rules')

    if difficulty not in DIFFICULTIES:
        raise GameError(f'Difficulty must be one of: {", ".join(DIFFICULTIES)}.')

    if model not in OPPONENT_MODELS:
        raise GameError(f'AI model must be one of: {", ".join(OPPONENT_MODELS)}.')

    match = Table(difficulty, model=model)
    match_id = secrets.token_urlsafe(16)

    with matches_lock:
        prune()
        matches[match_id] = (match, time.time())
        match_chat_rooms[match_id] = secrets.token_urlsafe(24)

    session['match_id'] = match_id

    with match.lock:
        return jsonify(match.response())


@app.get('/api/state')
@require_auth
def state():
    if session.get('match_id') not in matches:
        return jsonify(None)  # no match yet; the client stays in the lobby

    match = current_match()

    with match.lock:
        return jsonify({'frames': [], 'view': match.view()})


@app.post('/api/advice')
@require_auth
@require_csrf
def advice():
    match = current_match()

    with match.lock:
        if match.actor_seat != HUMAN_SEAT:
            raise GameError('The AI advisor is available only on your turn.')
        if not ADVISOR_CHECKPOINT.is_file():
            return jsonify({
                'error': 'No trained DQN checkpoint is installed. Train the agent first.',
            }), 503

        try:
            from dqn_model_surya.env import (
                ACTION_NAMES,
                action_to_engine_action,
                legal_action_mask,
                observation_from_table,
            )
            from dqn_model_surya.model import choose_greedy_action, load_policy
        except ImportError:
            app.logger.exception('PyTorch or the DQN advisor package is unavailable.')
            return jsonify({'error': 'The DQN advisor dependencies are not installed.'}), 503

        network, metadata = load_policy(
            str(ADVISOR_CHECKPOINT),
            ADVISOR_CHECKPOINT.stat().st_mtime_ns,
        )
        state_vector = observation_from_table(match)
        if (
                metadata.get('state_size') != len(state_vector)
                or metadata.get('action_count') != len(ACTION_NAMES)
                or metadata.get('action_names') != list(ACTION_NAMES)
        ):
            raise RuntimeError('The trained DQN checkpoint uses incompatible state/actions.')
        mask = legal_action_mask(match)
        chosen, q_values = choose_greedy_action(network, state_vector, mask)
        action, amount = action_to_engine_action(match, chosen)
        if action == 'fold':
            label = 'Fold'
        elif action == 'check_call':
            label = 'Check / Call'
        else:
            label = f'Raise to ${amount / 100:,.2f}'

        explanation = gemini_explanation(match.view(), action, label)
        return jsonify({
            'action': action,
            'amount': amount,
            'label': label,
            'q_value_bb': q_values[chosen],
            'explanation': explanation,
        })


@app.post('/api/action')
@require_auth
@require_csrf
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
@require_auth
@require_csrf
def next_hand():
    match = current_match()

    with match.lock:
        match.next_hand()

        return jsonify(match.response())


if __name__ == '__main__':
    # 0.0.0.0 lets phones and other computers on the same Wi-Fi connect.
    app.run(host=os.environ.get('HOST', '0.0.0.0'), port=int(os.environ.get('PORT', 5000)), threaded=True)
