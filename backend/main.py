"""FastAPI application for PokerPilot auth, AI play, and local multiplayer."""

from __future__ import annotations

import hmac
import os
import re
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PokerPilot-Website"))

from engine.bots import DIFFICULTIES  # noqa: E402
from engine.game import GameError, Table  # noqa: E402

from backend.auth import (  # noqa: E402
    SESSION_SECONDS, connect, create_user, find_user, init_db, new_session,
    public_user, revoke_session, session_user, verify_password,
)
from backend.rooms import Room, make_room_code  # noqa: E402

SESSION_COOKIE = "pokerpilot_session"
CSRF_COOKIE = "pokerpilot_csrf"
MAX_ROOM_PLAYERS = 6
ROOM_TTL = 60 * 60 * 6
rooms: dict[str, Room] = {}
room_guard = threading.RLock()
ai_tables: dict[int, Table] = {}
auth_attempts: dict[str, deque[float]] = defaultdict(deque)
auth_guard = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="PokerPilot", version="1.0.0", lifespan=lifespan)
STATIC = ROOT / "PokerPilot-Website" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Signup(BaseModel):
    username: str = Field(min_length=3, max_length=24)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=12, max_length=128)


class Login(BaseModel):
    identifier: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class DifficultyChoice(BaseModel):
    difficulty: str


class Action(BaseModel):
    action: str
    amount: int | None = None


def _secure_cookie() -> bool:
    return os.getenv("POKERPILOT_COOKIE_SECURE", "0").lower() in {"1", "true", "yes"}


def _set_auth_cookies(response: Response, user_id: int) -> None:
    token, _ = new_session(user_id)
    csrf = secrets.token_urlsafe(24)
    secure = _secure_cookie()
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="lax", path="/")
    response.set_cookie(CSRF_COOKIE, csrf, max_age=SESSION_SECONDS, httponly=False,
                        secure=secure, samesite="lax", path="/")


def current_user(request: Request) -> dict:
    row = session_user(request.cookies.get(SESSION_COOKIE))
    if row is None:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return public_user(row)


def csrf_guard(request: Request) -> None:
    cookie = request.cookies.get(CSRF_COOKIE, "")
    header = request.headers.get("x-csrf-token", "")
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF token is missing or invalid.")
    origin = request.headers.get("origin")
    if origin:
        origin_host = urlsplit(origin).netloc
        if not origin_host or origin_host.lower() != request.headers.get("host", "").lower():
            raise HTTPException(status_code=403, detail="Cross-origin request rejected.")


CurrentUser = Annotated[dict, Depends(current_user)]
Csrf = Annotated[None, Depends(csrf_guard)]


def _attempt_key(request: Request, identifier: str) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{ip}|{identifier.casefold()}"


def _throttle(key: str) -> None:
    now = time.monotonic()
    with auth_guard:
        if len(auth_attempts) > 5000:
            for old_key, old_attempts in list(auth_attempts.items()):
                while old_attempts and now - old_attempts[0] > 300:
                    old_attempts.popleft()
                if not old_attempts:
                    auth_attempts.pop(old_key, None)
        ip_key = "ip:" + key.split("|", 1)[0]
        for attempt_key in (key, ip_key):
            attempts = auth_attempts[attempt_key]
            while attempts and now - attempts[0] > 300:
                attempts.popleft()
            if len(attempts) >= 8:
                raise HTTPException(status_code=429, detail="Too many login attempts. Try again in a few minutes.")


def _record_failure(key: str) -> None:
    with auth_guard:
        now = time.monotonic()
        auth_attempts[key].append(now)
        auth_attempts["ip:" + key.split("|", 1)[0]].append(now)


def _clear_failures(key: str) -> None:
    with auth_guard:
        auth_attempts.pop(key, None)
        auth_attempts.pop("ip:" + key.split("|", 1)[0], None)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "fastapi"}


@app.get("/api/auth/csrf")
def csrf_seed(response: Response) -> dict:
    token = secrets.token_urlsafe(24)
    response.set_cookie(CSRF_COOKIE, token, max_age=SESSION_SECONDS, httponly=False,
                        secure=_secure_cookie(), samesite="lax", path="/")
    return {"csrf_token": token}


@app.post("/api/auth/signup")
def signup(data: Signup, response: Response, _: Csrf) -> dict:
    username = data.username.strip()
    email = data.email.strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,24}", username):
        raise HTTPException(status_code=422, detail="Username must be 3–24 letters, numbers, or underscores.")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    try:
        row = create_user(username, email, data.password)
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="That username or email is already registered.") from exc
        raise
    _set_auth_cookies(response, int(row["id"]))
    return {"user": public_user(row)}


@app.post("/api/auth/login")
def login(data: Login, request: Request, response: Response, _: Csrf) -> dict:
    identifier = data.identifier.strip()
    key = _attempt_key(request, identifier)
    _throttle(key)
    row = find_user(identifier)
    if row is None or not verify_password(data.password, row["password_hash"]):
        _record_failure(key)
        raise HTTPException(status_code=401, detail="Username/email or password is incorrect.")
    _clear_failures(key)
    _set_auth_cookies(response, int(row["id"]))
    return {"user": public_user(row)}


@app.get("/api/auth/me")
def me(user: CurrentUser) -> dict:
    return {"user": user}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response, _: Csrf, user: CurrentUser) -> dict:
    room = _user_room(user["id"])
    if room is not None:
        with room_guard:
            with room.lock:
                room.sockets.pop(user["id"], None)
                if room.status == "in_progress":
                    if room.host_id == user["id"] and len(room.players) > 1:
                        room.host_id = next((uid for uid in room.players if uid in room.sockets),
                                            next(uid for uid in room.players if uid != user["id"]))
                    _advance_disconnected_turns(room)
                    if room.game is not None and room.game.game_over:
                        room.status = "finished"
                else:
                    room.players.pop(user["id"], None)
                    if not room.players:
                        rooms.pop(room.code, None)
                    elif room.host_id == user["id"]:
                        room.host_id = next(iter(room.players))
                room.version += 1
        if room.players:
            event = "player_disconnected" if room.status == "in_progress" else "player_left"
            await _broadcast(room, {"type": event, "username": user["username"]})
    revoke_session(request.cookies.get(SESSION_COOKIE))
    ai_tables.pop(user["id"], None)
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True,
                           secure=_secure_cookie(), samesite="lax")
    response.delete_cookie(CSRF_COOKIE, path="/", secure=_secure_cookie(), samesite="lax")
    return {"ok": True}


def _prune_rooms() -> None:
    cutoff = time.monotonic() - ROOM_TTL
    with room_guard:
        for code, room in list(rooms.items()):
            with room.lock:
                if not room.sockets and room.last_active < cutoff:
                    rooms.pop(code, None)
        while len(rooms) >= 500:
            idle = [room for room in rooms.values() if not room.sockets]
            if not idle:
                break
            oldest = min(idle, key=lambda room: room.last_active)
            rooms.pop(oldest.code, None)


def _user_room(user_id: int) -> Room | None:
    with room_guard:
        for room in rooms.values():
            with room.lock:
                if user_id in room.players:
                    return room
    return None


def _room_or_404(code: str) -> Room:
    with room_guard:
        room = rooms.get(code.upper())
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found.")
    with room.lock:
        room.last_active = time.monotonic()
    return room


def _seat_for(room: Room, user_id: int) -> int:
    try:
        return list(room.players).index(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="You are not a member of this room.") from exc


def _room_snapshot(room: Room, user_id: int) -> dict:
    with room.lock:
        payload: dict[str, Any] = {"type": "room_state", "room": room.public_state()}
        if room.game is not None:
            seat = _seat_for(room, user_id)
            payload["type"] = "game_state"
            payload["game"] = room.game.view_for(seat)
        return payload


def _advance_disconnected_turns(room: Room) -> None:
    """Check through or fold seats whose browser is disconnected, never stall a hand."""
    if room.game is None or room.status != "in_progress":
        return
    user_ids = list(room.players)
    for _ in range(len(user_ids)):
        actor = room.game.actor_seat
        if actor is None or actor >= len(user_ids) or user_ids[actor] in room.sockets:
            return
        action = "check_call" if room.game.state.checking_or_calling_amount == 0 else "fold"
        room.game.human_action_for(actor, action)


async def _broadcast(room: Room, event: dict | None = None) -> None:
    with room.lock:
        sockets = list(room.sockets.items())
        messages = [(socket, _room_snapshot(room, uid)) for uid, socket in sockets]
    for socket, state in messages:
        try:
            if event:
                await socket.send_json(event)
            await socket.send_json(state)
        except Exception:
            pass


@app.post("/api/rooms")
async def create_room(user: CurrentUser, _: Csrf) -> dict:
    _prune_rooms()
    with room_guard:
        if _user_room(user["id"]):
            raise HTTPException(status_code=409, detail="Leave your current room before creating another.")
        if len(rooms) >= 500:
            raise HTTPException(status_code=503, detail="All room capacity is currently in use.")
        for _ in range(10):
            code = make_room_code()
            if code not in rooms:
                room = Room(code=code, host_id=user["id"], players={user["id"]: user["username"]})
                room.last_active = time.monotonic()
                rooms[code] = room
                return room.public_state()
    raise HTTPException(status_code=503, detail="Could not reserve a room code. Try again.")


class JoinRoom(BaseModel):
    room_code: str = Field(min_length=6, max_length=6)


@app.post("/api/rooms/join")
async def join_room(data: JoinRoom, user: CurrentUser, _: Csrf) -> dict:
    code = data.room_code.strip().upper()
    if not re.fullmatch(r"[A-HJ-NP-Z2-9]{6}", code):
        raise HTTPException(status_code=422, detail="Enter a valid six-character room code.")
    room = _room_or_404(code)
    with room_guard:
        with room.lock:
            if user["id"] in room.players:
                return room.public_state()
            if _user_room(user["id"]):
                raise HTTPException(status_code=409, detail="Leave your current room before joining another.")
            if room.status != "waiting":
                raise HTTPException(status_code=409, detail="This room has already started.")
            if len(room.players) >= MAX_ROOM_PLAYERS:
                raise HTTPException(status_code=409, detail="This room is full.")
            room.players[user["id"]] = user["username"]
            room.last_active = time.monotonic()
            room.version += 1
    await _broadcast(room, {"type": "player_joined", "username": user["username"]})
    return room.public_state()


@app.get("/api/rooms/current")
def current_room(user: CurrentUser) -> dict | None:
    room = _user_room(user["id"])
    if room is None:
        return None
    return _room_snapshot(room, user["id"])


@app.get("/api/rooms/{room_code}")
def get_room(room_code: str, user: CurrentUser) -> dict:
    room = _room_or_404(room_code)
    if user["id"] not in room.players:
        raise HTTPException(status_code=403, detail="You are not a member of this room.")
    return _room_snapshot(room, user["id"])


@app.post("/api/rooms/{room_code}/leave")
async def leave_room(room_code: str, user: CurrentUser, _: Csrf) -> dict:
    room = _room_or_404(room_code)
    with room_guard:
        with room.lock:
            if user["id"] not in room.players:
                raise HTTPException(status_code=403, detail="You are not a member of this room.")
            if room.status == "in_progress":
                raise HTTPException(status_code=409, detail="Finish the current hand before leaving. You can reconnect if disconnected.")
            room.players.pop(user["id"])
            room.sockets.pop(user["id"], None)
            if not room.players:
                rooms.pop(room.code, None)
            else:
                if room.host_id == user["id"]:
                    room.host_id = next(iter(room.players))
                room.last_active = time.monotonic()
                room.version += 1
    if room.players:
        await _broadcast(room, {"type": "player_left", "username": user["username"]})
    return {"ok": True}


@app.post("/api/rooms/{room_code}/start")
async def start_room(room_code: str, user: CurrentUser, _: Csrf) -> dict:
    room = _room_or_404(room_code)
    with room.lock:
        if room.host_id != user["id"]:
            raise HTTPException(status_code=403, detail="Only the host can start this game.")
        if room.status != "waiting":
            raise HTTPException(status_code=409, detail="This room has already started.")
        if len(room.players) < 2:
            raise HTTPException(status_code=409, detail="At least two players are needed to start.")
        room.game = Table("medium", human_names=list(room.players.values()), bot_count=0)
        room.status = "in_progress"
        room.version += 1
    await _broadcast(room, {"type": "game_started", "version": room.version})
    return _room_snapshot(room, user["id"])


@app.websocket("/ws/rooms/{room_code}")
async def room_socket(websocket: WebSocket, room_code: str) -> None:
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host", "").lower()
    if origin and urlsplit(origin).netloc.lower() != host:
        await websocket.close(code=1008, reason="Cross-origin WebSocket rejected")
        return
    user_row = session_user(websocket.cookies.get(SESSION_COOKIE))
    if user_row is None:
        await websocket.close(code=1008, reason="Authentication required")
        return
    user = public_user(user_row)
    with room_guard:
        room = rooms.get(room_code.upper())
    if room is None:
        await websocket.close(code=1008, reason="Room membership required")
        return
    with room.lock:
        if user["id"] not in room.players:
            await websocket.close(code=1008, reason="Room membership required")
            return
    await websocket.accept()
    with room.lock:
        room.sockets[user["id"]] = websocket
        room.last_active = time.monotonic()
        room.version += 1
        initial = _room_snapshot(room, user["id"])
    await websocket.send_json(initial)
    await _broadcast(room, {"type": "player_joined", "username": user["username"]})
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                await websocket.send_json({"type": "error", "message": "Room messages must be JSON objects."})
                continue
            event_type = data.get("type")
            error_message = None
            refresh = None
            event = None
            with room.lock:
                if room.status != "in_progress" or room.game is None:
                    error_message = "The game has not started."
                elif data.get("version") is not None and data["version"] != room.version:
                    error_message = "The table changed. State has been refreshed."
                    refresh = _room_snapshot(room, user["id"])
                else:
                    seat = _seat_for(room, user["id"])
                    try:
                        if event_type == "player_action":
                            kind = data.get("action")
                            amount = data.get("amount")
                            if kind == "raise":
                                if isinstance(amount, bool) or not isinstance(amount, int):
                                    raise GameError("Raise amount must be a whole number of cents.")
                            else:
                                amount = None
                            room.game.human_action_for(seat, kind, amount)
                        elif event_type == "next_hand":
                            if room.host_id != user["id"]:
                                raise GameError("Only the host can start the next hand.")
                            room.game.next_hand()
                        else:
                            raise GameError("Unknown room message.")
                    except GameError as exc:
                        error_message = str(exc)
                    if error_message is None:
                        _advance_disconnected_turns(room)
                        room.version += 1
                        if room.game.game_over:
                            room.status = "finished"
                        event = {"type": "player_action", "username": user["username"],
                                 "action": event_type, "version": room.version}
            if error_message:
                await websocket.send_json({"type": "error", "message": error_message})
                if refresh:
                    await websocket.send_json(refresh)
                continue
            await _broadcast(room, event)
    except WebSocketDisconnect:
        pass
    finally:
        with room.lock:
            if room.sockets.get(user["id"]) is websocket:
                room.sockets.pop(user["id"], None)
                room.last_active = time.monotonic()
                if room.host_id == user["id"] and len(room.players) > 1:
                    room.host_id = next((uid for uid in room.players if uid in room.sockets),
                                        next(uid for uid in room.players if uid != user["id"]))
                try:
                    _advance_disconnected_turns(room)
                    if room.game is not None and room.game.game_over:
                        room.status = "finished"
                    room.version += 1
                except GameError:
                    pass
        await _broadcast(room, {"type": "player_left", "username": user["username"]})


def _ai_table(user: dict) -> Table:
    table = ai_tables.get(user["id"])
    if table is None:
        raise HTTPException(status_code=400, detail="No AI game is in progress. Choose a difficulty first.")
    return table


@app.post("/api/ai/new")
@app.post("/api/new")
def new_ai_game(data: DifficultyChoice, user: CurrentUser, _: Csrf) -> dict:
    if data.difficulty not in DIFFICULTIES:
        raise HTTPException(status_code=422, detail=f"Difficulty must be one of: {', '.join(DIFFICULTIES)}.")
    table = Table(data.difficulty)
    ai_tables[user["id"]] = table
    return table.response()


@app.get("/api/ai/state")
@app.get("/api/state")
def ai_state(user: CurrentUser) -> dict | None:
    table = ai_tables.get(user["id"])
    if table is None:
        return None
    with table.lock:
        return {"frames": [], "view": table.view_for(0)}


@app.post("/api/ai/action")
@app.post("/api/action")
def ai_action(data: Action, user: CurrentUser, _: Csrf) -> dict:
    table = _ai_table(user)
    if data.action == "raise" and (data.amount is None or data.amount < 0):
        raise HTTPException(status_code=422, detail="A raise needs a nonnegative whole-cent amount.")
    try:
        with table.lock:
            table.human_action(data.action, data.amount)
            return table.response()
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ai/next")
@app.post("/api/next")
def ai_next(user: CurrentUser, _: Csrf) -> dict:
    table = _ai_table(user)
    try:
        with table.lock:
            table.next_hand()
            return table.response()
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.exception_handler(GameError)
async def game_error_handler(_: Request, exc: GameError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


def _ws_game_state(room: Room, user_id: int) -> dict:
    seat = _seat_for(room, user_id)
    return {"type": "game_state", "room": room.public_state(),
            "game": room.game.view_for(seat), "version": room.version}
