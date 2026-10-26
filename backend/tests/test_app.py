from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import auth
from backend.main import app, ai_tables, rooms


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POKERPILOT_DATABASE", str(tmp_path / "pokerpilot.sqlite3"))
    rooms.clear()
    ai_tables.clear()
    with TestClient(app) as test_client:
        yield test_client
    rooms.clear()
    ai_tables.clear()


def csrf(client: TestClient) -> dict[str, str]:
    client.get("/api/auth/csrf")
    return {"X-CSRF-Token": client.cookies["pokerpilot_csrf"]}


def register(client: TestClient, username: str, email: str | None = None) -> dict:
    headers = csrf(client)
    response = client.post("/api/auth/signup", headers=headers, json={
        "username": username,
        "email": email or f"{username}@example.test",
        "password": "a-very-long-test-password",
    })
    assert response.status_code == 200, response.text
    return response.json()["user"]


def receive_until(socket, event_type: str) -> dict:
    for _ in range(8):
        message = socket.receive_json()
        if message.get("type") == event_type:
            return message
    raise AssertionError(f"Did not receive {event_type} from room WebSocket")


def test_signup_login_logout_and_protected_game(client, tmp_path):
    user = register(client, "alice")
    assert client.get("/api/auth/me").json()["user"]["id"] == user["id"]
    assert client.get("/api/state").status_code == 200

    with sqlite3.connect(tmp_path / "pokerpilot.sqlite3") as db:
        password_hash = db.execute("SELECT password_hash FROM users WHERE id=?", (user["id"],)).fetchone()[0]
    assert password_hash.startswith("scrypt$")
    assert password_hash != "a-very-long-test-password"

    duplicate = client.post("/api/auth/signup", headers=csrf(client), json={
        "username": "ALICE", "email": "different@example.test", "password": "a-very-long-test-password",
    })
    assert duplicate.status_code == 409

    bad_csrf = client.post("/api/auth/logout", headers={"X-CSRF-Token": "wrong"}, json={})
    assert bad_csrf.status_code == 403
    logout = client.post("/api/auth/logout", headers=csrf(client), json={})
    assert logout.status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/new", headers=csrf(client), json={"difficulty": "easy"}).status_code == 401

    bad_login = client.post("/api/auth/login", headers=csrf(client), json={
        "identifier": "alice", "password": "not-the-password",
    })
    assert bad_login.status_code == 401
    login = client.post("/api/auth/login", headers=csrf(client), json={
        "identifier": "alice@example.test", "password": "a-very-long-test-password",
    })
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["user"]["username"] == "alice"


def test_authenticated_player_can_start_ai_game(client):
    register(client, "player")
    response = client.post("/api/ai/new", headers=csrf(client), json={"difficulty": "easy"})
    assert response.status_code == 200
    assert len(response.json()["view"]["seats"]) == 6
    state = client.get("/api/ai/state")
    assert state.status_code == 200
    assert state.json()["view"]["difficulty"] == "easy"


def test_waiting_room_transfers_host_when_host_leaves(client):
    register(client, "first")
    second_client = TestClient(app)
    try:
        second = register(second_client, "second")
        created = client.post("/api/rooms", headers=csrf(client), json={})
        code = created.json()["room_code"]
        joined = second_client.post("/api/rooms/join", headers=csrf(second_client), json={"room_code": code})
        assert joined.status_code == 200

        left = client.post(f"/api/rooms/{code}/leave", headers=csrf(client), json={})
        assert left.status_code == 200
        remaining = second_client.get(f"/api/rooms/{code}")
        assert remaining.status_code == 200
        assert remaining.json()["room"]["host_id"] == second["id"]
    finally:
        second_client.close()


def test_two_authenticated_players_start_and_act_in_private_room(client):
    host_client = client
    host = register(host_client, "host")
    guest_client = TestClient(app)
    try:
        guest = register(guest_client, "guest")
        created = host_client.post("/api/rooms", headers=csrf(host_client), json={})
        assert created.status_code == 200
        code = created.json()["room_code"]
        assert len(code) == 6

        not_member = guest_client.post(f"/api/rooms/{code}/start", headers=csrf(guest_client), json={})
        assert not_member.status_code == 403
        joined = guest_client.post("/api/rooms/join", headers=csrf(guest_client), json={"room_code": code.lower()})
        assert joined.status_code == 200

        with host_client.websocket_connect(f"/ws/rooms/{code}") as host_ws:
            assert host_ws.receive_json()["type"] == "room_state"
            with guest_client.websocket_connect(f"/ws/rooms/{code}") as guest_ws:
                assert guest_ws.receive_json()["type"] == "room_state"
                assert host_ws.receive_json()["type"] == "player_joined"
                assert host_ws.receive_json()["type"] == "room_state"
                assert guest_ws.receive_json()["type"] == "player_joined"
                assert guest_ws.receive_json()["type"] == "room_state"
                started = host_client.post(f"/api/rooms/{code}/start", headers=csrf(host_client), json={})
                assert started.status_code == 200
                receive_until(host_ws, "game_started")
                host_state = receive_until(host_ws, "game_state")
                assert host_state["type"] == "game_state"
                receive_until(guest_ws, "game_started")
                guest_state = receive_until(guest_ws, "game_state")
                assert guest_state["type"] == "game_state"
                assert host_state["game"]["seats"][0]["cards"] is not None
                assert host_state["game"]["seats"][1]["cards"] is None
                assert guest_state["game"]["seats"][0]["cards"] is not None
                assert guest_state["game"]["seats"][1]["cards"] is None

                room = rooms[code]
                version = room.version
                actor_id = list(room.players)[room.game.actor_seat]
                clients = {host["id"]: (host_client, host_ws), guest["id"]: (guest_client, guest_ws)}
                other_id = guest["id"] if actor_id == host["id"] else host["id"]
                clients[other_id][1].send_json({"type": "player_action", "action": "check_call", "version": version})
                assert clients[other_id][1].receive_json()["type"] == "error"

                clients[actor_id][1].send_json({"type": "player_action", "action": "check_call", "version": version})
                actor_socket = clients[actor_id][1]
                other_socket = clients[other_id][1]
                assert actor_socket.receive_json()["type"] == "player_action"
                assert actor_socket.receive_json()["type"] == "game_state"
                assert other_socket.receive_json()["type"] == "player_action"
                updated = other_socket.receive_json()
                assert updated["type"] == "game_state"
                assert updated["game"]["seats"][0]["cards"] is not None
                assert updated["game"]["seats"][1]["cards"] is None
            with guest_client.websocket_connect(f"/ws/rooms/{code}") as reconnect_ws:
                resumed = receive_until(reconnect_ws, "game_state")
                assert resumed["room"]["status"] == "in_progress"
                assert resumed["game"]["seats"][0]["name"] == "guest"
                assert resumed["game"]["seats"][0]["cards"] is not None
    finally:
        guest_client.close()
