"""In-memory, single-worker room state for the first local multiplayer build."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def make_room_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))


@dataclass
class Room:
    code: str
    host_id: int
    players: dict[int, str]
    status: str = "waiting"
    game: object | None = None
    version: int = 0
    last_active: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)
    sockets: dict[int, WebSocket] = field(default_factory=dict)

    def public_state(self) -> dict:
        return {
            "room_code": self.code,
            "host_id": self.host_id,
            "players": [
                {"user_id": uid, "username": name, "is_host": uid == self.host_id,
                 "connected": uid in self.sockets}
                for uid, name in self.players.items()
            ],
            "available_seats": 6 - len(self.players),
            "max_players": 6,
            "status": self.status,
            "version": self.version,
        }
