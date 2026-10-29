from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Player:
    player_id: str
    name: str
    starting_chips: float
    chips: float
    active: bool = True
    hands_played: int = 0
    left_at: Optional[str] = None

    @property
    def net_profit(self) -> float:
        return round(self.chips - self.starting_chips, 2)


class PokerPilotLeaderboard:
    def __init__(self):
        self.game_id: Optional[str] = None
        self.status = "idle"
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.players: Dict[str, Player] = {}

    def start_game(self, game_id: str) -> None:
        """Start a new game; discard rankings from the previous closed game."""
        if self.status == "active":
            raise ValueError("Finish the current game before starting another.")
        self.game_id = game_id
        self.status = "active"
        self.started_at = timestamp()
        self.finished_at = None
        self.players = {}

    def add_player(self, player_id: str, name: str, starting_chips: float) -> None:
        self._require_active()
        if player_id in self.players:
            raise ValueError("Player already joined this game.")
        if starting_chips < 0:
            raise ValueError("Starting chips cannot be negative.")
        self.players[player_id] = Player(player_id, name, round(starting_chips, 2), round(starting_chips, 2))

    def hand_finished(self, chip_balances: Dict[str, float], participants: Optional[List[str]] = None) -> None:
        """Apply authoritative chip balances AFTER a hand is fully settled.

        chip_balances: mapping of player_id -> total chips after the hand.
        participants: players who played the hand (used for hands_played).
        """
        self._require_active()
        for player_id, chips in chip_balances.items():
            if player_id not in self.players:
                raise ValueError("Unknown player: " + player_id)
            if not self.players[player_id].active:
                raise ValueError("Cannot change a player who already left: " + player_id)
            if chips < 0:
                raise ValueError("Chip balance cannot be negative.")
        for player_id in participants or []:
            if player_id not in self.players:
                raise ValueError("Unknown participant: " + player_id)
        for player_id, chips in chip_balances.items():
            self.players[player_id].chips = round(chips, 2)
        for player_id in participants or []:
            self.players[player_id].hands_played += 1

    def player_quit(self, player_id: str) -> None:
        """Freeze the player's last settled balance and keep them ranked."""
        self._require_active()
        player = self.players[player_id]
        player.active = False
        player.left_at = timestamp()
        # Once the last player leaves, the game is complete.
        if self.players and all(not p.active for p in self.players.values()):
            self.finish_game()

    def finish_game(self) -> None:
        """Freeze final results; a subsequent start_game() resets the board."""
        self._require_active()
        self.status = "finished"
        self.finished_at = timestamp()

    def standings(self) -> List[dict]:
        ordered = self._sorted_players()
        result = []
        for rank, player in enumerate(ordered, 1):
            result.append({
                "rank": rank,
                "player_id": player.player_id,
                "name": player.name,
                "starting_chips": player.starting_chips,
                "current_chips": player.chips,
                "net_profit": player.net_profit,
                "hands_played": player.hands_played,
                "status": "playing" if player.active and self.status == "active" else "finished",
            })
        return result

    def snapshot(self) -> dict:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "standings": self.standings(),
        }

    def print_board(self) -> None:
        print("\nPOKERPILOT LEADERBOARD | Game:", self.game_id, "|", self.status.upper())
        print(f"{'#':<4}{'Player':<18}{'Chips':>12}{'Profit':>12}{'Status':>12}")
        for row in self.standings():
            print(f'{row["rank"]:<4}{row["name"]:<18}{row["current_chips"]:>12.2f}'
                  f'{row["net_profit"]:>+12.2f}{row["status"]:>12}')

    def _sorted_players(self) -> List[Player]:
        return sorted(self.players.values(), key=lambda p: (-p.net_profit, p.name.lower(), p.player_id))

    def _require_active(self) -> None:
        if self.status != "active":
            raise ValueError("No active game. Call start_game() first.")
