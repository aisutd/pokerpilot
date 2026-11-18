"""PostgreSQL-backed multiplayer leaderboard storage."""

from collections.abc import Iterable
from typing import Any

from backend.auth import connect


class LeaderboardStore:
    def ensure_player(self, player_id: str | int, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('A display name is required.')

        with connect() as db:
            row = db.execute(
                '''
                INSERT INTO pokerpilot.leaderboard_players (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                RETURNING id, name
                ''',
                (str(player_id), name.strip()),
            ).fetchone()
        return dict(row)

    def save_rounds(
        self,
        player_id: str | int,
        game_id: str,
        results: Iterable[dict[str, int]],
    ) -> None:
        rows = []
        for result in results:
            round_number, net_cents = result['round_number'], result['net_cents']
            if type(round_number) is not int or round_number <= 0 or type(net_cents) is not int:
                raise ValueError('Expected a positive round number and integer cents.')
            rows.append((game_id, round_number, str(player_id), net_cents))

        if not rows:
            return

        with connect() as db:
            with db.cursor() as cursor:
                cursor.executemany(
                    '''
                    INSERT INTO pokerpilot.leaderboard_round_results
                        (game_id, round_number, player_id, net_cents)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (game_id, round_number, player_id) DO NOTHING
                    ''',
                    rows,
                )

    def standings(self, player_id: str | int) -> dict[str, Any]:
        with connect() as db:
            rows = db.execute(
                '''
                SELECT p.id AS player_id, p.name,
                       COALESCE(SUM(r.net_cents), 0)::BIGINT AS net_winnings_cents,
                       COUNT(r.round_number) AS rounds_participated
                FROM pokerpilot.leaderboard_players p
                LEFT JOIN pokerpilot.leaderboard_round_results r ON r.player_id = p.id
                GROUP BY p.id, p.name
                ORDER BY net_winnings_cents DESC, lower(p.name), p.id
                '''
            ).fetchall()

        standings = []
        prior_score = None
        rank = 0
        for index, row in enumerate(rows, 1):
            score = row['net_winnings_cents']
            if score != prior_score:
                rank, prior_score = index, score
            standings.append(dict(row) | {
                'rank': rank,
                'is_you': row['player_id'] == str(player_id),
            })

        return {
            'standings': standings,
            'ranking': 'cumulative_net_winnings',
            'currency': 'USD',
        }
