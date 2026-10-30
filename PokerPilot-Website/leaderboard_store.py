"""Supabase PostgreSQL leaderboard storage. No SQLite fallback.

The application must load DATABASE_URL from the root .env before using this
store. Apply backend/migrations/002_leaderboard.sql in Supabase first.
"""

from backend.auth import connect


class LeaderboardStore:
    def profile(self, player_id):
        with connect() as db:
            row = db.execute(
                'SELECT id, name FROM pokerpilot.leaderboard_players WHERE id = %s',
                (player_id,),
            ).fetchone()
        return dict(row) if row else None

    def ensure_player(self, player_id, name):
        """Register a server-authenticated account using its stable account ID.

        Existing historical browser profiles must be linked explicitly, rather
        than matched by display name. A name conflict raises a database error.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError('A display name is required.')
        with connect() as db:
            row = db.execute('''
                INSERT INTO pokerpilot.leaderboard_players (id, name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                RETURNING id, name
            ''', (str(player_id), name.strip())).fetchone()
        return dict(row)

    def save_rounds(self, player_id, game_id, results):
        """Save engine-settled results atomically; retries cannot double-credit.

        Results contain round_number and net_cents (integer cents). Callers
        must obtain these from the engine, never browser-supplied scores.
        """
        rows = []
        for row in results:
            number, cents = row['round_number'], row['net_cents']
            if type(number) is not int or number <= 0 or type(cents) is not int:
                raise ValueError('Expected a positive round number and integer cents.')
            rows.append((game_id, number, str(player_id), cents))
        if not rows:
            return
        with connect() as db:
            with db.cursor() as cursor:
                cursor.executemany('''
                    INSERT INTO pokerpilot.leaderboard_round_results
                        (game_id, round_number, player_id, net_cents)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (game_id, round_number, player_id) DO NOTHING
                ''', rows)

    def standings(self, player_id=None):
        with connect() as db:
            rows = db.execute('''
                SELECT p.id AS player_id, p.name,
                       COALESCE(SUM(r.net_cents), 0)::BIGINT AS net_winnings_cents,
                       COUNT(r.round_number) AS rounds_participated
                FROM pokerpilot.leaderboard_players p
                LEFT JOIN pokerpilot.leaderboard_round_results r ON r.player_id = p.id
                GROUP BY p.id, p.name
                ORDER BY net_winnings_cents DESC, lower(p.name), p.id
            ''').fetchall()
        standings, prior_score, rank = [], None, 0
        for index, row in enumerate(rows, 1):
            score = row['net_winnings_cents']
            if score != prior_score:
                rank, prior_score = index, score
            standings.append(dict(row) | {
                'rank': rank, 'is_you': row['player_id'] == str(player_id),
            })
        return {'standings': standings, 'ranking': 'cumulative_net_winnings', 'currency': 'USD'}
