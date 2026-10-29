"""Persistent, server-authoritative global winnings ledger (integer cents)."""
from contextlib import contextmanager
from pathlib import Path
import secrets
import sqlite3


class LeaderboardStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS players (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS round_results (
                    game_id TEXT NOT NULL,
                    round_number INTEGER NOT NULL CHECK(round_number > 0),
                    player_id TEXT NOT NULL REFERENCES players(id),
                    net_cents INTEGER NOT NULL,
                    saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(game_id, round_number, player_id)
                );
                CREATE INDEX IF NOT EXISTS results_by_player ON round_results(player_id);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys = ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def session_secret(self):
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO settings VALUES (?, ?)',
                       ('session_secret', secrets.token_hex(32)))
            return db.execute('SELECT value FROM settings WHERE key = ?',
                              ('session_secret',)).fetchone()['value']

    def profile(self, player_id):
        with self.connect() as db:
            row = db.execute('SELECT id, name FROM players WHERE id = ?', (player_id,)).fetchone()
            return dict(row) if row else None

    def create_player(self):
        player_id = secrets.token_urlsafe(24)
        name = 'Player-' + secrets.token_hex(5)
        with self.connect() as db:
            db.execute('INSERT INTO players(id, name) VALUES (?, ?)', (player_id, name))
        return {'id': player_id, 'name': name}

    def rename_player(self, player_id, name):
        if not isinstance(name, str):
            raise ValueError('Enter a display name.')
        name = name.strip()
        if not 2 <= len(name) <= 24 or any(ord(c) < 32 for c in name):
            raise ValueError('Use a display name of 2–24 characters.')
        try:
            with self.connect() as db:
                db.execute('UPDATE players SET name = ? WHERE id = ?', (name, player_id))
        except sqlite3.IntegrityError:
            raise ValueError('That display name is already taken.') from None
        return self.profile(player_id)

    def save_rounds(self, player_id, game_id, results):
        """Retries and page reloads cannot count the same settled round twice."""
        with self.connect() as db:
            db.executemany('''INSERT OR IGNORE INTO round_results
                (game_id, round_number, player_id, net_cents) VALUES (?, ?, ?, ?)''',
                [(game_id, row['round_number'], player_id, row['net_cents']) for row in results])

    def standings(self, player_id=None):
        with self.connect() as db:
            rows = db.execute('''
                SELECT p.id AS player_id, p.name,
                       COALESCE(SUM(r.net_cents), 0) AS net_winnings_cents,
                       COUNT(r.round_number) AS rounds_participated
                FROM players p LEFT JOIN round_results r ON r.player_id = p.id
                GROUP BY p.id
                ORDER BY net_winnings_cents DESC, p.name COLLATE NOCASE, p.id
            ''').fetchall()
        standings = []
        prior_score = None
        rank = 0
        for index, row in enumerate(rows, 1):
            score = row['net_winnings_cents']
            if score != prior_score:
                rank = index
                prior_score = score
            standings.append(dict(row) | {'rank': rank, 'is_you': row['player_id'] == player_id})
        return {'standings': standings, 'ranking': 'cumulative_net_winnings', 'currency': 'USD'}
