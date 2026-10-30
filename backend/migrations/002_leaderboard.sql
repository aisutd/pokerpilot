-- Apply in Supabase SQL Editor before using the leaderboard.
-- Profile IDs are text so legacy browser profiles can be migrated explicitly.
BEGIN;
CREATE SCHEMA IF NOT EXISTS pokerpilot;
CREATE TABLE IF NOT EXISTS pokerpilot.leaderboard_players (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS leaderboard_players_name_lower_unique
    ON pokerpilot.leaderboard_players (lower(name));
CREATE TABLE IF NOT EXISTS pokerpilot.leaderboard_round_results (
    game_id TEXT NOT NULL,
    round_number INTEGER NOT NULL CHECK (round_number > 0),
    player_id TEXT NOT NULL REFERENCES pokerpilot.leaderboard_players(id),
    net_cents BIGINT NOT NULL,
    saved_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, round_number, player_id)
);
CREATE INDEX IF NOT EXISTS leaderboard_results_by_player
    ON pokerpilot.leaderboard_round_results (player_id);
ALTER TABLE pokerpilot.leaderboard_players ENABLE ROW LEVEL SECURITY;
ALTER TABLE pokerpilot.leaderboard_round_results ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON pokerpilot.leaderboard_players FROM PUBLIC, anon, authenticated;
REVOKE ALL ON pokerpilot.leaderboard_round_results FROM PUBLIC, anon, authenticated;
COMMIT;
