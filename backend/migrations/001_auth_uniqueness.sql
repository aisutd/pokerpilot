-- Uses the existing pokerpilot.users table. It does not create or copy data.
-- First check and resolve any case-insensitive duplicates, then apply these
-- indexes through the Supabase SQL editor or a privileged migration runner.
CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower_unique
    ON pokerpilot.users (lower(username));

CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_unique
    ON pokerpilot.users (lower(email));
