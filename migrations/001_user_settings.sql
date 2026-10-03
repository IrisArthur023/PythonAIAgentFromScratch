-- Migration: 001_user_settings.sql
-- Description: Create user_settings table to persist preferred LLM provider, custom models, and encrypted API keys.

CREATE TABLE IF NOT EXISTS user_settings (
    user_id VARCHAR(64) PRIMARY KEY,
    preferred_provider VARCHAR(32) NOT NULL DEFAULT 'anthropic',
    preferred_model VARCHAR(64),
    encrypted_openai_key TEXT,
    encrypted_anthropic_key TEXT,
    ollama_base_url VARCHAR(255) NOT NULL DEFAULT 'http://localhost:11434',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index on preferred_provider for fast filtering if needed
CREATE INDEX IF NOT EXISTS idx_user_settings_provider ON user_settings(preferred_provider);

-- Seed default user settings if not present
INSERT INTO user_settings (user_id, preferred_provider, ollama_base_url)
VALUES ('default_user', 'anthropic', 'http://localhost:11434')
ON CONFLICT (user_id) DO NOTHING;
