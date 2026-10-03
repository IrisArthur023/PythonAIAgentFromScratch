"""Database integration for Neon (PostgreSQL) with local fallback.

Manages user settings, provider preferences, and encrypted API key storage.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger("field_notes_db")

# Optional cryptography for encrypting API keys at rest
try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

# Optional psycopg2 for Neon Postgres
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

# In-memory store fallback when DATABASE_URL is not set or Neon is offline
_IN_MEMORY_SETTINGS: Dict[str, Dict[str, Any]] = {
    "default_user": {
        "user_id": "default_user",
        "preferred_provider": "openrouter",
        "preferred_model": None,
        "encrypted_openai_key": None,
        "encrypted_anthropic_key": None,
        "encrypted_openrouter_key": None,
        "ollama_base_url": "http://localhost:11434",
    }
}


def _get_fernet() -> Optional[Any]:
    """Return a Fernet cipher instance if ENCRYPTION_KEY is available."""
    if not Fernet:
        return None
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        return None
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except Exception as error:
        logger.warning("Invalid ENCRYPTION_KEY configured: %s", error)
        return None


def encrypt_secret(secret: Optional[str]) -> Optional[str]:
    """Encrypt a secret key using Fernet if configured, else return raw string."""
    if not secret:
        return None
    fernet = _get_fernet()
    if fernet:
        return fernet.encrypt(secret.encode("utf-8")).decode("utf-8")
    return secret


def decrypt_secret(encrypted_secret: Optional[str]) -> Optional[str]:
    """Decrypt a secret key using Fernet if configured, else return raw string."""
    if not encrypted_secret:
        return None
    fernet = _get_fernet()
    if fernet:
        try:
            return fernet.decrypt(encrypted_secret.encode("utf-8")).decode("utf-8")
        except Exception as error:
            logger.warning("Failed to decrypt secret: %s", error)
            return None
    return encrypted_secret


def get_db_connection():
    """Create and return a database connection to Neon (PostgreSQL)."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url or not psycopg2:
        return None
    try:
        # Neon requires sslmode=require
        conn = psycopg2.connect(db_url, sslmode=os.getenv("PGSSLMODE", "require"))
        return conn
    except Exception as error:
        logger.warning("Could not connect to Neon Postgres (%s). Using local fallback.", error)
        return None


def init_db() -> bool:
    """Initialize user_settings table in Neon Postgres if connection is available."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id VARCHAR(64) PRIMARY KEY,
                    preferred_provider VARCHAR(32) NOT NULL DEFAULT 'openrouter',
                    preferred_model VARCHAR(64),
                    encrypted_openai_key TEXT,
                    encrypted_anthropic_key TEXT,
                    encrypted_openrouter_key TEXT,
                    ollama_base_url VARCHAR(255) NOT NULL DEFAULT 'http://localhost:11434',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_user_settings_provider ON user_settings(preferred_provider);
                INSERT INTO user_settings (user_id, preferred_provider, ollama_base_url)
                VALUES ('default_user', 'openrouter', 'http://localhost:11434')
                ON CONFLICT (user_id) DO NOTHING;
            """)
            conn.commit()
            return True
    except Exception as error:
        logger.error("Failed to initialize database schema: %s", error)
        conn.rollback()
        return False
    finally:
        conn.close()


def get_user_settings(user_id: str = "default_user") -> Dict[str, Any]:
    """Retrieve settings for a user from Neon or fallback store."""
    conn = get_db_connection()
    if conn and RealDictCursor:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id, preferred_provider, preferred_model, encrypted_openai_key, encrypted_anthropic_key, encrypted_openrouter_key, ollama_base_url FROM user_settings WHERE user_id = %s;",
                    (user_id,)
                )
                row = cur.fetchone()
                if row:
                    data = dict(row)
                    return {
                        "user_id": data["user_id"],
                        "preferred_provider": data["preferred_provider"],
                        "preferred_model": data["preferred_model"],
                        "ollama_base_url": data["ollama_base_url"] or "http://localhost:11434",
                        "openai_api_key": decrypt_secret(data.get("encrypted_openai_key")),
                        "anthropic_api_key": decrypt_secret(data.get("encrypted_anthropic_key")),
                        "openrouter_api_key": decrypt_secret(data.get("encrypted_openrouter_key")),
                        "has_custom_openai_key": bool(data.get("encrypted_openai_key")),
                        "has_custom_anthropic_key": bool(data.get("encrypted_anthropic_key")),
                        "has_custom_openrouter_key": bool(data.get("encrypted_openrouter_key")),
                    }
        except Exception as error:
            logger.warning("Error fetching settings from Neon (%s), using local fallback.", error)
        finally:
            conn.close()

    # Fallback store
    raw = _IN_MEMORY_SETTINGS.get(user_id, _IN_MEMORY_SETTINGS["default_user"])
    return {
        "user_id": raw["user_id"],
        "preferred_provider": raw["preferred_provider"],
        "preferred_model": raw.get("preferred_model"),
        "ollama_base_url": raw.get("ollama_base_url", "http://localhost:11434"),
        "openai_api_key": decrypt_secret(raw.get("encrypted_openai_key")),
        "anthropic_api_key": decrypt_secret(raw.get("encrypted_anthropic_key")),
        "openrouter_api_key": decrypt_secret(raw.get("encrypted_openrouter_key")),
        "has_custom_openai_key": bool(raw.get("encrypted_openai_key")),
        "has_custom_anthropic_key": bool(raw.get("encrypted_anthropic_key")),
        "has_custom_openrouter_key": bool(raw.get("encrypted_openrouter_key")),
    }


def update_user_settings(user_id: str = "default_user", updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update settings for a user in Neon or fallback store."""
    if updates is None:
        updates = {}

    current = get_user_settings(user_id)
    preferred_provider = str(updates.get("preferred_provider", current["preferred_provider"]))
    preferred_model = updates.get("preferred_model", current["preferred_model"])
    ollama_base_url = str(updates.get("ollama_base_url", current["ollama_base_url"]))

    # Optional key updates
    encrypted_openai_key = current.get("openai_api_key")
    if "openai_api_key" in updates and updates["openai_api_key"] is not None:
        encrypted_openai_key = encrypt_secret(updates["openai_api_key"]) if updates["openai_api_key"] else None

    encrypted_anthropic_key = current.get("anthropic_api_key")
    if "anthropic_api_key" in updates and updates["anthropic_api_key"] is not None:
        encrypted_anthropic_key = encrypt_secret(updates["anthropic_api_key"]) if updates["anthropic_api_key"] else None

    encrypted_openrouter_key = current.get("openrouter_api_key")
    if "openrouter_api_key" in updates and updates["openrouter_api_key"] is not None:
        encrypted_openrouter_key = encrypt_secret(updates["openrouter_api_key"]) if updates["openrouter_api_key"] else None

    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_settings (
                        user_id, preferred_provider, preferred_model, encrypted_openai_key, encrypted_anthropic_key, encrypted_openrouter_key, ollama_base_url, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        preferred_provider = EXCLUDED.preferred_provider,
                        preferred_model = EXCLUDED.preferred_model,
                        encrypted_openai_key = COALESCE(EXCLUDED.encrypted_openai_key, user_settings.encrypted_openai_key),
                        encrypted_anthropic_key = COALESCE(EXCLUDED.encrypted_anthropic_key, user_settings.encrypted_anthropic_key),
                        encrypted_openrouter_key = COALESCE(EXCLUDED.encrypted_openrouter_key, user_settings.encrypted_openrouter_key),
                        ollama_base_url = EXCLUDED.ollama_base_url,
                        updated_at = NOW();
                """, (user_id, preferred_provider, preferred_model, encrypted_openai_key, encrypted_anthropic_key, encrypted_openrouter_key, ollama_base_url))
                conn.commit()
        except Exception as error:
            logger.warning("Failed to update user_settings in Neon (%s), saving in fallback.", error)
            conn.rollback()
        finally:
            conn.close()

    # Also update in-memory
    _IN_MEMORY_SETTINGS[user_id] = {
        "user_id": user_id,
        "preferred_provider": preferred_provider,
        "preferred_model": preferred_model,
        "encrypted_openai_key": encrypted_openai_key,
        "encrypted_anthropic_key": encrypted_anthropic_key,
        "encrypted_openrouter_key": encrypted_openrouter_key,
        "ollama_base_url": ollama_base_url,
    }

    return get_user_settings(user_id)
