# config
"""
Central configuration for the LLM Security Gateway.

All settings are read from environment variables (see .env.example).
A tiny built-in .env loader is used instead of python-dotenv to avoid
pulling in an extra dependency for something this small.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Populate os.environ from a .env file without overriding existing vars."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Service ---
    env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development"))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))

    # --- Upstream LLM ---
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock"))
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))

    # --- Auth / RBAC ---
    api_keys_json: str = field(default_factory=lambda: os.environ.get("API_KEYS_JSON", ""))

    # --- Rate limiting (token bucket) ---
    rate_limit_capacity: int = field(default_factory=lambda: _env_int("RATE_LIMIT_CAPACITY", 20))
    rate_limit_refill_per_sec: float = field(default_factory=lambda: _env_float("RATE_LIMIT_REFILL_PER_SEC", 0.5))

    # --- Anomaly / extraction-signal detection ---
    anomaly_window_s: float = field(default_factory=lambda: _env_float("ANOMALY_WINDOW_S", 60.0))
    anomaly_burst_threshold: int = field(default_factory=lambda: _env_int("ANOMALY_BURST_THRESHOLD", 30))
    anomaly_diversity_min_requests: int = field(default_factory=lambda: _env_int("ANOMALY_DIVERSITY_MIN_REQUESTS", 15))
    anomaly_diversity_threshold: float = field(default_factory=lambda: _env_float("ANOMALY_DIVERSITY_THRESHOLD", 0.85))

    # --- Jailbreak detection ---
    jailbreak_threshold: float = field(default_factory=lambda: _env_float("JAILBREAK_THRESHOLD", 0.5))
    # Directory saved by app/jailbreak/train.py (trainer.save_model + tokenizer.save_pretrained),
    # not a single file — loaded via transformers.AutoModelForSequenceClassification.
    jailbreak_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "JAILBREAK_MODEL_PATH", str(BASE_DIR / "models" / "jailbreak_classifier")
        )
    )

    # --- PII scrubbing ---
    pii_redact_in_logs: bool = field(default_factory=lambda: _env_bool("PII_REDACT_IN_LOGS", True))
    pii_block_on_detect: bool = field(default_factory=lambda: _env_bool("PII_BLOCK_ON_DETECT", False))

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.environ.get("LOG_FILE", str(BASE_DIR / "logs" / "gateway.jsonl")))

    def api_keys(self) -> dict:
        """Return {api_key: role} mapping.

        Falls back to a demo set so the gateway is runnable out of the box.
        Override with API_KEYS_JSON='{"sk-...": "admin", "sk-...": "user"}' in production.
        """
        if self.api_keys_json:
            try:
                return json.loads(self.api_keys_json)
            except json.JSONDecodeError:
                pass
        return {
            "demo-admin-key": "admin",
            "demo-user-key": "user",
            "demo-readonly-key": "readonly",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
