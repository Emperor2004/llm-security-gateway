# config
"""
Central configuration for the LLM Security Gateway.

Supports:
1. config.yaml (declarative hierarchical settings)
2. .env file & OS environment variables (overrides YAML settings)
3. Safe defaults for out-of-the-box local testing
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Populate os.environ from a .env file without overriding existing vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _load_yaml_config(path: Path = BASE_DIR / "config.yaml") -> dict[str, Any]:
    """Safely load config.yaml if present and pyyaml is installed."""
    if not path.exists():
        return {}
    try:
        import yaml
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_YAML_CONFIG = _load_yaml_config()


def _get_yaml_val(*keys: str, default: Any = None) -> Any:
    """Navigate nested keys in config.yaml."""
    curr = _YAML_CONFIG
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return default
    return curr


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
    env: str = field(
        default_factory=lambda: os.environ.get("APP_ENV", _get_yaml_val("server", "env", default="development"))
    )
    host: str = field(
        default_factory=lambda: os.environ.get("HOST", _get_yaml_val("server", "host", default="0.0.0.0"))
    )
    port: int = field(
        default_factory=lambda: _env_int("PORT", _get_yaml_val("server", "port", default=8080))
    )

    # --- Upstream LLM ---
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", _get_yaml_val("llm", "provider", default="mock"))
    )
    llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_BASE_URL", _get_yaml_val("llm", "base_url", default="https://api.openai.com/v1")
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", _get_yaml_val("llm", "api_key", default=""))
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", _get_yaml_val("llm", "model", default="gpt-4o-mini"))
    )
    llm_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_S", _get_yaml_val("llm", "timeout_s", default=30.0))
    )

    # --- Auth / RBAC ---
    api_keys_json: str = field(default_factory=lambda: os.environ.get("API_KEYS_JSON", ""))

    # --- Rate limiting (token bucket) ---
    rate_limit_capacity: int = field(
        default_factory=lambda: _env_int(
            "RATE_LIMIT_CAPACITY", _get_yaml_val("pipeline", "stages", "rate_limit", "capacity", default=20)
        )
    )
    rate_limit_refill_per_sec: float = field(
        default_factory=lambda: _env_float(
            "RATE_LIMIT_REFILL_PER_SEC",
            _get_yaml_val("pipeline", "stages", "rate_limit", "refill_per_sec", default=0.5),
        )
    )

    # --- Anomaly / extraction-signal detection ---
    anomaly_window_s: float = field(
        default_factory=lambda: _env_float(
            "ANOMALY_WINDOW_S", _get_yaml_val("pipeline", "stages", "anomaly", "window_s", default=60.0)
        )
    )
    anomaly_burst_threshold: int = field(
        default_factory=lambda: _env_int(
            "ANOMALY_BURST_THRESHOLD",
            _get_yaml_val("pipeline", "stages", "anomaly", "burst_threshold", default=30),
        )
    )
    anomaly_diversity_min_requests: int = field(
        default_factory=lambda: _env_int(
            "ANOMALY_DIVERSITY_MIN_REQUESTS",
            _get_yaml_val("pipeline", "stages", "anomaly", "diversity_min_requests", default=15),
        )
    )
    anomaly_diversity_threshold: float = field(
        default_factory=lambda: _env_float(
            "ANOMALY_DIVERSITY_THRESHOLD",
            _get_yaml_val("pipeline", "stages", "anomaly", "diversity_threshold", default=0.85),
        )
    )

    # --- Jailbreak detection ---
    jailbreak_threshold: float = field(
        default_factory=lambda: _env_float(
            "JAILBREAK_THRESHOLD", _get_yaml_val("pipeline", "stages", "jailbreak", "threshold", default=0.5)
        )
    )
    jailbreak_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "JAILBREAK_MODEL_PATH",
            _get_yaml_val(
                "pipeline",
                "stages",
                "jailbreak",
                "model_path",
                default=str(BASE_DIR / "models" / "jailbreak_classifier"),
            ),
        )
    )

    # --- PII scrubbing ---
    pii_redact_in_logs: bool = field(
        default_factory=lambda: _env_bool(
            "PII_REDACT_IN_LOGS",
            _get_yaml_val("pipeline", "stages", "pii", "redact_in_logs", default=True),
        )
    )
    pii_block_on_detect: bool = field(
        default_factory=lambda: _env_bool(
            "PII_BLOCK_ON_DETECT",
            _get_yaml_val("pipeline", "stages", "pii", "block_on_detect", default=False),
        )
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.environ.get(
            "LOG_LEVEL", _get_yaml_val("pipeline", "stages", "audit", "log_level", default="INFO")
        )
    )
    log_file: str = field(
        default_factory=lambda: os.environ.get(
            "LOG_FILE",
            _get_yaml_val("pipeline", "stages", "audit", "log_file", default=str(BASE_DIR / "logs" / "gateway.jsonl")),
        )
    )

    # --- Modular Stage Activation Toggles ---
    stage_auth_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_AUTH_ENABLED",
            _get_yaml_val("pipeline", "stages", "auth", "enabled", default=True),
        )
    )
    stage_rate_limit_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_RATE_LIMIT_ENABLED",
            _get_yaml_val("pipeline", "stages", "rate_limit", "enabled", default=True),
        )
    )
    stage_anomaly_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_ANOMALY_ENABLED",
            _get_yaml_val("pipeline", "stages", "anomaly", "enabled", default=True),
        )
    )
    stage_jailbreak_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_JAILBREAK_ENABLED",
            _get_yaml_val("pipeline", "stages", "jailbreak", "enabled", default=True),
        )
    )
    stage_pii_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_PII_ENABLED",
            _get_yaml_val("pipeline", "stages", "pii", "enabled", default=True),
        )
    )
    stage_audit_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "STAGE_AUDIT_ENABLED",
            _get_yaml_val("pipeline", "stages", "audit", "enabled", default=True),
        )
    )

    def api_keys(self) -> dict[str, str]:
        """Return {api_key: role} mapping.

        Priority order:
        1. API_KEYS_JSON env var
        2. auth.api_keys block in config.yaml
        3. Built-in demo keys
        """
        if self.api_keys_json:
            try:
                parsed = json.loads(self.api_keys_json)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        yaml_keys = _get_yaml_val("auth", "api_keys")
        if isinstance(yaml_keys, dict) and yaml_keys:
            return yaml_keys

        return {
            "demo-admin-key": "admin",
            "demo-user-key": "user",
            "demo-readonly-key": "readonly",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
