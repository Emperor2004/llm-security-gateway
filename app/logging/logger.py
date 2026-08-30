# logger
"""
Structured JSON logging for the gateway.

Every event is a single JSON line written to LOG_FILE (and echoed to
stdout), which is what dashboard/app.py and scripts/simulate_traffic.py
consume. Any 'prompt' / 'response' field is passed through the PII
scrubber before it's written, so a misconfigured LOG_FILE can't become
a second place where user PII leaks — this is on by default
(PII_REDACT_IN_LOGS=true) and has to be deliberately turned off.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from app.config import get_settings

_TEXT_FIELDS_TO_SCRUB = {"prompt", "response"}


def _build_logger() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger("llm_gateway")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if logger.handlers:
        return logger  # already configured (e.g. re-imported in tests)

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, stream_handler):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


_logger = _build_logger()


def log_event(event_type: str, **fields) -> None:
    """Write one structured JSON log line.

    Example: log_event("request_blocked", reason="jailbreak", principal_id=p.id, score=0.87)
    """
    settings = get_settings()

    if settings.pii_redact_in_logs:
        # Local import avoids a hard import-time dependency cycle between
        # logging and pii at module load.
        from app.pii.scrubber import scrub

        for field_name in _TEXT_FIELDS_TO_SCRUB:
            if field_name in fields and isinstance(fields[field_name], str):
                fields[field_name] = scrub(fields[field_name]).text

    record = {"ts": time.time(), "event": event_type, **fields}
    _logger.info(json.dumps(record, default=str))
