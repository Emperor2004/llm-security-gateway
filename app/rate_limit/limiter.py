# limiter
"""
Per-principal token-bucket rate limiter.

In-memory and thread-safe. For multi-process deployment this should be
backed by Redis (INCR + TTL or a Lua token-bucket script) instead — the
in-memory version is correct for a single gateway instance only, which
is called out explicitly rather than silently pretending to be
horizontally scalable.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    def __init__(self, capacity: int | None = None, refill_per_sec: float | None = None):
        settings = get_settings()
        self.capacity = capacity if capacity is not None else settings.rate_limit_capacity
        self.refill_per_sec = refill_per_sec if refill_per_sec is not None else settings.rate_limit_refill_per_sec
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _refill(self, bucket: _Bucket, now: float) -> None:
        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_sec)
        bucket.last_refill = now

    def allow(self, principal_id: str, cost: float = 1.0) -> bool:
        """Consume `cost` tokens for principal_id. Returns True if allowed."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(principal_id, _Bucket(tokens=self.capacity, last_refill=now))
            self._refill(bucket, now)
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True
            return False

    def remaining(self, principal_id: str) -> float:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(principal_id)
            if bucket is None:
                return float(self.capacity)
            self._refill(bucket, now)
            return bucket.tokens


# Process-wide singleton shared across requests.
limiter = TokenBucketLimiter()
