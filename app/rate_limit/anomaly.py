# anomaly
"""
Burst detection and extraction-signal heuristics.

Two independent signals, both cheap enough to run on every request:

1. Burst detection: request count in a sliding time window vs threshold.
2. Extraction signal: a self-defined heuristic (see README) — high
   request volume combined with high *lexical diversity* across prompts
   is treated as a proxy for systematic probing / scraping behavior,
   since a legitimate user's prompts tend to cluster around a task while
   an extraction attempt tends to sweep across many distinct inputs.

This is intentionally lexical (token-set Jaccard-style diversity), not
embedding-based, so it runs inline on the request path without loading
a model. The jailbreak module's embedder can be reused later for a
semantic version of this if the lexical heuristic proves too noisy.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import get_settings


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in text.split() if t}


@dataclass
class _PrincipalHistory:
    timestamps: deque[float] = field(default_factory=deque)
    recent_token_sets: deque[set[str]] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class AnomalyResult:
    is_burst: bool
    request_count_in_window: int
    is_extraction_signal: bool
    diversity_score: float | None


class AnomalyDetector:
    def __init__(
        self,
        window_s: float | None = None,
        burst_threshold: int | None = None,
        diversity_min_requests: int | None = None,
        diversity_threshold: float | None = None,
    ):
        settings = get_settings()
        self.window_s = window_s if window_s is not None else settings.anomaly_window_s
        self.burst_threshold = burst_threshold if burst_threshold is not None else settings.anomaly_burst_threshold
        self.diversity_min_requests = (
            diversity_min_requests if diversity_min_requests is not None else settings.anomaly_diversity_min_requests
        )
        self.diversity_threshold = (
            diversity_threshold if diversity_threshold is not None else settings.anomaly_diversity_threshold
        )
        self._history: dict[str, _PrincipalHistory] = {}
        self._lock = threading.Lock()

    def _prune(self, history: _PrincipalHistory, now: float) -> None:
        cutoff = now - self.window_s
        while history.timestamps and history.timestamps[0] < cutoff:
            history.timestamps.popleft()

    def record_and_check(self, principal_id: str, prompt: str) -> AnomalyResult:
        now = time.monotonic()
        with self._lock:
            history = self._history.setdefault(principal_id, _PrincipalHistory())
            history.timestamps.append(now)
            self._prune(history, now)
            request_count = len(history.timestamps)
            is_burst = request_count > self.burst_threshold

            history.recent_token_sets.append(_tokenize(prompt))
            diversity_score = None
            is_extraction = False
            if request_count >= self.diversity_min_requests:
                diversity_score = self._diversity(history.recent_token_sets)
                is_extraction = is_burst and diversity_score >= self.diversity_threshold

            return AnomalyResult(
                is_burst=is_burst,
                request_count_in_window=request_count,
                is_extraction_signal=is_extraction,
                diversity_score=diversity_score,
            )

    @staticmethod
    def _diversity(token_sets: deque[set[str]]) -> float:
        """Average pairwise Jaccard distance across recent prompts (0=identical, 1=disjoint)."""
        sets = [s for s in token_sets if s]
        if len(sets) < 2:
            return 0.0
        distances = []
        # Compare consecutive pairs only — O(n) instead of O(n^2), sufficient
        # as a moving signal rather than an exact pairwise average.
        for a, b in zip(sets, list(sets)[1:]):
            union = a | b
            if not union:
                continue
            jaccard = len(a & b) / len(union)
            distances.append(1 - jaccard)
        return sum(distances) / len(distances) if distances else 0.0


anomaly_detector = AnomalyDetector()
