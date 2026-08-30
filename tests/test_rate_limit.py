# test rate limit
import time

from app.rate_limit.limiter import TokenBucketLimiter
from app.rate_limit.anomaly import AnomalyDetector


def test_limiter_allows_up_to_capacity():
    limiter = TokenBucketLimiter(capacity=3, refill_per_sec=0.0)
    assert limiter.allow("user-a")
    assert limiter.allow("user-a")
    assert limiter.allow("user-a")
    assert not limiter.allow("user-a")


def test_limiter_refills_over_time():
    limiter = TokenBucketLimiter(capacity=1, refill_per_sec=10.0)
    assert limiter.allow("user-b")
    assert not limiter.allow("user-b")
    time.sleep(0.15)  # ~1.5 tokens refilled
    assert limiter.allow("user-b")


def test_limiter_is_per_principal():
    limiter = TokenBucketLimiter(capacity=1, refill_per_sec=0.0)
    assert limiter.allow("user-c")
    assert limiter.allow("user-d")  # independent bucket
    assert not limiter.allow("user-c")


def test_anomaly_detects_burst():
    detector = AnomalyDetector(window_s=60.0, burst_threshold=3, diversity_min_requests=100, diversity_threshold=0.9)
    for _ in range(3):
        result = detector.record_and_check("user-e", "same prompt text")
    result = detector.record_and_check("user-e", "same prompt text")
    assert result.is_burst


def test_anomaly_flags_high_diversity_burst_as_extraction():
    detector = AnomalyDetector(window_s=60.0, burst_threshold=2, diversity_min_requests=3, diversity_threshold=0.5)
    prompts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    result = None
    for p in prompts:
        result = detector.record_and_check("user-f", p)
    assert result.is_extraction_signal


def test_anomaly_does_not_flag_focused_traffic():
    detector = AnomalyDetector(window_s=60.0, burst_threshold=2, diversity_min_requests=3, diversity_threshold=0.9)
    result = None
    for _ in range(5):
        result = detector.record_and_check("user-g", "summarize this document for me")
    assert not result.is_extraction_signal
