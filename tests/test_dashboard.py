# test dashboard functions
import io
import json
from collections import Counter, deque

from dashboard.app import _build_display, _tail_new_lines


def test_tail_new_lines_parses_json():
    content = io.StringIO(
        json.dumps({"event": "request_completed", "principal_id": "user:123"}) + "\n"
        + "invalid json line\n"
        + json.dumps({"event": "request_blocked", "reason": "jailbreak"}) + "\n"
    )
    records = list(_tail_new_lines(content, {}))
    assert len(records) == 2
    assert records[0]["event"] == "request_completed"
    assert records[1]["reason"] == "jailbreak"


def test_build_display_returns_renderable_group():
    events = deque([
        {"event": "request_completed", "principal_id": "user:1", "jailbreak_score": 0.1},
        {"event": "request_blocked", "principal_id": "user:2", "reason": "rate_limited"},
    ])
    counts = Counter({"request_completed": 1, "blocked:rate_limited": 1})

    display = _build_display(events, counts)
    assert display is not None
