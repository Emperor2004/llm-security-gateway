# app
"""
Terminal dashboard: tails the gateway's JSON-lines log file (LOG_FILE)
and renders a live summary — no separate web framework needed since
`rich` is already a project dependency.

Usage:
    python -m dashboard.app
"""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group

from app.config import get_settings


def _tail_new_lines(f, state: dict):
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _build_display(events: deque, counts: Counter) -> Group:
    summary = Table(title="Gateway Summary", expand=True)
    summary.add_column("Metric")
    summary.add_column("Count", justify="right")
    summary.add_row("Total events", str(sum(counts.values())))
    summary.add_row("Completed", str(counts.get("request_completed", 0)))
    summary.add_row("Blocked — jailbreak", str(counts.get("blocked:jailbreak", 0)))
    summary.add_row("Blocked — rate limit", str(counts.get("blocked:rate_limited", 0)))
    summary.add_row("Blocked — extraction signal", str(counts.get("blocked:extraction_signal", 0)))
    summary.add_row("Blocked — PII", str(counts.get("blocked:pii_detected", 0)))
    summary.add_row("Upstream errors", str(counts.get("upstream_error", 0)))

    recent = Table(title="Recent events", expand=True)
    recent.add_column("event")
    recent.add_column("principal")
    recent.add_column("detail")
    for e in list(events)[-15:][::-1]:
        detail = e.get("reason") or e.get("error") or f"score={e.get('jailbreak_score', '-')}"
        recent.add_row(e.get("event", "?"), str(e.get("principal_id", "-")), str(detail))

    return Group(Panel(summary), Panel(recent))


def run():
    settings = get_settings()
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    events: deque = deque(maxlen=500)
    counts: Counter = Counter()

    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(0, 2)  # start at end — only show new events from now on
        with Live(_build_display(events, counts), refresh_per_second=2) as live:
            while True:
                for record in _tail_new_lines(f, {}):
                    events.append(record)
                    event_type = record.get("event", "unknown")
                    if event_type == "request_blocked":
                        counts[f"blocked:{record.get('reason', 'unknown')}"] += 1
                    else:
                        counts[event_type] += 1
                live.update(_build_display(events, counts))
                time.sleep(0.5)


if __name__ == "__main__":
    run()
