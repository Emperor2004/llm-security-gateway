"""
Automated correlation check for label bias on training dataset CSVs.

Checks regex patterns (email, phone, SSN, credit card) against labeled rows.
Fails (exits non-zero) if any pattern matching >= 10 rows exhibits > 90% correlation with a single label class.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
import pandas as pd

DEFAULT_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"@[\w.-]+"),
    "phone": re.compile(r"\d{3}[-.]?\d{3}[-.]?\d{4}"),
    "ssn": re.compile(r"\d{3}-\d{2}-\d{4}"),
    "credit_card": re.compile(r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}"),
}

CORRELATION_THRESHOLD = 0.90
MIN_MATCHING_ROWS = 10


def check_label_bias(csv_path: str = "./data/train_set.csv", patterns: dict[str, re.Pattern] | None = None) -> bool:
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] Dataset file {csv_path} does not exist.", file=sys.stderr)
        return False

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        print(f"[ERROR] CSV {csv_path} must contain 'text' and 'label' columns.", file=sys.stderr)
        return False

    # Convert text column to string to handle any NaN or non-string values safely
    texts = df["text"].fillna("").astype(str)
    labels = df["label"]

    bias_detected = False
    print(f"Checking label bias for {csv_path} ({len(df)} total rows)...")

    for name, pattern in patterns.items():
        matches_mask = texts.str.contains(pattern, regex=True)
        matching_labels = labels[matches_mask]
        total_matches = len(matching_labels)

        if total_matches < MIN_MATCHING_ROWS:
            print(f"  Pattern '{name}' ({pattern.pattern}): {total_matches} matches (below min threshold {MIN_MATCHING_ROWS}), skipping.")
            continue

        count_1 = (matching_labels == 1).sum()
        count_0 = (matching_labels == 0).sum()
        corr_1 = count_1 / total_matches
        corr_0 = count_0 / total_matches

        max_corr = max(corr_1, corr_0)
        dominant_label = 1 if corr_1 > corr_0 else 0

        print(f"  Pattern '{name}' ({pattern.pattern}): {total_matches} matches -> Label 0: {count_0} ({corr_0:.1%}), Label 1: {count_1} ({corr_1:.1%})")

        if max_corr > CORRELATION_THRESHOLD:
            bias_detected = True
            print(
                f"[BIAS ERROR] Pattern '{name}' exhibits severe label bias! "
                f"{dominant_label} label correlation is {max_corr:.1%} (> {CORRELATION_THRESHOLD:.0%}) "
                f"across {total_matches} matching rows (Label 0: {count_0}, Label 1: {count_1}).",
                file=sys.stderr
            )

    if bias_detected:
        print(f"\n[FAIL] Label bias check FAILED for {csv_path}.", file=sys.stderr)
        return False

    print(f"\n[PASS] Label bias check PASSED for {csv_path}.")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/train_set.csv"
    success = check_label_bias(target)
    if not success:
        sys.exit(1)
    sys.exit(0)
