# classifier
"""
Two-tier jailbreak / prompt-injection detection.

Tier 1 (always on): keyword and pattern matching against known jailbreak
framings (role-play override, instruction override, obfuscation cues).
Zero dependencies, sub-millisecond, catches the common/lazy cases.

Tier 2 (optional): a fine-tuned DistilBERT sequence classifier, trained
by app/jailbreak/train.py (see that file + scripts/build_train_set.py
and scripts/check_contamination.py for the full data pipeline: rubend18
+ TrustAIRLab jailbreak prompts vs. Alpaca as train, with a cross-source
JailbreakBench + held-out-Alpaca val/test split, contamination-checked
via embedding cosine similarity before the reported metrics are trusted).
Loaded lazily from models/jailbreak_classifier/; if that directory
doesn't exist (e.g. training hasn't been run in this environment) the
classifier silently falls back to tier 1 only rather than failing the
request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

# Patterns are intentionally broad phrase-fragments, not exact strings,
# so trivial rewording doesn't bypass tier 1.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ignore_instructions", re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I)),
    ("role_override", re.compile(r"\byou\s+are\s+now\s+(DAN|[A-Z]{2,})\b", re.I)),
    ("no_restrictions", re.compile(r"\b(no|without)\s+(restrictions?|filters?|limitations?|guardrails?)\b", re.I)),
    ("dev_mode", re.compile(r"\b(developer|debug|jailbreak|god|admin)\s*mode\b", re.I)),
    ("pretend_no_rules", re.compile(r"\bpretend\s+(you\s+)?(have\s+no|there\s+are\s+no)\s+rules?\b", re.I)),
    ("system_prompt_leak", re.compile(r"\b(reveal|print|show|repeat)\s+(your\s+)?(system\s+prompt|instructions)\b", re.I)),
    ("hypothetical_bypass", re.compile(r"\bhypothetically,?\s+if\s+you\s+(had\s+no|could\s+ignore)\b", re.I)),
    ("token_smuggling", re.compile(r"[A-Za-z]\s*-\s*[A-Za-z]\s*-\s*[A-Za-z]\s*-\s*[A-Za-z]")),  # e.g. "b-o-m-b"
]


@dataclass
class JailbreakResult:
    is_jailbreak: bool
    score: float
    method: str  # "heuristic" | "ml" | "heuristic+ml"
    matched_patterns: list[str] = field(default_factory=list)


class JailbreakClassifier:
    def __init__(self):
        self._settings = get_settings()
        self._ml_model = None
        self._tokenizer = None
        self._ml_load_attempted = False

    def _try_load_ml(self) -> None:
        if self._ml_load_attempted:
            return
        self._ml_load_attempted = True
        model_dir = Path(self._settings.jailbreak_model_path)
        if not model_dir.exists():
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._ml_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self._ml_model.eval()
            self._torch = torch
        except Exception:
            # Missing optional deps or corrupt artifact: degrade to heuristic-only.
            self._ml_model = None
            self._tokenizer = None

    def _heuristic_score(self, text: str) -> tuple[float, list[str]]:
        matched = [name for name, pattern in _PATTERNS if pattern.search(text)]
        if not matched:
            return 0.0, []
        # Each match adds confidence; saturate at 1.0.
        score = min(1.0, 0.4 + 0.2 * len(matched))
        return score, matched

    def _ml_score(self, text: str) -> float | None:
        self._try_load_ml()
        if self._ml_model is None or self._tokenizer is None:
            return None
        try:
            inputs = self._tokenizer(
                text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
            )
            with self._torch.no_grad():
                logits = self._ml_model(**inputs).logits
                proba = self._torch.softmax(logits, dim=-1)[0]
            # Assumes class index 1 == "jailbreak" (see train.py label encoding).
            return float(proba[1])
        except Exception:
            return None

    def classify(self, text: str) -> JailbreakResult:
        if not text:
            return JailbreakResult(is_jailbreak=False, score=0.0, method="heuristic")

        heuristic_score, matched = self._heuristic_score(text)
        ml_score = self._ml_score(text)

        if ml_score is None:
            final_score = heuristic_score
            method = "heuristic"
        else:
            # Weighted blend: ML carries more weight when available, but a
            # strong heuristic hit can't be diluted away by a low ML score.
            final_score = max(heuristic_score, 0.6 * ml_score + 0.4 * heuristic_score)
            method = "heuristic+ml"

        threshold = self._settings.jailbreak_threshold
        return JailbreakResult(
            is_jailbreak=final_score >= threshold,
            score=round(final_score, 4),
            method=method,
            matched_patterns=matched,
        )


classifier = JailbreakClassifier()
