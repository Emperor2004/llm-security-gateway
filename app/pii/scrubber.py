# scrubber
"""
Regex-based PII / sensitive-data redaction.

Deliberately not ML-based: PII patterns (emails, phone numbers, SSNs,
card numbers, API keys, IPs) are structured enough that regex + checksum
validation (Luhn for cards) catches the overwhelming majority of cases
with zero inference cost, which matters because this runs on every
request and response.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9.\-_%+]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(
        r"(?<!\d)(\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    ),
    "api_key": re.compile(r"\b(?:sk|pk|api|key)-[A-Za-z0-9]{16,}\b", re.IGNORECASE),
}

REDACTION_TAG = "[REDACTED:{}]"


def _luhn_valid(digits: str) -> bool:
    d = [int(c) for c in digits]
    checksum = 0
    parity = len(d) % 2
    for i, digit in enumerate(d):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


@dataclass
class Finding:
    type: str
    count: int = 0


@dataclass
class ScrubResult:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return len(self.findings) > 0


def scrub(text: str) -> ScrubResult:
    if not text:
        return ScrubResult(text=text, findings=[])

    result = text
    counts: dict[str, int] = {}

    for pii_type, pattern in PATTERNS.items():
        def _replace(match: re.Match, pii_type=pii_type) -> str:
            value = match.group(0)
            if pii_type == "credit_card":
                digits = re.sub(r"[ -]", "", value)
                if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
                    return value  # not actually a card number, leave untouched
            counts[pii_type] = counts.get(pii_type, 0) + 1
            return REDACTION_TAG.format(pii_type)

        result = pattern.sub(_replace, result)

    findings = [Finding(type=t, count=c) for t, c in counts.items()]
    return ScrubResult(text=result, findings=findings)
