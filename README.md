# LLM Security Gateway
### (This project is currently in developing phase)

This project is a lightweight scaffold for an LLM security gateway that combines multiple defensive modules:

- Jailbreak and prompt-injection detection
- PII and sensitive-data redaction
- Token-bucket rate limiting and burst detection
- Extraction-signal monitoring for model theft or probing behavior

## Module overview

### 1. Jailbreak detection
The jailbreak classifier uses a simple keyword-based approach for initial detection and can be extended to train on Hugging Face datasets such as jailbreak prompt corpora and benign instruction data.

### 2. PII / sensitive data leakage
The PII scrubber redacts common patterns such as email addresses, phone numbers, and SSNs before data is logged or forwarded.

### 3. Token abuse
The rate limiter uses a token bucket to enforce per-user request budgets, while the anomaly detector scores bursty traffic patterns that may indicate abuse.

### 4. Model theft / extraction signals
This module uses a self-defined heuristic for suspicious extraction behavior based on sharply increased request volume and semantic diversity. There is no standard public benchmark for this task, so its ground truth is intentionally defined by the project rather than derived from an external labeled dataset.
