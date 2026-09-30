# LLM Security Gateway

### (Modular Enterprise Architecture v0.2.0 — Production-Ready Security Pipeline with Docker Support)

A security-focused, modular FastAPI gateway that sits in front of upstream LLMs (local Ollama with `llama3.1`, OpenAI-compatible endpoints, or mock providers) and enforces defense-in-depth security interceptors on every request before it reaches the model, and on every response before it reaches the client.

```
Client 
  │ [X-API-Key]
  ▼
[AuthStage]            -> API Key authentication & RBAC permission scope enforcement
  │
  ▼
[RateLimitStage]       -> Per-principal token bucket rate limiting (cheapest check first)
  │
  ▼
[AnomalyStage]         -> Sliding-window burst & token-set Jaccard diversity extraction detection
  │
  ▼
[JailbreakStage]       -> Two-tier defense: fast regex heuristics + DistilBERT sequence classifier
  │
  ▼
[PIIStage (outbound)]  -> Redacts sensitive PII (emails, cards, SSNs, phones, IPs, API keys)
  │
  ▼
[Upstream LLM Client]  -> Dispatches sanitized prompt to Ollama / OpenAI / Mock backend
  │
  ▼
[PIIStage (inbound)]   -> Redacts sensitive PII leaked from upstream model completions
  │
  ▼
[AuditStage]           -> Structured JSONL audit logging with sanitized prompts and telemetry
  │
  ▼
Client Response
```

---

## File structure

```
llm-security-gateway/
├── config.yaml                    # Declarative hierarchical gateway configuration
├── .env.example                   # Environment variable template overrides
├── params.yaml                    # Dataset sources + DistilBERT training hyperparameters
├── requirements.txt               # Dependencies (FastAPI, Uvicorn, PyTorch, Transformers, etc.)
├── Dockerfile                     # Multi-stage production container build (non-root user)
├── docker-compose.yml             # Orchestration: Gateway + Ollama services
│
├── app/
│   ├── main.py                    # Modular FastAPI app entrypoint
│   ├── config.py                  # Dual-source config loader (config.yaml + env overrides)
│   │
│   ├── core/                      # Core architectural abstractions
│   │   ├── context.py             # SecurityContext (request/response state & block verdicts)
│   │   └── pipeline.py            # Pluggable SecurityPipeline engine & PipelineStage base
│   │
│   ├── stages/                    # Modular security stages (individually toggleable)
│   │   ├── auth_stage.py          # API key & RBAC authorization stage
│   │   ├── rate_limit_stage.py    # Rate limiting enforcement stage
│   │   ├── anomaly_stage.py       # Burst & model extraction detection stage
│   │   ├── jailbreak_stage.py     # Heuristic + ML jailbreak filtering stage
│   │   ├── pii_stage.py           # Inbound/outbound PII scrubbing stage
│   │   └── audit_stage.py         # Structured JSON audit logging stage
│   │
│   ├── api/                       # API layer with separated routers
│   │   ├── router.py              # Aggregated API router
│   │   └── v1/
│   │       ├── chat.py            # POST /v1/chat endpoint
│   │       ├── admin.py           # GET /admin/stats endpoint
│   │       └── health.py          # GET /healthz endpoint
│   │
│   ├── schemas/                   # Pydantic request & response models
│   │   └── chat.py                # ChatRequest, ChatResponse, HealthResponse, AdminStatsResponse
│   │
│   ├── auth/                      # Authentication & RBAC engine
│   │   └── rbac.py                # API-key lookup + role/scope enforcement
│   ├── jailbreak/                 # Jailbreak classification engine
│   │   ├── classifier.py          # Two-tier jailbreak/prompt-injection detector
│   │   └── train.py               # Fine-tunes DistilBERT on data/train_set.csv
│   ├── llm/                       # Upstream LLM proxy client
│   │   └── client.py              # Async HTTP client: mock / ollama / OpenAI-compatible
│   ├── logging/                   # Audit logging subsystem
│   │   └── logger.py              # Structured JSONL audit logger (PII-redacted)
│   ├── pii/                       # PII scrubbing engine
│   │   └── scrubber.py            # Regex + Luhn-validated sensitive-data redaction
│   └── rate_limit/                # Rate limiting & anomaly engine
│       ├── limiter.py             # Token-bucket rate limiter (per principal)
│       └── anomaly.py             # Sliding-window burst + extraction-signal detector
│
├── dashboard/
│   └── app.py                     # Rich-based live terminal dashboard (tails LOG_FILE)
│
├── scripts/
│   ├── build_train_set.py         # Pulls HF datasets, assembles train/val/test CSVs
│   ├── check_contamination.py     # Embedding-similarity check: train vs val/test overlap
│   ├── check_label_bias.py        # Gate: fails if any PII pattern >90% correlated with one label
│   ├── generate_benign_pii_examples.py # Generates synthetic benign PII training rows
│   └── simulate_traffic.py        # CLI traffic generator (benign/jailbreak/PII/burst)
│
├── tests/
│   ├── test_api.py                # 9 FastAPI endpoint integration tests
│   ├── test_auth.py               # 5 RBAC & permission scope unit tests
│   ├── test_pipeline.py           # 5 Modular pipeline engine tests
│   ├── test_dashboard.py          # 2 Terminal dashboard unit tests
│   ├── test_jailbreak.py          # 7 Heuristic & ML classifier tests
│   ├── test_pii.py                # 7 PII redaction & Luhn card tests
│   ├── test_rate_limit.py         # 6 Rate limiter & anomaly detector tests
│   └── test_label_bias.py         # 9 Label bias gate tests
│
├── data/
│   ├── train_set.csv              # 4,225 rows (post-fix; see Dataset section)
│   ├── train_set_prefix.csv       # Pre-fix backup, kept for regression testing
│   ├── val_set_clean.csv
│   ├── test_set_clean.csv
│   └── *_contamination_report.json
│
└── models/
    ├── jailbreak_classifier/      # Fine-tuned DistilBERT: model.safetensors (~268MB),
    │                              #   config.json, tokenizer files, test_metrics.txt
    └── checkpoints/               # checkpoint-118 … checkpoint-393
```

---

## Modules

### 1. Auth & RBAC — `app/auth/rbac.py`
Custom header-based auth (not OAuth Bearer) with three roles: `admin`, `user`, `readonly`, each mapped to permission scopes (`chat`, `admin:view_stats`, `admin:manage_keys`).

- **Input:** `X-API-Key` request header
- **Process:** look up key in `API_KEYS_JSON` → resolve `Principal` (id + role) → check role has the scope the endpoint requires
- **Output:** authenticated `Principal` passed to the request handler, or a rejection (`401`/`403`) before any other module runs

### 2. Rate limiting — `app/rate_limit/limiter.py`
Token-bucket limiter, one bucket per principal, continuous refill.

- **Input:** principal ID, `RATE_LIMIT_CAPACITY`, `RATE_LIMIT_REFILL_PER_SEC`
- **Process:** deduct one token per request; refill continuously based on elapsed time; reject if bucket is empty
- **Output:** allow/deny decision for the request

### 3. Anomaly / extraction detection — `app/rate_limit/anomaly.py`
Self-defined heuristic (no public benchmark exists for this task) flagging probing/model-theft-style traffic.

- **Input:** rolling window of a principal's recent prompts (`ANOMALY_WINDOW_S`)
- **Process:** count requests in the window (burst check vs `ANOMALY_BURST_THRESHOLD`); compute Jaccard token diversity across consecutive prompts (extraction check vs `ANOMALY_DIVERSITY_MIN_REQUESTS` / `ANOMALY_DIVERSITY_THRESHOLD`) — high diversity + high volume looks like systematic probing rather than normal repeated use
- **Output:** burst flag / extraction flag, logged; does not currently block on its own (verify in code if you want hard blocking)

### 4. Jailbreak / prompt-injection detection — `app/jailbreak/classifier.py`
Two-tier classifier.

- **Input:** the raw prompt string
- **Process:**
  - **Tier 1 (heuristic):** regex pattern set (e.g. `ignore_instructions`, `system_prompt_leak`); each match adds to a saturating score
  - **Tier 2 (ML):** fine-tuned DistilBERT (lazy-loaded; degrades silently to Tier 1 only if the model/packages aren't available) produces its own score
  - **Blend:** `final_score = max(heuristic_score, 0.6 × ml_score + 0.4 × heuristic_score)` — a strong heuristic hit can't be diluted by a low ML score
- **Output:** `is_jailbreak` (bool, vs `JAILBREAK_THRESHOLD`), numeric score, method used (`heuristic` / `heuristic+ml`), matched pattern names — request is rejected before reaching the LLM if flagged

### 5. PII / sensitive-data redaction — `app/pii/scrubber.py`
- **Input:** prompt text (outbound, before Ollama) and response text (inbound, before logging/returning)
- **Process:** regex-based detection and redaction of emails, phone numbers, SSNs, and credit card numbers (Luhn-validated to avoid flagging random digit strings), plus IPs and API-key-shaped strings
- **Output:** redacted text + `pii_redacted` boolean; redaction is applied to what's logged (`PII_REDACT_IN_LOGS`), and can optionally block on detection (`PII_BLOCK_ON_DETECT`)

### 6. LLM client — `app/llm/client.py`
- **Input:** sanitized prompt, `LLM_PROVIDER` config
- **Process:** three provider modes — `mock` (canned response, no network call), `ollama` (bypasses API-key requirement, targets `{LLM_BASE_URL}/chat/completions`, e.g. `http://localhost:11434/v1/chat/completions`), or generic OpenAI-compatible (requires `LLM_API_KEY`)
- **Output:** model completion text, or a structured `LLMError` on failure/timeout

### 7. Structured logging — `app/logging/logger.py`
- **Input:** every pipeline event (`request_completed`, `request_blocked`, with reason)
- **Process:** redacts PII from prompt/response fields, timestamps, serializes to JSON
- **Output:** one line per event appended to `LOG_FILE` (`logs/gateway.jsonl`) and stdout

### 8. Dashboard — `dashboard/app.py`
- **Input:** `LOG_FILE`
- **Process:** tails the file live using `rich.live`
- **Output:** terminal UI showing running stats and recent request events (not yet covered by automated tests)

---

## Dataset — jailbreak classifier training data

**Sources** (pulled via `scripts/build_train_set.py`):
- `rubend18` and `TrustAIRLab` (`in_the_wild`) jailbreak-prompt corpora — positive (label `1`) examples
- `tatsu_lab/alpaca` — benign instruction data — negative (label `0`) examples
- `JailbreakBench` — additional positive examples
- `synthetic_benign_pii` (generated in-house via `scripts/generate_benign_pii_examples.py`) — 46 benign examples containing realistic-but-fake contact details (fake domains, fictitious area codes, templated ID/card formats), added specifically to correct a labeling bias (see below)

**Pipeline:**
1. `build_train_set.py` downloads and assembles disjoint `train` / `val` / `test` CSV splits (`text,label,source` schema) into `data/`
2. `check_contamination.py` computes sentence-transformer embedding cosine similarity between train and val/test sets and removes overlaps → produces `val_set_clean.csv`, `test_set_clean.csv`
3. `check_label_bias.py` (added after the bug below) scans for any regex-detectable pattern (email/phone/SSN/card-shaped) that correlates >90% with a single label across ≥10 rows, and **fails the build** if found — this runs both at the end of `build_train_set.py` and again as a pre-training gate inside `train.py`
4. `train.py` fine-tunes `distilbert-base-uncased` on `train_set.csv`, evaluates on `test_set_clean.csv`, exports to `models/jailbreak_classifier/`

**Known-fixed data bug:** every `@`-containing row in the original `train_set.csv` (35/35) was labeled jailbreak with zero benign counterexamples — the model learned "contains an email/phone pattern" as a proxy for "jailbreak," causing false-positive blocks on ordinary PII-mentioning prompts in production. Fixed by adding the 46 balanced synthetic benign examples (4,179 → 4,225 rows) and adding the permanent `check_label_bias.py` gate so this can't silently recur on a future data rebuild.

**Current model metrics** (`models/jailbreak_classifier/test_metrics.txt`, post-fix):

| Metric | Value |
|---|---|
| eval_loss | 0.7969 |
| eval_accuracy | 0.7930 |
| eval_precision | 1.0000 |
| eval_recall | 0.6468 |
| eval_f1 | 0.7855 |

Recall (~65%) is the known open gap, attributed to a stylistic mismatch between roleplay-framed training sources and more direct-phrased evaluation prompts.

---

## Current status

- **50 tests collected, 49 passed, 1 skipped (optional ML dependency)** (`pytest -v`):
  - API endpoint integration (`tests/test_api.py`): 9 tests passed
  - RBAC & Scope enforcement (`tests/test_auth.py`): 5 tests passed
  - Modular Security Pipeline (`tests/test_pipeline.py`): 5 tests passed
  - Terminal Dashboard (`tests/test_dashboard.py`): 2 tests passed
  - Jailbreak Heuristic & Blending (`tests/test_jailbreak.py`): 6 tests passed, 1 skipped
  - PII Redaction & Luhn validation (`tests/test_pii.py`): 7 tests passed
  - Token Bucket & Anomaly detection (`tests/test_rate_limit.py`): 6 tests passed
  - Training Data Bias gate (`tests/test_label_bias.py`): 9 tests passed
- **Modular Architecture Implemented:** Interceptor pipeline pattern (`app/core/pipeline.py`), decoupled stage components (`app/stages/`), structured Pydantic schemas (`app/schemas/`), and split API routers (`app/api/`).
- **Configuration Overhaul:** Hierarchical YAML configuration (`config.yaml`) with environment variable overrides and individual stage activation toggles.
- **Docker Packaging Complete:** Production multi-stage `Dockerfile` (non-root security, health checks) and `docker-compose.yml` for gateway + Ollama orchestration.