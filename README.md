# token-xray — makes zero network calls; your data never leaves your machine

**No telemetry, no update checks, no uploads. token-xray runs fully offline: your export never leaves your machine, and the report it writes contains aggregate statistics only — never prompt text. This is enforced by tests that fail if any code path opens a socket.**

A local, offline CLI that reads an LLM API usage/log export and prints where the tokens and money went: spend by model and day, input/output token distributions, near-duplicate prompt rate, model-mix, error rate, and long-context tail statistics.

It reports **what** and **how much**. It does not tell you what to change — interpretation is out of scope by design.

## Install

```bash
pipx install token-xray        # isolated, recommended
# or
pip install token-xray
```

## 60-second quickstart

```bash
# 1. Export your usage/logs from your provider (see "Supported formats" below).
# 2. Point token-xray at the file:
token-xray analyze path/to/export.csv

# Write a human-reviewable JSON report (aggregates only):
token-xray analyze path/to/export.csv --json-out xray_report.json

# Optional single-file HTML report:
token-xray analyze path/to/export.csv --html xray_report.html
```

Try it immediately against a bundled sample:

```bash
token-xray analyze tests/fixtures/litellm_proxy_sample.jsonl
```

## Supported formats (v0)

| Format | Source | Level | Prompt text present? |
|--------|--------|-------|----------------------|
| `openai_csv` | OpenAI usage/activity CSV export (`activity-*.csv`) | billing aggregate | no |
| `anthropic_csv` | Anthropic Console exports — usage (`claude_api_tokens_*.csv`) and cost (`claude_api_cost_*.csv`) | billing aggregate | no |
| `litellm_jsonl` | LiteLLM proxy request logs (JSONL) | per-request | yes |
| `helicone` | Helicone export (CSV or JSONL) | per-request | usually |

The format is auto-detected from the file's header/shape. Unknown files produce an
explicit error naming the supported formats — never a crash.

### Metric availability

Billing-level exports (OpenAI, Anthropic) do not contain individual requests or prompt
text, so per-request metrics (token histograms, near-duplicate rate, per-request error
rate) **cannot** be computed from them. token-xray labels every such metric
`unavailable from this export format` rather than guessing. Per-request logs
(LiteLLM, Helicone) unlock the full report.

## How near-duplicate detection works

Prompt text is **never stored**. At parse time, each prompt is normalized and reduced to
a bottom-k shingle sketch (stdlib `hashlib`, no third-party dependencies) plus a salted
hash; the raw text is discarded immediately. Duplicate analysis operates only on those
sketches. An optional `[embeddings]` extra enables a local semantic mode — still fully
offline.

```bash
pip install "token-xray[embeddings]"   # optional, heavy; not required
```

### Methodology & accuracy

- Each prompt is lowercased, whitespace-normalized, split into 3-word shingles, and
  reduced to the **k = 128 smallest 64-bit shingle hashes** (a bottom-k / KMV sketch).
- For prompts with **fewer than 128 shingles (under ~130 words) the Jaccard estimate
  is exact** — the sketch contains the complete shingle set. At or beyond that, the
  standard k-minimum-values estimator applies with error on the order of
  **±1/√k (~±0.09)**.
- Two prompts count as near-duplicates at estimated **Jaccard ≥ 0.5** — roughly "the
  same template with a word or two changed."
- Candidate pairing is sub-quadratic (prefix filtering over the smallest sketch values,
  each candidate verified before it counts). The caps that keep it fast can miss pairs
  in adversarial distributions, so treat the near-duplicate rate as an **aggregate
  statistic**, not an exact census.

## Privacy guarantee (enforced in tests)

- Zero network I/O. The test suite blocks all sockets and fails if any code path opens one.
- No prompt text in any output artifact. A test parses the JSON report and asserts no raw prompt strings are present.

## Development

```bash
uv venv
uv pip install -e ".[embeddings]" --group dev   # or drop [embeddings] for the light install
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
