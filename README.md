# pt-public-data-rag

Production-style RAG pipeline over Portuguese public data (INE / dados.gov.pt):
incremental ingestion, chunking, embeddings and vector search on Databricks, with
data governance (PII masking, chunk-level lineage) and retrieval evaluation.

> 🚧 Work in progress. Built step by step as a learning and portfolio project.

## Objective
<!-- What question does this system answer, for whom, and why it is a data engineering problem. -->

## Architecture
<!-- Diagram + short description of each stage: ingestion → parsing → chunking → embeddings → index → serving. -->

## How to run
Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Databricks CLI.

```bash
cp .env.example .env            # fill in ANTHROPIC_API_KEY
databricks auth login --host <workspace-url> --profile pt-rag   # OAuth, no stored PAT
uv sync                         # create .venv with dev dependencies
uv run pytest                   # unit tests

# Ingestion (raw landing zone in data/raw/, git-ignored)
uv run python -m ingestion.ine_destaques --theme prices --max-docs 50   # press-release PDFs
uv run python -m ingestion.ine_indicators                               # CPI tables (JSON API)
```

Both ingestion scripts are idempotent: re-running them only downloads what is new.

## Evaluation
<!-- Evaluation dataset, metrics (recall@k, ...), results table comparing configurations. -->

## Governance
<!-- PII handling, lineage (document → version → chunk → embedding), access control, cost tracking. -->

**Secrets.** The LLM API key lives in a git-ignored `.env` locally and will live in a Databricks
secret scope for jobs. The Databricks CLI uses OAuth (short-lived, auto-refreshed tokens) instead
of a personal access token. The API key is long-lived because Databricks Free Edition cannot use
workload identity federation; in a production setup I would use keyless auth instead. Spend is
capped with prepaid credits (no auto-reload) and a per-workspace monthly limit.

**Source provenance.** Every downloaded PDF is recorded in a manifest (`manifest.jsonl`) with its
source URL, INE release id, publication date and SHA-256. This is the first link of the lineage
chain document → version → chunk → embedding. Data is © Statistics Portugal (INE) and used under
its open-data terms, with the source cited.

**Polite crawling.** Requests are rate-limited (≥ 1 s apart, well below INE's documented 15–20 req/s),
retried with exponential backoff on 429/5xx, and identified by a descriptive User-Agent.

## Decisions and trade-offs
<!-- Chunking strategy, vector store choice, index type, incremental refresh strategy, and why. -->

**Data source: INE "Destaques" (theme *Prices*) + CPI tables.** INE publishes several press releases
per week, so the corpus grows continuously and incremental indexing is exercised for real, not
simulated. CPI releases come out twice a month (flash estimate, then final value), and the same
facts exist as structured tables. That enables questions that combine documents and SQL later.
The whole *Prices* theme (CPI, house prices, construction costs, industrial prices) is used rather
than CPI only: near-duplicate documents ("CPI September" vs "CPI October") make retrieval
evaluation realistic.

**Language: English.** INE publishes each release in PT and EN as separate documents. EN keeps the
demo readable for an international audience and lets an English embedding model be used. Trade-off:
the multilingual retrieval challenge is postponed (the scripts already support `--lang pt`).

**Raw landing zone is immutable and append-only.** PDFs are stored as delivered, keyed by INE's
release id. Indicator snapshots are named after INE's *last update* date, so a re-run with no new
INE release is a no-op, and a new release creates a new snapshot instead of overwriting history.
Files are written atomically (temp file + rename), and the manifest is appended only after the
file is on disk, so a crash never leaves the manifest pointing to a missing file.

**PDF ids are scraped, not derived.** On INE the release id (`DESTAQUESdest_boui`) and the PDF
attachment id (`look_parentBoui`) are different numbers, so the PDF link is read from the listing.

**Indicators: watermark check, then a bounded window.** The full CPI history (1992–today, ~13 MB)
took between 2.5 and 40+ minutes to stream depending on INE's load, so the extractor works like a CDC job:
1. a cheap metadata call (~3 s) returns INE's *last update* date and *last published period*;
2. if a snapshot for that last update already exists, the run stops (a no-op re-run costs one small request);
3. otherwise only the months from `--since` up to the last published period are requested.

Lesson learned: `urlopen(timeout=...)` bounds each socket read, not the whole request. A server that
trickles bytes kept one request alive for 40 minutes. `fetch()` now enforces a total deadline as well.

## Repository layout
| Folder | Role |
|---|---|
| `ingestion/` | Download and land raw documents and tables |
| `processing/` | Parsing, cleaning, PII masking, chunking |
| `indexing/` | Embeddings and vector index (incremental refresh) |
| `serving/` | Question-answering endpoint |
| `evaluation/` | Evaluation dataset and metrics |
| `docs/` | Architecture diagrams and notes |
| `tests/` | Unit tests (pytest) |
