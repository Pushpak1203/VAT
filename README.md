# VAT Chain Auditor

Production-oriented research/CLI implementation of an evidence-grounded EU cross-border VAT compliance pipeline.

## Python and environment

- Python **3.10.10**
- VS Code
- No Jupyter notebooks
- No web server
- Models download from Hugging Face at runtime
- GPU is optional; the intended development target is an NVIDIA RTX 3050 6 GB

### Windows / VS Code setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

For NVIDIA CUDA, install the PyTorch build appropriate for the CUDA runtime installed on the machine if the pinned wheel is not suitable for your environment. The application detects CPU/CUDA and refuses unsafe 4-bit loading when the required quantization stack is unavailable.

> Gemma 3 is optional and is not loaded by the core pipeline. Its Hugging Face repository may require access approval/authentication.

## Run

```powershell
python main.py --input data\sample_transactions.json
python main.py --input data\sample_invoices\sample_invoice.png --mode HYBRID_RAG
python main.py --input data\sample_transactions.json --mode RULES_ONLY
python main.py --benchmark
```

The five supported modes are:

- `LLM_ONLY`
- `RULES_ONLY`
- `HYBRID`
- `HYBRID_RAG`
- `DEEPSEEK_RAG`

## PostgreSQL / pgvector

The RAG layer uses PostgreSQL + pgvector. Create the database and enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The application creates the vector table/index when `ensure_schema()` is called. A pgvector-enabled PostgreSQL instance is required for database-backed retrieval. Tests use an in-memory fallback retriever so unit tests do not require PostgreSQL.

## Model stack

The core model stack is intentionally limited to the requested models:

1. Qwen3-8B — primary LLM
2. DeepSeek-R1-Distill-Qwen-7B — reasoning benchmark
3. Gemma 3 4B — optional modular extractor/classifier
4. BAAI/bge-m3 — legal embeddings
5. BAAI/bge-reranker-v2-m3 — legal reranking
6. microsoft/deberta-v3-base — transaction representation/classification
7. LightGBM — risk scoring
8. Isolation Forest — anomaly detection
9. LayoutLMv3 — document/layout representation
10. PaddleOCR — OCR
11. Isotonic Regression — confidence calibration

No other ML model is introduced.

## Important research limitation

The repository contains runnable model wrappers and deterministic inference scaffolding. Base checkpoints such as `microsoft/deberta-v3-base` and `microsoft/layoutlmv3-base` are not VAT-fine-tuned checkpoints. The code therefore includes deterministic prototype/heuristic fallbacks and explicit training hooks rather than pretending an untrained classification head is a validated VAT classifier.

Likewise, LightGBM and isotonic calibration load learned artifacts when present and otherwise use deterministic bootstrap behavior so the CLI remains runnable. For research claims, train these artifacts on a labelled VAT dataset and report the resulting metrics.

## Input formats

### Structured transaction JSON

A JSON object or a list of objects can be supplied. Example:

```json
{
  "transaction_id": "TX-001",
  "seller_country": "DE",
  "buyer_country": "FR",
  "buyer_vat_id": "FR12345678901",
  "customer_type": "B2B",
  "supply_type": "goods",
  "invoice_date": "2026-09-30",
  "currency": "EUR",
  "net_amount": 1000,
  "vat_amount": 0,
  "gross_amount": 1000,
  "description": "Intra-EU supply of goods",
  "destination_country": "FR"
}
```

### Invoice image

Use a PNG/JPEG/TIFF. PaddleOCR extracts OCR text and LayoutLMv3 processes the image/layout representation. Field extraction then applies deterministic VAT-ID, date and amount parsers.

## Output

The final `VATAuditResult` contains:

- VAT treatment decision
- compliance status
- risk score
- anomaly flag
- explanation/rationale
- legal references
- calibrated confidence
- intermediate evidence for reproducibility

## Tests

```powershell
pytest -q
```

The tests avoid downloading large models and therefore validate deterministic rules, schema behavior, in-memory RAG, and pipeline routing without requiring GPU weights.

## Benchmark output

```powershell
python main.py --benchmark
```

Results are written to `benchmarks/results/benchmark_results.json`.

The benchmark runner measures accuracy/F1 when labels are supplied, plus latency and confidence statistics. It does not manufacture performance improvements; measured results depend on the labelled evaluation dataset and locally available model artifacts.
