# Samsung Electronics Stock Forecasting Pipeline

A research-grade forecasting system for Samsung Electronics (005930.KS) built on a multi-agent orchestration architecture. The pipeline runs end-to-end from raw market data ingestion through feature engineering, walk-forward backtesting, and LLM-assisted evaluation.

> **Current Status:** The system works technically end-to-end with real market data and LLM integration. Model predictive performance is currently poor (near-random directional accuracy), which is expected and honest — this is the baseline from which improvement begins.

---

## Architecture

```
run_pipeline.py
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
│                                                         │
│  Step 1: PlannerAgent   ──► ExecutionPlan               │
│            │ (LLM: suggest feature windows              │
│            │  from past experiment history)             │
│                                                         │
│  Step 2: DataAgent      ──► DataSummary                 │
│            │ (download / cache / validate OHLCV)        │
│                                                         │
│  Step 3: ModelingAgent  ──► ModelingResult              │
│            │ (build features → LightGBM → walk-forward) │
│                                                         │
│  Step 4: EvaluationAgent ─► EvaluationReport            │
│            │ (rule-based metrics + optional LLM         │
│            │  interpretation and verdict)               │
│                                                         │
│  Step 5: ReportAgent    ──► ExperimentReport (JSON)     │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Fixed pipeline, not GroupChat | Deterministic handoff; no agent chatter |
| Two-pass evaluation | Rule-based always runs; LLM enriches only if available |
| DataFrame side-channel | Keeps all inter-agent schemas JSON-serialisable |
| Walk-forward (expanding window) | No data leakage; mirrors live trading retraining |
| LLM reads past reports | PlannerAgent suggests better features across experiments |

---

## Project Structure

```
samsung/
├── config/
│   └── default.yaml          # All hyperparameters and settings
├── src/
│   ├── agents/
│   │   ├── orchestrator.py   # run_experiment() — main entry point
│   │   ├── planner_agent.py  # Plans feature config; LLM-assisted on 2nd+ run
│   │   ├── data_agent.py     # Downloads/caches OHLCV data
│   │   ├── modeling_agent.py # Feature engineering + LightGBM walk-forward
│   │   ├── evaluation_agent.py # Metrics + LLM interpretation
│   │   ├── report_agent.py   # Saves JSON experiment report
│   │   ├── llm_client.py     # Thin LLM client (OpenAI / Anthropic / Mock)
│   │   └── schemas.py        # Pydantic v2 inter-agent contracts
│   ├── data/
│   │   ├── loader.py         # yfinance download + CSV cache
│   │   ├── schema.py         # DataConfig, OHLCVRow
│   │   └── synthetic.py      # GBM-based Samsung-like data (offline use)
│   ├── features/
│   │   ├── indicators.py     # Pure functions: RSI, ATR, MA ratio, etc.
│   │   └── pipeline.py       # build_feature_matrix()
│   ├── models/
│   │   ├── base.py           # BaseForecaster ABC
│   │   └── lgbm_model.py     # LGBMForecaster
│   ├── backtest/
│   │   ├── metrics.py        # DA, MAE, RMSE, Sharpe, IC
│   │   └── walk_forward.py   # Expanding-window WF validation
│   └── startup.py            # Preflight dependency checks
├── scripts/
│   ├── demo_agents.py        # Quick demo (synthetic or real data)
│   └── demo_agents_llm.py    # Demo with LLM modes (mock/openai/anthropic)
├── tests/                    # 92 tests across all modules
├── run_pipeline.py           # CLI entry point
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/coo001/stock-forecast.git
cd stock-forecast
pip install -r requirements.txt
```

### 2. Set API keys (Windows)

The pipeline can run with or without LLM integration.

**Without LLM (fully deterministic):**

Edit `config/default.yaml` and set:
```yaml
llm:
  provider: "none"
```

**With OpenAI:**

```cmd
# Command Prompt
set OPENAI_API_KEY=sk-...your-key-here...

# PowerShell
$env:OPENAI_API_KEY = "sk-...your-key-here..."
```

Then in `config/default.yaml`:
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
```

**With Anthropic:**

```cmd
set ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

```yaml
llm:
  provider: "anthropic"
  model: "claude-haiku-4-5-20251001"
  api_key_env: "ANTHROPIC_API_KEY"
```

> **Never hard-code API keys in config files.** The `api_key_env` field is just the environment variable name, not the key itself.

---

## Running the Pipeline

### Real Samsung market data (requires internet)

```bash
python run_pipeline.py
```

```bash
# Override LLM provider from the command line
python run_pipeline.py --llm-provider openai

# Force re-download data
python run_pipeline.py --force-refresh

# Use cached data only (no network)
python run_pipeline.py --no-download
```

### Synthetic data (offline, no yfinance needed)

```bash
python run_pipeline.py --synthetic
```

### Custom config or output directory

```bash
python run_pipeline.py --config config/default.yaml --reports-dir experiments/
```

### Fix experiment ID for reproducibility

```bash
python run_pipeline.py --experiment-id baseline_v1
```

---

## Experiment Reports

Every run saves a JSON report to `reports/<experiment_id>.json`.

**Report location:**
```
reports/
└── 20260319_143022_abc12345.json   ← auto-generated ID
└── baseline_v1.json                ← fixed ID via --experiment-id
```

**What the report contains:**
- Full `ExperimentConfig` (features, backtest params, model hyperparams)
- `DataSummary` (rows, date range, data quality flags)
- `ModelingResult` (per-fold metrics, feature importances)
- `EvaluationReport` (aggregate metrics, verdict, LLM interpretation if enabled)

**Text summary printed to stdout:**
```
============================== EXPERIMENT REPORT ==============================
Experiment : baseline_v1
Generated  : 2026-03-19T14:30:22+00:00
Ticker     : 005930.KS
Target     : next_day_log_return  (horizon=1)

------------------------------ DATA SUMMARY ----------------------------
Rows       : 2,345   |   2015-01-02  ->  2026-03-18
Sufficient : yes

------------------------------ MODEL RESULTS ---------------------------
Model      : LGBMForecaster
Folds      : 14   |   OOS observations: 882
DA         : 51.3%   |   Sharpe: 0.18   |   IC: 0.024

------------------------------ EVALUATION ------------------------------
Verdict    : marginal
...
```

---

## Current Experiment Results

> **Honest assessment:** The baseline model performs near-randomly on next-day return prediction, which is consistent with the efficient market hypothesis for large-cap liquid stocks.

| Metric | Typical Result | Interpretation |
|---|---|---|
| Directional Accuracy | ~50–52% | Near random (50% is coin flip) |
| Annualised Sharpe | 0.1–0.4 | Too low for live trading |
| Information Coefficient | 0.02–0.05 | Very weak signal |
| Verdict | `marginal` | Technically works; not tradeable |

**Top features by importance** (LightGBM gain):
- Short-term log returns (1d, 5d)
- RSI(14)
- ATR percentage (14d)
- Volume ratio

**What this means:** The current feature set captures some momentum and volatility structure, but not enough to overcome transaction costs. This is the expected starting point.

---

## Next Improvements

### 1. Better feature engineering
- **Calendar effects:** day-of-week, month, earnings season flag
- **Volatility regimes:** rolling realised vol, GARCH-like features
- **Price patterns:** gap-open size, intraday range ratio
- **Longer lookbacks:** 60d, 120d, 252d return windows

### 2. Market index and macro features
- **KOSPI / KOSDAQ index returns** — Samsung is ~25% of KOSPI; market-wide momentum matters
- **USD/KRW exchange rate** — Samsung exports ~80% overseas; FX directly impacts earnings
- **Semiconductor sector ETF** (e.g. SOXX) returns
- **VIX** — global risk sentiment
- **US 10Y treasury yield** — risk-free rate proxy

### 3. Alternative models
- **Elastic Net / Ridge regression** — interpretable linear baseline
- **XGBoost** — alternative gradient boosting
- **LSTM / Transformer** — sequence models for temporal dependencies
- **Ensemble** — stacked model combining LightGBM + linear

### 4. Better target definitions
- **5-day or 20-day forward return** — less noise than 1-day
- **Risk-adjusted return** (return / realised vol) — more stable signal
- **Direction classification** — frame as binary; use AUC instead of DA
- **Volatility target** — predict future vol rather than direction

### 5. Experiment tracking
- **MLflow** integration — log params, metrics, artifacts per run
- **Automated re-runs** — run grid search over feature configs
- **Report comparison** — `PlannerAgent` already reads past reports; extend to rank experiments
- **Alert on regime change** — detect when model degrades in live use

---

## Running Tests

```bash
# All 92 tests
pytest

# Specific module
pytest tests/test_agents.py -v
pytest tests/test_startup.py -v

# With coverage
pytest --cov=src
```

All tests use synthetic data and `MockLLMClient` — no API keys or internet required.

---

## Tech Stack

| Layer | Library |
|---|---|
| Data | `yfinance`, `pandas` |
| Features | `ta` (technical analysis), `numpy` |
| Model | `lightgbm` |
| Validation | Custom walk-forward (no sklearn leakage) |
| Agents | Custom pipeline (AutoGen-compatible design) |
| LLM | `openai` / `anthropic` (optional) |
| Schemas | `pydantic` v2 |
| Config | `pyyaml` |
| Tests | `pytest` (92 tests) |

---

## Environment Variables Reference

| Variable | Required | Used for |
|---|---|---|
| `OPENAI_API_KEY` | Only if `llm.provider: openai` | OpenAI API calls |
| `ANTHROPIC_API_KEY` | Only if `llm.provider: anthropic` | Anthropic API calls |

No other secrets or credentials are required. All data is fetched from Yahoo Finance (public, no key needed).

---

## License

MIT
