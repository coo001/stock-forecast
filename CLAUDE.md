# CLAUDE.md

## Project
This project builds a stock forecasting system for Samsung Electronics (005930.KS / 005930.KQ depending source mapping).
Goal: build a research-to-production pipeline, not a toy notebook.
Primary language: Python
Secondary: TypeScript for frontend if needed

## Product goal
We are building:
1. data ingestion pipeline
2. feature engineering pipeline
3. forecasting/backtesting module
4. FastAPI backend
5. dashboard frontend
6. AutoGen-based multi-agent orchestration for research and development workflow

## Core principles
- Prefer maintainable architecture over clever code
- Always separate research code and production code
- Every forecasting experiment must be reproducible
- Never leak future data into training features
- Always use walk-forward or time-series-safe validation
- Add logging, tests, and docstrings for important modules
- Avoid oversized files; refactor aggressively

## Tech stack
- Python 3.11+
- pandas, numpy, scikit-learn
- lightgbm or xgboost for tabular baseline
- pytorch for deep learning models
- fastapi for backend
- pydantic for schemas
- pytest for tests
- optionally streamlit or Next.js dashboard
- autogen for multi-agent orchestration

## Directory policy
- `src/data`: loaders, collectors, preprocessors
- `src/features`: technical indicators, macro/news features
- `src/models`: baseline and deep models
- `src/backtest`: walk-forward validation, metrics
- `src/api`: FastAPI routes and schemas
- `src/agents`: AutoGen agent definitions and orchestration
- `tests`: test files mirroring src layout
- `notebooks`: exploration only, not production logic

## Forecasting rules
- The target must be explicitly declared:
  - next-day close return
  - next-day direction
  - next 5-day return
- Every experiment must document:
  - target definition
  - input window
  - features used
  - train/valid/test split
  - metrics
- Prevent target leakage
- Prefer baseline first, then more complex models

## Coding rules
- Use type hints where practical
- Use small pure functions when possible
- Add comments only where logic is non-obvious
- Do not silently swallow exceptions
- Add structured logging for pipeline steps
- Keep config values in config files, not hard-coded

## AutoGen rules
- Multi-agent design should be explicit and minimal
- Define one responsibility per agent
- Prefer deterministic handoff logic over vague agent chatter
- Agent outputs should be structured JSON when possible
- Separate planning agents from execution agents
- Add retry and timeout handling for tool calls
- Log each agent’s task, output summary, and failure reason

## Development workflow
When asked to implement something:
1. inspect repository structure first
2. propose a short implementation plan
3. implement in small commits/steps
4. run relevant tests/lint if available
5. summarize changed files and remaining risks

## Output expectations
When coding:
- show plan first
- then edit files
- then explain what changed
- then mention follow-up tasks