# Project Structure (Beginner Friendly)

This guide explains the important files and folders in this repo and what they are for.

## Quick Map (Diagram)

```
data/ -> notebooks -> checkpoints/ -> src/infer.py -> src/api.py -> client (test.py)
  |         |              |              |              |
  |         |              |              |              +-- FastAPI endpoints
  |         |              |              +-- Loads SavedModels
  |         |              +-- SavedModel artifacts
  |         +-- Training pipelines
  +-- MovieLens parquet files
```

## Top-level

- `README.md`: Project overview, setup, and usage.
- `requirements.txt`: Python dependencies for training and serving.
- `.env` / `.env.template`: Environment variables for API ports, model paths, and Redis.
- `LICENSE`: License information.
- `Makefile`: Convenience commands (if used).
- `PROJECT_STRUCTURE.md`: This document.

## Data

- `data/`: Local datasets in Parquet format.
  - `100k-*.parquet`: MovieLens 100K dataset (users, movies, ratings).
  - `1m-*.parquet`: MovieLens 1M dataset (users, movies, ratings).

## Notebooks

- `train_retrieval.ipynb`: Trains the retrieval model and saves artifacts.
- `train_ranking.ipynb`: Trains the ranking model and saves artifacts.
- `train_retrieval.nbconvert.ipynb`: Executed output from running the retrieval notebook.
- `train_ranking.nbconvert.ipynb`: Executed output from running the ranking notebook.

## Models and Checkpoints

- `checkpoints/`: Saved models from training.
  - `retrieval/brute/`: Brute-force retrieval model (SavedModel).
  - `retrieval/scann/`: ScaNN retrieval model (if available).
  - `ranking/pointwise/`: Pointwise ranking model (SavedModel).

## API and Serving

- `src/main.py`: Starts the FastAPI server with Uvicorn + Prometheus metrics.
- `src/api.py`: API routes:
  - `/api/healthcheck`
  - `/api/v1/retrieval`
  - `/api/v1/ranking`
- `src/infer.py`: Loads models and provides retrieval/ranking inference helpers.
- `src/config.py`: Reads environment variables for ports, paths, and Redis.

## Core Model Code

- `src/model/embedding.py`: Embedding feature definitions.
- `src/model/tower.py`: Tower network used in two-tower model.
- `src/model/retrieval.py`: Retrieval model logic.
- `src/model/ranking/`: Ranking models (base, pointwise, listwise).
- `src/model/recommender.py`: Full recommender model wiring.
- `src/model/utils/utilities.py`: Helper utilities used in training.

## Scripts and Utilities

- `scripts/dataset.py`: Dataset preparation utilities.
- `scripts/baseline_metrics.py`: Computes baseline retrieval hit rate and API latency.
- `scripts/install.sh`: Optional install helper (for Unix-like environments).

## Client / Demo

- `test.py`: Example client workflow:
  - Loads data into Redis
  - Calls retrieval API
  - Calls ranking API

---

If you want, I can also add a simple diagram or update `README.md` with this summary.
