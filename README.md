# Multi-Stage Two-Tower Recommender

![tests](https://github.com/Theepankumargandhi/multi-stage-two-tower-recommender/actions/workflows/tests.yml/badge.svg?branch=main)
![docker-build](https://github.com/Theepankumargandhi/multi-stage-two-tower-recommender/actions/workflows/docker-build.yml/badge.svg?branch=main)
![dvc-sync](https://github.com/Theepankumargandhi/multi-stage-two-tower-recommender/actions/workflows/dvc-sync.yml/badge.svg?branch=main)

A multi-stage movie recommender using a Two-Tower retrieval model and a Ranking model. The project includes MLOps tooling (MLflow, DVC, Grafana/Prometheus, Airflow), A/B testing, CI/CD, and a Streamlit UI.

## Highlights
- Two-Tower retrieval + ranking pipeline (TensorFlow Recommenders + TensorFlow Ranking)
- Approximate nearest neighbor retrieval with FAISS (IndexIVFFlat)
- FastAPI serving + Prometheus metrics
- MLflow experiment tracking + model registry
- DVC data and model versioning (DagsHub remote)
- A/B testing with PostgreSQL logging and analysis script
- Airflow DAG for end-to-end pipeline orchestration
- Streamlit UI for demo and live metrics

## Quick Start (Local)
### 1) Create environment
Use either venv or conda. Conda is recommended for FAISS on Windows.

Conda:
```bash
conda create -n ml_env310 python=3.10 -y
conda activate ml_env310
pip install -r requirements.txt
```

Venv:
```bash
python -m venv ml_env310
ml_env310\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Set environment variables
```bash
copy .env.template .env
```
Update paths if needed (see .env.template for all keys).

### 3) Run API
```bash
python src\main.py
```
API docs: http://127.0.0.1:8000/docs

### 4) Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

### 5) Quick test
```bash
python test.py
```

## Data
MovieLens datasets are converted into separate parquet files for users, movies, and ratings using `scripts/dataset.py`.

Generated files (example):
- `data/raw/100k-users.parquet`
- `data/raw/100k-movies.parquet`
- `data/raw/100k-ratings.parquet`

## Training
Two notebooks perform training and save SavedModels:
- `train_retrieval.ipynb`
- `train_ranking.ipynb`

Run notebooks:
```bash
python -m jupyter nbconvert --to notebook --execute train_retrieval.ipynb --ExecutePreprocessor.kernel_name=python3
python -m jupyter nbconvert --to notebook --execute train_ranking.ipynb --ExecutePreprocessor.kernel_name=python3
```

Artifacts are stored in `checkpoints/`:
- `checkpoints/retrieval/brute/` (SavedModel)
- `checkpoints/retrieval/query_tower/` (SavedModel)
- `checkpoints/retrieval/faiss/` (FAISS index + IDs)
- `checkpoints/ranking/pointwise/` (SavedModel)

## Retrieval Modes
- FAISS (IndexIVFFlat) when `approximate=true` and FAISS artifacts exist
- ScaNN if installed and SavedModel exists
- Brute-force fallback

## Serving
- `src/infer.py` loads models and runs retrieval/ranking
- `src/api.py` exposes FastAPI endpoints
- `/metrics` exposes Prometheus metrics

See API details in `API_DOCUMENTATION.md`.

## Streamlit UI
`streamlit_app.py` provides:
- Recommendations tab: retrieval + ranking
- Metrics tab: live Prometheus metrics

Set `API_BASE_URL` if the API is not local.

## MLflow
- Run UI: `mlflow ui --backend-store-uri ./mlruns`
- Open: http://127.0.0.1:5000
- Experiments: `retrieval`, `ranking`
- Registry models: `ttr_retrieval`, `ttr_ranking`

See `MLFLOW_GUIDE.md`.

## Monitoring (Prometheus + Grafana)
- FastAPI exports `/metrics`
- Prometheus scrapes API + Pushgateway
- Grafana dashboards include latency, request rate, accuracy, NDCG

See `GRAFANA_GUIDE.md`.

## DVC
- Data and checkpoints tracked with DVC
- Remote: DagsHub

See `DVC_GUIDE.md`.

## A/B Testing
- 90/10 split based on user_id hash
- Logged to PostgreSQL in `predictions` table
- Analysis: `scripts/ab_test_analysis.py`

See `AB_TESTING.md`.

## Airflow
- DAG: `dags/ml_pipeline_dag.py`
- Orchestrates data retrieval, training, metrics, registry, DVC

See `ARCHITECTURE.md` for pipeline flow.

## Docker (Config-Only)
Docker files and compose are included but not tested locally in this environment.

```bash
cp .env.template .env
docker compose up --build -d
```

Services:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

Note: compose uses local volumes for `data/`, `checkpoints/`, `mlruns/`.

## Project Structure (Short)
- `src/`: model + API code
- `scripts/`: data prep, baseline metrics, A/B tooling
- `checkpoints/`: SavedModels and FAISS index
- `streamlit_app.py`: Streamlit UI
- `docker-compose.yml`, `Dockerfile.*`: container config

Full list: `PROJECT_STRUCTURE.md`.

## Troubleshooting
- `ModuleNotFoundError`: install missing package in the active env
- Port 8000 already in use: stop old API or change port
- No metrics in Grafana: run API and Pushgateway, then generate traffic
- FAISS missing on Windows: install via conda (faiss-cpu)
- DVC pull errors: ensure remote + token configured

## References
Videos:
- https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz

Papers:
- https://research.google.com/pubs/archive/45530.pdf

Libraries:
- https://github.com/tensorflow/recommenders
- https://github.com/tensorflow/ranking

## License
MIT (see LICENSE).