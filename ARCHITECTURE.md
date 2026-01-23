# Architecture

This document explains the model architecture, data flow, and MLOps components.

## 1) Two-Tower Model

The system uses a Two-Tower retrieval model and a separate ranking model.

- Query tower encodes user features into an embedding.
- Candidate tower encodes movie features into an embedding.
- Retrieval finds nearest candidates using FAISS/ScaNN/Brute.
- Ranking scores the candidate list with a pointwise model.

## 2) Data Flow (High Level)

```
MovieLens -> scripts/dataset.py -> data/raw
     |                         
     +-> train_retrieval.ipynb -> checkpoints/retrieval + FAISS index
     +-> train_ranking.ipynb   -> checkpoints/ranking

API -> src/infer.py loads SavedModels -> /api/v1/retrieval and /api/v1/ranking
```

## 3) Serving Flow

```
Client -> FastAPI (/api/v1/retrieval)
        -> retrieve() in src/infer.py
        -> FAISS (IndexIVFFlat) if approximate and available
        -> IDs
        -> FastAPI (/api/v1/ranking)
        -> rank() in src/infer.py
        -> scores -> response
```

## 4) MLOps Components

- MLflow: track experiments and model registry
- DVC: data + model versioning with DagsHub remote
- Prometheus + Grafana: live API metrics and training metrics
- Airflow: orchestration of the end-to-end pipeline
- GitHub Actions: CI for tests + Docker builds + DVC sync

## 5) Airflow DAG (ml_pipeline_dag)

Stages:
1. data_retrieval
2. preprocessing
3. feature_engineering
4. train_retrieval
5. train_ranking
6. evaluate_models
7. push_metrics
8. register_models
9. dvc_push

## 6) Artifacts

- `checkpoints/retrieval/brute/`: brute retrieval SavedModel
- `checkpoints/retrieval/query_tower/`: query tower SavedModel
- `checkpoints/retrieval/faiss/`: FAISS index + movie ids
- `checkpoints/ranking/pointwise/`: ranking SavedModel
- `mlruns/`: MLflow experiments and models

## 7) Monitoring Metrics

API metrics:
- recommendation_requests_total
- recommendation_latency_seconds
- model_load_time_seconds
- active_users_count

Training metrics (Pushgateway):
- retrieval_accuracy
- ranking_ndcg
- training_duration_seconds
