# MLflow Quick Guide

This guide shows how to start MLflow, open the UI, and find experiments.

## Start MLflow UI

From the repo root:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Then open:

```
http://127.0.0.1:5000
```

## Find Experiments

1) Click **Experiments** in the left sidebar.
2) Open:
   - `retrieval` for two‑tower retrieval training runs.
   - `ranking` for ranking model training runs.
3) Click the latest run to see:
   - **Metrics** (e.g., `retrieval_top_k_accuracy`, `val_ndcg`)
   - **Parameters** (learning rate, embedding dims, batch size, epochs)
   - **Artifacts** and **Logged models**

## Model Registry

1) Click **Models** in the left sidebar.
2) Look for:
   - `ttr_retrieval`
   - `ttr_ranking`
3) Click a model name to view versions, metrics, and artifacts.

---

Tip: If the UI doesn’t load, confirm the server is running and port `5000` is free.
