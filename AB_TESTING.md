# A/B Testing Workflow

This project uses a simple A/B split (90% model A, 10% model B) based on a deterministic hash of `user_id`. All prediction scores are logged to PostgreSQL for analysis.

## 1) Requirements

- PostgreSQL running locally on port `5432`
- `.env` contains:
  ```
  DB_HOST=127.0.0.1
  DB_PORT=5432
  DB_NAME=recommender_ab
  DB_USER=postgres
  DB_PASSWORD=YOUR_PASSWORD
  ```
- FastAPI running: `python src\main.py`

## 2) How A/B split works

In `src/infer.py`:
```
bucket = hash(user_id) % 100
model A if bucket < 90
model B otherwise
```

This is deterministic, so the same user always gets the same model version.

## 3) Logging predictions

When `/api/v1/ranking` is called, we log:

- `user_id`
- `item_id`
- `model_version` (A or B)
- `score`
- `created_at`

Table: `predictions`

## 4) Generating A/B traffic

```powershell
python scripts\ab_generate.py --samples 50
```

This sends a batch of users through retrieval + ranking and logs results.

## 5) Analyze results

```powershell
python scripts\ab_test_analysis.py
```

Output includes:
- Mean NDCG for model A and B
- T-test result (t‑stat, p‑value)

## 6) Promoting the winner

If B consistently outperforms A (higher mean NDCG and statistically significant p‑value):

1) Update the split logic to route more traffic to B.
2) Re-run analysis on fresh data.
3) Once stable, switch default to B and retire A.

---

Tip: For stronger results, collect more samples and run multiple days of traffic.
