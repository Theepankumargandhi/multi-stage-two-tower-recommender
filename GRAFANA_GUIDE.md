# Grafana Monitoring Guide (Beginner Friendly)

This guide shows exactly how we set up monitoring **without Docker** on Windows.

## 1) What you need running (3 terminals)

You must keep these running at the same time:

1) **FastAPI app** (metrics source)
```powershell
python src\main.py
```

2) **Prometheus** (scrapes metrics)
```powershell
& "C:\Users\theep\Downloads\prometheus-3.9.1.windows-amd64\prometheus-3.9.1.windows-amd64\prometheus.exe" `
  --config.file "D:\Studies\Theepan\Studies\Project\multi-stage-two-tower-recommender\multi-stage-two-tower-recommender\prometheus.yml" `
  --web.listen-address=":9092"
```

3) **Pushgateway** (receives training metrics)
```powershell
& "C:\Users\theep\Downloads\pushgateway-1.11.2.windows-amd64\pushgateway-1.11.2.windows-amd64\pushgateway.exe" `
  --web.listen-address=":9091"
```

## 2) Verify metrics are exposed

Open in your browser:
```
http://127.0.0.1:8000/metrics
```
You should see metrics like:
- `recommendation_requests_total`
- `recommendation_latency_seconds`
- `model_load_time_seconds`
- `active_users_count`

## 3) Prometheus config (already created)

File: `prometheus.yml`
```yaml
scrape_configs:
  - job_name: "fastapi"
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8000"]

  - job_name: "pushgateway"
    static_configs:
      - targets: ["127.0.0.1:9091"]
```

Prometheus UI:  
```
http://127.0.0.1:9092
```

## 4) Grafana setup

Open Grafana:  
```
http://127.0.0.1:3000
```
Login: `admin / admin` (then set a new password).

### Add Prometheus data source
Connections → Data sources → Add data source → Prometheus  
URL:
```
http://127.0.0.1:9092
```
Click **Save & Test** (should say success).

## 5) Dashboard panels we created

### API Performance Metrics (combined dashboard)

**Request rate (req/s)**
```
rate(recommendation_requests_total[1m])
```

**Latency (P50/P95/P99)**
```
histogram_quantile(0.50, sum(rate(recommendation_latency_seconds_bucket[1m])) by (le, endpoint))
histogram_quantile(0.95, sum(rate(recommendation_latency_seconds_bucket[1m])) by (le, endpoint))
histogram_quantile(0.99, sum(rate(recommendation_latency_seconds_bucket[1m])) by (le, endpoint))
```

**Error rate (5xx)**
```
sum(rate(http_requests_total{status=~"5.."}[1m])) / sum(rate(http_requests_total[1m])) OR vector(0)
```

**Active users**
```
active_users_count
```

**Retrieval accuracy trend**
```
retrieval_accuracy
```

**Ranking NDCG trend**
```
ranking_ndcg
```

**Training duration**
```
training_duration_seconds
```

## 6) How training metrics appear

Training metrics are pushed by the notebooks to Pushgateway:

```powershell
jupyter nbconvert --to notebook --execute train_retrieval.ipynb --ExecutePreprocessor.kernel_name=ml_env310
jupyter nbconvert --to notebook --execute train_ranking.ipynb --ExecutePreprocessor.kernel_name=ml_env310
```

After running these, refresh Grafana to see:
- `retrieval_accuracy`
- `ranking_ndcg`
- `training_duration_seconds`

## 7) Alerts we added

**Latency spike alert**
- Query: P95 latency
- Condition: `IS ABOVE 1` for `5m`
- Name: `Latency spike (P95 > 1s)`
- Folder: `Alerts`
- Evaluation group: `api-alerts`

**Retrieval accuracy drop alert**
- Query: `retrieval_accuracy`
- Condition: `IS BELOW 0.05` for `5m`
- Name: `Retrieval accuracy drop`
- Folder: `Alerts`
- Evaluation group: `api-alerts`

## 8) Export dashboard JSON

In Grafana:
Dashboard → **Export** → **Export as code**  
Save to:
```
dashboards/api_performance_metrics.json
```

## 9) Common issues

- **No data in charts**: run `python test.py` to generate API traffic.
- **No training metrics**: re-run notebooks (they push metrics to Pushgateway).
- **Port in use**: Prometheus uses `:9092`, Pushgateway uses `:9091`, API uses `:8000`.
- **Grafana can’t query Prometheus**: verify Prometheus is running and data source URL is correct.
