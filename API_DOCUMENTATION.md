# API Documentation

Base URL: `http://127.0.0.1:8000`

## Healthcheck
**GET** `/api/healthcheck`

Response:
```
OK
```

## Retrieval
**GET** `/api/v1/retrieval`

Query params:
- `top_k` (int, default 10)
- `approximate` (bool, default true)

Body (JSON):
```
{
  "user_id": "138",
  "user_gender": 1,
  "user_zip_code": "53211",
  "user_bucketized_age": 45.0,
  "user_occupation_label": 4
}
```

Response (list of movie ids):
```
["83", "187", "194", "204", "427", "211", "732", "202", "435", "317"]
```

Python example:
```
import requests
payload = {
  "user_id": "138",
  "user_gender": 1,
  "user_zip_code": "53211",
  "user_bucketized_age": 45.0,
  "user_occupation_label": 4,
}
resp = requests.get(
    "http://127.0.0.1:8000/api/v1/retrieval",
    params={"top_k": 10, "approximate": True},
    json=payload,
)
print(resp.json())
```

## Ranking
**GET** `/api/v1/ranking`

Body (JSON):
```
{
  "movies": [
    {"movie_id": "5", "movie_title": "Copycat", "movie_release_year": "1995"}
  ],
  "user": {
    "user_id": "138",
    "user_gender": 1,
    "user_zip_code": "53211",
    "user_bucketized_age": 45.0,
    "user_occupation_label": 4
  }
}
```

Response (movie_id -> score):
```
{"5": 3.80}
```

## Metrics
**GET** `/metrics`

Prometheus metrics for API requests and latency.

## Notes
- Retrieval uses FAISS (IndexIVFFlat) when `approximate=true` and FAISS artifacts exist.
- If FAISS is unavailable, ScaNN is used when installed; otherwise brute retrieval is used.
- Ranking logs predictions to PostgreSQL for A/B testing.