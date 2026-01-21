import argparse
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


def _load_users(path: str) -> pd.DataFrame:
    users = pd.read_parquet(path)
    users = users.set_index("user_id", drop=False)
    return users


def _load_ratings(path: str, sample_size: int, seed: int) -> pd.DataFrame:
    ratings = pd.read_parquet(path)
    if sample_size > 0 and sample_size < len(ratings):
        ratings = ratings.sample(n=sample_size, random_state=seed)
    return ratings


def _load_movies(path: str) -> pd.DataFrame:
    movies = pd.read_parquet(path)
    movies = movies.set_index("movie_id", drop=False)
    return movies


def _to_user_tensors(user: Dict[str, Any]) -> Dict[str, tf.Tensor]:
    return {k: tf.convert_to_tensor([v]) for k, v in user.items()}

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def _jsonify_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_jsonable(v) for k, v in data.items()}

def _filter_user_fields(user: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "user_id",
        "user_gender",
        "user_zip_code",
        "user_bucketized_age",
        "user_occupation_label",
    }
    filtered = {k: v for k, v in user.items() if k in allowed}
    if "user_id" in filtered:
        filtered["user_id"] = str(filtered["user_id"])
    if "user_zip_code" in filtered:
        filtered["user_zip_code"] = str(filtered["user_zip_code"])
    if "user_gender" in filtered:
        filtered["user_gender"] = int(filtered["user_gender"])
    if "user_bucketized_age" in filtered:
        filtered["user_bucketized_age"] = float(filtered["user_bucketized_age"])
    if "user_occupation_label" in filtered:
        filtered["user_occupation_label"] = int(filtered["user_occupation_label"])
    return filtered

def _filter_movie_fields(movie: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "movie_id",
        "movie_title",
        "movie_release_year",
    }
    filtered = {k: v for k, v in movie.items() if k in allowed}
    if "movie_id" in filtered:
        filtered["movie_id"] = str(filtered["movie_id"])
    if "movie_title" in filtered:
        filtered["movie_title"] = str(filtered["movie_title"])
    if "movie_release_year" in filtered:
        filtered["movie_release_year"] = str(filtered["movie_release_year"])
    return filtered


def _retrieval_hit_rate(
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    brute_model_path: str,
    top_k: int,
) -> float:
    model = tf.saved_model.load(brute_model_path)
    hits = 0
    total = 0

    for _, row in ratings.iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        if user_id not in users.index:
            continue

        user = _filter_user_fields(_jsonify_dict(users.loc[user_id].to_dict()))
        user_tensors = _to_user_tensors(user)
        result = model.signatures["call"](**user_tensors, k=top_k)
        retrieved = result["output_0"].numpy().tolist()[0]
        if str(movie_id) in [str(x) for x in retrieved]:
            hits += 1
        total += 1

    return hits / total if total else 0.0


def _percentiles(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def _api_latency(
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    api_base: str,
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    retrieval_times = []
    ranking_times = []

    for _, row in ratings.iterrows():
        user_id = row["user_id"]
        if user_id not in users.index:
            continue

        user = _jsonify_dict(users.loc[user_id].to_dict())

        start = time.perf_counter()
        response = requests.get(
            url=f"{api_base}/api/v1/retrieval",
            params={"approximate": False, "top_k": top_k},
            json=user,
            timeout=30,
        )
        response.raise_for_status()
        retrieval_times.append(time.perf_counter() - start)

        movie_ids = response.json()
        movie_list = []
        for movie_id in movie_ids:
            movie_id = str(movie_id)
            if movie_id in movies.index:
                movie_list.append(_filter_movie_fields(_jsonify_dict(movies.loc[movie_id].to_dict())))

        start = time.perf_counter()
        response = requests.get(
            url=f"{api_base}/api/v1/ranking",
            json={"movies": movie_list, "user": user},
            timeout=30,
        )
        response.raise_for_status()
        ranking_times.append(time.perf_counter() - start)

    r50, r95, r99 = _percentiles(retrieval_times)
    k50, k95, k99 = _percentiles(ranking_times)

    return {
        "retrieval": {"p50": r50, "p95": r95, "p99": r99},
        "ranking": {"p50": k50, "p95": k95, "p99": k99},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline metrics.")
    parser.add_argument("--dataset", default="100k", choices=["100k", "1m"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API latency measurement.",
    )
    args = parser.parse_args()

    users = _load_users(f"data/raw/{args.dataset}-users.parquet")
    ratings = _load_ratings(f"data/raw/{args.dataset}-ratings.parquet", args.sample_size, args.seed)
    movies = _load_movies(f"data/raw/{args.dataset}-movies.parquet")

    hit_rate = _retrieval_hit_rate(
        users=users,
        ratings=ratings,
        brute_model_path="checkpoints/retrieval/brute",
        top_k=args.top_k,
    )

    print(f"retrieval_hit_rate@{args.top_k}: {hit_rate:.4f}")

    if not args.skip_api:
        latency = _api_latency(
            users=users,
            ratings=ratings,
            movies=movies,
            api_base=args.api_base,
            top_k=args.top_k,
        )
        print("api_latency_seconds:", latency)


if __name__ == "__main__":
    main()
