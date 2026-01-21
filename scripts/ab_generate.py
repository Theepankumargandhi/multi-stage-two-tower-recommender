import argparse
import random

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A/B traffic.")
    parser.add_argument("--dataset", default="100k", choices=["100k", "1m"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    users = pd.read_parquet(f"data/raw/{args.dataset}-users.parquet")
    users = users.sample(n=min(args.samples, len(users)), random_state=42)

    for _, row in users.iterrows():
        user = row.to_dict()
        # Retrieval
        r = requests.get(
            url=f"{args.api_base}/api/v1/retrieval",
            params={"approximate": False, "top_k": 10},
            json=user,
            timeout=30,
        )
        r.raise_for_status()
        movie_ids = r.json()

        # Ranking
        movies = []
        for movie_id in movie_ids:
            movies.append(
                {
                    "movie_id": str(movie_id),
                    "movie_title": "",
                    "movie_release_year": "",
                }
            )
        rr = requests.get(
            url=f"{args.api_base}/api/v1/ranking",
            json={"movies": movies, "user": user},
            timeout=30,
        )
        rr.raise_for_status()

    print(f"Sent {len(users)} users through retrieval + ranking.")


if __name__ == "__main__":
    main()
