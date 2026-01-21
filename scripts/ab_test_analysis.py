import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import psycopg2
from dotenv import load_dotenv


def load_predictions() -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    query = """
        SELECT user_id, item_id, model_version, score, created_at
        FROM predictions
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def ndcg(scores: list[float]) -> float:
    # Use scores as relevance proxy; sorted by score in logged order.
    if not scores:
        return 0.0
    gains = np.array(scores, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains * discounts)
    ideal = np.sort(gains)[::-1]
    idcg = np.sum(ideal * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def main() -> None:
    load_dotenv()
    df = load_predictions()
    if df.empty:
        print("No predictions found.")
        return

    # Compute per-user NDCG per model version
    ndcg_by_version = defaultdict(list)
    for (model_version, user_id), group in df.groupby(["model_version", "user_id"]):
        scores = group["score"].tolist()
        ndcg_by_version[model_version].append(ndcg(scores))

    for version, values in ndcg_by_version.items():
        print(f"{version}: mean NDCG = {np.mean(values):.4f} (n={len(values)})")

    if "A" in ndcg_by_version and "B" in ndcg_by_version:
        tstat, pval = ttest_ind(ndcg_by_version["A"], ndcg_by_version["B"], equal_var=False)
        print(f"T-test A vs B: t={tstat:.4f}, p={pval:.6f}")
    else:
        print("Need both A and B samples to run significance test.")


if __name__ == "__main__":
    main()
