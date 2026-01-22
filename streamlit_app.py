import json
import os
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Two-Tower Recommender Demo",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get(path: str, payload: Dict, params: Optional[Dict] = None):
    url = f"{API_BASE_URL}{path}"
    return requests.get(url, json=payload, params=params, timeout=30)


@st.cache_data(show_spinner=False)
def load_movies() -> Optional[pd.DataFrame]:
    candidates = [
        "data/raw/100k-movies.parquet",
        "data/raw/1m-movies.parquet",
    ]
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_parquet(path)
            df["movie_id"] = df["movie_id"].astype(str)
            df["movie_release_year"] = df["movie_release_year"].fillna("").astype(str)
            return df
    return None


@st.cache_data(show_spinner=False)
def load_users() -> Optional[pd.DataFrame]:
    candidates = [
        "data/raw/100k-users.parquet",
        "data/raw/1m-users.parquet",
    ]
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_parquet(path)
            df["user_id"] = df["user_id"].astype(str)
            return df
    return None


def build_movie_payload(movie_ids: List[str], movies_df: Optional[pd.DataFrame]) -> List[Dict]:
    items: List[Dict] = []
    if movies_df is None:
        for movie_id in movie_ids:
            items.append(
                {
                    "movie_id": movie_id,
                    "movie_title": "Unknown",
                    "movie_release_year": "0000",
                }
            )
        return items

    movie_map = movies_df.set_index("movie_id").to_dict(orient="index")
    for movie_id in movie_ids:
        info = movie_map.get(movie_id, {})
        items.append(
            {
                "movie_id": movie_id,
                "movie_title": info.get("movie_title", "Unknown"),
                "movie_release_year": info.get("movie_release_year", "0000"),
            }
        )
    return items


def api_health() -> str:
    try:
        resp = requests.get(f"{API_BASE_URL}/api/healthcheck", timeout=5)
        return "OK" if resp.status_code == 200 else f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


st.title("🎬 Two-Tower Recommender — Streamlit Console")

with st.sidebar:
    st.subheader("Connection")
    st.caption("Set API_BASE_URL as env var if your API is not local.")
    st.code(API_BASE_URL, language="text")
    st.write(f"Healthcheck: **{api_health()}**")
    st.divider()

    st.subheader("User profile")
    users_df = load_users()

    if users_df is not None:
        sample_user_id = st.selectbox("Sample user_id", users_df["user_id"].head(50))
        sample = users_df[users_df["user_id"] == sample_user_id].iloc[0]
    else:
        sample = None

    user_id = st.text_input("user_id", value=str(sample["user_id"]) if sample is not None else "138")
    user_gender = st.selectbox("user_gender", options=[0, 1], index=1 if sample is None else int(sample["user_gender"]))
    user_zip_code = st.text_input("user_zip_code", value=str(sample["user_zip_code"]) if sample is not None else "53211")
    user_bucketized_age = st.number_input(
        "user_bucketized_age",
        value=float(sample["user_bucketized_age"]) if sample is not None else 45.0,
        step=1.0,
    )
    user_occupation_label = st.number_input(
        "user_occupation_label",
        value=int(sample["user_occupation_label"]) if sample is not None else 4,
        step=1,
    )

    user_payload = {
        "user_id": str(user_id),
        "user_gender": int(user_gender),
        "user_zip_code": str(user_zip_code),
        "user_bucketized_age": float(user_bucketized_age),
        "user_occupation_label": int(user_occupation_label),
    }

tab_rec, tab_metrics = st.tabs(["Recommendations", "Metrics"])

with tab_rec:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Retrieval")
        st.markdown('<div style="padding:10px">', unsafe_allow_html=True)
        top_k = st.slider("top_k", min_value=5, max_value=50, value=10, step=5)
        approximate = st.toggle("approximate (FAISS/ScaNN)", value=True)
        run_retrieval = st.button("Run Retrieval")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Ranking")
        st.markdown('<div style="padding:10px">', unsafe_allow_html=True)
        run_ranking = st.button("Run Ranking")
        st.caption("Ranking uses the retrieved items and attaches title/year if available.")
        st.markdown("</div>", unsafe_allow_html=True)

    movies_df = load_movies()

    if run_retrieval:
        try:
            resp = _get(
                "/api/v1/retrieval",
                payload=user_payload,
                params={"top_k": top_k, "approximate": approximate},
            )
            if resp.status_code != 200:
                st.error(f"Retrieval failed: HTTP {resp.status_code}")
                st.code(resp.text)
            else:
                st.session_state["retrieval_ids"] = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval error: {exc}")

    retrieval_ids = st.session_state.get("retrieval_ids", [])
    if retrieval_ids:
        st.success(f"Retrieved {len(retrieval_ids)} items.")
        st.write(retrieval_ids)

    if run_ranking:
        if not retrieval_ids:
            st.warning("Run retrieval first.")
        else:
            movies_payload = build_movie_payload(retrieval_ids, movies_df)
            try:
                resp = _get(
                    "/api/v1/ranking",
                    payload={"movies": movies_payload, "user": user_payload},
                )
                if resp.status_code != 200:
                    st.error(f"Ranking failed: HTTP {resp.status_code}")
                    st.code(resp.text)
                else:
                    scores = resp.json()
                    df = pd.DataFrame(
                        [
                            {
                                "movie_id": mid,
                                "score": score,
                                "movie_title": next(
                                    (m["movie_title"] for m in movies_payload if m["movie_id"] == mid),
                                    "Unknown",
                                ),
                                "movie_release_year": next(
                                    (m["movie_release_year"] for m in movies_payload if m["movie_id"] == mid),
                                    "0000",
                                ),
                            }
                            for mid, score in scores.items()
                        ]
                    ).sort_values("score", ascending=False)
                    st.session_state["ranking_df"] = df
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ranking error: {exc}")

    if "ranking_df" in st.session_state:
        st.subheader("Ranked results")
        st.dataframe(st.session_state["ranking_df"], use_container_width=True)

with tab_metrics:
    st.subheader("Live API Metrics")
    st.markdown('<div style="padding:10px">', unsafe_allow_html=True)
    try:
        resp = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if resp.status_code != 200:
            st.error(f"/metrics failed: HTTP {resp.status_code}")
            st.code(resp.text)
        else:
            text = resp.text.splitlines()
            metrics = {}
            for line in text:
                if line.startswith("#") or not line.strip():
                    continue
                key, value = line.split(" ", 1)
                metrics[key] = float(value)

            cols = st.columns(4)
            cols[0].metric("requests_total (retrieval)", metrics.get('recommendation_requests_total{endpoint="retrieval"}', 0))
            cols[1].metric("requests_total (ranking)", metrics.get('recommendation_requests_total{endpoint="ranking"}', 0))
            cols[2].metric("active_users", metrics.get("active_users_count", 0))
            cols[3].metric("model_load_time_sec", metrics.get("model_load_time_seconds", 0))

            st.caption("Raw metrics sample")
            st.code("\n".join(text[:60]))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Metrics error: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)