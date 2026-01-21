from typing import List
import threading
import time
from pydantic import BaseModel
from fastapi import FastAPI, Request, status
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# Third-party
from infer import retrieve, rank, choose_model_version
from db import insert_predictions


APP = FastAPI()
Instrumentator().instrument(APP).expose(APP)

# Custom metrics
RECOMMENDATION_REQUESTS = Counter(
    "recommendation_requests_total",
    "Total recommendation requests.",
    ["endpoint"],
)
RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Latency for recommendation endpoints.",
    ["endpoint"],
)
ACTIVE_USERS = Gauge(
    "active_users_count",
    "Count of unique users seen since process start.",
)
_active_users = set()
_active_users_lock = threading.Lock()


def _record_active_user(user_id: str) -> None:
    with _active_users_lock:
        _active_users.add(user_id)
        ACTIVE_USERS.set(len(_active_users))


class UserModel(BaseModel):
    user_id: str
    user_gender: int
    user_zip_code: str
    user_bucketized_age: float
    user_occupation_label: int

class MovieModel(BaseModel):
    movie_id: str
    movie_title: str
    movie_release_year: str


@APP.get(
    path = "/api/healthcheck",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'healthcheck'],
)
async def healthcheck(request: Request):
    return "OK"


@APP.get(
    path = "/api/v1/retrieval",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'v1', 'retrieval'],
)
async def api_v1_retrieval(
    user: UserModel,
    top_k: int = 10,
    approximate: bool = True
):
    start = time.perf_counter()
    data = user.model_dump()
    _record_active_user(data["user_id"])
    RECOMMENDATION_REQUESTS.labels(endpoint="retrieval").inc()
    model_version = choose_model_version(data["user_id"])
    try:
        return retrieve(
            user = data,
            k = top_k,
            approximate = approximate,
        )
    finally:
        RECOMMENDATION_LATENCY.labels(endpoint="retrieval").observe(
            time.perf_counter() - start
        )


@APP.get(
    path = "/api/v1/ranking",
    status_code = status.HTTP_200_OK,
    tags = ['api', 'v1', 'ranking'],
)
async def api_v1_rank(movies: List[MovieModel], user: UserModel):

    user_dict = user.model_dump()
    _record_active_user(user_dict["user_id"])
    start = time.perf_counter()
    RECOMMENDATION_REQUESTS.labels(endpoint="ranking").inc()

    movie_scores = {}
    model_version = choose_model_version(user_dict["user_id"])
    for movie in movies:
        movie_dict = movie.model_dump()
        movie_id = movie_dict.get('movie_id')

        score = rank(user_dict, movie_dict)
        
        movie_scores[movie_id] = score

    insert_predictions(
        user_id=user_dict["user_id"],
        model_version=model_version,
        items=[(movie_id, score) for movie_id, score in movie_scores.items()],
    )

    RECOMMENDATION_LATENCY.labels(endpoint="ranking").observe(
        time.perf_counter() - start
    )
    return movie_scores
