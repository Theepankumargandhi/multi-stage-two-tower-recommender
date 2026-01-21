from typing import Dict, Any, Tuple
import os
import time

import tensorflow as tf
from prometheus_client import Gauge

from config import (
    SCANN_PATH,
    BRUTE_PATH,
    RANKING_PATH,
)

try:
    import scann  # noqa: F401
    _has_scann = True
except Exception:
    _has_scann = False

def _has_saved_model(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "saved_model.pb")) or \
        os.path.isfile(os.path.join(path, "saved_model.pbtxt"))

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time spent loading models at startup.",
)

_load_start = time.perf_counter()
scann_retrieval = tf.saved_model.load(SCANN_PATH) if _has_scann and _has_saved_model(SCANN_PATH) else None
brute_retrieval = tf.saved_model.load(BRUTE_PATH) if _has_saved_model(BRUTE_PATH) else None
ranking = tf.saved_model.load(RANKING_PATH) if _has_saved_model(RANKING_PATH) else None
MODEL_LOAD_TIME.set(time.perf_counter() - _load_start)


def retrieve(
    user: Dict[str, Any],
    k: int,
    approximate: bool = True
) -> list:
    """
        Perform retrieval for a given user.

        Parameters:
            - user (Dict[str, Any]): A dictionary containing the user's features.
            - k (int): The number of items to retrieve.
            - approximate (bool): Whether to use an approximate nearest neighbors 
                search or an exact search. Defaults to `True`.

        Returns:
            - identifiers (list): A list of item identifiers.
    """
    user_tensors = {k: tf.convert_to_tensor([v]) for k, v in user.items()}

    if approximate and scann_retrieval is not None:
        _ = scann_retrieval.signatures['call'](**user_tensors, k=k)  # Approximate
    else:
        if brute_retrieval is None:
            raise RuntimeError("Brute-force retrieval model is not available.")
        _ = brute_retrieval.signatures['call'](**user_tensors, k=k)  # Exact

    identifiers = _['output_0'].numpy().tolist()
    affnities   = _['output_1'].numpy().tolist()

    return identifiers


def rank(
    user: Dict[str, Any],
    movie: Dict[str, Any],
) -> float:
    """
        Perform ranking for a given user and movie.

        Parameters:
            - user (Dict[str, Any]): A dictionary containing the user's features.
            - movie (Dict[str, Any]): A dictionary containing the movie's features.

        Returns:
            - score (float): A score representing how well the movie matches the user's preferences.
    """
    user_tensors  = {k: tf.convert_to_tensor([v]) for k, v in user.items()}
    movie_tensors = {k: tf.convert_to_tensor([v]) for k, v in movie.items()}

    if ranking is None:
        raise RuntimeError("Ranking model is not available.")
    _ = ranking.signatures['call'](**user_tensors, **movie_tensors)
    score = _['output_0'].numpy()[0][0].tolist()

    return score


def choose_model_version(user_id: str) -> str:
    # Deterministic 90/10 split based on user_id hash
    bucket = hash(user_id) % 100
    return "A" if bucket < 90 else "B"
