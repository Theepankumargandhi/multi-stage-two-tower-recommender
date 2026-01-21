import os
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values


def get_db_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def insert_predictions(
    user_id: str,
    model_version: str,
    items: list[tuple[str, float]],
    conn: Optional[psycopg2.extensions.connection] = None,
) -> None:
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO predictions (user_id, item_id, model_version, score)
            VALUES %s
            """,
            [(user_id, item_id, model_version, score) for item_id, score in items],
        )
    conn.commit()

    if close_conn:
        conn.close()
