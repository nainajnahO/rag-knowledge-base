from collections.abc import Iterator

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app.settings import settings

pool: ConnectionPool | None = None


def open_pool() -> None:
    global pool
    pool = ConnectionPool(settings.database_url, min_size=1, max_size=10, open=True)


def close_pool() -> None:
    global pool
    if pool is not None:
        pool.close()
        pool = None


def get_conn() -> Iterator[Connection]:
    assert pool is not None, "connection pool not initialized; check FastAPI lifespan"
    with pool.connection() as conn:
        yield conn
