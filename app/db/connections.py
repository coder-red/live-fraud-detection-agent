import os
from collections.abc import Generator

from dotenv import load_dotenv
from redis import asyncio as redis_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
# a session is the main way to interact with the database. It manages connections and transactions.

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

def _build_connect_args(database_url: str) -> dict:
    # Local SQLite uses a special thread-safety flag.
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}

    connect_args: dict[str, object] = {}

    sslmode = os.getenv("DATABASE_SSLMODE")
    if sslmode:
        connect_args["sslmode"] = sslmode

    connect_timeout = os.getenv("DATABASE_CONNECT_TIMEOUT")
    if connect_timeout:
        connect_args["connect_timeout"] = int(connect_timeout)

    return connect_args


connect_args = _build_connect_args(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Redis client configuration for caching, deduplication, and velocity tracking.
# We initialize the client once at the module level.
try:
    # decode_responses=True ensures we get strings back instead of bytes from Redis.
    redis_client = redis_asyncio.from_url(REDIS_URL, decode_responses=True)
except Exception:
    # If Redis is unavailable, we set the client to None to allow the app to run without it.
    redis_client = None

# SessionLocal creates one DB session per request or task.
# A session is the object you use to read, add, update, and commit rows like git
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)

# base is the parent class for all our models. It contains the metadata and other info about the database schema.
class Base(DeclarativeBase):
    pass


def init_db() -> None:
    # Import models so SQLAlchemy knows which tables exist.
    from app.db import models  # noqa: F401

    # create_all creates missing tables; it does not delete existing data.
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db # yield is for session mgmt - it allows the function to return a session and then resume to close it after the request is done.
    finally:
        db.close()


def get_redis() -> redis_asyncio.Redis | None:
    """
    Dependency for obtaining the Redis client.
    Used for fast-path deduplication and future stateful features (rate limiting, velocity).
    Returns None if Redis was not configured successfully at startup.
    """
    return redis_client
