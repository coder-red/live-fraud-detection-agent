import os
from collections.abc import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
# a session is the main way to interact with the database. It manages connections and transactions.

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

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
    # Yield one session, then close it after the request finishes.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
