import os
from collections.abc import Generator # Generator is used for type hinting the get_db function, which yields a database session

from dotenv import load_dotenv
from sqlalchemy import create_engine # create_engine is used to create a SQLAlchemy engine that will manage our database connections
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
# DeclarativeBase is the base class for our ORM models
# A session is a temporary conversation between your Python code and the db — it holds your changes in memory until you tell it to commit, then sends them all to the database at once.(like git)
# sessionmaker is a factory for creating new Session instances
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(
    DATABASE_URL,
    future=True, # future=True enables SQLAlchemy 2.0 style usage, which is more explicit and has better performance
    pool_pre_ping=True, # pool_pre_ping=True ensures that the connection is alive before using it, which helps prevent errors due to stale connections
)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)

# create a base class for our ORM models to inherit from
# # Base is the shared registry for our ORM models
class Base(DeclarativeBase):
    pass

# Create the database tables based on the models defined in app.db.models 
def init_db() -> None:
    # Import models here so SQLAlchemy registers them before create_all runs.
    from app.db import models
    # create_all will create the tables in the database based on the models defined in app.db.models. 
    # It checks if the tables already exist and only creates them if they don't, so it's safe to run multiple times without losing data.
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
