import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv() # Load env vars early for tracing and config

from app.routes import router
from app.db.connections import init_db, ping_db

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Live Fraud Detection API",
    description="Real-time XGBoost inference for transaction fraud detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "null",
        "https://live-fraud-detection-agent-1.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes we defined
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def health_check(): #async is used for allowing a task to run in the background while waiting for a response, improving performance
    return {"status": "online", "model": "XGBoost Fraud Agent"}

# Initialize the database when the app starts
@app.on_event("startup")
async def startup():
    # Kick off the DB maintenance loop in the background instead of blocking
    # startup: the app stays online even if the database is unreachable, and
    # the loop keeps retrying until the DB comes back.
    app.state.db_task = asyncio.create_task(db_maintenance_loop())


DB_RETRY_DELAY_SECONDS = 30
DB_KEEPALIVE_SECONDS = 240  # < 7 days of Supabase free-tier idle pause; ping every 4 min


async def db_maintenance_loop() -> None:
    """
    Keeps the app healthy and the database awake.

    - If init_db() fails (DB down/paused), log a warning and retry instead of
      crashing the app.
    - Once connected, run SELECT 1 every few minutes so Supabase's free tier
      never pauses the project for inactivity.
    """
    initialized = False
    while True:
        try:
            if not initialized:
                await asyncio.to_thread(init_db)
                initialized = True
                logger.info("Database initialized successfully")
            elif not await asyncio.to_thread(ping_db):
                initialized = False
                logger.warning("Database keep-alive ping failed; will re-initialize on retry")
        except Exception as exc:
            initialized = False
            logger.warning("Database unavailable (%s); retrying in %ss", exc, DB_RETRY_DELAY_SECONDS)

        await asyncio.sleep(DB_KEEPALIVE_SECONDS if initialized else DB_RETRY_DELAY_SECONDS)