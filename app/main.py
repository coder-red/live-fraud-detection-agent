from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv() # Load env vars early for tracing and config

from app.routes import router
from app.db.connections import init_db 

app = FastAPI(
    title="Live Fraud Detection API",
    description="Real-time XGBoost inference for transaction fraud detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "null"],
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
def startup():
    init_db()