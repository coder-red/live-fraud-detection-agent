from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Live Fraud Detection API",
    description="Real-time XGBoost inference for transaction fraud detection",
    version="1.0.0"
)

# Include the routes we defined
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def health_check(): #async is used for allowing a task to run in the background while waiting for a response, improving performance
    return {"status": "online", "model": "XGBoost Fraud Agent"}