from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# from config import SRC_PATH,BASE_DIR
from src.inference import FraudInference


router = APIRouter()

# Define the data we expect from the user
class Transaction(BaseModel):
    trans_date_trans_time: str
    amt: float
    category: str
    merchant: str
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city: str
    state: str
    city_pop: int
    dob: str
    gender: str
    job: str

# Initialize the model once when the server starts
base_path = Path(__file__).parent.parent / "model"
engine = FraudInference(
    model_path=str(base_path / "fraud_model.json"),
    feature_list_path=str(base_path / "feature_list.pkl")
)

@router.post("/predict")
async def predict_fraud(data: Transaction):
    try:
        # Convert Pydantic model to a standard dict for our engine
        transaction_dict = data.model_dump()
        result = engine.predict(transaction_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))