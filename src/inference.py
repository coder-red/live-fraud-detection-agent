import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from src.features import preprocess_features

class FraudInference:
    def __init__(self, model_path: str, feature_list_path: str, threshold: float = 0.5221):
        # 1. Initialize the wrapper
        self.model = xgb.XGBClassifier()
        
        # 2. Load from the JSON file (much more stable than .pkl)
        self.model.load_model(model_path)
        
        self.threshold = threshold
        
        # 3. Load feature names (this stays as joblib because it's just a list)
        self.feature_names = joblib.load(feature_list_path)

    def predict(self, raw_data: dict):
        # 1. Convert dict to DataFrame
        df = pd.DataFrame([raw_data])
        
        # 2. Preprocess (Calculates distance, hour, etc.)
        X = preprocess_features(df)
        
        # 3. Ensure columns are in the exact order the model expects
        X = X[self.feature_names]
        
        # 4. Get probability
        # [0, 1] gets the probability of class 1 (Fraud)
        prob = float(self.model.predict_proba(X)[0, 1])
        
        return {
            "is_fraud": bool(prob >= self.threshold),
            "probability": round(prob, 4),
            "threshold": self.threshold,
            "features": self.feature_names
        }

if __name__ == "__main__":
    # Correct paths based on tree
    base_path = Path(__file__).parent.parent / "model"
    
    # Use the .json file for our model
    m_path = str(base_path / "fraud_model.json")
    f_path = str(base_path / "feature_list.pkl")

    engine = FraudInference(model_path=m_path, feature_list_path=f_path)

    sample_tx = {
        "trans_date_trans_time": "2019-01-01 03:15:00",
        "amt": 950.00,
        "category": "shopping_net",
        "merchant": "fraud-merchant-xyz",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 34.0522,
        "merch_long": -118.2437,
        "city": "New York",
        "state": "NY",
        "city_pop": 8336817,
        "dob": "1990-01-01",
        "gender": "M",
        "job": "Engineer"
    }

    result = engine.predict(sample_tx)
    print(f"\nFraud Detected: {result['is_fraud']} ({result['probability']})")