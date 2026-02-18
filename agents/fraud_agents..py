import pandas as pd
import requests
import time
from config import DATA_DIR  

class LiveFraudAgent:
    def __init__(self, api_url="http://127.0.0.1:8000/api/v1/predict"):
        self.api_url = api_url
        self.stats = {"correct": 0, "missed": 0, "false_alarms": 0}

    def run_simulation(self, csv_path, num_rows=20):
        # Load the test data
        df = pd.read_csv(csv_path, nrows=num_rows)
        
        print(f"🚀 Starting Fraud Agent Simulation on {num_rows} transactions...")
        
        for _, row in df.iterrows():
            # 1. Prepare the data (convert row to dict)
            tx_payload = row.to_dict()
            actual_label = tx_payload.pop('is_fraud') # Remove the answer so the model can't "cheat"
            
            # 2. Ask the API for a prediction
            try:
                response = requests.post(self.api_url, json=tx_payload)
                prediction = response.json()
                
                is_fraud_pred = prediction['is_fraud']
                prob = prediction['probability']
                
                # 3. Logic: Compare prediction vs actual truth from CSV
                self._log_performance(is_fraud_pred, actual_label, tx_payload['amt'], prob)
                
                # Wait a split second to simulate "real time"
                time.sleep(0.5) 
                
            except Exception as e:
                print(f" Connection Error: {e}")

    def _log_performance(self, pred, actual, amt, prob):
        if pred and actual:
            print(f"🔥 [CRITICAL] CAUGHT FRAUD! Amt: ${amt} | Conf: {prob}")
            self.stats["correct"] += 1
        elif pred and not actual:
            print(f"⚠️ [FALSE ALARM] Flagged ${amt} but it was legitimate.")
            self.stats["false_alarms"] += 1
        elif not pred and actual:
            print(f"💀 [MISSED] Fraud of ${amt} slipped through!")
            self.stats["missed"] += 1
        else:
            print(f"✅ Approved legitimate transaction of ${amt}")

if __name__ == "__main__":
    # Point to your fraud_test.csv
    test_file = DATA_DIR / "fraud_test.csv"
    
    agent = LiveFraudAgent()
    agent.run_simulation(test_file, num_rows=50)