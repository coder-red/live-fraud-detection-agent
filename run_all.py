import subprocess
import time
import requests
import pandas as pd
import sys
import random
from pathlib import Path
from agents.fraud_agents import HITLFraudAgent

def start_api():
    """Launch the FastAPI server in the background."""
    print(" Starting FastAPI Server...")
    # Change DEVNULL to sys.stdout to see the server's internal logs
    process = subprocess.Popen(
        # subprocess.Popen tell python to run a terminal in the background and run the agent in it
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, # Suppress API logs for cleaner output
        stderr=subprocess.STDOUT # Redirect errors to the same place as stdout (which is currently DEVNULL)
    )
    return process

def wait_for_api(url: str, timeout: int = 30):
    """Wait for the API to be ready before starting the Agent."""
    print(" Waiting for API to wake up...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url.replace("/predict", "/"))
            if response.status_code == 200:
                print(" API is online!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def run_simulation(total_samples: int = 500):
    """Run a randomized simulation and track performance stats."""
    api_url = "http://127.0.0.1:8000/predict"
    agent = HITLFraudAgent(api_url=api_url)
    data_path = Path("data/fraudTest.csv")
    
    if not data_path.exists():
        print(f" Error: {data_path} not found!")
        return

    # Statistics Tracker
    stats = {"total": 0, "flagged": 0, "human": 0, "approved": 0}

    # Random Sampling: Skip a random amount to get fresh data each time
    random_skip = random.randint(1, 556000) 
    print(f" Sampling {total_samples} random transactions (Starting row: {random_skip})...")
    
    # Read the random chunk
    df = pd.read_csv(data_path, skiprows=range(1, random_skip), nrows=total_samples)
    
    print("-" * 50)

    for idx, row in df.iterrows():
        tx = row.to_dict()
        result = agent.run_on_transaction(tx)
        stats["total"] += 1
        
        action = result.get('action')
        
        # Log the outcomes
        if action == "BLOCK":
            stats["flagged"] += 1
            print(f" [Row {idx}] AI BLOCKED: {result.get('reasoning')[:60]}...")
        elif "human_verdict" in result:
            stats["human"] += 1
            print(f" [Row {idx}] HUMAN REVIEW: {result.get('human_verdict')}")
        else:
            stats["approved"] += 1
            # Quietly print a dot for every 10 approvals to show progress
            if stats["approved"] % 10 == 0: print(".", end="", flush=True)

    # Summarize the results in a clear format
    print("\n" + "="*40)
    print(" FINAL INVESTIGATION SUMMARY")
    print("="*40)
    print(f" Total Scanned:      {stats['total']}")
    print(f" AI Auto-Blocked:    {stats['flagged']}")
    print(f" Human Reviews:     {stats['human']}")
    print(f" Clean Transactions: {stats['approved']}")
    
    hit_rate = ((stats['flagged'] + stats['human']) / stats['total']) * 100
    print(f" Fraud Hit Rate:     {hit_rate:.2f}%")
    print("="*40)

if __name__ == "__main__":
    api_process = None
    try:
        api_process = start_api()
        if wait_for_api("http://127.0.0.1:8000/predict"):
            # Change this number to 500 for your final run
            run_simulation(total_samples=30) 
        else:
            print(" API failed to start.")
            
    except KeyboardInterrupt:
        print("\n Stopping...")
    finally:
        if api_process:
            print(" Killing API server...")
            api_process.terminate()