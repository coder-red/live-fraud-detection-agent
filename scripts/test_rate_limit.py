import requests

# Base URL for the local API
BASE_URL = "http://localhost:8000/api/v1"

# Sample transaction data
SAMPLE_TX = {
    "trans_date_trans_time": "2023-01-01 12:00:00",
    "amt": 50.0,
    "category": "entertainment",
    "merchant": "fraud_test_merchant",
    "lat": 40.7128,
    "long": -74.0060,
    "merch_lat": 40.7128,
    "merch_long": -74.0060,
    "city": "New York",
    "state": "NY",
    "city_pop": 8000000,
    "dob": "1990-01-01",
    "gender": "M",
    "job": "engineer",
}


def run_rate_limit_smoke_test():
    """
    Manual smoke test for live rate limiting against a running local server.
    This is intentionally not a pytest test because it depends on external services.
    """
    print("Starting Rate Limit Smoke Test...")
    print("Sending 12 requests in rapid succession (limit is 10 per 60 seconds)...")

    for i in range(1, 13):
        try:
            response = requests.post(f"{BASE_URL}/predict", json=SAMPLE_TX)
            status = response.status_code

            if status == 200:
                print(f"Request {i}: Success (200)")
            elif status == 429:
                print(f"Request {i}: Rate Limited (429) - SUCCESS")
                print(f"Detail: {response.json()['detail']}")
                return
            else:
                print(f"Request {i}: Unexpected status {status}")
                print(response.text)
        except Exception as e:
            print(f"Request {i}: Failed to connect. Is the server running? Error: {e}")
            break


if __name__ == "__main__":
    run_rate_limit_smoke_test()
