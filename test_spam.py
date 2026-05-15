import requests
import json

url = "http://localhost:8000/api/v1/predict"
payload = {
    "trans_date_trans_time": "2024-01-01 12:00:00",
    "amt": 100.0,
    "category": "shopping_net",
    "merchant": "SomeStore",
    "lat": 40.0,
    "long": -74.0,
    "merch_lat": 40.0,
    "merch_long": -74.0,
    "city": "New York",
    "state": "NY",
    "city_pop": 8000000,
    "dob": "1990-01-01",
    "gender": "M",
    "job": "engineer"
}

for i in range(15):
    r = requests.post(url, json=payload)
    print(f"Request {i+1}: {r.status_code}")
    if r.status_code == 429:
        print("Rate limit reached!")
        break
