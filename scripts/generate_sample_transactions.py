"""Generate data/sample_transactions.csv for dashboard demos (small file, not full Kaggle)."""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_transactions.csv"

random.seed(42)

LOCS = [
    ("Houston", "TX", 29.76, -95.37, 2_320_268),
    ("Phoenix", "AZ", 33.45, -112.07, 1_680_992),
    ("Chicago", "IL", 41.88, -87.63, 2_693_976),
    ("Seattle", "WA", 47.61, -122.33, 737_015),
    ("Denver", "CO", 39.74, -104.99, 715_522),
    ("Miami", "FL", 25.76, -80.19, 454_279),
    ("Portland", "OR", 45.52, -122.68, 652_503),
    ("Austin", "TX", 30.27, -97.74, 978_908),
    ("Boston", "MA", 42.36, -71.06, 675_647),
    ("San Diego", "CA", 32.72, -117.16, 1_386_932),
]
LEGIT_CATS = [
    "gas_transport",
    "grocery_pos",
    "food_dining",
    "pharmacy",
    "personal_care",
    "health_fitness",
]
FRAUD_CATS = ["shopping_net", "misc_net", "shopping_pos", "misc_pos"]
LEGIT_MERCHANTS = [
    "corner-store-24",
    "metro-pharmacy",
    "daily-grocer",
    "fuel-stop-77",
    "lunch-spot-downtown",
]
FRAUD_MERCHANTS = [
    "fraud-store-net",
    "flash-deal-999",
    "crypto-gift-cards",
    "ghost-checkout-co",
]
LEGIT_JOBS = ["Nurse", "Teacher", "Engineer", "Accountant", "Developer", "Designer", "Manager"]
FRAUD_JOBS = ["Engineer", "Consultant", "Self-employed", "Manager"]


def jitter_coord(lat: float, lon: float, deg: float = 0.05) -> tuple[float, float]:
    return round(lat + random.uniform(-deg, deg), 4), round(lon + random.uniform(-deg, deg), 4)


def main() -> None:
    rows: list[dict] = []
    start = datetime(2019, 1, 1, 8, 0, 0)

    for i in range(30):
        city, stc, lat, lon, pop = LOCS[i % len(LOCS)]
        mlat, mlon = jitter_coord(lat, lon, deg=0.05)
        amt = round(random.uniform(3.5, 89.0), 2)
        hour = random.choice(list(range(7, 22)))
        ts = start + timedelta(days=i * 2, hours=hour, minutes=random.randint(0, 59))
        rows.append(
            {
                "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "amt": amt,
                "category": random.choice(LEGIT_CATS),
                "merchant": random.choice(LEGIT_MERCHANTS),
                "lat": lat,
                "long": lon,
                "merch_lat": mlat,
                "merch_long": mlon,
                "city": city,
                "state": stc,
                "city_pop": pop,
                "dob": f"{1970 + random.randint(0, 35):04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "gender": random.choice(["M", "F"]),
                "job": random.choice(LEGIT_JOBS),
                "is_fraud": 0,
            }
        )

    for j in range(20):
        city, stc, lat, lon, pop = LOCS[(j + 3) % len(LOCS)]
        mlat, mlon = jitter_coord(lat, lon, deg=2.0)
        amt = round(random.uniform(480.0, 9800.0), 2)
        hour = random.choice(list(range(22, 24)) + list(range(0, 4)))
        ts = start + timedelta(days=j * 3 + 1, hours=hour, minutes=random.randint(0, 59))
        rows.append(
            {
                "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "amt": amt,
                "category": random.choice(FRAUD_CATS),
                "merchant": random.choice(FRAUD_MERCHANTS),
                "lat": lat,
                "long": lon,
                "merch_lat": mlat,
                "merch_long": mlon,
                "city": city,
                "state": stc,
                "city_pop": pop,
                "dob": f"{1985 + random.randint(0, 15):04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "gender": random.choice(["M", "F"]),
                "job": random.choice(FRAUD_JOBS),
                "is_fraud": 1,
            }
        )

    random.shuffle(rows)

    fieldnames = [
        "trans_date_trans_time",
        "amt",
        "category",
        "merchant",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "city",
        "state",
        "city_pop",
        "dob",
        "gender",
        "job",
        "is_fraud",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_fraud = sum(int(r["is_fraud"]) for r in rows)
    print(f"Wrote {OUT} ({len(rows)} rows, is_fraud=1 count: {n_fraud})")


if __name__ == "__main__":
    main()
