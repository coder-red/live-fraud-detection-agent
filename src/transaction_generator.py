"""Generate fresh random transactions for simulation demos.

Unlike the static CSV generator, this module produces unique transactions
on every call - no fixed seed, no file I/O. This ensures each simulation
run creates new predictions and review cases.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Literal

# ── Geographic templates ─────────────────────────────────────────────────────

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

# ── Legit (low-risk) templates ───────────────────────────────────────────────

LEGIT_CATS = [
    "gas_transport",
    "grocery_pos",
    "food_dining",
    "pharmacy",
    "personal_care",
    "health_fitness",
]
LEGIT_MERCHANTS = [
    "corner-store-24",
    "metro-pharmacy",
    "daily-grocer",
    "fuel-stop-77",
    "lunch-spot-downtown",
    "main-street-bakery",
    "city-supermarket",
    "quick-lube-oil",
]
LEGIT_JOBS = [
    "Nurse", "Teacher", "Engineer", "Accountant",
    "Developer", "Designer", "Manager", "Analyst",
]

# ── Fraud (high-risk) templates ──────────────────────────────────────────────

FRAUD_CATS = [
    "shopping_net",
    "misc_net",
    "shopping_pos",
    "misc_pos",
]
FRAUD_MERCHANTS = [
    "fraud-store-net",
    "flash-deal-999",
    "crypto-gift-cards",
    "ghost-checkout-co",
    "darkweb-deals",
    "suspicious-electronics",
    "fake-luxury-goods",
]
FRAUD_JOBS = [
    "Engineer", "Consultant", "Self-employed",
    "Manager", "Contractor", "Freelancer",
]


def _jitter_coord(
    lat: float, lon: float, deg: float = 0.05
) -> tuple[float, float]:
    """Add small random offset to coordinates."""
    return (
        round(lat + random.uniform(-deg, deg), 4),
        round(lon + random.uniform(-deg, deg), 4),
    )


def _generate_legit_transaction(base_date: datetime) -> dict:
    """Generate a single legitimate (low-risk) transaction."""
    city, state, lat, lon, pop = random.choice(LOCS)
    mlat, mlon = _jitter_coord(lat, lon, deg=0.05)

    # Legit: normal business hours, modest amounts
    hour = random.choice(list(range(7, 22)))
    amt = round(random.uniform(3.5, 150.0), 2)

    ts = base_date + timedelta(
        days=random.randint(0, 60),
        hours=hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    return {
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "amt": amt,
        "category": random.choice(LEGIT_CATS),
        "merchant": random.choice(LEGIT_MERCHANTS),
        "lat": lat,
        "long": lon,
        "merch_lat": mlat,
        "merch_long": mlon,
        "city": city,
        "state": state,
        "city_pop": pop,
        "dob": f"{1970 + random.randint(0, 35):04d}-"
               f"{random.randint(1, 12):02d}-"
               f"{random.randint(1, 28):02d}",
        "gender": random.choice(["M", "F"]),
        "job": random.choice(LEGIT_JOBS),
    }


def _generate_fraud_transaction(base_date: datetime) -> dict:
    """Generate a single fraudulent (high-risk) transaction."""
    city, state, lat, lon, pop = random.choice(LOCS)
    # Fraud: merchant location often far from customer location
    mlat, mlon = _jitter_coord(lat, lon, deg=2.0)

    # Fraud: late night hours, high amounts
    hour = random.choice(list(range(22, 24)) + list(range(0, 4)))
    amt = round(random.uniform(480.0, 9800.0), 2)

    ts = base_date + timedelta(
        days=random.randint(0, 60),
        hours=hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    return {
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "amt": amt,
        "category": random.choice(FRAUD_CATS),
        "merchant": random.choice(FRAUD_MERCHANTS),
        "lat": lat,
        "long": lon,
        "merch_lat": mlat,
        "merch_long": mlon,
        "city": city,
        "state": state,
        "city_pop": pop,
        "dob": f"{1985 + random.randint(0, 15):04d}-"
               f"{random.randint(1, 12):02d}-"
               f"{random.randint(1, 28):02d}",
        "gender": random.choice(["M", "F"]),
        "job": random.choice(FRAUD_JOBS),
    }


def generate_transactions(
    count: int = 50,
    fraud_ratio: float = 0.5,
    base_date: datetime | None = None,
) -> list[dict]:
    """Generate a balanced list of random transaction payloads.

    Args:
        count: Total number of transactions to generate.
        fraud_ratio: Proportion of fraud transactions (0.0 to 1.0).
        base_date: Starting date for transaction timestamps.
                   Defaults to 2019-01-01 to match the model's training era.

    Returns:
        List of transaction dictionaries ready for POST /predict.
    """
    if base_date is None:
        base_date = datetime(2019, 1, 1, 8, 0, 0)

    fraud_count = int(count * fraud_ratio)
    legit_count = count - fraud_count

    transactions: list[dict] = []

    for _ in range(legit_count):
        transactions.append(_generate_legit_transaction(base_date))

    for _ in range(fraud_count):
        transactions.append(_generate_fraud_transaction(base_date))

    # Shuffle so fraud/legit are interleaved
    random.shuffle(transactions)

    return transactions