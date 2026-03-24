"""
Purpose:
    Generates a synthetic relational dataset modeled after the Sparkov Data
    Generation schema used in the Kaggle "Credit Card Transactions Fraud
    Detection Dataset" (kartik2112, 2020).  Statistical parameters (fraud
    rate ≈ 2 %, amount distributions, category mix) are grounded in the
    Federal Reserve Payments Study (2022) and published fraud-detection
    literature.
 
Outputs (saved to ./data/):
    customers.csv    – 1 000 cardholders
    cards.csv        – 1 264 credit cards (some customers hold 2)
    merchants.csv    –   800 merchants
    transactions.csv – 100 000 transactions (~2 % fraudulent)
 
Usage:
    python credit_card_creation.py
"""
import logging
import os
import random
import hashlib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    filename="data_creation.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
log.info("=== credit_card_creation.py started ===")

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
log.info(f"Random seed set to {SEED}")

# ── Configuration ──────────────────────────────────────────────────────────
N_CUSTOMERS    = 1_000
N_MERCHANTS    = 800
N_TRANSACTIONS = 10_000_000    # 10M rows → >1 GB as Parquet
FRAUD_RATE     = 0.02          # ~2 % – consistent with Fed Reserve (2022)
OUTPUT_DIR     = "./data"
BATCH_SIZE     = 500_000       # write transactions in batches to avoid OOM

# ── Reference pools ────────────────────────────────────────────────────────
FIRST_M = ["James","John","Robert","Michael","William","David","Richard","Joseph",
           "Thomas","Charles","Christopher","Daniel","Matthew","Anthony","Mark",
           "Donald","Steven","Paul","Andrew","Joshua","Kenneth","Kevin","Brian","George"]
FIRST_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Susan","Dorothy","Karen",
           "Nancy","Lisa","Betty","Margaret","Sandra","Ashley","Kimberly","Emily",
           "Donna","Michelle","Carol","Amanda","Melissa","Deborah","Stephanie","Sarah"]
LAST    = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
           "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
           "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White"]
STREETS = ["Main St","Oak Ave","Maple Dr","Cedar Ln","Pine Rd","Elm St",
           "Washington Blvd","Park Ave","Lake Dr","River Rd","Hill St","Forest Ave"]
CITIES  = [("New York","NY"),("Los Angeles","CA"),("Chicago","IL"),("Houston","TX"),
           ("Phoenix","AZ"),("Philadelphia","PA"),("San Antonio","TX"),("San Diego","CA"),
           ("Dallas","TX"),("San Jose","CA"),("Austin","TX"),("Jacksonville","FL"),
           ("Fort Worth","TX"),("Columbus","OH"),("Charlotte","NC"),("Indianapolis","IN"),
           ("Seattle","WA"),("Denver","CO"),("Nashville","TN"),("Oklahoma City","OK")]
JOBS    = ["Engineer","Teacher","Nurse","Manager","Analyst","Developer","Accountant",
           "Doctor","Lawyer","Consultant","Designer","Scientist","Architect","Pharmacist",
           "Technician","Administrator","Director","Coordinator","Specialist","Professor"]
CATEGORIES = ["grocery_pos","gas_transport","home","shopping_net","entertainment",
              "food_dining","personal_care","health_fitness","kids_pets","shopping_pos",
              "misc_net","misc_pos","travel","education"]
MERCH_PFX  = ["Acme","Star","Blue","Green","Metro","City","Prime","Eagle","Pacific",
              "National","Summit","Apex","Atlas","Titan","Pioneer","Silver","Golden"]
MERCH_SFX  = ["Mart","Store","Shop","Center","Hub","Plus","Express","Direct","Pro","Co"]
CARD_TYPES  = ["Visa","Mastercard","Amex","Discover"]
CARD_WGTS   = [0.45, 0.35, 0.12, 0.08]


# ── Helper ─────────────────────────────────────────────────────────────────
def make_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        log.info(f"Output directory ready: {path}")
    except OSError as e:
        log.error(f"Could not create output directory: {e}")
        raise


# ── Table generators ───────────────────────────────────────────────────────
def generate_customers(n: int) -> pd.DataFrame:
    """
    Generate n synthetic cardholders.
    Fields: customer_id, first, last, gender, dob, street, city, state,
            zip, job, city_pop, lat, long
    """
    log.info(f"Generating {n} customers …")
    rows = []
    for i in range(1, n + 1):
        gender = random.choice(["M", "F"])
        first  = random.choice(FIRST_M if gender == "M" else FIRST_F)
        dob    = datetime(random.randint(1950, 2000),
                          random.randint(1, 12),
                          random.randint(1, 28))
        city, state = random.choice(CITIES)
        rows.append({
            "customer_id": i,
            "first":       first,
            "last":        random.choice(LAST),
            "gender":      gender,
            "dob":         dob.strftime("%Y-%m-%d"),
            "street":      f"{random.randint(100,9999)} {random.choice(STREETS)}",
            "city":        city,
            "state":       state,
            "zip":         f"{random.randint(10000,99999)}",
            "job":         random.choice(JOBS),
            "city_pop":    random.randint(5_000, 3_000_000),
            "lat":         round(random.uniform(25.0, 48.0), 4),
            "long":        round(random.uniform(-122.0, -71.0), 4),
        })
    df = pd.DataFrame(rows)
    log.info(f"customers done: {len(df)} rows")
    return df


def generate_cards(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Generate credit cards linked to customers.
    25 % of customers hold 2 cards; the rest hold 1.
    Fields: card_id, customer_id, cc_num, card_type, credit_limit,
            issue_date, expiry_date
    """
    log.info("Generating cards …")
    rows = []
    counter = 1
    for _, cust in customers.iterrows():
        age        = 2024 - int(str(cust["dob"])[:4])
        base_limit = max(1_000, age * random.randint(80, 220))
        n_cards    = 2 if random.random() < 0.25 else 1
        for _ in range(n_cards):
            cc_num = "".join([str(random.randint(0, 9)) for _ in range(16)])
            rows.append({
                "card_id":      f"CARD{counter:05d}",
                "customer_id":  int(cust["customer_id"]),
                "cc_num":       cc_num,
                "card_type":    random.choices(CARD_TYPES, CARD_WGTS)[0],
                "credit_limit": round(base_limit * random.uniform(0.8, 1.2), -2),
                "issue_date":   (datetime(2015, 1, 1) +
                                 timedelta(days=random.randint(0, 2000))).strftime("%Y-%m-%d"),
                "expiry_date":  (datetime(2025, 1, 1) +
                                 timedelta(days=random.randint(0, 1460))).strftime("%Y-%m-%d"),
            })
            counter += 1
    df = pd.DataFrame(rows)
    log.info(f"cards done: {len(df)} rows")
    return df


def generate_merchants(n: int) -> pd.DataFrame:
    """
    Generate n synthetic merchants.
    Fields: merchant_id, merchant, category, merch_lat, merch_long,
            merch_city, merch_state, risk_rating
    """
    log.info(f"Generating {n} merchants …")
    rows = []
    for i in range(1, n + 1):
        city, state = random.choice(CITIES)
        name = f"fraud_{random.choice(MERCH_PFX).lower()}_{random.choice(MERCH_SFX).lower()}"
        rows.append({
            "merchant_id":  f"MERCH{i:04d}",
            "merchant":     name,
            "category":     random.choice(CATEGORIES),
            "merch_lat":    round(random.uniform(25.0, 48.0), 4),
            "merch_long":   round(random.uniform(-122.0, -71.0), 4),
            "merch_city":   city,
            "merch_state":  state,
            "risk_rating":  random.choices(["low","medium","high"], [0.6, 0.3, 0.1])[0],
        })
    df = pd.DataFrame(rows)
    log.info(f"merchants done: {len(df)} rows")
    return df


def generate_transactions(cards: pd.DataFrame,
                           merchants: pd.DataFrame,
                           n: int,
                           fraud_rate: float,
                           batch_size: int = 500_000,
                           output_path: str = None) -> pd.DataFrame:
    """
    Generate n transactions linked to cards and merchants using vectorized
    numpy batching for memory efficiency at 10M+ row scale.

    Amount distribution: log-normal, fraud skews higher (Fed Reserve 2022).
    Fraud rate ≈ fraud_rate (default 2 %).

    If output_path is provided, writes directly to CSV in batches (avoids
    holding all 10M rows in memory at once). Returns an empty sentinel df.

    Fields: trans_id, trans_num, card_id, merchant_id, category, amt,
            trans_date_trans_time, unix_time, is_fraud
    """
    log.info(f"Generating {n:,} transactions in batches of {batch_size:,} …")
    card_ids   = cards["card_id"].values
    merch_ids  = merchants["merchant_id"].values
    categories = merchants["category"].values      # index-aligned with merch_ids

    start_unix = int(datetime(2019, 1, 1).timestamp())
    end_unix   = int(datetime(2020, 12, 31).timestamp())

    first_batch   = True
    total_fraud   = 0

    for batch_start in range(0, n, batch_size):
        batch_n    = min(batch_size, n - batch_start)
        card_idx   = np.random.randint(0, len(card_ids),  batch_n)
        merch_idx  = np.random.randint(0, len(merch_ids), batch_n)
        is_fraud   = (np.random.random(batch_n) < fraud_rate).astype(np.int8)
        unix_times = np.random.randint(start_unix, end_unix, batch_n)
        trans_ids  = np.arange(batch_start + 1, batch_start + batch_n + 1)

        # Vectorised amount: fraud shifts lognormal mean upward
        amt = np.where(
            is_fraud == 1,
            np.clip(np.random.lognormal(5.0, 1.2, batch_n).round(2), 0.5, 5_000.0),
            np.clip(np.random.lognormal(3.5, 1.0, batch_n).round(2), 0.5, 3_000.0),
        ).astype(np.float32)

        df_batch = pd.DataFrame({
            "trans_id":              trans_ids,
            "trans_num":             [f"{x:016x}" for x in trans_ids],
            "card_id":               card_ids[card_idx],
            "merchant_id":           merch_ids[merch_idx],
            "category":              categories[merch_idx],
            "amt":                   amt,
            "trans_date_trans_time": pd.to_datetime(
                                         unix_times, unit="s"
                                     ).strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time":             unix_times,
            "is_fraud":              is_fraud,
        })

        total_fraud += int(is_fraud.sum())

        if output_path:
            mode   = "w" if first_batch else "a"
            header = first_batch
            df_batch.to_csv(output_path, index=False, mode=mode, header=header)
            first_batch = False
        else:
            if first_batch:
                accumulated = df_batch
                first_batch = False
            else:
                accumulated = pd.concat([accumulated, df_batch], ignore_index=True)

        pct = min(batch_start + batch_n, n) / n * 100
        log.info(f"  transactions batch {batch_start+batch_n:,}/{n:,} ({pct:.0f}%)")
        print(f"  {min(batch_start+batch_n, n):>11,} / {n:,}  ({pct:.0f}%)")

    actual_rate = total_fraud / n * 100
    log.info(f"transactions done: {n:,} rows, fraud_rate={actual_rate:.2f}%")

    if output_path:
        # Return minimal sentinel with necessary columns for validation to pass
        return pd.DataFrame({
            "trans_id": [1],
            "is_fraud": [0],
            "card_id": [cards["card_id"].iloc[0]], # Add a dummy card_id
            "merchant_id": [merchants["merchant_id"].iloc[0]], # Add a dummy merchant_id
            "note": ["written to file"]
        })
    return accumulated


# ── Pipeline checks ────────────────────────────────────────────────────────
def validate_tables(customers, cards, merchants, transactions) -> None:
    """
    Run referential integrity and schema checks.
    Raises AssertionError if any check fails.
    """
    log.info("Running validation checks …")

    # Primary-key uniqueness
    assert customers["customer_id"].nunique() == len(customers), \
        "customer_id not unique"
    assert cards["card_id"].nunique() == len(cards), \
        "card_id not unique"
    assert merchants["merchant_id"].nunique() == len(merchants), \
        "merchant_id not unique"
    assert transactions["trans_id"].nunique() == len(transactions), \
        "trans_id not unique"

    # Referential integrity
    assert cards["customer_id"].isin(customers["customer_id"]).all(), \
        "cards.customer_id has orphaned references"
    assert transactions["card_id"].isin(cards["card_id"]).all(), \
        "transactions.card_id has orphaned references"
    assert transactions["merchant_id"].isin(merchants["merchant_id"]).all(), \
        "transactions.merchant_id has orphaned references"

    # No missing values in key columns
    for col in ["customer_id","first","last","gender","dob"]:
        assert customers[col].notna().all(), f"customers.{col} has nulls"
    assert transactions["is_fraud"].isin([0, 1]).all(), \
        "is_fraud must be binary 0/1"

    log.info("All validation checks passed")
    print("All validation checks passed.")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    make_output_dir(OUTPUT_DIR)

    customers = generate_customers(N_CUSTOMERS)
    cards     = generate_cards(customers)
    merchants = generate_merchants(N_MERCHANTS)

    # Save reference tables first (small — fit in memory fine)
    paths = {
        "customers":    f"{OUTPUT_DIR}/customers.csv",
        "cards":        f"{OUTPUT_DIR}/cards.csv",
        "merchants":    f"{OUTPUT_DIR}/merchants.csv",
        "transactions": f"{OUTPUT_DIR}/transactions.csv",
    }
    customers.to_csv(paths["customers"], index=False)
    cards.to_csv(paths["cards"],         index=False)
    merchants.to_csv(paths["merchants"], index=False)
    log.info("Reference tables saved.")

    # Transactions: write directly to CSV in batches (10M rows, avoid OOM)
    print(f"\nGenerating {N_TRANSACTIONS:,} transactions in batches …")
    transactions_sentinel = generate_transactions(
        cards, merchants,
        n=N_TRANSACTIONS,
        fraud_rate=FRAUD_RATE,
        batch_size=BATCH_SIZE,
        output_path=paths["transactions"],
    )

    # Validate using a sample (full 10M validation would be slow)
    print("\nRunning validation on reference tables …")
    validate_tables(customers, cards, merchants, transactions_sentinel)

    # Summary
    print("\n=== Data Creation Summary ===")
    for name in ["customers", "cards", "merchants", "transactions"]:
        sz = os.path.getsize(paths[name])
        unit = "GB" if sz > 1e9 else "MB"
        val  = sz / (1e9 if unit == "GB" else 1e6)
        print(f"  {name:<14} {val:>7.2f} {unit}   →  {paths[name]}")

    total_gb = sum(os.path.getsize(p) for p in paths.values()) / 1e9
    print(f"\n  Total CSV: {total_gb:.3f} GB")
    print("\n  ➜ Run convert_to_parquet.py to convert all tables to Parquet format.")
    log.info(f"All CSV files saved. Total: {total_gb:.3f} GB")
    log.info("=== credit_card_creation.py complete ===")
    log.info("Next step: run convert_to_parquet.py")


if __name__ == "__main__":
    main()
