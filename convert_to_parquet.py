"""
convert_to_parquet.py

Run this script locally (requires pyarrow) to convert all CSV tables to
standard Parquet format with Snappy compression.

Install dependency:
    pip install pyarrow

Usage:
    python convert_to_parquet.py

Input:  data/*.csv
Output: data/*.parquet  (replaces CSV as primary storage format)
"""

import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    filename="convert_to_parquet.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)
log.info("=== convert_to_parquet.py started ===")

DATA_DIR = "./data"

# Explicit schemas ensure correct types regardless of CSV parsing quirks
SCHEMAS = {
    "customers": pa.schema([
        pa.field("customer_id", pa.int32()),
        pa.field("first",       pa.string()),
        pa.field("last",        pa.string()),
        pa.field("gender",      pa.string()),
        pa.field("dob",         pa.string()),
        pa.field("street",      pa.string()),
        pa.field("city",        pa.string()),
        pa.field("state",       pa.string()),
        pa.field("zip",         pa.string()),
        pa.field("job",         pa.string()),
        pa.field("city_pop",    pa.int32()),
        pa.field("lat",         pa.float32()),
        pa.field("long",        pa.float32()),
    ]),
    "cards": pa.schema([
        pa.field("card_id",      pa.string()),
        pa.field("customer_id",  pa.int32()),
        pa.field("cc_num",       pa.string()),
        pa.field("card_type",    pa.string()),
        pa.field("credit_limit", pa.float32()),
        pa.field("issue_date",   pa.string()),
        pa.field("expiry_date",  pa.string()),
    ]),
    "merchants": pa.schema([
        pa.field("merchant_id",  pa.string()),
        pa.field("merchant",     pa.string()),
        pa.field("category",     pa.string()),
        pa.field("merch_lat",    pa.float32()),
        pa.field("merch_long",   pa.float32()),
        pa.field("merch_city",   pa.string()),
        pa.field("merch_state",  pa.string()),
        pa.field("risk_rating",  pa.string()),
    ]),
    "transactions": pa.schema([
        pa.field("trans_id",              pa.int64()),
        pa.field("trans_num",             pa.string()),
        pa.field("card_id",               pa.string()),
        pa.field("merchant_id",           pa.string()),
        pa.field("category",              pa.string()),
        pa.field("amt",                   pa.float32()),
        pa.field("trans_date_trans_time", pa.string()),
        pa.field("unix_time",             pa.int64()),
        pa.field("is_fraud",              pa.int8()),
    ]),
}


def csv_to_parquet(name: str, chunksize: int = 500_000) -> None:
    """
    Convert a single CSV file to Parquet using chunked reading for memory efficiency.
    Large tables (transactions) are written in chunks to avoid OOM errors.
    """
    csv_path     = os.path.join(DATA_DIR, f"{name}.csv")
    parquet_path = os.path.join(DATA_DIR, f"{name}.parquet")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    schema  = SCHEMAS[name]
    writer  = None
    total   = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            # Cast to explicit types
            for field in schema:
                col = field.name
                if col not in chunk.columns:
                    raise ValueError(f"Missing column '{col}' in {name}.csv")
                if pa.types.is_integer(field.type):
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).astype(int)
                elif pa.types.is_floating(field.type):
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0.0)
                else:
                    chunk[col] = chunk[col].fillna("").astype(str)

            table = pa.Table.from_pandas(chunk[schema.names], schema=schema,
                                         preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(parquet_path, schema,
                                          compression='snappy')
            writer.write_table(table)
            total += len(chunk)

    finally:
        if writer:
            writer.close()

    csv_size     = os.path.getsize(csv_path)
    parquet_size = os.path.getsize(parquet_path)
    ratio        = csv_size / parquet_size
    log.info(f"{name}: {total:,} rows | CSV {csv_size/1e6:.1f} MB → "
             f"Parquet {parquet_size/1e6:.1f} MB (ratio {ratio:.1f}×)")
    print(f"  ✓ {name:<14} {total:>11,} rows  |  "
          f"CSV {csv_size/1e6:>7.1f} MB  →  Parquet {parquet_size/1e6:>7.1f} MB  "
          f"({ratio:.1f}× compression)")


def main():
    print("Converting CSVs to Parquet (Snappy compression) …\n")
    total_csv     = 0
    total_parquet = 0

    for name in ["customers", "cards", "merchants", "transactions"]:
        csv_to_parquet(name)
        total_csv     += os.path.getsize(os.path.join(DATA_DIR, f"{name}.csv"))
        total_parquet += os.path.getsize(os.path.join(DATA_DIR, f"{name}.parquet"))

    print(f"\nTotal  CSV     : {total_csv/1e9:.3f} GB")
    print(f"Total  Parquet : {total_parquet/1e9:.3f} GB")
    print("\nAll Parquet files written to ./data/")
    log.info(f"Done. Total parquet: {total_parquet/1e9:.3f} GB")
    log.info("=== convert_to_parquet.py complete ===")


if __name__ == "__main__":
    main()
