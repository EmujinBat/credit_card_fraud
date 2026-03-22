import pandas as pd

# Load raw dataset
df = pd.read_csv("creditcard_2023.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Check basic structure
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

# Verify expected columns from the original dataset
required_cols = ["Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

print("\nAll expected columns are present.")

# Create surrogate key because original data has no id column
df.insert(0, "TransactionID", range(1, len(df) + 1))

# Check surrogate key uniqueness
assert df["TransactionID"].nunique() == len(df), "TransactionID is not unique."

print("\nTransactionID created successfully.")

# Save cleaned working copy
df.to_csv("creditcard_2023_clean.csv", index=False)

print("\nCleaned dataset saved as creditcard_2023_clean.csv")
