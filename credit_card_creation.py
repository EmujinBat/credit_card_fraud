import pandas as pd

# load raw data
df = pd.read_csv("creditcard_2023.csv")

# basic inspection
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)

# missing-value check
print(df.isna().sum())

# verify important columns
required_cols = ["id", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
print(all(col in df.columns for col in required_cols))

# optional: save cleaned working copy
df.columns = df.columns.str.strip()
df.to_csv("creditcard_2023_clean.csv", index=False)
