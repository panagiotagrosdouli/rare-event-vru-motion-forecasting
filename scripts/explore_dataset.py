from pathlib import Path
import pandas as pd

scenario = Path(
    "train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
)

df = pd.read_parquet(scenario)

print("=" * 60)
print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nFirst rows:")
print(df.head())
