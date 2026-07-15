from pathlib import Path
import pandas as pd

scenario = Path(
    "train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/"
    "scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
)

df = pd.read_parquet(scenario)

print("Unique agents:", df["track_id"].nunique())

print("\nObject types:")
print(df["object_type"].value_counts())

print("\nObject categories:")
print(df["object_category"].value_counts())

print("\nObserved values:")
print(df["observed"].value_counts())

print("\nTimesteps:")
print(df["timestep"].min(), "to", df["timestep"].max())

print("\nFocal track:")
print(df["focal_track_id"].iloc[0])

focal_id = df["focal_track_id"].iloc[0]
focal_df = df[df["track_id"] == focal_id]

print("\nFocal object type:")
print(focal_df["object_type"].iloc[0])

print("\nFocal trajectory length:")
print(len(focal_df))

print("\nFocal observed/future split:")
print(focal_df["observed"].value_counts())
