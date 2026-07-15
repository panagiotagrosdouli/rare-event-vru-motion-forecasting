from pathlib import Path
import pandas as pd

from src.preprocess import preprocess_focal_agent


scenario = Path(
    "train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/"
    "scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
)

df = pd.read_parquet(scenario)

past, future, obj, origin, angle = preprocess_focal_agent(df)

print("Object:", obj)
print("Origin:", origin)
print("Rotation angle:", angle)
print("Past shape:", past.shape)
print("Future shape:", future.shape)

print("\nLast observed point:")
print(past[-1])

print("\nPrevious observed point:")
print(past[-2])

print("\nFirst future points:")
print(future[:5])
