import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TAIL_PERCENTILE = 95

df = pd.read_csv("outputs/tail_statistics_train.csv")

features = [
    "max_speed",
    "max_acceleration",
    "max_heading_change",
]

scaler = MinMaxScaler()

normalized = scaler.fit_transform(df[features])

df["tail_score"] = (
    0.4 * normalized[:, 0] +
    0.4 * normalized[:, 1] +
    0.2 * normalized[:, 2]
)

threshold = df["tail_score"].quantile(TAIL_PERCENTILE / 100)

df["tail_event"] = (
    df["tail_score"] >= threshold
).astype(int)

print("=" * 60)
print("TAIL EVENT SUMMARY")
print("=" * 60)

print(df["tail_event"].value_counts())

print("\nThreshold:", threshold)

df.to_csv(
    "outputs/tail_dataset_train.csv",
    index=False,
)

print("\nSaved:")
print("outputs/tail_dataset_train.csv")
