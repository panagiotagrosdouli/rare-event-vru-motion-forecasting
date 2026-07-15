import pandas as pd

# Load statistics
df = pd.read_csv("outputs/tail_statistics_train.csv")

print("=" * 60)
print("VRU DATASET STATISTICS")
print("=" * 60)

print("\nClass distribution:")
print(df["object_type"].value_counts())

features = [
    "max_speed",
    "max_acceleration",
    "max_heading_change",
]

for feature in features:

    print("\n" + "=" * 60)
    print(feature.upper())
    print("=" * 60)

    print(df[feature].describe())

    print("\nPercentiles")

    for p in [90, 95, 97, 99]:

        value = df[feature].quantile(p / 100)

        count = (df[feature] >= value).sum()

        print(
            f"{p}% : {value:.4f} ({count} trajectories)"
        )
