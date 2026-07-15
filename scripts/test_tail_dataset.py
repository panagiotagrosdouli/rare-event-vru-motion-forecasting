from collections import Counter

from torch.utils.data import DataLoader

from src.tail_dataset import ArgoverseTailDataset


dataset = ArgoverseTailDataset(
    "outputs/tail_dataset_train.csv"
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("\nFirst sample:")
print("Object type:", sample["object_type"])
print("Object type ID:", sample["object_type_id"])
print("Tail event:", sample["tail_event"])
print("Tail score:", sample["tail_score"])
print("Past shape:", sample["past"].shape)
print("Future shape:", sample["future"].shape)
print("Scenario:", sample["scenario_path"])

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
)

batch = next(iter(loader))

print("\nBatch:")
print("Past:", batch["past"].shape)
print("Future:", batch["future"].shape)
print("Tail events:", batch["tail_event"])
print("Object types:", batch["object_type"])

counter = Counter(
    dataset.metadata["tail_event"].tolist()
)

print("\nTail distribution:")
print(counter)
