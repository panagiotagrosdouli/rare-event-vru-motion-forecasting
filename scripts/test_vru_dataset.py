from collections import Counter

from src.dataset import ArgoverseFocalDataset


dataset = ArgoverseFocalDataset(
    root_dir="train",
    only_vrus=True,
)

print("\nTotal VRU scenarios:", len(dataset))

counter = Counter()

for index in range(len(dataset)):
    sample = dataset[index]
    counter[sample["object_type"]] += 1

print("\nVRU class distribution:")

for object_type, count in counter.items():
    print(f"{object_type}: {count}")

sample = dataset[0]

print("\nFirst sample:")
print("Object type:", sample["object_type"])
print("Object type ID:", sample["object_type_id"])
print("Past shape:", sample["past"].shape)
print("Future shape:", sample["future"].shape)
print("Scenario:", sample["scenario_path"])
