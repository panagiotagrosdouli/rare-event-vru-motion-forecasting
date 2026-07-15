from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import ArgoverseFocalDataset


dataset = ArgoverseFocalDataset("train")

type_ids = []

for i in range(len(dataset)):
    sample = dataset[i]
    type_ids.append(sample["object_type_id"])

counts = Counter(type_ids)

print("Class counts:", counts)

class_weights = {
    class_id: len(type_ids) / count
    for class_id, count in counts.items()
}

sample_weights = torch.tensor(
    [class_weights[type_id] for type_id in type_ids],
    dtype=torch.double,
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=0,
)

batch = next(iter(loader))

print("Object types:")
print(batch["object_type"])

print("Object type IDs:")
print(batch["object_type_id"])
