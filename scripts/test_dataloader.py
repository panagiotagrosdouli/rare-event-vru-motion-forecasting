from torch.utils.data import DataLoader

from src.dataset import ArgoverseFocalDataset


dataset = ArgoverseFocalDataset("train")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

batch = next(iter(loader))

print("Past batch shape:", batch["past"].shape)
print("Future batch shape:", batch["future"].shape)

print("Object types:")
print(batch["object_type"])

print("Object type IDs:")
print(batch["object_type_id"])

print("Origin shape:", batch["origin"].shape)
print("Angle shape:", batch["angle"].shape)
