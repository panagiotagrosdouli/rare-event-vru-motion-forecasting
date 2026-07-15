from src.dataset import ArgoverseFocalDataset


dataset = ArgoverseFocalDataset("train")

print("Number of scenarios:", len(dataset))

sample = dataset[0]

print("Past shape:", sample["past"].shape)
print("Future shape:", sample["future"].shape)
print("Object type:", sample["object_type"])
print("Origin:", sample["origin"])
print("Angle:", sample["angle"])
print("Scenario:", sample["scenario_path"])
