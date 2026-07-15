from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

scenario = Path(
    "train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/"
    "scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
)

df = pd.read_parquet(scenario)

plt.figure(figsize=(10,10))

focal_id = df["focal_track_id"].iloc[0]

for track_id, track in df.groupby("track_id"):

    obj_type = track["object_type"].iloc[0]

    if obj_type == "vehicle":
        color = "blue"
    elif obj_type == "pedestrian":
        color = "red"
    elif obj_type == "cyclist":
        color = "green"
    else:
        color = "gray"

    past = track[track["observed"]]
    future = track[~track["observed"]]

    # Past
    plt.plot(
        past["position_x"],
        past["position_y"],
        color=color,
        linewidth=1
    )

    # Future
    plt.plot(
        future["position_x"],
        future["position_y"],
        "--",
        color=color,
        linewidth=1
    )

# Highlight focal agent
focal = df[df["track_id"] == focal_id]

past = focal[focal["observed"]]
future = focal[~focal["observed"]]

plt.plot(
    past["position_x"],
    past["position_y"],
    color="black",
    linewidth=3,
    label="Observed"
)

plt.plot(
    future["position_x"],
    future["position_y"],
    "--",
    color="magenta",
    linewidth=3,
    label="Future (Ground Truth)"
)

plt.scatter(
    past.iloc[-1]["position_x"],
    past.iloc[-1]["position_y"],
    s=80,
    color="orange",
    label="Prediction starts"
)

plt.axis("equal")
plt.legend()
plt.title("Past vs Future Trajectories")
plt.show()
