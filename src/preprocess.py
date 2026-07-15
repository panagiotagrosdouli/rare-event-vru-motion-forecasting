import numpy as np


def preprocess_focal_agent(df):
    focal_id = df["focal_track_id"].iloc[0]

    focal = df[df["track_id"] == focal_id].copy()
    focal = focal.sort_values("timestep")

    past = focal[focal["observed"]]
    future = focal[~focal["observed"]]

    past_xy = past[["position_x", "position_y"]].to_numpy(dtype=np.float32)
    future_xy = future[["position_x", "position_y"]].to_numpy(dtype=np.float32)

    if past_xy.shape != (50, 2):
        raise ValueError(f"Unexpected past shape: {past_xy.shape}")

    if future_xy.shape != (60, 2):
        raise ValueError(f"Unexpected future shape: {future_xy.shape}")

    origin = past_xy[-1].copy()

    past_xy = past_xy - origin
    future_xy = future_xy - origin

    # Direction from the last two observed points
    direction = past_xy[-1] - past_xy[-2]
    angle = np.arctan2(direction[1], direction[0])

    # Rotate by -angle so motion points toward +x
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    rotation = np.array(
        [
            [cos_a, -sin_a],
            [sin_a,  cos_a],
        ],
        dtype=np.float32,
    )

    past_xy = past_xy @ rotation.T
    future_xy = future_xy @ rotation.T

    object_type = focal["object_type"].iloc[0]

    return past_xy, future_xy, object_type, origin, angle
