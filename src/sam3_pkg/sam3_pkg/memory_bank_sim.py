"""memory_bank_sim.py

Simulation demo for `sam3_pkg.memory_bank.MemoryBank`.

Goal:
- Generate synthetic poses and 3D observations so that tracking is successful at
  the beginning, then gradually fails and finally becomes "all lost".

How it works:
- We create a set of fixed 3D landmarks in the WORLD frame.
- For each simulated frame we generate a camera pose.
- We synthesize observations (3D points in CAMERA frame) by transforming world
  landmarks into camera frame and adding measurement noise.
- In early frames: all landmarks are observable and noise is small -> matching succeeds.
- In later frames:
  - some landmarks get dropped (as if detector missed them)
  - plus we inject a growing outlier noise so distances exceed match_threshold
  - result: missed counts accumulate and eventually all points are marked lost.

Run (from workspace root):
    python3 -m sam3_pkg.memory_bank_sim

"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from memory_bank import MemoryBank, project_camera_points_to_image, world_points_to_camera


Pose6 = Tuple[float, float, float, float, float, float]
Vec3 = Tuple[float, float, float]


def make_landmarks_world() -> np.ndarray:
    """Create a simple square of 4 landmarks in front of the initial camera."""
    # World landmarks (meters)
    return np.array(
        [
            [0.20, 0.00, 3.00],
            [0.00, 0.20, 3.00],
            [-0.20, 0.00, 3.00],
            [0.00, -0.20, 3.00],
        ],
        dtype=float,
    )


def make_pose(frame: int) -> Pose6:
    """Simple camera motion: moving along +x and slowly yawing."""
    x = 0.02 * frame
    y = 0.0
    z = 0.0
    roll = 0.0
    pitch = 0.0
    yaw = math.radians(1.0 * frame)
    return (x, y, z, roll, pitch, yaw)


def simulate_observations(
    landmarks_world: np.ndarray,
    pose: Pose6,
    frame: int,
    rng: np.random.Generator,
) -> List[Vec3]:
    """Generate camera-frame 3D observations for this frame."""

    pts_cam = world_points_to_camera(landmarks_world, pose)

    # Keep only points in front of camera
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]

    # --- Dropouts: later frames miss more points (simulate detector failures) ---
    # Probability of dropping an observation.
    drop_p = 0.0 if frame < 10 else min(0.8, 0.05 * (frame - 10))

    kept = []
    for p in pts_cam:
        if rng.random() < drop_p:
            continue
        kept.append(p)

    if not kept:
        return []

    pts_cam_kept = np.array(kept, dtype=float)

    # --- Noise model ---
    # Small Gaussian noise early; later add a growing bias/outlier so matching breaks.
    sigma = 0.005 if frame < 15 else 0.02
    noise = rng.normal(0.0, sigma, size=pts_cam_kept.shape)

    # Outlier drift starts later and increases over time
    if frame >= 18:
        drift = 0.02 * (frame - 18)  # meters
        noise += np.array([drift, -drift, 0.0], dtype=float)

    pts_cam_obs = pts_cam_kept + noise

    return [tuple(map(float, row)) for row in pts_cam_obs]


def main() -> None:
    # --- Camera configuration ---
    intrinsics = (600.0, 600.0, 320.0, 240.0)
    image_size = (640, 480)

    # --- Tracker configuration ---
    match_threshold = 0.10
    max_missed_frames = 3
    bank = MemoryBank(match_threshold=match_threshold, max_missed_frames=max_missed_frames)

    rng = np.random.default_rng(0)

    landmarks_world = make_landmarks_world()

    # --- Initialize using frame 0 observations ---
    pose0 = make_pose(0)
    points_cam0 = simulate_observations(landmarks_world, pose0, frame=0, rng=rng)
    bank.initialize(points_cam0, pose0, intrinsics, image_size)

    print("Initialize:")
    print("  init_world_points:\n", bank.init_world_points)
    print()

    # --------- Collect simulation data for animation ---------
    frames: List[dict] = []

    for frame in range(1, 40):
        pose = make_pose(frame)

        if frame in (12, 13):
            points_cam = None
        else:
            points_cam = simulate_observations(landmarks_world, pose, frame=frame, rng=rng)

        res = bank.update(points_cam, pose)

        # Save per-frame data for visualization
        pts = []
        if points_cam is not None and len(points_cam) > 0 and res.matched_ids is not None:
            pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
            uv = project_camera_points_to_image(pts, *intrinsics)
            frames.append(
                {
                    "frame": frame,
                    "pose": pose,
                    "uv": uv,
                    "matched_ids": res.matched_ids,
                    "tracking_lost": res.tracking_lost,
                }
            )
        else:
            frames.append(
                {
                    "frame": frame,
                    "pose": pose,
                    "uv": np.zeros((0, 2), dtype=float),
                    "matched_ids": None,
                    "tracking_lost": res.tracking_lost,
                }
            )

        alive_count = int(np.sum(~bank._lost_mask))  # demo only
        obs_count = 0 if points_cam is None else len(points_cam)

        matched_count = None if res.matched_ids is None else sum(mid >= 0 for mid in res.matched_ids)
        matched_str = "None" if matched_count is None else f"{matched_count:02d}"
        odom_str = "None" if res.odometry is None else f"{res.odometry:.4f}"

        print(
            f"frame={frame:02d} obs={obs_count:02d} alive={alive_count:02d} "
            f"matched={matched_str} odom={odom_str} lost={res.tracking_lost}"
        )

        if res.tracking_lost:
            print("\nAll points lost. Stop simulation.")
            break

    # --------- Animation ---------
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as e:
        print("\nMatplotlib is required for animation. Install it and rerun:")
        print("  pip install matplotlib")
        print(f"Import error: {e}")
        return

    w, h = image_size
    fx, fy, cx, cy = intrinsics

    fig, ax = plt.subplots(figsize=(w / 160, h / 160), dpi=160)
    ax.set_title("MemoryBank tracking (image plane)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # image coords: origin top-left
    ax.set_aspect("equal", adjustable="box")

    # Draw a blank image background
    bg = np.zeros((h, w), dtype=np.uint8)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)

    scat = ax.scatter([], [], s=60)
    text_artists: List[any] = []
    info_text = ax.text(5, 15, "", color="yellow", fontsize=10, va="top")

    def _clear_texts() -> None:
        nonlocal text_artists
        for t in text_artists:
            try:
                t.remove()
            except Exception:
                pass
        text_artists = []

    def update_anim(i: int):
        data = frames[i]
        uv = data["uv"]
        mids = data["matched_ids"]

        _clear_texts()

        if uv.size == 0 or mids is None:
            scat.set_offsets(np.zeros((0, 2)))
            info_text.set_text(f"frame={data['frame']:02d}  (no observations)")
            return scat, info_text

        # Clip/cull points outside image bounds for display
        x = uv[:, 0]
        y = uv[:, 1]
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        uvv = uv[valid]
        mids_v = [mids[k] for k in range(len(mids)) if valid[k]]

        scat.set_offsets(uvv)

        for (u, v), mid in zip(uvv, mids_v):
            label = "?" if mid is None else str(mid)
            text_artists.append(ax.text(u + 4, v - 4, label, color="cyan", fontsize=10))

        info_text.set_text(
            f"frame={data['frame']:02d}  obs={len(uvv):02d}  lost={data['tracking_lost']}"
        )

        return (scat, info_text, *text_artists)

    ani = FuncAnimation(fig, update_anim, frames=len(frames), interval=250, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
