"""memory_bank_demo.py

A minimal usage example for `sam3_pkg.memory_bank.MemoryBank`.

This script demonstrates:
1) Initialization with points_cam + pose + intrinsics + image_size
2) Update with new points
3) Update with no 3D points (points_cam=None), pose only
4) Re-initialization (reset bank and initialize again)

Note:
- This is a standalone example. It uses small synthetic numbers.
- Angles are in radians.

Run (from workspace root):
    python3 -m sam3_pkg.memory_bank_demo

"""

from __future__ import annotations

import math

from sam3_pkg.memory_bank import MemoryBank


def main() -> None:
    # --- Camera configuration (example) ---
    # Intrinsics: fx, fy, cx, cy
    intrinsics = (600.0, 600.0, 320.0, 240.0)
    # Image size: width, height
    image_size = (640, 480)

    # --- Create MemoryBank ---
    # match_threshold in meters
    # max_missed_frames: allow K consecutive visible-but-unmatched frames before lost
    bank = MemoryBank(match_threshold=0.10, max_missed_frames=3)

    # --- 1) Initialize ---
    # 3D points in CAMERA coordinates (meters)
    points_cam_init = [
        (0.10, 0.00, 2.00),
        (0.00, 0.10, 2.00),
        (-0.10, 0.00, 2.00),
        (0.00, -0.10, 2.00),
    ]

    # Camera pose in WORLD: (x,y,z,roll,pitch,yaw)
    pose0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    id_map = bank.initialize(points_cam_init, pose0, intrinsics, image_size)
    print("Initialized, input_index -> id:", id_map)
    print("Init world points:\n", bank.init_world_points)

    # --- 2) Update with new observed points ---
    # Simulate a small camera motion and slightly perturbed measurements.
    pose1 = (0.05, 0.0, 0.0, 0.0, 0.0, math.radians(2.0))
    points_cam_1 = [
        (0.11, 0.01, 1.98),
        (0.01, 0.11, 1.99),
        (-0.09, 0.02, 2.02),
        (0.02, -0.09, 2.01),
    ]

    res1 = bank.update(points_cam_1, pose1)
    print("\nUpdate #1:")
    print("  matched_ids:", res1.matched_ids)
    print("  odometry:", res1.odometry)
    print("  tracking_lost:", res1.tracking_lost)
    print("  current world points:\n", bank.current_world_points)

    # --- 3) Update with no points (pose only) ---
    # E.g. detector failure for one frame.
    pose2 = (0.10, 0.0, 0.0, 0.0, 0.0, math.radians(4.0))
    res2 = bank.update(points_cam=None, pose=pose2)
    print("\nUpdate #2 (no points):")
    print("  matched_ids:", res2.matched_ids)
    print("  odometry:", res2.odometry)
    print("  tracking_lost:", res2.tracking_lost)

    # --- 4) Re-initialize ---
    # If you want to start a new tracking session, just call initialize() again.
    # This overwrites the internal state (init points, current points, masks, counters).
    print("\nRe-initialize with a new set of points...")
    points_cam_init2 = [
        (0.20, 0.00, 3.00),
        (0.00, 0.20, 3.00),
        (-0.20, 0.00, 3.00),
        (0.00, -0.20, 3.00),
    ]
    pose3 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    id_map2 = bank.initialize(points_cam_init2, pose3, intrinsics, image_size)
    print("Re-initialized, input_index -> id:", id_map2)
    print("New init world points:\n", bank.init_world_points)


if __name__ == "__main__":
    main()
