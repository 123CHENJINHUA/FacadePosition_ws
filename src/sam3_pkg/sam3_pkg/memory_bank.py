"""memory_bank.py

A small utility for tracking a set of 3D points across frames by maintaining a
"memory bank" of their world coordinates.

Requirements from user:
- Initialize with 3D points in camera coordinates and camera pose (x,y,z,r,p,y).
- If multiple points at init: label them clockwise.
- Store initial points in *world* coordinates (computed from pose).
- Update: given new camera-frame 3D points + pose -> compute world points,
  match them to current tracked points and output matched IDs.
- If matched pair distance > threshold => tracking lost for that point.
- If at least one match: update *current* tracked world positions using a new
  list (do not overwrite the initial list). Future matching uses current list.
- Odometer: continuously compute distance between newly observed (matched) points
  and the *initial* points. Output the mean distance as odometry.
- After update: return IDs for matched points; if all points lost return None.

Notes:
- Clockwise ordering is defined in XY plane of the *world* coordinates at init.
- Pose uses roll-pitch-yaw (r,p,y) in radians, ZYX convention (yaw->pitch->roll).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np


Vec3 = Tuple[float, float, float]
Pose6 = Tuple[float, float, float, float, float, float]  # x,y,z,roll,pitch,yaw


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return rotation matrix R (3x3) from roll-pitch-yaw.

    Uses ZYX order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return rz @ ry @ rx


def camera_points_to_world(points_cam: np.ndarray, pose: Pose6) -> np.ndarray:
    """Transform Nx3 points from camera frame to world frame."""

    if points_cam.size == 0:
        return points_cam.reshape(0, 3)

    x, y, z, r, p, yy = pose
    t = np.array([x, y, z], dtype=float)
    rmat = _rpy_to_rot(r, p, yy)

    # p_world = R * p_cam + t
    return (rmat @ points_cam.T).T + t


def world_points_to_camera(points_world: np.ndarray, pose: Pose6) -> np.ndarray:
    """Transform Nx3 points from world frame to camera frame."""
    if points_world.size == 0:
        return points_world.reshape(0, 3)

    x, y, z, r, p, yy = pose
    t = np.array([x, y, z], dtype=float)
    rmat = _rpy_to_rot(r, p, yy)

    # p_cam = R^T * (p_world - t)
    return (rmat.T @ (points_world - t).T).T


def project_camera_points_to_image(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project Nx3 camera-frame points to Nx2 pixel coordinates (u,v)."""
    if points_cam.size == 0:
        return points_cam.reshape(0, 2)

    z = points_cam[:, 2]
    # Avoid division by zero; invalid (z<=0) should be handled by caller.
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return np.stack([u, v], axis=1)


def _clockwise_order_xy(points_world: np.ndarray) -> List[int]:
    """Return indices that sort points clockwise by angle around centroid."""

    if len(points_world) <= 1:
        return list(range(len(points_world)))

    centroid = points_world[:, :2].mean(axis=0)
    dxdy = points_world[:, :2] - centroid
    angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])  # [-pi, pi]

    # Clockwise: descending angle; tie-breaker by radius (farther first)
    radii = np.linalg.norm(dxdy, axis=1)
    order = np.lexsort(( -radii, -angles))  # sort by -angles, then -radii
    return order.tolist()


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances between a (N,3) and b (M,3)."""
    # (N,1,3) - (1,M,3) -> (N,M,3)
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


@dataclass
class UpdateResult:
    matched_ids: Optional[List[int]]
    odometry: Optional[float]
    tracking_lost: bool


class MemoryBank:
    """Tracks a set of 3D points in world coordinates."""

    def __init__(self, match_threshold: float = 0.05, max_missed_frames: int = 0):
        self.match_threshold = float(match_threshold)
        self.max_missed_frames = int(max_missed_frames)

        # Camera model (set on initialize)
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._img_w: Optional[int] = None
        self._img_h: Optional[int] = None

        self._initialized: bool = False

        # Initial (never changed) world coordinates, ordered by assigned id.
        self._init_world: np.ndarray = np.zeros((0, 3), dtype=float)

        # Current tracked world coordinates (updated over time), aligned with ids.
        self._cur_world: np.ndarray = np.zeros((0, 3), dtype=float)

        # If a point is currently considered lost.
        self._lost_mask: np.ndarray = np.zeros((0,), dtype=bool)

        # Consecutive missed-frame counter per id (only meaningful while not lost).
        self._miss_counts: np.ndarray = np.zeros((0,), dtype=int)

        # Latest odometry value
        self._odometry: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def init_world_points(self) -> np.ndarray:
        return self._init_world.copy()

    @property
    def current_world_points(self) -> np.ndarray:
        return self._cur_world.copy()

    @property
    def odometry(self) -> Optional[float]:
        return self._odometry

    def initialize(
        self,
        points_cam: Sequence[Vec3],
        pose: Pose6,
        intrinsics: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        remain_odometry: bool = False,
    ) -> List[int]:
        """Initialize the bank.

        Args:
            points_cam: 3D points in camera frame.
            pose: camera pose in world (x,y,z,roll,pitch,yaw).
            intrinsics: (fx, fy, cx, cy).
            image_size: (width, height).

        Returns:
            List of assigned IDs aligned with the original input order.
        """

        self._fx, self._fy, self._cx, self._cy = map(float, intrinsics)
        self._img_w, self._img_h = int(image_size[0]), int(image_size[1])

        pts_cam = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        pts_world = camera_points_to_world(pts_cam, pose)

        order = _clockwise_order_xy(pts_world)
        pts_world_ordered = pts_world[order]

        self._init_world = pts_world_ordered.copy()
        self._cur_world = pts_world_ordered.copy()
        self._lost_mask = np.zeros((len(pts_world_ordered),), dtype=bool)
        self._miss_counts = np.zeros((len(pts_world_ordered),), dtype=int)
        self._initialized = True

        if not remain_odometry:
            self._odometry = 0.0

        inv = np.empty((len(order),), dtype=int)
        for new_id, old_i in enumerate(order):
            inv[old_i] = new_id
        return inv.tolist()

    def _visible_mask_current(self, pose: Pose6) -> np.ndarray:
        """Return boolean mask (len==n_ids) indicating which current points project into image."""
        if self._fx is None or self._fy is None or self._cx is None or self._cy is None:
            raise RuntimeError("Camera intrinsics not set. Call initialize(..., intrinsics, image_size)")
        if self._img_w is None or self._img_h is None:
            raise RuntimeError("Image size not set. Call initialize(..., intrinsics, image_size)")

        n_ids = len(self._cur_world)
        if n_ids == 0:
            return np.zeros((0,), dtype=bool)

        pts_cam = world_points_to_camera(self._cur_world, pose)
        z = pts_cam[:, 2]
        visible = z > 0

        uv = project_camera_points_to_image(pts_cam, self._fx, self._fy, self._cx, self._cy)
        u, v = uv[:, 0], uv[:, 1]

        visible &= (u >= 0) & (u < self._img_w) & (v >= 0) & (v < self._img_h)
        return visible

    def update(self, points_cam: Sequence[Vec3], pose: Pose6) -> UpdateResult:
        """Update tracking with a new observation.

        Matching is done against current tracked positions of not-lost points.

        Returns:
            UpdateResult:
              - matched_ids: IDs of matched points in the same order as the
                provided `points_cam`. For unmatched points: -1.
                If *all* tracked points are lost/unmatched -> None.
              - odometry: mean distance of matched observed points to their
                corresponding *initial* world positions.
              - tracking_lost: True if at least one point exceeds threshold.
        """

        if not self._initialized:
            raise RuntimeError("MemoryBank is not initialized. Call initialize() first.")

        if points_cam is None:
            points_cam = []

        obs_cam = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        obs_world = camera_points_to_world(obs_cam, pose)

        n_ids = len(self._cur_world)
        if n_ids == 0:
            return UpdateResult(matched_ids=None, odometry=None, tracking_lost=True)

        # Candidates are currently not lost
        alive_ids = np.where(~self._lost_mask)[0]
        if alive_ids.size == 0:
            return UpdateResult(matched_ids=None, odometry=None, tracking_lost=True)

        cur_alive = self._cur_world[alive_ids]

        matched_ids: List[int] = [-1] * len(obs_world)
        tracking_lost = False

        # Greedy nearest-neighbor assignment with one-to-one constraint.
        # For small number of points this is sufficient; if you need optimal,
        # replace with Hungarian algorithm.
        if len(obs_world) > 0:
            dmat = _pairwise_dist(obs_world, cur_alive)  # (Nobs, Nalive)
            # Flatten and sort by distance
            pairs = [(i, j, dmat[i, j]) for i in range(dmat.shape[0]) for j in range(dmat.shape[1])]
            pairs.sort(key=lambda x: x[2])

            used_obs = set()
            used_alive = set()

            # Collect updates and commit once at the end
            new_positions = self._cur_world.copy()

            for oi, aj, dist in pairs:
                if oi in used_obs or aj in used_alive:
                    continue

                real_id = int(alive_ids[aj])

                # Only accept a match if the observed point is within threshold.
                # (Re-check vs the latest track position to avoid any subtle
                # order effects and to match the intended semantics.)
                dist_to_latest = float(np.linalg.norm(obs_world[oi] - new_positions[real_id]))
                if dist <= self.match_threshold and dist_to_latest <= self.match_threshold:
                    used_obs.add(oi)
                    used_alive.add(aj)

                    matched_ids[oi] = real_id

                    # Update current track position with the newly observed world point
                    new_positions[real_id] = obs_world[oi]
                # else:
                #     print(
                #         f"Point {oi} unmatched (closest dist {dist:.4f} > threshold {self.match_threshold:.4f})"
                #     )

            # Visible points are the only ones eligible for missed-count increase.
            visible_mask = self._visible_mask_current(pose)

            # Update miss counters and decide whether a point becomes lost
            matched_id_set = {mid for mid in matched_ids if mid >= 0}

            # Reset miss count for matched ids
            for mid in matched_id_set:
                self._miss_counts[int(mid)] = 0

            # Increment miss count for visible+alive-but-unmatched ids; mark lost if exceeded
            for idx in alive_ids:
                idx_i = int(idx)
                if idx_i in matched_id_set:
                    continue
                if not bool(visible_mask[idx_i]):
                    # Not in image => do not count as missed / do not mark lost
                    continue

                self._miss_counts[idx_i] += 1
                if self._miss_counts[idx_i] > self.max_missed_frames:
                    self._lost_mask[idx_i] = True

            # Commit the new current positions (do not touch init positions)
            self._cur_world = new_positions
        else:
            # No observations: still update miss counters based on visibility
            visible_mask = self._visible_mask_current(pose)
            for idx in alive_ids:
                idx_i = int(idx)
                if not bool(visible_mask[idx_i]):
                    continue
                self._miss_counts[idx_i] += 1
                if self._miss_counts[idx_i] > self.max_missed_frames:
                    self._lost_mask[idx_i] = True

        # If all tracked points are lost, return None
        if bool(np.all(self._lost_mask)):
            self._odometry = None
            return UpdateResult(matched_ids=None, odometry=None, tracking_lost=True)

        # Odometry: mean distance from matched observed points to initial positions
        odom_vals: List[float] = []
        for oi, mid in enumerate(matched_ids):
            if mid < 0:
                continue
            odom_vals.append(float(np.linalg.norm(obs_world[oi] - self._init_world[mid])))

        self._odometry = float(np.mean(odom_vals)) if odom_vals else self._odometry

        # If no points matched in this frame, do NOT declare overall tracking lost
        # (points may re-appear within max_missed_frames)
        if all(mid < 0 for mid in matched_ids):
            return UpdateResult(matched_ids=None, odometry=self._odometry, tracking_lost=False)

        return UpdateResult(matched_ids=matched_ids, odometry=self._odometry, tracking_lost=False)