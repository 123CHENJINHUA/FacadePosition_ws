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
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math

import numpy as np


Vec3 = Tuple[float, float, float]
Pose6 = Tuple[float, float, float, float, float, float]  # x,y,z,roll,pitch,yaw
Odometry3 = Tuple[float, float, float]


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


def _estimate_rigid_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate rigid body transform (R, t) from matched point pairs using SVD.

    Solves: dst ≈ R @ src + t  (least squares, closed-form via SVD)

    Args:
        src: (N, 3) source points (previous positions of matched points).
        dst: (N, 3) destination points (newly observed positions).

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    assert src.shape == dst.shape and src.shape[0] >= 1

    # Centroids
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)

    # Center the point clouds
    A = src - c_src  # (N, 3)
    B = dst - c_dst  # (N, 3)

    # SVD of cross-covariance matrix
    H = A.T @ B  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det == +1, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])

    R = Vt.T @ D @ U.T  # (3, 3)
    t = c_dst - R @ c_src  # (3,)

    return R, t


@dataclass
class UpdateResult:
    matched_ids: Optional[List[int]]
    # Mean displacement (dx, dy, dz) from initial points to the newly observed matched points.
    odometry: Optional[Odometry3]
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

        # Per-point odometry baseline at the time the point was first added.
        # When a new point joins mid-way, its birth_odom = current global odometry,
        # so its contribution to global odometry = (obs - init) + birth_odom,
        # which equals (incremental displacement since joining) + (historical odometry).
        self._birth_odom: np.ndarray = np.zeros((0, 3), dtype=float)

        # Latest odometry value (dx, dy, dz)
        self._odometry: Optional[Odometry3] = None

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
    def odometry(self) -> Optional[Odometry3]:
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
        # All points initialized together share the same birth_odom = current odometry
        base = np.array(self._odometry if self._odometry is not None else (0.0, 0.0, 0.0), dtype=float)
        self._birth_odom = np.tile(base.reshape(1, 3), (len(pts_world_ordered), 1))
        self._initialized = True

        if not remain_odometry:
            self._odometry = (0.0, 0.0, 0.0)

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
              - odometry: weighted mean displacement (dx, dy, dz) from initial
                points to current points for alive+visible points.
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

            # Reset miss count + increase reliability for matched ids
            for mid in matched_id_set:
                mid_i = int(mid)
                self._miss_counts[mid_i] = 0

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

            # Assign new IDs to genuinely new observations (unmatched AND far from all alive points)
            unmatched_obs = [oi for oi in range(len(obs_world)) if matched_ids[oi] < 0]
            for oi in unmatched_obs:
                if cur_alive.shape[0] == 0:
                    min_d = float("inf")
                else:
                    min_d = float(np.min(dmat[oi, :]))

                if min_d > self.match_threshold:
                    new_id = int(len(self._cur_world))

                    # Add to bank. For new points, initial==first seen (start odometry from join time).
                    self._cur_world = np.vstack([self._cur_world, obs_world[oi][None, :]])
                    self._init_world = np.vstack([self._init_world, obs_world[oi][None, :]])
                    self._lost_mask = np.concatenate([self._lost_mask, np.array([False], dtype=bool)])
                    self._miss_counts = np.concatenate([self._miss_counts, np.array([0], dtype=int)])

                    # birth_odom = current global odometry at the moment this point joins.
                    # Later: total_odom for this point = (obs - init) + birth_odom
                    base = np.array(self._odometry if self._odometry is not None else (0.0, 0.0, 0.0), dtype=float)
                    self._birth_odom = np.vstack([self._birth_odom, base.reshape(1, 3)])

                    # Keep consistent with this frame's commit
                    new_positions = np.vstack([new_positions, obs_world[oi][None, :]])

                    matched_ids[oi] = new_id
                    matched_id_set.add(new_id)

            # Use matched point pairs to estimate a rigid body transform (R, t)
            # via SVD, then apply it to predict positions of unmatched alive points.
            # With only 1 matched pair, falls back to pure translation (no rotation).
            if used_obs:
                src_pts = []  # previous positions of matched points
                dst_pts = []  # newly observed positions of matched points
                for oi in used_obs:
                    mid = matched_ids[oi]
                    if mid >= 0:
                        src_pts.append(self._cur_world[mid])   # position before this frame
                        dst_pts.append(obs_world[oi])          # position this frame

                if src_pts:
                    src_arr = np.stack(src_pts, axis=0)  # (K, 3)
                    dst_arr = np.stack(dst_pts, axis=0)  # (K, 3)

                    if src_arr.shape[0] >= 3:
                        # Enough points: estimate full rigid transform (R + t)
                        R_est, t_est = _estimate_rigid_transform(src_arr, dst_arr)
                    else:
                        # Too few points for reliable rotation: pure translation
                        R_est = np.eye(3, dtype=float)
                        t_est = (dst_arr - src_arr).mean(axis=0)

                    matched_id_set_obs = {matched_ids[oi] for oi in used_obs if matched_ids[oi] >= 0}
                    for idx in alive_ids:
                        idx_i = int(idx)
                        if idx_i in matched_id_set_obs:
                            continue
                        # Apply rigid transform to predict new position
                        p = new_positions[idx_i]
                        new_positions[idx_i] = R_est @ p + t_est

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

        # Odometry: mean displacement vector from matched observed points to initial positions,
        # plus each point's birth_odom to account for historical odometry before it joined.
        # total_odom[point] = (obs - init) + birth_odom
        # NOTE: position update uses only (obs - init), i.e. the incremental displacement
        # since the point joined; birth_odom is only added when computing the global odometry.
        odom_vecs: List[np.ndarray] = []
        for oi, mid in enumerate(matched_ids):
            if mid < 0:
                continue
            incremental = obs_world[oi] - self._init_world[mid]
            birth = self._birth_odom[mid] if mid < len(self._birth_odom) else np.zeros(3, dtype=float)
            odom_vecs.append(incremental + birth)

        if odom_vecs:
            mean_delta = np.mean(np.stack(odom_vecs, axis=0), axis=0)
            self._odometry = (float(mean_delta[0]), float(mean_delta[1]), float(mean_delta[2]))

        # If no points matched in this frame, do NOT declare overall tracking lost
        # (points may re-appear within max_missed_frames)
        if all(mid < 0 for mid in matched_ids):
            return UpdateResult(matched_ids=None, odometry=self._odometry, tracking_lost=False)

        return UpdateResult(matched_ids=matched_ids, odometry=self._odometry, tracking_lost=False)