import numpy as np

def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw (radians, ZYX) to rotation matrix."""
    cr = float(np.cos(roll)); sr = float(np.sin(roll))
    cp = float(np.cos(pitch)); sp = float(np.sin(pitch))
    cy = float(np.cos(yaw)); sy = float(np.sin(yaw))

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return (Rz @ Ry @ Rx)

def _R_to_rpy(R: np.ndarray):
    """Convert rotation matrix to roll/pitch/yaw (radians, ZYX)."""
    r20 = float(R[2, 0])
    r21 = float(R[2, 1])
    r22 = float(R[2, 2])
    r10 = float(R[1, 0])
    r00 = float(R[0, 0])

    pitch = float(np.arctan2(-r20, np.sqrt(r21 * r21 + r22 * r22)))
    yaw = float(np.arctan2(r10, r00))
    roll = float(np.arctan2(r21, r22))
    return roll, pitch, yaw

def _pose6_to_T(pose6) -> np.ndarray:
    """(x,y,z,roll,pitch,yaw) -> 4x4 homogeneous transform."""
    x, y, z, roll, pitch, yaw = map(float, pose6)
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = _rpy_to_R(roll, pitch, yaw)
    T[0:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T

def _T_to_pose6(T: np.ndarray):
    """4x4 homogeneous transform -> (x,y,z,roll,pitch,yaw)."""
    x, y, z = map(float, T[0:3, 3])
    roll, pitch, yaw = _R_to_rpy(T[0:3, 0:3])
    return (float(x), float(y), float(z), float(roll), float(pitch), float(yaw))

def _depth_to_meters(depth_array):
    # 16UC1 likely in millimeters, 32FC1 already meters
    if depth_array is None:
        return None
    if depth_array.dtype == np.uint16:
        return depth_array.astype(np.float32) * 0.001
    return depth_array.astype(np.float32)

def _median_depth_in_circle(depth_m: np.ndarray, u: int, v: int, radius_px: int = 6):
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    x0 = max(0, u - radius_px)
    x1 = min(w - 1, u + radius_px)
    y0 = max(0, v - radius_px)
    y1 = min(h - 1, v + radius_px)

    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    circle = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    patch = depth_m[y0:y1 + 1, x0:x1 + 1]
    vals = patch[circle]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))

def _segment_intersection(a, b):
    """Return intersection point (x,y) of two infinite lines through segments a,b.
    a/b: (x1,y1,x2,y2). Returns None if nearly parallel.
    """
    x1, y1, x2, y2 = map(float, a)
    x3, y3, x4, y4 = map(float, b)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)

def _wrap_angle_pi(ang: float) -> float:
    """Normalize angle to [0, pi)."""
    ang = float(ang) % np.pi
    return ang

def _fit_line_from_mask_points(mask_u8: np.ndarray, min_points: int = 200):
    """Fit a line from all mask points using PCA, then refine with inlier selection.

    Returns dict or None:
        {
        'point': (x0,y0),         # point on line (centroid)
        'dir': (dx,dy),           # unit direction
        'angle': angle_in_[0,pi),
        'length': approx_extent,  # projected mask extent along the line
        'seg': (x1,y1,x2,y2)       # representative segment endpoints (for intersection math)
        }
    """
    ys, xs = np.where(mask_u8 > 0)
    if xs.size < min_points:
        return None

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    c = pts.mean(axis=0)
    X = pts - c
    # PCA: principal direction
    cov = (X.T @ X) / max(1.0, float(X.shape[0]))
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))]
    dx, dy = float(d[0]), float(d[1])
    nrm = float(np.hypot(dx, dy))
    if nrm < 1e-6:
        return None
    dx /= nrm
    dy /= nrm

    # refine: keep inliers near the line
    # distance = |(p-c) x d|
    dist = np.abs(X[:, 0] * dy - X[:, 1] * dx)
    thr = max(2.0, float(np.percentile(dist, 50)) * 1.5)
    inliers = pts[dist <= thr]
    if inliers.shape[0] >= min_points // 2:
        c = inliers.mean(axis=0)
        X2 = inliers - c
        cov2 = (X2.T @ X2) / max(1.0, float(X2.shape[0]))
        w2, v2 = np.linalg.eigh(cov2)
        d2 = v2[:, int(np.argmax(w2))]
        dx, dy = float(d2[0]), float(d2[1])
        nrm2 = float(np.hypot(dx, dy))
        if nrm2 >= 1e-6:
            dx /= nrm2
            dy /= nrm2
            pts = inliers

    # compute extent along direction
    proj = (pts[:, 0] - c[0]) * dx + (pts[:, 1] - c[1]) * dy
    tmin = float(np.min(proj))
    tmax = float(np.max(proj))
    length = float(tmax - tmin)

    x1 = float(c[0] + tmin * dx)
    y1 = float(c[1] + tmin * dy)
    x2 = float(c[0] + tmax * dx)
    y2 = float(c[1] + tmax * dy)

    ang = _wrap_angle_pi(np.arctan2(dy, dx))

    return {
        'point': (float(c[0]), float(c[1])),
        'dir': (dx, dy),
        'angle': float(ang),
        'length': float(length),
        'seg': (x1, y1, x2, y2),
    }

def _project_world_points_to_full_pixels(points_world: np.ndarray, pose6_cam_in_world, cam_K):
    """Project Nx3 world points to full-image pixels.

    Args:
        points_world: (N,3) world points.
        pose6_cam_in_world: camera pose in world (x,y,z,roll,pitch,yaw).

    Returns:
        uv_full: (N,2) float pixel coordinates in full image.
        valid: (N,) bool mask: in front of camera and inside image bounds.
    """
    if points_world is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    if cam_K is None or pose6_cam_in_world is None:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    fx, fy, cx, cy, W, H = cam_K
    pts_w = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts_w.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    # world <- cam
    T_w_c = _pose6_to_T(pose6_cam_in_world)
    R_w_c = T_w_c[0:3, 0:3]
    t_w_c = T_w_c[0:3, 3]

    # cam <- world
    pts_c = (R_w_c.T @ (pts_w - t_w_c).T).T
    z = pts_c[:, 2]
    valid = z > 1e-6

    u = fx * (pts_c[:, 0] / z) + cx
    v = fy * (pts_c[:, 1] / z) + cy
    uv = np.stack([u, v], axis=1)

    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, valid