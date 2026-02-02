import cv2
import numpy as np
from .sam3node_utils import _project_world_points_to_full_pixels

def _draw_infinite_line_on_crop(frame_bgr: np.ndarray, seg, res: int, color, thickness: int = 2):
    """Draw an infinite line (defined by a segment) across the crop image boundaries."""
    x1, y1, x2, y2 = map(float, seg[:4])
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return

    points = []
    # intersect with x=0 and x=res-1
    if abs(dx) > 1e-6:
        for x in (0.0, float(res - 1)):
            t = (x - x1) / dx
            y = y1 + t * dy
            if 0.0 <= y <= float(res - 1):
                points.append((int(round(x)), int(round(y))))
    # intersect with y=0 and y=res-1
    if abs(dy) > 1e-6:
        for y in (0.0, float(res - 1)):
            t = (y - y1) / dy
            x = x1 + t * dx
            if 0.0 <= x <= float(res - 1):
                points.append((int(round(x)), int(round(y))))

    uniq = []
    for p in points:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) < 2:
        return

    best = None
    best_d = -1.0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
            if d > best_d:
                best_d = d
                best = (uniq[i], uniq[j])
    if best is None:
        return

    cv2.line(frame_bgr, best[0], best[1], color, thickness)


def _draw_mask_index(display_frame: np.ndarray, cX: int, cY: int, idx: int, color=(255, 255, 255)):

    text = str(idx)
    # outline for readability
    cv2.putText(display_frame, text, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(display_frame, text, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



def _draw_world_points_on_crop(
    display_frame: np.ndarray,
    points_world: np.ndarray,
    prefix: str,
    start_x: int,
    start_y: int,
    res: int,
    color,
    last_pose_cam2world,
    cam_K
):
    """Draw projected world points onto cropped image.

    Args:
        display_frame: crop (res x res) BGR image.
        points_world: (N,3) points in world.
        prefix: label prefix, e.g. 'I' or 'C'.
        start_x/start_y: crop origin in full image.
        res: crop size.
        color: BGR tuple.
    """
    if points_world is None or cam_K is None or last_pose_cam2world is None:
        return

    uv_full, valid = _project_world_points_to_full_pixels(points_world, last_pose_cam2world, cam_K)
    if uv_full.shape[0] == 0:
        return

    for pid in range(uv_full.shape[0]):
        if not bool(valid[pid]):
            continue
        u_full, v_full = float(uv_full[pid, 0]), float(uv_full[pid, 1])

        # full -> crop
        u = int(round(u_full - start_x))
        v = int(round(v_full - start_y))
        if not (0 <= u < res and 0 <= v < res):
            continue

        cv2.circle(display_frame, (u, v), 4, color, -1)
        cv2.putText(
            display_frame,
            f'{prefix}{pid}',
            (u + 6, v - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

def _draw_bank_world_points(display_frame: np.ndarray, start_x: int, start_y: int, res: int ,init_w,cur_w,last_pose_cam2world,cam_K):
    # Init: magenta, Current: green
    _draw_world_points_on_crop(display_frame, init_w, 'I', start_x, start_y, res, (255, 0, 255),last_pose_cam2world,cam_K)
    _draw_world_points_on_crop(display_frame, cur_w, 'C', start_x, start_y, res, (0, 255, 0),last_pose_cam2world,cam_K)