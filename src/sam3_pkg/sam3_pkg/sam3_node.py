import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from qwen_pkg_interfaces.msg import QwenResponse

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped

import numpy as np
from PIL import Image as PILImage
import time
from pathlib import Path

from cv_bridge import CvBridge
import cv2

import torch
from .sam3 import build_sam3_image_model
from .sam3.model.sam3_image_processor import Sam3Processor

from .memory_bank import MemoryBank

import threading
import queue

import yaml


class SAM3_Process(Node):
    def __init__(self):
        super().__init__('sam3_process')
        self.bridge = CvBridge() if CvBridge else None

        # Parameters for topics and prompt
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('tcp_pose_topic', '/robot1/tcp_pose')

        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.tcp_pose_topic = self.get_parameter('tcp_pose_topic').get_parameter_value().string_value


        # Subscribers
        self.sub_image = self.create_subscription(Image, self.color_topic, self.on_color, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.sub_qwen = self.create_subscription(QwenResponse, 'qwen_service/response', self.on_qwen, 10)
        # TCP pose now published as TransformStamped (translation + quaternion)
        self.sub_tcp_pose = self.create_subscription(TransformStamped, self.tcp_pose_topic, self.on_tcp_pose, 10)

        # Publisher (publish processed overlay image)
        self.pub_result = self.create_publisher(Image, 'camera/sam3_result', 10)
        # Publish each point position in world frame
        self.pub_points_position = self.create_publisher(PointStamped, 'points_position', 10)

        # Publishing thread state (drop old frames to avoid backlog)
        self._pub_queue: "queue.Queue[tuple[np.ndarray, Image] | None]" = queue.Queue(maxsize=1)
        self._pub_stop = threading.Event()
        self._pub_thread = threading.Thread(target=self._publish_worker, daemon=True)
        self._pub_thread.start()

        # State
        self.last_color_msg = None
        self.last_color = None
        self.last_depth = None
        self.depth_encoding = None
        self.cam_K = None  # fx, fy, cx, cy

        # camera pose (world): (x,y,z,roll,pitch,yaw)
        self.last_pose_cam2world = None

        # NEW: sync caches (ROS time in seconds)
        self._last_color_stamp_s: float | None = None
        self._last_depth_stamp_s: float | None = None
        self._last_pose_stamp_s: float | None = None
        # accept messages within this window
        self._sync_slop_s: float = 0.05

        # MemoryBank for stable IDs
        self.bank = MemoryBank(match_threshold=0.01, max_missed_frames=30000)
        self.bank_init = False
        self.total_3dpoints = []
        self.total_2dpoints = []
        self.odometry = None
        self.offset2edge = 0

        # SAM3 model lazily initialized on first frame
        self.model = None
        self.processor = None
        self.device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        self.last_description = ''
        self.last_type = ''
        self.last_reason = ''
        self.target_resolution = 480

        pkg_name = 'sam3_pkg'
        self.bpe_path = None
        self.checkpoint_path = None
        self.hand_eye_path = None
     

        file_path = Path(__file__).resolve()
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                bpe_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'bpe_simple_vocab_16e6.txt.gz'
                checkpoint_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'sam3.pt'
                hand_eye_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'hand_eye.yaml'
                break
        self.bpe_path = str(bpe_path)
        self.checkpoint_path = str(checkpoint_path)
        self.hand_eye_path = str(hand_eye_path)

        # Hand-eye transform (TCP->Camera). Translation unit: meters.
        self.T_tcp_cam = self._load_hand_eye_4x4(self.hand_eye_path)
        self.get_logger().info(f'Hand-eye matrix:\n{self.T_tcp_cam}')
        if self.T_tcp_cam is None:
            raise RuntimeError(f'Failed to load hand-eye matrix from: {self.hand_eye_path}')


        self._init_sam3_if_needed()

        # Time reference (first received color frame = 0s)
        self._t0_first_color_frame: float | None = None

    def destroy_node(self):
        # stop publisher thread before shutting down
        try:
            self._pub_stop.set()
            try:
                self._pub_queue.put_nowait(None)
            except Exception:
                pass
            if getattr(self, '_pub_thread', None) is not None:
                self._pub_thread.join(timeout=1.0)
        finally:
            return super().destroy_node()

    def _enqueue_publish(self, frame_bgr: np.ndarray, color_msg: Image):
        if frame_bgr is None or color_msg is None:
            return
        # keep only latest
        try:
            if self._pub_queue.full():
                try:
                    _ = self._pub_queue.get_nowait()
                except Exception:
                    pass
            self._pub_queue.put_nowait((frame_bgr, color_msg))
        except Exception:
            pass

    def _publish_worker(self):
        # publish images from background thread
        while not self._pub_stop.is_set():
            item = None
            try:
                item = self._pub_queue.get(timeout=0.2)
            except Exception:
                continue
            if item is None:
                continue
            frame_bgr, color_msg = item
            try:
                if self.bridge is None:
                    continue
                out_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
                out_msg.header = color_msg.header
                self.pub_result.publish(out_msg)
            except Exception as e:
                try:
                    self.get_logger().warn(f'publish sam3_result failed: {e}')
                except Exception:
                    pass

    def on_qwen(self, msg: QwenResponse):
        self.last_description = (msg.description or '').strip()
        self.last_type = (msg.type or '').strip()
        self.last_reason = (msg.reason or '').strip()

    def on_camera_info(self, msg: CameraInfo):
        # Extract intrinsics
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.cam_K = (fx, fy, cx, cy, msg.width, msg.height)

    def _stamp_to_sec(self, stamp) -> float:
        """Convert builtin_interfaces/Time to float seconds."""
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return float('nan')

    def _synced_ready(self) -> bool:
        """True if color/depth/pose are present and within timestamp tolerance."""
        if self._last_color_stamp_s is None or self._last_depth_stamp_s is None or self._last_pose_stamp_s is None:
            return False
        if self.last_color is None or self.last_color_msg is None:
            return False
        if self.last_depth is None:
            return False
        if self.last_pose_cam2world is None:
            return False

        t0 = self._last_color_stamp_s
        # all must be close to color stamp
        if abs(self._last_depth_stamp_s - t0) > self._sync_slop_s:
            return False
        if abs(self._last_pose_stamp_s - t0) > self._sync_slop_s:
            return False
        return True

    def on_color(self, msg: Image):
        if self.bridge is None:
            return
        # Start time at the first received frame, regardless of processing.
        if self._t0_first_color_frame is None:
            self._t0_first_color_frame = time.time()

        self.last_color_msg = msg
        self._last_color_stamp_s = self._stamp_to_sec(msg.header.stamp)
        self.last_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Only process when depth+pose are aligned to this color frame
        if self._synced_ready():
            self.try_process_and_publish()

    def on_depth(self, msg: Image):
        if self.bridge is None:
            return
        # Keep encoding to infer scale
        self.depth_encoding = msg.encoding
        self._last_depth_stamp_s = self._stamp_to_sec(msg.header.stamp)
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.last_depth = depth

    def _load_hand_eye_4x4(self, path: str):
        """Load 4x4 hand-eye matrix from YAML (meters).

        Expected format:
          T_tcp_cam:
            - [r11,r12,r13,tx]
            - [r21,r22,r23,ty]
            - [r31,r32,r33,tz]
            - [0,0,0,1]
        """
        try:
            p = Path(path)
            if not p.exists():
                return None

            data = yaml.safe_load(p.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or 'T_tcp_cam' not in data:
                return None

            T = np.array(data['T_tcp_cam'], dtype=np.float64)
            if T.shape != (4, 4):
                return None

            # normalize last row
            T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            return T
        except Exception as e:
            try:
                self.get_logger().warn(f'load hand-eye failed: {e}')
            except Exception:
                pass
            return None

    def on_tcp_pose(self, msg: TransformStamped):
        # Expected TransformStamped with translation (m) and rotation quaternion (x,y,z,w)
        try:
            # cache timestamp for sync
            self._last_pose_stamp_s = self._stamp_to_sec(msg.header.stamp)

            t = msg.transform.translation
            q = msg.transform.rotation

            tx = float(t.x)
            ty = float(t.y)
            tz = float(t.z)

            qx = float(q.x)
            qy = float(q.y)
            qz = float(q.z)
            qw = float(q.w)

            # build rotation matrix from quaternion
            # Reference: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
            xx = qx * qx
            yy = qy * qy
            zz = qz * qz
            xy = qx * qy
            xz = qx * qz
            yz = qy * qz
            wx = qw * qx
            wy = qw * qy
            wz = qw * qz

            R = np.array([
                [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),           2.0 * (xz + wy)],
                [      2.0 * (xy + wz),   1.0 - 2.0 * (xx + zz),         2.0 * (yz - wx)],
                [      2.0 * (xz - wy),         2.0 * (yz + wx),     1.0 - 2.0 * (xx + yy)],
            ], dtype=np.float64)

            T_base_tcp = np.eye(4, dtype=np.float64)
            T_base_tcp[0:3, 0:3] = R
            T_base_tcp[0:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

            # Always apply hand-eye: base->cam = (base->tcp) @ (tcp->cam)
            T_base_cam = T_base_tcp @ self.T_tcp_cam
            self.last_pose_cam2world = self._T_to_pose6(T_base_cam)
        except Exception:
            return
        
    def _rpy_to_R(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll/pitch/yaw (radians, ZYX) to rotation matrix."""
        cr = float(np.cos(roll)); sr = float(np.sin(roll))
        cp = float(np.cos(pitch)); sp = float(np.sin(pitch))
        cy = float(np.cos(yaw)); sy = float(np.sin(yaw))

        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        return (Rz @ Ry @ Rx)

    def _R_to_rpy(self, R: np.ndarray):
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

    def _pose6_to_T(self, pose6) -> np.ndarray:
        """(x,y,z,roll,pitch,yaw) -> 4x4 homogeneous transform."""
        x, y, z, roll, pitch, yaw = map(float, pose6)
        T = np.eye(4, dtype=np.float64)
        T[0:3, 0:3] = self._rpy_to_R(roll, pitch, yaw)
        T[0:3, 3] = np.array([x, y, z], dtype=np.float64)
        return T

    def _T_to_pose6(self, T: np.ndarray):
        """4x4 homogeneous transform -> (x,y,z,roll,pitch,yaw)."""
        x, y, z = map(float, T[0:3, 3])
        roll, pitch, yaw = self._R_to_rpy(T[0:3, 0:3])
        return (float(x), float(y), float(z), float(roll), float(pitch), float(yaw))

    def track_ID(self, display_frame):
        # draw mask index (mask order)
        fx, fy, cx, cy, W, H = self.cam_K
        image_size = min(W, H) - self .offset2edge
        res1 = None
        if not self.bank_init:
            id_map = self.bank.initialize(self.total_3dpoints, self.last_pose_cam2world, (fx, fy, cx, cy), (image_size, image_size))
            self.bank_init = True
        else:
            res1 = self.bank.update(self.total_3dpoints, self.last_pose_cam2world)
            id_map = res1.matched_ids

        for i, (cX, cY) in enumerate(self.total_2dpoints):
            if id_map is not None and i < len(id_map):
                self._draw_mask_index(display_frame, int(cX), int(cY), int(id_map[i]))
            else:
                self._draw_mask_index(display_frame, int(cX), int(cY), i)

        # Publish world coordinates for each point (frame_id = id)
        try:
            if self.last_color_msg is not None:
                self._publish_points_world(id_map, self.last_color_msg.header)
        except Exception:
            pass
        
        if res1 is not None:
            self.odometry = res1.odometry
            if self.odometry is not None:
                odom = float(res1.odometry)
                odom = round(odom, 4) 
                cv2.putText(display_frame, f'Odometry: {odom}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, f'Odometry: {self.odometry}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            

    def _init_sam3_if_needed(self):
        if self.model is not None:
            return
        if build_sam3_image_model is None:
            self.get_logger().warn('SAM3 not available; skipping segmentation.')
            return
        try:
            self.model = build_sam3_image_model(checkpoint_path=self.checkpoint_path, bpe_path=self.bpe_path, image_size=self.target_resolution)
            self.model.to(self.device)
            self.model.eval()
            self.processor = Sam3Processor(self.model, resolution=self.target_resolution, confidence_threshold=0.4)
            self.get_logger().info(f'SAM3 initialized on {self.device} with resolution {self.target_resolution}')
        except Exception as e:
            self.get_logger().error(f'Failed to init SAM3: {e}')
            self.model = None
            self.processor = None

    def _depth_to_meters(self, depth_array):
        # 16UC1 likely in millimeters, 32FC1 already meters
        if depth_array is None:
            return None
        if depth_array.dtype == np.uint16:
            return depth_array.astype(np.float32) * 0.001
        return depth_array.astype(np.float32)

    def _median_depth_in_circle(self, depth_m: np.ndarray, u: int, v: int, radius_px: int = 6):
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

    def _segment_intersection(self, a, b):
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


    def _draw_infinite_line_on_crop(self, frame_bgr: np.ndarray, seg, res: int, color, thickness: int = 2):
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

    def _wrap_angle_pi(self, ang: float) -> float:
        """Normalize angle to [0, pi)."""
        ang = float(ang) % np.pi
        return ang

    def _fit_line_from_mask_points(self, mask_u8: np.ndarray, min_points: int = 200):
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

        ang = self._wrap_angle_pi(np.arctan2(dy, dx))

        return {
            'point': (float(c[0]), float(c[1])),
            'dir': (dx, dy),
            'angle': float(ang),
            'length': float(length),
            'seg': (x1, y1, x2, y2),
        }

    def _draw_mask_index(self, display_frame: np.ndarray, cX: int, cY: int, idx: int, color=(255, 255, 255)):

        text = str(idx)
        # outline for readability
        cv2.putText(display_frame, text, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(display_frame, text, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _project_world_points_to_full_pixels(self, points_world: np.ndarray, pose6_cam_in_world):
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
        if self.cam_K is None or pose6_cam_in_world is None:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

        fx, fy, cx, cy, W, H = self.cam_K
        pts_w = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if pts_w.size == 0:
            return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

        # world <- cam
        T_w_c = self._pose6_to_T(pose6_cam_in_world)
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

    def _draw_world_points_on_crop(
        self,
        display_frame: np.ndarray,
        points_world: np.ndarray,
        prefix: str,
        start_x: int,
        start_y: int,
        res: int,
        color,
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
        if points_world is None:
            return
        if self.last_pose_cam2world is None or self.cam_K is None:
            return

        uv_full, valid = self._project_world_points_to_full_pixels(points_world, self.last_pose_cam2world)
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

    def _draw_bank_world_points(self, display_frame: np.ndarray, start_x: int, start_y: int, res: int):
        """Draw both init and current world points from MemoryBank on the crop."""
        if not getattr(self, 'bank_init', False):
            return
        try:
            init_w = self.bank.init_world_points
            cur_w = self.bank.current_world_points
        except Exception:
            return

        # Init: magenta, Current: green
        self._draw_world_points_on_crop(display_frame, init_w, 'I', start_x, start_y, res, (255, 0, 255))
        self._draw_world_points_on_crop(display_frame, cur_w, 'C', start_x, start_y, res, (0, 255, 0))

    def try_process_and_publish(self):

        if self.last_color is None or self.last_color_msg is None:
            return

        # If prompt/type not ready, just publish raw color
        if self.last_type == '':
            self._enqueue_publish(self.last_color, self.last_color_msg)
            return

        frame_bgr = self.last_color.copy()
        h, w, _ = frame_bgr.shape

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.target_resolution
        start_x = (w - res) // 2
        start_y = (h - res) // 2
        image_cropped = frame_rgb[start_y:start_y+res, start_x:start_x+res]
        pil_image = PILImage.fromarray(image_cropped)

        t0 = time.time()
        masks = None
        try:
            state = self.processor.set_image(pil_image)
            state = self.processor.set_text_prompt(prompt=self.last_description, state=state)
            masks = state.get('masks')
            if masks is not None:
                masks = masks.cpu().numpy()  # [N,1,H,W]
        except Exception as e:
            self.get_logger().warn(f'SAM3 inference error: {e}')
            masks = None

        # Time since first received color frame
        if self._t0_first_color_frame is None:
            # Fallback (shouldn't happen unless on_color wasn't called)
            self._t0_first_color_frame = time.time()
        frame_time_s = float(time.time() - self._t0_first_color_frame)

        depth_m = self._depth_to_meters(self.last_depth)

        fx, fy, cx, cy, W, H = self.cam_K

        display_frame = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)

        self.total_2dpoints = []
        self.total_3dpoints = []

        if self.last_type == '0':  # points (solid point) -> sample depth INSIDE mask
            if masks is not None and masks.shape[0] > 0 and depth_m is not None and self.cam_K is not None:
                for i in range(masks.shape[0]):
                    mask = (masks[i, 0].astype(np.uint8) * 255)

                    # # log mask size (pixel area)
                    # try:
                    #     area_px = int(np.count_nonzero(mask))
                    #     self.get_logger().info(f'[SAM3] type=0 mask[{i}] area_px={area_px}')
                    # except Exception:
                    #     pass

                    # visualize mask (red)
                    color = (0, 0, 255)
                    colored_mask = np.zeros_like(display_frame, dtype=np.uint8)
                    colored_mask[:, :, 0] = mask * (color[0] / 255)
                    colored_mask[:, :, 1] = mask * (color[1] / 255)
                    colored_mask[:, :, 2] = mask * (color[2] / 255)
                    display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.35, 0)

                    M = cv2.moments(mask)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # inside-mask depth (map crop coords -> full coords)
                    inside_y, inside_x = np.where(mask > 0)
                    inside_y_full = inside_y + start_y
                    inside_x_full = inside_x + start_x

                    valid_depths = []
                    for rx, ry in zip(inside_x_full, inside_y_full):
                        if 0 <= ry < depth_m.shape[0] and 0 <= rx < depth_m.shape[1]:
                            d = float(depth_m[ry, rx])
                            if d > 0 and np.isfinite(d):
                                valid_depths.append(d)

                    label = f'({cX},{cY})'
                    if valid_depths:
                        Z = float(np.median(valid_depths))
                        u_full = cX + start_x
                        v_full = cY + start_y
                        X = (u_full - cx) / fx * Z
                        Y = (v_full - cy) / fy * Z
                        label = f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m'

                        self.total_2dpoints.append((cX, cY))
                        self.total_3dpoints.append((X, Y, Z))
                    else:
                        label += ' No Depth'

                    cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)

                self.track_ID(display_frame)

        elif self.last_type == '1':  # lines
            if masks is not None and masks.shape[0] > 0 and depth_m is not None and self.cam_K is not None:
                # log mask size (pixel area)
                # for i in range(masks.shape[0]):
                    # try:
                    #     m_area = int(np.count_nonzero(masks[i, 0]))
                    #     self.get_logger().info(f'[SAM3] type=1 mask[{i}] area_px={m_area}')
                    # except Exception:
                    #     pass

                # (A) show ALL masks with a light overlay
                for i in range(masks.shape[0]):
                    m = (masks[i, 0].astype(np.uint8) * 255)
                    colored = np.zeros_like(display_frame, dtype=np.uint8)
                    colored[:, :, 0] = m
                    colored[:, :, 1] = (m * 0.6).astype(np.uint8)
                    colored[:, :, 2] = (m * 0.2).astype(np.uint8)
                    display_frame = cv2.addWeighted(display_frame, 1.0, colored, 0.18, 0)

                # (B) fit ONE line per mask using all mask points (PCA)
                fitted = []
                for i in range(masks.shape[0]):
                    mask_u8 = (masks[i, 0].astype(np.uint8) * 255)
                    line = self._fit_line_from_mask_points(mask_u8)
                    if line is None:
                        continue
                    fitted.append(line)

                if len(fitted) >= 2:
                    # (C) group by orientation and keep the longest line in each direction
                    bins = []  # each: [rep_angle, best_line]
                    ang_thresh = np.deg2rad(15.0)
                    for ln in sorted(fitted, key=lambda x: -x['length']):
                        placed = False
                        for b in bins:
                            rep = b[0]
                            d = abs(ln['angle'] - rep)
                            d = min(d, np.pi - d)
                            if d < ang_thresh:
                                placed = True
                                if ln['length'] > b[1]['length']:
                                    b[0] = ln['angle']
                                    b[1] = ln
                                break
                        if not placed:
                            bins.append([ln['angle'], ln])
                    candidates = [b[1] for b in bins]

                    if len(candidates) >= 2:
                        candidates.sort(key=lambda x: -x['length'])

                        # (D) choose the best non-parallel pair (prefer longer)
                        best_pair = None
                        best_score = -1.0
                        for i in range(len(candidates)):
                            for j in range(i + 1, len(candidates)):
                                a = candidates[i]
                                b = candidates[j]
                                dang = abs(a['angle'] - b['angle'])
                                dang = min(dang, np.pi - dang)
                                if dang < np.deg2rad(20.0):
                                    continue
                                score = a['length'] + b['length']
                                if score > best_score:
                                    best_score = score
                                    best_pair = (a, b)
                        if best_pair is not None:
                            a, b = best_pair

                            # (E) intersection of the two fitted lines (use their representative seg)
                            inter = self._segment_intersection(a['seg'], b['seg'])
                            if inter is not None:
                                cX, cY = inter
                                cX = int(np.clip(round(cX), 0, res - 1))
                                cY = int(np.clip(round(cY), 0, res - 1))

                                u_full = cX + start_x
                                v_full = cY + start_y
                                Z = self._median_depth_in_circle(depth_m, u_full, v_full, radius_px=8)

                                label = f'({cX},{cY})'
                                if Z is not None:
                                    X = (u_full - cx) / fx * Z
                                    Y = (v_full - cy) / fy * Z
                                    label = f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m'

                                    self.total_2dpoints.append((cX, cY))
                                    self.total_3dpoints.append((X, Y, Z))
                                else:
                                    label += ' No Depth'

                                # (F) draw infinite lines
                                self._draw_infinite_line_on_crop(display_frame, a['seg'], res, (0, 255, 255), thickness=2)
                                self._draw_infinite_line_on_crop(display_frame, b['seg'], res, (255, 255, 0), thickness=2)

                                cv2.circle(display_frame, (cX, cY), 6, (0, 0, 255), -1)
                self.track_ID(display_frame)

        elif self.last_type == '2':  # holes
            if masks is not None and masks.shape[0] > 0:
                for i in range(masks.shape[0]):
                    mask = masks[i, 0]  # [resolution, resolution]

                    # log mask size (pixel area)
                    # try:
                    #     area_px = int(np.count_nonzero(mask))
                    #     self.get_logger().info(f'[SAM3] type=2 mask[{i}] area_px={area_px}')
                    # except Exception:
                    #     pass

                    mask = mask.astype(np.uint8) * 255
                    color = (255, 0, 0)

                    colored_mask = np.zeros_like(display_frame, dtype=np.uint8)
                    colored_mask[:, :, 0] = mask * (color[0] / 255)
                    colored_mask[:, :, 1] = mask * (color[1] / 255)
                    colored_mask[:, :, 2] = mask * (color[2] / 255)

                    display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.5, 0)

                    # 计算并绘制中心点
                    M = cv2.moments(mask)
                    if M["m00"] != 0 and depth_m is not None and self.cam_K is not None:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        kernel = np.ones((5, 5), np.uint8)
                        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
                        rim_mask = cv2.subtract(dilated_mask, mask)

                        rim_color = (0, 255, 0)
                        colored_rim = np.zeros_like(display_frame, dtype=np.uint8)
                        colored_rim[:, :, 0] = rim_mask * (rim_color[0] / 255)
                        colored_rim[:, :, 1] = rim_mask * (rim_color[1] / 255)
                        colored_rim[:, :, 2] = rim_mask * (rim_color[2] / 255)
                        display_frame = cv2.addWeighted(display_frame, 1.0, colored_rim, 0.5, 0)

                        rim_y, rim_x = np.where(rim_mask > 0)
                        rim_y_full = rim_y + start_y
                        rim_x_full = rim_x + start_x

                        valid_depths = []
                        for rx, ry in zip(rim_x_full, rim_y_full):
                            if 0 <= ry < depth_m.shape[0] and 0 <= rx < depth_m.shape[1]:
                                d = depth_m[ry, rx]
                                if d > 0 and np.isfinite(d):
                                    valid_depths.append(float(d))

                        label = f'({cX},{cY})'
                        if valid_depths:
                            Z = float(np.median(valid_depths))
                            u_full = cX + start_x
                            v_full = cY + start_y
                            X = (u_full - cx) / fx * Z
                            Y = (v_full - cy) / fy * Z
                            label = f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m'

                            self.total_2dpoints.append((cX, cY))
                            self.total_3dpoints.append((X, Y, Z))
                        else:
                            label += ' No Depth'

                        cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)
                self.track_ID(display_frame)

        # Overlay time and description
        # cv2.putText(
        #     display_frame,
        #     f'Time: {frame_time_s:.3f}s',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 255),
        #     2,
        # )
        cv2.putText(display_frame, f'Description: {self.last_description}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # NEW: draw all init/current world points (when bank has been initialized)
        self._draw_bank_world_points(display_frame, start_x, start_y, res)

        self._enqueue_publish(display_frame, self.last_color_msg)

    def _publish_points_world(self, id_map, color_header):
        """Publish each point's world coordinates as PointStamped.

        - topic: points_position
        - header.frame_id: point id (string)
        - point: world coordinates (meters)
        """
        if self.cam_K is None:
            return
        if self.last_pose_cam2world is None:
            return
        if not self.total_3dpoints:
            return

        try:
            T_base_cam = self._pose6_to_T(self.last_pose_cam2world)  # base/world <- cam
        except Exception:
            return

        for i, p_cam in enumerate(self.total_3dpoints):
            if p_cam is None:
                continue
            try:
                Xc, Yc, Zc = map(float, p_cam)
            except Exception:
                continue

            pid = None
            if id_map is not None and i < len(id_map):
                pid = id_map[i]
            else:
                pid = i

            p_cam_h = np.array([Xc, Yc, Zc, 1.0], dtype=np.float64)
            p_w = (T_base_cam @ p_cam_h).reshape(-1)

            msg = PointStamped()
            msg.header = color_header
            msg.header.frame_id = str(int(pid))
            msg.point.x = float(p_w[0])
            msg.point.y = float(p_w[1])
            msg.point.z = float(p_w[2])
            self.pub_points_position.publish(msg)

def main():
    rclpy.init()
    node = SAM3_Process()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
