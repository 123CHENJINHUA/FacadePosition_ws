import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped

import numpy as np
import time
from pathlib import Path

from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO

from .memory_bank import MemoryBank
from .utils import _T_to_pose6, _pose6_to_T, _depth_to_meters, _median_depth_in_circle, _segment_intersection, _fit_line_from_mask_points
from .vis import _draw_infinite_line_on_crop, _draw_bank_world_points, _draw_mask_index

import threading
import queue

import yaml

# YOLO class name -> processing type
#   'points'     -> '0'  (solid point, depth inside mask)
#   'black line' -> '1'  (line, fit + intersect)
#   'holes'      -> '2'  (hole, depth on rim)
# NOTE: update this dict to match your model's actual class names (check YOLO classes log on startup)
YOLO_CLASS_TYPE = {
    'points': '0',
    'black line': '1',
    'holes': '2',
    'hole': '2',
    'point': '0',
    'line': '1',
}


class YOLO_Process(Node):
    def __init__(self):
        super().__init__('yolo_process')
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
        # TCP pose now published as TransformStamped (translation + quaternion)
        self.sub_tcp_pose = self.create_subscription(TransformStamped, self.tcp_pose_topic, self.on_tcp_pose, 10)

        # Publisher (publish processed overlay image)
        self.pub_result = self.create_publisher(Image, 'camera/sam3_result', 10)
        # Publish each point position in world frame
        self.pub_points_position = self.create_publisher(PointStamped, 'points_position', 10)
        # Publish odometry displacement (dx,dy,dz) as PointStamped
        self.pub_odometry = self.create_publisher(PointStamped, 'yolo/odometry', 10)

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
        self.bank = MemoryBank(match_threshold=0.03, max_missed_frames=300)
        self.bank_init = False
        self.total_3dpoints = []
        self.total_2dpoints = []
        self.odometry = None
        self.offset2edge = 0

        # NEW: request to reinitialize MemoryBank init points on next frame
        self.reinit_bank_pending: bool = False

        # YOLO model
        self.model = None
        self.conf = 0.5
        self.target_resolution = 480

        pkg_name = 'yolo_pkg'
        self.checkpoint_path = None
        self.hand_eye_path = None

        file_path = Path(__file__).resolve()
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                checkpoint_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'best.pt'
                hand_eye_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'hand_eye.yaml'
                break
        self.checkpoint_path = str(checkpoint_path)
        self.hand_eye_path = str(hand_eye_path)

        # Hand-eye transform (TCP->Camera). Translation unit: meters.
        self.T_tcp_cam = self._load_hand_eye_4x4(self.hand_eye_path)
        self.get_logger().info(f'Hand-eye matrix:\n{self.T_tcp_cam}')
        if self.T_tcp_cam is None:
            raise RuntimeError(f'Failed to load hand-eye matrix from: {self.hand_eye_path}')

        self._init_yolo_if_needed()

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
                    self.get_logger().warn(f'publish yolo_result failed: {e}')
                except Exception:
                    pass

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
            self.last_pose_cam2world = _T_to_pose6(T_base_cam)
        except Exception:
            return
        
    

    def track_ID(self, display_frame, pts_2d: list, pts_3d: list):
        """Assign stable IDs to the given 2D/3D point set and draw them."""
        # draw mask index (mask order)
        fx, fy, cx, cy, W, H = self.cam_K
        image_size = min(W, H) - self.offset2edge
        res1 = None

        # If requested, reinitialize MemoryBank using current points as new baseline
        if self.reinit_bank_pending:
            id_map = self.bank.initialize(pts_3d, self.last_pose_cam2world, (fx, fy, cx, cy), (image_size, image_size), remain_odometry=True)
            self.get_logger().info('MemoryBank reinitialized.')
            self.bank_init = True
            self.reinit_bank_pending = False

        elif not self.bank_init:
            id_map = self.bank.initialize(pts_3d, self.last_pose_cam2world, (fx, fy, cx, cy), (image_size, image_size))
            self.get_logger().info('MemoryBank initialized.')
            self.bank_init = True
        else:
            res1 = self.bank.update(pts_3d, self.last_pose_cam2world)
            id_map = res1.matched_ids

        for i, (cX, cY) in enumerate(pts_2d):
            if id_map is not None and i < len(id_map) and id_map[i] >= 0:
                _draw_mask_index(display_frame, int(cX), int(cY), int(id_map[i]))

        # Publish world coordinates for each point (frame_id = id)
        try:
            if self.last_color_msg is not None:
                self._publish_points_world(id_map, pts_3d, self.last_color_msg.header)
        except Exception:
            pass

        if res1 is not None:
            self.odometry = res1.odometry
            if self.odometry is not None:
                dx, dy, dz = self.odometry
                dx = round(float(dx), 4)
                dy = round(float(dy), 4)
                dz = round(float(dz), 4)

                # Publish odometry
                try:
                    odom_msg = PointStamped()
                    if self.last_color_msg is not None:
                        odom_msg.header = self.last_color_msg.header
                    odom_msg.point.x = float(dx)
                    odom_msg.point.y = float(dy)
                    odom_msg.point.z = float(dz)
                    self.pub_odometry.publish(odom_msg)
                except Exception:
                    pass

                cv2.putText(
                    display_frame,
                    f'Odometry: dx={dx} dy={dy} dz={dz}',
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    def _init_yolo_if_needed(self):
        if self.model is not None:
            return
        try:
            self.model = YOLO(self.checkpoint_path)
            self.get_logger().info(f'YOLO model loaded from: {self.checkpoint_path}')
            self.get_logger().info(f'YOLO classes: {self.model.names}')
        except Exception as e:
            self.get_logger().error(f'Failed to init YOLO: {e}')
            self.model = None

    def try_process_and_publish(self):

        if self.last_color is None or self.last_color_msg is None:
            return

        if self.model is None:
            self._enqueue_publish(self.last_color, self.last_color_msg)
            return

        frame_bgr = self.last_color.copy()
        h, w, _ = frame_bgr.shape

        res = self.target_resolution
        start_x = (w - res) // 2
        start_y = (h - res) // 2
        image_cropped_bgr = frame_bgr[start_y:start_y + res, start_x:start_x + res]

        # Run YOLO segmentation on the cropped image
        try:
            results = self.model.predict(
                source=image_cropped_bgr,
                conf=self.conf,
                verbose=False
            )
        except Exception as e:
            self.get_logger().warn(f'YOLO inference error: {e}')
            self._enqueue_publish(image_cropped_bgr, self.last_color_msg)
            return

        result = results[0]

        # Time since first received color frame
        if self._t0_first_color_frame is None:
            self._t0_first_color_frame = time.time()

        depth_m = _depth_to_meters(self.last_depth)
        fx, fy, cx, cy, W, H = self.cam_K

        display_frame = image_cropped_bgr.copy()

        self.total_2dpoints = []
        self.total_3dpoints = []

        # Group detections by class type
        masks_by_type: dict[str, list[np.ndarray]] = {'0': [], '1': [], '2': []}

        if result.masks is not None and len(result.boxes) > 0:
            for i, cls_tensor in enumerate(result.boxes.cls):
                cls_id = int(cls_tensor.item())
                cls_name = result.names.get(cls_id, '')
                ptype = YOLO_CLASS_TYPE.get(cls_name)
                if ptype is None:
                    continue
                m = result.masks.data[i].cpu().numpy()  # (H, W) float in [0,1]
                if m.shape != (res, res):
                    m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
                masks_by_type[ptype].append(m)

        # ---- type '0': points (solid point) -> depth inside mask ----
        if masks_by_type['0']:
            masks = np.stack(masks_by_type['0'], axis=0)  # (N, H, W)
            pts_2d: list = []
            pts_3d: list = []
            if depth_m is not None and self.cam_K is not None:
                for i in range(masks.shape[0]):
                    mask = (masks[i] > 0.5).astype(np.uint8) * 255

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

                    inside_y, inside_x = np.where(mask > 0)
                    inside_y_full = inside_y + start_y
                    inside_x_full = inside_x + start_x

                    valid_depths = []
                    for rx, ry in zip(inside_x_full, inside_y_full):
                        if 0 <= ry < depth_m.shape[0] and 0 <= rx < depth_m.shape[1]:
                            d = float(depth_m[ry, rx])
                            if d > 0 and np.isfinite(d):
                                valid_depths.append(d)

                    if valid_depths:
                        Z = float(np.median(valid_depths))
                        u_full = cX + start_x
                        v_full = cY + start_y
                        X = (u_full - cx) / fx * Z
                        Y = (v_full - cy) / fy * Z
                        pts_2d.append((cX, cY))
                        pts_3d.append((X, Y, Z))

                    cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)

            if pts_3d:
                self.total_2dpoints = pts_2d
                self.total_3dpoints = pts_3d
                self.track_ID(display_frame, pts_2d, pts_3d)

        # ---- type '1': black lines -> PCA fit + intersection ----
        if masks_by_type['1']:
            masks = np.stack(masks_by_type['1'], axis=0)  # (N, H, W)
            pts_2d = []
            pts_3d = []
            if depth_m is not None and self.cam_K is not None:
                # (A) show all masks with a light overlay
                for i in range(masks.shape[0]):
                    m = ((masks[i] > 0.5).astype(np.uint8) * 255)
                    colored = np.zeros_like(display_frame, dtype=np.uint8)
                    colored[:, :, 0] = m
                    colored[:, :, 1] = (m * 0.6).astype(np.uint8)
                    colored[:, :, 2] = (m * 0.2).astype(np.uint8)
                    display_frame = cv2.addWeighted(display_frame, 1.0, colored, 0.18, 0)

                # (B) fit ONE line per mask using PCA
                fitted = []
                for i in range(masks.shape[0]):
                    mask_u8 = ((masks[i] > 0.5).astype(np.uint8) * 255)
                    line = _fit_line_from_mask_points(mask_u8)
                    if line is None:
                        continue
                    fitted.append(line)

                if len(fitted) >= 2:
                    # (C) group by orientation and keep the longest line per direction
                    bins = []
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

                        # (D) choose the best non-parallel pair
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
                            inter = _segment_intersection(a['seg'], b['seg'])
                            if inter is not None:
                                cX, cY = inter
                                cX = int(np.clip(round(cX), 0, res - 1))
                                cY = int(np.clip(round(cY), 0, res - 1))

                                u_full = cX + start_x
                                v_full = cY + start_y
                                Z = _median_depth_in_circle(depth_m, u_full, v_full, radius_px=15)

                                if Z is not None:
                                    X = (u_full - cx) / fx * Z
                                    Y = (v_full - cy) / fy * Z
                                    pts_2d.append((cX, cY))
                                    pts_3d.append((X, Y, Z))

                                _draw_infinite_line_on_crop(display_frame, a['seg'], res, (0, 255, 255), thickness=2)
                                _draw_infinite_line_on_crop(display_frame, b['seg'], res, (255, 255, 0), thickness=2)
                                cv2.circle(display_frame, (cX, cY), 6, (0, 0, 255), -1)

            if pts_3d:
                self.total_2dpoints = pts_2d
                self.total_3dpoints = pts_3d
                self.track_ID(display_frame, pts_2d, pts_3d)

        # ---- type '2': holes -> depth on rim ----
        if masks_by_type['2']:
            masks = np.stack(masks_by_type['2'], axis=0)  # (N, H, W)
            pts_2d = []
            pts_3d = []
            for i in range(masks.shape[0]):
                mask = ((masks[i] > 0.5).astype(np.uint8) * 255)
                color = (255, 0, 0)

                colored_mask = np.zeros_like(display_frame, dtype=np.uint8)
                colored_mask[:, :, 0] = mask * (color[0] / 255)
                colored_mask[:, :, 1] = mask * (color[1] / 255)
                colored_mask[:, :, 2] = mask * (color[2] / 255)
                display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.5, 0)

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

                    if valid_depths:
                        Z = float(np.median(valid_depths))
                        u_full = cX + start_x
                        v_full = cY + start_y
                        X = (u_full - cx) / fx * Z
                        Y = (v_full - cy) / fy * Z
                        pts_2d.append((cX, cY))
                        pts_3d.append((X, Y, Z))

                    cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)

            if pts_3d:
                self.total_2dpoints = pts_2d
                self.total_3dpoints = pts_3d
                self.track_ID(display_frame, pts_2d, pts_3d)

        self._enqueue_publish(display_frame, self.last_color_msg)

    def _publish_points_world(self, id_map, pts_3d: list, color_header):
        """Publish each point's world coordinates as PointStamped."""
        if self.cam_K is None:
            return
        if self.last_pose_cam2world is None:
            return
        if not pts_3d:
            return

        try:
            T_base_cam = _pose6_to_T(self.last_pose_cam2world)
        except Exception:
            return

        for i, p_cam in enumerate(pts_3d):
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
    node = YOLO_Process()
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
