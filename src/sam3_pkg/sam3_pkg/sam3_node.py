import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
from qwen_pkg_interfaces.msg import QwenResponse

import numpy as np
from PIL import Image as PILImage
import time
from pathlib import Path


from cv_bridge import CvBridge
import cv2

import torch
from .sam3 import build_sam3_image_model
from .sam3.model.sam3_image_processor import Sam3Processor


class SAM3_Process(Node):
    def __init__(self):
        super().__init__('sam3_process')
        self.bridge = CvBridge() if CvBridge else None

        # Parameters for topics and prompt
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        self.color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        

        # Subscribers
        self.sub_image = self.create_subscription(Image, self.color_topic, self.on_color, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, 10)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.sub_qwen = self.create_subscription(QwenResponse, 'qwen_service/response', self.on_qwen, 10)

        # State
        self.last_color = None
        self.last_depth = None
        self.depth_encoding = None
        self.cam_K = None  # fx, fy, cx, cy

        # SAM3 model lazily initialized on first frame
        self.model = None
        self.processor = None
        self.device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        self.last_description = ''
        self.last_type = ''
        self.last_reason = ''
        self.target_resolution = 480

        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

        pkg_name = 'sam3_pkg'
        self.bpe_path = None
        self.checkpoint_path = None

        file_path = Path(__file__).resolve()
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                bpe_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'bpe_simple_vocab_16e6.txt.gz'
                checkpoint_path = ws_root / 'src' / pkg_name / pkg_name / 'weight' / 'sam3.pt'
                break
        self.bpe_path = str(bpe_path)
        self.checkpoint_path = str(checkpoint_path)

        self._init_sam3_if_needed()

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

    def on_color(self, msg: Image):
        if self.bridge is None:
            return
        self.last_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_process_and_show()

    def on_depth(self, msg: Image):
        if self.bridge is None:
            return
        # Keep encoding to infer scale
        self.depth_encoding = msg.encoding
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.last_depth = depth

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

    def try_process_and_show(self):

        if self.last_type == '':
            cv2.imshow('camera', self.last_color)
            cv2.waitKey(1)
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
        fps = 1.0 / max(1e-6, (time.time() - t0))

        depth_m = self._depth_to_meters(self.last_depth)
        fx, fy, cx, cy, W, H = self.cam_K

        display_frame = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)

        if self.last_type == '0':  # points (solid point) -> sample depth INSIDE mask
            if masks is not None and masks.shape[0] > 0 and depth_m is not None and self.cam_K is not None:
                for i in range(masks.shape[0]):
                    mask = (masks[i, 0].astype(np.uint8) * 255)

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
                    else:
                        label += ' No Depth'

                    cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)
                    cv2.putText(display_frame, label, (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif self.last_type == '1':  # lines
            if masks is not None and masks.shape[0] > 0 and depth_m is not None and self.cam_K is not None:
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
                                ix, iy = inter
                                ix_i = int(np.clip(round(ix), 0, res - 1))
                                iy_i = int(np.clip(round(iy), 0, res - 1))

                                u_full = ix_i + start_x
                                v_full = iy_i + start_y
                                Z = self._median_depth_in_circle(depth_m, u_full, v_full, radius_px=8)

                                label = f'({ix_i},{iy_i})'
                                if Z is not None:
                                    X = (u_full - cx) / fx * Z
                                    Y = (v_full - cy) / fy * Z
                                    label = f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m'
                                else:
                                    label += ' No Depth'

                                # (F) draw infinite lines
                                self._draw_infinite_line_on_crop(display_frame, a['seg'], res, (0, 255, 255), thickness=2)
                                self._draw_infinite_line_on_crop(display_frame, b['seg'], res, (255, 255, 0), thickness=2)

                                cv2.circle(display_frame, (ix_i, iy_i), 6, (0, 0, 255), -1)
                                cv2.putText(display_frame, label, (ix_i + 10, iy_i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 255, 255), 2)

        elif self.last_type == '2':  # holes
            if masks is not None and masks.shape[0] > 0:
                for i in range(masks.shape[0]):
                    mask = masks[i, 0] # [resolution, resolution]
                    
                    mask = mask.astype(np.uint8) * 255
                    color = (255, 0, 0)  # 固定颜色为蓝色
                    
                    colored_mask = np.zeros_like(display_frame, dtype=np.uint8)
                    colored_mask[:, :, 0] = mask * (color[0] / 255)
                    colored_mask[:, :, 1] = mask * (color[1] / 255)
                    colored_mask[:, :, 2] = mask * (color[2] / 255)
                    
                    # 半透明叠加
                    display_frame = cv2.addWeighted(display_frame, 1.0, colored_mask, 0.5, 0)


                    # 计算并绘制中心点
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # --- 计算3D坐标 (针对孔洞优化) ---
                        # 策略：孔洞本身没有深度（或深度无效），我们需要采样孔洞边缘（Rim）的深度
                        
                        # 1. 膨胀掩码以获取边缘区域
                        kernel = np.ones((5, 5), np.uint8)
                        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
                        # 2. 减去原始掩码得到环状边缘 (Rim)
                        rim_mask = cv2.subtract(dilated_mask, mask)
                        
                        # 可视化膨胀区域 (Rim) - 绿色半透明
                        rim_color = (0, 255, 0) 
                        colored_rim = np.zeros_like(display_frame, dtype=np.uint8)
                        colored_rim[:, :, 0] = rim_mask * (rim_color[0] / 255)
                        colored_rim[:, :, 1] = rim_mask * (rim_color[1] / 255)
                        colored_rim[:, :, 2] = rim_mask * (rim_color[2] / 255)
                        display_frame = cv2.addWeighted(display_frame, 1.0, colored_rim, 0.5, 0)

                        # 3. 获取边缘像素的坐标
                        rim_y, rim_x = np.where(rim_mask > 0)
                        
                        # 4. 映射回原始全图坐标
                        rim_y_full = rim_y + start_y
                        rim_x_full = rim_x + start_x

                        valid_depths = []
                        for rx, ry in zip(rim_x_full, rim_y_full):
                            if 0 <= ry < depth_m.shape[0] and 0 <= rx < depth_m.shape[1]:
                                d = depth_m[ry, rx]
                                if d > 0:
                                    valid_depths.append(d)

                        label = f'({cX},{cY})'
                        if valid_depths:
                            Z = float(np.median(valid_depths))
                            u_full = cX + start_x
                            v_full = cY + start_y
                            X = (u_full - cx) / fx * Z
                            Y = (v_full - cy) / fy * Z
                            label = f'X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m'
                            # Log coordinates
                            # self.get_logger().info(f'Mask {i}: {label}')
                        else:
                            label += ' No Depth'

                        cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)
                        cv2.putText(display_frame, label, (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        

        # Overlay fps and description
        cv2.putText(display_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f'Description: {self.last_description}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('camera', display_frame)
        cv2.waitKey(1)


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
        if cv2 is not None:
            cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
