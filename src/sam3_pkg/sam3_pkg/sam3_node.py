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


class ImageShow(Node):
    def __init__(self):
        super().__init__('image_show')
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

        if self.last_type == '2':
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
    node = ImageShow()
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
