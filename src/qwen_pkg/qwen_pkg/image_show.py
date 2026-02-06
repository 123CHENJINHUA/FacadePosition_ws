import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image

import time
from pathlib import Path
import numpy as np

try:
    from cv_bridge import CvBridge
    import cv2
except Exception:
    CvBridge = None
    cv2 = None


class ImageShow(Node):
    def __init__(self):
        super().__init__('image_show')
        self.bridge = CvBridge() if CvBridge else None
        self.sub_image = self.create_subscription(Image, '/camera/sam3_result', self.on_image, 10)

        # Save video
        self._writer = None
        self._video_path = None
        self._fps = 30.0
        self._last_stamp_sec = None
        self._writer_size = None  # (w, h) determined by first frame

        # Create output directory under cwd
        out_dir = Path.cwd() / 'videos'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self._video_path = str(out_dir / f'sam3_result_{ts}.mp4')

        # Optional visualization
        if cv2 is not None:
            cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

        self.get_logger().info(f'Saving /camera/sam3_result to: {self._video_path}')

    def _ensure_writer(self, frame_bgr):
        if cv2 is None:
            return
        if self._writer is not None:
            return

        h, w = frame_bgr.shape[:2]
        self._writer_size = (w, h)

        # Common codecs: mp4v (portable), avc1 (needs h264 support)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(self._video_path, fourcc, self._fps, (w, h))
        if not self._writer.isOpened():
            self.get_logger().error(f'Failed to open VideoWriter: {self._video_path}')
            self._writer = None
            self._writer_size = None

    def _pad_to_writer_size(self, frame_bgr):
        """Pad frame to writer size if smaller; keep content at top-left."""
        if cv2 is None:
            return frame_bgr
        if self._writer_size is None:
            return frame_bgr

        target_w, target_h = self._writer_size
        h, w = frame_bgr.shape[:2]

        # If same size, return as-is
        if (w, h) == (target_w, target_h):
            return frame_bgr

        # If larger than writer size, center-crop to fit (avoid VideoWriter crash)
        if w > target_w or h > target_h:
            x0 = max(0, (w - target_w) // 2)
            y0 = max(0, (h - target_h) // 2)
            return frame_bgr[y0:y0 + target_h, x0:x0 + target_w]

        # Pad to match target size
        canvas = frame_bgr
        if frame_bgr.ndim == 2:
            canvas = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        out = np.zeros((target_h, target_w, 3), dtype=canvas.dtype)
        out[:h, :w, :] = canvas
        return out

    def on_image(self, msg: Image):
        if self.bridge is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # # Estimate fps from ROS header stamps if available
        # try:
        #     stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        #     if self._last_stamp_sec is not None:
        #         dt = stamp_sec - self._last_stamp_sec
        #         if dt > 1e-4:
        #             fps_est = 1.0 / dt
        #             # Limit to reasonable range to avoid spikes
        #             if 1.0 <= fps_est <= 120.0:
        #                 self._fps = float(fps_est)
        #     self._last_stamp_sec = stamp_sec
        # except Exception:
        #     pass

        # self._ensure_writer(frame)

        # # Ensure the frame matches writer resolution (pad if smaller)
        # frame_for_write = self._pad_to_writer_size(frame)

        # if self._writer is not None:
        #     try:
        #         self._writer.write(frame_for_write)
        #     except Exception as e:
        #         self.get_logger().warn(f'VideoWriter write failed: {e}')

        if cv2 is not None:
            cv2.imshow('camera', frame)
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
        try:
            if getattr(node, '_writer', None) is not None:
                node._writer.release()
        except Exception:
            pass
        node.destroy_node()
        if cv2 is not None:
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
