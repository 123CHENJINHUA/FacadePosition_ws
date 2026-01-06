import os
import glob
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from pathlib import Path

try:
    from cv_bridge import CvBridge
    import cv2
except Exception:
    CvBridge = None
    cv2 = None

class FolderImagePublisher(Node):
    def __init__(self):
        super().__init__('folder_image_publisher')
        self.bridge = CvBridge() if CvBridge else None

        # Resolve images_dir similar to your pattern
        file_path = Path(__file__).resolve()
        pkg_name = 'qwen_pkg'

        records_dir = None
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                records_dir = ws_root / 'src' / pkg_name / 'images'
                break
        if records_dir is None:
            pkg_dir = None
            for p in file_path.parents:
                if p.name == pkg_name:
                    pkg_dir = p
                    break
            records_dir = (pkg_dir / 'images') if pkg_dir is not None else (file_path.parent / 'images')

        self.declare_parameter('images_dir', str(records_dir))
        images_dir = Path(self.get_parameter('images_dir').get_parameter_value().string_value).resolve()
        self.get_logger().info(f'Using images_dir: {images_dir}')

        # Resolve specific files
        self.img1_path = str(images_dir / '1.jpg')
        self.img2_path = str(images_dir / '7.jpg')
        if not os.path.exists(self.img1_path):
            self.get_logger().warn(f'Image not found: {self.img1_path}')
        if not os.path.exists(self.img2_path):
            self.get_logger().warn(f'Image not found: {self.img2_path}')

        self.pub_cam1 = self.create_publisher(Image, 'camera1/image_raw', 10)
        self.pub_cam2 = self.create_publisher(Image, 'camera2/image_raw', 10)

        self.timer1 = self.create_timer(0.5, self.publish_cam1)
        self.timer2 = self.create_timer(0.5, self.publish_cam2)

    def publish_cam1(self):
        if self.bridge is None or not os.path.exists(self.img1_path):
            return
        img = cv2.imread(self.img1_path)
        if img is None:
            self.get_logger().warn(f'Failed to read image {self.img1_path}')
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.pub_cam1.publish(msg)

    def publish_cam2(self):
        if self.bridge is None or not os.path.exists(self.img2_path):
            return
        img = cv2.imread(self.img2_path)
        if img is None:
            self.get_logger().warn(f'Failed to read image {self.img2_path}')
            return
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.pub_cam2.publish(msg)


def main():
    rclpy.init()
    node = FolderImagePublisher()
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
