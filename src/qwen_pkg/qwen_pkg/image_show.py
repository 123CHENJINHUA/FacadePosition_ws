import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from qwen_pkg_interfaces.msg import QwenResponse

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
        self.sub_image = self.create_subscription(Image, '/camera/camera/color/image_raw', self.on_image, 10)
        self.sub_text = self.create_subscription(QwenResponse, 'qwen_service/response', self.on_text, 10)
        self.last_desc = ''
        self.last_type = ''
        self.last_reason = ''
        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

    def on_text(self, msg: QwenResponse):
        self.last_desc = msg.description or ''
        self.last_type = msg.type or ''
        self.last_reason = msg.reason or ''

    def on_image(self, msg: Image):
        if self.bridge is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Overlay fields
        lines = []
        if self.last_desc:
            lines.append(f'Desc: {self.last_desc}')
        if self.last_type:
            lines.append(f'Type: {self.last_type}')
        if self.last_reason:
            lines.append(f'Reason: {self.last_reason}')
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 30
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
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
