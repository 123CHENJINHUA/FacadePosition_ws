import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from qwen_pkg_interfaces.msg import QwenResponse


class QwenServiceNode(Node):
    def __init__(self):
        super().__init__('qwen_service_node')
        # Publisher for response content
        self.response_pub = self.create_publisher(QwenResponse, 'qwen_service/response', 10)

    def handle_order(self):
        out = QwenResponse()
        out.description = 'holes'
        out.type = '2'
        out.reason = 'aaa'
        self.response_pub.publish(out)


def main():
    rclpy.init()
    node = QwenServiceNode()
    node.handle_order()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
