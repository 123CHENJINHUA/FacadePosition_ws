import os
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from qwen_pkg_interfaces.srv import QwenOrder

class QwenServiceClient(Node):
    def __init__(self):
        super().__init__('qwen_service_client')
        # Parameter to choose order (1 for camera1, 2 for camera2)
        self.declare_parameter('order', '1')
        self.order = self.get_parameter('order').get_parameter_value().string_value

        # Service client
        self.cli = self.create_client(QwenOrder, 'qwen_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service qwen_service...')

        # Send request once on startup
        self.send_request()

    def send_request(self):
        req = QwenOrder.Request()
        req.order = self.order
        self.get_logger().info(f'Sending request with order={req.order}')
        future = self.cli.call_async(req)
        future.add_done_callback(self._handle_response)

    def _handle_response(self, future):
        try:
            resp = future.result()
            self.get_logger().info(f'Response:\nDescription: {resp.description}\nType: {resp.type}\nReason: {resp.reason}\n')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main():
    rclpy.init()
    node = QwenServiceClient()
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
