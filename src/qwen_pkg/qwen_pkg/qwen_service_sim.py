import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from qwen_pkg_interfaces.msg import QwenResponse
import time


class QwenServiceNode(Node):
    def __init__(self):
        super().__init__('qwen_service_node')
        # Publisher for response content
        self.response_pub = self.create_publisher(
            QwenResponse, 
            'qwen_service/response', 
            10
        )
        
        # 等待发布者建立连接
        self.get_logger().info('等待发布者建立连接...')
        time.sleep(1)  # 或者用 rclpy.wait_for_subscriber
        
        # 定时器方式发布（可选）
        # self.timer = self.create_timer(1.0, self.handle_order)

    def handle_order(self):
        out = QwenResponse()
        out.description = 'holes'
        out.type = '2'
        out.reason = 'aaa'
        
        # 确保发布者已建立
        while self.response_pub.get_subscription_count() == 0:
            self.get_logger().warn('等待订阅者连接...')
            time.sleep(0.5)
            
        self.response_pub.publish(out)
        self.get_logger().info(f'已发布消息: {out.description}')
        
        # 如果需要只发布一次，可以在此处停止节点
        # rclpy.shutdown()


def main():
    rclpy.init()
    node = QwenServiceNode()
    
    # 方法1：直接调用（确保有订阅者）
    node.handle_order()
    
    # 方法2：使用定时器持续发布
    # node.create_timer(2.0, node.handle_order)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()