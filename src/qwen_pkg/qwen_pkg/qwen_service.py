import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from qwen_pkg_interfaces.msg import QwenResponse
from openai import OpenAI
import base64
import io
import json
import os
from pathlib import Path

from rclpy.callback_groups import ReentrantCallbackGroup
from qwen_pkg_interfaces.srv import QwenOrder

try:
    from cv_bridge import CvBridge
    import cv2
except Exception:
    CvBridge = None
    cv2 = None

class QwenServiceNode(Node):
    def __init__(self):
        super().__init__('qwen_service_node')
        self.cb_group = ReentrantCallbackGroup()

        # Params for OpenAI (Qwen compatible) client
        api_key = "sk-55ae840bfc394775a122c42d494787f1"
        base_url = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv('QWEN_MODEL', 'qwen-vl-plus')

        # Load chat history path via Path parents
        file_path = Path(__file__).resolve()
        pkg_name = 'qwen_pkg'
        hist_path = None
        for p in file_path.parents:
            if p.name == 'install':
                ws_root = p.parent
                hist_path = ws_root / 'src' / pkg_name / 'chat_history' / 'chat_history.json'
                break
        if hist_path is None:
            pkg_dir = None
            for p in file_path.parents:
                if p.name == pkg_name:
                    pkg_dir = p
                    break
            if pkg_dir is not None:
                hist_path = pkg_dir / 'chat_history' / 'chat_history.json'
            else:
                hist_path = file_path.parent / 'chat_history.json'
        self.history_path = str(hist_path)
        self.messages = []

        self.get_logger().info(f'History path: {self.history_path}')
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
                self.get_logger().info(f'Loaded chat history with {len(self.messages)} messages.')
            except Exception as e:
                self.get_logger().warn(f'Failed to load chat history: {e}')
        if not self.messages:
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant for construction image analysis."}
            ]

        # Subscribers for images
        self.bridge = CvBridge() if CvBridge else None
        self.camera1_image = None
        self.camera2_image = None

        # Make topics configurable from launch
        self.declare_parameter('camera1_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera2_topic', '/camera/camera/depth/image_raw')
        cam1_topic = self.get_parameter('camera1_topic').get_parameter_value().string_value
        cam2_topic = self.get_parameter('camera2_topic').get_parameter_value().string_value

        self.get_logger().info(f'Subscribing camera1_topic={cam1_topic}')
        self.get_logger().info(f'Subscribing camera2_topic={cam2_topic}')

        self.sub1 = self.create_subscription(Image, cam1_topic, self._camera1_cb, 10)
        self.sub2 = self.create_subscription(Image, cam2_topic, self._camera2_cb, 10)

        # Publisher for response content
        self.response_pub = self.create_publisher(QwenResponse, 'qwen_service/response', 10)

        self.srv = self.create_service(QwenOrder, 'qwen_service', self.handle_order, callback_group=self.cb_group)

    def _camera1_cb(self, msg: Image):
        self.camera1_image = msg

    def _camera2_cb(self, msg: Image):
        self.camera2_image = msg

    def _image_to_base64(self, img_msg: Image) -> str:
        if self.bridge is None:
            raise RuntimeError('cv_bridge not available. Please install ros-<distro>-cv-bridge and OpenCV.')
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        success, buf = cv2.imencode('.jpg', cv_img)
        if not success:
            raise RuntimeError('Failed to encode image to JPEG')
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def _extract_json_object(self, text: str):
        s = text.strip()
        # Strip code fences if present
        if s.startswith('```'):
            lines = s.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith('```'):
                lines = lines[:-1]
            s = '\n'.join(lines).strip()
        # Find first top-level JSON object
        start = None
        depth = 0
        for i, ch in enumerate(s):
            if ch == '{':
                if start is None:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        obj = s[start:i+1]
                        try:
                            return json.loads(obj)
                        except Exception:
                            break
        # Fallback to direct parse
        try:
            return json.loads(s)
        except Exception:
            return None

    def call_qwen_with_image(self, image_b64_url: str) -> str:
        # Build content using current history + new user content
        messages = list(self.messages)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64_url}},
                {"type": "text", "text": "Please output only JSON, without any explanations, prefixes or suffixes.The format is: {\"description\": <string>, \"type\": <int>, \"reason\": <string>}"}
            ]
        })
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        assistant_response = completion.choices[0].message.content
        # Update history
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64_url}},
                {"type": "text", "text": "Please output only JSON, without any explanations, prefixes or suffixes.The format is: {\"description\": <string>, \"type\": <int>, \"reason\": <string>}"}
            ]
        })
        self.messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    def handle_order(self, request, response):
        # request.order is string representing a number
        order = request.order.strip()
        try:
            if order == '1':
                img_msg = self.camera1_image
            elif order == '2':
                img_msg = self.camera2_image
            else:
                response.description = None
                response.type = None
                response.reason = None
                return response

            if img_msg is None:
                response.description = None
                response.type = None
                response.reason = None
                return response

            image_b64_url = self._image_to_base64(img_msg)
            qwen_reply = self.call_qwen_with_image(image_b64_url)

            # Parse JSON structure from string reply (even with code fences)
            parsed = self._extract_json_object(qwen_reply)

            if isinstance(parsed, dict) and 'description' in parsed and 'type' in parsed and 'reason' in parsed:
                response.description = parsed.get('description')
                tval = parsed.get('type')
                response.type = str(tval) if tval is not None else None
                response.reason = parsed.get('reason')
            else:
                response.description = None
                response.type = None
                response.reason = None
            # Publish the response using custom msg
            out = QwenResponse()
            out.description = response.description or ''
            out.type = response.type or ''
            out.reason = response.reason or ''
            self.response_pub.publish(out)
            return response
        except Exception as e:
            response.description = None
            response.type = None
            response.reason = None
            out = QwenResponse()
            out.description = ''
            out.type = ''
            out.reason = ''
            self.response_pub.publish(out)
            return response


def main():
    rclpy.init()
    node = QwenServiceNode()
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
