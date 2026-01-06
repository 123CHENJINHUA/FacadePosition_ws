import os
import cv2
import numpy as np
from PIL import Image
import torch
import time
import pyrealsense2 as rs

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import traceback

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. 配置并启动 RealSense 相机
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 获取设备信息
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device_rs = pipeline_profile.get_device()
    
    print(f"RealSense Device: {device_rs.get_info(rs.camera_info.name)}")
    
    # 配置彩色流 - 使用常见的分辨率
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 配置深度流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 对齐对象：将深度对齐到彩色
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 启动流
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Error: Could not start RealSense pipeline: {e}")
        return
    
    # 等待相机稳定
    print("Warming up camera...")
    for _ in range(30):
        pipeline.wait_for_frames()
    
    # 获取流的分辨率
    color_stream = profile.get_stream(rs.stream.color)
    color_profile = color_stream.as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    w_orig = intrinsics.width
    h_orig = intrinsics.height
    
    resolution = min(h_orig, w_orig)
    print(f"RealSense resolution: {w_orig}x{h_orig}. Using crop resolution: {resolution}x{resolution}")

    # 2. 初始化模型
    print("Initializing SAM3 model...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f"{sam3_root}/sam3.pt"
    
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path, image_size=resolution)
    model.to(device)
    model.eval()
    
    processor = Sam3Processor(model, resolution=resolution, confidence_threshold=0.4)
    print(f"Model initialized with resolution {resolution}x{resolution}.")

    print("Starting video stream. Press 'q' to exit.")
    
    # 提示词
    text_prompt = "holes on the same object"

    try:
        while True:
            # 3. 从 RealSense 获取帧
            frames = pipeline.wait_for_frames()
            
            # 对齐帧
            aligned_frames = align.process(frames)
            
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                print("Warning: No frames received.")
                continue
            
            # 转换为 numpy 数组
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            
            # 4. 预处理图像
            # RealSense already provides BGR format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Center Crop logic to avoid distortion
            start_x = (w_orig - resolution) // 2
            start_y = (h_orig - resolution) // 2
            
            # Crop to square
            image_cropped = image_rgb[start_y:start_y+resolution, start_x:start_x+resolution]
            
            # No resize needed as we use the crop size as resolution
            pil_image = Image.fromarray(image_cropped)
            
            start_time = time.time()

            # 5. 推理
            try:
                state = processor.set_image(pil_image)
                state = processor.set_text_prompt(prompt=text_prompt, state=state)
                
                # 获取结果
                masks = state["masks"].cpu().numpy()  # [num_instances, 1, H, W]
                boxes = state["boxes"].cpu().numpy()  # [num_instances, 4]
                # scores = state["scores"].cpu().numpy()
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()
                masks = np.array([])
                boxes = np.array([])

            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0

            # Convert cropped image back to BGR for display
            display_frame = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)

            # 6. 可视化结果
            # 绘制掩码
            if masks.shape[0] > 0:
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
                        
                        # 5. 收集有效的深度值
                        valid_depths = []
                        for rx, ry in zip(rim_x_full, rim_y_full):
                            if 0 <= ry < h_orig and 0 <= rx < w_orig:
                                d = depth_image[ry, rx]
                                if d > 0: # 0 表示无效深度
                                    valid_depths.append(d)
                        
                        coord_text = f"({cX}, {cY})"
                        
                        if valid_depths:
                            # 使用中位数深度，过滤噪声
                            avg_depth_raw = np.median(valid_depths)
                            dist_meters = avg_depth_raw * aligned_depth_frame.get_units()
                            
                            # 反投影得到3D坐标 (X, Y, Z)
                            # 注意：使用全图坐标 (cX + start_x, cY + start_y)
                            cX_full = cX + start_x
                            cY_full = cY + start_y
                            
                            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cX_full, cY_full], dist_meters)
                            
                            coord_text = f"X:{point_3d[0]:.2f} Y:{point_3d[1]:.2f} Z:{point_3d[2]:.2f}m"
                        else:
                            coord_text += " No Depth"

                        cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)
                        cv2.putText(display_frame, coord_text, (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # # 绘制边框
                # for i in range(boxes.shape[0]):
                #     x0, y0, x1, y1 = boxes[i].astype(int)
                #     cv2.rectangle(display_frame, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
                #     # cv2.putText(display_frame, text_prompt, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示FPS
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(display_frame, f"Prompt: {text_prompt}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("SAM3 RealSense Real-time Segmentation", display_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # 7. 清理资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped.")

if __name__ == '__main__':
    main()
