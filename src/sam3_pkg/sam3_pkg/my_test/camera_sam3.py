import os
import cv2
import numpy as np
from PIL import Image
import torch
import time

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import traceback

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. 打开摄像头获取分辨率
    # 使用 cv2.CAP_DSHOW 后端以提高在 Windows 上的兼容性
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 读取一帧以获取尺寸
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    h_orig, w_orig = frame.shape[:2]
    resolution = min(h_orig, w_orig)
    print(f"Camera resolution: {w_orig}x{h_orig}. Using crop resolution: {resolution}x{resolution}")

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 3. 预处理图像
        # OpenCV uses BGR, PIL uses RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Center Crop logic to avoid distortion
        # We already know resolution = min(h, w)
        start_x = (w_orig - resolution) // 2
        start_y = (h_orig - resolution) // 2
        
        # Crop to square
        image_cropped = image_rgb[start_y:start_y+resolution, start_x:start_x+resolution]
        
        # No resize needed as we use the crop size as resolution
        pil_image = Image.fromarray(image_cropped)
        
        start_time = time.time()

        # 4. 推理
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

        # 5. 可视化结果
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
                    cv2.circle(display_frame, (cX, cY), 5, (0, 255, 255), -1)
                    cv2.putText(display_frame, f"({cX}, {cY})", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 绘制边框
            for i in range(boxes.shape[0]):
                x0, y0, x1, y1 = boxes[i].astype(int)
                cv2.rectangle(display_frame, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
                # cv2.putText(display_frame, text_prompt, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示FPS
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(display_frame, f"Prompt: {text_prompt}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.resize

        cv2.imshow("SAM3 Real-time Segmentation", display_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
