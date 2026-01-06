import os
import cv2
import numpy as np
from PIL import Image
import torch
 
import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
if __name__ == '__main__':
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f"{sam3_root}/sam3.pt"
    
        
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
    model.to(device)
    model.eval()
    
    processor = Sam3Processor(model, confidence_threshold=0.5)
 
    images_dir = os.path.join(sam3_root, "images")
    results_dir = os.path.join(sam3_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(supported_extensions)]

    print(f"Found {len(image_files)} images in {images_dir}")

    for image_file in image_files:
        print(f"Processing {image_file}...")
        image_path = os.path.join(images_dir, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not open {image_file}: {e}")
            continue
            
        width, height = image.size
        
        # --------------------------
        # 文本分割
        # --------------------------
        state = processor.set_image(image)
        # state = processor.set_text_prompt(prompt="shoe", state=state)
        state = processor.set_text_prompt(prompt="blue line", state=state)
        
        # 获取分割结果
        masks = state["masks"].cpu().numpy()  # [num_instances, H, W]
        boxes = state["boxes"].cpu().numpy()  # [num_instances, 4]
        scores = state["scores"].cpu().numpy()
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 绘制掩码
        for i in range(masks.shape[0]):
            mask = masks[i].astype(np.uint8) * 255  # 0/255
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            colored_mask = np.zeros_like(image_cv, dtype=np.uint8)
            colored_mask[:, :, 0] = mask * (color[0] / 255)
            colored_mask[:, :, 1] = mask * (color[1] / 255)
            colored_mask[:, :, 2] = mask * (color[2] / 255)
            # 半透明叠加
            image_cv = cv2.addWeighted(image_cv, 1.0, colored_mask, 0.5, 0)
        
        # 绘制边框
        for i in range(boxes.shape[0]):
            x0, y0, x1, y1 = boxes[i].astype(int)
            cv2.rectangle(image_cv, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
        
        # 保存结果
        save_path = os.path.join(results_dir, image_file)
        cv2.imwrite(save_path, image_cv)
        print(f"Saved result to {save_path}")