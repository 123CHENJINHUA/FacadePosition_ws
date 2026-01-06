import cv2
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建图像的路径 (假设 images 文件夹在上一级目录)
image_path = os.path.join(current_dir, '../images/1.jpg')

# 读取图像
img = cv2.imread(image_path)

if img is None:
    print(f"Error: 无法加载图像，请检查路径: {image_path}")
else:
    # 指定圆心的像素坐标 (x, y)
    center_coordinates = (200, 200)
    
    # 圆的半径
    radius = 50
    
    # 颜色 (B, G, R) - 这里使用红色
    color = (0, 0, 255)
    
    # 线条粗细 (-1 表示填充)
    thickness = 3
    
    # 在图像上画圆
    cv2.circle(img, center_coordinates, radius, color, thickness)
    
    # 显示图像
    cv2.imshow('Image with Circle', img)
    
    # 等待按键，然后关闭窗口
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
