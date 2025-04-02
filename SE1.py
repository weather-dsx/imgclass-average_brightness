import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 加载预训练的 ResNet18 模型
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
#print(resnet18)

# 检查并创建文件夹
def check_folders():
    if not os.path.exists("classify_img"):
        os.makedirs("classify_img")
        print("Created 'classify_img' folder.")
        
# 定义图像变换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载 ImageNet 的类别名称
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


def classify_images():
    check_folders()  

    # 遍历
    for img_name in os.listdir("input_img"):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join("input_img", img_name)
            img = Image.open(img_path).convert("RGB")
            
            img_cv = cv2.imread(img_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) 
            
            img_transformed = transform(img).unsqueeze(0)  # 一次处理一张图片
            
            # 分类
            with torch.no_grad():
                outputs = resnet18(img_transformed)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                top5_prob, top5_indices = torch.topk(probabilities, 5)
            top5_classes = [classes[idx] for idx in top5_indices]
            
            img_np = np.array(img)
            img= torch.tensor(img_np, dtype=torch.float32) / 255.0
            img_cv = torch.tensor(img_cv, dtype=torch.float32) / 255.0
            
            avg_color_PIL = np.array(img).mean(axis=(0, 1))  # 使用 PIL 计算平均颜色
            avg_color_PIL = avg_color_PIL.tolist()

            avg_color_opencv = img_cv.mean(axis=(0, 1))  # 使用 OpenCV 计算平均颜色
            avg_color_opencv = avg_color_opencv.tolist()

            avg_color = [(i + j) / 2 for i, j in zip(avg_color_PIL, avg_color_opencv)]

            # 绘制结果图像
            plt.figure(figsize=(20, 15))
            
            # 左上原始图像
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title("Original Image")

            # 右上 Top 5 Predictions 图
            plt.subplot(2, 2, 2)
            plt.barh(top5_classes, top5_prob, color=avg_color)  # 使用默认颜色
            plt.xlabel("Probability")
            plt.title("Top 5 Predictions")

           # 左下 PIL 平均亮度柱状图
            plt.subplot(2,2,3)
            plt.title("PIL Average Brightness")
            channels = ['R', 'G', 'B']
            plt.bar(channels, avg_color_PIL, color=[(avg_color_PIL[0], 0, 0), (0, avg_color_PIL[1], 0), (0, 0, avg_color_PIL[2])])
            for i, v in enumerate(avg_color_PIL):
                plt.text(i, v +0.005, f"{v:.4f}", ha='center') 

            # 右下 OpenCV 平均亮度柱状图
            plt.subplot(2,2,4)
            plt.title("OpenCV Average Brightness")
            plt.bar(channels, avg_color_opencv, color=[(avg_color_opencv[0], 0, 0), (0, avg_color_opencv[1], 0), (0, 0, avg_color_opencv[2])])
            for i, v in enumerate(avg_color_opencv):
                plt.text(i, v +0.005 , f"{v:.4f}", ha='center')

            
            # 保存结果图像
            save_name = f"{os.path.splitext(img_name)[0]}_{top5_classes[0]}.jpg"
            save_path = os.path.join("classify_img", save_name)
            plt.savefig(save_path)
            plt.close()

            print(f"Processed and saved: {save_name}")

classify_images()