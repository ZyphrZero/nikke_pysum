"""
CNN 数字识别器
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os


class SimpleDigitNet(nn.Module):
    """数字识别 CNN 模型"""

    def __init__(self, num_classes=10):
        super(SimpleDigitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNRecognizer:
    """CNN 数字识别器"""

    def __init__(self, model_path='sum10_model.pth', img_size=64, crop_ratio=0.8):
        """
        初始化 CNN 识别器

        Args:
            model_path: 模型文件路径
            img_size: 输入图像大小
            crop_ratio: 中心裁切比例
        """
        self.img_size = img_size
        self.crop_ratio = crop_ratio

        # 检查设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'使用设备: {self.device}')

        # 获取模型文件的实际路径
        import os
        actual_model_path = os.path.abspath(model_path)
        print(f'>> [CNN识别器] 模型路径: {actual_model_path}')

        # 加载模型
        if not os.path.exists(actual_model_path):
            raise FileNotFoundError(f'模型文件不存在: {actual_model_path}')

        self.model = SimpleDigitNet(num_classes=10).to(self.device)
        self.model.load_state_dict(torch.load(actual_model_path, map_location=self.device))
        self.model.eval()
        print(f'已加载模型: {actual_model_path}')

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def recognize_digit(self, cell_image: np.ndarray) -> int:
        """
        识别单个格子中的数字

        Args:
            cell_image: 格子图像（BGR 或灰度）

        Returns:
            识别出的数字（0-9）
        """
        # 中心裁切
        h, w = cell_image.shape[:2]
        new_h = int(h * self.crop_ratio)
        new_w = int(w * self.crop_ratio)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        cell_image = cell_image[start_y:start_y+new_h, start_x:start_x+new_w]

        # 转换为灰度图
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        # 转换为 PIL 图像
        pil_img = Image.fromarray(gray).convert('L')

        # 应用转换
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def recognize_grid(self, image: np.ndarray, rows: int, cols: int) -> list:
        """
        识别整个网格中的所有数字

        Args:
            image: 输入图像
            rows: 网格行数
            cols: 网格列数

        Returns:
            二维数字矩阵
        """
        h, w = image.shape[:2]
        matrix = [[0] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                # 计算格子位置
                y1 = int(r * (h / rows))
                y2 = int((r + 1) * (h / rows))
                x1 = int(c * (w / cols))
                x2 = int((c + 1) * (w / cols))

                # 提取格子
                cell = image[y1:y2, x1:x2]

                # 识别数字
                digit = self.recognize_digit(cell)
                matrix[r][c] = digit

        return matrix


def test_recognizer():
    """测试识别器"""
    recognizer = CNNRecognizer()
    print('CNN 识别器初始化成功！')
    print(f'设备: {recognizer.device}')
    print(f'图像大小: {recognizer.img_size}')
    print(f'裁切比例: {recognizer.crop_ratio}')


if __name__ == '__main__':
    test_recognizer()
