"""
图像识别模块
功能：截取屏幕区域、网格分割、CNN 数字识别
"""

import cv2
import numpy as np
from PIL import ImageGrab
from typing import List, Tuple, Optional

from .cnn_recognizer import CNNRecognizer


class ImageProcessor:
    """图像处理器，负责截图、网格分割和数字识别"""

    def __init__(self, model_path: str = 'sum10_model.pth'):
        """
        初始化图像处理器

        Args:
            model_path: CNN 模型文件路径
        """
        # 获取模型文件的实际路径
        import os
        actual_model_path = os.path.abspath(model_path)
        print(f'>> [图像处理器] 使用模型路径: {actual_model_path}')

        # 初始化 CNN 识别器
        self.recognizer = CNNRecognizer(model_path=actual_model_path)

        # 透视变换相关
        self.perspective_matrix = None  # 透视变换矩阵
        self.inverse_perspective_matrix = None  # 逆透视变换矩阵（用于坐标映射）
        self.corner_points = None  # 原始虚线四个点坐标

    def capture_screen_region(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        截取屏幕指定区域

        Args:
            bbox: 边界框 (left, top, right, bottom)

        Returns:
            截取的图像（OpenCV格式，BGR）
        """
        # 使用PIL截图
        screenshot = ImageGrab.grab(bbox=bbox)

        # 转换为OpenCV格式（BGR）
        img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        return img_cv

    def capture_and_warp_perspective(self, corner_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        根据四个角点截取屏幕并进行透视变换

        Args:
            corner_points: 四个角点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                          顺序：左上、右上、右下、左下

        Returns:
            透视变换后的图像（OpenCV格式，BGR）
        """
        # 保存原始角点
        self.corner_points = corner_points

        # 计算包围盒，用于截取整个区域
        xs = [p[0] for p in corner_points]
        ys = [p[1] for p in corner_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

        # 截取整个包围区域
        full_screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
        full_img = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2BGR)

        # 将角点坐标转换为相对于包围盒的坐标
        src_points = np.float32([
            (corner_points[0][0] - left, corner_points[0][1] - top),  # 左上
            (corner_points[1][0] - left, corner_points[1][1] - top),  # 右上
            (corner_points[2][0] - left, corner_points[2][1] - top),  # 右下
            (corner_points[3][0] - left, corner_points[3][1] - top)   # 左下
        ])

        # 计算目标矩形的宽度和高度
        # 使用四边形的边长来估算合适的输出尺寸
        width_top = np.linalg.norm(src_points[1] - src_points[0])
        width_bottom = np.linalg.norm(src_points[2] - src_points[3])
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(src_points[3] - src_points[0])
        height_right = np.linalg.norm(src_points[2] - src_points[1])
        height = int(max(height_left, height_right))

        # 定义目标矩形的四个角点（标准矩形）
        dst_points = np.float32([
            [0, 0],              # 左上
            [width, 0],          # 右上
            [width, height],     # 右下
            [0, height]          # 左下
        ])

        # 计算透视变换矩阵
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 计算逆透视变换矩阵（用于后续坐标映射）
        self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # 应用透视变换
        warped = cv2.warpPerspective(full_img, self.perspective_matrix, (width, height))

        return warped

    def map_grid_to_screen(self, row: int, col: int, rows: int, cols: int) -> Tuple[int, int]:
        """
        将网格坐标映射回原始屏幕坐标（考虑透视变换）

        Args:
            row: 网格行号
            col: 网格列号
            rows: 总行数
            cols: 总列数

        Returns:
            原始屏幕坐标 (x, y)
        """
        if self.inverse_perspective_matrix is None or self.corner_points is None:
            raise RuntimeError("未进行透视变换，无法映射坐标")

        # 计算在变换后图像中的坐标（网格中心点）
        # 首先需要知道变换后图像的尺寸
        xs = [p[0] for p in self.corner_points]
        ys = [p[1] for p in self.corner_points]
        left, top = min(xs), min(ys)

        # 重新计算变换后的尺寸（与capture_and_warp_perspective中的计算一致）
        src_points = np.float32([
            (self.corner_points[0][0] - left, self.corner_points[0][1] - top),
            (self.corner_points[1][0] - left, self.corner_points[1][1] - top),
            (self.corner_points[2][0] - left, self.corner_points[2][1] - top),
            (self.corner_points[3][0] - left, self.corner_points[3][1] - top)
        ])

        width_top = np.linalg.norm(src_points[1] - src_points[0])
        width_bottom = np.linalg.norm(src_points[2] - src_points[3])
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(src_points[3] - src_points[0])
        height_right = np.linalg.norm(src_points[2] - src_points[1])
        height = int(max(height_left, height_right))

        # 计算网格单元格的大小
        cell_width = width / cols
        cell_height = height / rows

        # 计算在变换后图像中的坐标（单元格中心点）
        warped_x = (col + 0.5) * cell_width
        warped_y = (row + 0.5) * cell_height

        # 使用逆透视变换矩阵映射回原始坐标
        point = np.array([[[warped_x, warped_y]]], dtype=np.float32)
        original_point = cv2.perspectiveTransform(point, self.inverse_perspective_matrix)

        # 转换为相对于包围盒的坐标，然后加上偏移量得到屏幕坐标
        x = int(original_point[0][0][0] + left)
        y = int(original_point[0][0][1] + top)

        return x, y

    def detect_grid(self, image: np.ndarray, rows: int, cols: int) -> List[List[Tuple[int, int, int, int]]]:
        """
        检测并分割网格（均匀分割）

        Args:
            image: 输入图像
            rows: 网格行数
            cols: 网格列数

        Returns:
            网格单元格的边界框列表 [[cell_bbox, ...], ...]
            每个cell_bbox是 (x, y, width, height)
        """
        height, width = image.shape[:2]

        # 计算每个单元格的大小
        cell_height = height // rows
        cell_width = width // cols

        # 生成网格单元格的边界框
        grid_cells = []
        for i in range(rows):
            row_cells = []
            for j in range(cols):
                x = j * cell_width
                y = i * cell_height
                w = cell_width
                h = cell_height

                # 如果是最后一行或最后一列，扩展到图像边界
                if i == rows - 1:
                    h = height - y
                if j == cols - 1:
                    w = width - x

                row_cells.append((x, y, w, h))
            grid_cells.append(row_cells)

        return grid_cells

    def detect_grid_auto(self, image: np.ndarray) -> Tuple[int, int, List[List[Tuple[int, int, int, int]]]]:
        """
        自动检测网格的行数和列数（改进版：多方法组合）

        Args:
            image: 输入图像

        Returns:
            (rows, cols, grid_cells)
        """
        print(">> [网格检测] 开始自动检测网格尺寸...")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 方法1：边缘检测 + 投影分析
        rows1, cols1, score1 = self._detect_by_edge_projection(gray)
        print(f">> [方法1] 边缘投影检测: {rows1}×{cols1} (置信度: {score1:.2f})")

        # 方法2：霍夫线检测
        rows2, cols2, score2 = self._detect_by_hough_lines(gray)
        print(f">> [方法2] 霍夫线检测: {rows2}×{cols2} (置信度: {score2:.2f})")

        # 方法3：自适应阈值 + 轮廓分析
        rows3, cols3, score3 = self._detect_by_contours(gray)
        print(f">> [方法3] 轮廓分析检测: {rows3}×{cols3} (置信度: {score3:.2f})")

        # 选择置信度最高的结果
        candidates = [
            (rows1, cols1, score1),
            (rows2, cols2, score2),
            (rows3, cols3, score3)
        ]

        # 按置信度排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        rows, cols, best_score = candidates[0]

        # 如果最佳结果的置信度太低，使用常见的网格尺寸
        if best_score < 0.3:
            print(f">> [警告] 检测置信度较低 ({best_score:.2f})，尝试常见网格尺寸...")
            # 尝试常见的网格尺寸
            common_sizes = [(15, 9), (10, 10), (12, 8), (20, 10)]
            rows, cols = self._try_common_sizes(image, common_sizes)
            print(f">> [网格检测] 使用常见尺寸: {rows}×{cols}")
        else:
            print(f">> [网格检测] 最终结果: {rows}×{cols} (置信度: {best_score:.2f})")

        # 生成网格
        grid_cells = self.detect_grid(image, rows, cols)

        return rows, cols, grid_cells

    def _detect_by_edge_projection(self, gray: np.ndarray) -> Tuple[int, int, float]:
        """使用边缘检测和投影分析检测网格"""
        edges = cv2.Canny(gray, 30, 100)

        # 水平和垂直投影
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)

        # 找到投影的峰值（网格线位置）
        h_peaks = self._find_peaks(h_projection, min_distance=20)
        v_peaks = self._find_peaks(v_projection, min_distance=20)

        rows = len(h_peaks) - 1 if len(h_peaks) > 1 else 15
        cols = len(v_peaks) - 1 if len(v_peaks) > 1 else 9

        # 计算置信度（基于峰值的规律性）
        h_regularity = self._calculate_regularity(h_peaks) if len(h_peaks) > 2 else 0
        v_regularity = self._calculate_regularity(v_peaks) if len(v_peaks) > 2 else 0
        confidence = (h_regularity + v_regularity) / 2

        return max(rows, 5), max(cols, 5), confidence

    def _detect_by_hough_lines(self, gray: np.ndarray) -> Tuple[int, int, float]:
        """使用霍夫线变换检测网格"""
        edges = cv2.Canny(gray, 50, 150)

        # 检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)

        if lines is None:
            return 15, 9, 0.0

        # 分类水平线和垂直线
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:  # 水平线
                h_lines.append((y1 + y2) / 2)
            elif 80 < angle < 100:  # 垂直线
                v_lines.append((x1 + x2) / 2)

        # 聚类相近的线
        h_lines = self._cluster_lines(h_lines, threshold=20)
        v_lines = self._cluster_lines(v_lines, threshold=20)

        rows = len(h_lines) - 1 if len(h_lines) > 1 else 15
        cols = len(v_lines) - 1 if len(v_lines) > 1 else 9

        # 置信度基于检测到的线条数量
        confidence = min(len(h_lines) * len(v_lines) / 200, 1.0)

        return max(rows, 5), max(cols, 5), confidence

    def _detect_by_contours(self, gray: np.ndarray) -> Tuple[int, int, float]:
        """使用轮廓分析检测网格（分析数字块的分布）"""
        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 10:
            return 15, 9, 0.0

        # 获取轮廓的中心点
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))

        if len(centers) < 10:
            return 15, 9, 0.0

        # 分析中心点的分布
        centers = np.array(centers)
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]

        # 使用直方图找到行列数
        x_hist, _ = np.histogram(x_coords, bins=50)
        y_hist, _ = np.histogram(y_coords, bins=50)

        # 找到直方图的峰值数量
        x_peaks = self._find_peaks(x_hist, min_distance=2)
        y_peaks = self._find_peaks(y_hist, min_distance=2)

        cols = len(x_peaks) if len(x_peaks) > 0 else 9
        rows = len(y_peaks) if len(y_peaks) > 0 else 15

        # 置信度基于检测到的轮廓数量和分布
        expected_cells = rows * cols
        confidence = min(len(centers) / expected_cells, 1.0) if expected_cells > 0 else 0.0

        return max(rows, 5), max(cols, 5), confidence

    def _find_peaks(self, signal: np.ndarray, min_distance: int = 10) -> List[int]:
        """在信号中找到峰值位置"""
        if len(signal) < 3:
            return []

        peaks = []
        threshold = np.mean(signal) + np.std(signal) * 0.5

        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # 检查与已有峰值的距离
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)

        return peaks

    def _calculate_regularity(self, peaks: List[int]) -> float:
        """计算峰值间距的规律性（越规律置信度越高）"""
        if len(peaks) < 3:
            return 0.0

        # 计算相邻峰值的间距
        distances = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]

        # 计算间距的标准差（越小越规律）
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        if mean_dist == 0:
            return 0.0

        # 规律性 = 1 - (标准差 / 平均值)
        regularity = max(0, 1 - (std_dist / mean_dist))

        return regularity

    def _cluster_lines(self, lines: List[float], threshold: float = 20) -> List[float]:
        """聚类相近的线条"""
        if not lines:
            return []

        lines = sorted(lines)
        clusters = [[lines[0]]]

        for line in lines[1:]:
            if line - clusters[-1][-1] < threshold:
                clusters[-1].append(line)
            else:
                clusters.append([line])

        # 返回每个聚类的平均值
        return [np.mean(cluster) for cluster in clusters]

    def _try_common_sizes(self, image: np.ndarray, sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
        """尝试常见的网格尺寸，选择最合适的"""
        height, width = image.shape[:2]

        best_size = sizes[0]
        best_score = 0

        for rows, cols in sizes:
            # 计算单元格大小
            cell_h = height / rows
            cell_w = width / cols

            # 评分：单元格应该接近正方形，且大小合理
            aspect_ratio = min(cell_w, cell_h) / max(cell_w, cell_h)
            size_score = 1.0 if 20 < cell_w < 100 and 20 < cell_h < 100 else 0.5

            score = aspect_ratio * size_score

            if score > best_score:
                best_score = score
                best_size = (rows, cols)

        return best_size

    def _count_lines(self, line_image: np.ndarray, axis: int) -> int:
        """
        统计线条数量

        Args:
            line_image: 线条图像
            axis: 投影轴（0=水平，1=垂直）

        Returns:
            线条数量
        """
        # 投影到指定轴
        projection = np.sum(line_image, axis=axis)

        # 找到峰值（线条位置）
        threshold = np.max(projection) * 0.3
        peaks = projection > threshold

        # 统计连续的峰值区域数量
        count = 0
        in_peak = False
        for val in peaks:
            if val and not in_peak:
                count += 1
                in_peak = True
            elif not val:
                in_peak = False

        return count

    def recognize_grid(self, image: np.ndarray, rows: int, cols: int) -> List[List[int]]:
        """
        识别整个网格中的所有数字

        Args:
            image: 输入图像
            rows: 网格行数
            cols: 网格列数

        Returns:
            二维数字矩阵
        """
        # 使用 CNN 识别器识别整个网格
        return self.recognizer.recognize_grid(image, rows, cols)

    def process_screenshot(self, bbox: Tuple[int, int, int, int],
                          rows: Optional[int] = None,
                          cols: Optional[int] = None) -> Tuple[List[List[int]], np.ndarray]:
        """
        完整的截图处理流程：截图 -> 网格分割 -> 数字识别

        Args:
            bbox: 屏幕区域边界框
            rows: 网格行数（如果为None则自动检测）
            cols: 网格列数（如果为None则自动检测）

        Returns:
            (数字矩阵, 原始图像)
        """
        # 截取屏幕
        image = self.capture_screen_region(bbox)

        # 检测网格
        if rows is None or cols is None:
            rows, cols, _ = self.detect_grid_auto(image)

        # 识别数字
        grid = self.recognize_grid(image, rows, cols)

        return grid, image

    def visualize_grid(self, image: np.ndarray,
                      grid_cells: List[List[Tuple[int, int, int, int]]],
                      grid: List[List[int]]) -> np.ndarray:
        """
        可视化网格和识别结果

        Args:
            image: 原始图像
            grid_cells: 网格单元格边界框
            grid: 识别出的数字矩阵

        Returns:
            可视化图像
        """
        vis_image = image.copy()

        rows = len(grid_cells)
        cols = len(grid_cells[0]) if grid_cells else 0

        for i in range(rows):
            for j in range(cols):
                x, y, w, h = grid_cells[i][j]

                # 绘制网格线
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 绘制识别出的数字
                digit = grid[i][j]
                text = str(digit) if digit != 0 else ""
                if text:
                    # 计算文本位置（居中）
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = x + (w - text_size[0]) // 2
                    text_y = y + (h + text_size[1]) // 2

                    # 绘制文本
                    cv2.putText(vis_image, text, (text_x, text_y),
                              font, font_scale, (0, 0, 255), thickness)

        return vis_image


def test_image_processor():
    """测试图像处理器"""
    print('初始化图像处理器...')
    processor = ImageProcessor()
    print('图像处理器初始化成功！')


if __name__ == "__main__":
    test_image_processor()
