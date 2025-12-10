"""
自动化控制模块
"""

import time
from typing import List, Tuple
import pydirectinput

# 配置 pydirectinput 以获得最快的响应速度
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.001


class AutomationController:
    """自动化控制器，负责执行鼠标操作"""

    def __init__(self, grid: List[List[int]], image_processor):
        """
        初始化自动化控制器

        Args:
            grid: 数字矩阵（用于计算网格尺寸）
            image_processor: ImageProcessor 实例（用于透视变换坐标映射）
        """

        self.image_processor = image_processor
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.should_stop = False  # 停止标志

        # 使用透视变换模式
        print(f">> [自动化] 输入方法: pydirectinput (透视变换模式)")
        print(f">> [自动化] 虚线四个点坐标: {image_processor.corner_points}")
        print(f">> [自动化] 网格大小: {self.rows} × {self.cols}")

    def grid_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """
        将网格坐标转换为屏幕坐标（单元格中心点）

        Args:
            row: 网格行号
            col: 网格列号

        Returns:
            屏幕坐标 (x, y)
        """
        # 使用透视变换映射坐标
        return self.image_processor.map_grid_to_screen(row, col, self.rows, self.cols)

    def move_to(self, x: int, y: int):
        """
        移动鼠标到指定位置

        Args:
            x: 屏幕X坐标
            y: 屏幕Y坐标
        """
        pydirectinput.moveTo(x, y)

    def execute_path(self, path: List[Tuple[int, int]]):
        """
        执行单条路径的鼠标操作

        Args:
            path: 路径坐标列表 [(row, col), ...]
        """
        if not path or len(path) < 2:
            return

        # 转换为屏幕坐标
        screen_coords = [self.grid_to_screen(row, col) for row, col in path]

        # 起点和终点
        start_x, start_y = screen_coords[0]
        end_x, end_y = screen_coords[-1]

        # 移动到起点
        pydirectinput.moveTo(start_x, start_y)
        time.sleep(0.015)

        # 按下鼠标
        pydirectinput.mouseDown()
        time.sleep(0.02)

        # 分步移动到终点（更平滑）
        steps = 4
        dx = (end_x - start_x) / steps
        dy = (end_y - start_y) / steps

        for i in range(1, steps + 1):
            pydirectinput.moveTo(int(start_x + dx * i), int(start_y + dy * i))

        # 确保到达终点
        pydirectinput.moveTo(end_x, end_y)
        time.sleep(0.015)

        # 释放鼠标
        pydirectinput.mouseUp()
        time.sleep(0.02)

        # 统一延迟
        time.sleep(0.03)

    def stop(self):
        """停止自动化执行"""
        self.should_stop = True
        print('>> [自动化] 收到停止信号')

    def execute_solution(self, solution: List[List[Tuple[int, int]]]):
        """
        执行完整的解决方案

        Args:
            solution: 路径列表，每个路径是一个坐标列表
        """

        # 重置停止标志
        self.should_stop = False
        print(f'>> [自动化] 开始执行，共{len(solution)}步')

        # 执行每条路径
        for i, path in enumerate(solution, 1):
            # 检查是否需要停止
            if self.should_stop:
                print(f'>> [自动化] 已停止（在第{i}步）')
                return

            print(f'>> 执行第{i}/{len(solution)}步: {path}')
            try:
                self.execute_path(path)
            except Exception as e:
                print(f'>> [错误] 执行第{i}步时出错: {e}')
                break

        print('>> [自动化] 执行完成！')
