"""
数字消除游戏求解器
引入Numba加速和算法优化

核心优化:
1. Numba JIT编译: 核心函数速度提升10-100倍
2. 智能提前终止: 多重退出条件，避免无效计算
3. 优化的Beam Search: 动态算力调整 + 连续无改进检测
4. 高效的Smart Rollback: 激进的残局处理策略
5. 启发式评估: 孤岛惩罚 + 中心引力 + 动态噪音
"""

import numpy as np
import random
import time
import os
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor

# 尝试导入Numba加速
try:
    from numba import njit
    HAS_NUMBA = True
    print(">> [系统] Numba加速引擎已启用")
except ImportError:
    HAS_NUMBA = False
    print(">> [警告] 未检测到Numba，性能将受限")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Numba加速的核心函数
# ============================================================================

@njit(fastmath=True, nogil=True, cache=True)
def _calc_prefix_sum(vals, rows, cols):
    """计算二维前缀和（Numba加速）"""
    P = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for r in range(rows):
        row_sum = 0
        for c in range(cols):
            row_sum += vals[r * cols + c]
            P[r + 1][c + 1] = P[r][c + 1] + row_sum
    return P


@njit(fastmath=True, nogil=True)
def _get_rect_sum(P, r1, c1, r2, c2):
    """获取矩形区域的和（Numba加速）"""
    return P[r2 + 1][c2 + 1] - P[r1][c2 + 1] - P[r2 + 1][c1] + P[r1][c1]


@njit(fastmath=True, nogil=True)
def _count_islands(map_data, rows, cols):
    """
    计算孤岛数量（Numba加速）
    孤岛定义：四周都没有相邻格子的孤立格子
    """
    islands = 0
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if map_data[idx] == 1:
                # 检查四周邻居（提前退出优化）
                if r > 0 and map_data[(r - 1) * cols + c] == 1:
                    continue
                if r < rows - 1 and map_data[(r + 1) * cols + c] == 1:
                    continue
                if c > 0 and map_data[r * cols + (c - 1)] == 1:
                    continue
                if c < cols - 1 and map_data[r * cols + (c + 1)] == 1:
                    continue
                islands += 1
    return islands


@njit(fastmath=True, nogil=True)
def _evaluate_state(score, map_data, rows, cols, w_island, w_fragment):
    """
    启发式状态评估（Numba加速）

    评分公式: H = Score * 2000 - Island_Penalty - Center_Penalty + Noise

    Args:
        score: 当前分数
        map_data: 地图状态
        rows, cols: 地图尺寸
        w_island: 孤岛惩罚权重
        w_fragment: 中心引力权重
    """
    # 基础攻击性（高权重让算法更想拿分）
    h = float(score * 2000)

    # 剩余格子统计
    remaining_count = np.sum(map_data)

    # 孤岛惩罚（残局恐惧机制）
    if w_island > 0:
        islands = _count_islands(map_data, rows, cols)

        # 残局时孤岛惩罚×5（关键优化）
        panic_multiplier = 1.0
        if remaining_count < 20 and remaining_count > 0:
            panic_multiplier = 5.0

        h -= islands * w_island * panic_multiplier

        # 极残局额外惩罚
        if remaining_count < 10:
            h -= remaining_count * w_island * 2.0

    # 中心引力（防止碎片化）
    if w_fragment > 0 and remaining_count > 30:
        center_mass = 0
        center_r, center_c = rows // 2, cols // 2
        for r in range(rows):
            for c in range(cols):
                if map_data[r * cols + c] == 1:
                    dist = abs(r - center_r) + abs(c - center_c)
                    center_mass += (20 - dist)
        h -= center_mass * w_fragment

    # 随机噪音（创造奇迹的源头）
    noise_level = 50.0
    if w_island < 20 and w_fragment < 1:
        # 赌徒/狂战士模式：注入巨大随机性
        noise_level = 2000.0

    if remaining_count < 30:
        # 残局求稳：降低随机性
        noise_level *= 0.2

    h += np.random.random() * noise_level
    return h


@njit(fastmath=True, nogil=True)
def _scan_rectangles(map_data, vals, rows, cols, active_indices):
    """
    扫描所有和为10的矩形（Numba加速）

    使用前缀和优化，时间复杂度: O(n^2)
    """
    moves = []
    n_active = len(active_indices)

    # 构建当前状态的值和计数数组
    current_vals = np.zeros(rows * cols, dtype=np.int32)
    current_counts = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows * cols):
        if map_data[i] == 1:
            current_vals[i] = vals[i]
            current_counts[i] = 1

    # 计算前缀和
    P_val = _calc_prefix_sum(current_vals, rows, cols)
    P_cnt = _calc_prefix_sum(current_counts, rows, cols)

    # 扫描所有可能的矩形
    for i in range(n_active):
        for j in range(i, n_active):
            idx1 = active_indices[i]
            idx2 = active_indices[j]

            r1_raw = idx1 // cols
            c1_raw = idx1 % cols
            r2_raw = idx2 // cols
            c2_raw = idx2 % cols

            min_r = min(r1_raw, r2_raw)
            max_r = max(r1_raw, r2_raw)
            min_c = min(c1_raw, c2_raw)
            max_c = max(c1_raw, c2_raw)

            # 前缀和快速校验
            if _get_rect_sum(P_val, min_r, min_c, max_r, max_c) != 10:
                continue

            count = _get_rect_sum(P_cnt, min_r, min_c, max_r, max_c)
            moves.append((min_r, min_c, max_r, max_c, count))

    return moves


@njit(fastmath=True, nogil=True)
def _apply_move(map_data, rect, cols):
    """应用移动（Numba加速）"""
    new_map = map_data.copy()
    r1, c1, r2, c2 = rect
    for r in range(r1, r2 + 1):
        base = r * cols
        for c in range(c1, c2 + 1):
            new_map[base + c] = 0
    return new_map


# ============================================================================
# 主求解器类
# ============================================================================

class GameSolver:
    """数字消除游戏求解器（性能优化版）"""

    def __init__(self, grid: List[List[int]]):
        """
        初始化求解器

        Args:
            grid: 二维数字矩阵
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

        # 转换为一维数组（用于Numba加速）
        self.grid_flat = np.array([grid[i][j] for i in range(self.rows) for j in range(self.cols)], dtype=np.int8)
        self.vals_flat = self.grid_flat.copy()

        # 初始化地图（1表示未使用，0表示已使用）
        self.initial_map = np.ones(self.rows * self.cols, dtype=np.int8)

        # 最优解
        self.best_solution = None
        self.best_score = 0

    def solve(self,
              mode: str = 'god',
              beam_width: int = 50,
              max_time: int = 30,
              use_rollback: bool = True,
              threads: Optional[int] = None) -> Optional[List[List[Tuple[int, int]]]]:
        """
        求解游戏

        Args:
            mode: 求解模式 ('classic', 'omni', 'god')
            beam_width: 束搜索宽度
            max_time: 最大搜索时间（秒）
            use_rollback: 是否使用智能回溯
            threads: 线程数（None=自动选择，1=单线程，>1=多线程）

        Returns:
            路径列表，每个路径是一个坐标列表 [(row, col), ...]
        """
        # 自动选择最优线程数
        if threads is None:
            cpu_count = os.cpu_count() or 1
            threads = min(cpu_count, 8)  # 最多使用8个线程
            print(f">> [求解器] 自动选择线程数: {threads} (CPU核心数: {cpu_count})")

        if threads > 1:
            return self._solve_multithread(mode, beam_width, max_time, threads)
        else:
            return self._solve_single(mode, beam_width, max_time, use_rollback)

    def _solve_single(self, mode: str, beam_width: int, max_time: int, use_rollback: bool):
        """单线程求解"""
        weights = {'w_island': 100, 'w_fragment': 0.5}

        # 第一阶段搜索
        if mode == 'god':
            # 两阶段搜索：classic阶段 + omni阶段
            p1_weights = weights.copy()
            p1_weights['w_island'] *= 0.5
            p1_state = self._run_beam_search(
                self.initial_map, beam_width, 'classic', 0, [], p1_weights
            )
            best_state = self._run_beam_search(
                p1_state['map'], beam_width, 'omni',
                p1_state['score'], p1_state['path'], weights
            )
        else:
            best_state = self._run_beam_search(
                self.initial_map, beam_width, mode, 0, [], weights
            )

        # 第二阶段：Smart Rollback
        if use_rollback and max_time > 5:
            best_state = self._smart_rollback(
                best_state, beam_width, max_time, weights
            )

        # 转换路径格式
        self.best_score = best_state['score']
        self.best_solution = self._convert_paths(best_state['path'])
        return self.best_solution

    def _run_beam_search(self, start_map, beam_width, mode, start_score, start_path, weights, max_depth=160):
        """
        核心Beam Search算法（优化版）

        优化点:
        1. 动态算力调整（残局×3，中局×1.5）
        2. 连续无改进提前终止
        3. Numba加速的核心函数
        """
        w_island = weights.get('w_island', 0)
        w_fragment = weights.get('w_fragment', 0)

        # 初始化
        initial_h = _evaluate_state(start_score, start_map, self.rows, self.cols, w_island, w_fragment)
        current_beam = [{
            'map': start_map.copy(),
            'path': list(start_path),
            'score': start_score,
            'h_score': initial_h
        }]
        best_state = current_beam[0]
        no_improvement_depth = 0

        # 迭代搜索
        for depth in range(max_depth):
            # 提前终止：连续20步无改进
            if no_improvement_depth > 20:
                break

            # 动态调整算力
            effective_beam = self._get_effective_beam(beam_width, best_state['score'])

            next_candidates = []
            found_any_move = False

            for state in current_beam:
                # 找到所有可用格子
                active_indices = np.where(state['map'] == 1)[0].astype(np.int32)
                if len(active_indices) < 2:
                    if state['score'] > best_state['score']:
                        best_state = state
                    continue

                # 扫描所有可能的矩形移动（Numba加速）
                moves = _scan_rectangles(state['map'], self.vals_flat, self.rows, self.cols, active_indices)

                if len(moves) == 0:
                    if state['score'] > best_state['score']:
                        best_state = state
                    continue

                # 过滤移动
                valid_moves = self._filter_moves(moves, mode)
                if len(valid_moves) == 0:
                    if state['score'] > best_state['score']:
                        best_state = state
                    continue

                found_any_move = True

                # 排序并截断
                valid_moves.sort(key=lambda x: x[4], reverse=True)
                window_size = 100 if best_state['score'] > 120 else 60
                top_moves = valid_moves[:window_size]

                # 生成后继状态
                for move in top_moves:
                    r1, c1, r2, c2, count = move
                    new_map = _apply_move(state['map'], (r1, c1, r2, c2), self.cols)
                    new_score = state['score'] + count
                    h = _evaluate_state(new_score, new_map, self.rows, self.cols, w_island, w_fragment)
                    new_path = list(state['path'])
                    new_path.append([int(r1), int(c1), int(r2), int(c2)])
                    next_candidates.append({
                        'map': new_map,
                        'path': new_path,
                        'score': new_score,
                        'h_score': h
                    })

            if not found_any_move or not next_candidates:
                break

            # 选择最优的N个状态
            next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
            current_beam = next_candidates[:effective_beam]

            # 更新最佳状态并跟踪改进
            if current_beam[0]['score'] > best_state['score']:
                best_state = current_beam[0]
                no_improvement_depth = 0
            else:
                no_improvement_depth += 1

        return best_state

    def _smart_rollback(self, best_state, beam_width, max_time, weights):
        """
        Smart Rollback（优化版）

        优化点:
        1. 降低退出阈值（10→5）
        2. 连续无改进退出（8次）
        3. 高分提前退出（95%覆盖率）
        4. 激进的残局处理（80%概率回滚10-30步）
        """
        start_time = time.time()
        iteration = 0
        best_final = best_state
        no_improvement_count = 0
        max_no_improvement = 8

        while (time.time() - start_time) < max_time:
            iteration += 1
            path = best_final['path']
            path_len = len(path)

            # 提前退出条件（三重保险）
            if path_len < 5:
                break

            if no_improvement_count >= max_no_improvement:
                break

            total_cells = np.sum(self.initial_map)
            if best_final['score'] >= total_cells * 0.95:
                break

            # 80%概率回滚10-30步（死磕残局）
            if random.random() < 0.8:
                rollback_steps = random.randint(10, 30)
            else:
                rollback_steps = random.randint(30, max(31, int(path_len * 0.6)))

            if rollback_steps >= path_len:
                rollback_steps = path_len - 2

            cut_start = path_len - rollback_steps
            prefix_path = path[:cut_start]

            # 重建状态
            temp_map = self.initial_map.copy()
            prefix_score = 0
            for rect in prefix_path:
                r1, c1, r2, c2 = rect
                count = 0
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        idx = r * self.cols + c
                        if temp_map[idx] == 1:
                            count += 1
                            temp_map[idx] = 0
                prefix_score += count

            # 极端性格修补
            repair_weights = weights.copy()
            dice = random.random()
            if dice < 0.4:
                repair_weights['w_island'] = random.randint(150, 300)
            elif dice < 0.7:
                repair_weights['w_island'] = random.randint(-50, -10)
            else:
                repair_weights['w_island'] += random.randint(-20, 20)

            # 使用5倍算力重新搜索
            huge_beam = int(beam_width * 5.0)
            repaired_state = self._run_beam_search(
                temp_map, huge_beam, 'omni',
                prefix_score, prefix_path, repair_weights
            )

            # 更新最优解
            if repaired_state['score'] > best_final['score']:
                best_final = repaired_state
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if repaired_state['score'] == best_final['score']:
                    if random.random() < 0.3:
                        best_final = repaired_state

        return best_final

    def _get_effective_beam(self, beam_width, current_score):
        """动态调整算力"""
        if current_score > 120:
            return int(beam_width * 3.0)  # 残局×3
        elif current_score > 80:
            return int(beam_width * 1.5)  # 中局×1.5
        else:
            return beam_width

    def _filter_moves(self, moves, mode):
        """根据模式过滤移动"""
        valid_moves = []
        for move in moves:
            count = move[4]
            if mode == 'classic':
                if count == 2:
                    valid_moves.append(move)
            else:  # omni 或 god
                if count >= 2:
                    valid_moves.append(move)
        return valid_moves

    def _convert_paths(self, paths):
        """转换路径格式：从矩形格式转换为坐标列表格式"""
        result = []
        for rect in paths:
            r1, c1, r2, c2 = rect
            path = []
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    path.append((r, c))
            result.append(path)
        return result

    def _solve_multithread(self, mode: str, beam_width: int, max_time: int, threads: int):
        """多线程求解（使用参考项目的三种预设）"""
        # 参考 sum10_Nikke 项目的三种求解预设
        PERSONALITIES = [
            {'w_island': 50, 'w_fragment': 2, 'role': 'Balancer (稳健)'},
            {'w_island': 24, 'w_fragment': 0.5, 'role': 'Striker (主攻)'},
            {'w_island': 80, 'w_fragment': 1.0, 'role': 'Heavy (重装)'}
        ]

        args_list = []
        for i in range(threads):
            # 循环使用三种预设
            personality = PERSONALITIES[i % len(PERSONALITIES)].copy()
            args_list.append((
                self.grid_flat.tolist(),
                self.vals_flat.tolist(),
                self.rows,
                self.cols,
                beam_width,
                mode,
                random.randint(0, 1000000) + i,
                max_time,
                personality
            ))

        # 并行求解
        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(_solve_worker, args_list))

        # 选择最优结果
        best_result = max(results, key=lambda x: x['score'])
        self.best_score = best_result['score']
        self.best_solution = self._convert_paths(best_result['path'])
        return self.best_solution

    def get_best_solution_info(self) -> dict:
        """获取最优解信息"""
        if self.best_solution is None:
            return {
                'found': False,
                'steps': 0,
                'score': 0,
                'coverage': 0.0
            }

        total_cells = np.sum(self.initial_map)
        return {
            'found': True,
            'steps': len(self.best_solution),
            'score': self.best_score,
            'coverage': self.best_score / total_cells if total_cells > 0 else 0.0
        }


def _solve_worker(args):
    """多线程工作函数"""
    grid_flat, vals_flat, rows, cols, beam_width, mode, seed, time_limit, personality = args

    np.random.seed(seed)
    random.seed(seed)

    # 重建网格
    grid = [[vals_flat[i * cols + j] for j in range(cols)] for i in range(rows)]

    # 创建求解器
    solver = GameSolver(grid)

    # 求解
    weights = {
        'w_island': personality.get('w_island', 100),
        'w_fragment': personality.get('w_fragment', 0.5)
    }

    initial_map = np.ones(rows * cols, dtype=np.int8)

    # 第一阶段
    if mode == 'god':
        p1_weights = weights.copy()
        p1_weights['w_island'] *= 0.5
        p1_state = solver._run_beam_search(
            initial_map, beam_width, 'classic', 0, [], p1_weights
        )
        best_state = solver._run_beam_search(
            p1_state['map'], beam_width, 'omni',
            p1_state['score'], p1_state['path'], weights
        )
    else:
        best_state = solver._run_beam_search(
            initial_map, beam_width, mode, 0, [], weights
        )

    # Smart Rollback
    best_state = solver._smart_rollback(best_state, beam_width, time_limit, weights)

    return {
        'worker_id': seed,
        'score': best_state['score'],
        'path': best_state['path'],
        'personality': personality
    }


def test_solver():
    """测试求解器"""
    # 测试用例：5x5网格
    grid = [
        [1, 2, 3, 4, 5],
        [6, 4, 3, 2, 1],
        [5, 5, 2, 3, 4],
        [1, 9, 1, 8, 2],
        [3, 4, 5, 1, 4]
    ]

    print("=" * 60)
    print("测试 Performance Optimized 求解器 V3.0")
    print("=" * 60)
    print("\n网格：")
    for row in grid:
        print(row)

    # 测试不同模式
    modes = ['classic', 'omni', 'god']

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"模式: {mode}")
        print(f"{'=' * 60}")

        solver = GameSolver(grid)
        start_time = time.time()
        solution = solver.solve(mode=mode, beam_width=30, max_time=10, use_rollback=True)
        elapsed = time.time() - start_time

        info = solver.get_best_solution_info()

        print(f"求解时间: {elapsed:.2f}秒")
        print(f"找到解: {info['found']}")
        print(f"步数: {info['steps']}")
        print(f"分数: {info['score']}")
        print(f"覆盖率: {info['coverage']:.2%}")

        if solution and len(solution) <= 5:
            print("\n路径：")
            for i, path in enumerate(solution, 1):
                path_sum = sum(grid[r][c] for r, c in path)
                print(f"  路径{i}: {path} (和={path_sum})")


if __name__ == "__main__":
    test_solver()
