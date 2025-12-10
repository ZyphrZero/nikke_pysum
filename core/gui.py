# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import Optional, Tuple, List
import threading
import os
import cv2
import json

# 全局热键支持
import keyboard

# Sun Valley 主题
import sv_ttk


class SettingsDialog:
    """设置对话框"""

    def __init__(self, parent, current_hotkeys):
        self.result = None
        self.hotkeys = current_hotkeys.copy()

        self.dialog = tk.Toplevel(parent)
        self.dialog.title('设置')
        self.dialog.geometry('450x350')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()

        # 居中显示
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f'+{x}+{y}')

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding='10')
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text='快捷键设置', font=('Arial', 12, 'bold')).pack(pady=10)

        hotkeys_frame = ttk.LabelFrame(main_frame, text='快捷键配置', padding='10')
        hotkeys_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.hotkey_entries = {}
        default_hotkeys = {
            'select_region': 'F1',
            'recognize_and_solve': 'F2',
            'execute': 'F3'
        }
        actions = [
            ('select_region', '截取屏幕区域'),
            ('recognize_and_solve', '识别数字并计算最优路径'),
            ('execute', '执行自动化（开始/停止）')
        ]

        for action, label in actions:
            row_frame = ttk.Frame(hotkeys_frame)
            row_frame.pack(fill=tk.X, pady=5)

            ttk.Label(row_frame, text=f'{label}:', width=22).pack(side=tk.LEFT)
            entry = ttk.Entry(row_frame, width=10)
            # 使用current_hotkeys中的值，如果没有则使用默认值
            entry.insert(0, self.hotkeys.get(action, default_hotkeys.get(action, '')))
            entry.pack(side=tk.LEFT, padx=5)
            self.hotkey_entries[action] = entry
            ttk.Label(row_frame, text='(如: F1, Ctrl+S)', foreground='gray').pack(side=tk.LEFT)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text='保存', command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='取消', command=self.cancel).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='恢复默认', command=self.reset_defaults).pack(side=tk.LEFT, padx=5)

    def save(self):
        for action, entry in self.hotkey_entries.items():
            self.hotkeys[action] = entry.get().strip()
        self.result = self.hotkeys
        self.dialog.destroy()

    def cancel(self):
        self.result = None
        self.dialog.destroy()

    def reset_defaults(self):
        defaults = {
            'select_region': 'F1',
            'recognize_and_solve': 'F2',
            'execute': 'F3'
        }
        for action, entry in self.hotkey_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults.get(action, ''))

    def show(self):
        self.dialog.wait_window()
        return self.result


class ScreenSelector:
    """屏幕区域选择器"""

    def __init__(self, callback):
        """
        初始化屏幕选择器

        Args:
            callback: 选择完成后的回调函数，参数为四个角点坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                     顺序：左上、右上、右下、左下
        """
        self.callback = callback
        self.points = []  # 存储已点击的角点
        self.point_markers = []  # 存储点的标记（圆圈）
        self.lines = []  # 存储连线
        self.root = None
        self.canvas = None
        self.hint_text = None

    def start_selection(self):
        """开始选择屏幕区域"""
        # 创建全屏透明窗口
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)  # 半透明
        self.root.attributes('-topmost', True)  # 始终置顶，确保在游戏窗口上方
        self.root.configure(bg='gray')

        # 强制提升窗口层级并获取焦点
        self.root.lift()
        self.root.focus_force()

        # 创建画布
        self.canvas = tk.Canvas(self.root, cursor='cross', bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Button-3>', self.reset_selection)  # 右键重置

        # 绑定键盘事件
        self.root.bind('<Escape>', lambda e: self.cancel_selection())
        self.root.bind('r', lambda e: self.reset_selection())  # R键重置

        # 显示提示文本
        self.hint_text = self.canvas.create_text(
            self.root.winfo_screenwidth() // 2,
            50,
            text='请依次点击虚线区域四个角坐标：左上 → 右上 → 右下 → 左下\n右键或按R重置，ESC取消',
            font=('Arial', 18),
            fill='white',
            justify='center'
        )

        self.root.mainloop()

    def on_click(self, event):
        """鼠标点击事件"""
        # 记录点击位置
        self.points.append((event.x, event.y))

        # 绘制点标记（圆圈）
        marker = self.canvas.create_oval(
            event.x - 8, event.y - 8,
            event.x + 8, event.y + 8,
            fill='red', outline='white', width=2
        )
        self.point_markers.append(marker)

        # 绘制点的序号
        self.canvas.create_text(
            event.x, event.y,
            text=str(len(self.points)),
            font=('Arial', 12, 'bold'),
            fill='white'
        )

        # 如果不是第一个点，绘制连线
        if len(self.points) > 1:
            prev_point = self.points[-2]
            line = self.canvas.create_line(
                prev_point[0], prev_point[1],
                event.x, event.y,
                fill='red', width=3
            )
            self.lines.append(line)

        # 更新提示文本
        self.update_hint()

        # 如果已经点击了4个点，完成选择
        if len(self.points) == 4:
            # 绘制最后一条连线（连接第4个点和第1个点）
            first_point = self.points[0]
            line = self.canvas.create_line(
                event.x, event.y,
                first_point[0], first_point[1],
                fill='red', width=3
            )
            self.lines.append(line)

            # 延迟500ms后关闭窗口，让用户看到完整的四边形
            self.root.after(500, self.complete_selection)

    def update_hint(self):
        """更新提示文本"""
        hints = [
            '请点击左上角',
            '请点击右上角',
            '请点击右下角',
            '请点击左下角',
            '完成！'
        ]

        current_hint = hints[len(self.points)]
        self.canvas.itemconfig(
            self.hint_text,
            text=f'已选择 {len(self.points)}/4 个角点 - {current_hint}\n右键或按R重置，ESC取消'
        )

    def reset_selection(self, event=None):
        """重置选择"""
        # 清除所有标记和连线
        for marker in self.point_markers:
            self.canvas.delete(marker)
        for line in self.lines:
            self.canvas.delete(line)

        # 清空数据
        self.points = []
        self.point_markers = []
        self.lines = []

        # 重置提示文本
        self.canvas.itemconfig(
            self.hint_text,
            text='请依次点击虚线区域四个角坐标：左上 → 右上 → 右下 → 左下\n右键或按R重置，ESC取消'
        )

    def complete_selection(self):
        """完成选择"""
        # 关闭窗口
        self.root.destroy()

        # 调用回调函数，传递四个角点坐标
        self.callback(self.points)

    def cancel_selection(self):
        """取消选择"""
        if self.root:
            self.root.destroy()


class GameAssistantGUI:

    def __init__(self):
        """初始化GUI"""
        self.root = tk.Tk()
        self.root.title('小游戏工具')
        self.root.geometry('820x750')

        # 应用 Sun Valley 主题
        sv_ttk.set_theme("light")

        # 数据
        self.selected_region = None  # 选择的屏幕区域
        self.recognized_grid = None  # 识别出的数字矩阵
        self.solution = None  # 求解结果
        self.original_image = None  # 原始截图
        self.image_processor = None  # 图像处理器实例（保存以便自动化使用）
        self.automation_controller = None  # 自动化控制器实例
        self.is_executing = False  # 是否正在执行自动化

        # 配置文件
        self.config_file = 'config.json'

        # 默认快捷键配置
        self.default_hotkeys = {
            'select_region': 'F1',
            'recognize_and_solve': 'F2',
            'execute': 'F3'
        }

        # 当前快捷键配置
        self.hotkeys = self.default_hotkeys.copy()

        # 已注册的全局热键列表
        self.registered_hotkeys = []

        # 创建界面
        self.create_widgets()

        # 加载保存的坐标和快捷键
        self.load_coordinates()
        self.load_hotkeys()

        # 绑定快捷键
        self.setup_hotkeys()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding='10')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # 配置区域（前置条件）
        config_frame = ttk.LabelFrame(main_frame, text='配置区域（前置条件）', padding='10')
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # 网格尺寸选择
        grid_size_frame = ttk.Frame(config_frame)
        grid_size_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(grid_size_frame, text='网格尺寸:').pack(side=tk.LEFT, padx=5)

        self.grid_size_var = tk.StringVar(value='15x9')

        ttk.Radiobutton(
            grid_size_frame,
            text='15×9',
            variable=self.grid_size_var,
            value='15x9'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            grid_size_frame,
            text='16×10',
            variable=self.grid_size_var,
            value='16x10'
        ).pack(side=tk.LEFT, padx=5)

        # 截取屏幕区域按钮
        btn_config = ttk.Frame(config_frame)
        btn_config.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.btn_select = ttk.Button(
            btn_config,
            text='虚线四个点坐标',
            command=self.select_screen_region
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)

        # 坐标状态显示
        self.coords_status_label = ttk.Label(btn_config, text='未配置', foreground='red')
        self.coords_status_label.pack(side=tk.LEFT, padx=10)

        # 设置按钮
        ttk.Button(
            btn_config,
            text='⚙ 设置',
            command=self.open_settings
        ).pack(side=tk.RIGHT, padx=5)

        # 主要操作区域
        control_frame = ttk.LabelFrame(main_frame, text='主要操作', padding='10')
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        # 操作按钮
        btn_row = ttk.Frame(control_frame)
        btn_row.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.btn_recognize_and_solve = ttk.Button(
            btn_row,
            text='识别数字并计算最优路径',
            command=self.recognize_and_solve,
            state=tk.DISABLED
        )
        self.btn_recognize_and_solve.pack(side=tk.LEFT, padx=5)

        self.btn_execute = ttk.Button(
            btn_row,
            text='执行自动化',
            command=self.execute_automation,
            state=tk.DISABLED
        )
        self.btn_execute.pack(side=tk.LEFT, padx=5)

        # 求解参数配置
        solver_frame = ttk.Frame(control_frame)
        solver_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(solver_frame, text='求解模式:').pack(side=tk.LEFT, padx=5)
        self.solver_mode_var = tk.StringVar(value='god')
        mode_combo = ttk.Combobox(solver_frame, textvariable=self.solver_mode_var, width=10, state='readonly')
        mode_combo['values'] = ('classic', 'omni', 'god')
        mode_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(solver_frame, text='束宽度:').pack(side=tk.LEFT, padx=5)
        self.beam_width_var = tk.StringVar(value='50')
        ttk.Entry(solver_frame, textvariable=self.beam_width_var, width=8).pack(side=tk.LEFT)

        ttk.Label(solver_frame, text='最大时间(秒):').pack(side=tk.LEFT, padx=5)
        self.max_time_var = tk.StringVar(value='30')
        ttk.Entry(solver_frame, textvariable=self.max_time_var, width=8).pack(side=tk.LEFT)

        ttk.Label(solver_frame, text='线程数:').pack(side=tk.LEFT, padx=5)
        optimal_threads = min(os.cpu_count() or 1, 8)
        self.threads_var = tk.StringVar(value=str(optimal_threads))
        self.threads_entry = ttk.Entry(solver_frame, textvariable=self.threads_var, width=8)
        self.threads_entry.pack(side=tk.LEFT)

        # 自动最优线程勾选框
        self.auto_threads_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            solver_frame,
            text='自动最优线程',
            variable=self.auto_threads_var,
            command=self._on_auto_threads_changed
        ).pack(side=tk.LEFT, padx=5)

        # 初始状态：勾选时禁用输入框
        self.threads_entry.config(state='disabled')

        # 显示区域
        display_frame = ttk.LabelFrame(main_frame, text='识别结果与日志', padding='10')
        display_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # 使用滚动文本框显示信息
        self.log_text = scrolledtext.ScrolledText(
            display_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Consolas', 10)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 状态栏
        self.status_var = tk.StringVar(value='就绪')
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)

        # 初始日志
        self.log('欢迎使用小游戏工具！')
        self.log('')
        self.log('使用步骤：')
        self.log('1. 点击"截取屏幕区域"按钮，框选游戏区域')
        self.log('2. 点击"识别数字并计算最优路径"按钮，自动识别并求解')
        self.log('3. 点击"执行自动化"按钮，自动执行消除操作')
        self.log('')
        self.log('**代码中模型及算法来自https://github.com/Small-tailqwq/sum10_Nikke，感谢原作者的贡献！**')
        self.log('-' * 60)

    def log(self, message: str):
        """
        在日志区域显示消息

        Args:
            message: 要显示的消息
        """
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update()

    def save_coordinates(self):
        """保存坐标到JSON文件"""
        if self.selected_region:
            import json
            import os
            # 读取现有配置
            config = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                except:
                    pass

            # 更新屏幕坐标部分
            config['screen_coordinates'] = {
                'corner_points': self.selected_region,
                'grid_size': self.grid_size_var.get()
            }

            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                self.log(f'✓ 坐标已保存到 {self.config_file}')
            except Exception as e:
                self.log(f'保存坐标失败: {str(e)}')

    def load_coordinates(self):
        """从JSON文件加载坐标"""
        import json
        import os
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                data = config.get('screen_coordinates', {})
                self.selected_region = data.get('corner_points')
                saved_grid_size = data.get('grid_size', '15x9')
                self.grid_size_var.set(saved_grid_size)

                if self.selected_region:
                    self.log(f'✓ 已加载保存的坐标: {self.config_file}')
                    self.log(f'  虚线四个点坐标: {self.selected_region}')
                    self.log(f'  网格尺寸: {saved_grid_size}')
                    self.log('配置完成，可以开始识别数字')
                    # 更新状态显示
                    self.coords_status_label.config(text='✓ 已配置', foreground='green')
                    # 启用主要操作按钮
                    self.btn_recognize_and_solve.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f'加载坐标失败: {str(e)}')
                self.coords_status_label.config(text='✗ 加载失败', foreground='red')
        else:
            self.log('未找到保存的坐标，请先配置屏幕区域')
            self.coords_status_label.config(text='✗ 未配置', foreground='red')

    def load_hotkeys(self):
        """从JSON文件加载快捷键配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                loaded_hotkeys = config.get('hotkeys', {})

                # 合并默认配置和加载的配置，确保所有必需的键都存在
                self.hotkeys = self.default_hotkeys.copy()
                self.hotkeys.update(loaded_hotkeys)

                self.log(f'✓ 已加载快捷键配置: {self.config_file}')
            except Exception as e:
                self.log(f'加载快捷键配置失败: {str(e)}，使用默认配置')
                self.hotkeys = self.default_hotkeys.copy()
        else:
            self.log('使用默认快捷键配置')
            self.hotkeys = self.default_hotkeys.copy()

        # 更新按钮文本以显示快捷键
        self.update_button_texts()

    def save_hotkeys(self):
        """保存快捷键配置到JSON文件"""
        try:
            import json
            import os
            # 读取现有配置
            config = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                except:
                    pass

            # 更新快捷键部分
            config['hotkeys'] = self.hotkeys

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.log(f'✓ 快捷键配置已保存到 {self.config_file}')
        except Exception as e:
            self.log(f'保存快捷键配置失败: {str(e)}')

    def update_button_texts(self):
        """更新按钮文本以显示快捷键"""
        # 更新选择区域按钮
        select_key = self.hotkeys.get('select_region', 'F1')
        self.btn_select.config(text=f'虚线四个点坐标 ({select_key})')

        # 更新识别并求解按钮
        recognize_and_solve_key = self.hotkeys.get('recognize_and_solve', 'F2')
        self.btn_recognize_and_solve.config(text=f'识别数字并计算最优路径 ({recognize_and_solve_key})')

        # 更新执行按钮
        execute_key = self.hotkeys.get('execute', 'F3')
        if self.is_executing:
            self.btn_execute.config(text=f'停止执行')
        else:
            self.btn_execute.config(text=f'执行自动化 ({execute_key})')

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self.root, self.hotkeys)
        result = dialog.show()

        if result:
            self.hotkeys = result
            self.save_hotkeys()
            # 重新绑定快捷键
            self.setup_hotkeys()
            # 更新按钮文本
            self.update_button_texts()
            self.log('快捷键配置已更新')

    def setup_hotkeys(self):
        """设置全局热键"""
        # 先清理旧的热键
        self.cleanup_hotkeys()

        # 绑定新的快捷键
        action_map = {
            'select_region': self.select_screen_region,
            'recognize_and_solve': self.recognize_and_solve,
            'execute': self.execute_automation
        }

        action_names = {
            'select_region': '截取屏幕区域',
            'recognize_and_solve': '识别数字并计算最优路径',
            'execute': '执行自动化（开始/停止）'
        }

        self.log('全局热键已启用（游戏内也可触发）:')
        for action, func in action_map.items():
            hotkey = self.hotkeys.get(action, '')
            if hotkey:
                try:
                    # 转换快捷键格式（keyboard 库使用小写）
                    hotkey_lower = hotkey.lower()

                    # 注册全局热键
                    keyboard.add_hotkey(hotkey_lower, func, suppress=False)
                    self.registered_hotkeys.append(hotkey_lower)

                    self.log(f'  {hotkey} - {action_names.get(action, action)}')
                except Exception as e:
                    self.log(f'绑定全局热键 {hotkey} 失败: {str(e)}')
                    self.log(f'  提示：可能需要管理员权限运行程序')

        self.log('-' * 60)

    def cleanup_hotkeys(self):
        """清理所有已注册的全局热键"""
        if self.registered_hotkeys:
            try:
                for hotkey in self.registered_hotkeys:
                    try:
                        keyboard.remove_hotkey(hotkey)
                    except:
                        pass
                self.registered_hotkeys.clear()
            except Exception as e:
                print(f'清理热键失败: {str(e)}')

    def select_screen_region(self):
        """选择屏幕区域"""
        self.log('请在屏幕上拖拽选择游戏区域...')
        self.status_var.set('等待选择屏幕区域...')

        # 创建选择器
        selector = ScreenSelector(self.on_region_selected)

        # 在新线程中启动选择器（避免阻塞主界面）
        threading.Thread(target=selector.start_selection, daemon=True).start()

    def on_region_selected(self, corner_points: List[Tuple[int, int]]):
        """
        屏幕区域选择完成的回调

        Args:
            corner_points: 虚线四个点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                          顺序：左上、右上、右下、左下
        """
        self.selected_region = corner_points
        self.log(f'已选择虚线四个点坐标区域:')
        self.log(f'  左上: {corner_points[0]}')
        self.log(f'  右上: {corner_points[1]}')
        self.log(f'  右下: {corner_points[2]}')
        self.log(f'  左下: {corner_points[3]}')
        self.status_var.set('已选择屏幕区域')
        # === 添加：保存测试截图用于检查 ===
        # try:
        #     from core.image_processor import ImageProcessor
        #     import os
        #     from datetime import datetime
        #
        #     # 创建调试目录
        #     debug_dir = 'debug_screenshots'
        #     os.makedirs(debug_dir, exist_ok=True)
        #
        #     # 创建图像处理器
        #     processor = ImageProcessor()
        #
        #     # 使用透视变换截取屏幕区域
        #     screenshot = processor.capture_and_warp_perspective(corner_points)
        #
        #     # 保存原始截图
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     screenshot_path = os.path.join(debug_dir, f'screenshot_{timestamp}.png')
        #     cv2.imwrite(screenshot_path, screenshot)
        #
        #     self.log(f'✓ 已保存透视变换后的截图: {screenshot_path}')
        #     self.log(f'  截图尺寸: {screenshot.shape[1]} × {screenshot.shape[0]} 像素')
        #
        #     # 使用用户选择的网格尺寸
        #     grid_size = self.grid_size_var.get()
        #     if grid_size == '15x9':
        #         rows, cols = 15, 9
        #     else:  # 16x10
        #         rows, cols = 16, 10
        #
        #     self.log(f'使用网格尺寸: {rows} × {cols}')
        #
        #     # 生成网格分割可视化
        #     grid_cells = processor.detect_grid(screenshot, rows, cols)
        #
        #     # 创建可视化图像（绘制网格线）
        #     vis_image = screenshot.copy()
        #     for i in range(rows):
        #         for j in range(cols):
        #             x, y, w, h = grid_cells[i][j]
        #             cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #             # 在每个格子中心标注坐标
        #             center_x = x + w // 2
        #             center_y = y + h // 2
        #             cv2.putText(vis_image, f'{i},{j}', (center_x-20, center_y),
        #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        #
        #     # 保存网格可视化
        #     grid_path = os.path.join(debug_dir, f'grid_{timestamp}.png')
        #     cv2.imwrite(grid_path, vis_image)
        #     self.log(f'✓ 已保存网格分割图: {grid_path}')
        #     self.log(f'✓ 检测到网格尺寸: {rows} × {cols}')
        #
        #     self.log('请检查截图和网格分割是否正确，然后点击"识别数字"按钮')
        #     self.log('-' * 60)
        #
        # except Exception as e:
        #     self.log(f'保存测试截图失败: {str(e)}')
        #     import traceback
        #     self.log(traceback.format_exc())

        # 保存坐标到JSON文件
        self.save_coordinates()

        # 更新状态显示
        self.coords_status_label.config(text='✓ 已配置', foreground='green')

        # 启用识别按钮
        self.btn_recognize_and_solve.config(state=tk.NORMAL)

    def _on_auto_threads_changed(self):
        """处理自动最优线程勾选框状态改变"""
        if self.auto_threads_var.get():
            # 勾选：禁用输入框，自动填写最优线程数
            self.threads_entry.config(state='disabled')
            optimal_threads = min(os.cpu_count() or 1, 8)
            self.threads_var.set(str(optimal_threads))
        else:
            # 取消勾选：启用输入框，允许手动输入
            self.threads_entry.config(state='normal')

    def recognize_and_solve(self):
        """识别数字并自动计算最优路径"""
        # 检查按钮状态，如果按钮被禁用则不执行（防止快捷键绕过）
        if str(self.btn_recognize_and_solve['state']) == 'disabled':
            return

        if not self.selected_region:
            messagebox.showerror('错误', '请先选择屏幕区域！')
            return

        # 获取用户选择的网格尺寸
        grid_size = self.grid_size_var.get()
        if grid_size == '15x9':
            rows, cols = 15, 9
        else:  # 16x10
            rows, cols = 16, 10

        self.log(f'正在使用 CNN 模型识别数字...')
        self.log(f'网格尺寸: {rows} × {cols}')
        self.status_var.set('识别中...')

        # 禁用按钮，防止重复点击
        self.btn_recognize_and_solve.config(state=tk.DISABLED)
        self.btn_execute.config(state=tk.DISABLED)

        # 在后台线程中执行识别，并在完成后自动求解
        thread = threading.Thread(
            target=self._recognize_and_solve_thread,
            args=(rows, cols),
            daemon=True
        )
        thread.start()

    def _recognize_and_solve_thread(self, rows: int, cols: int):
        """
        在后台线程中执行识别操作，识别完成后自动求解

        Args:
            rows: 网格行数
            cols: 网格列数
        """
        try:
            # 导入图像处理器
            from core.image_processor import ImageProcessor

            # 创建图像处理器（使用CNN模型）
            processor = ImageProcessor()

            # 使用透视变换截取并处理图像
            warped_image = processor.capture_and_warp_perspective(self.selected_region)

            # 识别数字（使用固定的网格尺寸）
            self.log('正在识别数字...')
            recognized_grid = processor.recognize_grid(warped_image, rows, cols)

            # 保存 processor 实例供后续使用
            self.image_processor = processor

            # 在主线程中更新GUI，传递auto_solve=True标志
            self.root.after(0, self._on_recognition_complete, recognized_grid, warped_image, None, True)

        except Exception as e:
            # 在主线程中显示错误
            self.root.after(0, self._on_recognition_complete, None, None, str(e), False)

    def _on_recognition_complete(self, recognized_grid, original_image, error, auto_solve=False):
        """
        识别完成后的回调（在主线程中执行）

        Args:
            recognized_grid: 识别出的数字矩阵
            original_image: 原始截图
            error: 错误信息（如果有）
            auto_solve: 识别成功后是否自动求解
        """
        if error:
            # 识别失败
            self.log(f'识别失败: {str(error)}')
            self.status_var.set('识别失败')
            messagebox.showerror('错误', f'识别失败：{str(error)}')
            # 重新启用识别按钮
            self.btn_recognize_and_solve.config(state=tk.NORMAL)
            self.btn_execute.config(state=tk.NORMAL)
        else:
            # 识别成功
            self.recognized_grid = recognized_grid
            self.original_image = original_image

            # 显示识别结果
            self.log('识别完成！识别结果：')
            self.log('-' * 60)
            for i, row in enumerate(self.recognized_grid):
                row_str = '  '.join(f'{num:2d}' for num in row)
                self.log(f'第{i+1}行: {row_str}')
            self.log('-' * 60)

            # 统计信息
            total_cells = len(self.recognized_grid) * len(self.recognized_grid[0])
            non_zero_cells = sum(1 for row in self.recognized_grid for num in row if num != 0)
            self.log(f'网格大小: {len(self.recognized_grid)} × {len(self.recognized_grid[0])}')
            self.log(f'非零格子数: {non_zero_cells}/{total_cells}')

            self.status_var.set('识别完成')

            # 启用识别按钮
            self.btn_recognize_and_solve.config(state=tk.NORMAL)

            # 如果是自动求解模式，延迟一点时间后自动开始求解
            if auto_solve:
                # 延迟500ms后自动求解，给用户时间看到识别结果
                self.root.after(500, self._auto_solve)
            else:
                # 非自动模式，启用执行按钮
                self.btn_execute.config(state=tk.NORMAL)

    def _auto_solve(self):
        """自动求解（识别完成后调用）"""
        if self.recognized_grid:
            self.log('\n=== 开始自动求解 ===\n')
            self.solve_game()

    def solve_game(self):
        """计算最优路径（异步执行）"""
        if not self.recognized_grid:
            messagebox.showerror('错误', '请先识别数字！')
            return

        # 获取求解参数
        try:
            mode = self.solver_mode_var.get()
            beam_width = int(self.beam_width_var.get())
            max_time = int(self.max_time_var.get())

            # 根据勾选状态决定线程数
            if self.auto_threads_var.get():
                threads = None  # 自动选择最优线程数
            else:
                threads = int(self.threads_var.get())
        except ValueError:
            messagebox.showerror('错误', '求解参数格式错误！请输入有效的数字。')
            return

        # 验证参数
        if beam_width < 10 or beam_width > 200:
            messagebox.showerror('错误', '束宽度应在10-200之间！')
            return
        if max_time < 5 or max_time > 300:
            messagebox.showerror('错误', '最大时间应在5-300秒之间！')
            return
        if threads is not None and (threads < 1 or threads > 16):
            messagebox.showerror('错误', '线程数应在1-16之间！')
            return

        self.log(f'正在计算最优路径...')
        self.log(f'  模式: {mode}')
        self.log(f'  束宽度: {beam_width}')
        self.log(f'  最大时间: {max_time}秒')
        if threads is None:
            self.log(f'  线程数: 自动选择')
        else:
            self.log(f'  线程数: {threads}')
        self.status_var.set('计算中...')

        # 禁用按钮
        self.btn_recognize_and_solve.config(state=tk.DISABLED)
        self.btn_execute.config(state=tk.DISABLED)

        # 在后台线程中执行求解
        thread = threading.Thread(
            target=self._solve_game_thread,
            args=(mode, beam_width, max_time, threads),
            daemon=True
        )
        thread.start()

    def _solve_game_thread(self, mode, beam_width, max_time, threads):
        """
        在后台线程中执行求解

        Args:
            mode: 求解模式
            beam_width: 束宽度
            max_time: 最大时间
            threads: 线程数
        """
        try:
            # 导入求解器
            from core.game_solver import GameSolver

            # 创建求解器（只搜索矩形区域）
            solver = GameSolver(self.recognized_grid)

            # 求解
            solution = solver.solve(
                mode=mode,
                beam_width=beam_width,
                max_time=max_time,
                use_rollback=True,
                threads=threads
            )

            # 获取求解信息
            info = solver.get_best_solution_info()

            # 在主线程中更新GUI
            self.root.after(0, self._on_solve_complete, solution, info, None)

        except Exception as e:
            # 在主线程中显示错误
            self.root.after(0, self._on_solve_complete, None, None, e)

    def _on_solve_complete(self, solution, info, error):
        """
        求解完成后的回调（在主线程中执行）

        Args:
            solution: 求解结果
            info: 求解信息
            error: 错误信息（如果有）
        """
        if error:
            # 求解失败
            self.log(f'计算失败: {str(error)}')
            self.status_var.set('计算失败')
            messagebox.showerror('错误', f'计算失败：{str(error)}')
            # 重新启用识别按钮
            self.btn_recognize_and_solve.config(state=tk.NORMAL)
            self.btn_execute.config(state=tk.NORMAL)
        else:
            # 求解成功
            self.solution = solution

            # 显示结果
            self.log('计算完成！')
            self.log('-' * 60)

            if info['found']:
                self.log(f'找到解决方案！')
                self.log(f'步数: {info["steps"]}')
                self.log(f'分数: {info["score"]}')
                self.log(f'覆盖率: {info["coverage"]:.2%}')
                self.log('')
                self.log('消除路径：')

                # 只显示前10条路径，避免日志过长
                display_count = min(10, len(self.solution))
                for i, path in enumerate(self.solution[:display_count], 1):
                    # 计算路径和
                    path_sum = sum(self.recognized_grid[r][c] for r, c in path)
                    path_str = ' -> '.join(f'({r},{c})' for r, c in path)
                    self.log(f'  路径{i}: {path_str}')
                    self.log(f'         数字: {[self.recognized_grid[r][c] for r, c in path]} (和={path_sum})')

                if len(self.solution) > display_count:
                    self.log(f'  ... 还有 {len(self.solution) - display_count} 条路径未显示')

                self.status_var.set('计算完成')

                # 启用按钮
                self.btn_recognize_and_solve.config(state=tk.NORMAL)
                self.btn_execute.config(state=tk.NORMAL)
            else:
                self.log('未找到完美解决方案')
                self.log('可能原因：')
                self.log('  1. OCR识别有误，请检查识别结果')
                self.log('  2. 游戏无解')
                self.log('  3. 需要更长的计算时间或更大的束宽度')
                self.status_var.set('未找到解')
                # 重新启用识别按钮
                self.btn_recognize_and_solve.config(state=tk.NORMAL)
                self.btn_execute.config(state=tk.NORMAL)

            self.log('-' * 60)

    def execute_automation(self):
        """执行自动化（切换开始/停止）"""
        # 检查按钮状态，如果按钮被禁用则不执行（防止快捷键绕过）
        if str(self.btn_execute['state']) == 'disabled':
            return

        # 如果正在执行，则停止
        if self.is_executing:
            self.log('正在停止自动化执行...')
            if self.automation_controller:
                self.automation_controller.stop()
            return

        # 如果没有解决方案，提示错误
        if not self.solution:
            self.log('错误：请先计算最优路径！')
            return

        # 开始执行
        self.is_executing = True
        self.log(f'开始执行自动化，共{len(self.solution)}步')
        self.status_var.set('执行中...')

        # 更新按钮文本以显示停止快捷键
        self.update_button_texts()

        # 禁用其他按钮
        self.btn_recognize_and_solve.config(state=tk.DISABLED)

        # 在后台线程中执行自动化
        thread = threading.Thread(
            target=self._execute_automation_thread,
            daemon=True
        )
        thread.start()

    def _execute_automation_thread(self):
        """
        在后台线程中执行自动化操作
        """
        try:
            # 导入自动化模块
            from core.automation import AutomationController

            # 创建自动化控制器（使用透视变换）
            self.automation_controller = AutomationController(
                self.recognized_grid,
                self.image_processor
            )

            # 执行自动化
            self.log('开始执行，请勿移动鼠标...')
            self.automation_controller.execute_solution(self.solution)

            # 在主线程中更新GUI
            self.root.after(0, self._on_automation_complete, None)

        except Exception as e:
            # 在主线程中显示错误
            self.root.after(0, self._on_automation_complete, e)

    def _on_automation_complete(self, error):
        """
        自动化完成后的回调（在主线程中执行）

        Args:
            error: 错误信息（如果有）
        """
        # 重置执行状态
        self.is_executing = False

        if error:
            # 执行失败
            self.log(f'执行失败: {str(error)}')
            self.status_var.set('执行失败')
        else:
            # 执行成功
            self.log('自动化执行完成！')
            self.status_var.set('执行完成')

        # 恢复按钮文本和状态
        self.update_button_texts()
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_recognize_and_solve.config(state=tk.NORMAL)

    def on_closing(self):
        """窗口关闭时的处理"""
        # 清理全局热键
        self.cleanup_hotkeys()
        # 销毁窗口
        self.root.destroy()

    def run(self):
        """运行GUI"""
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """主函数"""
    app = GameAssistantGUI()
    app.run()


if __name__ == '__main__':
    main()
