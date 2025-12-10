# -*- coding: utf-8 -*-

import sys
import os
import ctypes

# DPI感知
if sys.platform == 'win32':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass

    # 隐藏控制台窗口
    try:
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except:
        pass


def is_admin():
    """检查是否以管理员身份运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """请求管理员权限并重新启动程序"""
    try:
        script = os.path.abspath(sys.argv[0])
        params = ' '.join([script] + sys.argv[1:])
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, params, None, 1
        )
        if ret > 32:
            sys.exit(0)
    except Exception as e:
        print(f'>> [错误] 请求管理员权限失败: {e}')


def main():
    """主函数"""
    # 检查管理员权限
    if not is_admin():
        print('>> [警告] 需要管理员权限才能正常工作')
        print('>> 正在请求管理员权限...')
        run_as_admin()
        print('>> [警告] 未能获取管理员权限，程序可能无法正常工作')
        input('按回车键继续...')
    else:
        print('>> [✓] 已以管理员身份运行')

    print()

    # 启动GUI
    try:
        from core.gui import GameAssistantGUI
        app = GameAssistantGUI()
        app.run()
    except Exception as e:
        print(f'>> [错误] 启动失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
