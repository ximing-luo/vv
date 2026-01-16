import os
import ctypes
import threading
import time
from transformers import TrainerCallback

class SystemControlCallback(TrainerCallback):
    """
    HuggingFace Trainer 回调函数
    集成键盘监控和系统优化到 transformers 的训练生命周期中
    """
    def __init__(self):
        super().__init__()
        # 将内部方法作为回调传递给监控器
        self.kb_monitor = KeyboardMonitor(self._on_stop_signal)
        self.should_stop = False

    def _on_stop_signal(self):
        """键盘监控触发的内部回调"""
        print("\n[SystemCallback] 接收到停止信号，正在准备安全退出...")
        self.should_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始前执行系统优化"""
        SystemOptimizer.optimize_windows()
        self.kb_monitor.start()
        print("[SystemCallback] 训练环境已优化，键盘监控已就绪")

    def on_step_end(self, args, state, control, **kwargs):
        """每步结束检查是否需要停止"""
        if self.should_stop:
            control.should_training_stop = True
            control.should_evaluate = True   # 停止时触发最后一次评估和推理模拟
            control.should_save = True       # 触发保存断点，确保可以续训
            
            # 关键：禁用 load_best_model_at_end，防止 Trainer 在退出时加载旧的最佳模型
            # 从而确保最后保存的是当前最新的状态（即“最终检查点”包含了最新的训练进度）
            if hasattr(args, "load_best_model_at_end"):
                args.load_best_model_at_end = False
                
            print("[SystemCallback] 正在通过回调请求停止训练，将执行最后一次评估并保存断点...")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束停止监控线程并恢复系统状态"""
        self.kb_monitor.stop()
        SystemOptimizer.restore_windows()

class SystemOptimizer:
    """
    系统优化工具类 (Windows 专用)
    处理进程优先级、电源管理和防休眠
    """
    @staticmethod
    def optimize_windows():
        """执行所有 Windows 优化"""
        if os.name != 'nt':
            return

        try:
            kernel32 = ctypes.windll.kernel32
            
            # 1. 提升进程优先级 (High Priority)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.SetPriorityClass.restype = ctypes.c_bool
            
            # HIGH_PRIORITY_CLASS = 0x00000080
            success = kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)
            if success:
                print(f"[Win-Opt] 进程优先级已设置为 HIGH")
            else:
                print(f"[Win-Opt] 无法设置进程优先级: {kernel32.GetLastError()}")
                
            # 2. 禁用效率模式/电源限制 (EcoQoS / Power Throttling)
            class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                _fields_ = [
                    ("Version", ctypes.c_ulong),
                    ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong),
                ]
            
            ProcessPowerThrottling = 4
            PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
            
            throttling_state = PROCESS_POWER_THROTTLING_STATE()
            throttling_state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
            throttling_state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            throttling_state.StateMask = 0 # Set bit 0 to 0 to DISABLE throttling
            
            kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            kernel32.SetProcessInformation.restype = ctypes.c_bool
            
            success_eco = kernel32.SetProcessInformation(
                kernel32.GetCurrentProcess(),
                ProcessPowerThrottling,
                ctypes.byref(throttling_state),
                ctypes.sizeof(throttling_state)
            )
            if success_eco:
                print(f"[Win-Opt] 电源节流 (EcoQoS) 已禁用")
            else:
                print(f"[Win-Opt] 无法禁用电源节流")

            # 3. 防止系统休眠 (Keep Awake)
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            
            kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
            kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
            
            # 添加 ES_DISPLAY_REQUIRED 以防止屏幕关闭 (通常也关联休眠)
            kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            print(f"[Win-Opt] 系统休眠已禁用 (Execution State Set: System + Display)")
            
        except Exception as e:
            print(f"[Win-Opt] 优化失败: {e}")

    @staticmethod
    def restore_windows():
        """恢复系统默认设置 (允许休眠)"""
        if os.name != 'nt':
            return
        try:
            kernel32 = ctypes.windll.kernel32
            # ES_CONTINUOUS = 0x80000000
            # 使用 ES_CONTINUOUS 清除之前的状态
            kernel32.SetThreadExecutionState(0x80000000)
            print("[Win-Opt] 系统休眠限制已解除 (Execution State Restored)")
        except Exception as e:
            print(f"[Win-Opt] 恢复失败: {e}")

class KeyboardMonitor:
    """
    键盘监控工具，用于安全停止训练
    """
    def __init__(self, stop_callback):
        self.stop_callback = stop_callback
        self.running = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print("[System] 键盘监控已启动 (按 Alt+Shift+P 停止)")

    def _monitor(self):
        while self.running:
            # Alt+Shift+P
            try:
                if (ctypes.windll.user32.GetAsyncKeyState(0x12) & 0x8000) and \
                   (ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000) and \
                   (ctypes.windll.user32.GetAsyncKeyState(0x50) & 0x8000):
                    print("\n[System] 检测到 Alt+Shift+P ...")
                    self.stop_callback()
                    break
            except:
                pass
            time.sleep(0.1)
            
    def stop(self):
        self.running = False
