import asyncio
import functools
import logging
import os
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import viztracer

from rtp_llm.config.py_config_modules import StaticConfig

# 默认配置 - 从 StaticConfig.profiling_debug_config 读取
DEFAULT_OUTPUT_DIR = StaticConfig.profiling_debug_config.log_path
# 最小追踪时长，低于此时长的追踪不会保存文件 (环境变量: VIZTRACER_MIN_DURATION_MS)
DEFAULT_MIN_DURATION_MS = float(
    StaticConfig.profiling_debug_config.viztracer_min_duration_ms
)
# 全局 viztracer 开关，控制是否启用性能追踪 (环境变量: VIZTRACER_ENABLE)
DEFAULT_VIZTRACER_ENABLE = StaticConfig.profiling_debug_config.viztracer_enable

# 全局追踪状态管理
_tracer_lock = threading.Lock()
_active_traces = 0


def is_viztracer_enabled(force_trace=False):
    """检查是否应该启用 viztracer 追踪

    Args:
        force_trace: 是否强制追踪，无视全局开关

    Returns:
        bool: True 表示应该启用追踪
    """
    return DEFAULT_VIZTRACER_ENABLE or force_trace


# nullcontext 已在顶部导入


def get_global_tracer():
    """获取或创建全局 tracer"""
    tracer = viztracer.get_tracer()
    if tracer is None:
        tracer = viztracer.VizTracer(
            tracer_entries=2000000,
            log_gc=True,
            verbose=0,
            ignore_frozen=True,
            log_async=True,
        )
    return tracer


class SmartTraceScope:
    """智能追踪作用域，自动适配同步/异步环境"""

    def __init__(
        self,
        name: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        output_dir: Optional[str] = None,
        auto_save: bool = True,
        force_trace: bool = False,
    ):
        self.name = name
        self.min_duration_ms = min_duration_ms or DEFAULT_MIN_DURATION_MS
        self.output_dir = output_dir or StaticConfig.profiling_debug_config.log_path
        self.auto_save = auto_save
        self.force_trace = force_trace
        self.tracer = None
        self.start_time = None
        self.is_async = False
        self._trace_active = False
        self._is_root_trace = False  # 标记是否是根追踪

    def _generate_filename(self) -> str:
        """生成追踪文件名"""
        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        parts = [str(pid), timestamp]
        if self.name:
            safe_name = "".join(
                c for c in self.name if c.isalnum() or c in ("_", "-", ".")
            )
            parts.append(safe_name)

        if self.is_async:
            parts.append("async")

        return "_".join(parts) + ".json"

    def _should_save(self, duration_ms: float) -> bool:
        """判断是否应该保存追踪结果"""
        if self.force_trace:
            return True
        return duration_ms >= self.min_duration_ms

    def _save_trace(self) -> Optional[str]:
        """保存追踪结果"""
        if not self.auto_save or self.start_time is None or not self._is_root_trace:
            return None

        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000

        if not self._should_save(duration_ms):
            return None

        filepath = Path(self.output_dir) / self._generate_filename()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.tracer.save(str(filepath))
            logging.info(
                "[Trace] Saved to: %s (duration: %.2fms)", filepath, duration_ms
            )
            return str(filepath)
        except Exception as e:
            logging.error("[Trace Error] Failed to save trace: %s", e)
            return None

    def _start_trace(self):
        """开始追踪"""
        global _active_traces
        with _tracer_lock:
            if _active_traces == 0:
                # 只有第一个追踪才真正启动 tracer
                try:
                    self.tracer = get_global_tracer()
                    self.tracer.start()
                    self._is_root_trace = True
                    self._trace_active = True
                except Exception as e:
                    logging.error("[Trace Error] Failed to start trace: %s", e)
                    self._trace_active = False
                    return
            else:
                # 嵌套的追踪只记录时间
                self.tracer = get_global_tracer()
                self._is_root_trace = False
                self._trace_active = True

            _active_traces += 1

        self.start_time = time.perf_counter()

    def _stop_trace(self):
        """停止追踪并清理"""
        global _active_traces
        if not self._trace_active:
            return

        with _tracer_lock:
            _active_traces -= 1

            if self._is_root_trace and _active_traces == 0:
                # 只有根追踪才停止 tracer 并保存
                try:
                    self.tracer.stop()
                    self._save_trace()
                    self.tracer.clear()
                except Exception as e:
                    logging.error("[Trace Error] Failed to stop trace: %s", e)

            self._trace_active = False

    # 同步上下文管理器
    def __enter__(self):
        self.is_async = False
        self._start_trace()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_trace()
        return False

    # 异步上下文管理器
    async def __aenter__(self):
        self.is_async = True
        self._start_trace()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stop_trace()
        return False

    @classmethod
    def decorate(
        cls,
        name: Optional[Union[str, Callable]] = None,
        min_duration_ms: Optional[float] = None,
        output_dir: Optional[str] = None,
        force_trace: bool = False,
    ):
        """作为装饰器使用，支持同步和异步函数"""

        def decorator(func):
            # 提前检查是否启用 viztracer，如果禁用则直接返回原函数
            if not is_viztracer_enabled(force_trace):
                return func

            trace_name = name if isinstance(name, str) else None
            name_gen = name if callable(name) and not isinstance(name, str) else None

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    final_name = trace_name
                    if name_gen:
                        final_name = name_gen(func, *args, **kwargs)
                    elif not final_name:
                        final_name = func.__name__

                    async with cls(
                        final_name, min_duration_ms, output_dir, force_trace=force_trace
                    ):
                        return await func(*args, **kwargs)

                return async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    final_name = trace_name
                    if name_gen:
                        final_name = name_gen(func, *args, **kwargs)
                    elif not final_name:
                        final_name = func.__name__

                    with cls(
                        final_name, min_duration_ms, output_dir, force_trace=force_trace
                    ):
                        return func(*args, **kwargs)

                return sync_wrapper

        return decorator


# 便捷函数和别名
def trace_scope(
    name: Optional[str] = None,
    min_duration_ms: Optional[float] = None,
    output_dir: Optional[str] = None,
    force_trace: bool = False,
):
    if not is_viztracer_enabled(force_trace):
        return nullcontext()
    return SmartTraceScope(name, min_duration_ms, output_dir, force_trace=force_trace)


trace_func: Callable = SmartTraceScope.decorate
