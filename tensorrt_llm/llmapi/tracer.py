__all__ = ["enable_llm_tracer", "VizTracer", "log_sparse", "get_tracer"]

import os
from functools import wraps
from typing import Callable, Optional

from tensorrt_llm.logger import logger

_enable_llm_tracer_ = None


def enable_llm_tracer() -> bool:
    ''' Check if viztracer is enabled. '''
    global _enable_llm_tracer_
    if _enable_llm_tracer_ is not None:
        return _enable_llm_tracer_
    _enable_llm_tracer_ = os.environ.get("TLLM_LLM_ENABLE_TRACER", "0") == "1"
    return _enable_llm_tracer_


try:
    from viztracer import VizTracer, get_tracer, log_sparse
except ImportError:
    if enable_llm_tracer():
        logger.warning(
            "VizTracer is not installed. Disabling tracer in LLM API.")
        _enable_llm_tracer_ = False

if enable_llm_tracer():
    logger.warning("LLM tracer is enabled. This may affect performance.")
else:
    # Dummy placeholders for VizTracer

    class VizTracer:

        def log_instant(self, *args, **kwargs):
            pass

        def register_exit(self, *args, **kwargs):
            pass

        def start(self, *args, **kwargs):
            pass

    def log_sparse(func: Optional[Callable] = None,
                   stack_depth: int = 0,
                   dynamic_tracer_check: bool = False) -> Callable:
        if func is None:
            return lambda f: log_sparse(f, stack_depth, dynamic_tracer_check)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    _viz_tracer_dummy = VizTracer()

    def get_tracer():
        return _viz_tracer_dummy


_global_tracer_ = None


def global_tracer() -> VizTracer:
    ''' Get the global viztracer instance in the current process. '''
    if _global_tracer_ is None:
        return get_tracer()
    return _global_tracer_


def set_global_tracer(tracer: VizTracer):
    ''' Set the global viztracer instance in the current process. '''
    global _global_tracer_
    assert _global_tracer_ is None
    _global_tracer_ = tracer
