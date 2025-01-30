import importlib.util

from .backend_registry import create_py_executor, has_backend
from .pytorch_model_registry import create_pytorch_model_based_executor
from .simple_model_registy import create_simple_executor
from .trt_python_model_registry import create_trt_python_executor

# check if auto_deploy backend is available
if importlib.util.find_spec("auto_deploy"):
    import auto_deploy  # noqa: F401

__all__ = [
    "create_py_executor",
    "has_backend",
    "create_pytorch_model_based_executor",
    "create_trt_python_executor",
    "create_simple_executor",
]
