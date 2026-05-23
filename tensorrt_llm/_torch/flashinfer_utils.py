import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


if platform.system() != "Windows":
    # The default CuTe-DSL fused_add_rmsnorm in flashinfer >=0.6.8 raises
    # cudaErrorLaunchFailure during CUDA-graph capture on Blackwell. The CUDA
    # JIT path is graph-safe; flashinfer reads this env var at norm-module
    # import time, so it must be set before `import flashinfer`.
    os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
