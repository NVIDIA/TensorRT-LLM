import os
import platform
import traceback

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl():
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


def gen_env_disable_fused_add_rmsnorm_pdl():
    return os.environ.get("TRTLLM_DISABLE_FUSED_ADD_RMSNORM_PDL", "0") == "1"


if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        IS_FLASHINFER_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
