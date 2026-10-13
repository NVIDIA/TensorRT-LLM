import os
import platform
import traceback

import torch

from ..logger import logger

IS_FLASHINFER_AVAILABLE = False


def get_env_enable_pdl() -> bool:
    enabled = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    if enabled and not getattr(get_env_enable_pdl, "_printed", False):
        logger.info("PDL enabled")
        setattr(get_env_enable_pdl, "_printed", True)
    return enabled


original_value = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")

if platform.system() != "Windows":
    try:
        import flashinfer
        logger.info(f"flashinfer is available: {flashinfer.__version__}")
        major, minor = torch.cuda.get_device_capability()
        sm_version = major * 10 + minor
        IS_FLASHINFER_AVAILABLE = True
        if sm_version >= 100 and sm_version < 110 and sm_version not in [
                100, 103, 107
        ]:
            if original_value is None:
                os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "10.0f"
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )
