import platform
import traceback

# Re-exported: many modules import get_env_enable_pdl from here.
from .._utils import get_env_enable_pdl as get_env_enable_pdl  # noqa: F401
from ..logger import logger

IS_FLASHINFER_AVAILABLE = False

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

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer_cubin_pool import install_shared_cubin_pool

    install_shared_cubin_pool()
