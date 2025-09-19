import platform
import traceback

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "cutlass dsl is not installed properly, please try pip install nvidia-cutlass-dsl"
        )
