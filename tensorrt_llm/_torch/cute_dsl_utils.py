import platform

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

# Whether the additional CuTeDSL kernel set is present. Features that need those
# kernels stay disabled when it is False, falling back to their default paths.
IS_CUTLASS_DSL_EXTENDED_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True
    except ImportError:
        pass
