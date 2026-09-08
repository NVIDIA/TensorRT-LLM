import platform

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

# Whether the public CuTe DSL package provides the SM107/Rubin helper module.
# Rubin kernels stay disabled when it is absent and callers retain their
# existing fallback paths.
# TODO: flips to True once a Rubin-capable CuTe DSL package ships in the image.
IS_CUTLASS_DSL_RUBIN_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info(f"cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True

        try:
            import cutlass.utils.rubin_helpers  # noqa
        except ImportError:
            pass
        else:
            logger.info("cutlass dsl Rubin helpers are available")
            IS_CUTLASS_DSL_RUBIN_AVAILABLE = True
    except ImportError:
        pass
