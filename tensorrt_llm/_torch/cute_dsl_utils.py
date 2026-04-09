import os
import platform
import sys

from ..logger import logger

IS_CUTLASS_DSL_AVAILABLE = False

if platform.system() != "Windows":
    try:
        import cutlass  # noqa
    except ImportError:
        # nvidia-cutlass-dsl-libs-base installs cutlass under
        # nvidia_cutlass_dsl/python_packages/ and registers a .pth file to
        # make it importable.  The .pth file can be lost when pip upgrades
        # nvidia-cutlass-dsl (pip ordering bug).  Fall back to adding the
        # path manually.
        try:
            import nvidia_cutlass_dsl
            _pp = os.path.join(nvidia_cutlass_dsl.__path__[0],
                               "python_packages")
            if os.path.isdir(_pp) and _pp not in sys.path:
                sys.path.insert(0, _pp)
            import cutlass  # noqa
        except (ImportError, AttributeError):
            pass

    try:
        import cutlass  # noqa
        import cutlass.cute as cute  # noqa
        logger.info("cutlass dsl is available")
        IS_CUTLASS_DSL_AVAILABLE = True
    except ImportError:
        pass
