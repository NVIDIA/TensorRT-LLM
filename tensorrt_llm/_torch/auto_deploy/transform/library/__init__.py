"""AutoDeploy's library of transforms.

This file ensures that all publicly listed files/transforms in the library folder are auto-imported
and the corresponding transforms are registered.
"""

import importlib
import logging
import pkgutil

from ..._compat import TRTLLM_AVAILABLE

_logger = logging.getLogger(__name__)

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{module_name}")
        __all__.append(module_name)
    except (ModuleNotFoundError, ImportError, AttributeError) as exc:
        if not TRTLLM_AVAILABLE:
            _logger.debug("Skipping transform %s (standalone mode): %s", module_name, exc)
        else:
            raise
