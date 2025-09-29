"""AutoDeploy's library of transforms.

This file ensures that all publicly listed files/transforms in the library folder are auto-imported
and the corresponding transforms are registered.
"""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue

    # Handle optional dependencies with conditional imports
    if module_name == "visualization":
        try:
            importlib.import_module(f"{__name__}.{module_name}")
            __all__.append(module_name)
        except ImportError:
            pass  # Skip visualization if model_explorer is not available
    else:
        importlib.import_module(f"{__name__}.{module_name}")
        __all__.append(module_name)
