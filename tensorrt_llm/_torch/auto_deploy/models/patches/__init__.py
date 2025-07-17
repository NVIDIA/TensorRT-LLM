"""AutoDeploy's library of export patches for models.

This file ensures that all publicly listed files/patches in the library folder are auto-imported
and the corresponding patches are registered.
"""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue
    __all__.append(module_name)
    importlib.import_module(f"{__name__}.{module_name}")
