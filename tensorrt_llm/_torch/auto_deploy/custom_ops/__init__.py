"""Custom ops and make sure they are all registered."""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    __all__.append(module_name)
    importlib.import_module(f"{__name__}.{module_name}")
