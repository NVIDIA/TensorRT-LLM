"""Custom ops and make sure they are all registered."""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    __all__.append(module_name)
    importlib.import_module(f"{__name__}.{module_name}")

# Recursively import subpackages and modules so their side-effects (e.g.,
# op registrations) are applied even when nested in subdirectories.
for _, full_name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    __all__.append(full_name)
    importlib.import_module(full_name)
