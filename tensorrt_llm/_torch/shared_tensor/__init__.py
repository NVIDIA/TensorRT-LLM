from .shared_tensor import (SharedTensorContainer,
                            _SharedTensorRebuildMethodRegistry)

# Initialize the registry when the package is imported
_SharedTensorRebuildMethodRegistry.initialize()

__all__ = [
    'SharedTensorContainer',
]
