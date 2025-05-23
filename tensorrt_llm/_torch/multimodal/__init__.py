from .mm_utils import _SharedTensorRebuildMethodRegistry, SharedTensorContainer

# Initialize the registry when the package is imported
_SharedTensorRebuildMethodRegistry.initialize()

# Export the classes for easy access
__all__ = [
    'SharedTensorContainer',
]
