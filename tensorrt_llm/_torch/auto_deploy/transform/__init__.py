"""AutoDeploy's modular graph transform + inference optimizer pipeline."""

from . import (
    library,  # ensure all transforms are registered
    semantic_mask_registry,  # noqa: F401
)
from .interface import *
