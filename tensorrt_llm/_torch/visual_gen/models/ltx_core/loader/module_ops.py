"""Named transformations applied to a model shell before binding weights."""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import torch


class ModuleOps(NamedTuple):
    """A named structural transform applied while constructing a model shell.
    ``matcher`` selects a root module for ``mutator``, which may change or
    replace that module. ``name`` identifies the transform in shell cache keys.
    ``required_device_type`` optionally constrains where builders may bind the
    model's weights; it does not change meta-shell construction.
    """

    name: str
    matcher: Callable[[torch.nn.Module], bool]
    mutator: Callable[[torch.nn.Module], torch.nn.Module]
    required_device_type: str | None = None


def apply_module_ops(model: torch.nn.Module, operations: Sequence[ModuleOps]) -> torch.nn.Module:
    """Apply matching operations in order, including a replacement root module."""
    for operation in operations:
        if operation.matcher(model):
            model = operation.mutator(model)
    return model
