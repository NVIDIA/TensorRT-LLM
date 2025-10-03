"""Common utils for nn.Module."""

import itertools
from typing import Tuple

import torch.nn as nn


def get_submodule_of_param(gm: nn.Module, param_name: str) -> Tuple[nn.Module, str, str]:
    # Returns (module, module_path, attr_name)
    if "." not in param_name:
        # param on the root
        return gm, "", param_name
    mod_path, _, attr = param_name.rpartition(".")
    return gm.get_submodule(mod_path), mod_path, attr


def has_any_meta_tensor(m: nn.Module) -> bool:
    """Return True if any parameter or buffer is on the meta device."""
    for t in itertools.chain(m.parameters(recurse=True), m.buffers(recurse=True)):
        if t is None or t.numel() == 0:
            continue
        # t.is_meta as first check; device.type == 'meta' as fallback.
        if (
            getattr(t, "is_meta", False)
            or getattr(getattr(t, "device", None), "type", None) == "meta"
        ):
            return True
    return False
