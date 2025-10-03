"""Common utils for nn.Module."""

from typing import Tuple

import torch.nn as nn


def get_submodule_of_param(gm: nn.Module, param_name: str) -> Tuple[nn.Module, str, str]:
    # Returns (module, module_path, attr_name)
    if "." not in param_name:
        # param on the root
        return gm, "", param_name
    mod_path, _, attr = param_name.rpartition(".")
    return gm.get_submodule(mod_path), mod_path, attr
