# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve prepared-weight metadata inside opaque ops, including after offload moves."""

from weakref import WeakValueDictionary

import torch
from torch import nn

_OWNERS: WeakValueDictionary[int, nn.Module] = WeakValueDictionary()
_BY_ADDRESS: WeakValueDictionary[tuple, nn.Module] = WeakValueDictionary()


def _address(weight: torch.Tensor) -> tuple:
    return weight.device, weight.data_ptr(), tuple(weight.shape), tuple(weight.stride())


def register_nvfp4_owner(owner: nn.Module) -> None:
    _OWNERS[id(owner)] = owner
    _BY_ADDRESS[_address(owner.weight)] = owner


def find_nvfp4_owner(weight: torch.Tensor) -> nn.Module | None:
    """Validate cached addresses; offload may replace storage or the buffer itself."""
    key = _address(weight)
    owner = _BY_ADDRESS.get(key)
    if owner is not None and _address(owner.weight) == key:
        return owner
    _BY_ADDRESS.clear()
    for candidate in list(_OWNERS.values()):
        _BY_ADDRESS[_address(candidate.weight)] = candidate
    return _BY_ADDRESS.get(key)


def nvfp4_owner(weight: torch.Tensor) -> nn.Module:
    owner = find_nvfp4_owner(weight)
    if owner is None:
        raise RuntimeError("FLUX.3 NVFP4 prepared weight has no registered owner")
    return owner
