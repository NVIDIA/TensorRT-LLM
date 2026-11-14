# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for layer parity checks of a Kimi K3 module against the built-in module, on one GPU.

A check builds both modules at one rank's shapes with a single-rank ``Mapping()`` (for example 6 attention heads,
TP16's share of 96), gives them the same weights, runs both on the same inputs and compares the outputs: bit for bit
where both run the same kernels in the same order, else within a stated bound.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn


def copy_weights(
    source: nn.Module,
    target: nn.Module,
    rename: Optional[Callable[[str], Optional[str]]] = None,
    skip: Iterable[str] = (),
) -> Dict[str, str]:
    """Copy every parameter and buffer of ``source`` into ``target`` under ``rename(name)`` (the same name by
    default; None leaves it out). Every target tensor must be filled exactly once, so a renamed or reshaped weight
    cannot be missed silently. Returns the map from source to target names."""
    skip = set(skip)
    target_tensors = dict(target.named_parameters())
    target_tensors.update(dict(target.named_buffers()))
    filled: Dict[str, str] = {}
    for name, tensor in list(source.named_parameters()) + list(source.named_buffers()):
        dest = rename(name) if rename is not None else name
        if dest is None or dest in skip:
            continue
        assert dest in target_tensors, f"{name} -> {dest}: no such tensor in the new module"
        assert dest not in filled.values(), f"{dest} filled twice"
        out = target_tensors[dest]
        assert out.shape == tensor.shape and out.dtype == tensor.dtype, (
            f"{name} -> {dest}: {tuple(tensor.shape)} {tensor.dtype} into {tuple(out.shape)} {out.dtype}"
        )
        with torch.no_grad():
            out.copy_(tensor)
        filled[name] = dest
    missing = sorted(set(target_tensors) - set(filled.values()) - skip)
    assert not missing, f"tensors of the new module no source fills: {missing}"
    return filled


def assert_bitwise(new: torch.Tensor, old: torch.Tensor, what: str = "output") -> None:
    """The two tensors are the same bits."""
    assert new.shape == old.shape and new.dtype == old.dtype, (
        f"{what}: {tuple(new.shape)} {new.dtype} vs {tuple(old.shape)} {old.dtype}"
    )
    if not torch.equal(new, old):
        diff = (new.float() - old.float()).abs()
        raise AssertionError(
            f"{what}: {int((new != old).sum())} of {new.numel()} elements differ, max |diff| {float(diff.max())}"
        )


def assert_within(new: torch.Tensor, old: torch.Tensor, rel: float, what: str = "output") -> None:
    """``max |new - old| <= rel * max |old|``, for paths that do not run the same kernels in the same order."""
    assert new.shape == old.shape, f"{what}: {tuple(new.shape)} vs {tuple(old.shape)}"
    bound = rel * float(old.float().abs().max())
    worst = float((new.float() - old.float()).abs().max())
    assert worst <= bound, f"{what}: max |diff| {worst} above {rel} x max |old| = {bound}"
