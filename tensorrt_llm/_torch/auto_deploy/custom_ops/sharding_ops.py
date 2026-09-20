# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sharding-aware custom ops for the new hint-driven sharding architecture.

These ops encode sharding intent as metadata kwargs. At graph level they behave
identically to their torch.ops.aten counterparts. The ``apply_sharding_hints``
transform reads the hint kwargs together with a runtime ``Mapping`` to apply
deterministic, node-local sharding transformations.
"""

from typing import List

import torch


@torch.library.custom_op("auto_deploy::view", mutates_args=())
def view(
    x: torch.Tensor,
    shape: List[int],
    tp_scaled_dim: int = -1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Sharding-aware view/reshape.

    At runtime this behaves like ``x.reshape(shape).clone()``. When tensor-parallel
    sharding is enabled, ``apply_sharding_hints`` may scale one dimension of
    ``shape`` by ``1 / tp_size`` so the reshaped tensor matches per-rank shapes.

    Args:
        x: Input tensor to reshape.
        shape: Target shape, same semantics as :meth:`torch.Tensor.reshape`.
        tp_scaled_dim: Index into ``shape`` for TP scaling. When non-negative
            (``0``, ``1``, ``2``, ...), ``apply_sharding_hints`` divides
            ``shape[tp_scaled_dim]`` by ``tp_size``. ``-1`` means this dimension is
            not scaled for TP.
        layer_type: Layer classification for selective sharding via ``shard_layers``
            config. Values: ``"mha"``, ``"mla"``, ``"mlp"``, ``"moe"``, ``"ssm"``,
            ``"delta"``, ``"unknown"``.

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``tp_scaled_dim`` and ``layer_type`` are hints only; they do not change the
        unsharded reshape result.

    Returns:
        Reshaped tensor, same values as ``x.reshape(shape)`` up to clone semantics.
    """
    return x.reshape(shape).clone()


@view.register_fake
def _view_fake(
    x: torch.Tensor,
    shape: List[int],
    tp_scaled_dim: int = -1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    return x.reshape(shape).clone()


@torch.library.custom_op("auto_deploy::split_with_sizes", mutates_args=())
def split_with_sizes(
    x: torch.Tensor,
    split_sizes: List[int],
    dim: int = -1,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
) -> List[torch.Tensor]:
    """Sharding-aware :func:`torch.split` with explicit chunk sizes.

    At runtime this behaves like ``torch.split(x, split_sizes, dim=dim)``, with each
    chunk cloned. When ``enable_sharding`` is ``True`` and TP sharding is applied,
    ``apply_sharding_hints`` scales ``split_sizes`` so each rank splits its local
    activation width consistently.

    Args:
        x: Tensor to split along ``dim``.
        split_sizes: Size of each chunk along ``dim`` (same as PyTorch
            ``split_with_sizes``).
        dim: Dimension along which to split. May be negative (same semantics as
            PyTorch).
        enable_sharding: When ``True``, ``apply_sharding_hints`` divides every element of
            ``split_sizes`` by ``tp_size`` so splits match per-rank tensor shapes.
        layer_type: Layer classification for selective sharding via ``shard_layers``
            config. Values: ``"mha"``, ``"mla"``, ``"mlp"``, ``"moe"``, ``"ssm"``,
            ``"delta"``, ``"unknown"``.

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``enable_sharding``: When ``True``, ``apply_sharding_hints`` scales ``split_sizes``
        for TP (each chunk size is divided by ``tp_size`` when applicable).
        ``layer_type``: Selects whether this node is rewritten for a given
        ``shard_layers`` configuration.

    Returns:
        List of tensors, one per chunk, same as :func:`torch.split`.
    """

    return [t.clone() for t in torch.split(x, split_sizes, dim=dim)]


@split_with_sizes.register_fake
def _split_with_sizes_fake(
    x: torch.Tensor,
    split_sizes: List[int],
    dim: int = -1,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
) -> List[torch.Tensor]:
    return [t.clone() for t in torch.split(x, split_sizes, dim=dim)]


@torch.library.custom_op("auto_deploy::all_reduce", mutates_args=())
def all_reduce(x: torch.Tensor, layer_type: str = "unknown") -> torch.Tensor:
    """Sharding-aware all-reduce placeholder.

    At runtime this returns ``x.clone()``. After ``apply_sharding_hints``, the node
    may become a real ``dist.all_reduce`` when ``tp_size > 1`` and attention
    data-parallel replication is disabled; otherwise it remains an identity on the
    local tensor.

    Args:
        x: Activation tensor to combine across TP ranks when an all-reduce is
            inserted (e.g., partial attention outputs that must be summed).
        layer_type: Layer classification for selective sharding via ``shard_layers``
            config. Values: ``"mha"``, ``"mla"``, ``"mlp"``, ``"moe"``, ``"ssm"``,
            ``"delta"``, ``"unknown"``.

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``layer_type`` gates whether this placeholder is eligible for replacement by
        a collective in a given configuration.

    Returns:
        Tensor with the same shape and dtype as ``x`` (clone of input when unsharded).
    """
    return x.clone()


@all_reduce.register_fake
def _all_reduce_fake(x: torch.Tensor, layer_type: str = "unknown") -> torch.Tensor:
    return x.clone()
