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

    ``apply_sharding_hints`` divides ``shape[tp_scaled_dim]`` by ``tp_size``.

    Note: requires explicit non-negative scaling dimension (0, 1, 2, ...).
    ``-1`` means no sharding (the dimension is not scaled).
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
    shardable: bool = False,
    layer_type: str = "unknown",
) -> List[torch.Tensor]:
    """Sharding-aware split_with_sizes.

    When ``shardable`` is ``True``, ``apply_sharding_hints`` divides every
    element of ``split_sizes`` by ``tp_size``.
    """

    return [t.clone() for t in torch.split(x, split_sizes, dim=dim)]


@split_with_sizes.register_fake
def _split_with_sizes_fake(
    x: torch.Tensor,
    split_sizes: List[int],
    dim: int = -1,
    shardable: bool = False,
    layer_type: str = "unknown",
) -> List[torch.Tensor]:
    return [t.clone() for t in torch.split(x, split_sizes, dim=dim)]


@torch.library.custom_op("auto_deploy::all_reduce", mutates_args=())
def all_reduce(x: torch.Tensor, layer_type: str = "unknown") -> torch.Tensor:
    """Sharding-aware all-reduce placeholder.

    Identity when running unsharded.  ``apply_sharding_hints`` replaces this
    node with a real ``dist.all_reduce`` when ``tp_size > 1`` and attention DP
    is disabled, or leaves it as identity otherwise.
    """
    return x.clone()


@all_reduce.register_fake
def _all_reduce_fake(x: torch.Tensor, layer_type: str = "unknown") -> torch.Tensor:
    return torch.empty_like(x)
