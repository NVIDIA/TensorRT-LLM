# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Recurrent state a model declares from its configuration.

A model class answers ``get_recurrent_state_spec()`` with one entry per target-model layer: the
layer's kind and, for a recurrent layer, the planes of base state it carries from one step to the
next. The declaration is derived from the configuration alone, so it is available before a cache
manager exists, and it describes the model rather than one rank: shapes are whole, and each plane
says how the ranks that share a request split it.

Speculative buffers such as replay history are not base state; they are declared as regions (see
``pyexecutor/kv_cache/REGIONS_DEVELOPER_GUIDE.md``).

``pyexecutor/kv_cache/mamba_cache_manager/RECURRENT_STATE_DEVELOPER_GUIDE.md`` describes how the
declaration is found and what else a recurrent family provides.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.config_utils import MambaKVCacheParams
    from tensorrt_llm.runtime.kv_cache_manager_v2 import DataRole

__all__ = [
    "LayerKind",
    "LayerStateSpec",
    "PlaneShard",
    "RecurrentStateSpec",
    "StatePlaneSpec",
    "mamba_params_from_recurrent_state_spec",
    "recurrent_state_spec_from_mamba_params",
]


class LayerKind(Enum):
    """What one model layer keeps between steps."""

    ATTENTION = "attention"
    RECURRENT = "recurrent"
    NONE = "none"


@dataclass(frozen=True)
class PlaneShard:
    """How the ranks that share a request split one state plane.

    Attributes:
        axis: Axis of the per-slot shape split evenly across those ranks; ``None`` when every
            rank holds the whole plane.
        sections: Sizes of the consecutive sections along ``axis``. A rank holds each section's
            size divided by the number of ranks, rounded down. Empty when the axis is one
            section.
    """

    axis: int | None = None
    sections: tuple[int, ...] = ()


@dataclass(frozen=True)
class StatePlaneSpec:
    """One plane of base recurrent state of a layer.

    Attributes:
        role: Buffer role of the plane.
        shape: Shape of one slot, before any split across ranks.
        dtype: Element type.
        shard: How the ranks that share a request split the plane.
    """

    role: DataRole
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard: PlaneShard = PlaneShard()

    def __post_init__(self) -> None:
        axis, sections = self.shard.axis, self.shard.sections
        if axis is None:
            if sections:
                raise ValueError(f"Plane {self.role!r} has sections but no sharded axis")
            return
        if not 0 <= axis < len(self.shape):
            raise ValueError(
                f"Plane {self.role!r} is sharded on axis {axis}, outside its shape {self.shape}"
            )
        if sections and sum(sections) != self.shape[axis]:
            raise ValueError(
                f"Sections {sections} of plane {self.role!r} do not add up to its sharded "
                f"axis of size {self.shape[axis]}"
            )

    def local_shape(self, num_shards: int) -> tuple[int, ...]:
        """Return the shape of one slot on one of ``num_shards`` ranks sharing a request."""
        axis = self.shard.axis
        if axis is None:
            return self.shape
        size = self.shape[axis]
        if size % num_shards != 0:
            raise ValueError(
                f"Axis {axis} of plane {self.role!r} has size {size}, which {num_shards} ranks "
                "cannot split evenly"
            )
        return self.shape[:axis] + (size // num_shards,) + self.shape[axis + 1 :]

    def local_sections(self, num_shards: int) -> tuple[int, ...]:
        """Return the section sizes on one of ``num_shards`` ranks sharing a request.

        Each section is divided on its own and rounded down, so the sizes add up to the local
        axis only when every section is a multiple of ``num_shards``.
        """
        return tuple(section // num_shards for section in self.shard.sections)


@dataclass(frozen=True)
class LayerStateSpec:
    """Kind of one model layer and, for a recurrent layer, its base-state planes."""

    kind: LayerKind
    planes: tuple[StatePlaneSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.planes and self.kind is not LayerKind.RECURRENT:
            raise ValueError(f"A {self.kind.value} layer cannot carry recurrent state planes")
        roles = [plane.role for plane in self.planes]
        if len(set(roles)) != len(roles):
            raise ValueError(f"A layer declares a state plane role more than once: {roles}")


@dataclass(frozen=True)
class RecurrentStateSpec:
    """Layer kinds and base recurrent state of a target model.

    Attributes:
        layers: One entry per target-model layer; the position is the layer index. Draft layers
            appended for speculative decoding are not included.
    """

    layers: tuple[LayerStateSpec, ...]

    @property
    def recurrent_layer_mask(self) -> tuple[bool, ...]:
        """Whether each target-model layer carries recurrent state."""
        return tuple(layer.kind is LayerKind.RECURRENT for layer in self.layers)

    @property
    def attention_layer_mask(self) -> tuple[bool, ...]:
        """Whether each target-model layer keeps a KV cache."""
        return tuple(layer.kind is LayerKind.ATTENTION for layer in self.layers)


def _mamba_base_planes(
    params: MambaKVCacheParams, conv_sections: Literal["x_b_c", "q_k_v"]
) -> tuple[StatePlaneSpec, StatePlaneSpec]:
    """Return the convolution and SSM planes of a recurrent layer with the geometry of ``params``.

    Raises:
        ValueError: If ``conv_sections`` is neither order.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole

    inner_dim = params.head_dim * params.num_heads
    grouped_dim = params.n_groups * params.state_size
    if conv_sections == "x_b_c":
        sections = (inner_dim, grouped_dim, grouped_dim)
    elif conv_sections == "q_k_v":
        sections = (grouped_dim, grouped_dim, inner_dim)
    else:
        raise ValueError(f"Unsupported convolution section order: {conv_sections!r}")
    conv_plane = StatePlaneSpec(
        role=MambaRole.CONV_STATE,
        shape=(inner_dim + 2 * grouped_dim, params.conv_kernel - 1),
        dtype=params.dtype,
        shard=PlaneShard(axis=0, sections=sections),
    )
    ssm_plane = StatePlaneSpec(
        role=MambaRole.SSM_STATE,
        shape=(params.num_heads, params.head_dim, params.state_size),
        dtype=(
            params.mamba_ssm_cache_dtype
            if params.mamba_ssm_cache_dtype is not None
            else params.dtype
        ),
        shard=PlaneShard(axis=0),
    )
    return conv_plane, ssm_plane


def recurrent_state_spec_from_mamba_params(
    params: MambaKVCacheParams,
    *,
    conv_sections: Literal["x_b_c", "q_k_v"],
    extra_planes: Mapping[int, tuple[StatePlaneSpec, ...]] | None = None,
) -> RecurrentStateSpec:
    """Declare the state of a model whose recurrent layers share one Mamba-style geometry.

    Each recurrent layer gets a convolution plane ``[inner + 2 * grouped, conv_kernel - 1]`` and
    an SSM plane ``[num_heads, head_dim, state_size]``, where ``inner = head_dim * num_heads`` and
    ``grouped = n_groups * state_size``. Both are split across ranks along their first axis.

    Args:
        params: Geometry, target-layer masks and dtypes of the model.
        conv_sections: Order of the channel sections of the convolution plane: ``"x_b_c"`` is
            ``(inner, grouped, grouped)`` and ``"q_k_v"`` is ``(grouped, grouped, inner)``.
        extra_planes: Further planes of individual layers, keyed by layer index.

    Returns:
        The declaration of the target-model layers.

    Raises:
        ValueError: If ``conv_sections`` is neither order.
    """
    conv_plane, ssm_plane = _mamba_base_planes(params, conv_sections)

    extra_planes = extra_planes or {}
    layers = []
    for layer_idx, (is_recurrent, is_attention) in enumerate(
        zip(params.mamba_layer_mask, params.target_full_attention_layer_mask, strict=True)
    ):
        if is_recurrent:
            kind, planes = LayerKind.RECURRENT, (conv_plane, ssm_plane)
        elif is_attention:
            kind, planes = LayerKind.ATTENTION, ()
        else:
            kind, planes = LayerKind.NONE, ()
        layers.append(LayerStateSpec(kind, planes + tuple(extra_planes.get(layer_idx, ()))))
    return RecurrentStateSpec(layers=tuple(layers))


def mamba_params_from_recurrent_state_spec(
    spec: RecurrentStateSpec, *, num_draft_layers: int = 0
) -> MambaKVCacheParams:
    """Return the Mamba-style cache parameters of a declaration.

    The inverse of :func:`recurrent_state_spec_from_mamba_params`: it accepts the convolution and
    SSM planes that function declares, in either section order, and no others. The cache managers
    take one geometry for all recurrent layers, so these layers must declare the same two planes.

    Args:
        spec: Declaration of the target-model layers.
        num_draft_layers: Number of layers appended for one-model speculative decoding.

    Returns:
        The geometry, target-layer masks and dtypes the cache managers are built from.

    Raises:
        ValueError: If no layer is recurrent, the recurrent layers differ in their convolution or
            SSM plane, or these planes are not the ones
            :func:`recurrent_state_spec_from_mamba_params` declares for some parameters.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import MambaKVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole

    def base_planes(layer: LayerStateSpec) -> tuple[StatePlaneSpec | None, StatePlaneSpec | None]:
        planes = {plane.role: plane for plane in layer.planes}
        return planes.get(MambaRole.CONV_STATE), planes.get(MambaRole.SSM_STATE)

    recurrent_layers = [layer for layer in spec.layers if layer.kind is LayerKind.RECURRENT]
    if not recurrent_layers:
        raise ValueError("A declaration without a recurrent layer carries no state geometry")
    conv, ssm = base_planes(recurrent_layers[0])
    if conv is None or ssm is None:
        raise ValueError("A recurrent layer declares no convolution or no SSM plane")
    if any(base_planes(layer) != (conv, ssm) for layer in recurrent_layers[1:]):
        raise ValueError("Recurrent layers declare different convolution or SSM planes")

    inexpressible = ValueError(
        f"The cache parameters cannot express the base state {conv} and {ssm}: these are not the "
        "planes that recurrent_state_spec_from_mamba_params declares"
    )
    sections = conv.shard.sections
    if len(conv.shape) != 2 or len(ssm.shape) != 3 or len(sections) != 3 or ssm.shape[2] < 1:
        raise inexpressible
    num_heads, head_dim, state_size = ssm.shape
    params = MambaKVCacheParams(
        state_size=state_size,
        conv_kernel=conv.shape[1] + 1,
        num_heads=num_heads,
        n_groups=sections[1] // state_size,
        head_dim=head_dim,
        mamba_layer_mask=list(spec.recurrent_layer_mask),
        target_full_attention_layer_mask=list(spec.attention_layer_mask),
        num_mamba_layers=sum(spec.recurrent_layer_mask),
        num_draft_layers=num_draft_layers,
        dtype=conv.dtype,
        mamba_ssm_cache_dtype=ssm.dtype,
    )
    if (conv, ssm) not in (
        _mamba_base_planes(params, "x_b_c"),
        _mamba_base_planes(params, "q_k_v"),
    ):
        raise inexpressible
    return params
