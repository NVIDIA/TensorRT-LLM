# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA recurrent-state declaration: the state a prefix checkpoint restores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionLayer,
    RegionLifeCycle,
)

RECURRENT_STATE_REGION = "recurrent_state"


class KDARecurrentRegion:
    """Declare the recurrent matrix and convolution history of the local KDA layers as one region.

    The region is snapshotted at the request's commit boundaries and restored at
    a matched prefix. Commits stop at the prompt, where the two planes alone
    continue the sequence. With replay caches, a checkpoint taken during
    generation would also need the pending drafts replayed into ``SSM_STATE``
    and the replay convolution windows folded into ``CONV_STATE``.
    Recurrent-state slots are read through the manager's ``ssm_layer_group_id``,
    which resolves to this region's group.

    Args:
        mamba_pp_layers: The rank's KDA layers, as global model layer ids.
        layer_offsets: Global model layer to local manager layer mapping.
        ssm_bytes: Bytes of one layer's ``SSM_STATE`` in one slot.
        conv_bytes: Bytes of one layer's ``CONV_STATE`` in one slot.
    """

    def __init__(
        self,
        mamba_pp_layers: Sequence[int],
        layer_offsets: Mapping[int, int],
        ssm_bytes: int,
        conv_bytes: int,
    ) -> None:
        self.region = Region(
            name=RECURRENT_STATE_REGION,
            life_cycle=RegionLifeCycle(AllocationUnit.REQUEST, HandOff.DECLARED_POINTS),
            layers=tuple(
                RegionLayer(
                    model_layer=layer_offsets[layer],
                    planes=(
                        Plane(MambaRole.SSM_STATE, ssm_bytes),
                        Plane(MambaRole.CONV_STATE, conv_bytes),
                    ),
                )
                for layer in mamba_pp_layers
            ),
        )

    def declare_regions(self) -> Sequence[Region]:
        return (self.region,)


__all__ = ["KDARecurrentRegion", "RECURRENT_STATE_REGION"]
