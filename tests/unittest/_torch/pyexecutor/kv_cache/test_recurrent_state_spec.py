# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The recurrent state a model declares, against the geometry the cache managers use."""

import pytest
import torch

from tensorrt_llm._torch.modules.qwen4_exp.cache_manager import PLE_NGRAM_CONTEXT
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaRole
from tensorrt_llm._torch.recurrent_state_spec import (
    LayerKind,
    LayerStateSpec,
    PlaneShard,
    StatePlaneSpec,
)


def test_plane_splits_only_its_sharded_axis():
    conv = StatePlaneSpec(
        role=MambaRole.CONV_STATE,
        shape=(48, 3),
        dtype=torch.bfloat16,
        shard=PlaneShard(axis=0, sections=(32, 8, 8)),
    )
    assert conv.local_shape(4) == (12, 3)
    assert conv.local_sections(4) == (8, 2, 2)
    with pytest.raises(ValueError, match="cannot split evenly"):
        conv.local_shape(5)

    # Each section is split on its own, so sections that are not multiples of the rank count
    # leave part of the local axis outside every section.
    uneven = StatePlaneSpec(
        role=MambaRole.CONV_STATE,
        shape=(16, 3),
        dtype=torch.bfloat16,
        shard=PlaneShard(axis=0, sections=(5, 5, 6)),
    )
    assert uneven.local_shape(2) == (8, 3)
    assert uneven.local_sections(2) == (2, 2, 3)

    whole = StatePlaneSpec(role=PLE_NGRAM_CONTEXT, shape=(2,), dtype=torch.int64)
    assert whole.local_shape(4) == (2,)
    assert whole.local_sections(4) == ()


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(axis=0, sections=(32, 8))
            ),
            "do not add up",
        ),
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(sections=(48,))
            ),
            "no sharded axis",
        ),
        (
            lambda: StatePlaneSpec(
                MambaRole.CONV_STATE, (48, 3), torch.bfloat16, PlaneShard(axis=2)
            ),
            "outside its shape",
        ),
        (
            lambda: LayerStateSpec(
                LayerKind.ATTENTION, (StatePlaneSpec(MambaRole.SSM_STATE, (4,), torch.bfloat16),)
            ),
            "cannot carry recurrent state",
        ),
        (
            lambda: LayerStateSpec(
                LayerKind.RECURRENT,
                (StatePlaneSpec(MambaRole.SSM_STATE, (4,), torch.bfloat16),) * 2,
            ),
            "more than once",
        ),
    ],
)
def test_malformed_declaration_is_rejected(build, message):
    with pytest.raises(ValueError, match=message):
        build()
