# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 tep8 target refuses a cache pool dtype it was not gated on.

The attention call passes `quant_mode` 0 and no scales, and the SSM kernels read an fp32 state, so a
runtime configured otherwise (`kv_cache_config.dtype`, `mamba_ssm_cache_dtype`) would be read and
written in the wrong width without a shape error anywhere. The pools are the runtime's to size; the
target's job is to notice the mismatch when it takes them, which is what `_bind_cache_manager` does.

No GPU: the cache manager here is a stand-in carrying the attributes the method reads.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__tep8 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm.bindings import DataType

# A tiny geometry: the method compares pool shapes with the dimension bundle it was given.
_DIMS = SimpleNamespace(
    attn_layers=[2],
    mamba_layers=[0, 3],
    conv_r=6,
    conv_kernel=4,
    heads_r=2,
    head_dim_ssm=4,
    ssm_state=8,
)


def _bind(kv=DataType.BF16, conv=torch.bfloat16, ssm=torch.float32):
    cm = SimpleNamespace(
        dtype=kv,
        layer_offsets={2: 0},
        get_conv_states=lambda layer: torch.zeros(3, 6, 3, dtype=conv),
        get_ssm_states=lambda layer: torch.zeros(3, 2, 4, 8, dtype=ssm),
    )
    md = SimpleNamespace(
        kv_cache_manager=cm, effective_beam_width=1, cache_indirection=None, is_cross=False
    )
    core = SimpleNamespace(_binds=0, _bound=None, dims=_DIMS)
    M.NemotronUltraBf16Tep8ModelingV2._bind_cache_manager(core, md)
    return core, cm


def test_the_certified_dtypes_bind():
    core, cm = _bind()
    assert core._bound.cm is cm and core._bound.pool_row == {2: 0}


@pytest.mark.parametrize("kv", [DataType.FP8, DataType.NVFP4, DataType.HALF])
def test_a_kv_pool_that_is_not_bf16_is_refused(kv):
    with pytest.raises(AssertionError, match="bf16 KV cache"):
        _bind(kv=kv)


@pytest.mark.parametrize(
    "conv, ssm", [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16)]
)
def test_a_state_pool_that_is_not_bf16_and_fp32_is_refused(conv, ssm):
    with pytest.raises(AssertionError, match="needs bf16 / fp32"):
        _bind(conv=conv, ssm=ssm)
