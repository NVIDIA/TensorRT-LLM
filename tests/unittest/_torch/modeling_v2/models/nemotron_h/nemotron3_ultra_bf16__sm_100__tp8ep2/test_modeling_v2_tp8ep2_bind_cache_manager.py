# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 tp8ep2 target binds the cache pool dtypes it was gated on and refuses the rest.

The KV pool may be bf16 (`quant_mode` 0) or fp8-e4m3 at unit scales (`QM_FP8_KV_CACHE`, no scale
tensors), and the attention calls follow the bound pool; the SSM kernels read an fp32 state. A
runtime configured otherwise (`kv_cache_config.dtype`, `mamba_ssm_cache_dtype`) would be read and
written in the wrong width without a shape error anywhere. The pools are the runtime's to size; the
target's job is to notice the mismatch when it takes them, which is what `_bind_cache_manager` does.

No GPU: the cache manager here is a stand-in carrying the attributes the method reads.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.nemotron_h.nemotron3_ultra_bf16__sm_100__tp8ep2 import (  # noqa: E501
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
    # Rank 1: the method logs the pools' depth on rank 0 only.
    mapping = SimpleNamespace(rank=1)
    core = SimpleNamespace(
        _binds=0, _bound=None, dims=_DIMS, model_config=SimpleNamespace(mapping=mapping)
    )
    M.NemotronUltraBf16Tp8ep2ModelingV2._bind_cache_manager(core, md)
    return core, cm


@pytest.mark.parametrize("kv, quant_mode", [(DataType.BF16, 0), (DataType.FP8, QM_FP8_KV_CACHE)])
def test_the_certified_dtypes_bind(kv, quant_mode):
    core, cm = _bind(kv=kv)
    assert core._bound.cm is cm and core._bound.pool_row == {2: 0}
    assert core._bound.kv_quant_mode == quant_mode


@pytest.mark.parametrize("kv", [DataType.NVFP4, DataType.HALF])
def test_a_kv_pool_that_is_neither_bf16_nor_fp8_is_refused(kv):
    with pytest.raises(AssertionError, match="bf16 or fp8 KV cache"):
        _bind(kv=kv)


@pytest.mark.parametrize(
    "conv, ssm", [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16)]
)
def test_a_state_pool_that_is_not_bf16_and_fp32_is_refused(conv, ssm):
    with pytest.raises(AssertionError, match="needs bf16 / fp32"):
        _bind(conv=conv, ssm=ssm)
