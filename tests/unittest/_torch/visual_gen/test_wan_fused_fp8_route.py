# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pipeline_config fused_fp8 parsing; needs no GPU because route() raises or returns first."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.wan import fused_fp8_route
from tensorrt_llm._torch.visual_gen.models.wan.fused_fp8_route import FusedFp8Route
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import ParallelConfig, QuantAttentionConfig

pytestmark = pytest.mark.cpu_only


def _route_config(fused_fp8, fp8_attention=True, fp8_checkpoint=True, async_ulysses=False):
    quant_attn = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8") if fp8_attention else None
    return DiffusionModelConfig(
        attention=AttentionConfig(backend="CUDNN", quant_attention_config=quant_attn),
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8 if fp8_checkpoint else None),
        parallel=ParallelConfig(ulysses_size=2, async_ulysses=True)
        if async_ulysses
        else ParallelConfig(),
        visual_gen_mapping=SimpleNamespace(ulysses_size=2) if async_ulysses else None,
        extra_attrs={} if fused_fp8 is None else {"fused_fp8": fused_fp8},
    )


@pytest.mark.parametrize("fused_fp8", [None, "off"], ids=["unset", "off"])
def test_fused_fp8_route_off(fused_fp8):
    assert fused_fp8_route.route(_route_config(fused_fp8), tp_size=1) == FusedFp8Route()


def test_fused_fp8_route_require_raises_with_reason():
    with pytest.raises(ValueError, match="attention is not cuDNN"):
        fused_fp8_route.route(_route_config("require", fp8_attention=False), tp_size=1)
    with pytest.raises(ValueError, match="parallel.async_ulysses is enabled"):
        fused_fp8_route.route(_route_config("require", async_ulysses=True), tp_size=1)
    with pytest.raises(ValueError, match="expected one of"):
        fused_fp8_route.route(_route_config("on"), tp_size=1)
