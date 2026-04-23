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
"""Unit tests for ``MegaMoEFusedMoE`` — the DeepGEMM ``fp8_fp4_mega_moe``
MoE backend.

What the tests cover
--------------------
* ``can_implement()`` matrix — accepts W4A8_MXFP4_MXFP8 + bf16 + SM100
  and rejects every other combination with a descriptive reason.
* ``get_moe_cls("MEGAMOE")`` dispatch — returns ``MegaMoEFusedMoE`` for
  W4A8_MXFP4_MXFP8, falls back to ``CutlassFusedMoE`` for anything
  else (mirrors the TRTLLM / CUTEDSL fallback).
* ``apply_router_weight_on_input`` rejection at construction time (the
  DG kernel applies routing on the MoE output, not pre-scaling the
  input — the two paths aren't equivalent under SwiGLU).
* ADP topology guard — ``use_dp and parallel_size > 1`` requires
  ``ep_size == parallel_size`` so DG's fused kernel subsumes the
  cross-ADP-rank exchange.

These tests intentionally do not exercise the hot path — end-to-end
forward validation is covered by the multi-GPU harness in
``tmp_test_scripts/parity_vs_torch_ref.py`` which requires 4+ GPUs,
SM100 and a bundled DeepGEMM with ``fp8_fp4_mega_moe``. The tests here
skip cleanly when those prerequisites are missing.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.create_moe import get_moe_cls
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoEFusedMoE
from tensorrt_llm._torch.modules.fused_moe.mega_moe.backend import (
    _import_deep_gemm,
    _MegaMoEUnavailable,
    _ue8m0_uint8_to_fp32,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


# ============================================================================
# Single-process torch.distributed bootstrap for dist-requiring tests
# ============================================================================
@pytest.fixture(scope="module")
def _single_proc_dist():
    """Initialise a single-rank torch.distributed world for tests that need
    ``dist.is_initialized() == True`` (e.g. ``MegaMoEFusedMoE.__init__``'s
    EP PG resolution). If dist is already set up (mpirun / torchrun) we
    reuse it; otherwise bring up a 1-rank NCCL group on ``cuda:0``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MegaMoE tests")
    already = dist.is_initialized()
    if not already:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29560")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        yield
    finally:
        if not already and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================================
# Helpers
# ============================================================================
def _dg_available() -> bool:
    """The full MegaMoE hot path needs bundled DG with mega_moe symbols.

    ``can_implement`` does its own check; helpers here use the same gate.
    """
    try:
        _import_deep_gemm()
    except _MegaMoEUnavailable:
        return False
    return True


def _is_sm100() -> bool:
    try:
        return get_sm_version() == 100
    except Exception:
        return False


def _tiny_model_config(
    num_experts: int = 8,
    hidden_size: int = 512,
    intermediate_size: int = 512,
    quant_algo: Optional[QuantAlgo] = QuantAlgo.W4A8_MXFP4_MXFP8,
    moe_backend: str = "MEGAMOE",
    world_size: int = 1,
    rank: int = 0,
) -> ModelConfig:
    pc = PretrainedConfig()
    pc.num_experts = num_experts
    pc.hidden_size = hidden_size
    pc.intermediate_size = intermediate_size
    pc.torch_dtype = torch.bfloat16
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=world_size,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=world_size,
    )
    return ModelConfig(
        pretrained_config=pc,
        mapping=mapping,
        moe_backend=moe_backend,
        quant_config=(QuantConfig(quant_algo=quant_algo) if quant_algo is not None else None),
    )


# ============================================================================
# can_implement() — capability matrix
# ============================================================================
@pytest.mark.skipif(
    not _is_sm100(),
    reason="can_implement() short-circuits on non-SM100 with an SM100 reason; "
    "the per-input negative reasons only surface on SM100 runners.",
)
@pytest.mark.parametrize(
    "quant_algo,dtype,swiglu_gptoss,expected_reason_fragment",
    [
        # Accepted: the only supported combination.
        (QuantAlgo.W4A8_MXFP4_MXFP8, torch.bfloat16, False, None),
        # Wrong quant.
        (QuantAlgo.FP8_BLOCK_SCALES, torch.bfloat16, False, "W4A8_MXFP4_MXFP8"),
        (QuantAlgo.NVFP4, torch.bfloat16, False, "W4A8_MXFP4_MXFP8"),
        (None, torch.bfloat16, False, "W4A8_MXFP4_MXFP8"),
        # Wrong activation dtype.
        (QuantAlgo.W4A8_MXFP4_MXFP8, torch.float32, False, "activations"),
        # swiglu_gptoss_style is not supported by the fused kernel.
        (QuantAlgo.W4A8_MXFP4_MXFP8, torch.bfloat16, True, "swiglu_gptoss_style"),
    ],
)
def test_can_implement_matrix(quant_algo, dtype, swiglu_gptoss, expected_reason_fragment):
    ok, reason = MegaMoEFusedMoE.can_implement(
        quant_algo,
        dtype_activation=dtype,
        swiglu_gptoss_style=swiglu_gptoss,
    )
    if expected_reason_fragment is None:
        # Accepted case still needs bundled DG with mega_moe. On SM100
        # runners without that DG, can_implement returns a DG-missing
        # reason — that's correct behavior.
        if not _dg_available():
            assert ok is False
            assert "mega_moe" in reason or "DeepGEMM" in reason or "per_token_cast_to_fp8" in reason
        else:
            assert ok is True, f"expected True, got reason={reason!r}"
            assert reason is None
    else:
        assert ok is False
        assert expected_reason_fragment in reason, (
            f"reason {reason!r} missing expected fragment {expected_reason_fragment!r}"
        )


# ============================================================================
# get_moe_cls() — quant-gated dispatch
# ============================================================================
@pytest.mark.parametrize(
    "quant_algo,expected_cls_when_available",
    [
        (QuantAlgo.W4A8_MXFP4_MXFP8, MegaMoEFusedMoE),
        # Unsupported quant configs must fall back to CutlassFusedMoE
        # per the TRTLLM / CUTEDSL pattern, so a model with the wrong
        # quant never allocates MXFP4-specific tensors.
        (QuantAlgo.FP8_BLOCK_SCALES, CutlassFusedMoE),
        (QuantAlgo.NVFP4, CutlassFusedMoE),
        (None, CutlassFusedMoE),
    ],
)
def test_get_moe_cls_dispatch(quant_algo, expected_cls_when_available):
    mc = _tiny_model_config(quant_algo=quant_algo)
    # On non-SM100 runners (and SM100 without a DG mega_moe build),
    # ``get_moe_cls`` falls back to CutlassFusedMoE for the accepted
    # quant — that *is* the intended factory behaviour (same pattern as
    # TRTLLM / CUTEDSL), so the test expects Cutlass in that case.
    expected_cls = expected_cls_when_available
    if expected_cls is MegaMoEFusedMoE and (not _is_sm100() or not _dg_available()):
        expected_cls = CutlassFusedMoE
    assert get_moe_cls(mc) is expected_cls


# ============================================================================
# Weight-loader shape contract (shape-level, no kernel)
# ============================================================================
def _make_vanilla_weights(num_experts: int, hidden: int, inter: int):
    """Plausible MXFP4 bytes for each of the 3 matrices per expert."""
    w: dict = {}
    for eid in range(num_experts):
        w[f"{eid}.w1.weight"] = torch.zeros(inter, hidden // 2, dtype=torch.uint8)
        w[f"{eid}.w1.weight_scale"] = torch.zeros(inter, hidden // 32, dtype=torch.uint8)
        w[f"{eid}.w3.weight"] = torch.zeros(inter, hidden // 2, dtype=torch.uint8)
        w[f"{eid}.w3.weight_scale"] = torch.zeros(inter, hidden // 32, dtype=torch.uint8)
        w[f"{eid}.w2.weight"] = torch.zeros(hidden, inter // 2, dtype=torch.uint8)
        w[f"{eid}.w2.weight_scale"] = torch.zeros(hidden, inter // 32, dtype=torch.uint8)
    return w


def _make_fused_gate_up_weights(num_experts: int, hidden: int, inter: int):
    """FUSED_GATE_UP_PROJ schema: stacked tensors with a leading expert
    dim. Shapes match ``MoEWeightLoader.load_expert_weights`` for MXFP4:
    after ``.transpose(0, 1)`` + ``.chunk(2, dim=0)`` we want
    ``w1``/``w3`` each of shape ``[I, H/2]``, so
    ``gate_up_proj[eid]`` must be ``[H/2, 2*I]``. Similarly
    ``down_proj[eid]`` is ``[I/2, H]`` (transpose gives ``[H, I/2]``,
    which is the MXFP4 w2 shape).
    """
    return {
        "gate_up_proj": torch.zeros(num_experts, hidden // 2, 2 * inter, dtype=torch.uint8),
        "gate_up_proj_weight_scale": torch.zeros(
            num_experts, hidden // 32, 2 * inter, dtype=torch.uint8
        ),
        "down_proj": torch.zeros(num_experts, inter // 2, hidden, dtype=torch.uint8),
        "down_proj_weight_scale": torch.zeros(num_experts, inter // 32, hidden, dtype=torch.uint8),
    }


@pytest.mark.skipif(not _is_sm100(), reason="MegaMoE Phase 1 is SM100 only")
@pytest.mark.skipif(not _dg_available(), reason="bundled DeepGEMM lacks mega_moe symbols")
@pytest.mark.parametrize(
    "loading_mode,weight_builder",
    [
        (MoEWeightLoadingMode.VANILLA, _make_vanilla_weights),
        (MoEWeightLoadingMode.FUSED_GATE_UP_PROJ, _make_fused_gate_up_weights),
    ],
)
def test_load_weights_schemas(loading_mode, weight_builder, _single_proc_dist):
    """Verify the loader accepts both key schemas without error and
    produces contiguous MXFP4/UE8M0 tensors of the expected shapes."""
    num_experts, hidden, inter = 8, 512, 512
    mc = _tiny_model_config(num_experts=num_experts, hidden_size=hidden, intermediate_size=inter)

    moe = MegaMoEFusedMoE(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        dtype=torch.bfloat16,
        model_config=mc,
        weight_loading_mode=loading_mode,
    ).cuda()

    weights = weight_builder(num_experts, hidden, inter)
    moe.load_weights([weights])

    assert moe.w3_w1_weight.shape == (num_experts, 2 * inter, hidden // 2)
    assert moe.w3_w1_weight_scale.shape == (num_experts, 2 * inter, hidden // 32)
    assert moe.w2_weight.shape == (num_experts, hidden, inter // 2)
    assert moe.w2_weight_scale.shape == (num_experts, hidden, inter // 32)
    assert moe.w3_w1_weight.dtype == torch.uint8
    assert moe.w3_w1_weight_scale.dtype == torch.uint8


# ============================================================================
# apply_router_weight_on_input — hard reject, not silent ignore
# ============================================================================
@pytest.mark.skipif(not _is_sm100(), reason="MegaMoE Phase 1 is SM100 only")
@pytest.mark.skipif(not _dg_available(), reason="bundled DeepGEMM lacks mega_moe symbols")
def test_reject_apply_router_weight_on_input(_single_proc_dist):
    mc = _tiny_model_config()
    with pytest.raises(AssertionError, match="apply_router_weight_on_input"):
        MegaMoEFusedMoE(
            routing_method=RenormalizeMoeRoutingMethod(top_k=1),
            num_experts=8,
            hidden_size=512,
            intermediate_size=512,
            dtype=torch.bfloat16,
            model_config=mc,
            apply_router_weight_on_input=True,
        )


# ============================================================================
# ADP guard — reject ep_size < parallel_size
# ============================================================================
@pytest.mark.skipif(not _is_sm100(), reason="MegaMoE Phase 1 is SM100 only")
@pytest.mark.skipif(not _dg_available(), reason="bundled DeepGEMM lacks mega_moe symbols")
def test_reject_adp_gt_ep(_single_proc_dist):
    """A Mapping with ``enable_attention_dp=True`` and EP strictly smaller
    than ``parallel_size`` (i.e. attention-DP size > MoE EP size) must be
    refused — DG's fused kernel can only shuttle tokens that originate
    from ranks inside the EP group.

    We build a Mapping mock that reports ``use_dp=True``,
    ``parallel_size=4``, ``ep_size=2`` and assert construction raises.
    """
    pc = PretrainedConfig()
    pc.num_experts = 8
    pc.hidden_size = 512
    pc.intermediate_size = 512
    pc.torch_dtype = torch.bfloat16
    # Mock a Mapping where parallel_size > ep_size (ADP > EP).
    mapping_mock = mock.MagicMock()
    mapping_mock.rank = 0
    mapping_mock.tp_rank = 0
    mapping_mock.moe_tp_rank = 0
    mapping_mock.moe_tp_size = 1
    mapping_mock.moe_ep_size = 2
    mapping_mock.moe_ep_rank = 0
    mapping_mock.moe_cluster_size = 1
    mapping_mock.tp_size = 4
    mapping_mock.enable_attention_dp = True
    mapping_mock.moe_ep_group = [0, 1]
    mc = ModelConfig(
        pretrained_config=pc,
        mapping=mapping_mock,
        moe_backend="MEGAMOE",
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
    )
    with pytest.raises(AssertionError, match="ep_size == parallel_size"):
        MegaMoEFusedMoE(
            routing_method=RenormalizeMoeRoutingMethod(top_k=2),
            num_experts=8,
            hidden_size=512,
            intermediate_size=512,
            dtype=torch.bfloat16,
            model_config=mc,
        )


# ============================================================================
# UE8M0 bit helper — pure arithmetic, easy to verify
# ============================================================================
def test_ue8m0_uint8_to_fp32_roundtrip():
    """UE8M0 to fp32: ``value = 2**(exp - 127)``. Encoded in fp32 by
    shifting the exponent 23 bits left with sign=0 and mantissa=0."""
    # A few representative exponents:
    #   0   -> subnormal 2^-127 (kept as raw bits; not a regular float)
    #   127 -> 1.0
    #   128 -> 2.0
    #   126 -> 0.5
    sf = torch.tensor([127, 128, 126], dtype=torch.uint8)
    got = _ue8m0_uint8_to_fp32(sf)
    expected = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    torch.testing.assert_close(got, expected)
