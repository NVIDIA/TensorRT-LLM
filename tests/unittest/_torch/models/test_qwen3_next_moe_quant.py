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
"""Unquantized-MoE probe for the Qwen3Next / Qwen3.5 MTP layer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_exclude_modules
from tensorrt_llm._torch.models.modeling_qwen3_next import _experts_excluded_from_quant
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_HIDDEN_LAYERS = 40
# MTP layers are appended after the regular ones (MTPForCausalLM start_layer_idx).
MTP_LAYER_IDX = NUM_HIDDEN_LAYERS


def _model_config(exclude_modules, quant_algo=QuantAlgo.NVFP4):
    return SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=quant_algo,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=exclude_modules,
        ),
        pretrained_config=SimpleNamespace(
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_nextn_predict_layers=1,
        ),
    )


# All of these cover model.layers.40.mlp.experts under QuantConfig's matcher -- exact name,
# ancestor walk (a parent entry excludes everything below it), trailing ".*", "re:" regex -- in
# either the translated TRT-LLM namespace or the raw HF one, so every one must be recognized.
@pytest.mark.parametrize(
    "pattern",
    [
        f"model.layers.{MTP_LAYER_IDX}",
        f"model.layers.{MTP_LAYER_IDX}*",
        f"model.layers.{MTP_LAYER_IDX}.*",
        f"model.layers.{MTP_LAYER_IDX}.mlp",
        f"model.layers.{MTP_LAYER_IDX}.mlp*",
        f"model.layers.{MTP_LAYER_IDX}.mlp.*",
        f"model.layers.{MTP_LAYER_IDX}.mlp.experts",
        f"model.layers.{MTP_LAYER_IDX}.mlp.experts*",
        rf"re:model\.layers\.{MTP_LAYER_IDX}\..*",
        r"re:.*\.mlp\.experts$",
        "mtp.layers.0",
        "mtp.layers.0*",
        "mtp.layers.0.mlp",
        "mtp.layers.0.mlp*",
        "mtp.layers.0.mlp.experts*",
    ],
)
def test_experts_covered_by_exclusion(pattern):
    model_config = _model_config([pattern, "lm_head"])

    assert _experts_excluded_from_quant(model_config, MTP_LAYER_IDX)


# Leaf-level exclusions around the experts (what FP8 block-scale checkpoints emit) leave the
# experts quantized: forcing the CUTLASS fp8_blockscale GEMM there aborts on Blackwell.
@pytest.mark.parametrize(
    "pattern",
    [
        "mtp.fc",
        "mtp.norm",
        "mtp.layers.0.mlp.gate",
        "mtp.layers.0.mlp.shared_expert_gate",
        "mtp.layers.0.mlp.shared_expert.down_proj",
        "mtp.layers.0.self_attn.q_proj",
        f"model.layers.{MTP_LAYER_IDX}.mlp.gate",
        f"model.layers.{MTP_LAYER_IDX}.mlp.shared_expert.down_proj",
        f"model.layers.{MTP_LAYER_IDX - 1}*",
        "*kv_b_proj*",
        "*linear_attn.conv1d",
    ],
)
def test_experts_not_covered_by_exclusion(pattern):
    model_config = _model_config([pattern, "lm_head"])

    assert not _experts_excluded_from_quant(model_config, MTP_LAYER_IDX)


# The exclude_modules shapes checkpoints are known to emit, run through the same normalization
# the Qwen3.5 entry points apply. ``expected`` says whether the routed experts end up bf16.
@pytest.mark.parametrize(
    "shape,exclude_modules,quant_algo,expected",
    [
        ("whole MTP layer wildcard", ["mtp.layers.0*", "lm_head"], QuantAlgo.NVFP4, True),
        (
            "bare mtp subtree plus layer wildcard",
            ["mtp*", "mtp.layers.0*"],
            QuantAlgo.MIXED_PRECISION,
            True,
        ),
        (
            "leaf-level entries only",
            ["mtp.fc", "mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.shared_expert_gate"],
            QuantAlgo.FP8_BLOCK_SCALES,
            False,
        ),
        (
            "leaf-level entries incl. top-level norms",
            [
                "mtp.fc",
                "mtp.norm",
                "mtp.pre_fc_norm_embedding",
                "mtp.pre_fc_norm_hidden",
                "mtp.layers.0.input_layernorm",
                "mtp.layers.0.mlp.gate",
            ],
            QuantAlgo.FP8_BLOCK_SCALES,
            False,
        ),
    ],
)
def test_known_exclude_module_shapes(shape, exclude_modules, quant_algo, expected):
    model_config = _model_config(exclude_modules, quant_algo=quant_algo)
    _normalize_qwen35_exclude_modules(model_config)

    assert _experts_excluded_from_quant(model_config, MTP_LAYER_IDX) is expected, shape


def test_no_exclusions_needs_no_fallback():
    assert not _experts_excluded_from_quant(_model_config(None), MTP_LAYER_IDX)


@pytest.mark.parametrize(
    "pattern,layer_idx,expected",
    [
        # A regular layer's experts get the same treatment as the MTP layer's:
        # create_moe must see the unquantized config, or it picks a backend the
        # experts will not have once the exclusion pass runs.
        ("model.layers.5.mlp.experts", 5, True),
        ("model.layers.5.mlp", 5, True),
        ("model.layers.5*", 5, True),
        ("model.layers.5.mlp.experts", 6, False),
        # The raw HF mtp.* spelling only applies to layers past the last
        # regular one, so it must not leak onto a regular layer.
        ("mtp.layers.0.mlp.experts", 5, False),
        ("mtp.layers.0.mlp.experts", MTP_LAYER_IDX, True),
    ],
)
def test_regular_layers_are_covered_too(pattern, layer_idx, expected):
    model_config = _model_config([pattern, "lm_head"])

    assert _experts_excluded_from_quant(model_config, layer_idx) is expected


def test_missing_layer_idx_is_a_noop():
    assert not _experts_excluded_from_quant(_model_config(["model.layers.5*"]), None)


class _StopBlockInit(Exception):
    """Raised once the resolved MoE class is captured, to abort the build."""


def _build_moe_block(moe_backend, exclude_modules, layer_idx, *, sm, quant_config_dict=None):
    """Run sparse-MoE initialization through its ``create_moe`` call.

    ``sm`` is declared, not probed. Which implementation wins is a function of
    the machine, so pinning an expected class while the environment comes from
    whatever GPU CI happens to hand out asserts a property of the runner rather
    than of the code. Declaring it keeps the assertions about selection
    strategy and lets these cases run on any device. An empty dependency set
    goes with it, so an installed flashinfer cannot change the outcome either.
    """
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    from tensorrt_llm._torch.modules.fused_moe.impl_contract import MoEEnvironment
    from tensorrt_llm._torch.modules.fused_moe.impl_environment import override_moe_environment

    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            num_nextn_predict_layers=1,
            torch_dtype=torch.bfloat16,
            model_type="qwen3_next",
            mlp_bias=False,
        ),
        moe_backend=moe_backend,
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.NVFP4,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=exclude_modules,
        ),
        quant_config_dict=quant_config_dict,
    )
    captured = {}

    # ``fused_moe/__init__`` re-exports create_moe as a function, which shadows
    # the submodule of the same name, so ``import ...create_moe as m`` binds the
    # function. Both forms below resolve the submodule through sys.modules.
    from tensorrt_llm._torch.modules.fused_moe.create_moe import (
        resolve_moe_cls as real_resolve_moe_cls,
    )

    # Observe the resolution create_moe actually performs instead of
    # reproducing its argument list here: a copy drifts, and a copy that omits
    # e.g. swiglu_gptoss_style resolves to a different class than the layer
    # will be built with, which the assertions below would not catch.
    def _capture(model_config, **kwargs):
        captured["moe_backend"] = model_config.moe_backend
        captured["override"] = kwargs.get("override_quant_config")
        captured["moe_cls"] = real_resolve_moe_cls(model_config, **kwargs).__name__
        raise _StopBlockInit

    with (
        override_moe_environment(MoEEnvironment(sm=sm)),
        patch(
            "tensorrt_llm._torch.modules.fused_moe.create_moe.resolve_moe_cls",
            _capture,
        ),
        pytest.raises(_StopBlockInit),
    ):
        Qwen3NextSparseMoeBlock(model_config, aux_stream=None, layer_idx=layer_idx)
    return captured


@pytest.mark.parametrize("backend", ["CUTLASS", "TRTLLM", "DEEPGEMM", "CUTEDSL"])
@pytest.mark.parametrize("layer_idx", [5, MTP_LAYER_IDX])
def test_excluded_layer_builds_bf16_on_cutlass(backend, layer_idx):
    per_layer_quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    # The exclusion rewrites the request to CUTLASS and strips the quantization,
    # and unquantized Cutlass covers SM80+, so the declared SM only has to be a
    # real one -- this case is about the rewrite, not about hardware.
    captured = _build_moe_block(
        backend,
        [f"model.layers.{layer_idx}*"],
        layer_idx,
        sm=100,
        quant_config_dict={
            f"model.layers.{layer_idx}.mlp.experts": per_layer_quant_config,
        },
    )

    assert captured["moe_backend"] == "CUTLASS"
    assert captured["moe_cls"] == "CutlassFusedMoE"
    assert captured["override"] is not per_layer_quant_config
    assert not captured["override"].layer_quant_mode.has_any_quant(exclude_kv_cache=True)
    assert captured["override"].kv_cache_quant_algo == QuantAlgo.FP8


# "Requested backend wins" only holds where that backend can actually serve the
# layer, so each row pairs the request with an environment and an algorithm that
# make its family eligible. A single (algorithm, SM) shared by all four does not
# exist: FP8 block scales is Cutlass on SM90 and DeepGemm / TRTLLM-Gen on SM100,
# and CuteDSL only claims NVFP4.
@pytest.mark.parametrize(
    "backend,sm,quant_algo,expected_moe_cls",
    [
        ("CUTLASS", 90, QuantAlgo.FP8_BLOCK_SCALES, "CutlassFusedMoE"),
        ("TRTLLM", 100, QuantAlgo.FP8_BLOCK_SCALES, "TRTLLMGenFusedMoE"),
        ("DEEPGEMM", 100, QuantAlgo.FP8_BLOCK_SCALES, "DeepGemmFusedMoE"),
        ("CUTEDSL", 100, QuantAlgo.NVFP4, "CuteDslFusedMoE"),
    ],
    ids=[
        "cutlass_sm90_fp8_block",
        "trtllm_sm100_fp8_block",
        "deepgemm_sm100_fp8_block",
        "cutedsl_sm100_nvfp4",
    ],
)
def test_unexcluded_layer_keeps_configured_backend_and_layer_quant_config(
    backend, sm, quant_algo, expected_moe_cls
):
    per_layer_quant_config = QuantConfig(quant_algo=quant_algo)
    captured = _build_moe_block(
        backend,
        ["model.layers.7*"],
        5,
        sm=sm,
        quant_config_dict={
            "model.layers.5.mlp.experts": per_layer_quant_config,
        },
    )

    assert captured["moe_backend"] == backend
    assert captured["moe_cls"] == expected_moe_cls
    assert captured["override"] is per_layer_quant_config
