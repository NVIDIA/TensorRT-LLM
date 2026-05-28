# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static / config / weight-accounting tests for the Step3p7 bring-up.

These are the *reference_tier=static* checks called out in
``workspace/step-3-7-flash-fp8/acceptance-criteria.md``.  They do not need a
full B200 fleet to run — the goal is to prove that the Step3p7 checkpoint is
recognized as a Step3p7 text-generation model, that the per-layer attention
inventory and MoE topology are reported correctly, and that the weight
accounting cleanly separates consumed text-path keys from intentionally
ignored vision keys and plain-path MTP keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# Acceptance criterion #1 requires both checkpoint paths to be exercised.
CHECKPOINT_ROOT = Path("/home/scratch.kevxie_sw_1/workspace/quiet_harbor")
CHECKPOINT_FP8 = CHECKPOINT_ROOT / "step-3-7-flash-fp8_vv1"
CHECKPOINT_BF16 = CHECKPOINT_ROOT / "step-3-7-flash-bf16_vv1"

# Default checkpoint for the few helper tests that only need the layer-pattern /
# stacked-MoE / FP8 scan logic.  These are dtype-agnostic, so the FP8 checkpoint
# is the simpler choice (it also has the missing-scale tensors the scan test
# needs).
CHECKPOINT_PATH = CHECKPOINT_FP8


def _checkpoint_id(path: Path) -> str:
    name = path.name
    if "fp8" in name:
        return "fp8"
    if "bf16" in name:
        return "bf16"
    return name


CHECKPOINT_PATHS = [CHECKPOINT_FP8, CHECKPOINT_BF16]
CHECKPOINT_IDS = [_checkpoint_id(p) for p in CHECKPOINT_PATHS]
ALL_CHECKPOINTS_AVAILABLE = all(p.exists() for p in CHECKPOINT_PATHS)
SKIP_REASON_NO_CKPT = (
    "Step3p7 checkpoints not available; expected both "
    f"{CHECKPOINT_FP8} and {CHECKPOINT_BF16}.  Set up the workspace "
    "before running this test."
)


@pytest.fixture(scope="module", params=CHECKPOINT_PATHS, ids=CHECKPOINT_IDS)
def checkpoint_path(request) -> Path:
    if not request.param.exists():
        pytest.fail(
            f"Step3p7 checkpoint missing at {request.param}; acceptance criterion #1 "
            "requires both fp8 and bf16 checkpoints."
        )
    return request.param


@pytest.fixture(scope="module")
def config_dict(checkpoint_path):
    with open(checkpoint_path / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def safetensors_keys(checkpoint_path):
    import safetensors

    all_keys = []
    for f in sorted(checkpoint_path.glob("model-*.safetensors")):
        with safetensors.safe_open(str(f), framework="pt") as h:
            all_keys.extend(h.keys())
    return all_keys


def test_real_checkpoint_config_and_weight_accounting(
    checkpoint_path, config_dict, safetensors_keys
):
    """Acceptance criterion #1: recognize Step3p7 + weight accounting.

    Pass requires:

    - architectures == ["Step3p7ForConditionalGeneration"]
    - top model_type == "step3p7"; text model_type == "step3p5"
    - 45 text decoder layers; layer_types alternate full/sliding with the
      exact pattern documented in plan.md
    - All consumed text-path safetensors keys can be enumerated; only vision,
      plain-path MTP (layers 45..47), and the vision projector are ignored
    - TRT-LLM ``ModelConfig.from_pretrained`` resolves Step3p7 to the
      registered text-generation architecture and detects FP8_BLOCK_SCALES
      with the TRTLLM MoE backend hint.
    """
    # 1. Top-level config sanity.
    assert config_dict["architectures"] == ["Step3p7ForConditionalGeneration"]
    assert config_dict["model_type"] == "step3p7"
    text_cfg = config_dict["text_config"]
    assert text_cfg["model_type"] == "step3p5"
    assert text_cfg["num_hidden_layers"] == 45
    assert text_cfg["hidden_size"] == 4096
    assert text_cfg["vocab_size"] == 128896
    assert text_cfg["num_attention_heads"] == 64
    assert text_cfg["num_attention_groups"] == 8
    assert text_cfg["head_dim"] == 128
    assert text_cfg["moe_num_experts"] == 288
    assert text_cfg["moe_top_k"] == 8
    assert text_cfg["moe_router_scaling_factor"] == 3.0
    assert text_cfg["use_head_wise_attn_gate"] is True
    assert text_cfg["use_moe_router_bias"] is True
    assert text_cfg["need_fp32_gate"] is True

    # 2. Layer inventory: 45 decoder layers, layer_types alternating
    #    full/sliding/sliding/sliding pattern (full at idx 0,4,8,...,44).
    # The raw config.json layer_types array can be longer (48 entries) because
    # it also covers the 3 MTP layers (45..47).  We only validate the
    # decoder slice.
    layer_types = text_cfg["layer_types"]
    assert len(layer_types) >= 45
    for idx in range(45):
        lt = layer_types[idx]
        if idx % 4 == 0:
            assert lt == "full_attention", f"layer {idx}: expected full_attention, got {lt}"
        else:
            assert lt == "sliding_attention", f"layer {idx}: expected sliding_attention, got {lt}"

    # 3. Quant config: only the FP8 checkpoint carries it.  The BF16 reference
    #    checkpoint loads as plain bf16 with no quantization_config block.
    is_fp8_checkpoint = _checkpoint_id(checkpoint_path) == "fp8"
    if is_fp8_checkpoint:
        quant_cfg = config_dict["quantization_config"]
        assert quant_cfg["quant_method"] == "fp8"
        assert quant_cfg["weight_block_size"] == [128, 128]
        not_convert = set(quant_cfg["modules_to_not_convert"])
        # Attention / dense MLPs / shared experts / router gate must be excluded.
        for layer_idx in range(45):
            if layer_idx < 3:
                # Dense MLP layers: gate/up/down all bf16.
                for name in ("self_attn.q_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                    key = f"model.layers.{layer_idx}.{name}"
                    assert key in not_convert, f"dense layer {layer_idx} {name} must not be FP8"
            else:
                # MoE layers: routed gate/up/down are FP8; shared expert is bf16.
                for name in (
                    "self_attn.q_proj",
                    "moe.gate",
                    "share_expert.gate_proj",
                    "share_expert.up_proj",
                    "share_expert.down_proj",
                ):
                    key = f"model.layers.{layer_idx}.{name}"
                    assert key in not_convert, f"MoE layer {layer_idx} {name} must not be FP8"
    else:
        # BF16 checkpoint: no quantization_config and routed/shared experts both bf16.
        assert "quantization_config" not in config_dict, (
            "BF16 checkpoint should not carry a quantization_config block; "
            f"found: {config_dict.get('quantization_config')}"
        )

    # 4. Weight accounting on the actual safetensors keys.
    consumed_text_keys = []
    ignored_vision_keys = []
    ignored_mtp_keys = []
    text_layer_re = re.compile(r"^model\.layers\.(\d+)\.")
    for key in safetensors_keys:
        m = text_layer_re.match(key)
        if m:
            li = int(m.group(1))
            if li >= 45:
                ignored_mtp_keys.append(key)
            else:
                consumed_text_keys.append(key)
        elif key.startswith("vision_model.") or key.startswith("vit_large_projector"):
            ignored_vision_keys.append(key)
        elif key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            consumed_text_keys.append(key)
        else:
            # Anything else is an unaccounted-for key; fail.
            pytest.fail(f"Unaccounted-for safetensors key: {key}")

    # Sanity: every kind of key bucket is non-empty.
    assert len(consumed_text_keys) > 0, "no consumed text keys found"
    assert len(ignored_vision_keys) > 0, "no ignored vision keys found"
    assert len(ignored_mtp_keys) > 0, "no ignored MTP keys found; expected layers 45..47"

    # 5. TRT-LLM ModelConfig.from_pretrained resolves to the registered model.
    #    The FP8 checkpoint additionally exposes FP8_BLOCK_SCALES and the TRTLLM
    #    MoE backend; the BF16 checkpoint has no quant_algo.
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import Step3p7ForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import get_model_architecture
    from tensorrt_llm.quantization import QuantAlgo

    model_config = ModelConfig.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    pc = model_config.pretrained_config
    assert pc.architectures == ["Step3p7ForConditionalGeneration"]
    assert pc.num_hidden_layers == 45
    assert pc.vocab_size == 128896
    assert pc.hidden_size == 4096
    if is_fp8_checkpoint:
        assert model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        assert model_config.moe_backend.upper() == "TRTLLM"
    else:
        # BF16 checkpoint has no quantization; quant_algo should be unset.
        assert model_config.quant_config is None or model_config.quant_config.quant_algo is None, (
            f"BF16 checkpoint should not expose a quant_algo; got {model_config.quant_config.quant_algo}"
        )

    model_cls, _ = get_model_architecture(pc)
    assert model_cls is Step3p7ForCausalLM


@pytest.mark.parametrize("ckpt_path", CHECKPOINT_PATHS, ids=CHECKPOINT_IDS)
def test_per_layer_attention_geometry_helpers(ckpt_path):
    """Verify the per-layer head / RoPE helpers match config arithmetic.

    Runs against both FP8 and BF16 checkpoints because the per-layer attention
    geometry is identical between the two: only the routed-expert dtype differs.
    """
    if not ckpt_path.exists():
        pytest.fail(f"Step3p7 checkpoint missing at {ckpt_path}")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_step3p7 import (
        _is_moe_layer,
        _layer_attention_type,
        _layer_kv_heads,
        _layer_partial_rotary,
        _layer_query_heads,
        _layer_rope_theta,
        _layer_swiglu_limit,
        _layer_uses_rope_scaling,
    )

    model_config = ModelConfig.from_pretrained(str(ckpt_path), trust_remote_code=True)
    text_config = model_config.pretrained_config.text_config

    # Full vs sliding head counts.
    assert _layer_query_heads(text_config, 0) == 64
    assert _layer_kv_heads(text_config, 0) == 8
    assert _layer_query_heads(text_config, 1) == 96
    assert _layer_kv_heads(text_config, 1) == 8
    assert _layer_attention_type(text_config, 0) == "full_attention"
    assert _layer_attention_type(text_config, 1) == "sliding_attention"
    assert _layer_attention_type(text_config, 45) == "sliding_attention"
    assert _layer_query_heads(text_config, 45) == 96
    assert _layer_kv_heads(text_config, 45) == 8

    # RoPE per-layer.
    assert _layer_rope_theta(text_config, 0) == 5_000_000.0
    assert _layer_rope_theta(text_config, 1) == 10_000.0
    assert _layer_rope_theta(text_config, 45) == 10_000.0
    assert _layer_partial_rotary(text_config, 0) == 0.5
    assert _layer_partial_rotary(text_config, 1) == 1.0
    assert _layer_partial_rotary(text_config, 45) == 1.0
    assert _layer_uses_rope_scaling(text_config, 0) is True
    assert _layer_uses_rope_scaling(text_config, 1) is False

    # MoE vs dense.
    assert _is_moe_layer(text_config, 0) is False
    assert _is_moe_layer(text_config, 1) is False
    assert _is_moe_layer(text_config, 2) is False
    assert _is_moe_layer(text_config, 3) is True
    assert _is_moe_layer(text_config, 44) is True

    # SwiGLU clamp limits: nonzero only on layers 43 and 44 (per config).
    assert _layer_swiglu_limit(text_config, 0) is None
    assert _layer_swiglu_limit(text_config, 43) == 7.0
    assert _layer_swiglu_limit(text_config, 44) == 7.0
    assert _layer_swiglu_limit(text_config, 45) is None
    assert _layer_swiglu_limit(text_config, 43, shared=True) == 16.0
    assert _layer_swiglu_limit(text_config, 44, shared=True) == 16.0


def test_mtp_spec_config_defaults_to_checkpoint_layer_count():
    """Default ``MTPDecodingConfig`` should load all Step3p7 MTP layers."""
    if not CHECKPOINT_FP8.exists():
        pytest.fail(f"Step3p7 FP8 checkpoint missing at {CHECKPOINT_FP8}")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_step3p7 import _prepare_step3p7_mtp_spec_config
    from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec_config = MTPDecodingConfig()
    assert "max_draft_len" not in spec_config.model_fields_set
    model_config = ModelConfig.from_pretrained(
        str(CHECKPOINT_FP8),
        trust_remote_code=True,
        spec_config=spec_config,
    )

    update_spec_config_from_model_config(spec_config, model_config.pretrained_config)
    _prepare_step3p7_mtp_spec_config(model_config)

    assert spec_config.num_nextn_predict_layers == 3
    assert spec_config.max_draft_len == 3
    assert spec_config.max_total_draft_tokens == 3
    assert spec_config.spec_dec_mode.is_mtp_vanilla()

    explicit_spec_config = MTPDecodingConfig(max_draft_len=1)
    explicit_model_config = ModelConfig.from_pretrained(
        str(CHECKPOINT_FP8),
        trust_remote_code=True,
        spec_config=explicit_spec_config,
    )
    update_spec_config_from_model_config(
        explicit_spec_config,
        explicit_model_config.pretrained_config,
    )
    _prepare_step3p7_mtp_spec_config(explicit_model_config)

    assert explicit_spec_config.num_nextn_predict_layers == 3
    assert explicit_spec_config.max_draft_len == 1
    assert explicit_spec_config.max_total_draft_tokens == 1


def test_rewrite_mtp_weights_for_step3p7_uses_mtp_layout():
    """Checkpoint MTP keys should map to TRT-LLM's ``mtp_block`` layout."""
    import torch

    if not CHECKPOINT_FP8.exists():
        pytest.fail(f"Step3p7 FP8 checkpoint missing at {CHECKPOINT_FP8}")
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
    from tensorrt_llm._torch.models.modeling_step3p7 import rewrite_mtp_weights_for_step3p7

    model_config = ModelConfig.from_pretrained(str(CHECKPOINT_FP8), trust_remote_code=True)
    text_config = model_config.pretrained_config.text_config
    weights = {
        "model.layers.45.enorm.weight": torch.tensor([1.0]),
        "model.layers.45.eh_proj.weight": torch.tensor([2.0]),
        "model.layers.45.self_attn.q_proj.weight": torch.tensor([3.0]),
        "model.layers.45.mlp.gate_proj.weight": torch.tensor([4.0]),
        "model.layers.45.transformer.shared_head.norm.weight": torch.tensor([5.0]),
        "model.layers.45.transformer.shared_head.output.weight": torch.tensor([6.0]),
        "model.layers.44.self_attn.q_proj.weight": torch.tensor([7.0]),
    }
    weights = ConsumableWeightsDict(weights)

    rewritten = rewrite_mtp_weights_for_step3p7(weights, text_config)

    assert rewritten == 4
    assert "model.layers.45.enorm.weight" in weights
    assert "model.layers.45.eh_proj.weight" in weights
    assert "model.layers.45.mtp_block.self_attn.q_proj.weight" in weights
    assert "model.layers.45.mtp_block.mlp.gate_proj.weight" in weights
    assert "model.layers.45.shared_head.norm.weight" in weights
    assert "model.layers.45.shared_head.output.weight" in weights
    assert "model.layers.44.self_attn.q_proj.weight" in weights
    assert "model.layers.45.self_attn.q_proj.weight" not in weights
    assert "model.layers.45.transformer.shared_head.output.weight" not in weights


def test_step3p7_mtp_head_normalizes_before_output_projection(monkeypatch):
    """Step3p7 MTP applies shared-head norm only when producing draft logits."""
    import torch
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_step3p7 import Step3p7MTPHead
    from tensorrt_llm._torch.modules import rms_norm as rms_norm_module

    monkeypatch.setattr(rms_norm_module, "IS_FLASHINFER_AVAILABLE", False)

    text_config = PretrainedConfig()
    text_config.hidden_size = 2
    text_config.rms_norm_eps = 0.0
    text_config.torch_dtype = torch.float32
    text_config.vocab_size = 2
    top_config = PretrainedConfig()
    top_config.text_config = text_config
    head = Step3p7MTPHead(ModelConfig(pretrained_config=top_config))

    class CaptureOutput(torch.nn.Module):
        gather_output = True

        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, hidden_states, **kwargs):
            del kwargs
            self.seen = hidden_states.detach().clone()
            return hidden_states

    output = CaptureOutput()
    head.output = output
    hidden_states = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)

    logits = head(hidden_states, lm_head=None, attn_metadata=None)

    expected = hidden_states[-1:] / hidden_states[-1:].pow(2).mean(-1, keepdim=True).sqrt()
    assert torch.allclose(output.seen, expected)
    assert torch.allclose(logits, expected)


def test_split_stacked_moe_weights_expands_per_expert_keys():
    """Verify the stacked routed-expert tensors are expanded into per-expert keys.

    Source convention: ``...moe.gate_proj.weight`` shape ``(N, intermediate, hidden)``.
    Target convention (VANILLA MoE backend): ``...moe.experts.<e>.w1.weight``.

    Builds a synthetic weights dict over 2 small layers (one MoE, one dense)
    so the test runs in <1s without the real checkpoint.
    """
    import types

    import torch

    from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

    num_experts = 4
    intermediate = 8
    hidden = 16
    # Minimal text_config object: only the helpers used by
    # split_stacked_moe_weights need to be present.
    text_config = types.SimpleNamespace(
        num_hidden_layers=2,
        moe_num_experts=num_experts,
        moe_layers_enum=[1],  # Only layer 1 is MoE.
    )

    weights = {
        # Layer 0 (dense): no MoE stacked tensors.
        "model.layers.0.mlp.gate_proj.weight": torch.arange(
            intermediate * hidden, dtype=torch.float32
        ).reshape(intermediate, hidden),
        # Layer 1 (MoE): stacked routed-expert tensors.
        "model.layers.1.moe.gate_proj.weight": torch.arange(
            num_experts * intermediate * hidden, dtype=torch.float32
        ).reshape(num_experts, intermediate, hidden),
        "model.layers.1.moe.gate_proj.weight_scale_inv": torch.arange(
            num_experts * 1 * 2, dtype=torch.float32
        ).reshape(num_experts, 1, 2),
        "model.layers.1.moe.up_proj.weight": torch.arange(
            num_experts * intermediate * hidden, dtype=torch.float32
        ).reshape(num_experts, intermediate, hidden)
        + 1.0,
        "model.layers.1.moe.up_proj.weight_scale_inv": torch.arange(
            num_experts * 1 * 2, dtype=torch.float32
        ).reshape(num_experts, 1, 2)
        + 1.0,
        "model.layers.1.moe.down_proj.weight": torch.arange(
            num_experts * hidden * intermediate, dtype=torch.float32
        ).reshape(num_experts, hidden, intermediate),
        "model.layers.1.moe.down_proj.weight_scale_inv": torch.arange(
            num_experts * 2 * 1, dtype=torch.float32
        ).reshape(num_experts, 2, 1),
    }
    original_layer1_gate = weights["model.layers.1.moe.gate_proj.weight"]
    original_layer1_gate_scale = weights["model.layers.1.moe.gate_proj.weight_scale_inv"]

    layers_split = split_stacked_moe_weights(weights, text_config)
    assert layers_split == 1, f"expected exactly 1 MoE layer to be split, got {layers_split}"

    # Stacked keys must be removed.
    assert "model.layers.1.moe.gate_proj.weight" not in weights
    assert "model.layers.1.moe.gate_proj.weight_scale_inv" not in weights
    assert "model.layers.1.moe.up_proj.weight" not in weights
    assert "model.layers.1.moe.down_proj.weight" not in weights

    # Per-expert keys must be present with correct slicing.  Source
    # ``gate_proj`` → ``w1``, ``up_proj`` → ``w3``, ``down_proj`` → ``w2``.
    for expert_id in range(num_experts):
        key_w1 = f"model.layers.1.moe.experts.{expert_id}.w1.weight"
        key_w1_scale = f"model.layers.1.moe.experts.{expert_id}.w1.weight_scale_inv"
        key_w3 = f"model.layers.1.moe.experts.{expert_id}.w3.weight"
        key_w2 = f"model.layers.1.moe.experts.{expert_id}.w2.weight"
        assert key_w1 in weights, f"missing {key_w1}"
        assert key_w1_scale in weights, f"missing {key_w1_scale}"
        assert key_w3 in weights, f"missing {key_w3}"
        assert key_w2 in weights, f"missing {key_w2}"
        # Verify the slice corresponds to the original stacked element.
        assert torch.equal(weights[key_w1], original_layer1_gate[expert_id])
        assert torch.equal(weights[key_w1_scale], original_layer1_gate_scale[expert_id])
        # Shape sanity: per-expert tensor drops the leading expert dim.
        assert weights[key_w1].shape == (intermediate, hidden)
        assert weights[key_w2].shape == (hidden, intermediate)

    # Layer 0 (dense) must be untouched.
    assert "model.layers.0.mlp.gate_proj.weight" in weights


def test_split_stacked_moe_weights_no_op_when_no_stacked_keys():
    """When the weights dict has no stacked MoE keys, the helper is a no-op."""
    import types

    from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

    text_config = types.SimpleNamespace(
        num_hidden_layers=1,
        moe_num_experts=8,
        moe_layers_enum=[0],
    )
    weights = {"some.other.key": "not_a_tensor"}
    assert split_stacked_moe_weights(weights, text_config) == 0
    assert weights == {"some.other.key": "not_a_tensor"}


def test_split_stacked_moe_weights_handles_nvfp4_suffixes():
    """NVFP4 checkpoints carry ``weight_scale`` / ``weight_scale_2`` /
    ``input_scale`` alongside the packed ``weight``; the splitter must fan
    each suffix out to per-expert keys."""
    import types

    import torch

    from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

    num_experts = 3
    intermediate = 8
    hidden = 16
    text_config = types.SimpleNamespace(
        num_hidden_layers=1,
        moe_num_experts=num_experts,
        moe_layers_enum=[0],
    )
    weights = {
        # Packed FP4 weight: (E, I, H/2) uint8.
        "model.layers.0.moe.gate_proj.weight": torch.zeros(
            (num_experts, intermediate, hidden // 2), dtype=torch.uint8
        ),
        # Per-16 block scale: (E, I, H/16) fp8_e4m3fn.
        "model.layers.0.moe.gate_proj.weight_scale": torch.ones(
            (num_experts, intermediate, hidden // 16), dtype=torch.float8_e4m3fn
        ),
        # Per-tensor global scale: (E,) float32.
        "model.layers.0.moe.gate_proj.weight_scale_2": torch.arange(
            num_experts, dtype=torch.float32
        ),
        # Per-tensor input scale: (E,) float32.
        "model.layers.0.moe.gate_proj.input_scale": torch.arange(num_experts, dtype=torch.float32)
        + 10.0,
    }
    layers_split = split_stacked_moe_weights(weights, text_config)
    assert layers_split == 1
    assert "model.layers.0.moe.gate_proj.weight_scale" not in weights
    assert "model.layers.0.moe.gate_proj.weight_scale_2" not in weights
    assert "model.layers.0.moe.gate_proj.input_scale" not in weights
    for e in range(num_experts):
        assert weights[f"model.layers.0.moe.experts.{e}.w1.weight"].shape == (
            intermediate,
            hidden // 2,
        )
        assert weights[f"model.layers.0.moe.experts.{e}.w1.weight_scale"].shape == (
            intermediate,
            hidden // 16,
        )
        assert weights[f"model.layers.0.moe.experts.{e}.w1.weight_scale_2"].item() == float(e)
        assert weights[f"model.layers.0.moe.experts.{e}.w1.input_scale"].item() == float(e) + 10.0


def test_rewrite_language_model_keys_flattens_multimodal_namespace():
    """The NVFP4 multimodal Step3p7 checkpoint stores text decoder keys under
    ``model.language_model.*``; the loader must normalize them to ``model.*``
    so the existing module tree (built around ``model.layers.*``) is
    addressable. Vision keys move out from under ``model.`` to match the
    pre-existing ignored prefix list."""
    from tensorrt_llm._torch.models.modeling_step3p7 import rewrite_language_model_keys

    weights = {
        "lm_head.weight": "lm",
        "model.language_model.embed_tokens.weight": "embed",
        "model.language_model.layers.0.input_layernorm.weight": "in_ln",
        "model.language_model.norm.weight": "final_ln",
        "model.vision_model.conv1.weight": "vision",
        "model.vit_large_projector.weight": "projector",
    }
    n = rewrite_language_model_keys(weights)
    # 5 of 6 keys get renamed; lm_head is untouched.
    assert n == 5
    assert "lm_head.weight" in weights
    assert "model.embed_tokens.weight" in weights
    assert "model.layers.0.input_layernorm.weight" in weights
    assert "model.norm.weight" in weights
    assert "vision_model.conv1.weight" in weights
    assert "vit_large_projector.weight" in weights
    # Source keys must be gone.
    assert "model.language_model.embed_tokens.weight" not in weights
    assert "model.vision_model.conv1.weight" not in weights


def test_strip_language_model_prefix_preserves_regex_and_non_matching():
    """The quant-config exclude-modules normalizer must:
    - replace ``model.language_model.`` with ``model.``,
    - leave ``re:`` prefixed entries untouched, and
    - leave entries without the segment untouched."""
    from tensorrt_llm._torch.models.modeling_step3p7 import (
        strip_language_model_prefix_from_exclude_modules,
    )

    src = [
        "lm_head",
        "model.language_model.layers.0*",
        "model.language_model.layers.3.moe.gate",
        "re:^model\\.something\\..*",
        "model.vision_model*",
    ]
    out = strip_language_model_prefix_from_exclude_modules(src)
    assert out == [
        "lm_head",
        "model.layers.0*",
        "model.layers.3.moe.gate",
        "re:^model\\.something\\..*",
        "model.vision_model*",
    ]
    assert strip_language_model_prefix_from_exclude_modules(None) is None


def test_nvfp4_dequant_batched_round_trips_constant_values():
    """Sanity check the NVFP4 dequant helper. Packing a constant e2m1 index
    of 4 (= 2.0) and scaling with block_scale=1.0 and global_scale=0.5 must
    produce 1.0 everywhere."""
    import torch

    from tensorrt_llm._torch.models.modeling_step3p7 import _nvfp4_dequant_batched

    # Each byte holds (high<<4) | low.  Encoding index 4 in both nibbles:
    # (4 << 4) | 4 = 68.
    weight = torch.full((2, 8), 68, dtype=torch.uint8)
    block_scale = torch.ones((2, 1), dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor(0.5, dtype=torch.float32)
    out = _nvfp4_dequant_batched(weight, block_scale, global_scale)
    assert out.dtype == torch.bfloat16
    # 2.0 (e2m1) * 1.0 (block) * 0.5 (global) = 1.0
    assert torch.all(out == 1.0)
    # Encode a negative index (12 = -2.0): byte = (12 << 4) | 4 → high=-2, low=2.
    # Use K=16 so the per-16 block-scale shape works (K must be a multiple of 16).
    weight2 = torch.full((1, 8), (12 << 4) | 4, dtype=torch.uint8)
    out2 = _nvfp4_dequant_batched(
        weight2,
        torch.ones((1, 1), dtype=torch.float8_e4m3fn),
        torch.tensor(1.0, dtype=torch.float32),
    )
    # Even positions get value 2.0 (low nibble), odd positions get -2.0 (high nibble).
    assert out2[0, 0].item() == 2.0
    assert out2[0, 1].item() == -2.0
    # 3D batched form: (E=2, M=2, K_half=8) — exercises the per-expert
    # global-scale broadcast path used by the Python-clamp loader.
    w3 = torch.full((2, 2, 8), 68, dtype=torch.uint8)
    s1 = torch.ones((2, 2, 1), dtype=torch.float8_e4m3fn)
    s2 = torch.tensor([0.5, 0.25], dtype=torch.float32)
    out3 = _nvfp4_dequant_batched(w3, s1, s2)
    assert torch.all(out3[0] == 1.0)  # 2.0 * 1.0 * 0.5 = 1.0
    assert torch.all(out3[1] == 0.5)  # 2.0 * 1.0 * 0.25 = 0.5
