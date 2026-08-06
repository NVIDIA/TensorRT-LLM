# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DFlash structural scaffold: config schema + capture plumbing.

Mostly CPU-only (no checkpoint): validates that the synthetic drafter
generator emits the exact K2.7-Code-DFlash checkpoint schema the generic
DFlashForCausalLM loader expects, that DFlashDecodingConfig resolves to the
predicates the K3 one-engine wrapper relies on, and that KimiLinearModel
threads spec_metadata for per-layer hidden-state capture. The capture-buffer
round-trip needs a CUDA device (DFlashSpecMetadata allocates its buffer on
cuda) and is skip-guarded.
"""

import importlib.util
import inspect
import json
import os

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode


def _load_generator():
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "examples",
            "kimi_k3",
            "make_synthetic_dflash_drafter.py",
        )
    )
    if not os.path.exists(path):
        # The synthetic-drafter generator ships with the examples/kimi_k3
        # PR; on branches without it, the schema tests below skip.
        return None
    spec = importlib.util.spec_from_file_location("make_synthetic_dflash_drafter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()

requires_generator = pytest.mark.skipif(
    GEN is None,
    reason="examples/kimi_k3/make_synthetic_dflash_drafter.py not present on this branch",
)

# Reference: nvidia/Kimi-K2.7-Code-DFlash (69 tensors). The generator must
# reproduce this key/shape schema exactly at K2.7 dims — the generic
# DFlashForCausalLM.load_weights() key remapping is written against it.
K27_HIDDEN = 7168
K27_FC_SHAPE = (7168, 43008)  # hidden x (hidden * 6 capture layers)
K27_NUM_TENSORS = 69
K27_TARGET_LAYER_IDS = [1, 12, 24, 35, 47, 58]  # over 61 target layers

DFLASH_EXTRA_KEYS = {"fc.weight", "hidden_norm.weight"}
PER_LAYER_SUFFIXES = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
}


@requires_generator
def test_even_spacing_matches_k27_reference():
    assert GEN.even_target_layer_ids(61, 6) == K27_TARGET_LAYER_IDS


@requires_generator
def test_tensor_plan_matches_k27_schema():
    plan = GEN.drafter_tensor_plan(K27_HIDDEN, GEN.K27_DRAFTER, 6)
    assert len(plan) == K27_NUM_TENSORS
    assert plan["fc.weight"] == K27_FC_SHAPE
    keys = set(plan)
    assert DFLASH_EXTRA_KEYS | {"norm.weight"} <= keys
    for i in range(GEN.K27_DRAFTER["num_hidden_layers"]):
        for suffix in PER_LAYER_SUFFIXES:
            assert f"layers.{i}.{suffix}" in keys
    # No embeddings or lm_head: shared with the target model.
    assert not any("embed" in k or "lm_head" in k for k in keys)


# Reference: the K3 DSpark drafter dummy checkpoint from the training team
# (dummy-dspark0724, 2026-07-24; 73 tensors = K2.7's 69 + markov_w1/w2 +
# confidence_proj.{weight,bias}). Dims from its config.json.
K3_HIDDEN = 7168
K3_VOCAB = 163840
K3_NUM_TARGET_LAYERS = 93
K3_TARGET_LAYER_IDS = [1, 19, 37, 54, 72, 90]
K3_MASK_TOKEN_ID = 163606  # NOT vocab-2
K3_MARKOV_RANK = 256
K3_NUM_TENSORS = 73


@requires_generator
def test_even_spacing_matches_real_k3_drafter():
    """The real config's target_layer_ids follow the even-spacing convention."""
    assert GEN.even_target_layer_ids(K3_NUM_TARGET_LAYERS, 6) == K3_TARGET_LAYER_IDS


@requires_generator
def test_tensor_plan_matches_real_dspark_schema():
    """Plan must match the dummy-dspark0724 safetensors header exactly."""
    plan = GEN.drafter_tensor_plan(
        K3_HIDDEN,
        GEN.K3_DRAFTER,
        len(K3_TARGET_LAYER_IDS),
        vocab=K3_VOCAB,
        markov_rank=K3_MARKOV_RANK,
        use_confidence_head=True,
    )
    assert len(plan) == K3_NUM_TENSORS
    assert plan["fc.weight"] == (7168, 43008)
    assert plan["markov_w1.weight"] == (K3_VOCAB, K3_MARKOV_RANK)
    assert plan["markov_w2.weight"] == (K3_VOCAB, K3_MARKOV_RANK)
    # Confidence head reads [hidden, markov_features] concat -> in-dim 7424.
    assert plan["confidence_proj.weight"] == (1, K3_HIDDEN + K3_MARKOV_RANK)
    assert plan["confidence_proj.bias"] == (1,)
    # Per-layer backbone: K3 drafter has 32 Q heads / 8 KV heads / hd 128.
    assert plan["layers.0.self_attn.q_proj.weight"] == (4096, 7168)
    assert plan["layers.0.self_attn.k_proj.weight"] == (1024, 7168)
    assert plan["layers.0.mlp.gate_proj.weight"] == (12288, 7168)
    assert not any("embed" in k or "lm_head" in k for k in plan)


@requires_generator
def test_k3_drafter_config_is_dspark():
    cfg = GEN.drafter_config(
        K3_HIDDEN,
        K3_VOCAB,
        K3_NUM_TARGET_LAYERS,
        K3_TARGET_LAYER_IDS,
        K3_MASK_TOKEN_ID,
        GEN.K3_DRAFTER,
    )
    assert cfg["architectures"] == ["DFlashDraftModel"]
    assert cfg["model_type"] == "qwen3"
    dflash_cfg = cfg["dflash_config"]
    assert dflash_cfg["mask_token_id"] == K3_MASK_TOKEN_ID
    assert dflash_cfg["target_layer_ids"] == K3_TARGET_LAYER_IDS
    # DSpark extras, exactly as the real config declares them.
    assert dflash_cfg["projector_type"] == "dspark"
    assert dflash_cfg["causal"] is False
    assert dflash_cfg["use_swa"] is True and dflash_cfg["swa_window_size"] == 1024
    assert dflash_cfg["shift_label"] is True
    assert dflash_cfg["markov_rank"] == K3_MARKOV_RANK
    assert dflash_cfg["markov_head_type"] == "vanilla"
    assert dflash_cfg["use_confidence_head"] is True
    assert cfg["layer_types"] == ["sliding_attention"] * 6
    assert cfg["sliding_window"] == 1024
    assert cfg["rope_theta"] == 10000.0 and cfg["rope_scaling"] is None


@requires_generator
def test_drafter_config_drives_generic_dflash_path():
    cfg = GEN.drafter_config(K27_HIDDEN, 163840, 61, K27_TARGET_LAYER_IDS, 163838, GEN.K27_DRAFTER)
    # Unknown architecture label + model_type=qwen3 selects the generic
    # (non-Laguna) DFlashForCausalLM with a Qwen3ForCausalLM backbone.
    assert cfg["architectures"] == ["DFlashDraftModel"]
    assert not any("Laguna" in a for a in cfg["architectures"])
    assert cfg["model_type"] == "qwen3"
    assert cfg["dflash_config"] == {
        "mask_token_id": 163838,
        "target_layer_ids": K27_TARGET_LAYER_IDS,
    }
    assert cfg["synthetic_random_weights"] is True


@requires_generator
def test_generator_tiny_roundtrip(tmp_path):
    import subprocess
    import sys

    from safetensors import safe_open

    out = tmp_path / "tiny_dflash"
    subprocess.run(
        [sys.executable, GEN.__file__, "--tiny", "--out", str(out)],
        check=True,
    )
    with open(out / "config.json") as f:
        cfg = json.load(f)
    dflash_cfg = cfg["dflash_config"]
    plan = GEN.drafter_tensor_plan(
        cfg["hidden_size"],
        GEN.TINY,
        len(dflash_cfg["target_layer_ids"]),
        vocab=cfg["vocab_size"],
        markov_rank=GEN.TINY["markov_rank"],
        use_confidence_head=True,
    )
    with safe_open(out / "model.safetensors", "pt") as f:
        keys = set(f.keys())
        assert keys == set(plan)
        for k in keys:
            assert tuple(f.get_tensor(k).shape) == tuple(plan[k])
    assert dflash_cfg["mask_token_id"] == cfg["vocab_size"] - 2
    assert all(0 <= t < cfg["num_target_layers"] for t in dflash_cfg["target_layer_ids"])


@requires_generator
def test_generator_real_config_mode(tmp_path):
    """--config adopts a real drafter config.json verbatim (random weights,
    exact real module structure)."""
    import subprocess
    import sys

    from safetensors import safe_open

    real_cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "block_size": 4,
        "dflash_config": {
            "mask_token_id": 510,
            "target_layer_ids": [2, 5],
            "projector_type": "dspark",
            "markov_rank": 16,
            "use_confidence_head": True,
        },
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 512,
        "rope_theta": 12345.0,  # pass-through field the generator must keep
        "num_target_layers": 8,
    }
    cfg_path = tmp_path / "real_config.json"
    cfg_path.write_text(json.dumps(real_cfg))
    out = tmp_path / "synth"
    subprocess.run(
        [sys.executable, GEN.__file__, "--config", str(cfg_path), "--out", str(out)],
        check=True,
    )
    with open(out / "config.json") as f:
        emitted = json.load(f)
    # Verbatim adoption plus the synthetic marker.
    assert emitted["dflash_config"] == real_cfg["dflash_config"]
    assert emitted["rope_theta"] == 12345.0
    assert emitted["synthetic_random_weights"] is True
    plan = GEN.drafter_tensor_plan(
        real_cfg["hidden_size"],
        dict(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            block_size=4,
        ),
        len(real_cfg["dflash_config"]["target_layer_ids"]),
        vocab=real_cfg["vocab_size"],
        markov_rank=16,
        use_confidence_head=True,
    )
    assert {
        "markov_w1.weight",
        "markov_w2.weight",
        "confidence_proj.weight",
        "confidence_proj.bias",
    } <= set(plan)
    with safe_open(out / "model.safetensors", "pt") as f:
        assert set(f.keys()) == set(plan)


def test_dflash_decoding_config_predicates():
    from tensorrt_llm.llmapi import DFlashDecodingConfig

    cfg = DFlashDecodingConfig(max_draft_len=7)
    mode = cfg.spec_dec_mode
    assert mode == SpeculativeDecodingMode.DFLASH
    assert mode.is_dflash()
    # The K3 one-engine wrapper (SpecDecOneEngineForCausalLM) relies on
    # these to attach the external drafter and its capture metadata.
    assert mode.use_one_engine()
    assert mode.is_external_drafter()
    # KimiLinearForCausalLM admission: SA or DFlash only.
    assert mode.is_sa() or mode.is_dflash()
    assert not SpeculativeDecodingMode.NGRAM.is_sa()
    assert not SpeculativeDecodingMode.NGRAM.is_dflash()


def test_kimi_linear_model_threads_spec_metadata():
    pytest.importorskip("fla")
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearModel

    # The capture hook only fires if spec_metadata is an explicit parameter
    # (previously it was silently swallowed by **kwargs).
    params = inspect.signature(KimiLinearModel.forward).parameters
    assert "spec_metadata" in params
    src = inspect.getsource(KimiLinearModel.forward)
    assert "maybe_capture_hidden_states" in src


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlashSpecMetadata allocates capture buffer on cuda"
)
def test_dflash_spec_metadata_capture_prefix_sum_convention():
    """K3 passes the full post-layer prefix sum with residual=None; the
    buffer must then hold the hidden state verbatim (no residual add)."""
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    hidden_size, max_tokens = 16, 8
    md = DFlashSpecMetadata(
        max_draft_len=4,
        max_total_draft_tokens=4,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
        max_num_requests=2,
        max_num_tokens=max_tokens,
        hidden_size=hidden_size,
        layers_to_capture=[1, 3],
        dtype=torch.bfloat16,
    )
    assert md.is_layer_capture(1) and md.is_layer_capture(3)
    assert not md.is_layer_capture(2)

    h1 = torch.randn(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    h3 = torch.randn_like(h1)
    md.maybe_capture_hidden_states(1, h1, None)
    md.maybe_capture_hidden_states(2, torch.randn_like(h1), None)  # no-op
    md.maybe_capture_hidden_states(3, h3, None)
    captured = md.get_hidden_states(max_tokens)
    assert captured.shape == (max_tokens, 2 * hidden_size)
    torch.testing.assert_close(captured[:, :hidden_size], h1)
    torch.testing.assert_close(captured[:, hidden_size:], h3)
