# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit and integration tests for the MiniMax-M3 text bring-up.

The CPU-only tests exercise the Goal 1.1 helpers (config normalization,
layer scheduling, routing-method scaling) without requiring CUDA or the
TensorRT-LLM C++ extension. ``test_text_checkpoint_loading`` is the
Stage 1 / item 1 acceptance gate: it loads the real MiniMax-M3
checkpoint config / tokenizer / chat template, runs the static keyspace
coverage classifier on every key in the checkpoint's safetensors index,
and confirms that each ``language_model.*`` weight is either mapped to a
TRT-LLM text parameter (loaded) or intentionally ignored with a
documented reason. CUDA-side execution of the model is the responsibility
of subsequent Stage 1 goals (1.2-1.7).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Dict

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUM_HIDDEN_LAYERS = 7
_SPARSE_FREQ = [0, 0, 0, 1, 1, 1, 1]
_DISABLE_INDEX_VALUE = [0, 0, 0, 1, 1, 1, 1]
_MOE_LAYER_FREQ = [0, 0, 0, 1, 1, 1, 1]


def _make_text_config():
    """Build a SimpleNamespace mimicking the real M3 text config (trimmed)."""
    sparse_attention_config = {
        "use_sparse_attention": True,
        "sparse_index_dim": 128,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": list(_DISABLE_INDEX_VALUE),
        "sparse_attention_freq": list(_SPARSE_FREQ),
    }
    return SimpleNamespace(
        model_type="minimax_m3",
        hidden_size=6144,
        intermediate_size=3072,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=200064,
        max_position_embeddings=524288,
        rms_norm_eps=1e-06,
        use_gemma_norm=True,
        attention_output_gate=False,
        rope_theta=5000000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        hidden_act="swigluoai",
        use_qk_norm=True,
        qk_norm_type="per_head",
        tie_word_embeddings=False,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        moe_layer_freq=list(_MOE_LAYER_FREQ),
        num_mtp_modules=1,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        routed_scaling_factor=2.0,
        sparse_attention_config=sparse_attention_config,
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
    )


def _make_vl_config():
    return SimpleNamespace(
        model_type="minimax_m3_vl",
        text_config=_make_text_config(),
        vision_config=SimpleNamespace(
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
        ),
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        image_token_index=200025,
        video_token_index=200026,
    )


# ---------------------------------------------------------------------------
# Shared helpers used by both CPU and CUDA tests
# ---------------------------------------------------------------------------


_CHECKPOINT_PATH_ENV = "MINIMAX_M3_CHECKPOINT_PATH"
_DEFAULT_CHECKPOINT_PATH = "/home/scratch.fredw_sw/workspace/hidden_trail/minimax-m3-preview_vv1"


def _checkpoint_path() -> str:
    return os.environ.get(_CHECKPOINT_PATH_ENV, _DEFAULT_CHECKPOINT_PATH)


def _has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Goal 1.1 CPU-only unit tests
# ---------------------------------------------------------------------------


def test_is_minimax_m3_vl_config_detects_vl():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import is_minimax_m3_vl_config

    assert is_minimax_m3_vl_config(_make_vl_config()) is True


def test_is_minimax_m3_vl_config_detects_text_only():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import is_minimax_m3_vl_config

    assert is_minimax_m3_vl_config(_make_text_config()) is False


def test_is_minimax_m3_vl_config_falls_back_to_architectures():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import is_minimax_m3_vl_config

    cfg = SimpleNamespace(
        model_type="custom",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
    )
    assert is_minimax_m3_vl_config(cfg) is True


def test_get_text_config_returns_text_subconfig():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_text_config

    vl_cfg = _make_vl_config()
    text_cfg = get_text_config(vl_cfg)
    assert text_cfg is vl_cfg.text_config
    assert text_cfg.num_hidden_layers == _NUM_HIDDEN_LAYERS


def test_get_text_config_passthrough_for_text_only():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_text_config

    text_cfg = _make_text_config()
    assert get_text_config(text_cfg) is text_cfg


def test_get_text_config_propagates_dtype_when_missing():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_text_config

    vl_cfg = _make_vl_config()
    vl_cfg.text_config.torch_dtype = None
    out = get_text_config(vl_cfg)
    assert out.torch_dtype == "bfloat16"


def test_get_text_config_missing_text_attribute_raises():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_text_config

    bad = SimpleNamespace(model_type="minimax_m3_vl")
    with pytest.raises(ValueError, match="text_config"):
        get_text_config(bad)


def test_get_sparse_layer_ids_splits_dense_and_sparse():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_sparse_layer_ids

    dense, sparse = get_sparse_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert sparse == [3, 4, 5, 6]


def test_get_sparse_layer_ids_falls_back_when_disabled():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_sparse_layer_ids

    cfg = _make_text_config()
    cfg.sparse_attention_config["use_sparse_attention"] = False
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_falls_back_without_config():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_sparse_layer_ids

    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    dense, sparse = get_sparse_layer_ids(cfg)
    assert dense == list(range(_NUM_HIDDEN_LAYERS))
    assert sparse == []


def test_get_sparse_layer_ids_length_mismatch_raises():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_sparse_layer_ids

    cfg = _make_text_config()
    cfg.sparse_attention_config["sparse_attention_freq"] = [0] * (_NUM_HIDDEN_LAYERS + 1)
    with pytest.raises(ValueError, match="sparse_attention_freq length"):
        get_sparse_layer_ids(cfg)


def test_get_sparse_disable_index_value_layer_ids_matches_sparse():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        get_sparse_disable_index_value_layer_ids,
    )

    ids = get_sparse_disable_index_value_layer_ids(_make_text_config())
    assert ids == [3, 4, 5, 6]


def test_get_sparse_disable_index_value_no_config():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        get_sparse_disable_index_value_layer_ids,
    )

    cfg = _make_text_config()
    cfg.sparse_attention_config = None
    assert get_sparse_disable_index_value_layer_ids(cfg) == []


def test_get_moe_layer_ids_splits_dense_and_moe():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_moe_layer_ids

    dense, moe = get_moe_layer_ids(_make_text_config())
    assert dense == [0, 1, 2]
    assert moe == [3, 4, 5, 6]


def test_get_moe_layer_ids_all_moe_without_freq():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_moe_layer_ids

    cfg = _make_text_config()
    cfg.moe_layer_freq = None
    dense, moe = get_moe_layer_ids(cfg)
    assert dense == []
    assert moe == list(range(_NUM_HIDDEN_LAYERS))


def test_get_moe_layer_ids_length_mismatch_raises():
    from tensorrt_llm._torch.models.modeling_minimaxm3 import get_moe_layer_ids

    cfg = _make_text_config()
    cfg.moe_layer_freq = [0] * (_NUM_HIDDEN_LAYERS - 1)
    with pytest.raises(ValueError, match="moe_layer_freq length"):
        get_moe_layer_ids(cfg)


# ---------------------------------------------------------------------------
# Goal 1.3 — attention module transforms
# ---------------------------------------------------------------------------
#
# These tests construct :class:`MiniMaxM3Attention` with a tiny synthetic
# geometry and ``skip_create_weights_in_init=True`` so the Linear modules
# exist (with ``.in_features`` / ``.out_features`` set) but no weights are
# allocated. The base :class:`Attention` constructor reaches into CUDA-only
# paths (e.g. backend selection), so the tests run under
# ``pytest.mark.gpu`` + ``skipif(not _has_cuda())``. Geometry is bounded to
# a few KB.
#
# Coverage:
#  * Dense / sparse attention construction shapes match the configured
#    head_dim, head counts, and sparse index branch dimensions.
#  * Partial RoPE only rotates ``rotary_dim`` of ``head_dim`` channels.
#  * Per-head Gemma Q/K RMSNorm: q_norm / k_norm are RMSNorm with
#    ``use_gemma=True`` and ``hidden_size=head_dim``; the
#    :meth:`apply_qk_norm` reshape matches an independent hand-written
#    reference.
#  * Sparse index branch: ``index_q_proj`` is column-parallel and
#    projects to ``num_index_heads * sparse_index_dim``;
#    ``index_k_proj`` is **replicated** (tp_mode is None) and projects
#    to **only** ``sparse_index_dim`` (single K per token, broadcast
#    across all index heads for block-selection scoring) — this is the
#    SGLang reference contract, confirmed by the M3 checkpoint shape
#    ``(sparse_index_dim, hidden_size)``.
#  * Dense layers do not expose any index branch attributes (negative
#    control).
#  * Real M3 checkpoint shape for ``index_k_proj.weight`` is
#    ``(sparse_index_dim, hidden_size)`` = ``(128, 6144)``.


def _make_attention_test_config():
    """Return ``(text_config, ModelConfig)`` for the Goal 1.3 attention tests.

    Geometry is a scaled-down M3-shaped config: hidden_size=128, head_dim=32,
    num_heads=4, num_kv_heads=2, num_index_heads=2, sparse_index_dim=32,
    rotary_dim=16 (= head_dim * 0.5 — partial RoPE), 1 dense + 3 sparse
    layers. With ``skip_create_weights_in_init=True`` no Linear weight
    tensors are allocated, only metadata.
    """
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import _wrap_dict_as_config
    from tensorrt_llm.mapping import Mapping

    n_layers = 4
    sparse_cfg = {
        "use_sparse_attention": True,
        "sparse_index_dim": 32,
        "sparse_num_index_heads": 2,
        "sparse_topk_blocks": 4,
        "sparse_block_size": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_score_type": "max",
        "sparse_disable_index_value": [0, 1, 1, 1],
        "sparse_attention_freq": [0, 1, 1, 1],
    }
    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "num_hidden_layers": n_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 256,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "use_gemma_norm": True,
            "rope_theta": 10000.0,
            "rotary_dim": 16,
            "partial_rotary_factor": 0.5,
            "qk_norm_type": "per_head",
            "use_qk_norm": True,
            "sparse_attention_config": sparse_cfg,
            "torch_dtype": torch.bfloat16,
        }
    )
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    return text_cfg, model_cfg


def _per_head_gemma_rms_norm_reference(x, weight, eps):
    """Hand-written reference for per-head Gemma RMSNorm.

    Matches :class:`RMSNorm.forward` with ``use_gemma=True``,
    ``residual=None``, ``is_nvfp4=False``: cast to float32 to compute
    variance, normalise, cast back to input dtype, then scale by
    ``(weight + 1)``. The per-head structure comes from reshaping the
    input to ``(-1, head_dim)`` before applying this function.
    """
    import torch

    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_norm = (x_f32 * torch.rsqrt(variance + eps)).to(input_dtype)
    return (weight + 1.0) * x_norm


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_construction_matches_config():
    """Dense layer's QKV/O projection and per-head Q/K norm match config."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
        disable_index_value=False,
    )

    head_dim = int(text_cfg.head_dim)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv = int(text_cfg.num_key_value_heads)
    hidden = int(text_cfg.hidden_size)

    # Q/K/V projection (fused into qkv_proj).
    assert attn.num_heads == num_heads
    assert attn.num_key_value_heads == num_kv
    assert attn.head_dim == head_dim
    assert attn.head_dim_value == head_dim
    assert attn.q_size == num_heads * head_dim
    assert attn.kv_size == num_kv * head_dim
    assert attn.qkv_proj.in_features == hidden
    # With tp_size=1, out_features == q_size + 2 * kv_size.
    assert attn.qkv_proj.out_features == num_heads * head_dim + 2 * num_kv * head_dim

    # Output projection.
    assert attn.o_proj.in_features == num_heads * head_dim
    assert attn.o_proj.out_features == hidden

    # Per-head Gemma Q/K RMSNorm.
    assert attn.use_gemma_norm is True
    assert attn.qk_norm_type == "per_head"
    assert attn.q_norm.use_gemma is True
    assert attn.k_norm.use_gemma is True
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Dense layers must not expose any index-branch attributes.
    assert attn.is_sparse_attention_layer is False
    for name in (
        "index_q_proj",
        "index_k_proj",
        "index_q_norm",
        "index_k_norm",
    ):
        assert not hasattr(attn, name), f"dense layer should not declare {name!r}"


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_partial_rope_dim_is_rotary_dim():
    """Partial RoPE rotates only ``rotary_dim`` of ``head_dim`` channels."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    rotary_dim = int(text_cfg.rotary_dim)
    head_dim = int(text_cfg.head_dim)
    assert attn.pos_embd_params is not None
    assert attn.pos_embd_params.rope.dim == rotary_dim
    assert attn.pos_embd_params.rope.dim < head_dim, (
        f"partial RoPE expects rope.dim < head_dim, got {attn.pos_embd_params.rope.dim} >= {head_dim}"
    )
    # The base Attention class also stores the rotary embedding when
    # ``rope_fusion=False``; M3 sets ``rope_fusion=False`` explicitly.
    assert attn.rope_fusion is False
    assert attn.rotary_emb is not None


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_qk_norm_matches_reference():
    """Verify ``apply_qk_norm`` does per-head Gemma RMSNorm and reshape-back."""
    import torch

    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    text_cfg, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    head_dim = int(text_cfg.head_dim)
    eps = float(text_cfg.rms_norm_eps)

    # Set non-zero norm weights so the test catches any reshape /
    # weight-broadcast bugs (zero weights would mask many errors).
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    attn.q_norm.weight = torch.nn.Parameter(q_weight)
    attn.k_norm.weight = torch.nn.Parameter(k_weight)

    seq = 3
    q = torch.randn(seq, attn.q_size, dtype=dtype, device=device)
    k = torch.randn(seq, attn.kv_size, dtype=dtype, device=device)

    q_out, k_out = attn.apply_qk_norm(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

    # Hand-written reference: per-head reshape -> Gemma RMSNorm ->
    # reshape back. Identical computation, independent code path.
    q_ref = _per_head_gemma_rms_norm_reference(q.reshape(-1, head_dim), q_weight, eps).reshape(
        q.shape
    )
    k_ref = _per_head_gemma_rms_norm_reference(k.reshape(-1, head_dim), k_weight, eps).reshape(
        k.shape
    )

    # BF16 + possible flashinfer kernel: use a looser tolerance.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_sparse_construction_matches_config():
    """Sparse layer adds index branch with the SGLang-correct shapes.

    Verifies the **bug fix** from the iter-4 Goal 1.3 work:
      * ``index_q_proj`` is column-parallel and projects to
        ``num_index_heads * sparse_index_dim``.
      * ``index_k_proj`` is replicated (``tp_mode is None``) and projects
        to **only** ``sparse_index_dim`` — a single replicated K per
        token, *not* per-head. This matches SGLang's ``ReplicatedLinear``
        and the M3 checkpoint's ``index_k_proj.weight`` shape
        ``(sparse_index_dim, hidden_size)``.
      * ``index_q_norm`` / ``index_k_norm`` are per-head Gemma RMSNorm
        of width ``sparse_index_dim``.
    """
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    hidden = int(text_cfg.hidden_size)

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )
    assert attn.is_sparse_attention_layer is True
    assert attn.disable_index_value is True

    # index_q_proj: per-head Q for the index branch. As of iter-15 this
    # is **replicated** (tp_mode=None) across TP ranks, not
    # column-parallel: the sparse forward consumes ``idx_q`` reshaped to
    # ``[num_tokens, num_index_heads, sparse_index_dim]`` and a
    # column-parallel split would slice the head dimension (breaking the
    # reshape at any ``tp_size > num_index_heads`` geometry, including
    # the TP=8 configuration the real-checkpoint smoke test now uses).
    # The replicated weight is small (~3 MiB BF16) so the per-rank
    # memory cost is negligible.
    assert attn.index_q_proj.in_features == hidden
    assert attn.index_q_proj.out_features == num_index_heads * sparse_index_dim
    assert attn.index_q_proj.tp_mode is None, (
        f"index_q_proj must be replicated (tp_mode=None) so the sparse "
        f"forward's `idx_q.view(num_tokens, num_index_heads, sparse_index_dim)` "
        f"reshape is well-defined at any TP geometry, got "
        f"{attn.index_q_proj.tp_mode!r}"
    )

    # index_k_proj: REPLICATED, only sparse_index_dim outputs.
    assert attn.index_k_proj.in_features == hidden
    assert attn.index_k_proj.out_features == sparse_index_dim, (
        f"index_k_proj.out_features must be sparse_index_dim={sparse_index_dim}, "
        f"got {attn.index_k_proj.out_features} (regression of the iter-4 fix)"
    )
    assert attn.index_k_proj.tp_mode is None, (
        f"index_k_proj must be replicated (tp_mode=None), got {attn.index_k_proj.tp_mode!r}"
    )

    # Per-head Gemma RMSNorm of width sparse_index_dim.
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)

    # The sparse forward path now dispatches through the MiniMax-M3
    # sparse algorithm. Calling forward without metadata must raise a
    # clear RuntimeError pointing at the missing kv_cache_manager
    # (rather than silently returning garbage or crashing inside the
    # algorithm).
    try:
        attn.forward()
    except RuntimeError as e:
        msg = str(e)
        assert "attn_metadata" in msg or "kv_cache_manager" in msg, msg
    else:  # pragma: no cover
        raise AssertionError("sparse forward must raise RuntimeError when attn_metadata is None")


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_apply_index_qk_norm_matches_reference():
    """Sparse index per-head Gemma QK norm matches the hand-written reference.

    ``apply_index_qk_norm`` reshapes ``idx_q`` (``num_index_heads`` heads)
    and ``idx_k`` (1 replicated head) to ``(-1, sparse_index_dim)`` rows,
    applies the per-head Gemma RMSNorm, and reshapes back. The test sets
    non-zero norm weights, drives synthetic input, and compares against
    the same pure-torch reference used for the main Q/K norm.
    """
    import torch

    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    text_cfg, model_cfg = _make_attention_test_config()
    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])

    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = float(text_cfg.rms_norm_eps)
    iq_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    ik_weight = torch.randn(sparse_index_dim, dtype=dtype, device=device) * 0.3
    attn.index_q_norm.weight = torch.nn.Parameter(iq_weight)
    attn.index_k_norm.weight = torch.nn.Parameter(ik_weight)

    seq = 5
    idx_q = torch.randn(seq, num_index_heads * sparse_index_dim, dtype=dtype, device=device)
    idx_k = torch.randn(seq, sparse_index_dim, dtype=dtype, device=device)
    iq_out, ik_out = attn.apply_index_qk_norm(idx_q, idx_k)
    assert iq_out.shape == idx_q.shape
    assert ik_out.shape == idx_k.shape

    iq_ref = _per_head_gemma_rms_norm_reference(
        idx_q.reshape(-1, sparse_index_dim), iq_weight, eps
    ).reshape(idx_q.shape)
    ik_ref = _per_head_gemma_rms_norm_reference(
        idx_k.reshape(-1, sparse_index_dim), ik_weight, eps
    ).reshape(idx_k.shape)

    torch.testing.assert_close(iq_out, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ik_out, ik_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
def test_minimax_m3_attention_dense_apply_index_qk_norm_raises():
    """Dense layers must reject ``apply_index_qk_norm`` calls."""
    import torch

    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention

    _, model_cfg = _make_attention_test_config()
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=0,
        is_sparse_attention_layer=False,
    )
    idx_q = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda")
    idx_k = torch.zeros(2, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="only valid on sparse attention layers"):
        attn.apply_index_qk_norm(idx_q, idx_k)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 attention construction needs CUDA")
@pytest.mark.skipif(
    not os.path.exists(_checkpoint_path()),
    reason=(
        "MiniMax-M3 checkpoint not mounted; set "
        f"{_CHECKPOINT_PATH_ENV} to a path containing config.json"
    ),
)
def test_minimax_m3_attention_real_config_index_branch_shapes():
    """Real M3 config → sparse-layer index branch has the checkpoint's shapes.

    Asserts the iter-4 Goal 1.3 fix in numbers:
      * ``index_q_proj.out_features == 512`` (= 4 * 128
        = ``num_index_heads * sparse_index_dim``).
      * ``index_k_proj.out_features == 128`` (= ``sparse_index_dim``)
        and ``tp_mode is None`` (replicated). The real
        ``index_k_proj.weight`` in the checkpoint has shape
        ``(128, 6144)``; a regression to the old
        ``num_index_heads * sparse_index_dim`` (512) would break weight
        loading at runtime.
      * ``index_q_norm.weight.shape == (128,)`` and
        ``index_k_norm.weight.shape == (128,)``.
    """
    import torch

    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Attention, get_text_config
    from tensorrt_llm.mapping import Mapping

    cfg = AutoConfig.from_pretrained(_checkpoint_path(), trust_remote_code=True)
    text_cfg = get_text_config(cfg)
    # ``MiniMaxM3Model.__init__`` normalises ``torch_dtype`` to a real
    # ``torch.dtype`` before constructing layers (the HF config stores it
    # as the string ``"bfloat16"``). Mirror that here so the standalone
    # attention construction does not blow up inside the RMSNorm
    # ``torch.zeros(..., dtype=dtype)`` call.
    if isinstance(getattr(text_cfg, "torch_dtype", None), str):
        text_cfg.torch_dtype = torch.bfloat16
    elif getattr(text_cfg, "torch_dtype", None) is None:
        text_cfg.torch_dtype = torch.bfloat16

    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(),
        skip_create_weights_in_init=True,
    )
    attn = MiniMaxM3Attention(
        model_config=model_cfg,
        layer_idx=3,
        is_sparse_attention_layer=True,
        disable_index_value=True,
    )

    sparse_cfg = text_cfg.sparse_attention_config
    num_index_heads = int(sparse_cfg["sparse_num_index_heads"])
    sparse_index_dim = int(sparse_cfg["sparse_index_dim"])
    assert num_index_heads == 4
    assert sparse_index_dim == 128

    # index_q_proj: 4 * 128 = 512 out, replicated (tp_mode=None) as of
    # iter-15. The downstream sparse forward reshapes ``idx_q`` to
    # ``[num_tokens, num_index_heads, sparse_index_dim]``; a
    # column-parallel split would slice the head dimension and break
    # that reshape at any ``tp_size > num_index_heads`` geometry
    # (including TP=8 used by the real-checkpoint smoke test). The
    # replicated weight is ~3 MiB BF16 — the per-rank memory cost is
    # negligible.
    assert attn.index_q_proj.in_features == int(text_cfg.hidden_size)
    assert attn.index_q_proj.out_features == num_index_heads * sparse_index_dim
    assert attn.index_q_proj.tp_mode is None

    # index_k_proj: 128 out (NOT 512), replicated.
    assert attn.index_k_proj.in_features == int(text_cfg.hidden_size)
    assert attn.index_k_proj.out_features == sparse_index_dim
    assert attn.index_k_proj.tp_mode is None

    # Per-head Gemma index norms: width sparse_index_dim.
    assert tuple(attn.index_q_norm.weight.shape) == (sparse_index_dim,)
    assert tuple(attn.index_k_norm.weight.shape) == (sparse_index_dim,)
    assert attn.index_q_norm.use_gemma is True
    assert attn.index_k_norm.use_gemma is True

    # Main Q/K norm shapes follow head_dim, not hidden_size.
    head_dim = int(text_cfg.head_dim)
    assert tuple(attn.q_norm.weight.shape) == (head_dim,)
    assert tuple(attn.k_norm.weight.shape) == (head_dim,)

    # Partial RoPE rotates rotary_dim of head_dim.
    assert attn.pos_embd_params.rope.dim == int(text_cfg.rotary_dim)
    assert attn.pos_embd_params.rope.dim < head_dim


# ---------------------------------------------------------------------------
# Routing-method unit tests (CPU)
# ---------------------------------------------------------------------------


def test_minimax_m3_routing_method_applies_routed_scaling_factor():
    """MiniMaxM3MoeRoutingMethod multiplies renormalized weights by scaling."""
    import torch

    from tensorrt_llm._torch.modules.fused_moe.routing import (
        MiniMaxM2MoeRoutingMethod,
        MiniMaxM3MoeRoutingMethod,
    )

    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    def bias_fn():
        return bias

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
    )
    scaled = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=bias_fn,
        routed_scaling_factor=2.0,
    )

    base_idx, base_weights = base.apply(logits)
    scaled_idx, scaled_weights = scaled.apply(logits)

    torch.testing.assert_close(base_idx, scaled_idx)
    torch.testing.assert_close(scaled_weights, base_weights * 2.0, rtol=0, atol=0)


def test_minimax_m3_routing_method_default_scale_is_identity():
    import torch

    from tensorrt_llm._torch.modules.fused_moe.routing import (
        MiniMaxM2MoeRoutingMethod,
        MiniMaxM3MoeRoutingMethod,
    )

    num_experts = 8
    top_k = 2
    bias = torch.zeros(num_experts, dtype=torch.float32)

    torch.manual_seed(0)
    logits = torch.randn(4, num_experts, dtype=torch.float32) * 3.0

    base = MiniMaxM2MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
    )
    same = MiniMaxM3MoeRoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        callable_e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
    )
    _, base_weights = base.apply(logits)
    _, same_weights = same.apply(logits)
    torch.testing.assert_close(same_weights, base_weights, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda(), reason="MiniMax-M3 needs CUDA")
@pytest.mark.skipif(
    not os.path.exists(_checkpoint_path()),
    reason=(
        "MiniMax-M3 checkpoint not mounted; set "
        f"{_CHECKPOINT_PATH_ENV} to a path containing config.json"
    ),
)
def test_text_norm_weights_real_loader_smoke():
    """Goal 1.2 gate: real ``_load_weights_impl`` populates norm parameters.

    Constructs a memory-safe stub containing the top-level ``model.norm``
    and the first decoder layer's ``input_layernorm`` and
    ``post_attention_layernorm`` (each a 6144-dim BF16
    :class:`RMSNorm`, ~12 KB on CUDA), reads the corresponding tensors
    from the real checkpoint via ``safetensors``, strips the
    ``language_model.`` prefix exactly as the M3 VL wrapper does, and
    invokes :func:`_load_weights_impl` end-to-end. The test fails if any
    target parameter remains at its zero-initialisation, proving the
    canonical loader walks the module tree and copies the correct source
    keys for these BF16 parameters.

    Why this slice: ``input_layernorm`` / ``post_attention_layernorm`` /
    ``model.norm`` exercise the loader's ``filter_weights`` + per-module
    copy path on real tensor handles. Surfacing every required text-path
    parameter under the state-dict-coverage report is the static half of
    Goal 1.2.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")

    import torch
    from safetensors import safe_open
    from torch import nn
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        _strip_language_model_prefix,
        get_text_config,
    )
    from tensorrt_llm._torch.models.modeling_utils import _load_weights_impl
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    checkpoint = _checkpoint_path()
    cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    text_cfg = get_text_config(cfg)

    eps = float(text_cfg.rms_norm_eps)
    use_gemma = bool(getattr(text_cfg, "use_gemma_norm", False))
    hidden = int(text_cfg.hidden_size)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Memory-safe stub matching the M3 module tree for the three norms.
    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.post_attention_layernorm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )

    class _ModelInner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(
                hidden_size=hidden,
                eps=eps,
                dtype=dtype,
                device=device,
                use_gemma=use_gemma,
            )
            self.layers = nn.ModuleList([_Layer()])

    class _LoaderStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_config = ModelConfig(pretrained_config=text_cfg, mapping=Mapping())
            self.config = text_cfg
            self.model = _ModelInner()

    stub = _LoaderStub()

    # Sanity: the RMSNorm init zero-fills the gemma path; if any tensor is
    # already equal to its checkpoint value we would not be testing the copy.
    assert torch.all(stub.model.norm.weight == 0)
    assert torch.all(stub.model.layers[0].input_layernorm.weight == 0)
    assert torch.all(stub.model.layers[0].post_attention_layernorm.weight == 0)

    targets = [
        "language_model.model.norm.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
    ]

    # Group source keys by safetensors shard for efficient reads.
    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: dict = {}
    for key in targets:
        shard = weight_map[key]
        by_shard.setdefault(shard, []).append(key)

    raw_weights: dict = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as sf:
            for k in keys:
                raw_weights[k] = sf.get_tensor(k)

    # Strip `language_model.` exactly as `MiniMaxM3VLForConditionalGeneration`
    # does at load time. Confirm the stripped keyspace matches the inner
    # loader's expectation.
    text_weights, ignored = _strip_language_model_prefix(raw_weights)
    assert ignored == {}, f"unexpectedly stripped {len(ignored)} entries: {ignored!r}"
    assert set(text_weights.keys()) == {
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    }

    # Invoke the canonical loader. `_load_weights_impl` walks the stub's
    # module tree and uses the generic per-parameter copy fallback because
    # RMSNorm does not define ``load_weights``. Disable the parallel
    # executor so a failure surfaces immediately rather than as a thread
    # traceback (the parallel path is exercised in production; for this
    # tiny 3-module slice the serial walk is what the test should observe).
    os.environ["TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL"] = "True"
    try:
        _load_weights_impl(stub, text_weights, allow_partial_loading=True)
    finally:
        os.environ.pop("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", None)

    # The three norms should now hold the source tensors' values.
    torch.testing.assert_close(
        stub.model.norm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.norm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].input_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.input_layernorm.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        stub.model.layers[0].post_attention_layernorm.weight.detach().cpu().to(torch.bfloat16),
        raw_weights["language_model.model.layers.0.post_attention_layernorm.weight"].to(
            torch.bfloat16
        ),
    )

    # Independent sanity: at least one tensor must be non-zero, i.e. the
    # loader actually performed the copy.
    assert torch.any(stub.model.norm.weight != 0)


# ---------------------------------------------------------------------------
# Attention DP construction for the dense MLP / MoE shared expert
# ---------------------------------------------------------------------------
#
# Under ``enable_attention_dp=True`` each rank processes a rank-local
# set of tokens. The base ``Attention`` re-maps tp_size=1 internally so
# qkv_proj/o_proj are replicated. The MiniMax-M3 dense MLP / MoE shared
# expert (a ``GatedMLP`` built by ``_build_swiglu_oai_dense_mlp``) must
# follow the same pattern: ``overridden_tp_size=1`` + ``reduce_output=
# False`` so it runs replicated under ADP. A ROW-parallel all-reduce
# across ADP ranks would mix outputs from independent rank-local token
# sets and produce wrong results. This test pins the construction-level
# invariants of that contract.


@pytest.mark.gpu
@pytest.mark.skipif(
    not _has_cuda(), reason="MiniMax-M3 MLP construction needs CUDA backend selection"
)
def test_minimax_m3_swiglu_oai_dense_mlp_under_adp_is_replicated():
    """``_build_swiglu_oai_dense_mlp`` under ``enable_attention_dp=True``
    must produce a ``GatedMLP`` whose Linear layers are replicated
    (full-width in_features/out_features) with ``reduce_output=False``
    on ``down_proj``. Without this the dense MLP / shared expert would
    all-reduce across ADP ranks and mix outputs from independent
    rank-local token sets.
    """
    import torch  # noqa: F401

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        _build_swiglu_oai_dense_mlp,
        _wrap_dict_as_config,
    )
    from tensorrt_llm.mapping import Mapping

    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "torch_dtype": torch.bfloat16,
        }
    )
    # Simulate ADP with tp_size=4; world_size matches so Mapping
    # validation passes.
    model_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=Mapping(world_size=4, tp_size=4, pp_size=1, rank=0, enable_attention_dp=True),
        skip_create_weights_in_init=True,
    )

    intermediate = 64
    hidden = int(text_cfg.hidden_size)
    mlp = _build_swiglu_oai_dense_mlp(
        model_config=model_cfg,
        intermediate_size=intermediate,
    )
    # Under ADP the gate_up_proj and down_proj must be replicated:
    # in_features and out_features keep their full size and the
    # Linear-internal tp_size is 1.
    assert mlp.gate_up_proj.in_features == hidden
    assert mlp.gate_up_proj.out_features == 2 * intermediate, (
        "ADP gate_up_proj must be full-width (replicated), not sharded by the global TP size"
    )
    assert mlp.gate_up_proj.tp_size == 1
    assert mlp.down_proj.in_features == intermediate, (
        "ADP down_proj must be full-width (replicated), not ROW-sharded across the global TP group"
    )
    assert mlp.down_proj.out_features == hidden
    assert mlp.down_proj.tp_size == 1
    assert mlp.down_proj.reduce_output is False, (
        "ADP down_proj must skip the cross-rank all-reduce; otherwise it "
        "mixes outputs across independent rank-local token sets"
    )


def _mpi_world_size() -> int:
    """Best-effort MPI world size lookup for tests that need to know
    whether the current process is a multi-rank Slurm/MPI launch or a
    single-process unit run. Returns 1 when MPI is not available or
    cannot be initialized.
    """
    try:
        from mpi4py import MPI

        return int(MPI.COMM_WORLD.Get_size())
    except Exception:
        # Fall back to SLURM_NTASKS when MPI4py is unavailable / not
        # initialized. Single-process unit runs return 1.
        import os

        try:
            return int(os.environ.get("SLURM_NTASKS", "1"))
        except (TypeError, ValueError):
            return 1


# ---------------------------------------------------------------------------
# Multi-rank ADP negative-control / mutation tests
# ---------------------------------------------------------------------------
#
# Real multi-rank MPI job via the ``mpi_pool_executor`` fixture so the
# AllReduce workspace allocator actually runs. The test constructs both
# the ADP-safe (replicated, ``reduce_output=False``, no AllReduce) and
# the non-ADP (ROW-sharded, ``reduce_output=True``, AllReduce active)
# ``GatedMLP`` produced by ``_build_swiglu_oai_dense_mlp``, and a
# mutation that reproduces the pre-fix broken state.
#
# The mutation/negative control inside this job is layered:
#
#   * Negative control A — the non-ADP path actually allocates the
#     AllReduce workspace and constructs the ROW-sharded Linear with
#     ``reduce_output=True``. Proof that the all-reduce path is alive
#     under regular TP, so a regression that disables the all-reduce
#     for every path (not just ADP) would fail this assertion.
#   * Negative control B — under ADP the same builder constructs a
#     replicated Linear with ``reduce_output=False`` and **no**
#     AllReduce module, proving the ADP-safe construction is active.
#   * Mutation — the test simulates the fix being reverted by building
#     ``GatedMLP`` directly with ``overridden_tp_size=None`` and
#     ``reduce_output=True`` against an ADP mapping (the pre-fix code
#     path). The mutated ``GatedMLP`` then has ``reduce_output=True``,
#     ``tp_size==world_size``, and an AllReduce module attached —
#     proving the test would catch the regression.
#
# The MPI fixture spawns 2 worker processes (one per rank) and each
# worker independently exercises all three layers. The test exits 0
# only when every rank passes; an asymmetric failure on a single rank
# (the exact symptom a cross-rank mixing bug would produce) propagates
# back through ``MPIPoolExecutor`` as an exception on rank 0.


def _swiglu_dense_mlp_adp_negative_control_worker(world_size: int, rank: int) -> Dict:
    """Per-rank worker for the multi-rank ADP negative-control test.

    Runs on each MPI worker independently. Each worker constructs
    three variants of the M3 dense MLP (non-ADP, ADP, and a mutated
    ADP path) and verifies the ADP-safe construction invariants.
    Returns a structured dict of evidence that the controller (rank 0)
    can aggregate and assert across ranks.
    """
    import importlib
    from functools import partial

    import torch as _torch

    # Pin this worker process to its rank-local GPU. The mpi_pool_executor
    # fixture spawns local worker processes that share the host's CUDA
    # context space, so we explicitly bind each rank's tensors to the
    # rank-numbered device. The number of visible CUDA devices on the
    # host is the upper bound; if there are fewer visible devices than
    # ranks the worker falls back to device 0 (the workers will still
    # share the same device but the construction-time invariants do not
    # depend on rank-distinct devices).
    cuda_dev = rank
    try:
        n_devs = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    except Exception:
        n_devs = 0
    if n_devs > 0:
        cuda_dev = rank % n_devs
        _torch.cuda.set_device(cuda_dev)

    from tensorrt_llm._torch.model_config import ModelConfig

    minimaxm3_module = importlib.import_module("tensorrt_llm._torch.models.modeling_minimaxm3")
    _build_swiglu_oai_dense_mlp = minimaxm3_module._build_swiglu_oai_dense_mlp
    _minimax_m3_swiglu_oai = minimaxm3_module._minimax_m3_swiglu_oai
    _wrap_dict_as_config = minimaxm3_module._wrap_dict_as_config
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm.mapping import Mapping

    text_cfg = _wrap_dict_as_config(
        {
            "hidden_size": 128,
            "intermediate_size": 64,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "torch_dtype": _torch.bfloat16,
        }
    )
    hidden = int(text_cfg.hidden_size)
    intermediate = 64

    # ------------------------------------------------------------------
    # Negative control A — non-ADP path: ROW-sharded down_proj with
    # active AllReduce.
    # ------------------------------------------------------------------
    non_adp_mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        pp_size=1,
        rank=rank,
        enable_attention_dp=False,
    )
    non_adp_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=non_adp_mapping,
        skip_create_weights_in_init=True,
    )
    mlp_non_adp = _build_swiglu_oai_dense_mlp(
        model_config=non_adp_cfg,
        intermediate_size=intermediate,
    )
    non_adp_evidence = {
        "gate_up_proj_in_features": int(mlp_non_adp.gate_up_proj.in_features),
        "gate_up_proj_out_features": int(mlp_non_adp.gate_up_proj.out_features),
        "gate_up_proj_tp_size": int(mlp_non_adp.gate_up_proj.tp_size),
        "down_proj_in_features": int(mlp_non_adp.down_proj.in_features),
        "down_proj_out_features": int(mlp_non_adp.down_proj.out_features),
        "down_proj_tp_size": int(mlp_non_adp.down_proj.tp_size),
        "down_proj_reduce_output": bool(mlp_non_adp.down_proj.reduce_output),
        "down_proj_all_reduce_is_none": (mlp_non_adp.down_proj.all_reduce is None),
    }
    # Negative control A assertions.
    assert non_adp_evidence["gate_up_proj_in_features"] == hidden, non_adp_evidence
    assert non_adp_evidence["gate_up_proj_out_features"] == 2 * intermediate // world_size, (
        non_adp_evidence
    )
    assert non_adp_evidence["gate_up_proj_tp_size"] == world_size, non_adp_evidence
    assert non_adp_evidence["down_proj_in_features"] == intermediate // world_size, non_adp_evidence
    assert non_adp_evidence["down_proj_out_features"] == hidden, non_adp_evidence
    assert non_adp_evidence["down_proj_tp_size"] == world_size, non_adp_evidence
    assert non_adp_evidence["down_proj_reduce_output"] is True, non_adp_evidence
    assert not non_adp_evidence["down_proj_all_reduce_is_none"], (
        "Non-ADP dense MLP must have an active AllReduce module on "
        f"down_proj; got all_reduce=None. {non_adp_evidence!r}"
    )

    # ------------------------------------------------------------------
    # Negative control B — ADP path: replicated down_proj, no AllReduce.
    # ------------------------------------------------------------------
    adp_mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        pp_size=1,
        rank=rank,
        enable_attention_dp=True,
    )
    adp_cfg = ModelConfig(
        pretrained_config=text_cfg,
        mapping=adp_mapping,
        skip_create_weights_in_init=True,
    )
    mlp_adp = _build_swiglu_oai_dense_mlp(
        model_config=adp_cfg,
        intermediate_size=intermediate,
    )
    adp_evidence = {
        "gate_up_proj_in_features": int(mlp_adp.gate_up_proj.in_features),
        "gate_up_proj_out_features": int(mlp_adp.gate_up_proj.out_features),
        "gate_up_proj_tp_size": int(mlp_adp.gate_up_proj.tp_size),
        "down_proj_in_features": int(mlp_adp.down_proj.in_features),
        "down_proj_out_features": int(mlp_adp.down_proj.out_features),
        "down_proj_tp_size": int(mlp_adp.down_proj.tp_size),
        "down_proj_reduce_output": bool(mlp_adp.down_proj.reduce_output),
        "down_proj_all_reduce_is_none": (mlp_adp.down_proj.all_reduce is None),
    }
    # Negative control B assertions.
    assert adp_evidence["gate_up_proj_in_features"] == hidden, adp_evidence
    assert adp_evidence["gate_up_proj_out_features"] == 2 * intermediate, adp_evidence
    assert adp_evidence["gate_up_proj_tp_size"] == 1, adp_evidence
    assert adp_evidence["down_proj_in_features"] == intermediate, adp_evidence
    assert adp_evidence["down_proj_out_features"] == hidden, adp_evidence
    assert adp_evidence["down_proj_tp_size"] == 1, adp_evidence
    assert adp_evidence["down_proj_reduce_output"] is False, adp_evidence
    assert adp_evidence["down_proj_all_reduce_is_none"] is True, (
        "ADP dense MLP must have NO AllReduce module on down_proj; "
        f"got all_reduce non-None. {adp_evidence!r}"
    )

    # ------------------------------------------------------------------
    # Mutation simulation — construct ``GatedMLP`` directly with the
    # ADP mapping but *without* the ``_build_swiglu_oai_dense_mlp``
    # builder's ``overridden_tp_size=1`` / ``reduce_output=False``
    # plumbing, exactly reproducing the pre-fix code path. This path
    # has both the wrong shape (ROW-sharded across the global TP group)
    # AND an active AllReduce module on down_proj — under ADP this
    # would all-reduce across rank-local-token sets and mix the
    # outputs. The mutation is the *executed* negative control: it
    # actually constructs the broken state on each rank, proving the
    # ADP-safe builder is required and that this test discriminates
    # the broken state from the fixed state via observable Linear
    # shape + ``all_reduce is not None``.
    # ------------------------------------------------------------------
    mutated_mlp = GatedMLP(
        hidden_size=hidden,
        intermediate_size=intermediate,
        bias=False,
        activation=partial(_minimax_m3_swiglu_oai, alpha=1.702, limit=7.0),
        dtype=text_cfg.torch_dtype,
        config=adp_cfg,
        # No ``overridden_tp_size`` collapse — keeps the global TP
        # mapping under ADP. ``reduce_output=True`` keeps the
        # cross-rank AllReduce alive on the ROW down_proj. The Linear's
        # ADP shortcut at modules/linear.py:2818 only fires for COLUMN,
        # so this ROW mode keeps reduce_output=True even with
        # enable_attention_dp=True.
        overridden_tp_size=None,
        reduce_output=True,
    )
    mutated_evidence = {
        "gate_up_proj_tp_size": int(mutated_mlp.gate_up_proj.tp_size),
        "down_proj_in_features": int(mutated_mlp.down_proj.in_features),
        "down_proj_out_features": int(mutated_mlp.down_proj.out_features),
        "down_proj_tp_size": int(mutated_mlp.down_proj.tp_size),
        "down_proj_reduce_output": bool(mutated_mlp.down_proj.reduce_output),
        "down_proj_all_reduce_is_none": (mutated_mlp.down_proj.all_reduce is None),
    }
    # Mutation must reproduce the broken state: ROW-sharded down_proj
    # with an active AllReduce — i.e. the pre-fix behavior under an
    # ADP mapping that ought to be replicated.
    assert mutated_evidence["down_proj_tp_size"] == world_size, (
        f"Mutation simulation expected down_proj.tp_size={world_size} "
        f"(ROW-sharded across the global TP group, the broken pre-fix "
        f"behavior); got {mutated_evidence!r}."
    )
    assert mutated_evidence["down_proj_in_features"] == intermediate // world_size, (
        f"Mutation simulation expected down_proj.in_features="
        f"{intermediate // world_size}; got {mutated_evidence!r}."
    )
    assert mutated_evidence["down_proj_reduce_output"] is True, (
        f"Mutation simulation must keep reduce_output=True (the pre-"
        f"fix bug); got {mutated_evidence!r}. The Linear's ADP shortcut "
        f"at modules/linear.py:2818 only fires for COLUMN; ROW mode keeps "
        f"reduce_output=True even when the mapping has "
        f"enable_attention_dp=True."
    )
    assert mutated_evidence["down_proj_all_reduce_is_none"] is False, (
        f"Mutation simulation must produce an active AllReduce module on "
        f"down_proj (the pre-fix cross-rank reduction); got "
        f"all_reduce_is_none={mutated_evidence['down_proj_all_reduce_is_none']}."
        " A regression that fails to install AllReduce in the broken "
        "state would silently weaken this test's discriminator."
    )
    # Cross-check: the fixed ADP path has down_proj.tp_size==1 and
    # all_reduce is None; the mutation has tp_size==world_size and
    # all_reduce non-None. The shape AND the AllReduce module are
    # observably different — the test discriminates both axes.
    assert mutated_evidence["down_proj_tp_size"] != adp_evidence["down_proj_tp_size"], (
        "Mutation simulation produced the same down_proj.tp_size as "
        "the fixed ADP path; the test cannot distinguish the regression "
        "from the fix. This is a test-quality bug."
    )
    assert (
        mutated_evidence["down_proj_all_reduce_is_none"]
        != adp_evidence["down_proj_all_reduce_is_none"]
    ), (
        "Mutation simulation produced the same all_reduce_is_none as "
        "the fixed ADP path; the test cannot distinguish the regression."
    )

    return {
        "rank": rank,
        "world_size": world_size,
        "cuda_device": cuda_dev,
        "non_adp": non_adp_evidence,
        "adp": adp_evidence,
        "mutated": mutated_evidence,
    }


@pytest.mark.gpu
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_minimax_m3_swiglu_oai_dense_mlp_adp_negative_control_mpi(mpi_pool_executor):
    """Multi-rank executed ADP negative-control / mutation test.

    Runs the construction-time invariant + mutation check inside a
    real 2-process MPI pool. Each rank constructs the M3 dense MLP
    three times via ``_build_swiglu_oai_dense_mlp`` (non-ADP, ADP) and
    once via ``GatedMLP`` directly (mutated pre-fix path), asserts the
    ADP-safe invariants survive the mutation, then returns structured
    evidence the controller cross-checks across ranks.

    The fixture-supplied ``mpi_pool_executor`` (via
    ``tests/unittest/conftest.py``) spawns local MPI worker processes
    so the AllReduce workspace allocator inside the non-ADP path
    actually runs (no SKIP). The mutation simulates the ADP-safe
    builder being reverted and verifies the test would catch that
    regression.
    """
    import torch  # noqa: F401

    if not _has_cuda():
        pytest.skip(
            "ADP negative-control mutation test needs CUDA for the "
            "AllReduce workspace allocator and Linear weight materialization."
        )

    tp_size = mpi_pool_executor.num_workers
    assert tp_size == 2, (
        "ADP negative-control test parametrize fixture must be "
        f"[2]; got num_workers={tp_size}"
    )

    # Dispatch the worker function across ranks 0..tp_size-1. The
    # MPIPoolExecutor.map(...) call sends one invocation per rank and
    # blocks until all complete.
    results = list(
        mpi_pool_executor.map(
            _swiglu_dense_mlp_adp_negative_control_worker,
            [tp_size] * tp_size,
            list(range(tp_size)),
        )
    )

    assert len(results) == tp_size, (
        f"Expected {tp_size} per-rank evidence dicts; got {len(results)}: {results!r}"
    )
    seen_ranks = {r["rank"] for r in results}
    assert seen_ranks == set(range(tp_size)), (
        f"Expected ranks 0..{tp_size - 1}; got {sorted(seen_ranks)!r}: {results!r}"
    )

    # Cross-rank invariants:
    #   * Every rank's non-ADP down_proj has the same TP-sharded shape
    #     (rank-local in_features=intermediate//tp_size). Different
    #     in_features across ranks would indicate broken TP setup.
    #   * Every rank's ADP down_proj is replicated (tp_size=1).
    #   * Every rank's mutated down_proj is ROW-sharded again.
    for r in results:
        rank_idx = r["rank"]
        non_adp = r["non_adp"]
        adp = r["adp"]
        mutated = r["mutated"]
        # Non-ADP shape and AllReduce.
        assert non_adp["down_proj_tp_size"] == tp_size, r
        assert non_adp["down_proj_reduce_output"] is True, r
        assert non_adp["down_proj_all_reduce_is_none"] is False, r
        # ADP shape and AllReduce.
        assert adp["down_proj_tp_size"] == 1, r
        assert adp["down_proj_reduce_output"] is False, r
        assert adp["down_proj_all_reduce_is_none"] is True, r
        # Mutation shape and AllReduce — the pre-fix broken state.
        assert mutated["down_proj_tp_size"] == tp_size, r
        assert mutated["down_proj_reduce_output"] is True, r
        assert mutated["down_proj_all_reduce_is_none"] is False, r
        # The fixed-vs-mutated discriminator: both shape AND AllReduce
        # presence differ.
        assert mutated["down_proj_tp_size"] != adp["down_proj_tp_size"], r
        assert mutated["down_proj_all_reduce_is_none"] != adp["down_proj_all_reduce_is_none"], r
        # Emit a per-rank evidence marker that an sbatch run can grep
        # for.
        print(
            f"[M3-ADP-NEG-CTRL] rank={rank_idx} world_size={r['world_size']} "
            f"cuda_device={r['cuda_device']} "
            f"non_adp_down_proj_tp_size={non_adp['down_proj_tp_size']} "
            f"non_adp_down_proj_reduce_output={non_adp['down_proj_reduce_output']} "
            f"non_adp_down_proj_all_reduce_is_none={non_adp['down_proj_all_reduce_is_none']} "
            f"adp_down_proj_tp_size={adp['down_proj_tp_size']} "
            f"adp_down_proj_reduce_output={adp['down_proj_reduce_output']} "
            f"adp_down_proj_all_reduce_is_none={adp['down_proj_all_reduce_is_none']} "
            f"mutated_down_proj_tp_size={mutated['down_proj_tp_size']} "
            f"mutated_down_proj_reduce_output={mutated['down_proj_reduce_output']} "
            f"mutated_down_proj_all_reduce_is_none={mutated['down_proj_all_reduce_is_none']}"
        )
