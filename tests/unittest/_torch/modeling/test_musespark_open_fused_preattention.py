# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuseSparkOpen attention preprocessing, fused into one Triton kernel.

Between the QKV projection and the attention call the generic path spends, per
layer: two reshape copies to deinterleave the output gate out of the query
rows, a weightless per-head RMSNorm on Q and another on K, one elementwise
multiply for the query pre-scale, and an in-place RoPE launch. All of it is
per-layer fixed cost -- it does not amortize with batch -- and at 62 layers it
is a measurable share of a decode iteration.

``fused_qkv_rmsnorm_rope_gate`` does the whole chain in one pass and writes the
packed ``[Q, K, V]`` the attention op wants, so the concatenate afterwards goes
away too. This model needs three things the kernel did not previously offer: a
weightless norm rather than Gemma's ``1 + w``, GPT-J interleaved rotation
rather than NeoX halves, and the post-norm query multiplier.

The reference below is not a hand-written formula: it is the production
modules the fused kernel replaces, driven in the order
``Attention.preprocess_qkv`` drives them. That is what makes this a check on
the substitution rather than on someone's algebra.
"""

import ast
import inspect
import pathlib

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.kernels.fused_qk_norm_rope_gate import (
    fused_qkv_rmsnorm_rope_gate,
)
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import _fused_preattention_enabled
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import RotaryScalingType

# The checkpoint's attention geometry as one TP4 rank sees it: 128 query heads
# and 16 KV heads over four ranks, head_dim 64, theta 500000, and the source's
# query multiplier.
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 64
ROPE_THETA = 500000.0
RMS_EPS = 1e-5
SCALE_QUERY_BY = 43.7840518911 / (HEAD_DIM**0.5)
MAX_POSITIONS = 4096


def _method_source(module, class_name: str, method: str) -> str:
    """The source text of one method, read from the module file."""
    path = pathlib.Path(inspect.getsourcefile(module))
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    return ast.get_source_segment(text, sub)
    raise AssertionError(f"{class_name}.{method} not found in {path}")


def _rotary_embedding() -> RotaryEmbedding:
    rope_params = RopeParams(
        dim=HEAD_DIM,
        theta=ROPE_THETA,
        scale_type=RotaryScalingType.none,
        scale=1.0,
        max_positions=MAX_POSITIONS,
    )
    return RotaryEmbedding(rope_params, head_dim=HEAD_DIM, is_neox=False)


def _head_norm(dtype: torch.dtype) -> RMSNorm:
    return RMSNorm(hidden_size=HEAD_DIM, eps=RMS_EPS, dtype=dtype, has_weights=False).cuda()


def _unfused(qkv, q_norm, k_norm, rotary_emb, position_ids, apply_rope):
    """The chain the fused kernel replaces, in ``preprocess_qkv`` order.

    Deinterleave the gate exactly as ``Attention.preprocess_qkv`` does, run
    ``QKNormRoPEAttention.apply_qk_norm`` and MuseSparkOpen's query multiplier on
    top of it, then RoPE. ``apply_rope`` False is a NoPE layer, where the model
    passes ``skip_rope`` and the norm is the whole transform.
    """
    num_tokens = qkv.shape[0]
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q_gate, k, v = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
    orig_shape = q_gate.shape[:-1]
    q, gate = [
        t.reshape(*orig_shape, -1)
        for t in torch.chunk(q_gate.view(*orig_shape, NUM_Q_HEADS, -1), 2, dim=-1)
    ]
    q = q_norm(q.reshape(-1, HEAD_DIM)).reshape(num_tokens, q_size)
    k = k_norm(k.reshape(-1, HEAD_DIM)).reshape(num_tokens, kv_size)
    q = q * SCALE_QUERY_BY
    if apply_rope:
        q, k = rotary_emb(position_ids, [q.contiguous(), k.contiguous()])
    return torch.cat([q, k, v], dim=-1), gate.reshape(num_tokens, NUM_Q_HEADS, HEAD_DIM)


@pytest.mark.parametrize("num_tokens", (1, 3, 32, 256))
@pytest.mark.parametrize("apply_rope", (True, False), ids=("rope", "nope"))
def test_the_fused_kernel_reproduces_the_module_chain(num_tokens, apply_rope):
    """One kernel, same result as the four modules it stands in for.

    Tolerance rather than equality: both sides accumulate the RMSNorm sum of
    squares in FP32 but not necessarily in the same order, and every other
    rounding point is reproduced step for step. A mistake that matters here --
    the query scale applied on the wrong side of the norm, NeoX pairing
    substituted for GPT-J, the gate read from the wrong half -- moves the
    result far outside this band, not just past the last bit.
    """
    torch.manual_seed(20260908 + num_tokens)
    dtype = torch.bfloat16
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn((num_tokens, 2 * q_size + 2 * kv_size), dtype=dtype, device="cuda")
    position_ids = torch.randint(0, MAX_POSITIONS, (num_tokens,), dtype=torch.int32, device="cuda")

    q_norm = _head_norm(dtype)
    k_norm = _head_norm(dtype)
    rotary_emb = _rotary_embedding()

    expected_qkv, expected_gate = _unfused(
        qkv, q_norm, k_norm, rotary_emb, position_ids, apply_rope
    )
    actual_qkv, actual_gate = fused_qkv_rmsnorm_rope_gate(
        qkv,
        q_norm.weight,
        k_norm.weight,
        rotary_emb.rotary_cos_sin if apply_rope else None,
        position_ids if apply_rope else None,
        RMS_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        HEAD_DIM if apply_rope else 0,
        use_gemma_weight=False,
        q_scale=SCALE_QUERY_BY,
        is_neox=False,
    )

    torch.testing.assert_close(actual_qkv, expected_qkv, atol=0.05, rtol=0.02)
    # The gate and V are copied, not computed, so they are bit-exact.
    torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_qkv[:, q_size + kv_size :], qkv[:, 2 * q_size + kv_size :], atol=0, rtol=0
    )


def test_the_query_scale_is_not_applied_to_the_keys():
    """A multiplier on the K heads would be silent in a norm-only comparison."""
    torch.manual_seed(11)
    dtype = torch.bfloat16
    num_tokens = 8
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn((num_tokens, 2 * q_size + 2 * kv_size), dtype=dtype, device="cuda")
    q_norm = _head_norm(dtype)
    k_norm = _head_norm(dtype)

    scaled, _ = fused_qkv_rmsnorm_rope_gate(
        qkv,
        q_norm.weight,
        k_norm.weight,
        None,
        None,
        RMS_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        0,
        use_gemma_weight=False,
        q_scale=SCALE_QUERY_BY,
        is_neox=False,
    )
    plain, _ = fused_qkv_rmsnorm_rope_gate(
        qkv,
        q_norm.weight,
        k_norm.weight,
        None,
        None,
        RMS_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        0,
        use_gemma_weight=False,
        q_scale=1.0,
        is_neox=False,
    )
    torch.testing.assert_close(scaled[:, q_size:], plain[:, q_size:], atol=0, rtol=0)
    assert not torch.equal(scaled[:, :q_size], plain[:, :q_size])


def test_an_empty_batch_returns_empty_outputs():
    """The zero-token DP pad must not launch an empty grid or raise."""
    dtype = torch.bfloat16
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = torch.empty((0, 2 * q_size + 2 * kv_size), dtype=dtype, device="cuda")
    q_norm = _head_norm(dtype)
    rotary_emb = _rotary_embedding()
    out, gate = fused_qkv_rmsnorm_rope_gate(
        qkv,
        q_norm.weight,
        q_norm.weight,
        rotary_emb.rotary_cos_sin,
        torch.empty((0,), dtype=torch.int32, device="cuda"),
        RMS_EPS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        use_gemma_weight=False,
        q_scale=SCALE_QUERY_BY,
        is_neox=False,
    )
    assert out.shape == (0, q_size + 2 * kv_size)
    assert gate.shape == (0, NUM_Q_HEADS, HEAD_DIM)


def test_the_default_is_unfused(monkeypatch):
    """The fusion is opt-in.

    Two independent multi-turn canaries on this fusion's own image truncate a
    streamed answer by one character at the same serial turn, byte-identically,
    and no build without it does. Until that is traced, a caller who asks for
    nothing gets the chain that has never shown the defect.
    """
    monkeypatch.delenv("MUSESPARK_OPEN_FUSED_PREATTN", raising=False)
    assert _fused_preattention_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " on "])
def test_the_knob_selects_the_fused_treatment_leg(monkeypatch, value):
    monkeypatch.setenv("MUSESPARK_OPEN_FUSED_PREATTN", value)
    assert _fused_preattention_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "", "   ", "no", " off ", "onn"])
def test_anything_else_stays_unfused(monkeypatch, value):
    """A typo must not silently hand back the red path."""
    monkeypatch.setenv("MUSESPARK_OPEN_FUSED_PREATTN", value)
    assert _fused_preattention_enabled() is False


def test_the_attention_module_selects_the_fused_path_and_the_query_scale():
    """A source check on the wiring.

    Constructing the module needs a full model config and a mapping with a real
    TP group. What regresses cheaply is the wiring -- dropping the switch, or
    setting it without handing the kernel the query multiplier, which would
    silently drop a factor of 5.5 out of every query.
    """
    source = _method_source(modeling_musespark_open, "MuseSparkOpenAttention", "__init__")
    assert "self._fuse_qk_norm_rope_gate = _fused_preattention_enabled()" in source
    assert "self.fused_prep_q_scale = self.scale_query_by" in source
