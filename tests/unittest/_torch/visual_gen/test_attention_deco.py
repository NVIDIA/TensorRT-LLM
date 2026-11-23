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
"""Tests for the DeCo-Attention CuTe DSL backend of VisualGen.

``require_deco_kernels`` (conftest.py) skips a test on a GPU without a kernel for its V format.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.backends import (
    CuTeDSLAttention,
    UlyssesAttention,
    create_attention,
    get_visual_gen_attention_backend,
)
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import DeCoAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import deco as deco_backend
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

HEAD_DIM = 128
V_DTYPES = ("fp8", "nvfp4")
# Sequence lengths that no other forward of the process uses, one per request.
_UNSEEN_LENGTHS = itertools.count(2001)
MIN_SEQ_LEN = deco_backend.MIN_SEQ_LEN


@pytest.fixture(autouse=True)
def _kernels_serve_short_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests run the kernels on short sequences; ``test_deco_delegates_short_calls`` puts
    the backend's limit back."""
    monkeypatch.setattr(deco_backend, "MIN_SEQ_LEN", 1)


def _quant_config(v_dtype: str = "fp8") -> QuantAttentionConfig:
    return QuantAttentionConfig(qk_dtype="nvfp4", v_dtype=v_dtype, deco_attention=True)


def _backend(num_heads: int, v_dtype: str = "fp8", state: dict | None = None) -> DeCoAttention:
    """A DeCo layer from the factory, on the attention state of its model component."""
    attn = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        attention_config=AttentionConfig(
            backend="CUTEDSL", quant_attention_config=_quant_config(v_dtype)
        ),
        attention_metadata_state={} if state is None else state,
    )
    assert isinstance(attn, DeCoAttention)
    return attn


def _random_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
    seed: int = 0,
    q_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal Q and K, V with a per-channel offset, as [B, S, H, D] bfloat16.

    The scores have standard deviation ``q_scale``: 1 gives nearly uniform attention weights,
    4 gives rows that a few keys dominate.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (batch_size, seq_len, num_heads, HEAD_DIM)
    q = torch.randn(shape, generator=gen, device=device) * q_scale
    k = torch.randn(shape, generator=gen, device=device)
    v = torch.randn(shape, generator=gen, device=device)
    v = v + torch.randn((1, 1, num_heads, HEAD_DIM), generator=gen, device=device)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    """FP32 SDPA reference on [B, S, H, D] tensors."""
    out = F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2), **kwargs
    )
    return out.transpose(1, 2)


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref).norm() / ref.norm())


def _kernels(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, v_dtype: str = "fp8", **kwargs
) -> torch.Tensor:
    """The output of the kernel package on a workspace of its own; ``kwargs`` go to
    ``prepare``."""
    batch_size, seq_len, num_heads, _ = q.shape
    num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, v_dtype)
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=q.device)
    operands = deco_attention.prepare(q, k, v, workspace, v_dtype, **kwargs)
    return deco_attention.attention(operands, torch.empty_like(q))


def test_deco_selection_and_padding_capabilities() -> None:
    """CUTEDSL keeps its dense backend unless the quantization recipe asks for DeCo-Attention.
    The Ulysses wrapper answers the padding capabilities for the backend it wraps, per
    instance, and the SOL CuTe DSL backend takes a key padding mask."""
    assert not QuantAttentionConfig().deco_attention
    plain = QuantAttentionConfig(qk_dtype="nvfp4")
    assert get_visual_gen_attention_backend("CUTEDSL") is CuTeDSLAttention
    assert (
        get_visual_gen_attention_backend("CUTEDSL", quant_attention_config=plain)
        is CuTeDSLAttention
    )
    assert (
        get_visual_gen_attention_backend("CUTEDSL", quant_attention_config=_quant_config())
        is DeCoAttention
    )

    sol = get_visual_gen_attention_backend("CUTEDSL", "sol_attn")
    assert sol.__name__ == "SOLCuTeDSLAttention"
    assert sol.support_key_padding_mask() and not sol.support_valid_seq_len()
    for inner in (
        get_visual_gen_attention_backend("VANILLA"),
        get_visual_gen_attention_backend("FA4"),
        get_visual_gen_attention_backend("TRTLLM"),
        DeCoAttention,
    ):
        wrapper = UlyssesAttention.__new__(UlyssesAttention)
        wrapper.inner_backend = SimpleNamespace(
            support_key_padding_mask=inner.support_key_padding_mask,
            requires_trailing_padding=inner.requires_trailing_padding,
            support_valid_seq_len=inner.support_valid_seq_len,
        )
        assert wrapper.support_key_padding_mask() is inner.support_key_padding_mask()
        assert wrapper.requires_trailing_padding() is inner.requires_trailing_padding()
        assert wrapper.support_valid_seq_len() is inner.support_valid_seq_len()


def test_deco_layer_construction(require_deco_kernels) -> None:
    """A layer needs a recipe with a kernel on this GPU (NVFP4 PV: SM120 only) and the attention
    state of the model component. A layer whose head dimension or K/V head count the kernels
    do not take is built and runs all its calls on VANILLA."""
    device = require_deco_kernels()
    args = dict(num_heads=4, head_dim=HEAD_DIM, quant_attention_config=_quant_config())
    if "nvfp4" not in deco_backend.supported_v_dtypes():
        with pytest.raises(ValueError, match="no kernel for qk_dtype='nvfp4', v_dtype='nvfp4'"):
            DeCoAttention(**{**args, "quant_attention_config": _quant_config("nvfp4")})
    with pytest.raises(ValueError, match="attention_metadata_state"):
        create_attention(
            backend="CUTEDSL",
            layer_idx=0,
            num_heads=4,
            head_dim=HEAD_DIM,
            attention_config=AttentionConfig(
                backend="CUTEDSL", quant_attention_config=_quant_config()
            ),
        )

    for geometry in ({"head_dim": 64}, {"num_kv_heads": 2}):
        state: dict = {}
        attn = DeCoAttention(**{**args, **geometry}, attention_metadata_state=state)
        head_dim, num_kv_heads = attn.head_dim, attn.num_kv_heads

        gen = torch.Generator(device=device).manual_seed(0)
        q = torch.randn((1, 2048, 4, head_dim), generator=gen, device=device).bfloat16()
        k = torch.randn((1, 2048, num_kv_heads, head_dim), generator=gen, device=device).bfloat16()
        v = torch.randn((1, 2048, num_kv_heads, head_dim), generator=gen, device=device).bfloat16()
        out = attn.forward(q, k, v)

        ref = _sdpa(q, k, v, enable_gqa=num_kv_heads != 4)
        assert out.shape == q.shape and _relative_error(out, ref) < 2e-2
        assert "deco_workspace" not in state


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads",
    [(1, 1024, 4), (2, 333, 3), (1, 1, 1)],
    ids=["tiles", "partial_tile", "one_token"],
)
def test_deco_forward_matches_sdpa(
    require_deco_kernels, v_dtype: str, batch_size: int, seq_len: int, num_heads: int
) -> None:
    """The output follows the scores: its error against FP32 SDPA stays below half the error of
    an attention that ignores Q and K (uniform weights) on rows that a few keys dominate."""
    device = require_deco_kernels(v_dtype)
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, v_dtype)

    out = attn.forward(q, k, v)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    ref = _sdpa(q, k, v)
    uniform_weights = v.float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(out, ref) < max(0.5 * _relative_error(uniform_weights, ref), 0.02)
    # A repeated call returns the same bits.
    assert torch.equal(attn.forward(q, k, v), out)


def test_deco_layers_share_one_workspace(require_deco_kernels) -> None:
    """The layers of a model component share one workspace, which grows to the largest call and
    holds nothing a later call depends on; slices of a fused QKV projection give the result of
    contiguous inputs."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 1, 512, 4
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    fused = torch.cat([x.flatten(2) for x in (q, k, v)], dim=-1)
    q_s, k_s, v_s = (x.view(q.shape) for x in fused.chunk(3, dim=-1))
    assert not k_s.is_contiguous()

    state: dict = {}
    first, second = _backend(num_heads, state=state), _backend(num_heads, state=state)
    out = first.forward(q, k, v)
    workspace = state["deco_workspace"]

    assert torch.equal(second.forward(q_s, k_s, v_s), out)
    assert state["deco_workspace"] is workspace

    # A smaller call reuses the buffer; its leftovers do not reach the next call of this shape.
    second.forward(*_random_qkv(batch_size, 200, num_heads, device, seed=1))
    assert state["deco_workspace"] is workspace
    workspace.random_(0, 256)
    assert torch.equal(first.forward(q, k, v), out)

    # A larger call replaces it.
    first.forward(*_random_qkv(batch_size, 1024, num_heads, device))
    assert state["deco_workspace"].numel() > workspace.numel()


def test_deco_delegates_unsupported_calls(
    require_deco_kernels, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-attention, a causal mask and other dtypes run on VANILLA, unquantized; the layer
    notes its first delegated self-attention call with what keeps it from the kernels."""
    device = require_deco_kernels()
    num_heads = 2
    q, k, v = _random_qkv(1, 256, num_heads, device)
    attn = _backend(num_heads)

    # The note names the dtype of K or V when Q is bfloat16. VANILLA takes one dtype for all.
    notes = []
    monkeypatch.setattr(
        deco_backend.logger, "warning_once", lambda message, key=None: notes.append(message)
    )
    with pytest.raises(RuntimeError, match="same dtype"):
        attn.forward(q, k.float(), v.float())
    assert attn._noted_unquantized

    out = attn.forward(q, k[:, :128], v[:, :128])
    assert out.shape == q.shape
    assert _relative_error(out, _sdpa(q, k[:, :128], v[:, :128])) < 2e-2

    out = attn.forward(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert _relative_error(out, _sdpa(q, k, v, is_causal=True)) < 2e-2

    out = attn.forward(q.half(), k.half(), v.half())
    assert out.dtype == torch.float16
    assert _relative_error(out, _sdpa(q, k, v)) < 2e-2

    # A delegated call with a valid prefix attends the prefix alone; its padded rows are zero.
    out = attn.forward(q.half(), k.half(), v.half(), valid_seq_len=200)
    assert not out[:, 200:].any()
    assert _relative_error(out[:, :200], _sdpa(q[:, :200], k[:, :200], v[:, :200])) < 2e-2

    # A delegated call with a key padding mask attends the tokens; its masked rows are zero.
    key_padding_mask = torch.ones(1, 256, dtype=torch.bool, device=device)
    key_padding_mask[:, 100:156] = False
    out = attn.forward(q.half(), k.half(), v.half(), key_padding_mask=key_padding_mask)
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    assert not out[~key_padding_mask].any()
    assert _relative_error(out[key_padding_mask], ref[key_padding_mask]) < 2e-2

    # With both, the call keeps the prefix of the mask.
    out = attn.forward(
        q.half(), k.half(), v.half(), key_padding_mask=key_padding_mask, valid_seq_len=200
    )
    tokens = key_padding_mask[:, :200]
    ref = _sdpa(q[:, :200], k[:, :200], v[:, :200], attn_mask=tokens[:, None, None, :])
    assert not out[:, 200:].any() and not out[:, :200][~tokens].any()
    assert _relative_error(out[:, :200][tokens], ref[tokens]) < 2e-2

    # Cross-attention keeps all its keys: only the padded query rows are zero.
    out = attn.forward(q, k[:, :128], v[:, :128], valid_seq_len=200)
    assert not out[:, 200:].any()
    assert _relative_error(out[:, :200], _sdpa(q[:, :200], k[:, :128], v[:, :128])) < 2e-2

    assert "deco_workspace" not in attn._state
    # The layer notes once, whatever its later calls are delegated for.
    assert len(notes) == 1 and "torch.float32" in notes[0]


def test_deco_delegates_short_calls(require_deco_kernels, monkeypatch: pytest.MonkeyPatch) -> None:
    """A self-attention call over fewer than ``MIN_SEQ_LEN`` rows, such as the text sequence
    of a token refiner, runs on VANILLA without a note; a call of that many rows runs on the
    kernels."""
    device = require_deco_kernels()
    monkeypatch.setattr(deco_backend, "MIN_SEQ_LEN", MIN_SEQ_LEN)
    assert MIN_SEQ_LEN > 256
    num_heads = 2
    attn = _backend(num_heads)

    q, k, v = _random_qkv(2, 160, num_heads, device)
    key_padding_mask = torch.ones(2, 160, dtype=torch.bool, device=device)
    key_padding_mask[0, 130:] = False
    out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    assert not out[~key_padding_mask].any()
    assert _relative_error(out[key_padding_mask], ref[key_padding_mask]) < 2e-2
    assert "deco_workspace" not in attn._state and not attn._noted_unquantized

    attn.forward(*_random_qkv(1, MIN_SEQ_LEN, num_heads, device))
    assert "deco_workspace" in attn._state


def test_deco_token_tags_draw_the_basis_rows_by_modality(require_deco_kernels) -> None:
    """With the token tags of a packed sequence the preparation takes half of its rows from the
    text and audio tokens and none from the padding; the rows are drawn once per tags tensor
    and ``valid_seq_len``. With a ``valid_seq_len`` they come from the tokens of the prefix."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 1, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:100] = 1  # text
    token_tags[100:160] = 2  # audio
    token_tags[-8:] = -1  # padding
    state: dict = {}
    attn = _backend(num_heads, state=state)

    out = attn.forward(q, k, v, token_tags=token_tags)

    (plan,) = state["deco_plans"]
    assert plan.token_tags is token_tags and not plan.captured
    rows = plan.rows
    drawn = token_tags[rows.long()]
    assert (drawn >= 0).all()
    assert int((drawn > 0).sum()) == rows.numel() // 2

    # The kernels give the same output for these rows, and another one for their own draw.
    assert torch.equal(out, _kernels(q, k, v, rows=rows))
    assert not torch.equal(out, attn.forward(q, k, v))

    assert torch.equal(attn.forward(q, k, v, token_tags=token_tags), out)
    assert len(state["deco_plans"]) == 1
    with pytest.raises(ValueError, match="token tags"):
        attn.forward(q, k, v, token_tags=token_tags[:-1])

    # A group short of its half gives all its tokens and the other group fills the draw, so a
    # mostly-text sequence still draws the full count; a sequence of padding draws nothing. The
    # preparation takes a draw of fewer rows than twice its rank and rejects an empty one.
    mostly_text = torch.ones(seq_len, dtype=torch.int32, device=device)
    mostly_text[:5] = 0
    rows = deco_attention.sample_rows(seq_len, device, mostly_text)
    assert rows.numel() == deco_attention.inputs.sample_count(seq_len)
    assert int((mostly_text[rows.long()] == 0).sum()) == 5
    padding = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    assert deco_attention.sample_rows(seq_len, device, padding).numel() == 0
    lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    for short, valid_seq_lens in ((rows[:16], None), (rows[:16][None], lens)):
        out_short = _kernels(q, k, v, rows=short, valid_seq_lens=valid_seq_lens)
        assert torch.isfinite(out_short).all()
    with pytest.raises(ValueError, match="at least one token"):
        _kernels(q, k, v, rows=rows[:0])

    # Tags with fewer tokens than the draw takes give them all.
    few_tokens = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    few_tokens[:50] = 1
    assert torch.isfinite(attn.forward(q, k, v, token_tags=few_tokens)).all()
    rows = state["deco_plans"][0].rows
    assert sorted(rows.tolist()) == list(range(50))

    # A row outside the sequence stands for its nearest token.
    beyond, last = plan.rows.clone(), plan.rows.clone()
    beyond[0], last[0] = seq_len + 5, seq_len - 1
    assert torch.equal(_kernels(q, k, v, rows=beyond), _kernels(q, k, v, rows=last))

    # With ``valid_seq_len`` and no mask the rows come from the tokens of the prefix, and
    # another length gets its own draw.
    num_plans = len(state["deco_plans"])
    text_last = torch.zeros(seq_len, dtype=torch.int32, device=device)
    text_last[3000:] = 1  # text at the end; the tags mark no padding
    for valid_seq_len in (3500, 3100):
        out = attn.forward(q, k, v, token_tags=text_last, valid_seq_len=valid_seq_len)
        assert not out[:, valid_seq_len:].any()
        rows = state["deco_plans"][0].rows
        assert rows.dim() == 1 and int(rows.max()) < valid_seq_len
        assert int((rows >= 3000).sum()) == min(rows.numel() // 2, valid_seq_len - 3000)
    assert len(state["deco_plans"]) == num_plans + 2


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_valid_seq_len_attends_the_prefix(require_deco_kernels, v_dtype: str) -> None:
    """With ``valid_seq_len`` the rows after the prefix are padding: the prefix attends its own
    keys only, whatever the padding holds, and the padded rows come back zero. A mask whose
    tokens are that prefix gives the same bits, a mask without padding those of the call
    without a mask, and a mask with ``valid_seq_len`` those of the rows that both keep."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, valid_seq_len, num_heads = 2, 1024, 700, 3
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    state: dict = {}
    attn = _backend(num_heads, v_dtype, state=state)

    # A prefix that covers the sequence is the call without a length.
    assert torch.equal(attn.forward(q, k, v, valid_seq_len=seq_len), attn.forward(q, k, v))

    expected = attn.forward(q, k, v, valid_seq_len=valid_seq_len)

    assert expected.shape == q.shape and expected.dtype == torch.bfloat16
    assert not expected[:, valid_seq_len:].any()
    prefix = expected[:, :valid_seq_len]
    ref = _sdpa(q[:, :valid_seq_len], k[:, :valid_seq_len], v[:, :valid_seq_len])
    uniform_weights = v[:, :valid_seq_len].float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(prefix, ref) < max(0.5 * _relative_error(uniform_weights, ref), 0.02)

    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
    prefix_mask = positions < valid_seq_len
    assert torch.equal(attn.forward(q, k, v, key_padding_mask=prefix_mask), expected)
    full = torch.ones_like(prefix_mask)
    assert torch.equal(attn.forward(q, k, v, key_padding_mask=full), attn.forward(q, k, v))
    assert torch.equal(
        attn.forward(q, k, v, key_padding_mask=positions < 800, valid_seq_len=valid_seq_len),
        expected,
    )
    # Newest first: a prefix mask with a valid_seq_len, a mask without padding, a prefix mask.
    plans = state["deco_plans"]
    assert [plan.token_counts is None for plan in plans] == [False, True, False]
    assert all(plan.order is None for plan in plans)

    # The mask is read once per tensor and content: an in-place change is seen.
    full[:, valid_seq_len:] = False
    assert torch.equal(attn.forward(q, k, v, key_padding_mask=full), expected)
    assert state["deco_plans"][0].token_counts.tolist() == [valid_seq_len] * batch_size

    # The compaction of a prefix is the prefix: a hole in sample 0 changes that sample only.
    holes = prefix_mask.clone()
    holes[0, 100:200] = False
    out = attn.forward(q, k, v, key_padding_mask=holes)
    assert torch.equal(out[1], expected[1]) and not torch.equal(out[0], expected[0])

    for padding in (0.0, 1e30, float("nan")):
        for tensor in (q, k, v):
            tensor[:, valid_seq_len:] = padding
        out = attn.forward(q, k, v, valid_seq_len=valid_seq_len)
        assert torch.equal(out[:, :valid_seq_len], prefix)
        assert not out[:, valid_seq_len:].any()

    with pytest.raises(ValueError, match="valid_seq_len"):
        attn.forward(q, k, v, valid_seq_len=seq_len + 1)

    # A batch element without valid tokens comes back zero, whatever its padding holds.
    lens = torch.tensor([0, valid_seq_len], dtype=torch.int32, device=device)
    out = _kernels(q, k, v, v_dtype, valid_seq_lens=lens)
    assert not out[0].any()
    assert torch.equal(out[1, :valid_seq_len], prefix[1])
    assert not out[1, valid_seq_len:].any()


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_key_padding_mask_attends_the_tokens(require_deco_kernels, v_dtype: str) -> None:
    """With a ``key_padding_mask`` every sample attends its own tokens only, wherever they sit
    and whatever the masked rows hold, and the masked rows come back zero. A sample whose mask
    excludes every row comes back zero, on the kernels and on a delegated call, with and
    without token tags, next to a sample that is served."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 1024, 3
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    state: dict = {}
    attn = _backend(num_heads, v_dtype, state=state)

    # Sample 0 has holes between its tokens; sample 1 has another token count, as a prefix.
    gen = torch.Generator(device=device).manual_seed(1)
    key_padding_mask = torch.rand((batch_size, seq_len), generator=gen, device=device) < 0.7
    key_padding_mask[1] = torch.arange(seq_len, device=device) < 600

    out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert not out[~key_padding_mask].any()
    tokens = out[key_padding_mask].clone()
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    for sample in range(batch_size):
        valid = key_padding_mask[sample]
        uniform_weights = v[sample, valid].float().mean(dim=0, keepdim=True)
        uniform_error = _relative_error(
            uniform_weights.expand_as(ref[sample, valid]), ref[sample, valid]
        )
        error = _relative_error(out[sample, valid], ref[sample, valid])
        assert error < max(0.5 * uniform_error, 0.02)
    (plan,) = state["deco_plans"]
    assert plan.order is not None

    for padding in (0.0, 1e30, float("nan")):
        for tensor in (q, k, v):
            tensor[~key_padding_mask] = padding
        out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)
        assert torch.equal(out[key_padding_mask], tokens)
        assert not out[~key_padding_mask].any()
    assert len(state["deco_plans"]) == 1

    with pytest.raises(ValueError, match="key_padding_mask"):
        attn.forward(q, k, v, key_padding_mask=key_padding_mask[:, :-1])

    # Sample 0 without tokens, sample 1 with a prefix of 900.
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    empty_sample = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    empty_sample[0] = False
    empty_sample[1, 900:] = False
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[800:] = 1
    ref = _sdpa(q[1:], k[1:], v[1:], attn_mask=empty_sample[1:, None, None, :])[0, :900]

    for kwargs in ({}, {"token_tags": token_tags}):
        out = attn.forward(q, k, v, key_padding_mask=empty_sample, **kwargs)
        assert not out[0].any() and not out[1, 900:].any()
        assert torch.isfinite(out).all()
        assert _relative_error(out[1, :900], ref) < 0.2
    (rows,) = [plan.rows for plan in state["deco_plans"] if plan.token_tags is token_tags]
    assert rows.shape[0] == batch_size and (rows[1] < 900).all()
    # The draw of sample 1 keeps its rows, half of them text, next to the empty sample.
    assert int((rows[1] >= 800).sum()) == rows.shape[1] // 2

    out = attn.forward(q.half(), k.half(), v.half(), key_padding_mask=empty_sample)
    assert not out[0].any() and not out[1, 900:].any()
    assert _relative_error(out[1, :900], ref) < 2e-2


def test_deco_token_tags_with_a_key_padding_mask(require_deco_kernels) -> None:
    """With token tags and a mask whose tokens are not a prefix, the basis rows are the
    positions of the drawn tokens after the compaction. When the samples of a batch have
    padding on positions with a modality, such as text of two lengths, every sample gets the
    draw of its own tokens, half of it from its text, once per tags tensor and mask."""
    device = require_deco_kernels()
    batch_size, seq_len, num_heads = 2, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    state: dict = {}
    attn = _backend(num_heads, state=state)

    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:100] = 1  # text
    token_tags[100:160] = 2  # audio
    token_tags[-8:] = -1  # padding
    key_padding_mask = (token_tags >= 0).expand(batch_size, -1).clone()
    key_padding_mask[0, 40:100] = False  # the text of sample 0 is shorter

    out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask)

    assert not out[~key_padding_mask].any()
    rows = state["deco_plans"][0].rows
    order = torch.argsort(~key_padding_mask, dim=1, stable=True)
    index = order[:, :, None, None].expand_as(q)
    position = torch.argsort(order, dim=1)
    compacted = _kernels(
        q.gather(1, index),
        k.gather(1, index),
        v.gather(1, index),
        rows=position.gather(1, rows.long()).to(torch.int32).contiguous(),
        valid_seq_lens=key_padding_mask.sum(dim=1, dtype=torch.int32),
    )
    assert torch.equal(out, torch.empty_like(q).scatter_(1, index, compacted))

    # Video, then text with the padding of every sample last: the tags mark no padding.
    num_video = 3600
    token_tags = torch.ones(seq_len, dtype=torch.int32, device=device)
    token_tags[:num_video] = 0
    lengths = torch.tensor([num_video + 400, num_video + 20], device=device)
    key_padding_mask = torch.arange(seq_len, device=device) < lengths[:, None]

    out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask)

    plan = state["deco_plans"][0]
    assert plan.token_tags is token_tags and plan.key_padding_mask is key_padding_mask
    rows = plan.rows
    assert rows.shape[0] == batch_size
    assert (rows < lengths[:, None]).all()
    text_rows = (rows >= num_video).sum(dim=1)
    assert text_rows.tolist() == [rows.shape[1] // 2, 20]
    assert torch.equal(out, _kernels(q, k, v, rows=rows, valid_seq_lens=lengths.to(torch.int32)))

    assert torch.equal(
        attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=key_padding_mask), out
    )
    assert len(state["deco_plans"]) == 2


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_cuda_graph_with_a_key_padding_mask(require_deco_kernels, v_dtype: str) -> None:
    """A captured forward with a mask computes the counts and the order of the tokens inside
    the graph: a replay follows the content of the mask tensor of the capture. A capture takes
    the basis rows that an eager forward drew for the token tags, also when its mask tensor is
    another object than the mask of that forward."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 2, 512, 2
    attn = _backend(num_heads, v_dtype)
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    gen = torch.Generator(device=device).manual_seed(1)
    key_padding_mask = torch.rand((batch_size, seq_len), generator=gen, device=device) < 0.7

    attn.forward(q, k, v, key_padding_mask=key_padding_mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)

    new_q, new_k, new_v = _random_qkv(batch_size, seq_len, num_heads, device, seed=1)
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
    for new_mask in (
        torch.rand((batch_size, seq_len), generator=gen, device=device) < 0.5,
        positions < 300,
        torch.ones_like(key_padding_mask),
    ):
        expected = attn.forward(new_q, new_k, new_v, key_padding_mask=new_mask).clone()
        q.copy_(new_q)
        k.copy_(new_k)
        v.copy_(new_v)
        key_padding_mask.copy_(new_mask)
        graph_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out, expected)

    seq_len = 2048
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[1800:] = 1
    lengths = torch.tensor([[2000], [1850]], device=device)
    warm_mask = torch.arange(seq_len, device=device) < lengths
    expected = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=warm_mask).clone()

    capture_mask = warm_mask.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v, token_tags=token_tags, key_padding_mask=capture_mask)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, expected)

    with pytest.raises(RuntimeError, match="eagerly once before capturing"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            attn.forward(q, k, v, token_tags=token_tags.clone(), key_padding_mask=capture_mask)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_cuda_graph(require_deco_kernels, v_dtype: str) -> None:
    """A forward is captured after an eager forward of its shape and replays onto new inputs,
    also after eager forwards of other shapes reused and regrew the component's workspace and
    other requests' token tags passed through the state, which keeps the captured rows. A
    sequence length or token tags first met under capture are rejected with a clear error."""
    device = require_deco_kernels(v_dtype)
    batch_size, seq_len, num_heads = 1, 384, 2
    state: dict = {}
    attn = _backend(num_heads, v_dtype, state=state)
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:64] = 1

    attn.forward(q, k, v, token_tags=token_tags)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v, token_tags=token_tags)
    (plan,) = state["deco_plans"]
    assert plan.captured

    for other_len in (128, 1024, 256, 512, 640, 768):
        other_tags = torch.zeros(other_len, dtype=torch.int32, device=device)
        other_tags[:64] = 1
        attn.forward(*_random_qkv(batch_size, other_len, num_heads, device), token_tags=other_tags)
    assert plan in state["deco_plans"] and len(state["deco_plans"]) == 5
    new_q, new_k, new_v = _random_qkv(batch_size, seq_len, num_heads, device, seed=1)
    expected = attn.forward(new_q, new_k, new_v, token_tags=token_tags).clone()
    q.copy_(new_q)
    k.copy_(new_k)
    v.copy_(new_v)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, expected)

    # The preparation builds its inputs for a sequence length on first use, outside capture.
    unseen_len = next(_UNSEEN_LENGTHS)
    unseen = _random_qkv(batch_size, unseen_len, num_heads, device)
    num_bytes = deco_attention.workspace_size(batch_size, unseen_len, num_heads, device, v_dtype)
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    with pytest.raises(RuntimeError, match="eagerly once before capturing"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            deco_attention.prepare(*unseen, workspace, v_dtype)
    # So does the backend for the draw of token tags it has not seen.
    with pytest.raises(RuntimeError, match="eagerly once before capturing"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            attn.forward(q, k, v, token_tags=token_tags.clone())
