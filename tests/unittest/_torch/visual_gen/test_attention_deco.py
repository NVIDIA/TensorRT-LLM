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
"""Tests for the DeCo-Attention CuTe DSL backend of VisualGen."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.backends import (
    CuTeDSLAttention,
    DeCoAttention,
    create_attention,
    get_visual_gen_attention_backend,
)
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

HEAD_DIM = 128
V_DTYPES = ("fp8", "nvfp4")
# Sequence lengths that no other forward of the process uses, one per request.
_UNSEEN_LENGTHS = itertools.count(2001)


def _require_deco_gpu(v_dtype: str = "fp8") -> torch.device:
    """Skip unless the GPU has a DeCo kernel with this V format (NVFP4 PV: SM120 only)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DeCo-Attention kernels.")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3), (10, 7), (12, 0)):
        pytest.skip("DeCo-Attention kernels need an SM100, SM103, SM107 or SM120 GPU.")
    if v_dtype == "nvfp4" and capability != (12, 0):
        pytest.skip("DeCo-Attention with NVFP4 PV needs an SM120 GPU.")
    return torch.device("cuda:0")


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


def test_deco_attention_is_opt_in() -> None:
    """CUTEDSL keeps its dense backend unless the quantization recipe asks for DeCo-Attention."""
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


def test_deco_rejects_what_its_kernels_do_not_take() -> None:
    """The kernels need head dimension 128, as many K/V heads as Q heads, a recipe with a kernel
    on this GPU (NVFP4 PV: SM120 only) and the attention state of the model component."""
    _require_deco_gpu()
    args = dict(num_heads=4, head_dim=HEAD_DIM, quant_attention_config=_quant_config())
    with pytest.raises(ValueError, match="head_dim 128"):
        DeCoAttention(**{**args, "head_dim": 64})
    with pytest.raises(ValueError, match="as many K/V heads as Q heads"):
        DeCoAttention(**{**args, "num_kv_heads": 2})
    if torch.cuda.get_device_capability() != (12, 0):
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


@pytest.mark.parametrize("v_dtype", V_DTYPES)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads",
    [(1, 1024, 4), (2, 333, 3), (1, 1, 1)],
    ids=["tiles", "partial_tile", "one_token"],
)
def test_deco_forward_matches_sdpa(
    v_dtype: str, batch_size: int, seq_len: int, num_heads: int
) -> None:
    """The output follows the scores: its error against FP32 SDPA stays below half the error of
    an attention that ignores Q and K (uniform weights) on rows that a few keys dominate."""
    device = _require_deco_gpu(v_dtype)
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


def test_deco_layers_share_one_workspace() -> None:
    """The layers of a model component share one workspace, which grows to the largest call and
    holds nothing a later call depends on; slices of a fused QKV projection give the result of
    contiguous inputs."""
    device = _require_deco_gpu()
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


def test_deco_delegates_unsupported_calls() -> None:
    """Cross-attention, masks and other dtypes run on VANILLA, unquantized."""
    device = _require_deco_gpu()
    num_heads = 2
    q, k, v = _random_qkv(1, 256, num_heads, device)
    attn = _backend(num_heads)

    out = attn.forward(q, k[:, :128], v[:, :128])
    assert out.shape == q.shape
    assert _relative_error(out, _sdpa(q, k[:, :128], v[:, :128])) < 2e-2

    out = attn.forward(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert _relative_error(out, _sdpa(q, k, v, is_causal=True)) < 2e-2

    key_padding_mask = torch.ones(1, 256, dtype=torch.bool, device=device)
    key_padding_mask[:, 200:] = False
    out = attn.forward(q, k, v, key_padding_mask=key_padding_mask)
    ref = _sdpa(q, k, v, attn_mask=key_padding_mask[:, None, None, :])
    assert _relative_error(out, ref) < 2e-2

    out = attn.forward(q.half(), k.half(), v.half())
    assert out.dtype == torch.float16
    assert _relative_error(out, _sdpa(q, k, v)) < 2e-2

    # A delegated call with a valid prefix attends the prefix alone; its padded rows are zero.
    out = attn.forward(q.half(), k.half(), v.half(), valid_seq_len=200)
    assert not out[:, 200:].any()
    assert _relative_error(out[:, :200], _sdpa(q[:, :200], k[:, :200], v[:, :200])) < 2e-2

    assert "deco_workspace" not in attn._state
    # The causal call has the shape of a self-attention call: the layer notes it once.
    assert attn._noted_unquantized


def test_deco_token_tags_draw_the_basis_rows_by_modality() -> None:
    """With the token tags of a packed sequence the preparation takes half of its rows from the
    text and audio tokens and none from the padding; the rows are drawn once per tags tensor."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention

    device = _require_deco_gpu()
    batch_size, seq_len, num_heads = 1, 4096, 2
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    token_tags = torch.zeros(seq_len, dtype=torch.int32, device=device)
    token_tags[:100] = 1  # text
    token_tags[100:160] = 2  # audio
    token_tags[-8:] = -1  # padding
    state: dict = {}
    attn = _backend(num_heads, state=state)

    out = attn.forward(q, k, v, token_tags=token_tags)

    ((kept_tags, rows),) = state["deco_tagged_rows"]
    assert kept_tags is token_tags
    drawn = token_tags[rows.long()]
    assert (drawn >= 0).all()
    assert int((drawn > 0).sum()) == rows.numel() // 2

    # The kernels give the same output for these rows, and another one for their own draw.
    num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, device, "fp8")
    workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    operands = deco_attention.prepare(q, k, v, workspace, "fp8", rows=rows)
    assert torch.equal(out, deco_attention.attention(operands, torch.empty_like(q)))
    assert not torch.equal(out, attn.forward(q, k, v))

    assert torch.equal(attn.forward(q, k, v, token_tags=token_tags), out)
    assert len(state["deco_tagged_rows"]) == 1
    with pytest.raises(ValueError, match="token tags"):
        attn.forward(q, k, v, token_tags=token_tags[:-1])

    # A group short of its half gives all its tokens and the other group fills the draw, so a
    # mostly-text sequence still draws the full count; a sequence of padding draws nothing, which
    # the preparation rejects instead of running on no rows.
    mostly_text = torch.ones(seq_len, dtype=torch.int32, device=device)
    mostly_text[:5] = 0
    rows = deco_attention.sample_rows(seq_len, device, mostly_text)
    assert rows.numel() == deco_attention.inputs.sample_count(seq_len)
    assert int((mostly_text[rows.long()] == 0).sum()) == 5
    padding = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    assert deco_attention.sample_rows(seq_len, device, padding).numel() == 0
    with pytest.raises(ValueError, match="at least one token"):
        deco_attention.prepare(
            q, k, v, workspace, "fp8", rows=torch.empty(0, dtype=torch.int32, device=device)
        )


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_valid_seq_len_attends_the_prefix(v_dtype: str) -> None:
    """With ``valid_seq_len`` the rows after the prefix are padding: the prefix attends its own
    keys only, whatever the padding holds, and the padded rows come back zero."""
    device = _require_deco_gpu(v_dtype)
    batch_size, seq_len, valid_seq_len, num_heads = 2, 1024, 700, 3
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device, q_scale=4.0)
    attn = _backend(num_heads, v_dtype)

    # A prefix that covers the sequence is the call without a length.
    assert torch.equal(attn.forward(q, k, v, valid_seq_len=seq_len), attn.forward(q, k, v))

    out = attn.forward(q, k, v, valid_seq_len=valid_seq_len)

    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert not out[:, valid_seq_len:].any()
    prefix = out[:, :valid_seq_len].clone()
    ref = _sdpa(q[:, :valid_seq_len], k[:, :valid_seq_len], v[:, :valid_seq_len])
    uniform_weights = v[:, :valid_seq_len].float().mean(dim=1, keepdim=True).expand_as(ref)
    assert _relative_error(prefix, ref) < max(0.5 * _relative_error(uniform_weights, ref), 0.02)

    for padding in (0.0, 1e30, float("nan")):
        for tensor in (q, k, v):
            tensor[:, valid_seq_len:] = padding
        out = attn.forward(q, k, v, valid_seq_len=valid_seq_len)
        assert torch.equal(out[:, :valid_seq_len], prefix)
        assert not out[:, valid_seq_len:].any()

    with pytest.raises(ValueError, match="valid_seq_len"):
        attn.forward(q, k, v, valid_seq_len=seq_len + 1)


@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_deco_cuda_graph(v_dtype: str) -> None:
    """A forward is captured after an eager forward of its shape and replays onto new inputs,
    also after eager forwards of other shapes reused and regrew the component's workspace. A
    sequence length first met under capture is rejected with a clear error."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention

    device = _require_deco_gpu(v_dtype)
    batch_size, seq_len, num_heads = 1, 384, 2
    attn = _backend(num_heads, v_dtype)
    q, k, v = _random_qkv(batch_size, seq_len, num_heads, device)

    attn.forward(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = attn.forward(q, k, v)

    for other_len in (128, 1024):
        attn.forward(*_random_qkv(batch_size, other_len, num_heads, device))
    new_q, new_k, new_v = _random_qkv(batch_size, seq_len, num_heads, device, seed=1)
    expected = attn.forward(new_q, new_k, new_v).clone()
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
