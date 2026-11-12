# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-causal sliding-window block attention for DSpark drafters.

A DSpark block query ``j`` at position ``C + j`` (``C`` context tokens) attends
every key within ``W - 1`` positions on both sides: context keys from
``C + j - W + 1`` and every block key. The TRTLLM backend serves that mask as
the causal windowed decode plus the block keys after each query, merged by
log-sum-exp. These tests check the mask resolution (CPU), the merge against a
dense oracle (CPU), and the full trtllm-gen path against a dense fp32 oracle
over a paged cache (SM100/SM103 only).
"""

import math
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import (
    dflash_merge_future_block_attention,
    dflash_trtllm_folded_block_attention,
    dflash_trtllm_windowed_block_attention,
)
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM
from tensorrt_llm._torch.speculative.dflash_attention import (
    DFLASH_FOLDED_DECODE_LOWERING,
    get_dflash_folded_decode_chunk_size,
    validate_dflash_trtllm_gen_runtime,
)

# ---------------------------------------------------------------------------
# Dense oracle
# ---------------------------------------------------------------------------


def _oracle(query, keys, values, ctx_lens, window_left, window_right, causal):
    """Dense fp32 block attention with the trained mask.

    query: [B, blk, Hq, D]; keys/values: per request [C_b + blk, Hkv, D].
    Returns ([B * blk, Hq, D] output, [B * blk, Hq] natural-log lse).
    """
    B, blk, num_heads, head_dim = query.shape
    outs, lses = [], []
    for b in range(B):
        k = keys[b].float()
        v = values[b].float()
        num_kv_heads = k.shape[1]
        group = num_heads // num_kv_heads
        q = query[b].float()  # [blk, Hq, D]
        k_rep = k.repeat_interleave(group, dim=1)  # [L, Hq, D]
        v_rep = v.repeat_interleave(group, dim=1)
        logits = torch.einsum("jhd,lhd->hjl", q, k_rep) * head_dim**-0.5
        pos_q = ctx_lens[b] + torch.arange(blk, device=q.device).view(-1, 1)
        pos_k = torch.arange(k.shape[0], device=q.device).view(1, -1)
        allowed = torch.ones_like(pos_q - pos_k, dtype=torch.bool)
        if window_left >= 0:
            allowed &= pos_k >= pos_q - window_left
        if causal:
            allowed &= pos_k <= pos_q
        elif window_right >= 0:
            allowed &= pos_k <= pos_q + window_right
        logits = logits.masked_fill(~allowed, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)  # [Hq, blk]
        out = torch.einsum("hjl,lhd->jhd", torch.softmax(logits, dim=-1), v_rep)
        outs.append(out)
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs), torch.cat(lses)


# ---------------------------------------------------------------------------
# Merge (CPU): causal oracle + merge == non-causal oracle
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("block_size", [1, 2, 4, 7])
@pytest.mark.parametrize("window", [3, 8, 64])
@pytest.mark.parametrize("heads", [(4, 1), (8, 2)])
def test_merge_matches_the_non_causal_oracle(block_size, window, heads):
    torch.manual_seed(0)
    num_heads, num_kv_heads = heads
    head_dim = 16
    ctx_lens = [0, 1, window - block_size, window - 1, window, window + 5, 3 * window + 2]
    ctx_lens = [c for c in ctx_lens if c >= 0]
    B = len(ctx_lens)
    query = torch.randn(B, block_size, num_heads, head_dim)
    keys = [torch.randn(c + block_size, num_kv_heads, head_dim) for c in ctx_lens]
    values = [torch.randn(c + block_size, num_kv_heads, head_dim) for c in ctx_lens]
    block_key = torch.stack([k[-block_size:] for k in keys])
    block_value = torch.stack([v[-block_size:] for v in values])
    w = window - 1

    out_causal, lse_causal = _oracle(query, keys, values, ctx_lens, w, -1, causal=True)
    expected, _ = _oracle(query, keys, values, ctx_lens, w, w, causal=False)
    merged = dflash_merge_future_block_attention(
        out_causal, lse_causal, query, block_key, block_value, head_dim**-0.5, w
    )
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)
    if block_size > 1:
        # The causal output alone is a different attention: the merge matters.
        assert not torch.allclose(out_causal, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Mask resolution (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("causal", [False, True])
def test_windowed_decode_and_merge_use_cache_precision(cache_dtype, causal):
    """Check the decode boundary and future-block merge against rounded cache values."""
    torch.manual_seed(17)
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 3, 4, 2, 8
    query = torch.randn(batch, block_size, num_heads, head_dim, dtype=torch.bfloat16)
    block_key = torch.randn(batch, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16)
    block_value = torch.randn_like(block_key)
    keys = block_key.to(cache_dtype).float().unbind()
    values = block_value.to(cache_dtype).float().unbind()
    context_lengths = [0] * batch
    causal_out, causal_lse = _oracle(query, keys, values, context_lengths, 1, 0, causal=True)
    expected, _ = _oracle(query, keys, values, context_lengths, 1, 1, causal=causal)
    cache = torch.empty(1, 2, num_kv_heads, 4, head_dim, dtype=cache_dtype)
    out = torch.empty_like(causal_out)

    def decode(**kwargs):
        assert kwargs["query"].dtype == query.dtype
        assert kwargs["kv_cache"][0].dtype == cache_dtype
        torch.testing.assert_close(
            kwargs["query"].float(), query.float().reshape_as(kwargs["query"])
        )
        kwargs["out"].copy_(causal_out)
        if kwargs["return_lse"]:
            return kwargs["out"], causal_lse / math.log(2.0)
        return kwargs["out"]

    dflash_trtllm_windowed_block_attention(
        SimpleNamespace(batch_decode_with_kv_cache=decode),
        query,
        block_key,
        block_value,
        cache,
        torch.zeros(batch, 1, dtype=torch.int32),
        torch.full((batch,), block_size, dtype=torch.int32),
        4,
        causal,
        (1, 0 if causal else 1),
        torch.empty(0),
        torch.empty(0),
        out,
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def _wrapper(cls, config, layer_windows=(), backend="TRTLLM"):
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = config
    wrapper._is_dflash2 = False
    wrapper._uses_lilicorr = False
    wrapper._sliding_layers_causal = False
    wrapper.dflash_attention_backend = backend
    wrapper._layer_windows = list(layer_windows)
    return wrapper


def _minimax_like_config(num_layers=5, window=4096):
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=["sliding_attention"] * num_layers,
        sliding_window=window,
    )


@pytest.mark.cpu_only
def test_dspark_windowed_layers_are_non_causal_and_two_sided():
    """The MiniMax-M3.1 drafter: five sliding layers, window 4096, plus the
    use_swa window the config flattening adds."""
    config = _minimax_like_config()
    wrapper = _wrapper(GQADSparkForCausalLM, config, layer_windows=[(4095, 4095)] * 5)
    for layer_idx in range(5):
        assert wrapper._get_attention_mask_args(layer_idx) == (False, (4095, 4095))
        assert wrapper._resolve_block_attention(layer_idx) == (False, (4095, 4095))
    assert wrapper.validate_block_attention_windows() == ["causal decode + future-block merge"] * 5


@pytest.mark.cpu_only
def test_dspark_unwindowed_layers_stay_non_causal():
    config = SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"] * 2)
    wrapper = _wrapper(GQADSparkForCausalLM, config)
    assert wrapper._resolve_block_attention(0) == (False, (-1, -1))
    assert wrapper.validate_block_attention_windows() == ["non-causal folded decode"] * 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"is_causal": True}, (True, (511, 0))),
        ({"sliding_window_non_causal": False}, (True, (511, 0))),
        ({"is_causal": False}, (False, (511, 511))),
        ({"sliding_window_non_causal": True}, (False, (511, 511))),
        ({"is_causal": False, "sliding_window_non_causal": False}, (False, (511, 511))),
    ],
)
def test_dspark_honors_explicit_window_causality_metadata(metadata, expected):
    """A config that states its window is causal keeps the DFlash resolution
    (causal windowed decode); ``is_causal`` outranks the speculators spelling."""
    config = _minimax_like_config(num_layers=2, window=512)
    for key, value in metadata.items():
        setattr(config, key, value)
    wrapper = _wrapper(GQADSparkForCausalLM, config)
    assert wrapper._get_attention_mask_args(0) == expected
    assert wrapper._resolve_block_attention(0) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("metadata", [{"is_causal": True}, {"sliding_window_non_causal": False}])
def test_dspark_causal_metadata_applies_to_a_use_swa_only_window(metadata):
    """A window that comes only from ``dflash_config.use_swa`` (no top-level
    ``sliding_window``) still gets the causal decode when the config says so."""
    config = SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"] * 2, **metadata)
    wrapper = _wrapper(GQADSparkForCausalLM, config, layer_windows=[(511, 511)] * 2)
    assert wrapper._get_attention_mask_args(0) == (True, (511, 0))
    assert wrapper._resolve_block_attention(0)[0] is True
    assert wrapper.validate_block_attention_windows() == ["causal decode"] * 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("head_dim", [72, 80, 96])
def test_decode_layers_are_not_held_to_context_kernel_head_sizes(head_dim: int) -> None:
    """Windowed and folded non-causal layers both use the decode kernel."""
    windowed = _wrapper(GQADSparkForCausalLM, _minimax_like_config(num_layers=2, window=512))
    lowerings = windowed.validate_block_attention_windows()
    assert DFLASH_FOLDED_DECODE_LOWERING not in lowerings
    unwindowed = _wrapper(
        GQADSparkForCausalLM,
        SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"] * 2),
    )
    assert DFLASH_FOLDED_DECODE_LOWERING in unwindowed.validate_block_attention_windows()
    shape = dict(
        dtype=torch.bfloat16, num_heads=16, num_kv_heads=2, head_dim=head_dim, tokens_per_block=32
    )
    with patch("tensorrt_llm._torch.speculative.dflash_attention.get_dflash_trtllm_gen_ops"):
        validate_dflash_trtllm_gen_runtime(**shape)
        validate_dflash_trtllm_gen_runtime(**shape, folded_block_size=4)


# ---------------------------------------------------------------------------
# Full trtllm-gen path (SM100/SM103): paged cache, decode + merge vs oracle
# ---------------------------------------------------------------------------


def _trtllm_gen_ops_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
        get_dflash_trtllm_gen_ops,
    )

    reason = dflash_trtllm_gen_unavailability_reason()
    if reason is not None:
        pytest.skip(f"DFlash TRTLLM backend unavailable: {reason}")
    return get_dflash_trtllm_gen_ops()


class _PagedCase:
    """Requests with their own context lengths in one shuffled page pool."""

    def __init__(self, ops, ctx_lens, block_size, num_heads, num_kv_heads, head_dim, page_size):
        dev = torch.device("cuda")
        self.ops = ops
        self.ctx_lens = ctx_lens
        self.block_size = block_size
        B = len(ctx_lens)
        seq_lens = [c + block_size for c in ctx_lens]
        pages_per_req = [math.ceil(s / page_size) for s in seq_lens]
        self.max_pages = max(pages_per_req) + 1
        num_pages = sum(pages_per_req) + 3
        perm = torch.randperm(num_pages).tolist()
        self.block_tables = torch.zeros(B, self.max_pages, dtype=torch.int32, device=dev)
        cursor = 0
        for b, n in enumerate(pages_per_req):
            self.block_tables[b, :n] = torch.tensor(perm[cursor : cursor + n], dtype=torch.int32)
            cursor += n
        self.page_size = page_size
        self.cache = torch.zeros(
            num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.bfloat16, device=dev
        )
        self.keys = [
            torch.randn(s, num_kv_heads, head_dim, device=dev).to(torch.bfloat16) for s in seq_lens
        ]
        self.values = [
            torch.randn(s, num_kv_heads, head_dim, device=dev).to(torch.bfloat16) for s in seq_lens
        ]
        # Context K/V: written straight into the pages.
        for b, c in enumerate(ctx_lens):
            for pos in range(0, c, page_size):
                n = min(page_size, c - pos)
                page = self.block_tables[b, pos // page_size]
                self.cache[page, 0, :, :n] = self.keys[b][pos : pos + n].transpose(0, 1)
                self.cache[page, 1, :, :n] = self.values[b][pos : pos + n].transpose(0, 1)
        self.query = torch.randn(B, block_size, num_heads, head_dim, device=dev).to(torch.bfloat16)
        self.block_key = torch.stack([k[-block_size:] for k in self.keys])
        self.block_value = torch.stack([v[-block_size:] for v in self.values])
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        self.workspace = torch.empty(
            ops.get_workspace_size(
                dtype=torch.bfloat16,
                num_tokens=B * block_size,
                num_gen_tokens=B * block_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                max_num_requests=B,
                rotary_embedding_dim=0,
                fp8_context_fmha=False,
            ),
            dtype=torch.uint8,
            device=dev,
        )
        sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
        self.counters = torch.zeros(
            ops.get_multi_ctas_kv_counter_size(num_heads, B, sm_count),
            dtype=torch.uint8,
            device=dev,
        )

    def append_block(self):
        """The block K/V go in through the forward's append op."""
        B, blk = len(self.ctx_lens), self.block_size
        dev = self.query.device
        last = ((self.seq_lens - 1) % self.page_size) + 1
        self.ops.append_paged_kv_cache(
            append_key=self.block_key.reshape(B * blk, *self.block_key.shape[2:]).contiguous(),
            append_value=self.block_value.reshape(
                B * blk, *self.block_value.shape[2:]
            ).contiguous(),
            batch_indices=torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(blk),
            positions=(
                torch.tensor(self.ctx_lens, dtype=torch.int32, device=dev).view(-1, 1)
                + torch.arange(blk, dtype=torch.int32, device=dev)
            ).reshape(-1),
            paged_kv_cache=self.cache,
            kv_indices=self.block_tables.flatten(),
            kv_indptr=torch.arange(
                0, (B + 1) * self.max_pages, self.max_pages, dtype=torch.int32, device=dev
            ),
            kv_last_page_len=last,
            kv_layout="HND",
        )

    def run(self, causal, window_size, out=None):
        B, blk, num_heads, head_dim = self.query.shape
        out = (
            torch.empty(B * blk, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            if out is None
            else out
        )
        dflash_trtllm_windowed_block_attention(
            self.ops,
            self.query,
            self.block_key,
            self.block_value,
            self.cache,
            self.block_tables,
            self.seq_lens,
            self.max_pages * self.page_size,
            causal,
            window_size,
            self.workspace,
            self.counters,
            out,
        )
        return out


def _ctx_lens_around(window, block_size, page_size):
    w = window
    return sorted(
        {
            1,
            max(1, w - block_size - 1),
            max(1, w - block_size),
            w - 1,
            w,
            w + 1,
            w + page_size - 1,
            w + page_size,
            3 * w + 17,
        }
    )


@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("block_size", [1, 2, 4, 7])
@pytest.mark.parametrize("heads", [(16, 1), (16, 2)])
@pytest.mark.parametrize("window", [64, 4096])
def test_trtllm_windowed_non_causal_matches_the_oracle(page_size, block_size, heads, window):
    """Mixed context lengths in one batch, straddling the window and page
    boundaries, through the forward's append + decode + merge."""
    ops = _trtllm_gen_ops_or_skip()
    torch.manual_seed(block_size * 1000 + window + page_size)
    num_heads, num_kv_heads = heads
    ctx_lens = _ctx_lens_around(window, block_size, page_size)
    case = _PagedCase(ops, ctx_lens, block_size, num_heads, num_kv_heads, 128, page_size)
    case.append_block()
    w = window - 1
    got = case.run(False, (w, w)).float()
    expected, _ = _oracle(case.query, case.keys, case.values, ctx_lens, w, w, causal=False)
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    # The stock causal path is the same kernel without the merge; with more
    # than one block token it is a different attention.
    causal_expected, _ = _oracle(case.query, case.keys, case.values, ctx_lens, w, -1, causal=True)
    torch.testing.assert_close(
        case.run(True, (w, 0)).float(), causal_expected, atol=2e-2, rtol=2e-2
    )


def test_trtllm_windowed_non_causal_long_context():
    """Contexts far past the window: only the window may be read."""
    ops = _trtllm_gen_ops_or_skip()
    torch.manual_seed(7)
    ctx_lens = [40_000, 131_000, 4_095, 70]
    case = _PagedCase(ops, ctx_lens, 7, 16, 1, 128, 64)
    case.append_block()
    expected, _ = _oracle(case.query, case.keys, case.values, ctx_lens, 4095, 4095, causal=False)
    torch.testing.assert_close(
        case.run(False, (4095, 4095)).float(), expected, atol=2e-2, rtol=2e-2
    )


def test_trtllm_windowed_non_causal_replays_under_cuda_graph():
    """Captured once, replayed with new queries, block K/V and lengths."""
    ops = _trtllm_gen_ops_or_skip()
    torch.manual_seed(11)
    ctx_lens = [100, 4_090, 5_000]
    case = _PagedCase(ops, ctx_lens, 7, 16, 1, 128, 64)
    case.append_block()
    out = torch.empty(len(ctx_lens) * 7, 16, 128, dtype=torch.bfloat16, device="cuda")
    case.run(False, (4095, 4095), out)  # warm up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(False, (4095, 4095), out)

    # New data in the same buffers: shorter contexts, fresh queries and block.
    new_ctx = [90, 4_089, 4_999]
    case.ctx_lens[:] = new_ctx
    case.seq_lens.copy_(torch.tensor([c + 7 for c in new_ctx], dtype=torch.int32))
    case.query.copy_(torch.randn_like(case.query, dtype=torch.float32).to(torch.bfloat16))
    for b, c in enumerate(new_ctx):
        case.keys[b] = case.keys[b][: c + 7].clone()
        case.values[b] = case.values[b][: c + 7].clone()
        case.keys[b][-7:] = torch.randn_like(case.keys[b][-7:], dtype=torch.float32).to(
            torch.bfloat16
        )
        case.values[b][-7:] = torch.randn_like(case.values[b][-7:], dtype=torch.float32).to(
            torch.bfloat16
        )
    case.block_key.copy_(torch.stack([k[-7:] for k in case.keys]))
    case.block_value.copy_(torch.stack([v[-7:] for v in case.values]))
    case.append_block()
    graph.replay()
    expected, _ = _oracle(case.query, case.keys, case.values, new_ctx, 4095, 4095, causal=False)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.cpu_only
def test_windowed_decode_preserves_query_precision() -> None:
    query = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    block_kv = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    cache = torch.empty(1, 2, 1, 16, 8, dtype=torch.float8_e4m3fn)
    out = torch.empty(2, 4, 8, dtype=torch.bfloat16)
    seen = []

    def decode(**kwargs):
        seen.append(kwargs["query"])
        kwargs["out"].zero_()
        return kwargs["out"]

    dflash_trtllm_windowed_block_attention(
        SimpleNamespace(batch_decode_with_kv_cache=decode),
        query,
        block_kv,
        block_kv,
        cache,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        16,
        True,
        (7, 0),
        torch.empty(0),
        torch.empty(0),
        out,
    )
    assert seen[0].dtype == torch.bfloat16
    torch.testing.assert_close(seen[0], query.reshape_as(seen[0]))


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
@pytest.mark.parametrize("cache_as_list", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_dflash_paged_cache_dtype_is_validated_before_allocation(
    backend, cache_as_list, cache_dtype
):
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    class ReachedAllocation(Exception):
        pass

    def allocation_boundary(*args, **kwargs):
        raise ReachedAllocation

    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.dflash_attention_backend = backend
    wrapper._dflash_trtllm_gen_ops = None
    wrapper._dflash_fa4_fwd = None
    wrapper._dflash_paged_append = None
    wrapper._fused_kv_weight = object()
    wrapper._has_qk_norm = False
    wrapper._use_fused_qk_norm_rope = False
    wrapper.model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_size=32, kv_size=8, head_dim=8, num_heads=4, num_key_value_heads=1
                )
            )
        ]
    )
    wrapper._get_rope_cos_sin = lambda *args, **kwargs: (torch.empty(0), torch.empty(0))
    wrapper._prepare_dflash_trtllm_gen_buffers = allocation_boundary
    wrapper._prepare_dflash_index_buffers = allocation_boundary
    layer_cache = torch.empty(1, 2, 1, 16, 8, dtype=cache_dtype)
    cache = [layer_cache] if cache_as_list else layer_cache.unsqueeze(0)
    expected = ReachedAllocation if cache_dtype == torch.bfloat16 else ValueError
    match = None if cache_dtype == torch.bfloat16 else "draft activation dtype"
    with pytest.raises(expected, match=match):
        wrapper.dflash_forward(
            noise_embedding=torch.zeros(1, 2, 32, dtype=torch.bfloat16),
            query_positions=torch.tensor([[0, 1]]),
            num_ctx_per_req=torch.tensor([0]),
            ctx_k_cache=torch.empty(0),
            ctx_v_cache=torch.empty(0),
            ctx_cache_batch_idx=torch.tensor([0]),
            ctx_kv_cache=cache,
            ctx_page_table=torch.zeros(1, 1, dtype=torch.int32),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "block,heads,kv_heads,expected_heads",
    [
        (1, 4, 2, [4]),
        (4, 8, 2, [32]),
        (4, 16, 1, [64]),
        (8, 16, 1, [128]),
        (9, 16, 1, [128, 16]),
        (15, 32, 2, [256, 224]),
        (16, 24, 2, [240, 144]),
        (16, 12, 1, [120, 72]),
        (21, 6, 2, [126]),
        (43, 6, 2, [252, 6]),
    ],
)
def test_folded_decode_chunks_preserve_full_attention(
    block: int, heads: int, kv_heads: int, expected_heads: list[int]
) -> None:
    """Independent dense attention checks head grouping, query order and the tail chunk."""
    torch.manual_seed(302 + block)
    batch, dim = 2, 8
    context_lengths = [1, 3]
    query = torch.randn(batch, block, heads, dim)
    keys = [torch.randn(length + block, kv_heads, dim) for length in context_lengths]
    values = [torch.randn_like(key) for key in keys]
    output = torch.empty(batch * block, heads, dim)
    table = torch.zeros(batch, 1, dtype=torch.int32)
    lengths = torch.tensor([length + block for length in context_lengths], dtype=torch.int32)
    calls = []

    def decode(**kwargs):
        folded = kwargs["query"]
        calls.append(folded.shape[1])
        assert kwargs["seq_lens"] is lengths
        assert kwargs["block_tables"] is table
        assert kwargs["q_len_per_req"] == 1
        assert kwargs["window_left"] == -1
        for row in range(batch):
            result = torch.nn.functional.scaled_dot_product_attention(
                folded[row].unsqueeze(0).unsqueeze(2),
                keys[row].permute(1, 0, 2).unsqueeze(0),
                values[row].permute(1, 0, 2).unsqueeze(0),
                enable_gqa=True,
            )
            kwargs["out"][row].copy_(result[0, :, 0])
        return kwargs["out"]

    dflash_trtllm_folded_block_attention(
        SimpleNamespace(batch_decode_with_kv_cache=decode),
        query,
        torch.empty(1, 2, kv_heads, block + 3, dim),
        table,
        lengths,
        block + 3,
        torch.empty(0),
        torch.empty(0),
        output,
    )
    expected, _ = _oracle(query, keys, values, context_lengths, -1, -1, False)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
    assert calls == expected_heads


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "block,heads,kv_heads", [(16, 12, 1), (16, 24, 2)] + [(b, 16, 1) for b in range(9, 16)]
)
def test_folded_runtime_accepts_large_blocks_by_chunking(
    block: int, heads: int, kv_heads: int
) -> None:
    with patch("tensorrt_llm._torch.speculative.dflash_attention.get_dflash_trtllm_gen_ops"):
        validate_dflash_trtllm_gen_runtime(
            dtype=torch.bfloat16,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=128,
            tokens_per_block=32,
            folded_block_size=block,
        )
    chunk = get_dflash_folded_decode_chunk_size(block, heads, kv_heads)
    assert chunk > 0
    assert heads // kv_heads * chunk <= 128


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("tokens_per_block", "has_context_attention", "folded_block_size", "supported"),
    [
        (64, True, None, True),
        (64, False, 16, True),
        (128, False, None, True),
        (256, False, None, True),
        (96, False, None, False),
        (128, True, None, False),
        # The folded decode is not validated on trtllm-gen's dynamic page sizes.
        (128, False, 16, False),
        (256, False, 16, False),
    ],
)
def test_runtime_admits_large_pages_only_without_context_or_folded_layers(
    tokens_per_block: int,
    has_context_attention: bool,
    folded_block_size: Optional[int],
    supported: bool,
) -> None:
    """Pages of 128+ tokens serve drafters whose layers are all causal or windowed."""
    shape = dict(
        dtype=torch.bfloat16,
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=tokens_per_block,
        has_context_attention=has_context_attention,
        folded_block_size=folded_block_size,
    )
    with patch("tensorrt_llm._torch.speculative.dflash_attention.get_dflash_trtllm_gen_ops"):
        if supported:
            validate_dflash_trtllm_gen_runtime(**shape)
        else:
            with pytest.raises(RuntimeError, match="page size"):
                validate_dflash_trtllm_gen_runtime(**shape)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "block,heads,kv_heads",
    [(0, 16, 1), (-1, 16, 1), (4, 0, 1), (4, 16, 0), (4, 7, 2), (4, 64, 1)],
)
def test_folded_chunk_plan_rejects_unsupported_base_shapes(
    block: int, heads: int, kv_heads: int
) -> None:
    with pytest.raises(RuntimeError, match="positive|divisible|validated"):
        get_dflash_folded_decode_chunk_size(block, heads, kv_heads)


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
@pytest.mark.parametrize("bad_layer", [0, 1])
def test_paged_fp8_rejection_precedes_model_or_cache_mutation(backend: str, bad_layer: int) -> None:
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    wrapper = DFlashForCausalLM.__new__(DFlashForCausalLM)
    torch.nn.Module.__init__(wrapper)
    wrapper.dflash_attention_backend = backend
    caches = [torch.randn(1, 2, 1, 16, 8).to(torch.bfloat16) for _ in range(2)]
    caches[bad_layer] = caches[bad_layer].to(torch.float8_e4m3fn)
    snapshots = [cache.view(torch.uint8).clone() for cache in caches]
    before_state = dict(wrapper.__dict__)
    with pytest.raises(ValueError, match="draft activation dtype"):
        wrapper.dflash_forward(
            noise_embedding=torch.zeros(1, 2, 32, dtype=torch.bfloat16),
            query_positions=torch.tensor([[0, 1]]),
            num_ctx_per_req=torch.tensor([0]),
            ctx_k_cache=None,
            ctx_v_cache=None,
            ctx_cache_batch_idx=torch.tensor([0]),
            ctx_kv_cache=caches,
            ctx_page_table=torch.zeros(1, 1, dtype=torch.int32),
        )
    assert wrapper.__dict__ == before_state
    for cache, snapshot in zip(caches, snapshots):
        assert torch.equal(cache.view(torch.uint8), snapshot)


@pytest.mark.parametrize(
    "block,heads,kv_heads",
    [(1, 4, 2), (4, 8, 2), (4, 16, 1), (8, 16, 1), (16, 12, 1), (16, 24, 2)]
    + [(block, 16 * kv_heads, kv_heads) for block in range(9, 16) for kv_heads in (1, 2)],
)
def test_trtllm_folded_chunks_match_full_attention_and_graph(
    block: int, heads: int, kv_heads: int
) -> None:
    """Large effective ratios, multiple KV groups and partial final chunks."""
    ops = _trtllm_gen_ops_or_skip()
    torch.manual_seed(403 + block + heads)
    case = _PagedCase(ops, [0, 31, 65], block, heads, kv_heads, 128, 32)
    batch = len(case.ctx_lens)
    chunk = get_dflash_folded_decode_chunk_size(block, heads, kv_heads)
    workspace_bytes = ops.get_workspace_size(
        dtype=torch.bfloat16,
        num_tokens=batch,
        num_gen_tokens=batch,
        num_heads=heads * chunk,
        num_kv_heads=kv_heads,
        head_size=128,
        max_num_requests=batch,
        rotary_embedding_dim=0,
        fp8_context_fmha=False,
    )
    if case.workspace.numel() < workspace_bytes:
        case.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    case.counters = torch.zeros(
        ops.get_multi_ctas_kv_counter_size(
            heads * chunk, batch, torch.cuda.get_device_properties("cuda").multi_processor_count
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    output = torch.empty(batch * block, heads, 128, dtype=torch.bfloat16, device="cuda")

    decoded_heads = []

    def decode(**kwargs):
        decoded_heads.append(kwargs["query"].shape[1])
        return ops.batch_decode_with_kv_cache(**kwargs)

    observed_ops = SimpleNamespace(batch_decode_with_kv_cache=decode)

    def run():
        dflash_trtllm_folded_block_attention(
            observed_ops,
            case.query,
            case.cache,
            case.block_tables,
            case.seq_lens,
            case.max_pages * case.page_size,
            case.workspace,
            case.counters,
            output,
        )

    case.append_block()
    run()
    assert decoded_heads == [heads * min(chunk, block - start) for start in range(0, block, chunk)]
    if heads * block // kv_heads <= 128:
        assert decoded_heads == [heads * block]
    expected, _ = _oracle(case.query, case.keys, case.values, case.ctx_lens, -1, -1, False)
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        case.query.normal_()
        for row in range(batch):
            case.keys[row][-block:].normal_()
            case.values[row][-block:].normal_()
        case.block_key.copy_(torch.stack([key[-block:] for key in case.keys]))
        case.block_value.copy_(torch.stack([value[-block:] for value in case.values]))
        case.append_block()
        graph.replay()
        expected, _ = _oracle(case.query, case.keys, case.values, case.ctx_lens, -1, -1, False)
        torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
