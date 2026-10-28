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
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import (
    dflash_merge_future_block_attention,
    dflash_trtllm_windowed_block_attention,
)
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM
from tensorrt_llm._torch.speculative.dflash_attention import (
    DFLASH_CONTEXT_LOWERING,
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
    assert wrapper.validate_block_attention_windows() == ["non-causal context"] * 2


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
def test_windowed_layers_are_not_held_to_the_context_kernel_head_sizes(head_dim):
    """Regression: a windowed non-causal layer is lowered to the block decode,
    so the context kernel's head-size limits must not reject it; an unwindowed
    non-causal layer still is."""
    windowed = _wrapper(GQADSparkForCausalLM, _minimax_like_config(num_layers=2, window=512))
    lowerings = windowed.validate_block_attention_windows()
    assert DFLASH_CONTEXT_LOWERING not in lowerings
    unwindowed = _wrapper(
        GQADSparkForCausalLM,
        SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention"] * 2),
    )
    assert DFLASH_CONTEXT_LOWERING in unwindowed.validate_block_attention_windows()
    shape = dict(
        dtype=torch.bfloat16, num_heads=16, num_kv_heads=2, head_dim=head_dim, tokens_per_block=32
    )
    with patch("tensorrt_llm._torch.speculative.dflash_attention.get_dflash_trtllm_gen_ops"):
        validate_dflash_trtllm_gen_runtime(
            **shape, has_context_attention=DFLASH_CONTEXT_LOWERING in lowerings
        )
        with pytest.raises(RuntimeError, match="head dimension"):
            validate_dflash_trtllm_gen_runtime(**shape, has_context_attention=True)


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
