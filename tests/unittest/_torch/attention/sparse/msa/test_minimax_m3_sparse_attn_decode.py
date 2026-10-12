# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# PyTorch oracle derived from vLLM (Apache-2.0), _reference_sparse_attn:
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L755
"""Correctness tests for the Triton MiniMax-M3 sparse block decode attention.

The PyTorch oracle follows the vLLM reference linked in the file header
(v0.26.1rc0-77-g6f91edf96): softmax over exactly the selected blocks, each
truncated at the query token's own causal extent.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import run_msa_nvfp4_sparse_gqa
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    require_msa_module,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    NVFP4_SF_VEC_SIZE,
    SPARSE_BLOCK_SIZE,
    _sm100f_nvfp4_launch_options,
    _sm100f_nvfp4_merge_launch_options,
    _sm100f_nvfp4_num_topk_chunks,
    _sm100f_nvfp4_query_group_size,
    _sm100f_nvfp4_use_linear_softmax,
    minimax_m3_sparse_attn_decode,
    resolve_num_topk_chunks,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="MiniMax-M3 NVFP4 and fmha_sm100 require SM100/SM103.",
)


def _flat_page_table(block_table: torch.Tensor, kv_lens_cpu: torch.Tensor) -> torch.Tensor:
    """Flatten a block table into the per-request page ids fmha_sm100 consumes.

    The production helper concatenates the valid prefix of each request's
    block-id row according to its KV length.
    """
    return build_kv_page_indices(block_table.cpu().to(torch.int32), kv_lens_cpu, PAGE_SIZE)


def _reference_sparse_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    decode_query_len: int,
) -> torch.Tensor:
    """fp32 softmax over the selected blocks, truncated at each token's extent."""
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_paged.shape[1]
    group = num_heads // num_kv_heads
    max_topk = topk_idx.shape[-1]
    out = torch.zeros(total_q, num_heads, head_dim, device=q.device, dtype=torch.float32)
    k_f32, v_f32 = k_paged.float(), v_paged.float()
    positions = torch.arange(PAGE_SIZE, device=q.device)

    for req, seq_len in enumerate(seq_lens.tolist()):
        for intra in range(decode_query_len):
            token = req * decode_query_len + intra
            kv_len = max(seq_len - decode_query_len + intra + 1, 0)
            num_blocks = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
            real_topk = min(max_topk, num_blocks)
            if real_topk == 0:
                continue
            for kv_head in range(num_kv_heads):
                blocks = topk_idx[kv_head, token, :real_topk].tolist()
                pages = [int(block_table[req, int(b)]) for b in blocks]
                keys = torch.cat([k_f32[p, kv_head] for p in pages])
                values = torch.cat([v_f32[p, kv_head] for p in pages])
                valid = torch.cat([int(b) * PAGE_SIZE + positions < kv_len for b in blocks])
                q_rows = q[token, kv_head * group : (kv_head + 1) * group].float()
                logits = (q_rows @ keys.T) * sm_scale
                logits = logits.masked_fill(~valid[None, :], -float("inf"))
                probs = torch.softmax(logits, dim=-1)
                out[token, kv_head * group : (kv_head + 1) * group] = probs @ values
    return out


def _random_topk(
    seq_lens,
    *,
    num_kv_heads: int,
    decode_query_len: int,
    topk: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Ascending block ids with -1 tail padding, as the selector emits them."""
    total_q = len(seq_lens) * decode_query_len
    table = torch.full((num_kv_heads, total_q, topk), -1, device="cuda", dtype=torch.int32)
    for req, seq_len in enumerate(seq_lens):
        for intra in range(decode_query_len):
            token = req * decode_query_len + intra
            kv_len = max(seq_len - decode_query_len + intra + 1, 0)
            num_blocks = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
            real_topk = min(topk, num_blocks)
            if real_topk == 0:
                continue
            for kv_head in range(num_kv_heads):
                perm = torch.randperm(num_blocks, device="cuda", generator=generator)
                chosen = perm[:real_topk].sort().values
                table[kv_head, token, :real_topk] = chosen.to(torch.int32)
    return table


def _make_inputs(
    seq_lens,
    *,
    kv_dtype,
    q_dtype=torch.bfloat16,
    num_kv_heads=1,
    group=8,
    decode_query_len=1,
    topk=MSA_REQUIRED_TOPK,
    seed=0,
):
    """Build (q, k_paged, v_paged, topk_idx, block_table, seq_lens), in that order.

    The order matches both the kernel wrapper and the oracle, so a test can
    splat the tuple into either.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    total_q = batch * decode_query_len
    num_heads = num_kv_heads * group
    max_blocks = max(1, (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE)
    num_pages = batch * max_blocks

    # Shuffled pages: a kernel that ignored the block table and indexed the
    # cache by logical block would still pass with an identity mapping.
    block_table = torch.randperm(num_pages, device="cuda", generator=generator)
    block_table = block_table.to(torch.int32).reshape(batch, max_blocks)
    seq_lens_dev = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    q = torch.randn(
        total_q, num_heads, HEAD_DIM, device="cuda", generator=generator, dtype=torch.float32
    ).to(q_dtype)
    kv = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        PAGE_SIZE,
        HEAD_DIM,
        device="cuda",
        generator=generator,
        dtype=torch.float32,
    ).to(kv_dtype)
    # Non-contiguous K/V views of one coalesced pool, exactly as msa_paged_kv
    # hands them over.
    k_paged, v_paged = kv[:, 0], kv[:, 1]

    topk_idx = _random_topk(
        seq_lens,
        num_kv_heads=num_kv_heads,
        decode_query_len=decode_query_len,
        topk=topk,
        generator=generator,
    )
    return q, k_paged, v_paged, topk_idx, block_table, seq_lens_dev


def _dequant_nvfp4_flat(data: torch.Tensor, scale: torch.Tensor, dequant: float) -> torch.Tensor:
    """Dequantize fp4_quantize output in its own flat [rows, D] layout.

    Deliberately independent of any paged placement: this is the value oracle
    the paged readers are checked against.
    """
    rows = data.reshape(-1, data.shape[-1]).to(torch.int32)
    nibbles = torch.stack([rows & 0x0F, (rows >> 4) & 0x0F], dim=-1).reshape(rows.shape[0], -1)
    exponent = (nibbles & 7) >> 1
    mantissa = (nibbles & 1).float()
    magnitude = torch.where(
        exponent == 0, mantissa * 0.5, torch.exp2((exponent - 1).float()) * (1.0 + mantissa * 0.5)
    )
    values = torch.where((nibbles >> 3) & 1 == 1, -magnitude, magnitude)
    factors = scale.reshape(rows.shape[0], -1).view(torch.float8_e4m3fn).float()
    return values * factors.repeat_interleave(NVFP4_SF_VEC_SIZE, dim=-1) * dequant


def _make_nvfp4_inputs(seq_lens, *, num_kv_heads=1, group=8, decode_query_len=1, seed=0):
    """Build a paged NVFP4 cache with the production scatter kernel.

    Returns the kernel arguments alongside `k_ref`/`v_ref`: the same values
    dequantized in fp4_quantize's flat layout and then placed by slot id. A
    reference run on those measures the kernel's error, not the quantizer's.
    """
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
        fused_write_layer_caches_nvfp4,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    total_q = batch * decode_query_len
    num_heads = num_kv_heads * group
    max_blocks = max(1, (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE)
    num_pages = batch * max_blocks
    num_slots = num_pages * PAGE_SIZE
    scale_cols = HEAD_DIM // NVFP4_SF_VEC_SIZE

    block_table = torch.randperm(num_pages, device="cuda", generator=generator)
    block_table = block_table.to(torch.int32).reshape(batch, max_blocks)
    seq_lens_dev = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    q = torch.randn(
        total_q, num_heads, HEAD_DIM, device="cuda", generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    # One [pages, 2, ...] pool per role, matching get_block_scale_buffers and
    # msa_paged_kv: K is role 0, V role 1.
    data_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE_SIZE, HEAD_DIM // 2, device="cuda", dtype=torch.uint8
    )
    scale_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE_SIZE, scale_cols, device="cuda", dtype=torch.uint8
    )
    source = {}
    quant = torch.empty(3, device="cuda", dtype=torch.float32)
    quant[0] = 1.0
    for role in (0, 1):
        values = torch.randn(
            num_slots, num_heads, HEAD_DIM, device="cuda", generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        # One head group per KV head, so the cache holds num_kv_heads of them.
        values = values[:, :num_kv_heads].reshape(num_slots, num_kv_heads * HEAD_DIM)
        quant[role + 1] = (448.0 * 6.0) / values.abs().max().float().item()
        source[role] = values
    assert fused_write_layer_caches_nvfp4(
        data_pool[:, 0],
        data_pool[:, 1],
        scale_pool[:, 0],
        scale_pool[:, 1],
        None,
        torch.arange(num_slots, device="cuda", dtype=torch.int32),
        source[0],
        source[1],
        None,
        quant,
    )

    # Slot s lives at page s // PAGE_SIZE, token s % PAGE_SIZE.
    reference = []
    for role in (0, 1):
        flat = torch.ops.trtllm.fp4_quantize(
            source[role].reshape(num_slots, num_kv_heads, HEAD_DIM).contiguous(),
            quant[role + 1 : role + 2],
            NVFP4_SF_VEC_SIZE,
            False,
            False,
        )
        dequantized = _dequant_nvfp4_flat(
            flat[0].view(torch.uint8), flat[1].view(torch.uint8), 1.0 / quant[role + 1].item()
        )
        reference.append(
            dequantized.reshape(num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    topk_idx = _random_topk(
        seq_lens,
        num_kv_heads=num_kv_heads,
        decode_query_len=decode_query_len,
        topk=MSA_REQUIRED_TOPK,
        generator=generator,
    )
    scales = {
        "k_block_scale": scale_pool[:, 0],
        "v_block_scale": scale_pool[:, 1],
        "k_global_scale": (1.0 / quant[1:2]).contiguous(),
        "v_global_scale": (1.0 / quant[2:3]).contiguous(),
    }
    return SimpleNamespace(
        q=q,
        k_paged=data_pool[:, 0],
        v_paged=data_pool[:, 1],
        topk_idx=topk_idx,
        block_table=block_table,
        seq_lens=seq_lens_dev,
        scales=scales,
        scale_pool=scale_pool,
        k_ref=reference[0],
        v_ref=reference[1],
        decode_query_len=decode_query_len,
    )


def _run_nvfp4(case, **kwargs):
    # An E4M3 q is widened in-register, so the output stays at compute dtype.
    out_dtype = torch.bfloat16 if case.q.dtype == torch.float8_e4m3fn else case.q.dtype
    out = torch.empty(case.q.shape, device=case.q.device, dtype=out_dtype)
    minimax_m3_sparse_attn_decode(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        output=out,
        decode_query_len=case.decode_query_len,
        **case.scales,
        **kwargs,
    )
    return out


def _run(q, k_paged, v_paged, topk_idx, block_table, seq_lens, decode_query_len, **kwargs):
    out = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q,
        k_paged,
        v_paged,
        topk_idx,
        block_table,
        seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        output=out,
        decode_query_len=decode_query_len,
        **kwargs,
    )
    return out


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("num_kv_heads", "group", "decode_query_len"),
    [(1, 8, 1), (1, 8, 2), (2, 16, 1), (4, 4, 3)],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [128],
        [1, 128, 129],
        [1025, 4097],
        [300, 1500, 2049, 4096, 5000, 7777],
    ],
    ids=["one-block", "short-mixed", "two-req", "batch6"],
)
def test_sparse_decode_matches_reference(kv_dtype, num_kv_heads, group, decode_query_len, seq_lens):
    """Attention over the selected blocks must match the oracle across shapes."""
    seq_lens = [max(s, decode_query_len) for s in seq_lens]
    inputs = _make_inputs(
        seq_lens,
        kv_dtype=kv_dtype,
        num_kv_heads=num_kv_heads,
        group=group,
        decode_query_len=decode_query_len,
    )
    out = _run(*inputs, decode_query_len)
    expected = _reference_sparse_decode(
        *inputs, sm_scale=HEAD_DIM**-0.5, decode_query_len=decode_query_len
    )
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("num_kv_heads", "group", "decode_query_len"),
    [(1, 8, 1), (2, 16, 1), (4, 4, 3)],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[128], [1, 128, 129], [300, 1500, 2049, 4096]],
    ids=["one-block", "short-mixed", "batch4"],
)
def test_sparse_decode_fp8_q_matches_prewidened_q(num_kv_heads, group, decode_query_len, seq_lens):
    """FP8 q must give bitwise the same answer as a caller-widened BF16 q.

    A fused QK-norm/RoPE producer emits E4M3 q, which the kernel widens
    in-register so that no standalone widening kernel runs per sparse layer.
    E4M3 -> BF16 is exact, so this is an equality rather than a tolerance; a
    regression that let the narrow q reach tl.dot would run the whole attention
    in FP8 and show up here immediately.
    """
    seq_lens = [max(s, decode_query_len) for s in seq_lens]
    q, *rest = _make_inputs(
        seq_lens,
        kv_dtype=torch.float8_e4m3fn,
        q_dtype=torch.float8_e4m3fn,
        num_kv_heads=num_kv_heads,
        group=group,
        decode_query_len=decode_query_len,
        seed=61,
    )
    k_paged, v_paged, topk_idx, block_table, seq_lens_dev = rest

    def run(query):
        out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
        minimax_m3_sparse_attn_decode(
            query,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens_dev,
            sm_scale=HEAD_DIM**-0.5,
            output=out,
            decode_query_len=decode_query_len,
        )
        return out

    torch.testing.assert_close(run(q), run(q.to(torch.bfloat16)), rtol=0, atol=0)


@pytest.mark.parametrize("num_topk_chunks", [1, 2, 4, 8, 16])
def test_sparse_decode_split_k_invariant(num_topk_chunks):
    """Flash-decoding must merge to the same answer for any split-K factor.

    A wrong LSE merge shows up here and nowhere else, because the default
    chunk count for a small batch is usually large enough to hide it.
    """
    seq_lens = [4097, 300, 8192]
    inputs = _make_inputs(seq_lens, kv_dtype=torch.float8_e4m3fn, num_kv_heads=2, group=8, seed=7)
    reference = _run(*inputs, 1, num_topk_chunks=1)
    out = _run(*inputs, 1, num_topk_chunks=num_topk_chunks)
    torch.testing.assert_close(out.float(), reference.float(), rtol=1e-2, atol=1e-2)


def test_sparse_decode_ignores_padded_topk_entries():
    """Rows whose valid block count is below topk must not read the -1 tail.

    Every request here is one or two blocks long against topk=16, so all but a
    couple of entries are -1; dereferencing them would fault or corrupt.
    """
    seq_lens = [1, 64, 128, 129, 200]
    inputs = _make_inputs(seq_lens, kv_dtype=torch.bfloat16, seed=13)
    topk_idx = inputs[3]
    assert (topk_idx == -1).any()

    out = _run(*inputs, 1)
    expected = _reference_sparse_decode(*inputs, sm_scale=HEAD_DIM**-0.5, decode_query_len=1)
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


def test_sparse_decode_zero_length_rows_are_zero_not_nan():
    """CUDA-graph padding rows attend nothing; they must emit zeros.

    A NaN here would be discarded from the padded output but would still reach
    the residual stream and the tensor-parallel all-reduce.
    """
    seq_lens = [1024, 0, 512, 0]
    inputs = _make_inputs(seq_lens, kv_dtype=torch.float8_e4m3fn, seed=21)
    out = _run(*inputs, 1)

    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    assert out[0].abs().sum() > 0


def test_sparse_decode_accepts_token_major_topk_table():
    """The kernel reads the top-k table by stride, so either backing works."""
    seq_lens = [1025, 4097]
    q, k_paged, v_paged, head_major, block_table, seq_lens_dev = _make_inputs(
        seq_lens, kv_dtype=torch.bfloat16, num_kv_heads=2, group=8, seed=31
    )
    token_major = head_major.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    assert head_major.is_contiguous() and not token_major.is_contiguous()

    args = (q, k_paged, v_paged)
    out_hm = _run(*args, head_major, block_table, seq_lens_dev, 1)
    out_tm = _run(*args, token_major, block_table, seq_lens_dev, 1)
    assert torch.equal(out_hm, out_tm)


@pytest.mark.parametrize("num_topk_chunks", [1, 4], ids=["direct-output", "split-merge"])
def test_sparse_decode_cuda_graph_replay_tracks_inputs(num_topk_chunks):
    """Replay must recompute from the live buffers, not reuse captured values."""
    seq_lens = [2048, 4097, 900]
    q, k_paged, v_paged, topk_idx, block_table, seq_lens_dev = _make_inputs(
        seq_lens, kv_dtype=torch.float8_e4m3fn, num_kv_heads=2, group=8, seed=41
    )
    out = torch.empty_like(q)

    def run():
        minimax_m3_sparse_attn_decode(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens_dev,
            sm_scale=HEAD_DIM**-0.5,
            output=out,
            decode_query_len=1,
            num_topk_chunks=num_topk_chunks,
        )

    # Warm up on a side stream so the Triton JIT and the scratch arena are
    # settled before capture, which forbids both.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    generator = torch.Generator(device="cuda").manual_seed(99)
    q.copy_(torch.randn(q.shape, device="cuda", generator=generator, dtype=torch.float32))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_sparse_decode(
        q, k_paged, v_paged, topk_idx, block_table, seq_lens_dev, HEAD_DIM**-0.5, 1
    )
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_msa_prefill_head_major_selector_cuda_graph(dtype: torch.dtype) -> None:
    """A short context must not read a generation token's selector backing."""
    msa = require_msa_module()
    num_queries, num_heads, num_kv_heads, topk = 6, 32, 2, MSA_REQUIRED_TOPK
    q = torch.zeros((num_queries, num_heads, HEAD_DIM), device="cuda", dtype=dtype)
    k = torch.zeros((2 * topk, num_kv_heads, PAGE_SIZE, HEAD_DIM), device="cuda", dtype=dtype)
    v = torch.ones_like(k)
    page_ids = torch.arange(2 * topk, device="cuda", dtype=torch.int32)

    # Six context tokens followed by eight generation tokens, in the indexer's
    # head-major backing. The context slice retains the full batch's head stride.
    selected = torch.empty((num_kv_heads, 14, topk), device="cuda", dtype=torch.int32)
    selected = selected.transpose(0, 1)
    selected[:] = torch.arange(topk, device="cuda", dtype=torch.int32)
    # Contiguous-pointer addressing misreads this generation row as context
    # query 4/head 0. Its pages are allocated but beyond the context's causal
    # extent, so the unfixed kernel produces NaNs from entirely finite inputs.
    selected[8, 0] = torch.arange(topk, 2 * topk, device="cuda", dtype=torch.int32)
    context_selected = selected[:num_queries]
    assert not context_selected.is_contiguous()

    kv_len = topk * PAGE_SIZE
    plan = msa.fmha_sm100_plan(
        torch.tensor([num_queries], dtype=torch.int32),
        torch.tensor([kv_len], dtype=torch.int32),
        num_heads,
        num_kv_heads=num_kv_heads,
        qo_offset=torch.tensor([kv_len - num_queries], dtype=torch.int32),
        page_size=PAGE_SIZE,
        kv_block_num=topk,
        causal=True,
        num_kv_splits=1,
        use_fp8_kvcache=dtype == torch.float8_e4m3fn,
    )
    assert not plan[3]["MM-SA-Nv"], "Regression must exercise the CUTLASS short-query path"
    out = torch.empty_like(q, dtype=torch.bfloat16)
    kwargs = dict(
        kv_indices=page_ids,
        kv_block_indexes=context_selected,
        out=out,
        sm_scale=HEAD_DIM**-0.5,
        output_maxscore=False,
    )
    msa.fmha_sm100(q, k, v, plan, **kwargs)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        msa.fmha_sm100(q, k, v, plan, **kwargs)
    for value in (1.0, 2.0):
        v.fill_(value)
        graph.replay()
        torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    ("layout", "num_kv_splits", "group"),
    [
        ("contiguous", 1, 16),
        ("head-slice", 1, 16),
        ("column-slice", 2, 16),
        ("broadcast", 2, 1),
        ("column-broadcast", 2, 16),
    ],
)
def test_msa_prefill_selector_strides_cuda_graph(
    dtype: torch.dtype, layout: str, num_kv_splits: int, group: int
) -> None:
    """Loader, padding scan and mask must read the same live selector view."""
    msa = require_msa_module()
    num_queries, num_kv_heads, topk = 6, 2, MSA_REQUIRED_TOPK
    num_heads = num_kv_heads * group
    num_pages = 2 * topk
    q = torch.zeros((num_queries, num_heads, HEAD_DIM), device="cuda", dtype=dtype)
    k = torch.zeros((num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM), device="cuda", dtype=dtype)
    # Page- and head-dependent values make wrong selector rows observable. With
    # Q=K=0 the output is
    # the mean over selected pages, which exposes finite-but-wrong row reads.
    page_values = torch.arange(num_pages, device="cuda")[:, None] // 4
    page_values = page_values + 8 * torch.arange(num_kv_heads, device="cuda")[None, :]
    v = page_values[:, :, None, None].expand_as(k).to(dtype).contiguous()
    page_ids = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    if layout == "contiguous":
        selected = torch.empty((num_queries, num_kv_heads, topk), device="cuda", dtype=torch.int32)
    elif layout == "head-slice":
        backing = torch.full((num_kv_heads, 14, topk), -1, device="cuda", dtype=torch.int32)
        selected = backing.transpose(0, 1)[3 : 3 + num_queries]
    elif layout == "column-slice":
        backing = torch.full((num_kv_heads, 14, 2 * topk + 1), -1, device="cuda", dtype=torch.int32)
        selected = backing.transpose(0, 1)[3 : 3 + num_queries, :, 1::2]
    elif layout == "column-broadcast":
        backing = torch.empty((num_queries, num_kv_heads, 1), device="cuda", dtype=torch.int32)
        selected = backing.expand(num_queries, num_kv_heads, topk)
        assert selected.stride(2) == 0
    elif layout == "large-head-gap":
        head_stride = 2**31
        numel = head_stride + num_queries * topk
        required_bytes = numel * 4
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"Large-offset preflight: required={required_bytes}, free={free_bytes}, total={total_bytes}"
        )
        if free_bytes < required_bytes + 4 * 2**30:
            pytest.skip("Large-offset validation needs 8 GiB backing plus 4 GiB headroom")
        # Allocate but never initialize the gap. Only the 192 logical elements
        # are written and consumed; head 1 uses an actual offset above INT32_MAX.
        backing = torch.empty((numel,), device="cuda", dtype=torch.int32)
        selected = backing.as_strided((num_queries, num_kv_heads, topk), (topk, head_stride, 1))
        assert selected.stride(1) == 2**31
    else:
        backing = torch.empty((1, 1, topk), device="cuda", dtype=torch.int32)
        selected = backing.expand(num_queries, num_kv_heads, topk)

    rows = torch.arange(num_queries, device="cuda")[:, None, None]
    heads = torch.arange(num_kv_heads, device="cuda")[None, :, None]
    columns = torch.arange(topk, device="cuda")[None, None, :]
    valid_count = torch.where((rows + heads) % 2 == 0, 4, 8)
    logical = torch.where(columns < valid_count, (rows + heads) % 4 + columns, -1).to(torch.int32)
    if layout == "broadcast":
        logical = logical[:1, :1].expand(num_queries, num_kv_heads, topk)
        backing.copy_(logical[:1, :1])
    elif layout == "column-broadcast":
        logical = (4 * ((rows + heads) % 4)).expand(num_queries, num_kv_heads, topk).to(torch.int32)
        backing.copy_(logical[:, :, :1])
    else:
        selected.copy_(logical)

    kv_len = num_pages * PAGE_SIZE
    plan = msa.fmha_sm100_plan(
        torch.tensor([num_queries], dtype=torch.int32),
        torch.tensor([kv_len], dtype=torch.int32),
        num_heads,
        num_kv_heads=num_kv_heads,
        qo_offset=torch.tensor([kv_len - num_queries], dtype=torch.int32),
        page_size=PAGE_SIZE,
        kv_block_num=topk,
        causal=True,
        num_kv_splits=num_kv_splits,
        use_fp8_kvcache=dtype == torch.float8_e4m3fn,
    )
    assert not plan[3]["MM-SA-Nv"]
    assert plan[3]["num_kv_splits"] == num_kv_splits
    out = torch.empty_like(q, dtype=torch.bfloat16)
    kwargs = dict(kv_indices=page_ids, kv_block_indexes=selected, out=out, output_maxscore=False)
    msa.fmha_sm100(q, k, v, plan, **kwargs)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        msa.fmha_sm100(q, k, v, plan, **kwargs)

    for shift in (0, 4):
        current = torch.where(logical >= 0, logical + shift, -1)
        if layout == "broadcast":
            backing.copy_(current[:1, :1])
        elif layout == "column-broadcast":
            backing.copy_(current[:, :, :1])
        else:
            selected.copy_(current)
        # Gather the known page values directly, independently of MSA's
        # implementation. All selected pages are fully inside the causal range.
        values = page_values[current.clamp_min(0).long(), heads]
        valid = current >= 0
        expected = (values * valid).sum(-1).float() / valid.sum(-1)
        expected = (
            expected.repeat_interleave(group, dim=1).unsqueeze(-1).expand_as(out).to(out.dtype)
        )
        if shift == 0:
            torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@skip_not_sm100
@pytest.mark.skipif(
    os.environ.get("MSA_VALIDATE_LARGE_OFFSET") != "1", reason="8 GiB boundary validation is opt-in"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_msa_prefill_selector_large_offset_cuda_graph(dtype: torch.dtype) -> None:
    test_msa_prefill_selector_strides_cuda_graph(dtype, "large-head-gap", 2, 16)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("group", [1, 16])
@pytest.mark.parametrize("num_kv_splits", [1, 2])
def test_msa_nonpaged_original_q_metadata(
    dtype: torch.dtype, group: int, num_kv_splits: int
) -> None:
    """Nonpaged load arguments must retain the original packed-Q layout."""
    msa = require_msa_module()
    torch.manual_seed(20260929)
    num_queries, kv_len, num_kv_heads = 6, 2048, 2
    num_heads = num_kv_heads * group
    # Vary every Q row/head/channel: uniform Q would hide lost layout metadata.
    q = torch.randn((num_queries, num_heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    k = torch.randn((kv_len, num_kv_heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(dtype)
    v = torch.randn((kv_len, num_kv_heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16).to(dtype)
    plan = msa.fmha_sm100_plan(
        torch.tensor([num_queries], dtype=torch.int32),
        torch.tensor([kv_len], dtype=torch.int32),
        num_heads,
        num_kv_heads=num_kv_heads,
        causal=True,
        qo_offset=torch.tensor([kv_len - num_queries], dtype=torch.int32),
        page_size=-1,
        num_kv_splits=num_kv_splits,
        output_maxscore=False,
    )
    assert not plan[3]["MM-SA-Nv"]
    assert plan[3]["pack_factor"] == group
    assert plan[3]["num_kv_splits"] == num_kv_splits
    assert q.stride() == (num_heads * HEAD_DIM, HEAD_DIM, 1)
    out, _ = msa.fmha_sm100(q, k, v, plan, output_maxscore=False)

    # Selecting every identity-mapped page gives dense attention. The existing
    # oracle's decode extent kv_len - num_queries + token + 1 matches qo_offset.
    num_pages = kv_len // PAGE_SIZE
    k_pages = k.reshape(num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM).transpose(1, 2)
    v_pages = v.reshape(num_pages, PAGE_SIZE, num_kv_heads, HEAD_DIM).transpose(1, 2)
    all_pages = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    selected = all_pages.view(1, 1, num_pages).expand(num_kv_heads, num_queries, num_pages)
    expected = _reference_sparse_decode(
        q,
        k_pages,
        v_pages,
        selected,
        all_pages.view(1, num_pages),
        torch.tensor([kv_len], device="cuda", dtype=torch.int32),
        HEAD_DIM**-0.5,
        num_queries,
    )
    cosine = torch.nn.functional.cosine_similarity(out.float().flatten(), expected.flatten(), dim=0)
    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
    reference_norm = torch.linalg.vector_norm(expected)
    relative_l2 = torch.linalg.vector_norm(out.float() - expected) / reference_norm
    norm_ratio = torch.linalg.vector_norm(out.float()) / reference_norm
    print(
        f"nonpaged oracle: dtype={dtype}, group={group}, splits={num_kv_splits}, "
        f"cosine={cosine.item():.9g}, relative_l2={relative_l2.item():.9g}, norm_ratio={norm_ratio.item():.9g}"
    )

    assert torch.isfinite(out).all()
    assert cosine > threshold
    # Cosine alone accepts uniformly rescaled output. Final reference checks
    # observed a maximum norm deviation below 0.07% across both dtypes.
    torch.testing.assert_close(norm_ratio, torch.ones_like(norm_ratio), rtol=0, atol=1e-2)


@skip_not_sm100
def test_sparse_decode_matches_msa_kernel():
    """A/B against the fmha_sm100 sparse GQA path this kernel replaces."""
    from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import run_msa_sparse_gqa

    seq_lens = [1025, 4097, 300, 8192]
    q, k_paged, v_paged, topk_idx, block_table, seq_lens_dev = _make_inputs(
        seq_lens, kv_dtype=torch.float8_e4m3fn, num_kv_heads=1, group=8, seed=53
    )
    sm_scale = HEAD_DIM**-0.5
    batch = len(seq_lens)

    triton_out = _run(q, k_paged, v_paged, topk_idx, block_table, seq_lens_dev, 1)

    qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
    kv_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    kv_indices = _flat_page_table(block_table, kv_lens_cpu).cuda()
    msa_out = torch.empty_like(q)
    run_msa_sparse_gqa(
        q.to(torch.float8_e4m3fn),
        k_paged,
        v_paged,
        topk_idx.permute(1, 0, 2).contiguous(),
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=kv_lens_cpu - qo_lens_cpu,
        causal=True,
        head_dim=HEAD_DIM,
        out=msa_out,
        use_fp8=True,
    )

    # fmha_sm100 quantizes q to E4M3 while the Triton kernel keeps it in
    # bf16, so the two differ by q's quantization error alone.
    torch.testing.assert_close(triton_out.float(), msa_out.float(), rtol=6e-2, atol=6e-2)


@skip_not_sm100
@pytest.mark.parametrize(
    ("num_kv_heads", "group", "decode_query_len"),
    [(1, 8, 1), (1, 8, 2), (2, 16, 1), (4, 4, 3)],
)
@pytest.mark.parametrize(
    "seq_lens",
    [[128], [1, 128, 129], [1025, 4097], [300, 1500, 2049, 4096]],
    ids=["one-block", "short-mixed", "two-req", "batch4"],
)
def test_nvfp4_sparse_decode_matches_reference(num_kv_heads, group, decode_query_len, seq_lens):
    """The packed cache read must agree with the same values dequantized."""
    seq_lens = [max(s, decode_query_len) for s in seq_lens]
    case = _make_nvfp4_inputs(
        seq_lens, num_kv_heads=num_kv_heads, group=group, decode_query_len=decode_query_len
    )
    out = _run_nvfp4(case)
    expected = _reference_sparse_decode(
        case.q,
        case.k_ref,
        case.v_ref,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=decode_query_len,
    )
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@skip_not_sm100
@pytest.mark.parametrize("local_batch", [11, 12, 14])
def test_nvfp4_sparse_decode_eagle_pair_matches_divergent_reference(local_batch):
    """The paired producer must remain exact when adjacent selections differ."""
    seq_lens = [131 + 137 * row for row in range(local_batch)]
    case = _make_nvfp4_inputs(
        seq_lens,
        num_kv_heads=4,
        group=16,
        decode_query_len=4,
        seed=113 + local_batch,
    )
    case.q = case.q.to(torch.float8_e4m3fn)
    assert not torch.equal(case.topk_idx[:, 0::4], case.topk_idx[:, 1::4])

    out = _run_nvfp4(case)
    expected = _reference_sparse_decode(
        case.q.to(torch.bfloat16),
        case.k_ref,
        case.v_ref,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=case.decode_query_len,
    )
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@skip_not_sm100
def test_nvfp4_sparse_decode_fp8_q_matches_reference():
    """Production FP8 q may use FP16 dot operands without changing the cache math."""
    case = _make_nvfp4_inputs([300, 1500, 4097], num_kv_heads=2, group=8, seed=67)
    case.q = case.q.to(torch.float8_e4m3fn)
    out = _run_nvfp4(case)
    expected = _reference_sparse_decode(
        case.q.to(torch.bfloat16),
        case.k_ref,
        case.v_ref,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=case.decode_query_len,
    )
    torch.testing.assert_close(out.float(), expected, rtol=3e-2, atol=3e-2)


@skip_not_sm100
def test_nvfp4_paged_scale_layouts_are_the_ones_the_kernel_reads():
    """Lock the two block-scale layouts and the nibble order.

    The scatter kernel writes K's scale bytes token-major linear and V's in
    vLLM's 4x4 token-quad order; the decode kernel inverts both from the same
    formulas. Reproducing them here means a change to either side has to change
    this test too, rather than silently reading a neighbor's scale.
    """
    case = _make_nvfp4_inputs([4096], num_kv_heads=2, group=8, seed=71)
    num_pages, num_kv_heads = case.k_paged.shape[0], case.k_paged.shape[1]
    scale_cols = HEAD_DIM // NVFP4_SF_VEC_SIZE
    token = torch.arange(PAGE_SIZE, device="cuda")[:, None]
    col = torch.arange(scale_cols, device="cuda")[None, :]
    linear = token * scale_cols + col
    swizzled = (token // 4) * (4 * scale_cols) + 4 * col + (token % 4)

    for role, expected, offsets in (
        (0, case.k_ref, linear),
        (1, case.v_ref, swizzled),
    ):
        data = case.k_paged if role == 0 else case.v_paged
        flat = case.scale_pool[:, role].reshape(num_pages, num_kv_heads, PAGE_SIZE * scale_cols)
        gathered = flat[..., offsets.reshape(-1)].reshape(
            num_pages, num_kv_heads, PAGE_SIZE, scale_cols
        )
        dequant = case.scales["k_global_scale" if role == 0 else "v_global_scale"].item()
        decoded = _dequant_nvfp4_flat(data, gathered, dequant).reshape(
            num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM
        )
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


@skip_not_sm100
@pytest.mark.parametrize("num_topk_chunks", [1, 2, 4, 8, 16])
def test_nvfp4_sparse_decode_split_k_invariant(num_topk_chunks):
    """Flash-decoding over an NVFP4 cache must merge to one answer.

    V's per-tensor scale is applied per partial, so this also covers the claim
    that doing so survives the merge.
    """
    case = _make_nvfp4_inputs([4097, 300, 8192], num_kv_heads=2, group=8, seed=83)
    reference = _run_nvfp4(case, num_topk_chunks=1)
    out = _run_nvfp4(case, num_topk_chunks=num_topk_chunks)
    torch.testing.assert_close(out.float(), reference.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(("num_kv_heads", "group"), [(1, 8), (2, 16)], ids=["h1-g8", "h2-g16"])
@skip_not_sm100
def test_nvfp4_sparse_decode_matches_msa_csr_kernel(num_kv_heads, group):
    """A/B against MSA's NVFP4 CSR kernel over the same cache bytes.

    Three independent readings of one layout contract have to agree here: the
    Triton scatter that wrote the bytes, MSA's CuTe reader, and this kernel's
    reader. A torch oracle alone could share a misreading of the layout.

    FP8 staging introduces ordinary rounding after full NVFP4 reconstruction;
    it must not clip the unscaled FP4/block-scale product.
    """
    require_msa_module()
    seq_lens = [1025, 4097, 300, 8192]
    case = _make_nvfp4_inputs(seq_lens, num_kv_heads=num_kv_heads, group=group, seed=97)
    # E4M3 is what the fused M3 producer emits and what the CSR kernel needs.
    # Sharing it removes q quantization from the comparison; the Triton kernel
    # widens it back exactly.
    case.q = case.q.to(torch.float8_e4m3fn)
    batch = len(seq_lens)
    kv_lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    cu_kv = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    torch.cumsum(kv_lens, 0, out=cu_kv[1:])
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata.__dict__.update(
        _msa_live_batch=batch,
        msa_cu_q_lens=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        msa_cu_kv_lens=cu_kv,
        _msa_max_q_len=1,
        _msa_max_kv_len_all=max(seq_lens),
        _msa_total_k=sum(seq_lens),
        _msa_total_k_rows=sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens),
        msa_block_table=case.block_table,
        msa_seq_lens_cuda=case.seq_lens,
        _num_contexts=2,
        _msa_context_prefix_bounds=(
            1,
            max(seq_lens[:2]),
            sum(seq_lens[:2]),
            sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens[:2]),
        ),
        _num_generations=batch - 2,
    )

    triton_out = _run_nvfp4(case)
    msa_out = torch.empty(case.q.shape, device="cuda", dtype=torch.bfloat16)
    run_msa_nvfp4_sparse_gqa(
        case.q,
        case.k_paged,
        case.v_paged,
        case.scale_pool,
        case.topk_idx.permute(1, 0, 2).contiguous(),
        metadata,
        sm_scale=HEAD_DIM**-0.5,
        k_global_scale=case.scales["k_global_scale"],
        v_global_scale=case.scales["v_global_scale"],
        out=msa_out,
    )

    # A mixed step sends only its context prefix to CSR prefill. The longer
    # generation suffix must not affect the prefix worklist or output.
    prefix_rows = metadata.num_contexts
    prefix_out = torch.empty_like(msa_out[:prefix_rows])
    run_msa_nvfp4_sparse_gqa(
        case.q[:prefix_rows],
        case.k_paged,
        case.v_paged,
        case.scale_pool,
        case.topk_idx[:, :prefix_rows].permute(1, 0, 2).contiguous(),
        metadata,
        sm_scale=HEAD_DIM**-0.5,
        k_global_scale=case.scales["k_global_scale"],
        v_global_scale=case.scales["v_global_scale"],
        num_rows=prefix_rows,
        out=prefix_out,
    )
    torch.testing.assert_close(prefix_out, msa_out[:prefix_rows], rtol=1e-2, atol=1e-2)

    reference = _reference_sparse_decode(
        case.q.to(torch.bfloat16),
        case.k_ref,
        case.v_ref,
        case.topk_idx,
        case.block_table,
        case.seq_lens,
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=1,
    )
    torch.testing.assert_close(triton_out.float(), reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(msa_out.float(), reference, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("pair_dequant", [False, True], ids=["scalar", "pair"])
@pytest.mark.parametrize("kv_len", [1, 2], ids=["one-token", "k-sensitive"])
@skip_not_sm100
def test_msa_nvfp4_calibrated_staging_and_scale_replay(
    q_dtype: torch.dtype, pair_dequant: bool, kv_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Global scales must precede FP8 conversion and remain live during replay."""
    monkeypatch.setenv("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", str(int(pair_dequant)))
    require_msa_module()
    q = torch.zeros((1, 8, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    q[:, :, 0] = 1.0
    q = q.to(q_dtype)
    data = torch.zeros((1, 2, 1, PAGE_SIZE, HEAD_DIM // 2), device="cuda", dtype=torch.uint8)
    # E2M1 nibble 7 is +6, nibble 15 is -6. With block scale 448,
    # the unscaled magnitude is 2688, outside E4M3's finite range.
    data[:, 0, :, 0] = 0xFF
    data[:, 0, :, 1] = 0x77
    data[:, 1, :, 0] = 0x77
    data[:, 1, :, 1] = 0xFF
    # Constant scales work for both K-linear and V-swizzle4x4 layouts.
    scales = (
        torch.full(
            (1, 2, 1, PAGE_SIZE, HEAD_DIM // NVFP4_SF_VEC_SIZE),
            448.0,
            device="cuda",
            dtype=torch.float32,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k_global = torch.tensor([2.0 / 2688.0], device="cuda", dtype=torch.float32)
    v_global = torch.tensor([1.0 / 2688.0], device="cuda", dtype=torch.float32)
    metadata = object.__new__(MiniMaxM3MsaSparseAttentionMetadata)
    metadata._msa_live_batch = 1
    metadata.msa_cu_q_lens = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    metadata.msa_cu_kv_lens = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    metadata._msa_max_q_len = 1
    metadata._msa_max_kv_len_all = kv_len
    metadata._msa_total_k = kv_len
    metadata._msa_total_k_rows = 1
    metadata.msa_block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    metadata.msa_seq_lens_cuda = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    topk = torch.full((1, 1, MSA_REQUIRED_TOPK), -1, device="cuda", dtype=torch.int32)
    topk[..., 0] = 0
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)

    def run() -> None:
        run_msa_nvfp4_sparse_gqa(
            q,
            data[:, 0],
            data[:, 1],
            scales,
            topk,
            metadata,
            sm_scale=1.0,
            k_global_scale=k_global,
            v_global_scale=v_global,
            out=out,
        )

    # One token must reconstruct V=1, not the premature-saturation result 1/6.
    # Two tokens also expose K saturation through softmax([-2, 2]).
    logits = torch.tensor([-2.0, 2.0], device="cuda")[:kv_len]
    values = torch.tensor([1.0, -1.0], device="cuda")[:kv_len]
    expected = (logits.softmax(0) * values).sum().expand_as(out)
    run()
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    k_ptr, v_ptr = k_global.data_ptr(), v_global.data_ptr()
    k_global.mul_(0.5)
    v_global.mul_(2.0)
    expected_updated = ((logits * 0.5).softmax(0) * values * 2.0).sum().expand_as(out)
    graph.replay()
    torch.testing.assert_close(out.float(), expected_updated, rtol=2e-2, atol=2e-2)
    run()
    torch.testing.assert_close(out.float(), expected_updated, rtol=2e-2, atol=2e-2)
    assert (k_global.data_ptr(), v_global.data_ptr()) == (k_ptr, v_ptr)


@pytest.mark.parametrize(
    ("total_q", "num_kv_heads", "max_topk"),
    [(1, 1, 16), (8, 1, 16), (64, 2, 16), (512, 4, 16), (4096, 8, 16)],
)
def test_resolve_num_topk_chunks_is_shape_only_power_of_two(total_q, num_kv_heads, max_topk):
    """A power of two no larger than max_topk, decided by shape alone.

    The kernel splits the top-k blocks by shifting, and the launch geometry has
    to be the same at capture and at replay, so it may not depend on anything
    the shapes do not carry.
    """
    chunks = resolve_num_topk_chunks(total_q, num_kv_heads, max_topk)
    assert 1 <= chunks <= max_topk
    assert chunks & (chunks - 1) == 0
    assert chunks == resolve_num_topk_chunks(total_q, num_kv_heads, max_topk)


@pytest.mark.parametrize(
    ("local_batch", "expected"),
    [
        (1, {"num_warps": 4, "num_stages": 1}),
        (2, {"num_warps": 4, "num_stages": 1}),
        (4, {"num_warps": 2, "num_stages": 1}),
        (8, {"num_warps": 4, "num_stages": 2}),
        (28, {"num_warps": 4, "num_stages": 2}),
        (29, {"num_warps": 2, "num_stages": 1}),
        (32, {"num_warps": 2, "num_stages": 1}),
        (33, {}),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_sm100f_nvfp4_launch_options(local_batch, expected, capability):
    if capability != (10, 3):
        expected = {}
    assert (
        _sm100f_nvfp4_launch_options(
            total_q=local_batch * 4,
            num_kv_heads=4,
            gqa_group_size=16,
            max_topk=16,
            decode_query_len=4,
            capability=capability,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("local_batch", "expected"),
    [
        (1, None),
        (2, 8),
        (4, 8),
        (5, 16),
        (6, 4),
        (8, 2),
        (10, 8),
        (11, 4),
        (12, 2),
        (13, 2),
        (14, 2),
        (15, None),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_sm100f_nvfp4_num_topk_chunks(local_batch, expected, capability):
    if capability != (10, 3):
        expected = None
    assert (
        _sm100f_nvfp4_num_topk_chunks(
            total_q=local_batch * 4,
            num_kv_heads=4,
            gqa_group_size=16,
            max_topk=16,
            decode_query_len=4,
            capability=capability,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("local_batch", "expected"),
    [(10, 1), (11, 2), (12, 2), (14, 2), (15, 1)],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_sm100f_nvfp4_query_group_size_is_narrowly_scoped(local_batch, expected, capability):
    if capability != (10, 3):
        expected = 1
    common = dict(
        total_q=local_batch * 4,
        num_kv_heads=4,
        gqa_group_size=16,
        max_topk=16,
        decode_query_len=4,
    )
    assert _sm100f_nvfp4_query_group_size(**common, capability=capability) == expected
    assert _sm100f_nvfp4_query_group_size(**common, capability=(12, 0)) == 1
    assert (
        _sm100f_nvfp4_query_group_size(**(common | {"num_kv_heads": 2}), capability=capability) == 1
    )


def test_sm100f_nvfp4_launch_options_rejects_unmeasured_shapes():
    common = dict(
        total_q=32,
        num_kv_heads=4,
        gqa_group_size=16,
        max_topk=16,
        decode_query_len=4,
    )
    assert _sm100f_nvfp4_launch_options(**common, capability=(12, 0)) == {}
    assert _sm100f_nvfp4_launch_options(**(common | {"num_kv_heads": 2}), capability=(10, 3)) == {}
    assert (
        _sm100f_nvfp4_launch_options(**(common | {"decode_query_len": 1}), capability=(10, 3)) == {}
    )


@pytest.mark.parametrize(
    ("num_kv_heads", "local_batch", "expected"),
    [
        (1, 1, False),
        (1, 2, True),
        (1, 16, True),
        (2, 1, True),
        (2, 16, True),
        (4, 1, True),
        (4, 14, True),
        (8, 14, False),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_sm100f_nvfp4_linear_softmax_policy(num_kv_heads, local_batch, expected, capability):
    if capability != (10, 3):
        expected = False
    common = dict(
        total_q=local_batch * 4,
        num_kv_heads=num_kv_heads,
        gqa_group_size=16,
        max_topk=16,
        decode_query_len=4,
    )
    assert _sm100f_nvfp4_use_linear_softmax(**common, capability=capability) is expected
    assert not _sm100f_nvfp4_use_linear_softmax(**common, capability=(12, 0))


@pytest.mark.parametrize(
    ("local_batch", "expected"),
    [
        (1, {}),
        (2, {"num_warps": 1}),
        (3, {}),
        (4, {"num_warps": 1}),
        (5, {"num_warps": 1}),
        (6, {"num_warps": 1}),
        (7, {"num_warps": 1}),
        (8, {}),
        (10, {"num_warps": 1}),
        (11, {"num_warps": 1}),
        (12, {}),
    ],
)
@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_sm100f_nvfp4_merge_launch_options_is_narrowly_scoped(local_batch, expected, capability):
    if capability != (10, 3):
        expected = {}
    common = dict(
        num_kv_heads=4,
        gqa_group_size=16,
        max_topk=16,
        decode_query_len=4,
    )
    assert (
        _sm100f_nvfp4_merge_launch_options(total_q=local_batch * 4, **common, capability=capability)
        == expected
    )
    assert (
        _sm100f_nvfp4_merge_launch_options(total_q=local_batch * 4, **common, capability=(12, 0))
        == {}
    )


@skip_not_sm100
def test_msa_jit_cache_tracks_sources_and_build_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    require_msa_module()
    # MSA is optional outside supported GPU test environments; import after its
    # required-package check rather than during collection on other platforms.
    from fmha_sm100 import jit

    package = tmp_path / "package"
    csrc = package / "csrc"
    csrc.mkdir(parents=True)
    header = csrc / "params.h"
    header.write_text("int field;\n")
    template = csrc / "inst.jinja"
    template.write_text("{{ dtype }}\n")
    build_script = package / "jit.py"
    build_script.write_text("compiler_flags = '-std=c++20'\n")
    monkeypatch.setattr(jit, "_PACKAGE_DIR", package)
    monkeypatch.setattr(jit, "_FMHA_VARLEN_DIR", csrc)
    for name in ("GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK"):
        monkeypatch.delenv(name, raising=False)
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(cache_root))
    with caplog.at_level(logging.INFO, logger=jit.__name__):
        original = jit._compute_cache_base()
    assert original.parent == cache_root
    assert original == jit._compute_cache_base()
    assert str(original) in caplog.text
    assert "require recompilation" in caplog.text

    for source in (header, template, build_script):
        content = source.read_bytes()
        source.write_bytes(content + b"// changed\n")
        assert jit._compute_cache_base() != original
        source.write_bytes(content)
        assert jit._compute_cache_base() == original
    renamed = header.with_name("renamed.h")
    header.rename(renamed)
    assert jit._compute_cache_base() != original
    renamed.rename(header)

    # Presence enables these flags even when the environment value is empty.
    for name in ("GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK"):
        monkeypatch.setenv(name, "")
        assert jit._compute_cache_base() != original
        monkeypatch.delenv(name)
    assert jit._compute_cache_base() == original
    monkeypatch.delenv("MINFER_FMHA_CACHE_DIR")
    monkeypatch.setattr(
        jit.os.path, "expanduser", lambda _: str(tmp_path / ".cache/minfer/fmha_sm100")
    )
    default = jit._compute_cache_base()
    assert default.parent == tmp_path / ".cache/minfer/fmha_sm100"
    assert default.name == original.name
    monkeypatch.setenv("FMHA_GMEM_CHECK", "1")
    checked = jit._compute_cache_base()
    assert checked.parent == tmp_path / ".cache/minfer/fmha_sm100_gmem_check"
    assert checked.name != default.name
