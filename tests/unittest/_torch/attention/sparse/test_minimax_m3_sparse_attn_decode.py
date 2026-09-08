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

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_utils import (
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    msa_package_available,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.triton_sparse_decode import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
    resolve_num_topk_chunks,
)
from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = SPARSE_BLOCK_SIZE
HEAD_DIM = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="fmha_sm100 A/B comparison requires SM100/SM103.",
)


def _flat_page_table(block_table: torch.Tensor, kv_lens_cpu: torch.Tensor) -> torch.Tensor:
    """Flatten a block table into the per-request page ids fmha_sm100 consumes.

    build_kv_page_indices reads those ids out of a token-level slot map, so this
    rebuilds the map the block table implies and lets the production helper do
    the flattening.
    """
    batch, max_pages = block_table.shape
    intra = torch.arange(PAGE_SIZE, dtype=torch.int32)
    req_to_token = (block_table.cpu().to(torch.int32) * PAGE_SIZE).unsqueeze(2) + intra
    return build_kv_page_indices(
        req_to_token.reshape(batch, max_pages * PAGE_SIZE),
        torch.arange(batch, dtype=torch.int32),
        kv_lens_cpu,
        PAGE_SIZE,
    )


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


def test_sparse_decode_cuda_graph_replay_tracks_inputs():
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
            num_topk_chunks=4,
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
@pytest.mark.skipif(not msa_package_available(), reason="fmha_sm100 (MSA submodule) required")
def test_sparse_decode_matches_msa_kernel():
    """A/B against the fmha_sm100 sparse GQA path this kernel replaces."""
    from tensorrt_llm._torch.attention.backends.fmha.msa_sparse_gqa import run_msa_sparse_gqa

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
