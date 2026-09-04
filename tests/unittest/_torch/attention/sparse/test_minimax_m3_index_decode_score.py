# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# PyTorch oracle vendored from vLLM (Apache-2.0), _reference_decode_index_score:
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L188
"""Correctness tests for the CuTe DSL MiniMax-M3 indexer decode scorer.

The PyTorch oracle is ported from the vLLM reference linked in the file header
(v0.26.1rc0-77-g6f91edf96).
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import _cutedsl_score
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3_kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    msa_package_available,
    select_blocks_from_maxscore,
)
from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = 128
HEAD_DIM = 128

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="CuTe DSL MiniMax-M3 index decode scoring requires SM100/SM103.",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _flat_page_table(block_table: torch.Tensor, kv_lens_cpu: torch.Tensor) -> torch.Tensor:
    """Flatten a block table into the per-request page ids fmha_sm100 consumes."""
    return build_kv_page_indices(block_table.cpu().to(torch.int32), kv_lens_cpu, PAGE_SIZE)


def _runner():
    """Return the CuTe DSL runner class, skipping the test if unavailable."""
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    runner = getattr(cute_dsl_custom_ops, "CuteDSLMiniMaxM3IndexDecodeScoreRunner", None)
    if runner is None:
        pytest.skip("CuTe DSL custom ops are not registered in this build.")
    return runner


def _reference_decode_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    score_block_stride: int,
) -> torch.Tensor:
    """Per-block causal max of Q.K, in the kernel's [head, token, block] layout."""
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, score_block_stride),
        -float("inf"),
        device=idx_q.device,
        dtype=torch.float32,
    )
    for req_id, seq_len in enumerate(seq_lens.tolist()):
        num_blocks = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        if num_blocks == 0:
            continue
        token_start = req_id * decode_query_len
        q = idx_q[token_start : token_start + decode_query_len].float()
        pages = block_table[req_id, :num_blocks]
        k = index_kv_cache[pages].reshape(num_blocks * PAGE_SIZE, -1).float()
        score = torch.einsum("qhd,kd->hqk", q, k)
        q_pos = seq_len - decode_query_len + torch.arange(decode_query_len, device=idx_q.device)
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        out[:, token_start : token_start + decode_query_len, :num_blocks] = (
            score.reshape(num_idx_heads, decode_query_len, num_blocks, PAGE_SIZE).max(dim=3).values
        )
    return out


def _make_inputs(seq_lens, *, dtype, num_heads, decode_query_len, seed=0):
    """Build the kernel inputs, its score buffer and the expected scores.

    Returns (idx_q, index_k_cache, block_table, seq_lens_dev, backing, score,
    expected), where score is the [head, token, block] view the kernel writes
    and backing is the [head, block, token] buffer behind it.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens_dev = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    batch = len(seq_lens)
    total_q = batch * decode_query_len
    max_blocks = (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    # Mirror the plan's max_k_tiles alignment so the tests exercise a score
    # buffer wider than any single request's block count.
    score_block_stride = ((max_blocks + 15) // 16) * 16
    num_pages = batch * max_blocks

    # Shuffled pages: a kernel that ignored the block table and walked pages
    # linearly would still pass with an identity mapping.
    block_table = torch.randperm(num_pages, device="cuda", generator=generator).to(torch.int32)
    block_table = block_table.reshape(batch, max_blocks)

    idx_q = torch.randn(total_q, num_heads, HEAD_DIM, device="cuda", generator=generator).to(dtype)
    index_k_cache = torch.randn(
        num_pages, PAGE_SIZE, HEAD_DIM, device="cuda", generator=generator
    ).to(dtype)

    # Production hands the kernel a transposed view of the selector's
    # [heads, blocks, tokens] buffer, so build it the same way here.
    backing = torch.full(
        (num_heads, score_block_stride, total_q),
        -float("inf"),
        device="cuda",
        dtype=torch.float32,
    )
    score = backing.transpose(1, 2)

    expected = _reference_decode_index_score(
        idx_q, index_k_cache, block_table, seq_lens_dev, decode_query_len, score_block_stride
    )
    return idx_q, index_k_cache, block_table, seq_lens_dev, backing, score, expected


def _assert_scores_close(actual, expected):
    """Compare fp32 scores.

    Both sides accumulate 128 exactly-representable products in fp32, but the
    kernel's k-split order differs from the reference matmul's, so a few ULP of
    drift is expected.
    """
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("num_heads", "decode_query_len"),
    [(1, 1), (1, 3), (4, 1), (4, 8)],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        # Exactly one block, and one block plus a partial.
        [128],
        [1, 128, 129],
        # Non-multiples of the page size, several requests.
        [1025, 4097],
        [300, 1500, 2049, 4096, 5000, 7777, 8192, 9001],
    ],
    ids=["one-block", "short-mixed", "two-req", "batch8"],
)
def test_index_decode_score_matches_reference(dtype, num_heads, decode_query_len, seq_lens):
    """Per-block scores must match the oracle over the shapes decode produces."""
    if num_heads * decode_query_len > 32:
        pytest.skip("BLOCK_Q must not exceed 32.")
    # A request cannot have fewer KV positions than it has query tokens.
    seq_lens = [max(s, decode_query_len) for s in seq_lens]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens, dtype=dtype, num_heads=num_heads, decode_query_len=decode_query_len
    )

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, decode_query_len
    )
    _assert_scores_close(score, expected)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_index_decode_score_multi_round_split_k(dtype):
    """A request longer than split_k pages makes the per-CTA block loop iterate.

    split_k is 256, so 33000 tokens is 258 blocks and CTAs 0 and 1 each handle
    two blocks while the rest handle one.
    """
    seq_lens = [33000, 32768 + 1]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=3
    )
    assert (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE > 256

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1
    )
    _assert_scores_close(score, expected)


@skip_not_sm100
def test_index_decode_score_transposed_view_matches_contiguous():
    """The transposed selector view must produce the same values as a direct write.

    This is the zero-copy trick the indexer relies on: the kernel writes
    [head, token, block] into a buffer that is contiguous as
    [head, block, token], so the selector reads it without a transpose.
    """
    seq_lens = [1025, 4097, 300]
    idx_q, k_cache, block_table, seq_lens_dev, backing, transposed, expected = _make_inputs(
        seq_lens, dtype=torch.bfloat16, num_heads=4, decode_query_len=1, seed=11
    )
    contiguous = torch.full_like(expected, -float("inf"))

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, transposed, 1
    )
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, contiguous, 1
    )

    assert backing.is_contiguous()
    assert not transposed.is_contiguous()
    assert torch.equal(transposed, contiguous)
    _assert_scores_close(contiguous, expected)


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_index_decode_score_feeds_selector(dtype):
    """The scores must select the same blocks as the oracle's would.

    The selector reads the contiguous backing rather than the view the kernel
    writes, so this is where a disagreement between the two layouts would turn
    into a wrong block choice.
    """
    seq_lens = [129, 1025, 4097, 8192]
    idx_q, k_cache, block_table, seq_lens_dev, backing, score, expected = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=5
    )
    n_valid = torch.tensor(
        [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens], device="cuda", dtype=torch.int32
    )

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1
    )

    actual_topk = select_blocks_from_maxscore(
        backing, topk=MSA_REQUIRED_TOPK, n_valid_blocks=n_valid, init_blocks=0, local_blocks=1
    )
    expected_topk = select_blocks_from_maxscore(
        expected.transpose(1, 2).contiguous(),
        topk=MSA_REQUIRED_TOPK,
        n_valid_blocks=n_valid,
        init_blocks=0,
        local_blocks=1,
    )
    assert torch.equal(actual_topk, expected_topk)


@skip_not_sm100
@pytest.mark.skipif(not msa_package_available(), reason="fmha_sm100 (MSA submodule) required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_index_decode_score_matches_msa_proxy(dtype):
    """A/B against the fmha_sm100 proxy pass the CuTe DSL scorer replaces.

    Both sides report the per-block max of raw Q.K: the proxy's max_score is
    read off the MMA accumulator before the softmax scale, which in
    output_maxscore mode is never applied, so the values compare directly.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import _proxy_max_score

    seq_lens = [1025, 4097, 300]
    idx_q, k_cache, block_table, seq_lens_dev, backing, score, _ = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=17
    )
    sm_scale = HEAD_DIM**-0.5
    batch = len(seq_lens)
    n_valid_list = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    n_valid = torch.tensor(n_valid_list, device="cuda", dtype=torch.int32)

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1
    )

    qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
    kv_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    kv_indices = _flat_page_table(block_table, kv_lens_cpu).cuda()
    msa_score = _proxy_max_score(
        idx_q,
        k_cache.unsqueeze(1),
        qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=kv_lens_cpu - qo_lens_cpu,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        causal=True,
    )

    # The proxy plan sizes its block dim independently of the score buffer, and
    # entries past a token's valid count are undefined on both sides.
    width = min(backing.shape[1], msa_score.shape[1])
    for token, num_valid in enumerate(n_valid_list):
        assert num_valid <= width
        torch.testing.assert_close(
            backing[:, :num_valid, token],
            msa_score[:, :num_valid, token],
            rtol=2e-2,
            atol=2e-2,
        )

    kwargs = dict(topk=MSA_REQUIRED_TOPK, n_valid_blocks=n_valid, init_blocks=0, local_blocks=1)
    assert torch.equal(
        select_blocks_from_maxscore(backing[:, :width].contiguous(), **kwargs),
        select_blocks_from_maxscore(msa_score[:, :width].contiguous(), **kwargs),
    )


@skip_not_sm100
def test_index_decode_score_cuda_graph_replay_tracks_inputs():
    """Replay must recompute from the live buffers, not reuse captured values."""
    seq_lens = [1025, 4097]
    idx_q, k_cache, block_table, seq_lens_dev, backing, score, _ = _make_inputs(
        seq_lens, dtype=torch.bfloat16, num_heads=1, decode_query_len=1, seed=23
    )

    def run():
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q, k_cache, block_table, seq_lens_dev, score, 1
        )

    # Warm up outside capture so the JIT compile never lands inside it.
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Shrink one request and shuffle its pages; the replay must track both.
    new_seq_lens = [513, 4097]
    seq_lens_dev.copy_(torch.tensor(new_seq_lens, device="cuda", dtype=torch.int32))
    block_table.copy_(block_table.flip(1))
    backing.fill_(-float("inf"))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_decode_index_score(
        idx_q,
        k_cache,
        block_table,
        torch.tensor(new_seq_lens, device="cuda", dtype=torch.int32),
        1,
        backing.shape[1],
    )
    _assert_scores_close(score, expected)


@skip_not_sm100
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"num_heads": 8, "max_decode_query_len": 8}, "BLOCK_Q above 32"),
        ({"page_size": 64}, "unsupported page size"),
        ({"head_dim": 64}, "unsupported head dim"),
        ({"q_dtype": torch.float16}, "unsupported dtype"),
    ],
)
def test_index_decode_score_declines_unsupported_geometry(kwargs, reason):
    """The caller picks the scorer off is_supported, so it has to be exact."""
    runner = _runner()
    supported = dict(
        q_dtype=torch.bfloat16,
        num_heads=1,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        max_decode_query_len=1,
    )
    assert runner.is_supported(**supported), "baseline geometry must be supported"
    assert not runner.is_supported(**{**supported, **kwargs}), reason


@skip_not_sm100
def test_cutedsl_score_helper_falls_back_on_unsupported_geometry():
    """_cutedsl_score must report False rather than raise, so the caller can
    fall back to the fmha_sm100 proxy."""
    _runner()
    batch, num_heads, dql = 2, 1, 1
    idx_q = torch.randn(batch * dql, num_heads, 64, device="cuda", dtype=torch.bfloat16)
    k_paged = torch.randn(4, 1, PAGE_SIZE, 64, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros(batch, 2, device="cuda", dtype=torch.int32)
    seq_lens = torch.full((batch,), 128, device="cuda", dtype=torch.int32)
    max_score = torch.zeros(num_heads, 2, batch * dql, device="cuda").transpose(1, 2)

    assert not _cutedsl_score(
        idx_q,
        k_paged,
        max_score,
        block_table=block_table,
        seq_lens_cuda=seq_lens,
        decode_query_len=dql,
    )
