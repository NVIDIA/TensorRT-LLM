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

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.index_k_nvfp4 import (
    INDEX_K_NVFP4_PAGE_BYTES,
    INDEX_K_NVFP4_SCALE_OFFSET,
    dequantize_index_k_nvfp4_reference,
    quantize_index_k_nvfp4_reference,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    msa_package_available,
    select_blocks_from_maxscore,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import _cutedsl_score
from tensorrt_llm._utils import get_sm_version

PAGE_SIZE = 128
HEAD_DIM = 128

# (Q dtype, K format) pairs the scorer serves.
FORMATS = [
    pytest.param(torch.bfloat16, None, id="bf16"),
    pytest.param(torch.float8_e4m3fn, None, id="fp8"),
    pytest.param(torch.float8_e4m3fn, "nvfp4", id="nvfp4"),
]

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="CuTe DSL MiniMax-M3 index decode scoring requires SM100/SM103.",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _flat_page_table(block_table: torch.Tensor, kv_lens_cpu: torch.Tensor) -> torch.Tensor:
    """Flatten a block table into the per-request page ids fmha_sm100 consumes.

    The production helper concatenates the valid prefix of each request's
    block-id row according to its KV length.
    """
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


def _make_nvfp4_cache(num_pages, generator, page_stride=INDEX_K_NVFP4_PAGE_BYTES):
    """Quantize random rows into a uint8 [num_pages, page_stride] NVFP4 index-K cache."""
    rows = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM, device="cuda", generator=generator)
    data, scales = quantize_index_k_nvfp4_reference(rows)
    cache = torch.zeros(num_pages, page_stride, device="cuda", dtype=torch.uint8)
    cache[:, :INDEX_K_NVFP4_SCALE_OFFSET] = data.reshape(num_pages, -1)
    cache[:, INDEX_K_NVFP4_SCALE_OFFSET:INDEX_K_NVFP4_PAGE_BYTES] = scales.reshape(num_pages, -1)
    return cache


def _dequantize_nvfp4_cache(cache):
    """Exact FP32 values [num_pages, PAGE_SIZE, HEAD_DIM] of an NVFP4 index-K cache."""
    num_pages = cache.shape[0]
    data = cache[:, :INDEX_K_NVFP4_SCALE_OFFSET].reshape(num_pages, PAGE_SIZE, -1)
    scales = cache[:, INDEX_K_NVFP4_SCALE_OFFSET:INDEX_K_NVFP4_PAGE_BYTES]
    return dequantize_index_k_nvfp4_reference(data, scales.reshape(num_pages, PAGE_SIZE, -1))


def _make_inputs(
    seq_lens,
    *,
    dtype,
    num_heads,
    decode_query_len,
    seed=0,
    k_format=None,
    nvfp4_page_stride=INDEX_K_NVFP4_PAGE_BYTES,
):
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
    if k_format == "nvfp4":
        index_k_cache = _make_nvfp4_cache(num_pages, generator, nvfp4_page_stride)
        k_values = _dequantize_nvfp4_cache(index_k_cache)
    else:
        index_k_cache = torch.randn(
            num_pages, PAGE_SIZE, HEAD_DIM, device="cuda", generator=generator
        ).to(dtype)
        k_values = index_k_cache

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
        idx_q, k_values, block_table, seq_lens_dev, decode_query_len, score_block_stride
    )
    return idx_q, index_k_cache, block_table, seq_lens_dev, backing, score, expected


def _assert_scores_close(actual, expected):
    """Compare fp32 scores.

    Both sides accumulate 128 exactly-representable products in fp32, but the
    kernel's k-split order differs from the reference matmul's, so a few ULP of
    drift is expected.
    """
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


SEQ_LENS = [
    # Exactly one block, and one block plus a partial.
    pytest.param([128], id="one-block"),
    pytest.param([1, 128, 129], id="short-mixed"),
    # Non-multiples of the page size, several requests.
    pytest.param([1025, 4097], id="two-req"),
    pytest.param([300, 1500, 2049, 4096, 5000, 7777, 8192, 9001], id="batch8"),
]


@skip_not_sm100
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("num_heads", "decode_query_len"),
    [(1, 1), (1, 3), (4, 1), (4, 8)],
)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
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
@pytest.mark.parametrize(
    ("num_heads", "decode_query_len"),
    [(1, 1), (1, 3), (2, 2), (2, 4), (4, 1), (8, 4)],
)
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
def test_index_decode_score_nvfp4_matches_reference(num_heads, decode_query_len, seq_lens):
    """NVFP4 K must score as FP32 q . dequant(k) over the shapes decode produces."""
    seq_lens = [max(s, decode_query_len) for s in seq_lens]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens,
        dtype=torch.float8_e4m3fn,
        num_heads=num_heads,
        decode_query_len=decode_query_len,
        k_format="nvfp4",
    )

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, decode_query_len, "nvfp4"
    )
    _assert_scores_close(score, expected)


@skip_not_sm100
def test_index_decode_score_nvfp4_long_context_strided_pool():
    """Several blocks per CTA, so the two stages wrap, read from a pool whose
    page stride exceeds the 9216-byte page."""
    seq_lens = [131072 + 5, 70000, 4097]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens,
        dtype=torch.float8_e4m3fn,
        num_heads=2,
        decode_query_len=4,
        seed=7,
        k_format="nvfp4",
        nvfp4_page_stride=INDEX_K_NVFP4_PAGE_BYTES + 512,
    )
    assert (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE > 4 * 256

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 4, "nvfp4"
    )
    _assert_scores_close(score, expected)


@skip_not_sm100
def test_index_decode_score_nvfp4_zeroed_pages_score_zero():
    """A zeroed page (code 0, scale 0) dequantizes to zeros, never NaN."""
    seq_lens = [300, 129]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens, dtype=torch.float8_e4m3fn, num_heads=4, decode_query_len=1, k_format="nvfp4"
    )
    k_cache.zero_()

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1, "nvfp4"
    )
    # With one query per request every written block has a visible key.
    torch.testing.assert_close(score, torch.where(expected.isinf(), expected, 0.0))


@skip_not_sm100
def test_index_decode_score_nvfp4_rejects_short_pages():
    """A cache row shorter than a page would let the kernel read past the pool."""
    _runner()
    idx_q, k_cache, block_table, seq_lens_dev, _, score, _ = _make_inputs(
        [256], dtype=torch.float8_e4m3fn, num_heads=1, decode_query_len=1, k_format="nvfp4"
    )
    with pytest.raises(ValueError, match="NVFP4"):
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q,
            k_cache[:, :INDEX_K_NVFP4_SCALE_OFFSET],
            block_table,
            seq_lens_dev,
            score,
            1,
            "nvfp4",
        )


@skip_not_sm100
@pytest.mark.parametrize(("dtype", "k_format"), FORMATS)
@pytest.mark.parametrize("decode_query_len", [1, 4])
def test_index_decode_score_masks_keys_after_each_query(dtype, k_format, decode_query_len):
    """No key after a query's position may reach that query's block maximum.

    Q selects dim 0, along which every key after the first query outscores all
    earlier keys and the unwritten tail of the last page outscores everything,
    so a leaked key changes the maximum. The lengths put the first query at
    each edge of the 32-key warp slices and the 128-key blocks.
    """
    seq_lens = [130, 131, 161, 163, 191, 192, 255, 256, 4098, 4099]
    generator = torch.Generator(device="cuda").manual_seed(29)
    batch, num_heads = len(seq_lens), 2
    max_blocks = (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE
    block_table = torch.randperm(batch * max_blocks, device="cuda", generator=generator)
    block_table = block_table.to(torch.int32).reshape(batch, max_blocks)
    seq_lens_dev = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)

    idx_q = torch.zeros(batch * decode_query_len, num_heads, HEAD_DIM, device="cuda")
    idx_q[..., 0] = 1.0
    idx_q = idx_q.to(dtype)
    keys = torch.rand(batch, max_blocks * PAGE_SIZE, HEAD_DIM, device="cuda", generator=generator)
    keys = keys * 2 - 1
    for req, seq_len in enumerate(seq_lens):
        q_start = seq_len - decode_query_len
        keys[req, q_start + 1 : seq_len, 0] = torch.tensor(
            [2.0 ** (2 + j) for j in range(1, decode_query_len)], device="cuda"
        )
        keys[req, seq_len:, 0] = 64.0
    pages = torch.empty(batch * max_blocks, PAGE_SIZE, HEAD_DIM, device="cuda")
    pages[block_table.flatten().long()] = keys.reshape(batch * max_blocks, PAGE_SIZE, HEAD_DIM)
    if k_format == "nvfp4":
        data, scales = quantize_index_k_nvfp4_reference(pages)
        k_cache = torch.cat(
            (data.reshape(len(pages), -1), scales.reshape(len(pages), -1)), dim=1
        ).contiguous()
        k_values = _dequantize_nvfp4_cache(k_cache)
    else:
        k_cache = pages.to(dtype)
        k_values = k_cache

    width = ((max_blocks + 15) // 16) * 16
    backing = torch.full((num_heads, width, batch * decode_query_len), -float("inf"), device="cuda")
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q,
        k_cache,
        block_table,
        seq_lens_dev,
        backing.transpose(1, 2),
        decode_query_len,
        k_format,
    )
    expected = _reference_decode_index_score(
        idx_q, k_values, block_table, seq_lens_dev, decode_query_len, width
    )
    _assert_scores_close(backing.transpose(1, 2), expected)


@skip_not_sm100
@pytest.mark.parametrize(("dtype", "k_format"), FORMATS)
def test_index_decode_score_multi_round_split_k(dtype, k_format):
    """A request longer than split_k pages makes the per-CTA block loop iterate.

    split_k is 256, so 33000 tokens is 258 blocks and CTAs 0 and 1 each handle
    two blocks while the rest handle one.
    """
    seq_lens = [33000, 32768 + 1]
    idx_q, k_cache, block_table, seq_lens_dev, _, score, expected = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=3, k_format=k_format
    )
    assert (max(seq_lens) + PAGE_SIZE - 1) // PAGE_SIZE > 256

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1, k_format
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
@pytest.mark.parametrize(("dtype", "k_format"), FORMATS)
def test_index_decode_score_feeds_selector(dtype, k_format):
    """The scores must select the same blocks as the oracle's would.

    The selector reads the contiguous backing rather than the view the kernel
    writes, so this is where a disagreement between the two layouts would turn
    into a wrong block choice.
    """
    seq_lens = [129, 1025, 4097, 8192]
    idx_q, k_cache, block_table, seq_lens_dev, backing, score, expected = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=5, k_format=k_format
    )
    n_valid = torch.tensor(
        [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens], device="cuda", dtype=torch.int32
    )

    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, k_cache, block_table, seq_lens_dev, score, 1, k_format
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
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
        _proxy_max_score,
    )

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
@pytest.mark.parametrize(("dtype", "k_format"), FORMATS)
def test_index_decode_score_cuda_graph_replay_tracks_inputs(dtype, k_format):
    """Replay must recompute from the live buffers, not reuse captured values."""
    seq_lens = [1025, 4097]
    idx_q, k_cache, block_table, seq_lens_dev, backing, score, _ = _make_inputs(
        seq_lens, dtype=dtype, num_heads=1, decode_query_len=1, seed=23, k_format=k_format
    )

    def run():
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q, k_cache, block_table, seq_lens_dev, score, 1, k_format
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
        _dequantize_nvfp4_cache(k_cache) if k_format == "nvfp4" else k_cache,
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
        ({"k_format": "nvfp4"}, "NVFP4 K needs FP8 Q"),
        ({"k_format": "mxfp4"}, "unsupported K format"),
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
    assert runner.is_supported(**{**supported, "q_dtype": torch.float8_e4m3fn, "k_format": "nvfp4"})
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
