# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused MiniMax-M3 MSA block selector."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import _INIT_SCORE, _LOCAL_SCORE
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    select_blocks_from_maxscore,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_select_blocks(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    scores = max_score_kv.permute(2, 0, 1).to(torch.float32).clone()
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.long)
    nvb = n_valid_blocks.to(device=device, dtype=torch.long)

    if init_blocks > 0:
        init_mask = block_ids.view(1, 1, -1) < init_blocks
        scores = torch.where(init_mask, torch.full_like(scores, _INIT_SCORE), scores)
    if local_blocks > 0:
        local_start = (nvb - local_blocks).clamp_min(0)
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < nvb.view(-1, 1)
        )
        scores = torch.where(local_mask.unsqueeze(1), torch.full_like(scores, _LOCAL_SCORE), scores)
    block_valid = block_ids.view(1, -1) < nvb.view(-1, 1)
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    k = min(topk, n_blocks)
    vals, idx = scores.topk(k=k, dim=-1)
    idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
    sort_key = torch.where(idx < 0, torch.full_like(idx, n_blocks), idx)
    sort_key, _ = torch.sort(sort_key, dim=-1)
    idx = torch.where(sort_key >= n_blocks, torch.full_like(sort_key, -1), sort_key)
    if k < topk:
        pad = torch.full(
            (total_q, num_kv_heads, topk - k),
            -1,
            dtype=idx.dtype,
            device=device,
        )
        idx = torch.cat([idx, pad], dim=-1)
    return idx.to(torch.int32)


# Rows wider than this go to the histogram select instead of the register-
# resident bitonic sorts; see kSmallMaxBlocks in
# cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu.
HISTOGRAM_MIN_BLOCKS = 129


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize(
    "num_blocks",
    [1, 8, 16, 17, 32, 33, 64, 65, 96, 127, 128, 129, 1024, 1537, 4096, 8192],
)
def test_fused_selector_matches_reference_random(num_kv_heads, num_blocks):
    total_q = 19
    generator = torch.Generator(device="cuda").manual_seed(num_blocks)
    scores = torch.randn(
        num_kv_heads,
        num_blocks,
        total_q,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    n_valid_blocks = torch.randint(
        0,
        num_blocks + 1,
        (total_q,),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == (total_q, num_kv_heads, 16)
    assert actual.stride() == (num_kv_heads * 16, 16, 1)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_blocks", [65, 96, 128])
def test_fused_selector_128_path_matches_reference_ties_and_forcing(num_blocks):
    scores = torch.zeros((2, num_blocks, 5), device="cuda", dtype=torch.float32)
    n_valid_blocks = torch.tensor(
        [0, 8, 16, num_blocks - 1, num_blocks], device="cuda", dtype=torch.int32
    )

    expected = torch.tensor(
        [
            [-1] * 16,
            list(range(8)) + [-1] * 8,
            list(range(16)),
            list(range(8)) + list(range(num_blocks - 13, num_blocks - 5)),
            list(range(8)) + list(range(num_blocks - 12, num_blocks - 4)),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    expected = expected[:, None, :].expand(-1, scores.shape[0], -1)
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_blocks", [16, 48, 96, 129])
def test_fused_selector_head_major_output_is_zero_copy_q2k(num_blocks):
    total_q, num_kv_heads = 7, 4
    generator = torch.Generator(device="cuda").manual_seed(num_blocks)
    scores = torch.randn(
        num_kv_heads,
        num_blocks,
        total_q,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    n_valid_blocks = torch.randint(
        0,
        num_blocks + 1,
        (total_q,),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
        head_major_output=True,
    )
    q2k = actual.permute(1, 0, 2).contiguous().to(torch.int32)

    assert torch.equal(actual, expected)
    assert actual.shape == (total_q, num_kv_heads, 16)
    assert actual.stride() == (16, total_q * 16, 1)
    assert not actual.is_contiguous()
    assert q2k.shape == (num_kv_heads, total_q, 16)
    assert q2k.is_contiguous()
    assert q2k.data_ptr() == actual.data_ptr()
    assert q2k.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()


def test_fused_selector_head_major_output_through_msa_q2k_consumer():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")

    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
        msa_package_available,
    )

    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")

    from fmha_sm100.sparse_fmha_adapter import _convert_kv_block_indexes_to_q2k

    total_q, num_kv_heads, num_blocks = 7, 4, 96
    scores = torch.randn(num_kv_heads, num_blocks, total_q, device="cuda")
    n_valid_blocks = torch.arange(32, 32 + total_q, device="cuda", dtype=torch.int32)
    selected = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
        head_major_output=True,
    )
    q2k = _convert_kv_block_indexes_to_q2k(
        selected,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_kv_heads,
        qhead_per_kv=1,
    )

    assert q2k.shape == (num_kv_heads, total_q, 16)
    assert q2k.is_contiguous()
    assert q2k.data_ptr() == selected.data_ptr()
    assert q2k.untyped_storage().data_ptr() == selected.untyped_storage().data_ptr()
    assert torch.equal(q2k.permute(1, 0, 2), selected)


@pytest.mark.parametrize(
    ("init_blocks", "local_blocks"),
    [(0, 0), (0, 1), (2, 3), (16, 1), (20, 0), (0, 20)],
)
def test_fused_selector_matches_reference_forced_and_padded(init_blocks, local_blocks):
    scores = (
        torch.tensor(
            [
                [
                    float("-inf"),
                    4.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                    -1.0,
                    -2.0,
                    -3.0,
                    -4.0,
                    -5.0,
                    -6.0,
                    -7.0,
                    -8.0,
                    -9.0,
                    -10.0,
                    -11.0,
                    -12.0,
                    -13.0,
                    -14.0,
                ]
            ],
            device="cuda",
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .expand(-1, -1, 5)
    )
    n_valid_blocks = torch.tensor([0, 1, 7, 16, 20], dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("fill_value", [0.0, 1.0e30, 1.0e29])
def test_fused_selector_matches_reference_equal_score_ties(fill_value):
    scores = torch.full((2, 64, 3), fill_value, device="cuda", dtype=torch.float32)
    n_valid_blocks = torch.tensor([15, 32, 64], dtype=torch.int32)

    expected = torch.tensor(
        [
            list(range(15)) + [-1],
            list(range(16)),
            list(range(16)),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    expected = expected[:, None, :].expand(-1, scores.shape[0], -1)
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=20,
        local_blocks=0,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_matches_reference_nonfinite_and_validity_bounds():
    scores = (
        torch.tensor(
            [
                float("nan"),
                float("inf"),
                float("-inf"),
                -1.0,
                0.0,
                1.0,
                float("nan"),
                float("-inf"),
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            device="cuda",
            dtype=torch.float32,
        )
        .view(1, 20, 1)
        .expand(-1, -1, 4)
    )
    n_valid_blocks = torch.tensor([-3, 0, 17, 25], dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_supports_strided_scores_and_cuda_validity():
    generator = torch.Generator(device="cuda").manual_seed(7)
    backing = torch.randn(2, 73, 22, generator=generator, device="cuda")
    scores = backing[:, 1:72:2, ::2]
    assert not scores.is_contiguous()
    n_valid_blocks = torch.tensor(
        [0, 1, 3, 8, 15, 16, 17, 20, 30, 35, 36], device="cuda", dtype=torch.int32
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("scores_dtype", "expects float32 scores"),
        ("validity_dtype", "expects int32 n_valid_blocks"),
        ("validity_contiguous", "n_valid_blocks must be contiguous"),
        ("topk", "supports topk=16"),
        ("negative_init", "init_blocks must be non-negative"),
        ("negative_local", "local_blocks must be non-negative"),
        ("too_many_blocks", "supports at most 65535 blocks"),
    ],
)
def test_fused_selector_rejects_invalid_operator_contracts(case, match):
    scores = torch.randn(1, 32, 2, device="cuda")
    n_valid_blocks = torch.tensor([16, 32], device="cuda", dtype=torch.int32)
    topk, init_blocks, local_blocks = 16, 0, 1

    if case == "scores_dtype":
        scores = scores.to(torch.bfloat16)
    elif case == "validity_dtype":
        n_valid_blocks = n_valid_blocks.to(torch.int64)
    elif case == "validity_contiguous":
        n_valid_blocks = torch.tensor([16, 0, 32, 0], device="cuda", dtype=torch.int32)[::2]
    elif case == "topk":
        topk = 8
    elif case == "negative_init":
        init_blocks = -1
    elif case == "negative_local":
        local_blocks = -1
    elif case == "too_many_blocks":
        scores = torch.empty(1, 65_536, 2, device="cuda")

    with pytest.raises(RuntimeError, match=match):
        torch.ops.trtllm.minimax_m3_select_blocks(
            scores,
            n_valid_blocks,
            topk,
            init_blocks,
            local_blocks,
            False,
        )


@pytest.mark.parametrize("num_blocks", [HISTOGRAM_MIN_BLOCKS, 512, 3000])
def test_histogram_selector_matches_reference_ties_and_forcing(num_blocks):
    """All-equal scores on the histogram path.

    Every block lands in one histogram bin, so the whole top-k comes out of the
    boundary sort. That sort must break ties towards the smaller block index to
    match torch.topk, which is the part of the histogram select that does not
    come for free from the vendored algorithm.
    """
    scores = torch.zeros((2, num_blocks, 5), device="cuda", dtype=torch.float32)
    n_valid_blocks = torch.tensor(
        [0, 16, 17, num_blocks - 1, num_blocks], device="cuda", dtype=torch.int32
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=8,
        local_blocks=12,
    )

    assert torch.equal(actual, expected)


def test_histogram_selector_matches_reference_mixed_length_batch():
    """Per-row valid-block bounds, from a single block up to the full row.

    The histogram select reads each row through its own n_valid_blocks bound
    rather than a batch-wide one, so a batch whose rows span the trivial
    path, the boundary-sort path, and everything between must still agree.
    """
    num_kv_heads, num_blocks = 3, 2048
    generator = torch.Generator(device="cuda").manual_seed(11)
    n_valid_blocks = torch.tensor(
        [0, 1, 15, 16, 17, 63, 200, 1023, 2047, 2048, 4096, -5],
        device="cuda",
        dtype=torch.int32,
    )
    scores = torch.randn(
        num_kv_heads,
        num_blocks,
        n_valid_blocks.numel(),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )

    assert torch.equal(actual, expected)


def test_histogram_selector_matches_reference_fp16_indistinguishable_scores():
    """Scores that share one fp16 bin but are all distinct in fp32.

    Step 0 of the histogram select bins an fp16 cast of the score, so this row
    lands entirely in one bin, overflows the boundary staging buffer, and
    forces the kernel through the exact fp32 refinement steps. Consecutive fp32
    values keep every score distinct, which keeps the expected answer
    well-defined: more than 2048 bit-identical scores in one row is the one
    case the histogram select resolves arbitrarily.
    """
    num_kv_heads, num_blocks, total_q = 2, 3000, 4
    generator = torch.Generator(device="cuda").manual_seed(5)
    # 3000 consecutive fp32 values above 1.0 span ~3.6e-4, inside the ~9.8e-4
    # rounding interval of fp16's 1.0.
    ladder = 1.0 + torch.arange(num_blocks, device="cuda", dtype=torch.float32) * 2.0**-23
    noise = torch.rand(num_kv_heads, total_q, num_blocks, generator=generator, device="cuda")
    scores = ladder[noise.argsort(dim=-1)].permute(0, 2, 1).contiguous()
    assert scores.shape == (num_kv_heads, num_blocks, total_q)
    assert torch.unique(scores[0, :, 0]).numel() == num_blocks
    n_valid_blocks = torch.tensor([17, 1000, 2999, 3000], device="cuda", dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )

    assert torch.equal(actual, expected)


def test_histogram_selector_matches_reference_nonfinite():
    """NaN, +-inf and forced sentinels together on the histogram path.

    NaN must outrank +inf (torch.topk on CUDA orders it that way), the init and
    local sentinels must outrank ordinary scores, and query 1 has fewer than
    sixteen non--inf valid blocks so some selected slots must come back as -1
    rather than as a block index.
    """
    num_blocks, total_q = 300, 4
    generator = torch.Generator(device="cuda").manual_seed(3)
    scores = torch.randn(2, num_blocks, total_q, generator=generator, device="cuda")
    scores[:, 5, :] = float("nan")
    scores[:, 6, :] = float("inf")
    scores[:, 7, :] = float("-inf")
    # Query 1 keeps only ten finite blocks inside its 200 valid ones.
    scores[:, 10:, 1] = float("-inf")
    n_valid_blocks = torch.tensor([17, 200, 290, num_blocks], device="cuda", dtype=torch.int32)

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=3,
        local_blocks=4,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=3,
        local_blocks=4,
    )

    assert (expected[1] == -1).any(), "query 1 should exercise the -inf -> -1 rule"
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_nan", [16, 48])
def test_histogram_selector_matches_reference_repeated_nan(num_nan):
    """A row whose threshold bin holds more than one NaN.

    NaN outranks every finite score and +inf, so a row with at least topk NaNs
    makes the NaN bin the threshold bin and sends every NaN through the
    boundary sort. That sort therefore has to rank NaN against NaN, which
    comparing the scores as floats cannot do.
    """
    num_kv_heads, num_blocks, total_q = 2, 300, 3
    generator = torch.Generator(device="cuda").manual_seed(29)
    scores = torch.randn(num_kv_heads, num_blocks, total_q, generator=generator, device="cuda")
    nan_blocks = torch.arange(20, 20 + 2 * num_nan, 2, device="cuda")
    scores[:, nan_blocks, :] = float("nan")
    n_valid_blocks = torch.full((total_q,), num_blocks, device="cuda", dtype=torch.int32)

    kwargs = dict(topk=16, n_valid_blocks=n_valid_blocks, init_blocks=0, local_blocks=0)
    actual = select_blocks_from_maxscore(scores, **kwargs)

    # The NaNs are bit-identical, so the smaller-block-index tie-break decides.
    expected = nan_blocks[:16].to(torch.int32).view(1, 1, 16).expand(total_q, num_kv_heads, 16)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, _reference_select_blocks(scores, **kwargs))


@pytest.mark.parametrize("num_blocks", [HISTOGRAM_MIN_BLOCKS, 1023])
def test_histogram_selector_matches_reference_contiguous_single_query(num_blocks):
    """The float4 read path, which needs a unit block stride.

    A [num_kv_heads, n_blocks, total_q] score tensor has block stride total_q,
    so only a single-query batch reads through float4. An odd n_blocks with
    several KV heads misaligns the row base for some heads and leaves a
    remainder, so the scalar lead-in and the scalar tail run alongside the
    vector body.
    """
    num_kv_heads = 3
    generator = torch.Generator(device="cuda").manual_seed(23)
    scores = torch.randn(num_kv_heads, num_blocks, 1, generator=generator, device="cuda")
    assert scores.stride(1) == 1
    assert num_blocks % 4 != 0
    n_valid_blocks = torch.tensor([num_blocks], device="cuda", dtype=torch.int32)

    kwargs = dict(topk=16, n_valid_blocks=n_valid_blocks, init_blocks=2, local_blocks=3)
    assert torch.equal(
        select_blocks_from_maxscore(scores, **kwargs),
        _reference_select_blocks(scores, **kwargs),
    )


def test_histogram_selector_supports_strided_scores():
    """The histogram select reads through blockStride, with no transpose."""
    generator = torch.Generator(device="cuda").manual_seed(13)
    backing = torch.randn(2, 601, 22, generator=generator, device="cuda")
    scores = backing[:, 1:600:2, ::2]
    assert not scores.is_contiguous()
    assert scores.shape[1] >= HISTOGRAM_MIN_BLOCKS

    n_valid_blocks = torch.tensor(
        [0, 1, 16, 17, 100, 200, 299, 300, 400, 17, 33],
        device="cuda",
        dtype=torch.int32,
    )

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )
    actual = select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=2,
        local_blocks=3,
    )

    assert torch.equal(actual, expected)


def test_fused_selector_cuda_graph_replay_updates_inputs():
    scores = torch.randn(1, 64, 4, device="cuda")
    n_valid_blocks = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)

    for _ in range(3):
        output = select_blocks_from_maxscore(
            scores,
            topk=16,
            n_valid_blocks=n_valid_blocks,
            init_blocks=0,
            local_blocks=1,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = select_blocks_from_maxscore(
            scores,
            topk=16,
            n_valid_blocks=n_valid_blocks,
            init_blocks=0,
            local_blocks=1,
        )

    scores.copy_(torch.arange(64, device="cuda", dtype=torch.float32).view(1, 64, 1))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_select_blocks(
        scores,
        topk=16,
        n_valid_blocks=n_valid_blocks,
        init_blocks=0,
        local_blocks=1,
    )
    assert torch.equal(output, expected)
