# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise equality of the CuTe DSL context block selector with minimax_m3_select_blocks."""

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.minimax_m3_select_context_blocks import (
    minimax_m3_select_context_blocks,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="SM100 family required",
)

DATA_KINDS = ("randn", "ties", "equal", "neg_inf", "mostly_neg_inf", "pos_nan_inf")
N_VALID_KINDS = ("causal", "random", "short", "full")
# (init_blocks, local_blocks); the last overlaps on rows with fewer than 20 valid blocks.
FORCING = ((0, 0), (1, 2), (8, 12))


def _select(scores, n_valid, init_blocks, local_blocks):
    return minimax_m3_select_context_blocks(
        scores, n_valid, init_blocks=init_blocks, local_blocks=local_blocks
    )


def _make_scores(kind, shape, gen):
    x = torch.randn(shape, generator=gen, device="cuda")
    u = torch.rand(shape, generator=gen, device="cuda")
    if kind == "ties":
        x = torch.randint(0, 4, shape, generator=gen, device="cuda").float() * 0.5
    elif kind == "equal":
        # Every row has more than 64 keys at its threshold: the exact-extraction fallback.
        x = torch.full(shape, 0.25, device="cuda")
    elif kind == "neg_inf":
        x[u < 0.3] = float("-inf")
    elif kind == "mostly_neg_inf":
        # Many rows keep fewer than 16 finite blocks, so -1 padding follows.
        x[u < 0.97] = float("-inf")
    elif kind == "pos_nan_inf":
        x[u < 0.02] = float("nan")
        x[(u >= 0.02) & (u < 0.04)] = float("inf")
    return x


def _make_n_valid(kind, num_blocks, num_tokens, gen):
    if kind == "causal":
        # A chunk whose last token sees every block.
        pos = num_blocks * 128 - num_tokens + torch.arange(num_tokens, device="cuda")
        return ((pos + 128) // 128).clamp_min(0).to(torch.int32)
    if kind == "random":
        # Includes negative counts and counts past num_blocks.
        high = num_blocks + 6
        return torch.randint(-3, high, (num_tokens,), generator=gen, device="cuda").to(torch.int32)
    if kind == "short":
        return torch.randint(0, 40, (num_tokens,), generator=gen, device="cuda").to(torch.int32)
    return torch.full((num_tokens,), num_blocks, dtype=torch.int32, device="cuda")


def _layout(x, layout):
    if layout == "tok":
        # The indexer scorers' [heads, blocks, tokens] layout.
        return x
    if layout == "row":
        # Token-major storage viewed as [heads, blocks, tokens].
        return x.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    # Misaligned by one element, so 16-byte loads are off.
    buf = torch.empty(x.numel() + 1, device="cuda")
    view = buf[1:].view(x.shape)
    view.copy_(x)
    return view


def _assert_matches(scores, n_valid, init_blocks, local_blocks, tag):
    expected = torch.ops.trtllm.minimax_m3_select_blocks(
        scores, n_valid, 16, init_blocks, local_blocks, True
    )
    actual = _select(scores, n_valid, init_blocks, local_blocks)
    assert actual.shape == expected.shape and actual.stride() == expected.stride(), tag
    if not torch.equal(actual, expected):
        t, h = (actual != expected).any(-1).nonzero()[0].tolist()
        pytest.fail(
            f"{tag}: row (t={t}, h={h}) n_valid={int(n_valid[t])}\n"
            f"  expected {expected[t, h].tolist()}\n  actual   {actual[t, h].tolist()}"
        )


# 1-32 take the C++ op's small path, 33-128 its bitonic paths and the rest its histogram path.
# Here 2048 blocks is the widest row kept in registers, 4096 the widest not sent to the C++ op.
@pytest.mark.parametrize("layout", ["tok", "row", "offset"])
@pytest.mark.parametrize(
    "num_blocks", [1, 17, 32, 33, 100, 128, 129, 300, 600, 1000, 1088, 1200, 1600, 2048, 2500, 4096]
)
def test_matches_cpp_op(num_blocks, layout):
    gen = torch.Generator(device="cuda").manual_seed(num_blocks)
    # Past 2048 tied keys the C++ radix select fills a tie group in atomic order.
    kinds = DATA_KINDS if num_blocks <= 2048 else ("randn", "neg_inf", "pos_nan_inf")
    case = 0
    for kind in kinds:
        for n_valid_kind in N_VALID_KINDS:
            case += 1
            num_heads = (1, 2, 4)[case % 3]
            num_tokens = (64, 77)[(case // 3) % 2]  # 16-byte loads need a multiple of 4
            init_blocks, local_blocks = FORCING[(case // 2) % 3]
            x = _make_scores(kind, (num_heads, num_blocks, num_tokens), gen)
            n_valid = _make_n_valid(n_valid_kind, num_blocks, num_tokens, gen)
            _assert_matches(
                _layout(x, layout),
                n_valid,
                init_blocks,
                local_blocks,
                f"{kind} n_valid={n_valid_kind} H={num_heads} T={num_tokens} "
                f"forcing=({init_blocks}, {local_blocks})",
            )


@pytest.mark.parametrize("num_blocks", [32, 100, 1088, 2500])
def test_negative_nan_ranks_with_neg_inf(num_blocks):
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(2, num_blocks, 77, generator=gen, device="cuda")
    neg_nan = torch.tensor([0xFFC00000 - 2**32], dtype=torch.int32).view(torch.float32)
    x[torch.rand(x.shape, generator=gen, device="cuda") < 0.3] = neg_nan.item()
    n_valid = _make_n_valid("random", num_blocks, 77, gen)
    expected = torch.ops.trtllm.minimax_m3_select_blocks(
        torch.where(x.isnan(), float("-inf"), x), n_valid, 16, 1, 2, True
    )
    assert torch.equal(_select(x, n_valid, 1, 2), expected)


def test_edge_shapes():
    # Wider than 4096 blocks: handed to the C++ op.
    _assert_matches(
        torch.randn(1, 4097, 5, device="cuda"),
        torch.tensor([0, 17, 4000, 4097, 9000], dtype=torch.int32, device="cuda"),
        1,
        2,
        "4097 blocks",
    )
    empty = _select(
        torch.empty(4, 100, 0, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        1,
        2,
    )
    assert empty.shape == (0, 4, 16)


def test_context_rows_dispatch(monkeypatch):
    import tensorrt_llm._torch.attention.kernels.blackwell.minimax_m3_select_context_blocks as kernel_module
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        select_blocks_from_maxscore,
    )

    calls, real = [], kernel_module.minimax_m3_select_context_blocks

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "minimax_m3_select_context_blocks", spy)
    scores = torch.randn(4, 1088, 37, device="cuda")
    n_valid = torch.arange(1050, 1087, device="cuda")  # int64, as the CPU fallback produces
    kwargs = dict(topk=16, n_valid_blocks=n_valid, init_blocks=1, local_blocks=2)
    actual = select_blocks_from_maxscore(
        scores, **kwargs, head_major_output=True, context_rows=True
    )
    assert calls == [1]
    expected = select_blocks_from_maxscore(scores, **kwargs, head_major_output=True)
    assert calls == [1]
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)
