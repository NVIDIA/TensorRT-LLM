# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fused per-expert post-norm combine against the reference composition.

On CUDA `post_expert_norm_combine` is one kernel. These check it against the
op-by-op composition at the checkpoint's width and at a narrow one, and pin what
only the kernel can get wrong: a row behind an invalid slot must never be read,
because the MoE kernel leaves it unwritten and a NaN there would reach the sum,
and the kernel must replay from a CUDA graph.
"""

import pytest
import torch

from tensorrt_llm._torch.models import musespark_open_kernels
from tensorrt_llm._torch.models.modeling_musespark_open import (
    post_expert_norm_combine,
    post_expert_norm_rows,
)

EPS = 1e-6
# The checkpoint's expert width and top-k.
HIDDEN = 4096
TOP_K = 8

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_combine(gemm2, index, routing, weight, eps, *, num_tokens, top_k, hidden_size):
    """The op-by-op composition the fused kernel stands for."""
    rows, _ = post_expert_norm_rows(
        gemm2, index, weight, eps, num_tokens=num_tokens, top_k=top_k, hidden_size=hidden_size
    )
    return (rows * routing.reshape(num_tokens, top_k, 1).to(torch.float32)).sum(dim=1)


def _case(
    num_tokens,
    *,
    hidden=HIDDEN,
    top_k=TOP_K,
    dtype=torch.bfloat16,
    invalid=(),
    pad=0,
    seed=0,
    device="cuda",
):
    """A finalize input with `invalid` flat slots set to -1 and NaN behind them.

    NaN is what the MoE kernel can leave in the rows of experts that are not
    local to this rank: it never writes them.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    slots = num_tokens * top_k
    gemm2 = torch.randn(slots, hidden + pad, generator=gen, device=device, dtype=torch.float32)
    index = torch.arange(slots, dtype=torch.int32, device=device)
    for slot in invalid:
        index[slot] = -1
        gemm2[slot].fill_(float("nan"))
    if pad:
        gemm2[:, hidden:].fill_(float("nan"))
    routing = torch.rand(num_tokens, top_k, generator=gen, device=device, dtype=torch.float32)
    weight = torch.randn(hidden, generator=gen, device=device, dtype=torch.float32) * 0.1
    return gemm2.to(dtype), index, routing, weight.to(dtype)


@cuda_only
@pytest.mark.parametrize("num_tokens", [1, 7, 256])
def test_fused_matches_the_reference_at_the_checkpoint_width(num_tokens):
    """The kernel sums in another order, far below the BF16 grain the result is cast to."""
    gemm2, index, routing, weight = _case(num_tokens, invalid=(0, 3))
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    fused = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)
    reference = _reference_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert torch.isfinite(fused).all()
    max_abs = (fused - reference).abs().max().item()
    assert max_abs < 1e-4 * reference.abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        fused.reshape(1, -1).double(), reference.reshape(1, -1).double()
    ).item()
    assert cosine > 1 - 1e-9
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)


@cuda_only
def test_fused_never_reads_a_row_behind_an_invalid_slot():
    """A rank owning none of the selected experts contributes exact zeros, not NaN."""
    num_tokens = 16
    gemm2, index, routing, weight = _case(num_tokens, invalid=tuple(range(num_tokens * TOP_K)))

    combined = post_expert_norm_combine(
        gemm2, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    assert torch.equal(combined, torch.zeros_like(combined))


@cuda_only
def test_fused_selects_out_a_permuted_index_past_the_end_of_gemm2():
    """An index past the end of `gemm2` is selected out exactly like a -1 slot."""
    num_tokens, slots = 4, 4 * TOP_K
    gemm2, index, routing, weight = _case(num_tokens)
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    out_of_range = index.clone()
    out_of_range[1] = slots
    out_of_range[5] = slots + 1024
    as_invalid = index.clone()
    as_invalid[1] = -1
    as_invalid[5] = -1

    got = post_expert_norm_combine(gemm2, out_of_range, routing, weight, EPS, **kwargs)
    want = post_expert_norm_combine(gemm2, as_invalid, routing, weight, EPS, **kwargs)

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_the_padding_columns_of_a_wider_gemm2_row_are_ignored(device):
    """The MoE kernel's rows may be padded, and the padding can hold anything."""
    num_tokens = 4
    gemm2, index, routing, weight = _case(num_tokens, pad=64, device=device)
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    combined = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert combined.shape == (num_tokens, HIDDEN)
    assert torch.isfinite(combined).all()
    torch.testing.assert_close(
        combined,
        _reference_combine(gemm2[:, :HIDDEN], index, routing, weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_combine_accumulates_in_fp32_whatever_the_expert_output_dtype(device, dtype):
    num_tokens = 8
    gemm2, index, routing, weight = _case(
        num_tokens, hidden=256, top_k=4, dtype=dtype, device=device
    )
    kwargs = dict(num_tokens=num_tokens, top_k=4, hidden_size=256)

    combined = post_expert_norm_combine(gemm2, index, routing.to(dtype), weight, EPS, **kwargs)

    assert combined.dtype is torch.float32
    torch.testing.assert_close(
        combined,
        _reference_combine(gemm2, index, routing.to(dtype), weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@cuda_only
def test_fused_masks_the_tail_of_a_narrow_row():
    """Eight live lanes of a block padded to the next power of two, and two slots of top-k."""
    num_tokens, hidden, top_k = 4, 8, 2
    gemm2, index, routing, weight = _case(
        num_tokens, hidden=hidden, top_k=top_k, dtype=torch.float32, invalid=(3,)
    )
    kwargs = dict(num_tokens=num_tokens, top_k=top_k, hidden_size=hidden)

    fused = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert torch.isfinite(fused).all()
    torch.testing.assert_close(
        fused,
        _reference_combine(gemm2, index, routing, weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@cuda_only
def test_fused_finalize_replays_from_a_cuda_graph():
    """Warm up, capture, then replay against fresh inputs."""
    num_tokens = 32
    gemm2, index, routing, weight = _case(num_tokens, invalid=(1, 2))
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    fresh_gemm2, _, fresh_routing, _ = _case(num_tokens, invalid=(1, 2), seed=5)
    gemm2.copy_(fresh_gemm2)
    routing.copy_(fresh_routing)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_combine(gemm2, index, routing, weight, EPS, **kwargs)
    assert torch.isfinite(static_out).all()
    torch.testing.assert_close(static_out, expected, rtol=2e-5, atol=2e-5)


@cuda_only
@pytest.mark.parametrize("layout", ["no_tokens", "column_major"])
def test_unsupported_inputs_take_the_composition(layout, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("an unsupported input reached the fused combine")

    monkeypatch.setattr(musespark_open_kernels, "post_expert_norm_combine", unexpected)
    num_tokens = 0 if layout == "no_tokens" else 4
    gemm2, index, routing, weight = _case(num_tokens, dtype=torch.float32)
    if layout == "column_major":
        gemm2 = gemm2.t().contiguous().t()
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    combined = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert combined.shape == (num_tokens, HIDDEN)
    torch.testing.assert_close(
        combined, _reference_combine(gemm2, index, routing, weight, EPS, **kwargs)
    )
