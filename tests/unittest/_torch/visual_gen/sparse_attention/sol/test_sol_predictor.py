# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification tests for the two-stage VisualGen SOL predictor."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.kernels import (
    block_pool,
    block_thresholds,
)
from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.predictor import (
    BLOCK_SIZE,
    SolPredictorOutputs,
    _runtime_scalar,
    predict,
    support_reason,
)

_CPU_ONLY = pytest.mark.cpu_only
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_LOG2_E = math.log2(math.e)
_POLICIES = ("diag", "exact")


@_CPU_ONLY
def test_sol_predictor_support_reason_covers_layout_shape_dtype_and_device() -> None:
    good = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    cases = (
        ((good.view(64, 128), good, good), "compact BSHD"),
        ((good, torch.zeros(1, 65, 1, 128, dtype=torch.bfloat16), good), "uniform self-attention"),
        ((good, good.float(), good), "matching BF16"),
        ((good, good, good), "requires CUDA"),
        ((None, good, good), "torch tensors"),
    )
    for tensors, message in cases:
        assert message in support_reason(*tensors)
    with pytest.raises(ValueError, match="requires CUDA"):
        predict(good, good, good, tau=0.5, sm_scale=0.125)


@_CPU_ONLY
def test_sol_predictor_validates_runtime_scalars() -> None:
    assert _runtime_scalar(0.1, "tau") == 0.1
    assert _runtime_scalar(2, "sm_scale", positive=True) == 2.0
    invalid = (
        (True, "tau", {}),
        (math.nan, "tau", {}),
        (math.inf, "sm_scale", {"positive": True}),
        (0.0, "sm_scale", {"positive": True}),
        (-0.125, "sm_scale", {"positive": True}),
        ("0.5", "tau", {}),
    )
    for value, name, kwargs in invalid:
        with pytest.raises((TypeError, ValueError), match=name):
            _runtime_scalar(value, name, **kwargs)


def _summary_oracle(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = (k.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_summary = torch.empty(
        (k.shape[0], blocks, k.shape[2], k.shape[3]),
        dtype=torch.bfloat16,
        device=k.device,
    )
    v_summary = torch.empty_like(k_summary)
    for block_idx in range(blocks):
        begin = block_idx * BLOCK_SIZE
        end = min(begin + BLOCK_SIZE, k.shape[1])
        k_summary[:, block_idx] = k[:, begin:end].float().mean(dim=1).to(torch.bfloat16)
        v_summary[:, block_idx] = v[:, begin:end].float().sum(dim=1).to(torch.bfloat16)
    return k_summary, v_summary


def _pack_bits(exact: torch.Tensor) -> torch.Tensor:
    words = (exact.shape[-1] + 31) // 32
    padded = F.pad(exact, (0, words * 32 - exact.shape[-1])).view(*exact.shape[:-1], words, 32)
    powers = 1 << torch.arange(32, dtype=torch.int64, device=exact.device)
    return (padded.to(torch.int64) * powers).sum(dim=-1).to(torch.uint32)


def _centroid_oracle(q: torch.Tensor) -> torch.Tensor:
    blocks = (q.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    padded_q = F.pad(q, (0, 0, 0, 0, 0, blocks * BLOCK_SIZE - q.shape[1]))
    q_blocks = padded_q.view(q.shape[0], blocks, BLOCK_SIZE, q.shape[2], q.shape[3])
    q_lengths = torch.clamp(
        q.shape[1] - torch.arange(blocks, device=q.device) * BLOCK_SIZE, min=1, max=BLOCK_SIZE
    )
    return q_blocks.double().sum(dim=2) / q_lengths[None, :, None, None]


def _threshold_oracle(
    q_centroids: torch.Tensor,
    k_summary: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
    thresh_type: str,
) -> torch.Tensor:
    """fp64 ``[batch, heads, num_q_blocks]`` thresholds of the selected policy."""
    c = q_centroids.double().permute(0, 2, 1, 3)
    keys = k_summary.double().permute(0, 2, 1, 3)
    k_mean = keys.mean(dim=2)
    mean = torch.einsum("bhqd,bhd->bhq", c, k_mean)
    if thresh_type == "diag":
        k_var = torch.clamp(keys.square().mean(dim=2) - k_mean.square(), min=0.0)
        var = torch.einsum("bhqd,bhd->bhq", c.square(), k_var)
    else:
        centered = keys - k_mean.unsqueeze(2)
        covariance = torch.matmul(centered.transpose(-1, -2), centered) / keys.shape[2]
        var = torch.einsum("bhqd,bhqd->bhq", torch.matmul(c, covariance), c)
    log2_scale = float(sm_scale) * _LOG2_E
    return mean * log2_scale + float(tau) * torch.sqrt(var * log2_scale * log2_scale + 1.0e-6)


def _predictor_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
    thresh_type: str = "diag",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_summary, v_summary = _summary_oracle(k, v)
    q_centroids = _centroid_oracle(q)
    threshold = _threshold_oracle(
        q_centroids, k_summary, tau=tau, sm_scale=sm_scale, thresh_type=thresh_type
    )
    scores = torch.einsum("bqhd,bkhd->bhqk", q_centroids, k_summary.double()) * (
        float(sm_scale) * _LOG2_E
    )
    exact = scores > threshold.unsqueeze(-1)
    blocks = k_summary.shape[1]
    block_ids = torch.arange(blocks, device=q.device)
    exact |= (block_ids[:, None] - block_ids[None, :]).abs()[None, None] <= 1
    return _pack_bits(exact), k_summary, v_summary


def _small_integer_bf16(shape: tuple[int, ...], *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-2, 3, shape, generator=generator, device="cuda", dtype=torch.bfloat16)


def _inputs(
    shape: tuple[int, int, int, int], seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _small_integer_bf16(shape, seed=seed),
        _small_integer_bf16(shape, seed=seed + 1),
        _small_integer_bf16(shape, seed=seed + 2),
    )


def _assert_outputs_match(
    outputs: SolPredictorOutputs,
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    assert torch.equal(outputs.exact_block_bits, expected[0])
    torch.testing.assert_close(outputs.k_summary, expected[1], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(outputs.v_summary, expected[2], rtol=1e-2, atol=2e-2)


@_REQUIRES_CUDA
def test_sol_predictor_support_reason_on_cuda_tensors() -> None:
    q, k, v = _inputs((1, 64, 2, 128), 5)
    assert support_reason(q, k, v) is None
    strided = q.transpose(1, 2).contiguous().transpose(1, 2)
    assert strided.shape == q.shape and not strided.is_contiguous()
    assert "contiguous" in support_reason(strided, k, v)
    narrow = torch.zeros((1, 64, 2, 64), device="cuda", dtype=torch.bfloat16)
    assert "head_dim=128" in support_reason(narrow, narrow, narrow)
    empty = torch.zeros((1, 0, 2, 128), device="cuda", dtype=torch.bfloat16)
    assert "positive" in support_reason(empty, empty, empty)
    with pytest.raises(ValueError, match="thresh_type"):
        predict(q, k, v, tau=0.5, sm_scale=0.125, thresh_type="diagonal")


@_REQUIRES_CUDA
@pytest.mark.parametrize("thresh_type", _POLICIES)
def test_sol_predictor_matches_oracle(thresh_type: str) -> None:
    q, k, v = _inputs((2, 257, 3, 128), 11)
    outputs = predict(q, k, v, tau=0.75, sm_scale=0.125, thresh_type=thresh_type)
    assert outputs.exact_block_bits.shape == (2, 3, 5, 1)
    assert outputs.k_summary.shape == outputs.v_summary.shape == (2, 5, 3, 128)
    _assert_outputs_match(
        outputs,
        _predictor_oracle(q, k, v, tau=0.75, sm_scale=0.125, thresh_type=thresh_type),
    )


@_REQUIRES_CUDA
def test_sol_predictor_exact_policy_sees_key_channel_correlation() -> None:
    """Keys whose channels move together have a full-covariance variance the diagonal policy misses."""
    q, _, v = _inputs((1, 320, 2, 128), 23)
    shared = _small_integer_bf16((1, 320, 2, 1), seed=29)
    k = shared.expand(-1, -1, -1, 128).contiguous()
    diag = predict(q, k, v, tau=0.5, sm_scale=0.125, thresh_type="diag")
    exact = predict(q, k, v, tau=0.5, sm_scale=0.125, thresh_type="exact")
    _assert_outputs_match(diag, _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125))
    _assert_outputs_match(
        exact, _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125, thresh_type="exact")
    )
    assert not torch.equal(diag.exact_block_bits, exact.exact_block_bits)


@_REQUIRES_CUDA
def test_sol_predictor_runtime_scale_and_tau_extremes() -> None:
    q, k, v = _inputs((1, 257, 2, 128), 61)
    normal = predict(q, k, v, tau=0.5, sm_scale=0.125).exact_block_bits
    tiny = predict(q, k, v, tau=0.5, sm_scale=1.0e-5).exact_block_bits
    assert torch.equal(normal, _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125)[0])
    assert torch.equal(tiny, _predictor_oracle(q, k, v, tau=0.5, sm_scale=1.0e-5)[0])
    assert not torch.equal(normal, tiny)

    blocks = 5
    block_ids = torch.arange(blocks, device=q.device)
    local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    expected_extremes = (
        _pack_bits(local[None, None].expand(1, 2, -1, -1)),
        _pack_bits(torch.ones((1, 2, blocks, blocks), device=q.device, dtype=torch.bool)),
    )
    for tau, expected in zip((1.0e6, -1.0e6), expected_extremes, strict=True):
        outputs = predict(q, k, v, tau=tau, sm_scale=128**-0.5)
        assert torch.equal(outputs.exact_block_bits, expected)


@_REQUIRES_CUDA
def test_sol_predictor_long_proxy_group_keeps_tail_mass_and_clears_padding_bits() -> None:
    tokens = 16_451
    q, k, _ = _inputs((1, tokens, 1, 128), 31)
    v = torch.ones_like(q)
    outputs = predict(q, k, v, tau=1.0e6, sm_scale=0.125)
    expected, expected_k, expected_v = _predictor_oracle(q, k, v, tau=1.0e6, sm_scale=0.125)

    assert outputs.k_summary.shape[1] == 258
    assert outputs.exact_block_bits.shape[-1] == 9
    assert torch.equal(outputs.exact_block_bits, expected)
    torch.testing.assert_close(outputs.k_summary[:, -1], expected_k[:, -1], rtol=1e-2, atol=1e-2)
    assert torch.equal(outputs.v_summary[:, -1], expected_v[:, -1])
    assert torch.all(outputs.v_summary[:, -1] == 3)
    assert int(outputs.exact_block_bits[..., -1].to(torch.int64).max()) < 4


@_REQUIRES_CUDA
@pytest.mark.parametrize("thresh_type", _POLICIES)
def test_sol_predictor_cuda_graph_replay_refreshes_captured_outputs(thresh_type: str) -> None:
    q, k, v = _inputs((1, 257, 2, 128), 41)
    predict(q, k, v, tau=0.5, sm_scale=0.125, thresh_type=thresh_type)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = predict(q, k, v, tau=0.5, sm_scale=0.125, thresh_type=thresh_type)

    next_q, next_k, next_v = _inputs(q.shape, 51)
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()

    _assert_outputs_match(
        captured, _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125, thresh_type=thresh_type)
    )


@_REQUIRES_CUDA
def test_sol_predictor_operator_is_functional_and_compiles_fullgraph() -> None:
    op = torch.ops.trtllm.visual_gen_sol_predictor.default
    schema = str(op._schema)
    assert "!" not in schema, "the predictor operator must not mutate its inputs"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "trtllm::visual_gen_sol_predictor", "Meta"
    )
    meta_q = torch.empty((1, 65, 3, 128), device="meta", dtype=torch.bfloat16)
    bits, k_summary, v_summary = op(meta_q, meta_q, meta_q, BLOCK_SIZE, 0.5, 0.125, "diag")
    assert bits.shape == (1, 3, 2, 1) and bits.dtype == torch.uint32
    assert k_summary.shape == v_summary.shape == (1, 2, 3, 128)

    q, k, v = _inputs((1, 257, 2, 128), 81)
    compiled = torch.compile(predict, backend="eager", fullgraph=True)
    for thresh_type in _POLICIES:
        outputs = compiled(q, k, v, tau=0.5, sm_scale=0.125, thresh_type=thresh_type)
        _assert_outputs_match(
            outputs,
            _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125, thresh_type=thresh_type),
        )


@_REQUIRES_CUDA
@pytest.mark.parametrize("thresh_type", _POLICIES)
def test_sol_predictor_thresholds_match_the_fused_kernel_preprocess(thresh_type: str) -> None:
    """Both backends must route from the same per-block threshold."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("the fused kernel preprocess needs TMA descriptors")
    preprocess = pytest.importorskip(
        "tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn.preprocess"
    )
    generator = torch.Generator(device="cuda").manual_seed(97)
    shape = (2, 320, 2, 128)
    q, k, v = (
        torch.randn(shape, generator=generator, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    _, _, fused_threshold = preprocess.prepare(
        q, k, v, tau=0.75, scale=128**-0.5, thresh_type=thresh_type
    )

    summary_shape = (shape[0], shape[1] // BLOCK_SIZE, shape[2], shape[3])
    centroid = torch.empty(summary_shape, device="cuda", dtype=torch.float32)
    k_summary = torch.empty(summary_shape, device="cuda", dtype=torch.bfloat16)
    block_pool(q, centroid, block_size=BLOCK_SIZE, reduce="mean")
    block_pool(k, k_summary, block_size=BLOCK_SIZE, reduce="mean")
    threshold = block_thresholds(
        centroid, k_summary, tau=0.75, sm_scale=128**-0.5, thresh_type=thresh_type
    )

    torch.testing.assert_close(threshold.permute(0, 2, 1), fused_threshold, rtol=2e-2, atol=2e-2)
