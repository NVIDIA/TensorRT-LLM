# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification tests for the two-stage VisualGen SOL predictor."""

from __future__ import annotations

import dataclasses
import math
import struct

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol.predictor import (
    SolPredictorGeometry,
    SolPredictorOutputs,
    SolPredictorPlanKey,
    SOLSparsePredictor,
    _normalize_runtime_scalars,
)

_CPU_ONLY = pytest.mark.cpu_only


@_CPU_ONLY
def test_sol_predictor_geometry_and_static_plan_contract() -> None:
    geometry = SolPredictorGeometry.create(batch_size=2, seq_len=257, num_heads=3)
    assert (
        geometry.tensor_shape,
        geometry.summary_shape,
        geometry.stats_shape,
        geometry.exact_block_bits_shape,
    ) == ((2, 257, 3, 128), (2, 5, 3, 128), (2, 3, 128), (2, 3, 5, 1))
    assert (geometry.num_q_blocks, geometry.num_kv_blocks, geometry.tail_tokens) == (5, 5, 1)

    boundary_cases = (
        (64, 1, 1, 64),
        (65, 2, 1, 1),
        (64 * 32, 32, 1, 64),
        (64 * 32 + 1, 33, 2, 1),
    )
    for seq_len, blocks, words, tail in boundary_cases:
        current = SolPredictorGeometry.create(batch_size=1, seq_len=seq_len, num_heads=1)
        assert (current.num_q_blocks, current.exact_words, current.tail_tokens) == (
            blocks,
            words,
            tail,
        )

    key = SolPredictorPlanKey(geometry=geometry, device_index=1, dtype=torch.bfloat16)
    assert tuple(field.name for field in dataclasses.fields(key)) == (
        "geometry",
        "device_index",
        "dtype",
    )
    assert "tau" not in repr(key) and "sm_scale" not in repr(key)
    assert SOLSparsePredictor().num_plans == 0


@_CPU_ONLY
def test_sol_predictor_validates_geometry_and_runtime_scalars() -> None:
    invalid_geometry = (
        ({"batch_size": 0, "seq_len": 64, "num_heads": 1}, "batch_size"),
        ({"batch_size": 1, "seq_len": 0, "num_heads": 1}, "seq_len"),
        ({"batch_size": 1, "seq_len": 64, "num_heads": 0}, "num_heads"),
        ({"batch_size": True, "seq_len": 64, "num_heads": 1}, "batch_size"),
        ({"batch_size": 1, "seq_len": 64, "num_heads": 1, "head_dim": 64}, "head_dim=128"),
    )
    for kwargs, message in invalid_geometry:
        with pytest.raises((TypeError, ValueError), match=message):
            SolPredictorGeometry.create(**kwargs)

    tau, sm_scale = _normalize_runtime_scalars(tau=0.1, sm_scale=math.sqrt(0.5))
    expected_tau = struct.unpack("=f", struct.pack("=f", 0.1))[0]
    expected_scale = struct.unpack("=f", struct.pack("=f", math.sqrt(0.5)))[0]
    assert (tau, sm_scale) == (expected_tau, expected_scale)

    invalid_scalars = (
        (True, 0.125, "tau"),
        (math.nan, 0.125, "tau"),
        (0.0, True, "sm_scale"),
        (0.0, math.inf, "sm_scale"),
        (0.0, 0.0, "sm_scale"),
        (0.0, -0.125, "sm_scale"),
    )
    for invalid_tau, invalid_scale, message in invalid_scalars:
        with pytest.raises((TypeError, ValueError), match=message):
            _normalize_runtime_scalars(tau=invalid_tau, sm_scale=invalid_scale)


_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_LOG2_E = math.log2(math.e)


def _summary_oracle(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = (k.shape[1] + 63) // 64
    k_summary = torch.empty(
        (k.shape[0], blocks, k.shape[2], k.shape[3]),
        dtype=torch.bfloat16,
        device=k.device,
    )
    v_summary = torch.empty_like(k_summary)
    for block_idx in range(blocks):
        begin = block_idx * 64
        end = min(begin + 64, k.shape[1])
        k_summary[:, block_idx] = k[:, begin:end].float().mean(dim=1).to(torch.bfloat16)
        v_summary[:, block_idx] = v[:, begin:end].float().sum(dim=1).to(torch.bfloat16)
    return k_summary, v_summary


def _pack_bits(exact: torch.Tensor) -> torch.Tensor:
    words = (exact.shape[-1] + 31) // 32
    padded = F.pad(exact, (0, words * 32 - exact.shape[-1])).view(*exact.shape[:-1], words, 32)
    powers = 1 << torch.arange(32, dtype=torch.int64, device=exact.device)
    return (padded.to(torch.int64) * powers).sum(dim=-1).to(torch.uint32)


def _predictor_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_summary, v_summary = _summary_oracle(k, v)
    blocks = k_summary.shape[1]
    padded_q = F.pad(q, (0, 0, 0, 0, 0, blocks * 64 - q.shape[1]))
    q_blocks = padded_q.view(q.shape[0], blocks, 64, q.shape[2], q.shape[3])
    q_lengths = torch.clamp(
        q.shape[1] - torch.arange(blocks, device=q.device) * 64,
        min=1,
        max=64,
    )
    q_centroids = q_blocks.float().sum(dim=2) / q_lengths[None, :, None, None]
    k_float = k_summary.float()
    k_mean = k_float.mean(dim=1)
    k_var = torch.clamp(k_float.square().mean(dim=1) - k_mean.square(), min=0.0)
    log2_scale = float(sm_scale) * _LOG2_E
    projected_mean = torch.einsum("bqhd,bhd->bqh", q_centroids, k_mean) * log2_scale
    projected_var = (
        torch.einsum("bqhd,bhd->bqh", q_centroids.square(), k_var) * log2_scale * log2_scale
    )
    threshold = projected_mean + float(tau) * torch.sqrt(projected_var + 1.0e-6)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_centroids, k_float) * log2_scale
    exact = scores > threshold.permute(0, 2, 1).unsqueeze(-1)
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


def _output_tensors(
    outputs: SolPredictorOutputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return outputs.exact_block_bits, outputs.k_summary, outputs.v_summary


def _assert_outputs_match(
    outputs: SolPredictorOutputs,
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    assert torch.equal(outputs.exact_block_bits, expected[0])
    torch.testing.assert_close(outputs.k_summary, expected[1], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(outputs.v_summary, expected[2], rtol=1e-2, atol=2e-2)


def _run_custom_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *buffers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.ops.trtllm.visual_gen_sol_predictor(q, k, v, *buffers, 64, 0.5, 0.125)
    return buffers[0], buffers[1], buffers[2]


@_REQUIRES_CUDA
def test_sol_predictor_s257_bh_gt_one_matches_oracle_and_reuses_storage() -> None:
    q, k, v = _inputs((2, 257, 3, 128), 11)
    predictor = SOLSparsePredictor()
    plan = predictor.prepare(q, k, v)
    output_ids = tuple(map(id, _output_tensors(plan.outputs)))
    scratch_ids = (id(plan.k_mean), id(plan.k_var_diag))

    outputs = predictor.predict(q, k, v, tau=0.75, sm_scale=0.125)
    reference = _predictor_oracle(q, k, v, tau=0.75, sm_scale=0.125)

    assert outputs is plan.outputs
    assert tuple(map(id, _output_tensors(outputs))) == output_ids
    assert (id(plan.k_mean), id(plan.k_var_diag)) == scratch_ids
    _assert_outputs_match(outputs, reference)
    expected_k_mean = reference[1].float().mean(dim=1)
    expected_k_var = torch.clamp(
        reference[1].float().square().mean(dim=1) - expected_k_mean.square(),
        min=0.0,
    )
    torch.testing.assert_close(plan.k_mean, expected_k_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(plan.k_var_diag, expected_k_var, rtol=1e-5, atol=1e-5)

    second = predictor.predict(q, k, v, tau=-0.25, sm_scale=0.0625)
    assert second is outputs
    assert tuple(map(id, _output_tensors(second))) == output_ids
    assert predictor.num_plans == 1


@_REQUIRES_CUDA
def test_sol_predictor_s257_runtime_scale_and_tau_extremes() -> None:
    q, k, v = _inputs((1, 257, 2, 128), 61)
    predictor = SOLSparsePredictor()

    normal = predictor.predict(q, k, v, tau=0.5, sm_scale=0.125)
    normal_bits = normal.exact_block_bits.clone()
    expected_normal = _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125)[0]
    tiny = predictor.predict(q, k, v, tau=0.5, sm_scale=1.0e-5)
    expected_tiny = _predictor_oracle(q, k, v, tau=0.5, sm_scale=1.0e-5)[0]

    assert tiny is normal
    assert torch.equal(normal_bits, expected_normal)
    assert torch.equal(tiny.exact_block_bits, expected_tiny)
    assert not torch.equal(expected_normal, expected_tiny)

    blocks = 5
    block_ids = torch.arange(blocks, device=q.device)
    local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    expected_extremes = (
        _pack_bits(local[None, None].expand(1, 2, -1, -1)),
        _pack_bits(torch.ones((1, 2, blocks, blocks), device=q.device, dtype=torch.bool)),
    )
    for tau, expected in zip((1.0e6, -1.0e6), expected_extremes, strict=True):
        outputs = predictor.predict(q, k, v, tau=tau, sm_scale=128**-0.5)
        assert torch.equal(outputs.exact_block_bits, expected)


@_REQUIRES_CUDA
def test_sol_predictor_long_proxy_group_keeps_tail_mass_and_clears_padding_bits() -> None:
    tokens = 16_451
    q, k, _ = _inputs((1, tokens, 1, 128), 31)
    v = torch.ones_like(q)
    outputs = SOLSparsePredictor().predict(q, k, v, tau=1.0e6, sm_scale=0.125)
    expected, expected_k, expected_v = _predictor_oracle(q, k, v, tau=1.0e6, sm_scale=0.125)

    assert outputs.k_summary.shape[1] == 258
    assert outputs.exact_block_bits.shape[-1] == 9
    assert torch.equal(outputs.exact_block_bits, expected)
    torch.testing.assert_close(outputs.k_summary[:, -1], expected_k[:, -1], rtol=1e-2, atol=1e-2)
    assert torch.equal(outputs.v_summary[:, -1], expected_v[:, -1])
    assert torch.all(outputs.v_summary[:, -1] == 3)
    assert int(outputs.exact_block_bits[..., -1].to(torch.int64).max()) < 4


@_REQUIRES_CUDA
def test_sol_predictor_cuda_graph_replay_updates_live_outputs() -> None:
    q, k, v = _inputs((1, 257, 2, 128), 41)
    predictor = SOLSparsePredictor()
    plan = predictor.prepare(q, k, v)
    predictor.predict(q, k, v, tau=0.5, sm_scale=0.125)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = predictor.predict(q, k, v, tau=0.5, sm_scale=0.125)
    assert captured is plan.outputs

    next_q, next_k, next_v = _inputs(q.shape, 51)
    q.copy_(next_q)
    k.copy_(next_k)
    v.copy_(next_v)
    graph.replay()
    torch.cuda.synchronize()
    expected = _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125)

    _assert_outputs_match(captured, expected)


@_REQUIRES_CUDA
def test_sol_predictor_rejects_plan_miss_during_capture_and_reuses_prepared_plan(
    monkeypatch,
) -> None:
    q, k, v = _inputs((1, 193, 1, 128), 71)

    with monkeypatch.context() as capture:
        capture.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="plan must be prepared"):
            SOLSparsePredictor().predict(q, k, v, tau=0.5, sm_scale=0.125)

    predictor = SOLSparsePredictor()
    plan = predictor.prepare(q, k, v)
    with monkeypatch.context() as capture:
        capture.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        assert predictor.predict(q, k, v, tau=0.5, sm_scale=0.125) is plan.outputs
    torch.cuda.synchronize()
    _assert_outputs_match(plan.outputs, _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125))


@_REQUIRES_CUDA
def test_sol_predictor_compiled_public_predict_owns_each_instance_plan(recwarn) -> None:
    output_ptrs = []
    for seed in (81, 91):
        q, k, v = _inputs((1, 257, 2, 128), seed)
        predictor = SOLSparsePredictor()
        compiled_predict = torch.compile(predictor.predict, backend="eager", fullgraph=False)

        outputs = compiled_predict(q, k, v, tau=0.5, sm_scale=0.125)
        expected = _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125)

        assert predictor.num_plans == 1
        _assert_outputs_match(outputs, expected)
        output_ptrs.append(tuple(tensor.data_ptr() for tensor in _output_tensors(outputs)))

    assert output_ptrs[0] != output_ptrs[1]
    assert not any("recompile_limit" in str(warning.message) for warning in recwarn)


@_REQUIRES_CUDA
def test_sol_predictor_custom_op_fake_schema_and_fullgraph_compile() -> None:
    from tensorrt_llm._torch.visual_gen.attention_backend.sparse.sol import kernels  # noqa: F401

    op = torch.ops.trtllm.visual_gen_sol_predictor.default
    schema = str(op._schema)
    assert "exact_block_bits" in schema
    assert "k_summary" in schema
    assert "v_summary" in schema
    assert "!" in schema
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "trtllm::visual_gen_sol_predictor", "Meta"
    )

    meta_q = torch.empty((1, 65, 1, 128), device="meta", dtype=torch.bfloat16)
    meta_summary = torch.empty((1, 2, 1, 128), device="meta", dtype=torch.bfloat16)
    meta_stats = torch.empty((1, 1, 128), device="meta", dtype=torch.float32)
    meta_args = (
        meta_q,
        torch.empty_like(meta_q),
        torch.empty_like(meta_q),
        torch.empty((1, 1, 2, 1), device="meta", dtype=torch.uint32),
        meta_summary,
        torch.empty_like(meta_summary),
        meta_stats,
        torch.empty_like(meta_stats),
        torch.empty((1, 2, 1, 128), device="meta", dtype=torch.float32),
    )
    assert op(*meta_args, 64, 0.5, 0.125) is None

    q, k, v = _inputs((1, 257, 2, 128), 81)
    plan = SOLSparsePredictor().prepare(q, k, v)
    buffers = (
        plan.outputs.exact_block_bits,
        plan.outputs.k_summary,
        plan.outputs.v_summary,
        plan.k_mean,
        plan.k_var_diag,
        plan.q_centroid,
    )
    actual = torch.compile(_run_custom_op, backend="eager", fullgraph=True)(q, k, v, *buffers)
    expected = _predictor_oracle(q, k, v, tau=0.5, sm_scale=0.125)

    assert all(
        actual_tensor is plan_tensor
        for actual_tensor, plan_tensor in zip(actual, buffers[:3], strict=True)
    )
    _assert_outputs_match(plan.outputs, expected)
