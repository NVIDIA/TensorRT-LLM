# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster boundary rounding, live graph inputs, and bounded dispatch."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary_available
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not cluster_boundary_available(torch.device("cuda", 0)),
    reason="GB300 and CuTe DSL required",
)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16, 17, 32])
@pytest.mark.parametrize("with_gain", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_cluster_boundary_rounding_and_mutable_graph(
    tokens: int, with_gain: bool, eps: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(1294 + tokens)
    residual = torch.randn(tokens, 8192, device="cuda", dtype=torch.float32)
    branch = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    gate = MuseSparkOpenGatedResidual(8192, torch.bfloat16).cuda()
    gate.gate.normal_(0, 0.5)
    gate.cache_derived_state()
    norm = RMSNorm(hidden_size=8192, eps=eps, dtype=torch.bfloat16, use_gemma=True).cuda()
    norm.weight.normal_(0, 0.02)
    gain = torch.empty(8192, device="cuda", dtype=torch.bfloat16) if with_gain else None
    if gain is not None:
        gain.normal_(1, 0.1)

    # Compile on the eager stream, then capture on the graph stream.
    gate.add_after_rmsnorm_and_norm(residual, branch, eps, norm, gain)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gate.add_after_rmsnorm_and_norm(residual, branch, eps, norm, gain)
    for scale in (1e-3, 1.0, 1e3):
        residual.normal_().mul_(scale)
        branch.normal_().mul_(scale)
        norm.weight.normal_(0, 0.02)
        if gain is not None:
            gain.normal_(1, 0.1)
        graph.replay()
        with monkeypatch.context() as reference:
            reference.setattr(
                "tensorrt_llm._torch.models.musespark_open_boundary.cluster_boundary_supported",
                lambda *args: False,
            )
            expected = gate.add_after_rmsnorm_and_norm(residual, branch, eps, norm, gain)
        torch.cuda.synchronize()
        assert actual[0].dtype == torch.float32 and actual[1].dtype == torch.bfloat16
        for got in actual:
            assert torch.isfinite(got).all()
        # Different variance trees can cross individual BF16 thresholds.
        # The tight aggregate bound rejects omitting a BF16 rounding stage.
        residual_error = (actual[0] - expected[0]).norm() / expected[0].norm().clamp_min(1e-12)
        normalized_error = (actual[1].float() - expected[1].float()).norm() / expected[
            1
        ].float().norm().clamp_min(1e-12)
        assert residual_error < 2e-4
        assert normalized_error < 1e-3
        eager = gate.add_after_rmsnorm_and_norm(residual, branch, eps, norm, gain)
        for got, want in zip(actual, eager):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,hidden", [(33, 8192), (2, 4096)])
@torch.inference_mode()
def test_cluster_boundary_unsupported_shapes_keep_fallback(
    tokens: int, hidden: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.models import musespark_open_boundary

    def unexpected(*args, **kwargs):
        raise AssertionError("Unsupported shape reached the cluster kernel")

    monkeypatch.setattr(musespark_open_boundary, "cluster_boundary", unexpected)
    residual = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
    branch = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    gate = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    actual = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
    expected = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@torch.inference_mode()
def test_cluster_boundary_unaligned_contiguous_branch_keeps_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.models import musespark_open_boundary

    def unexpected(*args, **kwargs):
        raise AssertionError("Unaligned input reached the vectorized cluster kernel")

    monkeypatch.setattr(musespark_open_boundary, "cluster_boundary", unexpected)
    residual = torch.randn(2, 8192, device="cuda", dtype=torch.float32)
    branch = torch.randn(2 * 8192 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(2, 8192)
    assert branch.is_contiguous() and branch.data_ptr() % 16 != 0
    gate = MuseSparkOpenGatedResidual(8192, torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=8192, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    actual = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
    expected = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@torch.inference_mode()
def test_cluster_boundary_rejects_empty_inputs() -> None:
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary_supported

    residual = torch.empty(0, 8192, device="cuda", dtype=torch.float32)
    branch = torch.empty_like(residual, dtype=torch.bfloat16)
    coefficient = torch.ones(8192, device="cuda", dtype=torch.float32)
    weight = torch.ones(8192, device="cuda", dtype=torch.bfloat16)
    assert not cluster_boundary_supported(residual, branch, None, coefficient, coefficient, weight)


@pytest.mark.parametrize("device_index", [0, 1])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("tokens", [2, 16])
@torch.inference_mode()
def test_cluster_boundary_side_stream_and_gate_cache(
    device_index: int, cache_enabled: bool, tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if torch.cuda.device_count() <= device_index:
        pytest.skip("Second GPU unavailable")
    with torch.cuda.device(device_index):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            residual = torch.randn(tokens, 8192, device="cuda", dtype=torch.float32)
            branch = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
            gain = torch.ones(8192, device="cuda", dtype=torch.bfloat16)
            gate = MuseSparkOpenGatedResidual(8192, torch.bfloat16).cuda()
            gate.gate.normal_(0, 0.5)
            if cache_enabled:
                # Uncached, the coefficients are recomputed from the live gate.
                gate.cache_derived_state()
            norm = RMSNorm(hidden_size=8192, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
            norm.weight.normal_(0, 0.02)
            gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm, gain)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm, gain)
            for _ in range(8):
                residual.normal_()
                branch.normal_()
                gain.normal_(1, 0.1)
                norm.weight.normal_(0, 0.02)
                if not cache_enabled:
                    gate.gate.normal_(0, 0.5)
                graph.replay()
                with monkeypatch.context() as reference:
                    reference.setattr(
                        "tensorrt_llm._torch.models.musespark_open_boundary.cluster_boundary_supported",
                        lambda *args: False,
                    )
                    expected = gate.add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm, gain)
                stream.synchronize()
                for got, want, tolerance in zip(actual, expected, (2e-4, 1e-3)):
                    assert got.device.index == device_index
                    assert torch.isfinite(got).all()
                    error = (got.float() - want.float()).norm() / want.float().norm()
                    assert error < tolerance


def test_cluster_boundary_does_not_import_the_finalize_module() -> None:
    import ast
    import inspect

    from tensorrt_llm._torch.models import musespark_open_boundary

    tree = ast.parse(inspect.getsource(musespark_open_boundary))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("finalize_ar" in name for name in imported)


def test_cluster_boundary_availability_ignores_finalize_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling the finalize fusion must not disable the boundary kernel."""
    from tensorrt_llm._torch.models import musespark_open_boundary, musespark_open_finalize_ar

    monkeypatch.setattr(musespark_open_finalize_ar, "HAS_CUTEDSL", False)
    musespark_open_boundary.cluster_boundary_available.cache_clear()
    try:
        assert musespark_open_boundary.cluster_boundary_available(torch.device("cuda", 0))
    finally:
        musespark_open_boundary.cluster_boundary_available.cache_clear()
