# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""draft_front (the draft_front contract) against the native ``DeepseekV3MTP`` front end.

The oracle runs the native ops of ``DeepseekV3MTP.forward``: TensorRT-LLM
``RMSNorm`` modules for ``enorm``/``hnorm`` (run in parallel through
``maybe_execute_in_parallel``), ``torch.concat`` and ``torch.chunk``. The
candidate must reproduce chunk ``tp_rank`` bit for bit, with the native view's
shape and strides.

Fixtures:
* synthetic: random activations and weights (labelled ``synthetic``).
* real: ``enorm``/``hnorm``/``eh_proj``/``model.norm``/``shared_head.norm``
  weights and embedding rows read from the GLM-5.2 NVFP4 checkpoint
  (``GLM52_NVFP4_CKPT``); the hidden states entering the draft layer are
  synthetic pre-norm activations passed through the real ``model.norm`` (first
  draft pass) or ``shared_head.norm`` (later passes) weights.

The native-norm-path tests check that draft_front serves a call only when the native
``enorm``/``hnorm`` run the FlashInfer CuTe DSL RMSNorm that it mirrors, and
leaves every other native path (gemma, cuda.tile, NVFP4, torch fallback,
FlashInfer CUDA JIT norm, unverified FlashInfer) to the native front.
"""

import functools
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_front import (  # noqa: E501
    native_norm,
    native_norm_reason,
    pack_draft_front,
    plan_segments,
    unsupported_reason,
)
from tensorrt_llm._torch.modules import rms_norm as rms_norm_module
from tensorrt_llm._torch.modules.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

GLM_H = 6144
GLM_EPS = 1e-5
GLM_TP = 8
CKPT = os.environ.get(
    "GLM52_NVFP4_CKPT",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
# Draft passes at K=5 for one request: first pass verifies 6 rows, then 1 row each.
DRAFT_ROWS = (6, 1)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402


class NativeDraftFront:
    """Native ``DeepseekV3MTP`` front end: parallel enorm/hnorm, concat, chunk."""

    def __init__(
        self,
        enorm_weight: torch.Tensor,
        hnorm_weight: torch.Tensor,
        enorm_eps: float = GLM_EPS,
        hnorm_eps: float = GLM_EPS,
    ):
        hidden_size = enorm_weight.shape[0]
        self.enorm = RMSNorm(hidden_size=hidden_size, eps=enorm_eps, dtype=torch.bfloat16).cuda()
        self.hnorm = RMSNorm(hidden_size=hidden_size, eps=hnorm_eps, dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            self.enorm.weight.copy_(enorm_weight)
            self.hnorm.weight.copy_(hnorm_weight)
        self.aux_stream = torch.cuda.Stream()
        self.events = (torch.cuda.Event(), torch.cuda.Event())

    def __call__(
        self, embeds: torch.Tensor, hidden: torch.Tensor, tp_size: int, tp_rank: int
    ) -> torch.Tensor:
        e, h = maybe_execute_in_parallel(
            lambda: self.enorm(embeds),
            lambda: self.hnorm(hidden),
            self.events[0],
            self.events[1],
            self.aux_stream,
            disable_on_compile=True,
        )
        x = torch.concat([e, h], dim=-1)
        if tp_size > 1:
            x = torch.chunk(x, tp_size, dim=-1)[tp_rank]
        return x

    def candidate(
        self, embeds: torch.Tensor, hidden: torch.Tensor, tp_size: int, tp_rank: int
    ) -> torch.Tensor | None:
        return pack_draft_front(
            embeds,
            hidden,
            self.enorm.weight,
            self.enorm.variance_epsilon,
            self.hnorm.weight,
            self.hnorm.variance_epsilon,
            tp_rank,
            tp_size,
        )


def assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, what: str = "") -> None:
    assert actual is not None, f"{what}: candidate fell back to native"
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    assert actual.stride() == expected.stride(), (what, actual.stride(), expected.stride())
    assert actual.dtype == expected.dtype
    a = actual.contiguous().view(torch.int16)
    b = expected.contiguous().view(torch.int16)
    mismatch = a != b
    if bool(mismatch.any()):
        diff = (actual.float() - expected.float()).abs()
        pytest.fail(
            f"{what}: {int(mismatch.sum())}/{mismatch.numel()} elements differ, "
            f"max |diff| {float(diff.max()):.3e}"
        )


def make_inputs(
    rows: int, hidden_size: int = GLM_H, seed: int = 0, scale_spread: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic BF16 embeddings / hidden states with per-row scale spread and outliers."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    e = torch.randn(rows, hidden_size, device="cuda", generator=gen)
    h = torch.randn(rows, hidden_size, device="cuda", generator=gen)
    if scale_spread:
        scales = torch.logspace(-2, 2, steps=max(rows, 1), device="cuda")[:rows, None]
        e = e * scales.flip(0) * 0.05
        h = h * scales
        h[:, 7] *= 50.0  # an outlier channel, as in residual streams
    return e.bfloat16(), h.bfloat16()


def make_weights(hidden_size: int = GLM_H, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    we = (1.0 + 0.5 * torch.randn(hidden_size, device="cuda", generator=gen)).bfloat16()
    wh = (0.5 + 0.25 * torch.randn(hidden_size, device="cuda", generator=gen)).bfloat16()
    return we, wh


def fp64_reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x64 = x.double()
    return x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * w.double()


# --------------------------------------------------------------------------- planning (CPU)


@pytest.mark.parametrize("hidden_size", [GLM_H, 7168, 2048])
@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
def test_plan_segments_cover_native_chunks(hidden_size, tp_size):
    if (2 * hidden_size) % tp_size:
        pytest.skip("not a native chunking")
    width = 2 * hidden_size // tp_size
    for rank in range(tp_size):
        cols = []
        for seg in plan_segments(hidden_size, tp_size, rank):
            assert 0 <= seg.begin < seg.end <= hidden_size
            cols.extend(
                range(seg.source * hidden_size + seg.begin, seg.source * hidden_size + seg.end)
            )
        assert cols == list(range(rank * width, (rank + 1) * width))


def test_plan_segments_glm_tp8_one_source_per_rank():
    for rank in range(GLM_TP):
        (seg,) = plan_segments(GLM_H, GLM_TP, rank)
        assert seg.source == (0 if rank < 4 else 1)
        assert seg.end - seg.begin == 1536


def test_plan_segments_rejects_bad_tp():
    with pytest.raises(ValueError):
        plan_segments(GLM_H, 5, 0)
    with pytest.raises(ValueError):
        plan_segments(GLM_H, 8, 8)


# --------------------------------------------------------------------------- equivalence


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2, 3, 6, 7, 64, 128, 1000])
@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
def test_matches_native_all_ranks_synthetic(rows, tp_size):
    e, h = make_inputs(rows, seed=rows)
    native = NativeDraftFront(*make_weights())
    for rank in range(tp_size):
        assert_bitwise_equal(
            native.candidate(e, h, tp_size, rank),
            native(e, h, tp_size, rank),
            f"synthetic rows={rows} tp={tp_size} rank={rank}",
        )


@requires_cuda
@pytest.mark.parametrize("hidden_size", [2048, 4096, 7168])
@pytest.mark.parametrize("rows", [1, 6])
def test_matches_native_other_hidden_sizes(hidden_size, rows):
    e, h = make_inputs(rows, hidden_size, seed=3)
    native = NativeDraftFront(*make_weights(hidden_size))
    for rank in range(GLM_TP):
        assert_bitwise_equal(
            native.candidate(e, h, GLM_TP, rank),
            native(e, h, GLM_TP, rank),
            f"H={hidden_size} rows={rows} rank={rank}",
        )


@requires_cuda
@pytest.mark.parametrize("enorm_eps, hnorm_eps", [(1e-5, 1e-5), (1e-6, 1e-3), (1e-2, 1e-6)])
def test_epsilon_per_source(enorm_eps, hnorm_eps):
    # Small magnitudes so that eps dominates mean(x^2) for some rows.
    e, h = make_inputs(6, seed=11)
    e = (e.float() * 1e-3).bfloat16()
    h = (h.float() * 1e-3).bfloat16()
    native = NativeDraftFront(*make_weights(), enorm_eps=enorm_eps, hnorm_eps=hnorm_eps)
    for tp_size in (1, 3, 8):
        for rank in range(tp_size):
            assert_bitwise_equal(
                native.candidate(e, h, tp_size, rank),
                native(e, h, tp_size, rank),
                f"eps=({enorm_eps},{hnorm_eps}) tp={tp_size} rank={rank}",
            )


@requires_cuda
def test_extreme_rows():
    rows = 8
    e, h = make_inputs(rows, seed=5, scale_spread=False)
    e[0].zero_()
    h[1].zero_()
    e[2] = 3.0e4
    h[3] = -2.0e4
    e[4] = 1.0e-30
    h[5].copy_(torch.linspace(-1.0e3, 1.0e3, GLM_H, device="cuda").bfloat16())
    h[6, ::2] = 0.0
    e[7, 100] = 6.0e4
    native = NativeDraftFront(*make_weights())
    for rank in range(GLM_TP):
        assert_bitwise_equal(
            native.candidate(e, h, GLM_TP, rank), native(e, h, GLM_TP, rank), f"extreme rank={rank}"
        )


@requires_cuda
def test_strided_and_gathered_inputs():
    """Draft-loop operands: row slices of a larger buffer and gathered rows."""
    native = NativeDraftFront(*make_weights())
    big_e, big_h = make_inputs(16, seed=9)
    wide = torch.empty(16, 2 * GLM_H, dtype=torch.bfloat16, device="cuda")
    wide[:, GLM_H:] = big_h
    cases = {
        "row_slice": (big_e[3:9], big_h[3:9]),
        "wide_row_stride": (big_e[:6], wide[:6, GLM_H:]),
        "gathered": (
            big_e[torch.tensor([5, 0, 9], device="cuda")],
            big_h[torch.tensor([2, 2, 15], device="cuda")],
        ),
    }
    for name, (e, h) in cases.items():
        for rank in range(GLM_TP):
            assert_bitwise_equal(
                native.candidate(e, h, GLM_TP, rank),
                native(e, h, GLM_TP, rank),
                f"{name} rank={rank}",
            )


@requires_cuda
def test_padded_rows_do_not_leak():
    """Rows past the live count hold stale data; live rows must not depend on them."""
    native = NativeDraftFront(*make_weights())
    live = 2
    e, h = make_inputs(6, seed=21)
    e2, h2 = e.clone(), h.clone()
    e2[live:] = 1.0e4
    h2[live:] = -7.0
    for rank in range(GLM_TP):
        out_a = native.candidate(e, h, GLM_TP, rank)[:live].clone()
        out_b = native.candidate(e2, h2, GLM_TP, rank)
        assert torch.equal(out_a, out_b[:live])
        assert_bitwise_equal(out_b, native(e2, h2, GLM_TP, rank), f"padded rank={rank}")


@requires_cuda
@pytest.mark.parametrize("pdl", ["0", "1"])
def test_cuda_graph_replay(monkeypatch, pdl):
    """Capture once per rank and draft-row count; replay with fresh inputs."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", pdl)
    native = NativeDraftFront(*make_weights())
    for rows in DRAFT_ROWS:
        for rank in range(GLM_TP):
            static_e, static_h = make_inputs(rows, seed=100)
            native.candidate(static_e, static_h, GLM_TP, rank)  # compile outside capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = native.candidate(static_e, static_h, GLM_TP, rank)
            for step in range(8):
                e, h = make_inputs(rows, seed=200 + step)
                static_e.copy_(e)
                static_h.copy_(h)
                graph.replay()
                torch.cuda.synchronize()
                assert_bitwise_equal(
                    static_out,
                    native(static_e, static_h, GLM_TP, rank),
                    f"graph pdl={pdl} rows={rows} rank={rank} step={step}",
                )


@requires_cuda
def test_graph_with_native_neighbors():
    """Embedding lookup -> candidate -> eh_proj GEMM captured in one multi-stream graph."""
    vocab = 4096
    gen = torch.Generator(device="cuda").manual_seed(7)
    table = torch.randn(vocab, GLM_H, device="cuda", generator=gen).bfloat16()
    w_eh = (torch.randn(GLM_H, 2 * GLM_H, device="cuda", generator=gen) * 0.02).bfloat16()
    native = NativeDraftFront(*make_weights())
    width = 2 * GLM_H // GLM_TP
    for rows in DRAFT_ROWS:
        ids = torch.randint(0, vocab, (rows,), device="cuda", generator=gen)
        _, h = make_inputs(rows, seed=31)
        for rank in range(GLM_TP):
            w_r = w_eh[:, rank * width : (rank + 1) * width].contiguous()

            def native_step():
                return F.linear(native(F.embedding(ids, table), h, GLM_TP, rank), w_r)

            def candidate_step():
                return F.linear(native.candidate(F.embedding(ids, table), h, GLM_TP, rank), w_r)

            expected = native_step()
            candidate_step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True), torch.cuda.graph(graph):
                out_native = native_step()
                out_cand = candidate_step()
            for step in range(4):
                ids.copy_(torch.randint(0, vocab, (rows,), device="cuda", generator=gen))
                graph.replay()
                torch.cuda.synchronize()
                expected = native_step()
                assert_bitwise_equal(out_native, expected, f"native-in-graph rank={rank}")
                assert_bitwise_equal(out_cand, expected, f"neighbors rows={rows} rank={rank}")


@requires_cuda
def test_capture_before_compile_falls_back():
    """A configuration first seen inside CUDA-graph capture takes the native path."""
    hidden_size, rank = 1024, 5
    e, h = make_inputs(6, hidden_size, seed=55)
    native = NativeDraftFront(*make_weights(hidden_size))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert native.candidate(e, h, GLM_TP, rank) is None
    assert_bitwise_equal(
        native.candidate(e, h, GLM_TP, rank),
        native(e, h, GLM_TP, rank),
        "eager after capture fallback",
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = native.candidate(e, h, GLM_TP, rank)
    assert static_out is not None
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise_equal(static_out, native(e, h, GLM_TP, rank), "capture after compile")


# --------------------------------------------------------------------------- served neighbours


def make_eh_shard(tp_size: int = GLM_TP, seed: int = 9) -> torch.Tensor:
    """A ``[H, 2H / tp]`` BF16 stand-in for the local ROW-parallel ``eh_proj`` weight shard."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(GLM_H, 2 * GLM_H // tp_size, device="cuda", generator=gen) * 0.02
    ).bfloat16()


@requires_cuda
@pytest.mark.parametrize("pdl", ["0", "1"])
def test_graph_with_served_neighbors(monkeypatch, pdl):
    """LM-head-sized L2 eviction -> embedding -> front -> eh_proj GEMM -> input RMSNorm."""
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", pdl)
    vocab = 4096
    gen = torch.Generator(device="cuda").manual_seed(17)
    table = torch.randn(vocab, GLM_H, device="cuda", generator=gen).bfloat16()
    flush_w = torch.randn(19360, GLM_H, device="cuda", generator=gen).bfloat16()
    flush_x = torch.randn(1, GLM_H, device="cuda", generator=gen).bfloat16()
    native = NativeDraftFront(*make_weights())
    w_eh = make_eh_shard()
    input_norm = RMSNorm(hidden_size=GLM_H, eps=GLM_EPS, dtype=torch.bfloat16).cuda()
    for rows in DRAFT_ROWS:
        ids = torch.randint(0, vocab, (rows,), device="cuda", generator=gen)
        _, h = make_inputs(rows, seed=33)
        for rank in (0, 5):

            def native_step():
                F.linear(flush_x, flush_w)
                return input_norm(F.linear(native(F.embedding(ids, table), h, GLM_TP, rank), w_eh))

            def candidate_step():
                F.linear(flush_x, flush_w)
                front = native.candidate(F.embedding(ids, table), h, GLM_TP, rank)
                return input_norm(F.linear(front, w_eh))

            native_step()
            candidate_step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True), torch.cuda.graph(graph):
                out_native = native_step()
                out_cand = candidate_step()
            for step in range(4):
                ids.copy_(torch.randint(0, vocab, (rows,), device="cuda", generator=gen))
                graph.replay()
                torch.cuda.synchronize()
                assert_bitwise_equal(
                    out_cand, out_native, f"served pdl={pdl} rows={rows} rank={rank}"
                )


# --------------------------------------------------------------------------- fallback


@requires_cuda
def test_unsupported_operands_take_native_path():
    e, h = make_inputs(4)
    we, wh = make_weights()
    flat = torch.empty(4 * GLM_H + 8, dtype=torch.bfloat16, device="cuda")
    misaligned = flat[1 : 1 + 4 * GLM_H].view(4, GLM_H)
    wide = torch.empty(4, 2 * GLM_H, dtype=torch.bfloat16, device="cuda")
    cases = {
        "fp32": (e.float(), h.float(), we.float(), wh.float(), 8, 0),
        "fp16_mixed": (e.half(), h, we, wh, 8, 0),
        "misaligned": (misaligned, h, we, wh, 8, 0),
        "column_stride": (e, wide[:, ::2], we, wh, 8, 0),
        "rows_mismatch": (e, h[:3], we, wh, 8, 0),
        "tp_not_dividing": (e, h, we, wh, 5, 0),
        "rank_out_of_range": (e, h, we, wh, 8, 8),
        "weight_shape": (e, h, we[:-8], wh, 8, 0),
        "cpu": (e.cpu(), h.cpu(), we.cpu(), wh.cpu(), 8, 0),
    }
    for name, (ce, ch, cwe, cwh, tp_size, rank) in cases.items():
        assert unsupported_reason(ce, ch, cwe, cwh, tp_size, rank) is not None, name
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = pack_draft_front(ce, ch, cwe, GLM_EPS, cwh, GLM_EPS, rank, tp_size)
            torch.cuda.synchronize()
        assert out is None, name
        kernels = [
            ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert not kernels, (name, kernels)


@requires_cuda
def test_empty_rows():
    e, h = make_inputs(1)
    we, wh = make_weights()
    out = pack_draft_front(e[:0], h[:0], we, GLM_EPS, wh, GLM_EPS, 3, GLM_TP)
    assert out is not None and out.shape == (0, 2 * GLM_H // GLM_TP)


# --------------------------------------------------------------------------- native norm path


@pytest.fixture
def fresh_norm_path():
    """Recompute the per-device FlashInfer path decision inside the test and after it."""
    native_norm._flashinfer_path_reason.cache_clear()
    yield
    native_norm._flashinfer_path_reason.cache_clear()


def glm_norm(**kwargs) -> RMSNorm:
    """``enorm``/``hnorm`` as ``DeepseekV3MTP`` builds them, on the CPU."""
    return RMSNorm(hidden_size=GLM_H, eps=GLM_EPS, dtype=torch.bfloat16, **kwargs)


def _with_attr(name: str, value) -> RMSNorm:
    norm = glm_norm()
    setattr(norm, name, value)
    return norm


class _OverridingNorm(RMSNorm):
    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states, *args, **kwargs)


class _InheritingNorm(RMSNorm):
    pass


def _module_cases() -> dict:
    """Norm modules whose native forward is not the plain FlashInfer RMSNorm."""
    return {
        "gemma": glm_norm(use_gemma=True),
        "cuda_tile": _with_attr("use_cuda_tile", True),
        "nvfp4": _with_attr("is_nvfp4", True),
        "hp_output": _with_attr("return_hp_output", True),
        "forward_override": _OverridingNorm(hidden_size=GLM_H, eps=GLM_EPS, dtype=torch.bfloat16),
        "foreign_module": torch.nn.RMSNorm(GLM_H, eps=GLM_EPS),
    }


def test_native_norm_module_paths_decline():
    plain = glm_norm()
    assert native_norm.norm_module_reason("enorm", plain) is None
    inheriting = _InheritingNorm(hidden_size=GLM_H, eps=GLM_EPS, dtype=torch.bfloat16)
    assert native_norm.norm_module_reason("hnorm", inheriting) is None
    for case, norm in _module_cases().items():
        assert native_norm.norm_module_reason("enorm", norm) is not None, case
        reason = native_norm_reason(norm, plain)
        assert reason is not None and reason.startswith("enorm"), (case, reason)
        reason = native_norm_reason(plain, norm)
        assert reason is not None and reason.startswith("hnorm"), (case, reason)


@requires_cuda
def test_native_norm_glm_config_takes_mirrored_path(monkeypatch, fresh_norm_path):
    """GLM's enorm/hnorm reach FlashInfer's CuTe DSL RMSNorm, so draft_front serves and is bitwise."""
    import flashinfer.norm as flashinfer_norm

    native = NativeDraftFront(*make_weights())
    assert native_norm_reason(native.enorm, native.hnorm) is None
    calls = {"cute": [], "cuda_jit": 0}
    cute = flashinfer_norm.rmsnorm_cute
    jit = flashinfer_norm.get_norm_module

    def cute_spy(input, *args, **kwargs):
        calls["cute"].append(input.dim())
        return cute(input, *args, **kwargs)

    def jit_spy():
        calls["cuda_jit"] += 1
        return jit()

    monkeypatch.setattr(flashinfer_norm, "rmsnorm_cute", cute_spy)
    monkeypatch.setattr(flashinfer_norm, "get_norm_module", jit_spy)
    for rows in DRAFT_ROWS:
        e, h = make_inputs(rows, seed=rows + 70)
        assert unsupported_reason(e, h, native.enorm.weight, native.hnorm.weight, GLM_TP, 0) is None
        for rank in (0, 4):
            calls["cute"].clear()
            expected = native(e, h, GLM_TP, rank)
            assert calls["cute"] == [2, 2], (rows, rank, calls)
            assert_bitwise_equal(
                native.candidate(e, h, GLM_TP, rank),
                expected,
                f"mirrored path rows={rows} rank={rank}",
            )
    assert calls["cuda_jit"] == 0


@requires_cuda
def test_gemma_native_differs_from_candidate():
    """A gemma enorm computes x * rstd * (w + 1): draft_front would change the result."""
    we, wh = make_weights()
    native = NativeDraftFront(we, wh)
    gemma = glm_norm(use_gemma=True).cuda()
    with torch.no_grad():
        gemma.weight.copy_(we)
    e, h = make_inputs(6, seed=81)
    rank = 0  # an embedding-source rank
    width = 2 * GLM_H // GLM_TP
    native_gemma = gemma(e)[:, :width]
    candidate = native.candidate(e, h, GLM_TP, rank)
    assert not torch.equal(candidate, native_gemma)
    assert native_norm_reason(gemma, native.hnorm) is not None


def _process_case(monkeypatch, case: str) -> None:
    import flashinfer
    import flashinfer.norm as flashinfer_norm

    if case == "flashinfer_unavailable":
        monkeypatch.setattr(rms_norm_module, "IS_FLASHINFER_AVAILABLE", False)
    elif case == "cuda_jit_norm":
        monkeypatch.setattr(flashinfer_norm, "_use_cuda_norm", lambda device: True)
    elif case == "unverified_flashinfer":
        monkeypatch.setattr(flashinfer, "__version__", "0.0.0")
    elif case == "no_norm_dispatch":
        monkeypatch.delattr(flashinfer_norm, "_use_cuda_norm")
    else:
        raise AssertionError(case)


@requires_cuda
@pytest.mark.parametrize(
    "case", ["flashinfer_unavailable", "cuda_jit_norm", "unverified_flashinfer", "no_norm_dispatch"]
)
def test_native_norm_process_paths_fall_back(monkeypatch, fresh_norm_path, case):
    """Per-process conditions under which native RMSNorm leaves FlashInfer's CuTe kernel."""
    native = NativeDraftFront(*make_weights())
    e, h = make_inputs(6, seed=91)
    rank = 4  # a hidden-source rank
    candidate = native.candidate(e, h, GLM_TP, rank)
    assert candidate is not None
    _process_case(monkeypatch, case)
    native_norm._flashinfer_path_reason.cache_clear()
    assert native_norm.flashinfer_path_reason(e.device) is not None, case
    assert native_norm_reason(native.enorm, native.hnorm) is not None, case
    weights = (native.enorm.weight, native.hnorm.weight)
    assert unsupported_reason(e, h, *weights, GLM_TP, rank) is not None, case
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = pack_draft_front(e, h, weights[0], GLM_EPS, weights[1], GLM_EPS, rank, GLM_TP)
        torch.cuda.synchronize()
    assert out is None, case
    kernels = [ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    assert not kernels, (case, kernels)
    if case == "flashinfer_unavailable":
        # The torch path rounds x * rstd to BF16 before the weight multiply.
        torch_path = native(e, h, GLM_TP, rank)
        assert not torch.equal(torch_path, candidate)


@requires_cuda
def test_adapter_native_norm_gate(monkeypatch, fresh_norm_path):
    """The draft layer's adoption binds draft_front only when enorm/hnorm take the mirrored path.

    Builds the built-in ``DeepseekV3MTP`` with the decoder-layer base, ``eh_proj`` and the
    shared head stubbed out, then runs the draft step of its adoption as ``GlmMtpLayer``.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
        reclass,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMtpLayer,
    )
    from tensorrt_llm._torch.models import modeling_deepseekv3 as model_module

    plan_testlib.select(dispatch, "draft_front", monkeypatch)
    monkeypatch.setattr(
        model_module.DeepseekV3DecoderLayer,
        "__init__",
        lambda self, *args, **kwargs: torch.nn.Module.__init__(self),
    )
    monkeypatch.setattr(model_module, "Linear", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        model_module, "DeepseekV3MTPHead", lambda *args, **kwargs: torch.nn.Identity()
    )
    aux_streams = {model_module.AuxStreamType.MoeShared: torch.cuda.Stream()}

    def build(norm_factory=RMSNorm, attention_dp: bool = False, tp_size: int = GLM_TP):
        monkeypatch.setattr(model_module, "RMSNorm", norm_factory)
        config = SimpleNamespace(
            pretrained_config=SimpleNamespace(
                hidden_size=GLM_H,
                moe_intermediate_size=2048,
                n_routed_experts=256,
                n_shared_experts=1,
                num_experts_per_tok=8,
                rms_norm_eps=GLM_EPS,
                torch_dtype=torch.bfloat16,
            ),
            mapping=SimpleNamespace(enable_attention_dp=attention_dp, tp_size=tp_size, tp_rank=0),
            skip_create_weights_in_init=True,
            use_cute_dsl_blockscaling_mm=False,
        )
        layer = model_module.DeepseekV3MTP(config, 78, aux_streams)
        # The stubbed decoder-layer constructor does not keep the model config.
        layer.model_config = config
        reclass(layer, model_module.DeepseekV3MTP, GlmMtpLayer)
        layer._adopt_draft()
        return layer.draft_front

    def factory(name, value):
        def make(**kwargs):
            norm = RMSNorm(**kwargs)
            setattr(norm, name, value)
            return norm

        return make

    assert build() is pack_draft_front
    assert build(functools.partial(RMSNorm, use_gemma=True)) is None
    assert build(factory("use_cuda_tile", True)) is None
    assert build(factory("is_nvfp4", True)) is None
    assert build(attention_dp=True) is None
    assert build(tp_size=1) is None
    monkeypatch.setattr(rms_norm_module, "IS_FLASHINFER_AVAILABLE", False)
    native_norm._flashinfer_path_reason.cache_clear()
    assert build() is None


def test_dispatch_registration(monkeypatch):
    plan_testlib.select(dispatch, "draft_front", monkeypatch)
    assert dispatch.is_enabled("draft_front")
    assert dispatch.get_impl("draft_front", "draft_front") is pack_draft_front
    assert dispatch.get_impl("draft_front", "unsupported_reason") is unsupported_reason
    assert dispatch.get_impl("draft_front", "native_norm_reason") is native_norm_reason
    # draft_front is the draft_front contract only: nothing that changes the eh_proj reduction
    # or the draft layer's numerics is registered under this path.
    with pytest.raises(KeyError):
        dispatch.get_impl("draft_front", "input_norm_fold")


def test_dispatch_default_off(monkeypatch):
    """Without an installed plan (a process that does not build the target) draft_front is off."""
    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    assert not dispatch.is_enabled("draft_front")


# --------------------------------------------------------------------------- real checkpoint


@functools.cache
def _real_fixture() -> dict | None:
    index_path = os.path.join(CKPT, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    from safetensors import safe_open

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    def load(name: str, rows: list[int] | None = None) -> torch.Tensor:
        with safe_open(os.path.join(CKPT, weight_map[name]), framework="pt", device="cpu") as f:
            if rows is None:
                return f.get_tensor(name)
            sl = f.get_slice(name)
            return torch.cat([sl[r : r + 1] for r in rows])

    token_ids = [
        0,
        1,
        2,
        3,
        13,
        198,
        271,
        1000,
        9999,
        12345,
        50000,
        77777,
        100000,
        151329,
        154000,
        154879,
    ]
    return {
        "enorm": load("model.layers.78.enorm.weight").cuda(),
        "hnorm": load("model.layers.78.hnorm.weight").cuda(),
        "eh_proj": load("model.layers.78.eh_proj.weight").cuda(),
        "model_norm": load("model.norm.weight").cuda(),
        "shared_head_norm": load("model.layers.78.shared_head.norm.weight").cuda(),
        "embeds": load("model.embed_tokens.weight", token_ids).cuda(),
        "token_ids": token_ids,
    }


def _real_hidden(rows: int, norm_weight: torch.Tensor, seed: int) -> torch.Tensor:
    """Synthetic pre-norm residual activations through a real output RMSNorm weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pre = torch.randn(rows, GLM_H, device="cuda", generator=gen) * 3.0
    pre[:, 11] *= 40.0
    pre = pre.bfloat16()
    norm = RMSNorm(hidden_size=GLM_H, eps=GLM_EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(norm_weight)
    return norm(pre)


@requires_cuda
@pytest.mark.parametrize("draft_pass", [0, 1])
def test_real_checkpoint_all_ranks(draft_pass):
    fx = _real_fixture()
    if fx is None:
        pytest.skip(f"checkpoint not found at {CKPT}")
    rows = DRAFT_ROWS[0] if draft_pass == 0 else DRAFT_ROWS[1]
    e = fx["embeds"][draft_pass : draft_pass + rows]
    h = _real_hidden(
        rows, fx["model_norm"] if draft_pass == 0 else fx["shared_head_norm"], seed=40 + draft_pass
    )
    native = NativeDraftFront(fx["enorm"], fx["hnorm"])
    width = 2 * GLM_H // GLM_TP
    full_ref = torch.cat(
        [fp64_reference(e, fx["enorm"], GLM_EPS), fp64_reference(h, fx["hnorm"], GLM_EPS)], dim=-1
    )
    for rank in range(GLM_TP):
        out_native = native(e, h, GLM_TP, rank)
        out_cand = native.candidate(e, h, GLM_TP, rank)
        assert_bitwise_equal(out_cand, out_native, f"real pass={draft_pass} rank={rank}")
        ref = full_ref[:, rank * width : (rank + 1) * width]
        for label, out in (("native", out_native), ("candidate", out_cand)):
            err = out.double() - ref
            rel_rms = float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            print(
                f"GLM_MOE_DSA_DRAFT_FRONT_STAT real pass={draft_pass} rank={rank} {label} "
                f"max_abs={float(err.abs().max()):.3e} rel_rms={rel_rms:.3e}"
            )
        # eh_proj consumer: the local ROW-parallel GEMM sees an identical operand.
        w_r = fx["eh_proj"][:, rank * width : (rank + 1) * width].contiguous()
        assert_bitwise_equal(
            F.linear(out_cand, w_r),
            F.linear(out_native, w_r),
            f"real eh_proj partial pass={draft_pass} rank={rank}",
        )


@requires_cuda
def test_real_checkpoint_draft_loop_rows():
    """All 16 real embedding rows through one first pass and later single-row passes."""
    fx = _real_fixture()
    if fx is None:
        pytest.skip(f"checkpoint not found at {CKPT}")
    native = NativeDraftFront(fx["enorm"], fx["hnorm"])
    e_all = fx["embeds"]
    h_all = _real_hidden(e_all.shape[0], fx["shared_head_norm"], seed=77)
    for start in range(0, e_all.shape[0]):
        for rows in (1, 6):
            if start + rows > e_all.shape[0]:
                continue
            e, h = e_all[start : start + rows], h_all[start : start + rows]
            for rank in range(GLM_TP):
                assert_bitwise_equal(
                    native.candidate(e, h, GLM_TP, rank),
                    native(e, h, GLM_TP, rank),
                    f"real loop start={start} rows={rows} rank={rank}",
                )
