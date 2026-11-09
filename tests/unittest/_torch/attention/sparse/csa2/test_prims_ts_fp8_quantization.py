# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent unit-scale Q8 checks for the packed native FP8 PrimTS adapter."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_quantization import UnitScaleQuery
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import SwaCacheFormat

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="packed PrimTS FP8 requires SM100, SM103 or SM107",
)


def _query(rows: int = 3, heads: int = 3) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.linspace(-1, 1, rows * heads * 512, device=device)
    # Span E4M3 subnormals, normals and saturating magnitudes in every row.
    magnitudes = torch.tensor([2.0**-12, 2.0**-7, 1.0, 96.0, 900.0], device=device)
    values = values * magnitudes[torch.arange(values.numel(), device=device) % magnitudes.numel()]
    return values.bfloat16().view(rows, 1, heads, 512)


def _expected(q: torch.Tensor) -> torch.Tensor:
    """RN/satfinite E4M3 at unit scale, independently of the PTX converter."""
    return q.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _plan(q: torch.Tensor) -> tuple[UnitScaleQuery, torch.Tensor]:
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    return UnitScaleQuery.plan(q.device, q.shape[0], q.shape[2], q8), q8


@pytest.mark.parametrize(
    "fill", [None, 0.0, 448.0, 450.0, -450.0, 2.0**-12, 2.0**120, 3.3895313892515355e38]
)
@torch.inference_mode()
def test_unit_scale_query_bytes(fill: float | None) -> None:
    q = _query()
    if fill is not None:
        q.fill_(fill)
    plan, q8 = _plan(q)
    plan.run(q, q8)
    torch.testing.assert_close(q8.view(torch.uint8), _expected(q), atol=0, rtol=0)
    # Saturation keeps every converted value finite.
    assert torch.isfinite(q8.float()).all()


@torch.inference_mode()
def test_unit_scale_query_saturates_infinities() -> None:
    q = _query(1, 1)
    q[0, 0, 0, :2] = torch.tensor([float("inf"), float("-inf")], device=q.device)
    plan, q8 = _plan(q)
    plan.run(q, q8)
    torch.testing.assert_close(q8.view(torch.uint8), _expected(q), atol=0, rtol=0)
    assert q8[0, 0, 0, :2].float().tolist() == [448.0, -448.0]


@torch.inference_mode()
def test_unit_scale_query_plans_share_kernels_across_rows() -> None:
    q = _query()
    plans = []
    for rows in (1, 2, 3):
        plan, q8 = _plan(q[:rows])
        plan.run(q[:rows], q8)
        torch.testing.assert_close(q8.view(torch.uint8), _expected(q[:rows]), atol=0, rtol=0)
        plans.append(plan)
    assert all(plan.kernel is plans[0].kernel for plan in plans)


@torch.inference_mode()
def test_unit_scale_query_graph_live_and_allocation_free() -> None:
    q = _query()
    plan, q8 = _plan(q)
    plan.run(q, q8)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    plan.run(q, q8)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(q, q8)
    addresses = (q.data_ptr(), q8.data_ptr())
    q.mul_(-3)
    graph.replay()
    torch.testing.assert_close(q8.view(torch.uint8), _expected(q), atol=0, rtol=0)
    q.zero_()
    graph.replay()
    assert torch.count_nonzero(q8.view(torch.uint8)) == 0
    assert (q.data_ptr(), q8.data_ptr()) == addresses


@torch.inference_mode()
def test_packed_fp8_persistent_formats_have_distinct_plans() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS

    from .test_backend import _backend

    q = torch.zeros((1, 3, 512), device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    helper = CSA2PrimsTS(_backend(3, compute_backend="prims_ts"))
    legacy = helper._get_plan(q.flatten(1), metadata)
    ordinary = helper._get_plan(q.flatten(1), metadata, "swa_fp8")
    assert helper._get_plan(q.flatten(1), metadata, "swa") is legacy
    assert helper._get_plan(q.flatten(1), metadata, "swa_fp8") is ordinary
    assert len(metadata.fmha_plan_caches["csa2_prims_ts"]) == 2
    assert len(metadata.fmha_plan_caches["csa2_prims_ts_arenas"]) == 2
    assert legacy["arena"] is not ordinary["arena"]
    legacy_state, ordinary_state = legacy["wrapper"]._state, ordinary["wrapper"]._state
    assert legacy_state is not None and ordinary_state is not None
    assert legacy_state.swa_cache_format == "swa"
    assert ordinary_state.swa_cache_format == "swa_fp8"
    assert legacy_state.compiled is not ordinary_state.compiled
    # Q staging does not depend on the SWA format.
    assert legacy_state.query.kernel is ordinary_state.query.kernel
    assert legacy["wrapper"].workspace_size_bytes == ordinary["wrapper"].workspace_size_bytes

    # Equal workspace size does not make the two raw-pool ABIs interchangeable.
    query = q.view(1, 1, 3, 512)
    output = torch.empty_like(query)
    for record, expected_width, wrong_width in ((legacy, 528, 512), (ordinary, 512, 528)):
        pool = torch.zeros((1, wrong_width), device=q.device, dtype=torch.uint8)
        prepared = record["metadata"]
        with pytest.raises(ValueError, match=rf"UINT8\[N,{expected_width}\]"):
            record["wrapper"].run(
                query,
                pool,
                prepared.indices,
                prepared.lengths,
                softmax_scale=512**-0.5,
                sinks=torch.zeros((3,), device=q.device),
                out=output,
            )


@pytest.mark.parametrize("swa_cache_format", ["swa", "swa_fp8"])
@torch.inference_mode()
def test_packed_fp8_quantization_waits_for_query(
    monkeypatch: pytest.MonkeyPatch, swa_cache_format: SwaCacheFormat
) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    from .test_backend import _backend, _helper_forward
    from .test_backend import _inputs as backend_inputs

    q = torch.zeros((1, 1, 512), device="cuda", dtype=torch.bfloat16)
    swa = torch.ones((1, 4, 512), device=q.device, dtype=q.dtype)
    sinks = torch.zeros((1,), device=q.device)
    valid = torch.ones((1, 4), device=q.device, dtype=torch.bool)
    args = backend_inputs(q, swa, None, valid, None, sinks, swa_cache_format=swa_cache_format)
    args.output = torch.empty_like(q).flatten(1)
    ready = False

    def wait() -> None:
        nonlocal ready
        ready = True
        q.fill_(450)

    def checked_run(self, *arguments) -> None:
        assert ready, "Q quantization ran before query_ready.wait()"
        original_run(self, *arguments)

    original_run = UnitScaleQuery.run
    monkeypatch.setattr(UnitScaleQuery, "run", checked_run)
    args.sparse_backend_args.query_ready = SimpleNamespace(wait=wait)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    attention = _backend(
        1,
        compute_backend="prims_ts",
        kv_cache_dtype="fp8" if swa_cache_format == "swa_fp8" else "auto",
    )
    _helper_forward(attention, q.flatten(1), metadata, args)
    record = next(iter(metadata.fmha_plan_caches["csa2_prims_ts"].values()))
    state = record["wrapper"]._state
    assert state.swa_cache_format == swa_cache_format
    # The Q written in the wait is the one quantized, saturated at unit scale.
    assert (state.buffers["q8"].float() == 448).all()
