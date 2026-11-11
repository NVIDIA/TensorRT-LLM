# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Routing observation contracts; genuine CUDA graph coverage is local/QA."""

import importlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeAlias, cast

import pytest
import torch
from profiler_test_utils import (
    decode_node_metadata,
    execution_trace_observer,
    profiler_nodes,
    select_regions,
)

from tensorrt_llm import _profiler as profiler
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEInputRequirement,
    MoERunContext,
    MoEStaticCapability,
    PreroutedRouting,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm._torch.route_capture import RouteCapture

if TYPE_CHECKING:
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

TraceRecords: TypeAlias = list[dict[str, profiler.MetadataValue]]
TraceRouting: TypeAlias = Callable[[Callable[[], object]], TraceRecords]


@pytest.fixture(autouse=True)
def select_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    select_regions(monkeypatch, "moe")


@pytest.fixture
def trace_routing(tmp_path: Path) -> Iterator[TraceRouting]:
    """Use the real profiler/observer; metadata expectations are hand-derived."""
    index = 0

    def capture(operation: Callable[[], object]) -> TraceRecords:
        """Capture routing records and check execution/Chrome event counts
        agree.
        """
        nonlocal index
        index += 1
        execution = tmp_path / f"execution-{index}.json"
        chrome = tmp_path / f"chrome-{index}.json"
        with execution_trace_observer(execution) as observer:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                execution_trace_observer=observer,
            ) as active:
                operation()
        active.export_chrome_trace(str(chrome))
        nodes = profiler_nodes(execution, "trtllm::moe_routing", "trtllm::moe_routing_ids")
        events = json.loads(chrome.read_text())["traceEvents"]
        assert sum(
            event.get("name") in {"trtllm::moe_routing", "trtllm::moe_routing_ids"}
            for event in events
        ) == len(nodes)
        records = []
        for node in nodes:
            if not node["inputs"]["values"]:
                records.append({})
                continue
            region, fields = decode_node_metadata(node)
            assert region == "moe"
            if node["name"] == "trtllm::moe_routing":
                assert set(fields) == {
                    "moe.layer",
                    "moe.num_tokens",
                    "moe.num_experts",
                    "moe.experts_per_token",
                }
            records.append(fields)
        return records

    yield capture


class _Routing(DefaultMoeRoutingMethod):
    def __init__(self, ledger: list[str], top_k: int = 2) -> None:
        super().__init__(top_k=top_k, force_enable_pytorch_op=True)
        self.ledger = ledger
        self.last: tuple[torch.Tensor, torch.Tensor] | None = None

    def apply(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.ledger.append("route")
        self.last = super().apply(logits, input_ids)
        return self.last


class _Backend:
    capabilities = MoEStaticCapability()
    input_requirement = MoEInputRequirement(routing_scales_dtype=torch.bfloat16)

    def __init__(self, ledger: list[str]) -> None:
        self.ledger = ledger
        self.contexts: list[MoERunContext] = []
        self.fused_result: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )
        self.error: RuntimeError | None = None

    def _supports_load_balancer(self) -> bool:
        return True

    def try_fused_route_quant(
        self, x: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        self.ledger.append("try_fused")
        return self.fused_result

    def quantize_input(
        self, x: torch.Tensor, *, post_quant_comm: bool = False
    ) -> tuple[torch.Tensor, None]:
        self.ledger.append("quantize")
        return x, None

    def run_moe(
        self, context: MoERunContext, *, workspace: dict[str, object] | None = None
    ) -> torch.Tensor:
        """Record scheduler inputs and preserve any configured backend
        failure.
        """
        self.ledger.append("run")
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return context.x.clone()


class _Moe:
    """Only compute/comm are doubles; the complete real scheduler is exercised."""

    layer_idx = 7
    num_experts = 4
    num_slots = 8
    comm = None
    sort_routing_applies = False
    use_dp = False
    parallel_size = 1
    reduce_results = False
    enable_alltoall = False
    aux_stream = None
    repeat_idx = 0
    repeat_count = 1
    moe_max_num_tokens = 2
    rank = 3
    _enable_perfect_router_expert_ids = False
    mapping = SimpleNamespace(moe_ep_rank=1, moe_ep_size=2, tp_rank=1)

    def __init__(self, *, eplb: bool = False, fold: bool = False, top_k: int = 2) -> None:
        self.ledger: list[str] = []
        self.backend = _Backend(self.ledger)
        self.routing_method = _Routing(self.ledger, top_k)
        self.layer_load_balancer = eplb
        self.apply_router_weight_on_input = fold

    def _maybe_get_perfect_router_expert_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return ids

    def _load_balancer_uses_replica_plan(self) -> bool:
        return False

    def _using_load_balancer(self) -> bool:
        return self.layer_load_balancer

    def _load_balancer_start_wait_gpu_stage(self, first: bool) -> None:
        self.ledger.append(f"start_wait:{first}")

    def _load_balancer_done_wait_gpu_stage(self, first: bool) -> None:
        self.ledger.append(f"done_wait:{first}")

    def _load_balancer_update_statistic(
        self, ids: torch.Tensor, first: bool, last: bool, *, ignore_allreduce: bool
    ) -> None:
        self.ledger.append(f"stats:{first}:{last}:{ignore_allreduce}")

    def _load_balancer_route(self, ids: torch.Tensor, use_dp: bool) -> torch.Tensor:
        self.ledger.append("remap")
        return ids + 4

    def _load_balancer_start_set_cpu_stage(self, last: bool) -> None:
        self.ledger.append(f"start_cpu:{last}")

    def _load_balancer_done_set_cpu_stage(self, last: bool) -> None:
        self.ledger.append(f"done_cpu:{last}")

    @staticmethod
    def split_chunk(count: int, chunks: int) -> list[int]:
        """Distribute token counts evenly while retaining empty trailing
        chunks.
        """
        quotient, remainder = divmod(count, chunks)
        return [quotient + (index < remainder) for index in range(chunks)]


def _external(
    moe: _Moe, x: torch.Tensor, logits: torch.Tensor, row_offset: int | None = 0
) -> torch.Tensor:
    return moe_scheduler.ExternalCommMoEScheduler(cast("ConfigurableMoE", moe))._forward_chunk_impl(
        x, logits, x.dtype, [len(x)], False, True, True, row_offset=row_offset
    )


def _fused(moe: _Moe, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return moe_scheduler.FusedCommMoEScheduler(cast("ConfigurableMoE", moe))._forward_chunk(
        x,
        logits,
        output_dtype=x.dtype,
        all_rank_num_tokens=None,
        do_finalize=True,
    )


def _replay_buffer(device: torch.device | str = "cpu") -> RouteCapture:
    capture = RouteCapture(rank=0)
    capture._buf = torch.full((8, 1, 2), -1, dtype=torch.int32, device=device)
    capture._layer_pos = {7: 0}
    return capture


@pytest.mark.cpu_only
class TestMoeProfilerCPU:
    @pytest.mark.parametrize("layer", [None, 7])
    def test_shape_only_noncontiguous_metadata(
        self, layer: int | None, monkeypatch: pytest.MonkeyPatch, trace_routing: TraceRouting
    ) -> None:
        """Describe routing shape without reading values or copying the source."""
        ids = torch.tensor([[3, 99, 1, 99], [0, 99, 2, 99]], dtype=torch.int32)[:, ::2]

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("shape observation must not read or copy tensor values")

        def run() -> None:
            with monkeypatch.context() as patches:
                for name in ("detach", "cpu", "tolist", "clone"):
                    patches.setattr(torch.Tensor, name, forbidden)
                profiler.record_moe_routing(layer, 4, ids)

        assert trace_routing(run) == [
            {
                "moe.layer": layer,
                "moe.num_tokens": 2,
                "moe.num_experts": 4,
                "moe.experts_per_token": 2,
            }
        ]
        assert ids.tolist() == [[3, 1], [0, 2]]

    @pytest.mark.parametrize("selected,active", [(False, True), (True, False), (False, False)])
    def test_inactive_never_inspects_tensors_or_cuda(
        self, selected: bool, active: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify inactive routing observation never reads tensors or queries
        CUDA.
        """
        ids = torch.tensor([[1, 0]], dtype=torch.int32)
        select_regions(monkeypatch, *(("moe",) if selected else ()))
        monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: active)
        inspected: list[str] = []

        def forbidden(*args: object, **kwargs: object) -> None:
            inspected.append("forbidden")
            raise AssertionError("inactive observation must not inspect or initialize")

        monkeypatch.setattr(torch.Tensor, "detach", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
        monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
        monkeypatch.setattr(profiler, "encode_metadata", forbidden)
        profiler.record_moe_routing(0, 2, ids)
        assert inspected == []

    def test_cpu_active_does_not_query_or_initialize_cuda(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        def forbidden() -> bool:
            raise AssertionError("CPU observation must not query CUDA")

        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
        records = trace_routing(
            lambda: profiler.record_moe_routing(7, 4, torch.empty((0, 2), dtype=torch.int32))
        )
        assert len(records) == 1
        assert records[0]["moe.num_tokens"] == 0
        assert records[0]["moe.experts_per_token"] == 2

    @pytest.mark.parametrize("fused", [False, True])
    def test_pre_eplb_trace_and_router_replay_preserve_execution(
        self,
        fused: bool,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        """Verify pre-EPLB observation preserves replay buffers and scheduler
        ordering.
        """
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0], [0.0, 5.0, 2.0, 4.0]])
        replay = _replay_buffer()
        monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: replay)
        moe = _Moe(eplb=True)
        outputs = []
        run = _fused if fused else _external
        records = trace_routing(lambda: outputs.append(run(moe, x, logits)))
        assert len(records) == 1
        assert replay._buf[:2, 0].tolist() == [[0, 2], [1, 3]]
        assert moe.routing_method.last is not None
        ids, weights = moe.routing_method.last
        assert torch.equal(replay._buf[:2, 0], ids)
        context = moe.backend.contexts[0]
        assert torch.equal(context.token_selected_experts, ids + 4)
        expected_dtype = torch.float32 if fused else torch.bfloat16
        assert context.token_final_scales.dtype == expected_dtype
        torch.testing.assert_close(context.token_final_scales, weights.to(expected_dtype))
        assert torch.equal(outputs[0], x)
        enabled_ledger = moe.ledger.copy()
        select_regions(monkeypatch)
        moe_disabled = _Moe(eplb=True)
        assert torch.equal(run(moe_disabled, x, logits), outputs[0])
        assert moe_disabled.ledger == enabled_ledger
        assert enabled_ledger == [
            "start_wait:True",
            "route",
            "done_wait:True",
            "stats:True:True:False",
            "remap",
            "quantize",
            "run",
            "start_cpu:True",
            "done_cpu:True",
        ]
        assert moe.repeat_idx == 0

    def test_shape_observation_preserves_input_weight_fold(
        self, trace_routing: TraceRouting
    ) -> None:
        moe = _Moe(fold=True, top_k=1)
        x = torch.ones((2, 4))
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0], [0.0, 5.0, 2.0, 4.0]])
        outputs = []
        records = trace_routing(lambda: outputs.append(_external(moe, x, logits)))
        assert len(records) == 1
        assert moe.routing_method.last is not None
        _, weights = moe.routing_method.last
        assert moe.backend.contexts[0].token_final_scales is None
        assert torch.equal(outputs[0], x * weights.to(torch.bfloat16).to(x.dtype))

    def test_materialized_fused_route_quant_is_not_recomputed(
        self, trace_routing: TraceRouting
    ) -> None:
        moe = _Moe()
        ids = torch.tensor([[3, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.125, 0.75]])
        quantized = torch.full((1, 4), 5.0)
        sf = torch.tensor([[2.0]])
        moe.backend.fused_result = ids, weights, quantized, sf
        records = trace_routing(lambda: _external(moe, torch.zeros((1, 4)), torch.zeros((1, 4))))
        assert records[0]["moe.num_tokens"] == 1
        torch.testing.assert_close(moe.backend.contexts[0].token_selected_experts, ids)
        torch.testing.assert_close(
            moe.backend.contexts[0].token_final_scales, weights.to(torch.bfloat16)
        )
        assert moe.ledger == [
            "start_wait:True",
            "try_fused",
            "run",
            "start_cpu:True",
            "done_cpu:True",
        ]
        assert moe.backend.contexts[0].x is quantized
        assert moe.backend.contexts[0].x_sf is sf

    @pytest.mark.parametrize("selected", [False, True])
    @pytest.mark.parametrize("active", [False, True])
    @pytest.mark.parametrize("force_separated", [False, True])
    def test_prerouted_routing_metadata_matches_backend_input(
        self,
        selected: bool,
        active: bool,
        force_separated: bool,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        """Verify prerouted observations match the inputs consumed by the
        backend.
        """
        moe = _Moe()
        monkeypatch.setattr(moe.backend, "_supports_load_balancer", lambda: False)
        monkeypatch.setattr(moe_scheduler, "FORCE_SEPARATED_ROUTING", force_separated)
        select_regions(monkeypatch, *(("moe",) if selected else ()))
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        prerouted = PreroutedRouting(
            tables=torch.empty(0, dtype=torch.int32),
            tactic=[8, 0],
            topk_ids=torch.tensor([[3, 1], [0, 2]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.bfloat16),
        )
        scheduler = moe_scheduler.ExternalCommMoEScheduler(cast("ConfigurableMoE", moe))
        outputs = []

        def run() -> None:
            outputs.append(
                scheduler._forward_chunk_impl(
                    x,
                    torch.zeros((2, 4)),
                    x.dtype,
                    [2],
                    False,
                    True,
                    True,
                    prerouted=prerouted,
                )
            )

        if active:
            records = trace_routing(run)
        else:
            run()
            records = []
        assert records == (
            [
                {
                    "moe.layer": 7,
                    "moe.num_tokens": 2,
                    "moe.num_experts": 4,
                    "moe.experts_per_token": 2,
                }
            ]
            if active and selected
            else []
        )
        assert torch.equal(outputs[0], x)
        context = moe.backend.contexts[0]
        assert context.prerouted is prerouted
        if force_separated:
            assert moe.routing_method.last is not None
            ids, weights = moe.routing_method.last
            assert torch.equal(context.token_selected_experts, ids)
            assert torch.equal(context.token_final_scales, weights.to(torch.bfloat16))
        else:
            assert context.token_selected_experts is None and context.token_final_scales is None
        assert moe.ledger == (
            ["start_wait:True"]
            + (["try_fused", "route"] if force_separated else [])
            + ["quantize", "run", "start_cpu:True", "done_cpu:True"]
        )

    def test_internal_fused_route_remains_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        moe = _Moe()
        monkeypatch.setattr(moe.backend, "_supports_load_balancer", lambda: False)
        monkeypatch.setattr(moe_scheduler, "FORCE_SEPARATED_ROUTING", False)
        assert trace_routing(lambda: _external(moe, torch.ones((1, 4)), torch.zeros((1, 4)))) == []
        assert moe.ledger == [
            "start_wait:True",
            "quantize",
            "run",
            "start_cpu:True",
            "done_cpu:True",
        ]
        assert moe.backend.contexts[0].token_selected_experts is None

    def test_fused_adp_chunks_keep_empty_launches_and_ep_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        """Verify fused ADP strips padding while retaining empty launches and
        EP rank.
        """
        moe = _Moe(eplb=True)
        replay = _replay_buffer()
        monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: replay)
        scheduler = moe_scheduler.FusedCommMoEScheduler(cast("ConfigurableMoE", moe))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0]] * 4)
        outputs = []
        records = trace_routing(
            lambda: outputs.append(
                scheduler.forward(
                    x,
                    logits,
                    do_finalize=True,
                    output_dtype=x.dtype,
                    all_rank_num_tokens=[4, 1],
                    use_dp_padding=True,
                    input_ids=None,
                )
            )
        )
        assert [record["moe.num_tokens"] for record in records] == [1, 0]
        assert [len(context.x) for context in moe.backend.contexts] == [1, 0]
        assert moe.ledger.count("quantize") == moe.ledger.count("run") == 2
        assert torch.equal(outputs[0], x[:1])
        assert replay._buf[0, 0].tolist() == [0, 2]
        assert replay._buf[1:, 0].tolist() == [[-1, -1]] * 7
        assert "stats:True:False:False" in moe.ledger
        assert "stats:False:True:False" in moe.ledger
        assert moe.repeat_idx == 0

    def test_substituted_work_is_observed_but_not_recaptured(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        moe = _Moe()
        replay = _replay_buffer()
        monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: replay)
        records = trace_routing(
            lambda: _external(moe, torch.ones((1, 4)), torch.tensor([[4.0, 1.0, 3.0, 0.0]]), None)
        )
        assert records[0]["moe.num_tokens"] == 1
        assert replay._buf.tolist() == [[[-1, -1]]] * 8

    def test_backend_exception_identity_survives_observation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        moe = _Moe()
        original = RuntimeError("backend failure")
        moe.backend.error = original

        def fail_encode(region: str, metadata: object) -> str:
            raise ValueError("observational metadata failure")

        monkeypatch.setattr(profiler, "encode_metadata", fail_encode)

        def run() -> None:
            with pytest.raises(RuntimeError) as caught:
                _external(moe, torch.ones((1, 4)), torch.zeros((1, 4)))
            assert caught.value is original

        assert trace_routing(run) == [{}]

    @pytest.mark.parametrize("fused", [False, True])
    def test_shape_observation_preserves_scheduler_weight_contract(
        self, fused: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep scheduler weight requirements even though profiling reads only shape."""
        moe = _Moe()

        def missing_weights(
            logits: torch.Tensor, input_ids: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, None]:
            return torch.zeros((1, 2), dtype=torch.int32), None

        monkeypatch.setattr(moe.routing_method, "apply", missing_weights)
        run = _fused if fused else _external
        with pytest.raises(AttributeError):
            run(moe, torch.ones((1, 4)), torch.zeros((1, 4)))
        assert moe.backend.contexts == []


@pytest.mark.cpu_only
def test_completed_cpu_ids_keep_layer_mapping_and_step(
    trace_routing: TraceRouting,
) -> None:
    """Serialize completed R3 rows in top-k order with their original attribution."""
    host_rows = torch.tensor(
        [[[0, 99, 1, 99], [3, 99, 2, 99]], [[2, 99, 3, 99], [1, 99, 0, 99]]],
        dtype=torch.int32,
    )[:, :, ::2]
    records = trace_routing(
        lambda: profiler.record_moe_routing_ids(
            host_rows, step=37, rank=2, layer_pos=((7, 1), (3, 0))
        )
    )
    assert sorted(records, key=lambda fields: fields["moe.layer"]) == [
        {
            "moe.layer": layer,
            "moe.step": 37,
            "moe.rank": 2,
            "moe.num_tokens": 2,
            "moe.experts_per_token": 2,
            "moe.selected_expert_ids": values,
        }
        for layer, values in ((3, [[0, 1], [2, 3]]), (7, [[3, 2], [1, 0]]))
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows,top_k", [(65, 2), (3, 2048)])
def test_completed_cpu_ids_preserve_all_rows_and_values(
    rows: int, top_k: int, trace_routing: TraceRouting
) -> None:
    """Preserve every route beyond the former row and total-ID thresholds."""
    host_rows = torch.arange(rows * top_k, dtype=torch.int32).reshape(rows, 1, top_k)
    records = trace_routing(
        lambda: profiler.record_moe_routing_ids(host_rows, step=None, rank=0, layer_pos=((7, 0),))
    )
    assert records == [
        {
            "moe.layer": 7,
            "moe.step": None,
            "moe.rank": 0,
            "moe.num_tokens": rows,
            "moe.experts_per_token": top_k,
            "moe.selected_expert_ids": host_rows[:, 0].tolist(),
        }
    ]


@pytest.mark.cpu_only
def test_completed_integer_ids_avoid_recursive_python_validation(
    monkeypatch: pytest.MonkeyPatch, trace_routing: TraceRouting
) -> None:
    """CPU integer tensors guarantee ID types without another walk over every ID."""
    host_rows = torch.arange(8192, dtype=torch.int32).reshape(128, 1, 64)
    validate = profiler._validate_metadata_value
    visited = 0

    def counted(value: profiler.MetadataValue, *args: object) -> None:
        nonlocal visited
        visited += 1
        validate(value, *args)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed CPU IDs must not add copies, finite reductions, or waits")

    monkeypatch.setattr(profiler, "_validate_metadata_value", counted)
    for method in ("cpu", "cuda", "to", "copy_", "item"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    records = trace_routing(
        lambda: profiler.record_moe_routing_ids(host_rows, step=3, rank=1, layer_pos=((7, 0),))
    )
    assert records[0]["moe.selected_expert_ids"] == host_rows[:, 0].tolist()
    assert visited < 32


@pytest.mark.cpu_only
def test_completed_routing_ids_reject_floating_rows(
    trace_routing: TraceRouting,
) -> None:
    """Only completed CPU integer rows qualify for trusted ID serialization."""
    rows = torch.tensor([[[1.5, float("nan")]]])
    assert trace_routing(
        lambda: profiler.record_moe_routing_ids(rows, step=1, rank=0, layer_pos=((7, 0),))
    ) == [{}]


@pytest.mark.cpu_only
@pytest.mark.parametrize("selected,active", [(False, True), (True, False), (False, False)])
def test_completed_ids_inactive_never_inspect_rows(
    selected: bool, active: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep completed-copy observation lazy when its region or profiler is inactive."""
    select_regions(monkeypatch, *(("moe",) if selected else ()))
    monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: active)
    profiler.record_moe_routing_ids(
        cast(torch.Tensor, object()), step=1, rank=0, layer_pos=((7, 0),)
    )


@pytest.mark.cpu_only
def test_completed_ids_metadata_failure_does_not_mutate_host_rows(
    monkeypatch: pytest.MonkeyPatch, trace_routing: TraceRouting
) -> None:
    """Drop a failed annotation without changing completed R3 data or caller flow."""
    host_rows = torch.tensor([[[3, 1]]], dtype=torch.int32)

    def fail(region: str, metadata: object) -> str:
        raise ValueError("failed completed-ID annotation")

    monkeypatch.setattr(profiler, "encode_metadata", fail)
    assert trace_routing(
        lambda: profiler.record_moe_routing_ids(host_rows, step=1, rank=0, layer_pos=((7, 0),))
    ) == [{}]
    assert host_rows.tolist() == [[[3, 1]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local/QA genuine CUDA graph gate")
def test_graph_capture_skips_only_profiler_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real current-stream capture and replay, with the real RouterReplay write."""
    moe = _Moe()
    moe.backend.input_requirement = MoEInputRequirement(routing_scales_dtype=torch.float32)
    x = torch.ones((1, 4), device="cuda")
    logits = torch.tensor([[4.0, 1.0, 3.0, 0.0]], device="cuda")
    replay = _replay_buffer("cuda")
    monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: replay)
    _external(moe, x, logits)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _external(moe, x, logits)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    serialized: list[str] = []

    def forbidden(self: torch.Tensor) -> torch.Tensor:
        serialized.append("cpu/tolist")
        raise AssertionError("no blocking profiler CPU serialization in a graph")

    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as graph_patches:
        graph_patches.setattr(torch.Tensor, "cpu", forbidden)
        graph_patches.setattr(torch.Tensor, "tolist", forbidden)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
            with torch.cuda.graph(graph, stream=stream):
                assert torch.cuda.is_current_stream_capturing()
                output = _external(moe, x, logits)
        assert serialized == []
    logits.copy_(torch.tensor([[0.0, 4.0, 1.0, 3.0]], device="cuda"))
    x.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, x)
    assert replay._buf[0, 0].tolist() == [1, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local/QA real R3 D2H gate")
def test_completed_r3_copy_emits_ids_without_an_additional_d2h(
    monkeypatch: pytest.MonkeyPatch, trace_routing: TraceRouting
) -> None:
    """Annotate the existing completed host copy while retaining the R3 commit."""
    from tensorrt_llm._torch.route_capture import RouteCopier

    device_rows = torch.tensor([[[3, 1]], [[0, 2]]], dtype=torch.int32, device="cuda")
    copier = RouteCopier(device_rows.device, tuple(device_rows.shape))
    copies: list[int] = []
    committed: list[object] = []
    original_copy = torch.Tensor.copy_

    def copy(
        destination: torch.Tensor, source: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if destination.device.type == "cpu" and source.is_cuda:
            copies.append(source.numel())
        return original_copy(destination, source, *args, **kwargs)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed R3 data must not trigger Tensor.cpu()")

    def commit(rows: torch.Tensor, layout: list) -> None:
        committed.append(rows.tolist())

    def ready(rows: torch.Tensor) -> None:
        profiler.record_moe_routing_ids(rows, step=17, rank=2, layer_pos=((7, 0),))

    def run() -> None:
        with monkeypatch.context() as patches:
            patches.setattr(torch.Tensor, "copy_", copy)
            patches.setattr(torch.Tensor, "cpu", forbidden)
            copier.stage(device_rows, 2, [(1, 0), (1, 1)], commit, on_ready=ready)
            copier.drain(commit, force=True)

    assert trace_routing(run) == [
        {
            "moe.layer": 7,
            "moe.step": 17,
            "moe.rank": 2,
            "moe.num_tokens": 2,
            "moe.experts_per_token": 2,
            "moe.selected_expert_ids": [[3, 1], [0, 2]],
        }
    ]
    assert copies == [4]
    assert committed == [[[[3, 1]], [[0, 2]]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local/QA CUDA shape/CPU-only gate")
def test_cuda_shape_only_scope_and_cpu_ids_device_guard(
    monkeypatch: pytest.MonkeyPatch, trace_routing: TraceRouting
) -> None:
    """Read only CUDA shape and reject CUDA values at the completed-CPU boundary."""
    device_rows = torch.tensor([[[3, 1]]], dtype=torch.int32, device="cuda")
    ids = device_rows[:, 0, :]

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("these observation boundaries must not read CUDA values")

    def run() -> None:
        with monkeypatch.context() as patches:
            for name in ("detach", "cpu", "tolist", "clone"):
                patches.setattr(torch.Tensor, name, forbidden)
            profiler.record_moe_routing(7, 4, ids)
            profiler.record_moe_routing_ids(device_rows, step=1, rank=0, layer_pos=((7, 0),))

    assert trace_routing(run) == [
        {
            "moe.layer": 7,
            "moe.num_tokens": 1,
            "moe.num_experts": 4,
            "moe.experts_per_token": 2,
        },
        {},
    ]


@pytest.mark.cpu_only
def test_moe_observation_is_fullgraph_compile_safe() -> None:
    def forward(logits: torch.Tensor) -> torch.Tensor:
        weights, ids = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1)
        profiler.record_moe_routing(None, logits.shape[-1], ids)
        return weights + 1

    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    expected = forward(logits)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        actual = compiled(logits)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("selected", [False, True])
def test_unsupported_moe_paths_warn_once(monkeypatch: pytest.MonkeyPatch, selected: bool) -> None:
    """Verify unsupported routing paths emit one warning per selected path."""
    from tensorrt_llm.logger import logger

    select_regions(monkeypatch, *(("moe",) if selected else ()))
    monkeypatch.setattr(logger, "_appeared_keys", set())
    messages: list[str] = []
    monkeypatch.setattr(logger, "log", lambda level, *parts: messages.append(" ".join(parts)))
    for backend in ("VanillaMoE", "TritonFusedMoE", "Step3p7 Python clamp path"):
        for _ in range(2):
            profiler.warn_unsupported_moe_routing(backend)
    assert len(messages) == (3 if selected else 0)
    assert all("ConfigurableMoE scheduler" in message for message in messages)


@pytest.mark.cpu_only
@pytest.mark.parametrize("quant_algo", ["FP8", "FP8_PER_CHANNEL_PER_TOKEN"])
@pytest.mark.parametrize("clamp", [7.0, 0.0, None], ids=["clamp", "zero-clamp", "no-clamp"])
def test_step3p7_warns_only_for_python_expert_path(
    monkeypatch: pytest.MonkeyPatch, quant_algo: str, clamp: float | None
) -> None:
    """Verify Step3p7 warns only when Python executes the routed experts."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models import modeling_step3p7 as step3p7
    from tensorrt_llm.models.modeling_utils import QuantConfig

    text_config = PretrainedConfig(
        num_hidden_layers=1,
        hidden_size=4,
        moe_num_experts=2,
        moe_top_k=2,
        moe_intermediate_size=8,
        torch_dtype=torch.float32,
        swiglu_limits=[clamp],
    )
    model_config = ModelConfig(
        pretrained_config=text_config, quant_config=QuantConfig(quant_algo=quant_algo)
    )
    monkeypatch.delenv("STEP3P7_PYTHON_FALLBACK_THRESHOLD", raising=False)
    # Leave path selection and warning ownership real; avoid allocating experts.
    monkeypatch.setattr(step3p7, "Linear", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(step3p7, "create_moe", lambda **kwargs: torch.nn.Identity())
    allocations: list[torch.dtype] = []
    monkeypatch.setattr(
        step3p7.Step3p7MoE,
        "_allocate_clamp_buffers",
        lambda self, dtype: allocations.append(dtype),
    )
    warnings: list[str] = []
    monkeypatch.setattr(step3p7, "warn_unsupported_moe_routing", warnings.append)

    moe = step3p7.Step3p7MoE(model_config, layer_idx=0, aux_stream_dict={})

    use_python = clamp == 7.0
    assert moe._use_python_experts is use_python
    assert warnings == (["Step3p7 Python clamp path"] if use_python else [])
    assert allocations == ([torch.float32] if use_python else [])


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend_name", ["VanillaMoE", "TritonFusedMoE", "CutlassFusedMoE"])
def test_create_moe_backend_warns_for_standalone_subclasses(
    monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Verify unsupported standalone backend subclasses warn before
    construction.
    """
    factory = importlib.import_module("tensorrt_llm._torch.moe.fused_moe.create_moe")
    backend_cls = type(f"Test{backend_name}", (getattr(factory, backend_name),), {})
    warnings: list[str] = []
    monkeypatch.setattr(factory, "warn_unsupported_moe_routing", warnings.append)

    def stop_after_warning(*args: object, **kwargs: object) -> None:
        raise RuntimeError("shape derivation reached")

    # Backend allocation is downstream of the real factory warning gate.
    monkeypatch.setattr(factory, "derive_moe_layer_shapes", stop_after_warning)
    with pytest.raises(RuntimeError, match="shape derivation reached"):
        factory.create_moe_backend(backend_cls, DefaultMoeRoutingMethod(top_k=2))

    assert warnings == ([] if backend_name == "CutlassFusedMoE" else [backend_cls.__name__])
