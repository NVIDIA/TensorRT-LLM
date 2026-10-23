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

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeAlias, cast

import pytest
import torch
from profiler_test_utils import decode_profiler_metadata

from tensorrt_llm import _profiler as profiler
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEInputRequirement,
    MoERunContext,
    MoEStaticCapability,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm._torch.route_capture import RouteCapture

if TYPE_CHECKING:
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

TraceRecords: TypeAlias = list[dict[str, profiler.MetadataValue]]
TraceRouting: TypeAlias = Callable[[Callable[[], object]], TraceRecords]


@pytest.fixture(autouse=True)
def select_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"moe"}))


@pytest.fixture
def trace_routing(tmp_path: Path) -> Iterator[TraceRouting]:
    """Use the real profiler/observer; metadata expectations are hand-derived."""
    index = 0

    def capture(operation: Callable[[], object]) -> TraceRecords:
        nonlocal index
        index += 1
        execution = tmp_path / f"execution-{index}.json"
        chrome = tmp_path / f"chrome-{index}.json"
        observer = torch.profiler.ExecutionTraceObserver().register_callback(str(execution))
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                execution_trace_observer=observer,
            ) as active:
                operation()
        finally:
            observer.unregister_callback()
        active.export_chrome_trace(str(chrome))
        nodes = [
            node
            for node in json.loads(execution.read_text())["nodes"]
            if node["name"] == "trtllm::moe_routing"
        ]
        events = json.loads(chrome.read_text())["traceEvents"]
        assert sum(event.get("name") == "trtllm::moe_routing" for event in events) == len(nodes)
        records = []
        for node in nodes:
            region, fields = decode_profiler_metadata(node["inputs"]["values"][0])
            assert region == "moe"
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
    use_dp = False
    parallel_size = 1
    reduce_results = False
    enable_alltoall = False
    aux_stream = None
    repeat_idx = 0
    repeat_count = 1
    moe_max_num_tokens = 2
    rank = 3
    mapping = SimpleNamespace(moe_ep_rank=1, moe_ep_size=2, tp_rank=1)

    def __init__(self, *, eplb: bool = False, fold: bool = False, top_k: int = 2) -> None:
        self.ledger: list[str] = []
        self.backend = _Backend(self.ledger)
        self.routing_method = _Routing(self.ledger, top_k)
        self.layer_load_balancer = eplb
        self.apply_router_weight_on_input = fold

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
    @pytest.mark.parametrize("weights_available", [False, True])
    def test_exact_noncontiguous_metadata(
        self,
        layer: int | None,
        weights_available: bool,
        trace_routing: TraceRouting,
    ) -> None:
        ids = torch.tensor([[3, 99, 1, 99], [0, 99, 2, 99]], dtype=torch.int32)[:, ::2]
        weights = torch.tensor([[0.125, 99, 0.75, 99], [0.5, 99, 0.25, 99]])[:, ::2]
        records = trace_routing(
            lambda: profiler.record_moe_routing(
                layer, 4, ids, weights if weights_available else None
            )
        )
        assert records == [
            {
                "moe.layer": layer,
                "moe.num_tokens": 2,
                "moe.num_experts": 4,
                "moe.experts_per_token": 2,
                "moe.selected_expert_ids": [[3, 1], [0, 2]],
                "moe.routing_weights": [[0.125, 0.75], [0.5, 0.25]] if weights_available else None,
            }
        ]
        assert ids.tolist() == [[3, 1], [0, 2]]
        assert weights.tolist() == [[0.125, 0.75], [0.5, 0.25]]

    @pytest.mark.parametrize("selected,active", [(False, True), (True, False), (False, False)])
    def test_inactive_never_inspects_tensors_or_cuda(
        self, selected: bool, active: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ids = torch.tensor([[1, 0]], dtype=torch.int32)
        monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"moe"} if selected else {}))
        monkeypatch.setattr(torch.autograd, "_profiler_enabled", lambda: active)
        inspected: list[str] = []

        def forbidden(*args: object, **kwargs: object) -> None:
            inspected.append("forbidden")
            raise AssertionError("inactive observation must not inspect or initialize")

        monkeypatch.setattr(torch.Tensor, "detach", forbidden)
        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
        monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
        monkeypatch.setattr(profiler, "_encode_metadata", forbidden)
        profiler.record_moe_routing(0, 2, ids, None)
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
            lambda: profiler.record_moe_routing(
                7, 4, torch.empty((0, 2), dtype=torch.int32), torch.empty((0, 2))
            )
        )
        assert len(records) == 1
        assert records[0]["moe.num_tokens"] == 0
        assert records[0]["moe.experts_per_token"] == 2
        assert records[0]["moe.selected_expert_ids"] == []
        assert records[0]["moe.routing_weights"] == []

    @pytest.mark.parametrize("fused", [False, True])
    def test_pre_eplb_trace_and_router_replay_preserve_execution(
        self,
        fused: bool,
        monkeypatch: pytest.MonkeyPatch,
        trace_routing: TraceRouting,
    ) -> None:
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0], [0.0, 5.0, 2.0, 4.0]])
        replay = _replay_buffer()
        monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: replay)
        moe = _Moe(eplb=True)
        outputs = []
        run = _fused if fused else _external
        records = trace_routing(lambda: outputs.append(run(moe, x, logits)))
        assert len(records) == 1
        assert records[0]["moe.selected_expert_ids"] == [[0, 2], [1, 3]]
        assert moe.routing_method.last is not None
        ids, weights = moe.routing_method.last
        assert records[0]["moe.routing_weights"] == weights.tolist()
        assert torch.equal(replay._buf[:2, 0], ids)
        context = moe.backend.contexts[0]
        assert torch.equal(context.token_selected_experts, ids + 4)
        expected_dtype = torch.float32 if fused else torch.bfloat16
        assert context.token_final_scales.dtype == expected_dtype
        assert torch.equal(outputs[0], x)
        enabled_ledger = moe.ledger.copy()
        monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset())
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

    def test_input_weight_fold_keeps_original_trace_weights(
        self, trace_routing: TraceRouting
    ) -> None:
        moe = _Moe(fold=True, top_k=1)
        x = torch.ones((2, 4))
        logits = torch.tensor([[4.0, 1.0, 3.0, 0.0], [0.0, 5.0, 2.0, 4.0]])
        outputs = []
        records = trace_routing(lambda: outputs.append(_external(moe, x, logits)))
        assert moe.routing_method.last is not None
        _, weights = moe.routing_method.last
        assert records[0]["moe.routing_weights"] == weights.tolist()
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
        assert records[0]["moe.selected_expert_ids"] == [[3, 1]]
        assert records[0]["moe.routing_weights"] == [[0.125, 0.75]]
        assert moe.ledger == [
            "start_wait:True",
            "try_fused",
            "run",
            "start_cpu:True",
            "done_cpu:True",
        ]
        assert moe.backend.contexts[0].x is quantized
        assert moe.backend.contexts[0].x_sf is sf

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
        assert records[1]["moe.selected_expert_ids"] == []
        assert records[1]["moe.routing_weights"] == []
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

        monkeypatch.setattr(profiler, "_encode_metadata", fail_encode)

        def run() -> None:
            with pytest.raises(RuntimeError) as caught:
                _external(moe, torch.ones((1, 4)), torch.zeros((1, 4)))
            assert caught.value is original

        assert trace_routing(run) == []

    @pytest.mark.parametrize("fused", [False, True])
    def test_optional_helper_weights_do_not_relax_scheduler_contract(
        self, fused: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
