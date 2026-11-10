# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB balancer and of the hooks the MoE modules call on it."""

from types import SimpleNamespace

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference

from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer as base
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import AdaptiveMoeLoadBalancer
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertStatistics,
    TimeModel,
    build_route_tables,
    statistics_buffers,
)
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

NUM_EXPERTS, NUM_SLOTS, EP_SIZE, TOP_K = 8, 12, 4, 2
ASSIGNMENTS = [(2 * r + j) % 8 for r in range(4) for j in range(3)]  # rank r, slot j < 3


@pytest.fixture(autouse=True)
def cpu_ops(monkeypatch):
    """The route and statistics ops on CPU tensors: their Python versions."""
    monkeypatch.setattr(
        torch.ops.trtllm, "adaptive_eplb_route_into", reference.route_into, raising=False
    )
    monkeypatch.setattr(torch.ops.trtllm, "adaptive_eplb_record", reference.record, raising=False)


def _balancer(ep_rank: int = 1) -> AdaptiveMoeLoadBalancer:
    config = MoeAdaptiveLoadBalancerConfig(num_slots=NUM_SLOTS)
    config.setup(ep_rank=ep_rank, ep_size=EP_SIZE)
    return AdaptiveMoeLoadBalancer(ep_rank=ep_rank, ep_size=EP_SIZE, config=config, mapping=None)


def _observe(balancer, generation: int, context: int = 0, warmup: bool = False) -> None:
    """The executor's call before a batch of ``generation`` generation and ``context`` context
    requests (no request progress is asked for: no runtime)."""
    balancer.observe_scheduled_batch(generation, not context, warmup, lambda start, stop: (0, 0, 0))


def _counts(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_layers_are_checked_against_the_slots():
    balancer = _balancer()
    with pytest.raises(ValueError, match="multiple of `ep_size`"):
        balancer.add_layer(6, TOP_K, NUM_SLOTS // EP_SIZE)
    with pytest.raises(ValueError, match="local slots"):
        balancer.add_layer(NUM_EXPERTS, TOP_K, 2)


@pytest.mark.parametrize(
    "arch,ep_size,cluster_size,error",
    [
        ("LlamaForCausalLM", 4, 1, "does not support model architecture"),
        ("DeepseekV3ForCausalLM", 1, 1, "requires expert parallelism"),
        ("DeepseekV3ForCausalLM", 4, 2, "is incompatible with smart routing"),
    ],
)
def test_factory_refuses_what_adaptive_eplb_cannot_balance(arch, ep_size, cluster_size, error):
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=ep_size, moe_cluster_size=cluster_size)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[arch]),
        moe_load_balancer=MoeAdaptiveLoadBalancerConfig(num_slots=NUM_SLOTS),
    )
    with pytest.raises(ValueError, match=f"adaptive EPLB {error}"):
        base.maybe_create_moe_load_balancer(model_config, mapping)


def test_statistics_record_nothing_while_the_flag_is_clear():
    off = torch.zeros(1, dtype=torch.int32)
    stats = ExpertStatistics.of(statistics_buffers(1, 4, "cpu"), 0, off)
    stats.record(torch.tensor([[0, 1], [1, 3]], dtype=torch.int32))
    assert not stats.rows.any() and stats.forwards.item() == 0


def test_layer_routes_by_the_installed_tables():
    balancer = _balancer()
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    layer.prepare_device_state(torch.int64, "cpu")
    called = []
    experts = torch.tensor([[7, 0], [0, 5]], dtype=torch.int32)
    slots = layer.route(experts, offset_by_ep_rank=True, overlap_fn=lambda: called.append(1))
    assert called == [1] and slots.dtype == torch.int64 and slots.tolist() == [[10, 0], [0, 7]]
    # Installed tables route the forwards enqueued after them: a replica of expert 1 on slot 5
    # takes all of its rows; an id outside the experts goes to slot num_slots.
    layout = layer.home_layout()
    shares = [1.0] * NUM_SLOTS
    layout[5] = 1
    shares[1] = 0.0
    tables = build_route_tables(layout, shares, NUM_EXPERTS, NUM_SLOTS).packed()
    balancer.route_tables("cpu")[layer.layer_idx].copy_(torch.tensor(tables, dtype=torch.int32))
    ids = torch.tensor([[1, 1], [9, -1]], dtype=torch.int32)
    assert layer.route(ids).tolist() == [[5, 5], [NUM_SLOTS, NUM_SLOTS]]


def test_layer_route_stages_into_the_backend_and_records_the_statistics():
    balancer = _balancer()
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    bucket = SimpleNamespace(
        topk_idx=torch.full((4, TOP_K), 9, dtype=torch.int64),
        topk_weights=torch.full((4, TOP_K + 1), 9.0),
        bucket=4,
    )
    asked = []

    def staging(num_tokens: int, top_k: int):
        asked.append((num_tokens, top_k))
        return bucket if num_tokens < 4 else None

    layer.prepare_device_state(torch.int64, "cpu", staging)
    _observe(balancer, 2)
    experts = torch.tensor([[7, 0], [0, 5]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.25], [1.0, 2.0]])
    layer.update_statistic_with_local_ids(experts, True, True, allreduce=object())
    slots = layer.route(experts, token_final_scales=weights)
    # The slots fill the bucket's first rows and -1 / 0 the other rows, the columns after top_k
    # stay untouched, and the same launch adds the ids to the statistics.
    assert slots.data_ptr() == bucket.topk_idx.data_ptr()
    assert slots.tolist() == [[10, 0], [0, 7]]
    assert bucket.topk_idx[2:].tolist() == [[-1, -1], [-1, -1]]
    assert torch.equal(bucket.topk_weights[:2, :TOP_K], weights)
    assert (bucket.topk_weights[2:, :TOP_K] == 0).all()
    assert (bucket.topk_weights[:, TOP_K] == 9).all()
    first = _counts(2, 0, 0, 0, 0, 1, 0, 1)
    assert torch.equal(layer.statistics.rows, first) and layer.statistics.forwards.item() == 1
    assert asked == [(2, TOP_K)]
    # When the backend declines (a forward it does not stage), the route returns its own tensor;
    # without the weights the backend is not asked.
    four = torch.tensor([[1, 2]] * 4, dtype=torch.int32)
    own = layer.route(four, token_final_scales=torch.ones(4, TOP_K))
    assert own.data_ptr() != bucket.topk_idx.data_ptr() and own.tolist() == [[1, 3]] * 4
    layer.route(experts)
    assert asked == [(2, TOP_K), (4, TOP_K)]
    # The statistics of ids other than the routed ones are added before the route.
    layer.update_local_statistic(four, True, True)
    layer.route(experts)
    assert torch.equal(layer.statistics.rows, first + _counts(0, 4, 4, 0, 0, 0, 0, 0))
    assert layer.statistics.forwards.item() == 2


def _prepare(monkeypatch, gaps, calibrated=True):
    """``prepare_runtime`` of rank 0 of a balancer of one layer whose backend has a time model for
    ``calibrated`` GPUs, the EP ranks reporting ``gaps`` (:meth:`platform_gap`) -> the balancer,
    its layer and what the runtime start-up saw."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer as module

    balancer = _balancer(ep_rank=0)
    balancer.mapping = SimpleNamespace(gpus_per_node=4)
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    seen = {}
    fields = dict(
        per_active_slot_us=1.0,
        per_extra_tile_us=2.0,
        per_row_us=3.0,
        hot_slot_extra_tile_us=4.0,
        scale=1.0,
        tile_rows=8,
    )

    def decode_time_model(expert_bytes: int, device_name: str) -> dict[str, float]:
        seen["args"] = (expert_bytes, device_name)
        return fields

    backend = SimpleNamespace(
        decode_time_model=decode_time_model,
        has_decode_time_model=lambda device_name: calibrated,
        slot_id_dtype=torch.int32,
    )
    transport = SimpleNamespace(expert_bytes=lambda index: 1000 + index, ep_rank=0)
    gathered = []

    def allgather(value):
        """The platform gaps first, then the copy times, of which the slowest 2.0 is agreed."""
        gathered.append(value)
        return gaps if len(gathered) == 1 else [value, 2.0, 1.0, 1.5]

    transport.allgather = allgather
    transport.copy = lambda experts, *args: seen.setdefault("copies", []).append(experts)
    event = SimpleNamespace(record=None, synchronize=lambda: None, elapsed_time=lambda end: 0.5)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA GB300")
    monkeypatch.setattr(module, "find_backends", lambda model, layers: {layer.layer_idx: backend})
    monkeypatch.setattr(module, "PeerSlotTransport", lambda mapping: transport)
    monkeypatch.setattr(module, "peer_copy_gap", lambda mapping: None)
    monkeypatch.setattr(
        module, "migrate_model_weights", lambda *args: seen.update(moved=args) or "stream"
    )

    def prepare_layers(backends, layers, device):
        # CPU tensors here: the route op runs as its Python version.
        for handle in layers:
            handle.prepare_device_state(backends[handle.layer_idx].slot_id_dtype, "cpu")

    monkeypatch.setattr(module, "prepare_layers", prepare_layers)
    monkeypatch.setattr(balancer, "start_runtime", lambda *args: seen.update(start=args))
    balancer.prepare_runtime("model")
    return balancer, layer, seen, fields, backend


def test_prepare_runtime_times_a_layer_copy_and_takes_the_time_model_from_the_backend(monkeypatch):
    balancer, layer, seen, fields, backend = _prepare(monkeypatch, [None] * EP_SIZE)
    assert seen["args"] == (1000, "NVIDIA GB300")
    # Rank 0's free slot 2 of layer 0 takes the home expert of rank 2, untimed and then timed.
    assert seen["copies"] == [[(0, 2, 2, 0)]] * 2
    assert seen["start"] == (TimeModel(**fields), 4, 2.0)
    assert (balancer.transport, balancer.copy_stream) == (seen["moved"][1], "stream")
    assert balancer.off_reason is None
    assert balancer.platform_gap(backend) is None


@pytest.mark.parametrize("calibrated", [True, False])
def test_adaptive_eplb_stays_off_where_any_rank_cannot_run_it(monkeypatch, calibrated):
    """A GPU without a calibrated time model, or EP ranks outside one NVLink domain, turn adaptive
    EPLB off on every rank: no weights move, no runtime starts, the layers route by their home
    layouts and record nothing."""
    gap = "the EP ranks span 2 nodes outside one MNNVL domain"
    balancer, layer, seen, _, backend = _prepare(monkeypatch, [None, gap, None, None], calibrated)
    assert balancer.off_reason == f"{gap} (EP rank 1)"
    assert "moved" not in seen and "start" not in seen and balancer.runtime is None
    if not calibrated:
        assert "no decode time model calibrated on NVIDIA GB300" in balancer.platform_gap(backend)
    _observe(balancer, 4)
    assert not balancer.recording
    experts = torch.tensor([[1, 2], [7, 4]], dtype=torch.int32)
    layer.update_statistic_with_local_ids(experts, True, True)
    # Expert e is home on slot e // 2 * 3 + e % 2 of the 3 slots per rank.
    assert layer.route(experts).tolist() == [[1, 3], [10, 6]]
    assert layer.statistics.forwards.item() == 0
    assert balancer.shutdown() is True


def test_start_runtime_needs_torch_distributed(monkeypatch):
    balancer = _balancer(ep_rank=0)
    balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="needs torch.distributed"):
        balancer.start_runtime(TimeModel(1.0, 1.0, 1.0, 1.0, 1.0, 8), 4)
    assert balancer.runtime is None


def test_forwards_count_on_the_device_without_iter_context():
    balancer = _balancer(ep_rank=0)
    layers = [balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE) for _ in range(2)]
    for layer in layers:
        layer.set_initial_weight_assignments(ASSIGNMENTS)
        layer.prepare_device_state(torch.int32, "cpu")
    balancer.finalize_model()
    experts = torch.tensor([[1, 2]], dtype=torch.int32)

    def forward():
        for layer in layers:
            layer.update_statistic_with_local_ids(experts, True, True)
            layer.route(experts)

    def forwards() -> list[int]:
        return [layer.statistics.forwards.item() for layer in layers]

    _observe(balancer, 4)
    with base.MoeLoadBalancerIterContext(balancer):
        forward()
    forward()  # the first replay after a graph capture runs without the context
    with base.MoeLoadBalancerIterContext(balancer):
        forward()
    assert forwards() == [3, 3]
    # Warmup, context and mixed batches are not recorded: the flag the op reads is cleared.
    for generation, context, warmup in ((4, 0, True), (0, 2, False), (4, 1, False)):
        _observe(balancer, generation, context, warmup)
        assert not balancer.recording
        forward()
    assert forwards() == [3, 3]
    _observe(balancer, 1)
    forward()
    assert forwards() == [4, 4]
    with pytest.raises(RuntimeError):
        balancer.end_iter()
    assert balancer.shutdown() is True


def test_the_layer_handle_is_a_moe_layer_load_balancer():
    layer = _balancer().add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    assert layer.get_load_expert_ids() == [2, 3]
    # Rank r loads experts (2r + j) % 8 into its slots j < 3; its first two slots are its homes.
    assert layer.home_layout() == [0, 1, -1, 2, 3, -1, 4, 5, -1, 6, 7, -1]
    layer.set_initial_weight_assignments(ASSIGNMENTS)
    assert isinstance(layer, base.MoeLayerLoadBalancer)
    assert not layer.is_dynamic_routing() and layer.records_statistics()
    assert layer.stages_routes and not layer.produces_replica_plan


def test_a_moe_layer_runs_a_forward_through_the_load_balancer_hooks():
    from tensorrt_llm._torch.moe.fused_moe.impl_blocks import MoEEplbWeightLayoutMixin
    from tensorrt_llm._torch.moe.fused_moe.interface import MoE
    from tensorrt_llm._torch.moe.fused_moe.quantization import FusedMoEMethodBase

    class Module(MoEEplbWeightLayoutMixin):
        """A MoE backend's load balancer hooks: the mixin's and the MoE layer's."""

        _load_balancer_start_wait_gpu_stage = MoE._load_balancer_start_wait_gpu_stage
        _load_balancer_done_wait_gpu_stage = MoE._load_balancer_done_wait_gpu_stage
        _load_balancer_update_statistic = MoE._load_balancer_update_statistic
        _load_balancer_records_statistics = MoE._load_balancer_records_statistics
        _load_balancer_route = MoE._load_balancer_route
        _load_balancer_uses_replica_plan = MoE._load_balancer_uses_replica_plan
        _load_balancer_start_set_cpu_stage = MoE._load_balancer_start_set_cpu_stage
        _load_balancer_done_set_cpu_stage = MoE._load_balancer_done_set_cpu_stage

        def __init__(self, layer: object) -> None:
            self.layer_load_balancer = layer
            self.allreduce = None
            self._load_balancer_overlap_fn = None

    balancer = _balancer(ep_rank=0)
    layer = balancer.add_layer(NUM_EXPERTS, TOP_K, NUM_SLOTS // EP_SIZE)
    module = Module(layer)
    # Construction, then a backend's weight load, then its post-load.
    assert layer.get_repeat_count() == 1
    layer.set_initial_weight_assignments(ASSIGNMENTS)
    assert not FusedMoEMethodBase.need_load_shared_weights(module)
    module._add_raw_shared_weights_for_unmap([torch.zeros(4)])
    layer.set_initial_weight_assignments(ASSIGNMENTS)
    balancer.register_weight_slots_after_to_cuda()
    balancer.finalize_model()
    layer.prepare_device_state(torch.int32, "cpu")
    _observe(balancer, 1)
    # One forward of two chunks.
    experts = torch.tensor([[0, 5], [7, 7]], dtype=torch.int32)
    for first, last in ((True, False), (False, True)):
        module._load_balancer_start_wait_gpu_stage(first)
        module._load_balancer_done_wait_gpu_stage(first)
        module._load_balancer_update_statistic(experts, first, last)
        slots = module._load_balancer_route(experts, True)
        module._load_balancer_start_set_cpu_stage(last)
        module._load_balancer_done_set_cpu_stage(last)
    assert slots.tolist() == [[0, 7], [10, 10]]
    assert torch.equal(layer.statistics.rows, _counts(2, 0, 0, 0, 0, 2, 0, 4))
    assert layer.statistics.forwards.item() == 1
