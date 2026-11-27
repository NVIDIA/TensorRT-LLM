# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FusedCommMoEScheduler's ``try_fused_route_quant`` hook: every collaborator is a spy appending to one ordered log."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import pytest
import torch

import tensorrt_llm._torch.moe.fused_moe.moe_resolution  # noqa: F401  (registers every impl)
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler as ms
from tensorrt_llm._torch.moe.fused_moe.impl_base import MoEImplBase
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEFusedRouteQuant
from tensorrt_llm._torch.moe.fused_moe.impl_identity import MOE_IMPL_REGISTRY
from tensorrt_llm._torch.moe.fused_moe.interface import MoESchedulerKind
from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import Mode

pytestmark = pytest.mark.cpu_only

NUM_SLOTS = 16
TOP_K = 4
HIDDEN = 8
LAYER_IDX = 3
ROW_OFFSET = 5
PADDED_ROWS = 9
# Model-specific variants that offer the hook (registered by their model families).
VARIANT_KERNEL_NAMES = frozenset({"mega_moe_local"})
# Stock fused-comm impls that offer the hook themselves.
OFFERING_STOCK_IMPLS = frozenset({"trtllm.cutedsl.mega_moe.nvfp4"})


def test_only_the_offering_stock_fused_comm_impls_override_the_hook():
    fused_comm = [
        (identity, MOE_IMPL_REGISTRY.lookup(identity))
        for identity in MOE_IMPL_REGISTRY.identities()
        if getattr(MOE_IMPL_REGISTRY.lookup(identity), "scheduler_kind", None)
        == MoESchedulerKind.FUSED_COMM
    ]
    stock = [(i, cls) for i, cls in fused_comm if i.kernel_name not in VARIANT_KERNEL_NAMES]
    assert {i.canonical() for i, _ in stock} >= {
        "trtllm.cutedsl.mega_moe.nvfp4",
        "deepgemm.cuda.mega_moe.w4a8_mxfp4_mxfp8",
    }
    for identity, cls in stock:
        assert issubclass(cls, MoEImplBase), identity.canonical()
        inherits = cls.try_fused_route_quant is MoEImplBase.try_fused_route_quant
        assert inherits != (identity.canonical() in OFFERING_STOCK_IMPLS), identity.canonical()


@dataclass
class Scenario:
    num_tokens: int = 6
    aux: bool = False
    eplb: bool = False
    calibrator_mode: Mode = Mode.NONE
    capture: bool = False
    fused_prepare: bool = False

    @property
    def hook_allowed(self) -> bool:
        """Whether the scheduler offers the backend the fused launch: rows to route and no routing
        aux. A load balancer's slots and REPLAY rewrite the routes after the launch (the backends
        restage the rewritten ids), so neither keeps the backend from being consulted."""
        return self.num_tokens > 0 and not self.aux


class _Log(list):
    def names(self) -> List[str]:
        return [name for name, _ in self]

    def args(self, name: str):
        (hit,) = [args for n, args in self if n == name]
        return hit


def _routing_outputs(num_tokens: int):
    ids = torch.arange(num_tokens * TOP_K, dtype=torch.int64).reshape(num_tokens, TOP_K)
    ids = ids % NUM_SLOTS
    scales = torch.linspace(0.1, 0.9, num_tokens * TOP_K, dtype=torch.bfloat16)
    return ids, scales.reshape(num_tokens, TOP_K)


class _Routing:
    experts_per_token = TOP_K

    def __init__(self, log: _Log):
        self.log = log

    def apply(self, logits, input_ids):
        self.log.append(("routing.apply", (logits, input_ids)))
        return _routing_outputs(logits.shape[0])

    def apply_with_aux(self, logits, input_ids, aux):
        self.log.append(("routing.apply_with_aux", (logits, input_ids, aux)))
        return _routing_outputs(logits.shape[0])


class _Backend:
    """Spy backend; ``fused`` is what ``try_fused_route_quant`` returns."""

    scheduler_kind = MoESchedulerKind.FUSED_COMM
    fused_route_quant_enabled = True

    def __init__(self, log: _Log, fused=None, fused_prepare: bool = False):
        self.log = log
        self.fused = fused
        self.fused_prepare = fused_prepare
        self.quantized = (torch.full((1, HIDDEN), 7, dtype=torch.uint8), torch.ones(1))

    def try_fused_route_quant(self, x, router_logits):
        self.log.append(("backend.try_fused_route_quant", (x, router_logits)))
        return self.fused

    def supports_fused_prepare(self):
        return self.fused_prepare

    def quantize_input(self, x):
        self.log.append(("backend.quantize_input", (x,)))
        return self.quantized

    def set_adaptive_launch_tokens(self, num_tokens):
        self.log.append(("backend.set_adaptive_launch_tokens", (num_tokens,)))

    def run_moe(self, ctx):
        self.log.append(("backend.run_moe", (ctx,)))
        return torch.zeros(1)


class _LoadBalancer:
    def __init__(self, log: _Log):
        self.log = log


def _fake_moe(log: _Log, scn: Scenario, backend: _Backend):
    lb = _LoadBalancer(log) if scn.eplb else None

    def _hook(name, result=None):
        def _call(*args, **kwargs):
            log.append((f"moe.{name}", args))
            return result(*args) if result is not None else None

        return _call

    return SimpleNamespace(
        apply_router_weight_on_input=False,
        mapping=SimpleNamespace(moe_ep_rank=0),
        layer_idx=LAYER_IDX,
        num_slots=NUM_SLOTS,
        use_dp=True,
        routing_method=_Routing(log),
        backend=backend,
        layer_load_balancer=lb,
        _using_load_balancer=lambda: lb is not None,
        _using_dynamic_load_balancer=lambda: False,
        _warmup_placement_route_counters=lambda row_offset: None,
        _load_balancer_count_warmup=_hook("lb_count_warmup"),
        _load_balancer_uses_replica_plan=lambda: False,
        _maybe_get_perfect_router_expert_ids=lambda ids: ids,
        _load_balancer_start_wait_gpu_stage=_hook("lb_start_wait_gpu_stage"),
        _load_balancer_done_wait_gpu_stage=_hook("lb_done_wait_gpu_stage"),
        _load_balancer_update_statistic=_hook("lb_update_statistic"),
        _load_balancer_route=_hook("lb_route", lambda ids, use_dp: (ids + 1) % NUM_SLOTS),
        _load_balancer_start_set_cpu_stage=_hook("lb_start_set_cpu_stage"),
        _load_balancer_done_set_cpu_stage=_hook("lb_done_set_cpu_stage"),
    )


@pytest.fixture
def harness(monkeypatch):
    def _install(log: _Log, scn: Scenario):
        class _Capture:
            def capture(self, layer_idx, ids, row_offset):
                log.append(("r3.capture", (layer_idx, ids, row_offset)))

        class _Statistic:
            @staticmethod
            def set_layer(layer_idx):
                log.append(("stat.set_layer", (layer_idx,)))

            @staticmethod
            def maybe_add_info(num_slots, slots):
                log.append(("stat.maybe_add_info", (num_slots, slots)))

        class _Calibrator:
            mode = scn.calibrator_mode

            def maybe_collect_or_replay_slots(self, num_slots, slots):
                log.append(("calibrator.collect_or_replay", (num_slots, slots)))
                if self.mode == Mode.REPLAY:
                    return torch.flip(slots, dims=(0,))
                return slots

        capture = _Capture() if scn.capture else None
        calibrator = _Calibrator()
        monkeypatch.setattr(ms, "get_active_route_capture", lambda: capture)
        monkeypatch.setattr(ms, "ExpertStatistic", _Statistic)
        monkeypatch.setattr(ms, "get_calibrator", lambda: calibrator)

    return _install


def _run(harness, scn: Scenario, fused=None, backend_cls=None):
    log = _Log()
    harness(log, scn)
    backend = (backend_cls or _Backend)(log, fused=fused, fused_prepare=scn.fused_prepare)
    moe = _fake_moe(log, scn, backend)
    x = torch.randn(PADDED_ROWS, HIDDEN, dtype=torch.bfloat16)
    logits = torch.randn(PADDED_ROWS, NUM_SLOTS, dtype=torch.float32)
    input_ids = torch.arange(PADDED_ROWS)
    aux = torch.randn(PADDED_ROWS, 2) if scn.aux else None
    ms.FusedCommMoEScheduler(moe)._forward_chunk(
        x,
        logits,
        output_dtype=torch.bfloat16,
        all_rank_num_tokens=[scn.num_tokens, PADDED_ROWS],
        do_finalize=True,
        input_ids=input_ids,
        row_offset=ROW_OFFSET,
        routing_aux=aux,
    )
    return log, backend, x, logits


def _two_call_sequence(scn: Scenario, consulted: bool) -> List[str]:
    """The call order of the routing-then-quantize path, hook entry included iff consulted."""
    seq = ["moe.lb_start_wait_gpu_stage", "backend.set_adaptive_launch_tokens"]
    if consulted:
        seq.append("backend.try_fused_route_quant")
    if scn.num_tokens > 0:
        seq.append("routing.apply_with_aux" if scn.aux else "routing.apply")
        if scn.capture:
            seq.append("r3.capture")
    if scn.eplb:
        if scn.num_tokens > 0:
            seq.append("moe.lb_count_warmup")
        seq += ["moe.lb_done_wait_gpu_stage", "moe.lb_update_statistic", "moe.lb_route"]
    seq += ["stat.set_layer", "stat.maybe_add_info", "calibrator.collect_or_replay"]
    if not scn.fused_prepare:
        seq.append("backend.quantize_input")
    seq += [
        "backend.run_moe",
        "moe.lb_start_set_cpu_stage",
        "moe.lb_done_set_cpu_stage",
    ]
    return seq


DECLINE_SCENARIOS = {
    "plain": Scenario(),
    "zero_rows": Scenario(num_tokens=0),
    "aux": Scenario(aux=True),
    "eplb": Scenario(eplb=True),
    "replay": Scenario(calibrator_mode=Mode.REPLAY),
    "r3_eplb": Scenario(capture=True, eplb=True),
    "fused_prepare": Scenario(fused_prepare=True),
}


@pytest.mark.parametrize("name", list(DECLINE_SCENARIOS))
def test_declining_backend_sees_the_two_call_path(harness, name):
    """Outside the contract an offering backend is not consulted at all (its result would poison the step)."""
    scn = DECLINE_SCENARIOS[name]
    fused = None
    if not scn.hook_allowed:
        ids, scales = _routing_outputs(max(scn.num_tokens, 1))
        fused = MoEFusedRouteQuant(ids, scales, torch.empty(0), torch.empty(0))
    log, backend, x, logits = _run(harness, scn, fused=fused)
    assert log.names() == _two_call_sequence(scn, consulted=scn.hook_allowed)

    n = scn.num_tokens
    assert log.args("backend.set_adaptive_launch_tokens") == (PADDED_ROWS,)
    if scn.hook_allowed:
        hook_x, hook_logits = log.args("backend.try_fused_route_quant")
        assert torch.equal(hook_x, x[:n]) and torch.equal(hook_logits, logits[:n])

    ctx = log.args("backend.run_moe")[0]
    ref_ids, ref_scales = _routing_outputs(n)
    ref_slots = ref_ids.to(torch.int32)
    if scn.eplb:
        ref_slots = (ref_slots + 1) % NUM_SLOTS
    if scn.calibrator_mode == Mode.REPLAY:
        ref_slots = torch.flip(ref_slots, dims=(0,))
    assert ctx.token_selected_experts.dtype == torch.int32
    assert torch.equal(ctx.token_selected_experts, ref_slots)
    assert ctx.token_final_scales.dtype == torch.float32
    assert torch.equal(ctx.token_final_scales, ref_scales.to(torch.float32))
    if scn.fused_prepare:
        assert torch.equal(ctx.x, x[:n]) and ctx.x_sf is None
    else:
        assert ctx.x is backend.quantized[0] and ctx.x_sf is backend.quantized[1]
    if scn.capture and n > 0:
        layer_idx, captured, row_offset = log.args("r3.capture")
        assert (layer_idx, row_offset) == (LAYER_IDX, ROW_OFFSET)
        assert captured.dtype == torch.int32 and torch.equal(captured, ref_ids.to(torch.int32))


@pytest.mark.parametrize(
    "scn",
    [Scenario(), Scenario(capture=True), Scenario(calibrator_mode=Mode.COLLECT)],
    ids=["plain", "r3", "collect"],
)
def test_accepted_fused_result_replaces_routing_and_quantize_only(harness, scn):
    ids, scales = _routing_outputs(scn.num_tokens)
    fused_x = torch.full((scn.num_tokens, HIDDEN // 2), 3, dtype=torch.uint8)
    fused_sf = torch.full((scn.num_tokens,), 2, dtype=torch.uint8)
    log, _, x, logits = _run(harness, scn, fused=MoEFusedRouteQuant(ids, scales, fused_x, fused_sf))

    expected = [
        name
        for name in _two_call_sequence(scn, consulted=True)
        if name not in ("routing.apply", "backend.quantize_input")
    ]
    assert log.names() == expected

    hook_x, hook_logits = log.args("backend.try_fused_route_quant")
    assert torch.equal(hook_x, x[: scn.num_tokens])
    assert torch.equal(hook_logits, logits[: scn.num_tokens])

    int32_ids = ids.to(torch.int32)
    for hook in ("stat.maybe_add_info", "calibrator.collect_or_replay"):
        num_slots, slots = log.args(hook)
        assert num_slots == NUM_SLOTS
        assert slots.dtype == torch.int32 and torch.equal(slots, int32_ids)
    if scn.capture:
        layer_idx, captured, row_offset = log.args("r3.capture")
        assert (layer_idx, row_offset) == (LAYER_IDX, ROW_OFFSET)
        assert captured.dtype == torch.int32 and torch.equal(captured, int32_ids)

    ctx = log.args("backend.run_moe")[0]
    assert ctx.token_selected_experts.dtype == torch.int32
    assert torch.equal(ctx.token_selected_experts, int32_ids)
    assert ctx.token_final_scales.dtype == torch.float32
    assert torch.equal(ctx.token_final_scales, scales.to(torch.float32))
    assert ctx.x is fused_x and ctx.x_sf is fused_sf


class _PartialsBackend(_Backend):
    """A backend that also takes the router logits as ``RouterLogitPartials``."""

    def takes_router_logit_partials(self):
        return True


@pytest.mark.parametrize("mode", [Mode.NONE, Mode.COLLECT, Mode.REPLAY])
def test_logit_partials_are_advertised_whatever_the_calibrator_mode(harness, mode):
    """The chunk gate is the routing aux and the replica plan alone; REPLAY rewrites the slots
    after the launch, so it does not stop the fused launch from routing."""
    scn = Scenario(calibrator_mode=mode)
    log = _Log()
    harness(log, scn)
    moe = _fake_moe(log, scn, _PartialsBackend(log))
    assert ms.FusedCommMoEScheduler(moe).routes_from_logit_partials(None) is True
