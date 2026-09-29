# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.expert_statistic import ExpertStatistic
from tensorrt_llm._torch.moe.fused_moe import moe_scheduler as scheduler
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEInputRequirement
from tensorrt_llm._torch.utils import model_extra_attrs


@pytest.mark.parametrize("backend_ok,comm_ok", [(True, True), (False, True), (True, False)])
def test_graph_padding_capability_gate(backend_ok, comm_ok):
    padding = torch.tensor([False, True])
    moe = SimpleNamespace(
        backend=SimpleNamespace(
            capabilities=SimpleNamespace(supports_graph_padding_trim=backend_ok)
        ),
        comm=SimpleNamespace(supports_graph_padding_trim=comm_ok),
    )
    with model_extra_attrs({"moe_graph_padding": padding}):
        assert scheduler._get_supported_graph_padding(moe) is (
            padding if backend_ok and comm_ok else None
        )


def test_graph_padding_rejects_invalid_chunk_bounds():
    slots = torch.tensor([[0], [1]], dtype=torch.int32)
    padding = torch.tensor([True])
    for offset in (0, -1):
        with pytest.raises(ValueError, match="exceeds"):
            scheduler._mask_prefill_padding_routes(slots, padding, offset)


@pytest.mark.parametrize("eplb", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_graph_padding_scheduler_statistics_and_replay(monkeypatch, eplb, replay):
    original = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=torch.int32)
    replay_slots = torch.ones_like(original)
    padding = torch.tensor([False, False, True, True])
    statistic = ExpertStatistic(0, 0, 2)
    statistic._set_iter(0)
    monkeypatch.setattr(ExpertStatistic, "expert_statistic_obj", statistic)
    calibrator = SimpleNamespace(
        maybe_collect_or_replay_slots=lambda n, slots: replay_slots if replay else slots
    )
    monkeypatch.setattr(scheduler, "get_calibrator", lambda: calibrator)
    moe = Mock()
    moe.backend = SimpleNamespace(
        capabilities=SimpleNamespace(supports_graph_padding_trim=True),
        input_requirement=MoEInputRequirement(),
        _supports_load_balancer=lambda: True,
        quantize_input=lambda x: (x, None),
        run_moe=lambda ctx, **kwargs: ctx.x,
        try_fused_route_quant=lambda *args: None,
    )
    moe._using_load_balancer.return_value = eplb
    dispatched = []

    def dispatch(**kwargs):
        dispatched.append(kwargs["token_selected_slots"])
        return (
            kwargs["hidden_states"],
            kwargs["hidden_states_sf"],
            kwargs["token_selected_slots"],
            kwargs["token_final_scales"],
        )

    moe.comm = SimpleNamespace(
        supports_graph_padding_trim=True,
        supports_post_quant_dispatch=lambda: True,
        uses_internal_dispatch_quantization=lambda: False,
        dispatch=dispatch,
        combine=lambda x, **kwargs: x,
    )
    moe.routing_method = SimpleNamespace(
        experts_per_token=2,
        apply=lambda *args: (original, torch.ones_like(original, dtype=torch.float32)),
    )
    moe.apply_router_weight_on_input = False
    moe.layer_load_balancer = eplb
    moe.num_slots = 4
    moe.use_dp = True
    moe.layer_idx = 0
    moe.enable_dummy_allreduce = False
    moe.quant_scales = None
    moe._load_balancer_route.side_effect = lambda slots, dp: torch.where(slots < 0, 4, slots)
    captured = []
    monkeypatch.setattr(
        scheduler,
        "get_active_route_capture",
        lambda: SimpleNamespace(
            capture=lambda layer, routes, offset: captured.append(routes.clone())
        ),
    )
    runner = scheduler.ExternalCommMoEScheduler(moe)
    monkeypatch.setattr(runner, "_build_run_context", lambda **kwargs: SimpleNamespace(**kwargs))
    with model_extra_attrs({"moe_graph_padding": padding}):
        runner._forward_chunk_impl(
            torch.ones(4, 8), torch.ones(4, 4), torch.float32, [4], True, True, True
        )
    expected = replay_slots.clone() if replay else original.clone()
    expected[2:] = -1
    torch.testing.assert_close(dispatched[0], expected)
    torch.testing.assert_close(statistic._records["0_0"], torch.ones(4, dtype=torch.int64))
    assert (original >= 0).all() and (replay_slots == 1).all()
    torch.testing.assert_close(captured[0], original)
    if eplb:
        counted = moe._load_balancer_update_statistic.call_args.args[0]
        assert (counted[2:] == -1).all()


def test_graph_padding_input_preparation_and_opt_out(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import model_engine

    engine = object.__new__(model_engine.PyTorchModelEngine)
    engine.mapping = None
    engine.spec_config = None
    engine.prefill_cuda_graph_backend = model_engine.PrefillCudaGraphBackend.BREAKABLE
    engine._moe_row_is_padding_cuda = torch.ones(8, dtype=torch.bool)
    state = SimpleNamespace(prefill=False)
    monkeypatch.setattr(
        model_engine, "get_per_request_prefill_cuda_graph_flag", lambda: state.prefill
    )
    monkeypatch.setattr(
        model_engine,
        "set_per_request_prefill_cuda_graph_flag",
        lambda value: setattr(state, "prefill", value),
    )

    def prepare(real, padded, prefill=True):
        result = ({"input_ids": torch.zeros(padded)}, None)

        def prepare_tp(*args, **kwargs):
            assert not state.prefill and engine._moe_graph_padding is None
            state.prefill = prefill
            return result

        engine._prepare_tp_inputs = prepare_tp
        assert engine._prepare_inputs(None, None, SimpleNamespace(num_tokens=real)) is result

    prepare(3, 8)
    assert engine._moe_graph_padding.tolist() == [False] * 3 + [True] * 5
    pointer = engine._moe_graph_padding.data_ptr()
    prepare(8, 8)
    assert not engine._moe_graph_padding.any()
    assert engine._moe_graph_padding.data_ptr() == pointer
    prepare(3, 8, prefill=False)
    assert engine._moe_graph_padding is None
    engine._moe_row_is_padding_cuda = None
    prepare(3, 8)
    assert engine._moe_graph_padding is None


def test_piecewise_graph_never_publishes_trim_mask(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import model_engine

    engine = object.__new__(model_engine.PyTorchModelEngine)
    monkeypatch.setattr(model_engine, "get_per_request_prefill_cuda_graph_flag", lambda: True)
    engine.prefill_cuda_graph_backend = model_engine.PrefillCudaGraphBackend.PIECEWISE
    engine._moe_row_is_padding_cuda = torch.ones(8, dtype=torch.bool)
    engine._moe_graph_padding = engine._moe_row_is_padding_cuda
    engine._publish_moe_graph_padding(3, 8)
    assert engine._moe_graph_padding is None
    assert engine._moe_row_is_padding_cuda.all()


@pytest.mark.parametrize("offset", [0, 2, None])
def test_graph_padding_chunk_offsets(offset):
    routes = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    padding = torch.tensor([False, False, False, True])
    expected = routes.clone()
    if offset is None:
        expected.fill_(-1)
    elif offset == 2:
        expected[1] = -1
    torch.testing.assert_close(
        scheduler._mask_prefill_padding_routes(routes, padding, offset), expected
    )
