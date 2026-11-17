# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for matching direct-stage and idle-rank collective sequences."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import FusedCommMoEScheduler

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def rubin_scheduler_type() -> type[FusedCommMoEScheduler]:
    # Importing the target loads its SM107 native library. Execute just its scheduler class,
    # inheriting the real shared scheduler, so this CPU test needs neither that library nor JIT.
    source = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/modeling_v2/models/glm_moe_dsa"
        / "glm_5_2_nvfp4__sm_107__high_throughput/ops/moe/megamoe.py"
    )
    tree = ast.parse(source.read_text())
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RubinFusedCommScheduler"
    )
    namespace = {
        "FusedCommMoEScheduler": FusedCommMoEScheduler,
        "MoERunContext": MoERunContext,
        "get_active_route_capture": lambda: None,
        "ExpertStatistic": SimpleNamespace(should_record=lambda: False),
        "get_calibrator": lambda: SimpleNamespace(mode=None),
        "Mode": SimpleNamespace(NONE=None),
    }
    exec(compile(ast.Module(body=[scheduler], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["RubinFusedCommScheduler"]


@pytest.mark.parametrize("enabled", [False, True])
def test_direct_stage_and_idle_rank_match_dummy_collectives(
    monkeypatch: pytest.MonkeyPatch,
    rubin_scheduler_type: type[FusedCommMoEScheduler],
    enabled: bool,
) -> None:
    monkeypatch.setattr(moe_scheduler, "_record_routes", lambda moe, routes, slots: routes)
    monkeypatch.setattr(moe_scheduler, "get_active_route_capture", lambda: None)
    rank_events = []
    for rank, num_tokens in enumerate((5, 0)):
        events = []

        def run_moe(ctx: MoERunContext) -> torch.Tensor:
            events.append("run_moe")
            assert ctx.x.shape == (num_tokens, 8)
            return ctx.x + 1

        stage = SimpleNamespace(
            topk_idx=torch.zeros((5, 1), dtype=torch.int32),
            topk_weights=torch.ones((5, 1)),
        )
        prepare = Mock(return_value=stage)
        quantize_into = Mock(side_effect=lambda x, stage: (x, None))
        quantize = Mock(side_effect=lambda x: (x, None))
        moe = SimpleNamespace(
            mapping=SimpleNamespace(moe_ep_rank=rank),
            enable_dummy_allreduce=enabled,
            dummy_allreduce=lambda: events.append("allreduce"),
            layer_load_balancer=None,
            apply_router_weight_on_input=False,
            _enable_perfect_router_expert_ids=False,
            _load_balancer_uses_replica_plan=lambda: False,
            _load_balancer_start_wait_gpu_stage=Mock(),
            _load_balancer_start_set_cpu_stage=Mock(),
            _load_balancer_done_set_cpu_stage=Mock(),
            num_slots=4,
            routing_method=SimpleNamespace(experts_per_token=1),
            backend=SimpleNamespace(
                prepare_direct_stage=prepare,
                route_and_quantize_into=lambda *args: stage,
                quantize_input_into=quantize_into,
                quantize_input=quantize,
                set_adaptive_launch_tokens=Mock(),
                run_moe=run_moe,
            ),
        )
        x = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        out = rubin_scheduler_type(moe)._forward_chunk(
            x,
            torch.zeros((5, 4)),
            output_dtype=x.dtype,
            all_rank_num_tokens=[5, 0],
            do_finalize=True,
        )
        if num_tokens:
            prepare.assert_called_once()
            quantize_into.assert_called_once()
            quantize.assert_not_called()
        else:
            prepare.assert_not_called()
            quantize_into.assert_not_called()
            quantize.assert_called_once()
        torch.testing.assert_close(out, x[:num_tokens] + 1, atol=0, rtol=0)
        rank_events.append(events)

    expected = ["allreduce", "run_moe", "allreduce"] if enabled else ["run_moe"]
    assert rank_events == [expected, expected]
