# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.speculative.dspark_device_select as device_select
import tensorrt_llm._torch.speculative.dspark_schedule as dspark_schedule
from tensorrt_llm._torch.speculative.dspark_device_select import select_windows_device
from tensorrt_llm._torch.speculative.dspark_ragged import fill_bucket_device
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_fused_fill,
)


def _legacy_schedule_and_fill(
    *,
    survival: torch.Tensor,
    budget: int,
    num_real: int,
    pad_len: int,
    graph_num_tokens: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    rows = torch.arange(survival.shape[0], device=survival.device)
    real_survival = torch.where(
        (rows < num_real).unsqueeze(1),
        survival,
        torch.zeros_like(survival),
    )
    scheduled = schedule_verify_lens_topk(survival=real_survival, budget=budget, cfg=cfg)
    return fill_bucket_device(
        scheduled + 1,
        num_real=torch.tensor(num_real, device=survival.device),
        graph_num_tokens=graph_num_tokens,
        max_verify_len=cfg.resolved_max_verify_len + 1,
        pad_fill=pad_len,
    )


@pytest.mark.parametrize(
    "num_rows,num_real,budget,survival_eps",
    [
        (1, 0, 0, 1e-6),
        (1, 1, 0, 1e-6),
        (2, 1, 1, 1e-6),
        (2, 1, 9, 1e-6),
        (4, 2, 3, 1e-6),
        (4, 4, 16, 1e-6),
        (8, 4, 9, 1e-6),
        (2, 2, 1, 0.35),
        (4, 1, 3, 0.35),
        (8, 8, 32, 0.35),
    ],
)
def test_cpu_fused_helper_matches_established_schedule_and_fill(
    num_rows, num_real, budget, survival_eps
):
    cfg = DSparkScheduleConfig(
        block_size=5,
        min_verify_len=1,
        max_verify_len=5,
        survival_eps=survival_eps,
    )
    generator = torch.Generator().manual_seed(
        10000 * num_rows + 1000 * num_real + 10 * budget + int(survival_eps * 10)
    )
    survival = torch.cumprod(torch.rand(num_rows, 5, generator=generator), dim=1)
    survival[num_real:] = 0
    pad_len = 1
    token_floor = cfg.min_verify_len + 1
    max_token_len = cfg.resolved_max_verify_len + 1
    pad_tokens = (num_rows - num_real) * pad_len
    capacity = min(max(budget, 0), num_real * cfg.schedulable_per_request)
    minimum = num_real * token_floor + pad_tokens
    maximum = num_real * max_token_len + pad_tokens
    graph_num_tokens = min(maximum, minimum + capacity + min(num_real, 2))

    expected = _legacy_schedule_and_fill(
        survival=survival,
        budget=budget,
        num_real=num_real,
        pad_len=pad_len,
        graph_num_tokens=graph_num_tokens,
        cfg=cfg,
    )
    actual = schedule_verify_lens_topk_fused_fill(
        survival=survival,
        budget=budget,
        num_real=num_real,
        pad_len=pad_len,
        graph_num_tokens=graph_num_tokens,
        cfg=cfg,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_round_robin_slack_is_not_promoted_into_confidence_budget():
    cfg = DSparkScheduleConfig(
        block_size=5,
        min_verify_len=1,
        max_verify_len=5,
        survival_eps=0.5,
    )
    survival = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.1, 0.1],
            [1.0, 0.8, 0.49, 0.1, 0.1],
        ]
    )
    actual = schedule_verify_lens_topk_fused_fill(
        survival=survival,
        budget=2,
        num_real=2,
        pad_len=1,
        graph_num_tokens=7,
        cfg=cfg,
    )
    # Budget two selects one position per row, then the already-paid graph
    # remainder goes to row zero. Promoting that remainder into top-k would
    # instead select row one's 0.49 candidate and produce [3, 4].
    assert actual.tolist() == [4, 3]


def test_zero_epsilon_declines_fusion_and_preserves_tensor_semantics(monkeypatch):
    cfg = DSparkScheduleConfig(
        block_size=5,
        min_verify_len=1,
        max_verify_len=5,
        survival_eps=0.0,
    )

    def _must_not_run(**_kwargs):
        raise AssertionError("zero-epsilon selector must use the tensor path")

    monkeypatch.setattr(device_select, "schedule_verify_lens_topk_fused_fill", _must_not_run)
    result = select_windows_device(
        confidence_logits=torch.zeros(3, 5),
        slot_idx=torch.tensor([0, 1, 2]),
        num_real=2,
        budget=2,
        graph_num_tokens=7,
        cfg=cfg,
        pad_len=1,
        use_fused_exact=True,
    )
    assert int(result.verify_lens.sum()) == 7


def test_tensor_controls_decline_fusion(monkeypatch):
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)

    def _must_not_run(**_kwargs):
        raise AssertionError("tensor controls must use the capture-safe tensor path")

    monkeypatch.setattr(device_select, "schedule_verify_lens_topk_fused_fill", _must_not_run)
    result = select_windows_device(
        confidence_logits=torch.zeros(2, 5),
        slot_idx=torch.tensor([0, 1]),
        num_real=torch.tensor(2, dtype=torch.int64),
        budget=torch.tensor(1, dtype=torch.int64),
        graph_num_tokens=5,
        cfg=cfg,
        pad_len=1,
        use_fused_exact=True,
    )
    assert int(result.verify_lens.sum()) == 5


def test_fused_helper_rejects_infeasible_graph_without_dispatch():
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)
    with pytest.raises(ValueError, match="cannot realize"):
        schedule_verify_lens_topk_fused_fill(
            survival=torch.ones(2, 5),
            budget=4,
            num_real=2,
            pad_len=1,
            graph_num_tokens=5,
            cfg=cfg,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_bs", [16, 32, 64, 128])
def test_cuda_fused_matches_tensor_oracle_across_all_graph_sizes(graph_bs):
    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)
    generator = torch.Generator(device="cuda").manual_seed(347805 + graph_bs)
    confidence = torch.rand((graph_bs, 5), generator=generator, device="cuda")
    survival = torch.cumprod(confidence, dim=1)
    # Include exact ties and non-finite candidates in every graph shape.
    survival[:2] = 0.75
    survival[2, 2] = torch.nan
    survival[3, 1] = torch.inf
    num_real = graph_bs - 3
    survival[num_real:] = 0
    budget = min(2 * num_real, num_real * cfg.schedulable_per_request)
    pad_len = 1
    minimum = num_real * (cfg.min_verify_len + 1) + (graph_bs - num_real)
    graph_num_tokens = minimum + budget + num_real

    expected = _legacy_schedule_and_fill(
        survival=survival,
        budget=budget,
        num_real=num_real,
        pad_len=pad_len,
        graph_num_tokens=graph_num_tokens,
        cfg=cfg,
    )
    actual = schedule_verify_lens_topk_fused_fill(
        survival=survival,
        budget=budget,
        num_real=num_real,
        pad_len=pad_len,
        graph_num_tokens=graph_num_tokens,
        cfg=cfg,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_engine_prewarm_requires_device_windows():
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    engine = object.__new__(PyTorchModelEngine)
    engine._dspark_fused_scheduler_enabled = True
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = False

    def _must_not_get_worker():
        raise AssertionError("disabled device-window mode must not prewarm fusion")

    engine._get_spec_worker = _must_not_get_worker
    PyTorchModelEngine._warmup_dspark_fused_scheduler(engine)
    assert engine._dspark_fused_schedule_ready_sizes == set()


def test_model_engine_prewarm_records_only_successful_graph_sizes(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import model_engine
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)
    planner = SimpleNamespace(
        cfg=cfg,
        exact_cost_table=SimpleNamespace(tables={16: object(), 32: object()}),
    )
    engine = object.__new__(PyTorchModelEngine)
    engine._dspark_fused_scheduler_enabled = True
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = True
    engine._get_spec_worker = lambda: SimpleNamespace(verify_planner=planner)

    original_ones = torch.ones
    monkeypatch.setattr(
        model_engine.torch,
        "ones",
        lambda shape, dtype=None, device=None: original_ones(shape, dtype=dtype),
    )
    monkeypatch.setattr(model_engine.torch.cuda, "synchronize", lambda: None)
    calls = []

    def _fake_fused(**kwargs):
        calls.append(kwargs["num_real"])
        if kwargs["num_real"] == 32:
            raise RuntimeError("synthetic compile failure")
        return torch.empty(kwargs["num_real"], dtype=torch.int32)

    monkeypatch.setattr(dspark_schedule, "schedule_verify_lens_topk_fused_fill", _fake_fused)
    PyTorchModelEngine._warmup_dspark_fused_scheduler(engine)
    assert calls == [16, 32]
    assert engine._dspark_fused_schedule_ready_sizes == {16}


def test_runtime_failure_retires_only_one_shape_and_retries_once():
    from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
    from tensorrt_llm._torch.speculative.dspark_schedule import DSparkFusedScheduleError

    engine = object.__new__(PyTorchModelEngine)
    engine._dspark_fused_schedule_ready_sizes = {16, 32}
    calls = []

    def _select(**kwargs):
        calls.append(kwargs["use_fused_exact"])
        if kwargs["use_fused_exact"]:
            raise DSparkFusedScheduleError("synthetic launch failure")
        return "tensor-result"

    result = PyTorchModelEngine._select_dspark_windows_with_fused_fallback(
        engine,
        select_fn=_select,
        selector_kwargs={},
        padded_bs=16,
    )
    assert result == "tensor-result"
    assert calls == [True, False]
    assert engine._dspark_fused_schedule_ready_sizes == {32}
    assert engine._dspark_fused_schedule_failure_counts == {16: 1}

    calls.clear()
    result = PyTorchModelEngine._select_dspark_windows_with_fused_fallback(
        engine,
        select_fn=_select,
        selector_kwargs={},
        padded_bs=16,
    )
    assert result == "tensor-result"
    assert calls == [False]
    assert engine._dspark_fused_schedule_failure_counts == {16: 1}
