# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.speculative.dspark_device_select as device_select
import tensorrt_llm._torch.speculative.dspark_schedule as dspark_schedule
from tensorrt_llm._torch.speculative.dspark_device_select import select_windows_device
from tensorrt_llm._torch.speculative.dspark_schedule import (
    DSparkScheduleConfig,
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_fused_fill,
)
from tensorrt_llm._torch.speculative.ragged_helpers import fill_bucket_device


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


@pytest.mark.parametrize("num_real,budget,graph_tokens", [(2, 2, 9), (4, 4, 14)])
def test_fused_workspace_selection_matches_tensor_and_reuses_storage(
    num_real, budget, graph_tokens
):
    cfg = DSparkScheduleConfig(block_size=3, min_verify_len=1)
    workspace = device_select.DeviceWindowWorkspace.allocate(
        max_rows=6, max_tokens=16, device="cpu"
    )
    workspace.verify_lens.fill_(-777)
    pointers = {
        name: getattr(workspace, name).data_ptr()
        for name in ("verify_lens", "qo_indptr", "req_idx", "kv_correction")
    }
    logits = torch.tensor([[10.0, 5.0, 1.0], [-2.0, -3.0, -4.0], [0.5, 0.2, 0.1], [1.0, 2.0, 3.0]])
    for confidence in (logits, logits.flip(0)):
        kwargs = dict(
            confidence_logits=confidence,
            slot_idx=torch.tensor([0, 1, 2, 3]),
            num_real=num_real,
            budget=budget,
            graph_num_tokens=graph_tokens,
            cfg=cfg,
            pad_len=1,
        )
        expected = select_windows_device(**kwargs, use_fused_exact=False)
        actual = select_windows_device(**kwargs, use_fused_exact=True, workspace=workspace)
        assert expected.workspace is None
        assert actual.workspace is workspace
        for name, pointer in pointers.items():
            result = getattr(actual, name)
            assert result.data_ptr() == pointer
            torch.testing.assert_close(result, getattr(expected, name), rtol=0, atol=0)
        assert expected.verify_lens.data_ptr() != workspace.verify_lens.data_ptr()
        assert workspace.verify_lens[4:].tolist() == [-777, -777]


@pytest.mark.parametrize("num_rows", [0, 2])
def test_fused_helper_preserves_output_address_and_unused_capacity(num_rows):
    cfg = DSparkScheduleConfig(block_size=3, min_verify_len=1)
    output = torch.full((num_rows + 2,), -777, dtype=torch.int32)
    actual = schedule_verify_lens_topk_fused_fill(
        survival=torch.ones(num_rows, 3),
        budget=0,
        num_real=num_rows,
        pad_len=1,
        graph_num_tokens=2 * num_rows,
        cfg=cfg,
        out=output,
    )
    assert actual.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
    assert actual.tolist() == [2] * num_rows
    assert output[num_rows:].tolist() == [-777, -777]


@pytest.mark.parametrize(
    "output,error,message",
    [
        (torch.empty(2, dtype=torch.int32, device="meta"), ValueError, "share survival.device"),
        (torch.empty(2, dtype=torch.int64), TypeError, "dtype torch.int32"),
        (torch.empty(1, 2, dtype=torch.int32), ValueError, "1-D tensor"),
        (torch.empty(1, dtype=torch.int32), ValueError, "at least 2 elements"),
        (torch.empty(4, dtype=torch.int32)[::2], ValueError, "contiguous"),
    ],
)
def test_fused_helper_rejects_incompatible_output_before_dispatch(output, error, message):
    cfg = DSparkScheduleConfig(block_size=3, min_verify_len=1)
    with pytest.raises(error, match=message):
        schedule_verify_lens_topk_fused_fill(
            survival=torch.ones(2, 3),
            budget=0,
            num_real=2,
            pad_len=1,
            graph_num_tokens=4,
            cfg=cfg,
            out=output,
        )


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
@pytest.mark.parametrize("graph_bs", [16, 32, 64, 128, 192, 256])
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
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner

    engine = object.__new__(DecoderRunner)
    engine._dspark_fused_scheduler_enabled = True
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = False

    def _must_not_get_worker():
        raise AssertionError("disabled device-window mode must not prewarm fusion")

    engine._get_spec_worker = _must_not_get_worker
    DecoderRunner._warmup_dspark_fused_scheduler(engine)
    assert engine._dspark_fused_schedule_ready_sizes == set()


def test_model_engine_prewarm_records_only_successful_graph_sizes(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as decoder_runner

    cfg = DSparkScheduleConfig(block_size=5, min_verify_len=1, max_verify_len=5)
    planner = SimpleNamespace(
        cfg=cfg,
        exact_cost_table=SimpleNamespace(tables={16: object(), 32: object()}),
    )
    engine = object.__new__(DecoderRunner)
    engine._dspark_fused_scheduler_enabled = True
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = True
    engine._get_spec_worker = lambda: SimpleNamespace(verify_planner=planner)

    original_ones = torch.ones
    monkeypatch.setattr(
        decoder_runner.torch,
        "ones",
        lambda shape, dtype=None, device=None: original_ones(shape, dtype=dtype),
    )
    monkeypatch.setattr(decoder_runner.torch.cuda, "synchronize", lambda: None)
    calls = []

    def _fake_fused(**kwargs):
        calls.append(kwargs["num_real"])
        if kwargs["num_real"] == 32:
            raise RuntimeError("synthetic compile failure")
        return torch.empty(kwargs["num_real"], dtype=torch.int32)

    monkeypatch.setattr(dspark_schedule, "schedule_verify_lens_topk_fused_fill", _fake_fused)
    DecoderRunner._warmup_dspark_fused_scheduler(engine)
    assert calls == [16, 32]
    assert engine._dspark_fused_schedule_ready_sizes == {16}


def test_runtime_failure_retires_only_one_shape_and_retries_once():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
    from tensorrt_llm._torch.speculative.dspark_schedule import DSparkFusedScheduleError

    engine = object.__new__(DecoderRunner)
    engine._dspark_fused_schedule_ready_sizes = {16, 32}
    calls = []

    def _select(**kwargs):
        calls.append(kwargs["use_fused_exact"])
        if kwargs["use_fused_exact"]:
            raise DSparkFusedScheduleError("synthetic launch failure")
        return "tensor-result"

    result = DecoderRunner._select_dspark_windows_with_fused_fallback(
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
    result = DecoderRunner._select_dspark_windows_with_fused_fallback(
        engine,
        select_fn=_select,
        selector_kwargs={},
        padded_bs=16,
    )
    assert result == "tensor-result"
    assert calls == [False]
    assert engine._dspark_fused_schedule_failure_counts == {16: 1}


def _tensor_prologue_case():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner

    confidence = torch.tensor([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    planner = SimpleNamespace(
        cfg=DSparkScheduleConfig(block_size=3, min_verify_len=1),
        apply_calibration=None,
    )
    worker = SimpleNamespace(
        verify_planner=planner,
        staged_confidence_buffer=lambda: confidence,
        batch_slot_view=lambda rows: torch.arange(rows),
        verified_draft_seq_cuda=lambda: None,
    )
    engine = object.__new__(DecoderRunner)
    engine.cuda_graph_runner = SimpleNamespace(agreed_ragged_bucket=6, ragged_pad_verify_len=0)
    engine._config = SimpleNamespace(use_mrope=False)
    engine._dspark_device_budget = 2
    engine._dspark_prev_covers_batch = True
    engine._dspark_last_padded_bs = 2
    engine._dspark_last_num_real = 2
    engine._dspark_fused_schedule_ready_sizes = set()
    engine._get_spec_worker = lambda: worker
    engine.ragged_verify_lens_cuda = torch.tensor([3, 3], dtype=torch.int32)
    engine.ragged_qo_indptr_cuda = torch.tensor([0, 3, 6], dtype=torch.int32)
    engine.previous_batch_indices_cuda = torch.tensor([0, 1], dtype=torch.int64)
    engine.input_ids_cuda = torch.full((6,), -1, dtype=torch.int32)
    engine.position_ids_cuda = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)
    engine.previous_pos_indices_cuda = torch.full((6,), -1, dtype=torch.int64)
    engine.previous_pos_id_offsets_cuda = torch.full((6,), -1, dtype=torch.int32)
    engine.previous_kv_lens_offsets_cuda = torch.tensor([-1, -2], dtype=torch.int32)
    engine.draft_tokens_cuda = torch.full((4,), -1, dtype=torch.int32)
    events = []
    attn_metadata = SimpleNamespace(
        num_contexts=0,
        kv_lens_cuda=torch.tensor([16, 26], dtype=torch.int32),
        apply_device_ragged_layout=lambda lens, owners, correction: events.append(
            ("attention", lens.clone(), owners.clone(), correction.clone())
        ),
    )
    spec_metadata = SimpleNamespace(
        remap_expanded_sampling_params=lambda owners, count: events.append(
            ("sampling", owners.clone(), count)
        ),
    )
    new_tensors = SimpleNamespace(
        new_tokens=torch.tensor([[100, 200], [101, 201], [102, 202], [103, 203]]),
        new_tokens_lens=torch.tensor([2, 1], dtype=torch.int32),
        next_draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]]),
    )
    return (
        engine,
        {"spec_metadata": spec_metadata, "attn_metadata": attn_metadata},
        new_tensors,
        events,
    )


def test_tensor_prologue_rebuilds_owners_positions_and_both_kv_terms():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner

    engine, inputs, new_tensors, events = _tensor_prologue_case()
    lens_address = engine.ragged_verify_lens_cuda.data_ptr()
    qo_address = engine.ragged_qo_indptr_cuda.data_ptr()
    assert DecoderRunner._apply_device_window_prologue(engine, inputs, new_tensors)
    assert engine._dspark_device_budget is None
    assert engine.ragged_verify_lens_cuda.tolist() == [4, 2]
    assert engine.ragged_qo_indptr_cuda.tolist() == [0, 4, 6]
    assert engine.ragged_verify_lens_cuda.data_ptr() == lens_address
    assert engine.ragged_qo_indptr_cuda.data_ptr() == qo_address
    assert engine.input_ids_cuda.tolist() == [100, 101, 102, 103, 200, 201]
    assert engine.position_ids_cuda.tolist() == [10, 11, 12, 13, 20, 21]
    assert engine.previous_pos_indices_cuda.tolist() == [0, 0, 0, 0, 1, 1]
    assert engine.previous_pos_id_offsets_cuda.tolist() == [2, 2, 2, 2, 1, 1]
    assert engine.draft_tokens_cuda.tolist() == [11, 12, 13, 21]
    assert inputs["attn_metadata"].kv_lens_cuda.tolist() == [18, 24]
    assert engine.previous_kv_lens_offsets_cuda.tolist() == [-2, -1]
    assert (
        inputs["attn_metadata"].kv_lens_cuda + engine.previous_kv_lens_offsets_cuda
    ).tolist() == [16, 23]
    assert [event[0] for event in events] == ["sampling", "attention"]
    assert events[0][1].tolist() == [0, 0, 0, 0, 1, 1]
    assert events[0][2] == 6


def test_decode_only_prologue_declines_local_prefill_without_mutating_buffers():
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner

    engine, inputs, new_tensors, events = _tensor_prologue_case()
    inputs["attn_metadata"].num_contexts = 1
    tensors = (
        engine.input_ids_cuda,
        engine.position_ids_cuda,
        engine.ragged_verify_lens_cuda,
        engine.ragged_qo_indptr_cuda,
        inputs["attn_metadata"].kv_lens_cuda,
        engine.previous_kv_lens_offsets_cuda,
    )
    originals = [tensor.clone() for tensor in tensors]
    assert not DecoderRunner._apply_device_window_prologue(engine, inputs, new_tensors)
    for tensor, original in zip(tensors, originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    assert not events


@pytest.mark.parametrize("confidence,fused", [(False, False), (True, False), (True, True)])
def test_workspace_allocation_follows_decoder_state_initialization(monkeypatch, confidence, fused):
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as decoder_runner

    original_empty, original_zeros = torch.empty, torch.zeros

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(decoder_runner, "prefer_pinned", lambda: False)
    monkeypatch.setattr(decoder_runner, "LoraParamBuilder", lambda **kwargs: SimpleNamespace())
    engine = object.__new__(DecoderRunner)
    engine._config = SimpleNamespace(
        is_spec_decode=True,
        max_draft_loop_tokens=3,
        max_batch_size=4,
        max_num_tokens=16,
        max_beam_width=1,
        max_seq_len=16,
        use_mrope=False,
        attention_backend="TRTLLM",
        spec_config=SimpleNamespace(
            enable_confidence_scheduling=confidence,
            enable_fused_confidence_scheduler=fused,
        ),
    )
    engine._compute_dynamic_draft_len_mapping = lambda: {}
    # Follow the constructor order: buffers exist before mutable state/flags.
    engine._allocate_decoder_buffers()
    engine._init_decoder_state()
    if confidence and fused:
        workspace = engine._dspark_device_window_workspace
        assert workspace.verify_lens.numel() == 5
        assert workspace.req_idx.numel() == 16
        assert workspace.verify_lens.device == engine.input_ids_cuda.device
    else:
        assert engine._dspark_device_window_workspace is None
