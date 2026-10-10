# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for optional DSpark mixed allocation; prices are synthetic."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import dspark_mixed_runtime as runtime
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.speculative.dspark_mixed_evidence import (
    AdmittedMixedCosts,
    MixedCostCell,
    MixedEvidenceBinding,
    MixedGeometry,
    MixedRuntimeIdentity,
)
from tensorrt_llm._torch.speculative.dspark_schedule import DSparkScheduleConfig
from tensorrt_llm._torch.speculative.dspark_verify import DSparkVerifyPlanner


class _Event:
    def __init__(self, ready=True):
        self.ready = ready
        self.records = 0

    def query(self):
        return self.ready

    def record(self):
        self.records += 1


def _request(request_id, *, context=False):
    request = SimpleNamespace(
        py_request_id=request_id,
        py_seq_slot=request_id,
        is_dummy=False,
        py_batch_idx=0,
        py_draft_tokens=[1, 2, 3],
        py_verify_len=None,
        state=SimpleNamespace(name="GENERATION_IN_PROGRESS"),
        py_orig_prompt_len=128,
        py_max_new_tokens=768,
        context_chunk_size=64 if context else 0,
        context_current_position=128 if context else 0,
        py_num_compressed_tokens=0,
        is_first_context_chunk=False,
    )
    request.get_num_tokens = lambda beam: 128
    return request


def _group(monkeypatch, *, all_decode=False, stale=False):
    monkeypatch.setattr(runtime, "_greedy", lambda _batch: True)
    identity = MixedRuntimeIdentity(*(c * 64 for c in "abcdef"))
    binding = MixedEvidenceBinding("1" * 64, "2" * 64, "3" * 64)
    costs = AdmittedMixedCosts(identity, binding, 0.01, ())
    group = []
    for rank in range(8):
        config = SimpleNamespace(
            max_draft_len=3,
            max_seq_len=4096,
            max_num_tokens=4096,
            enable_attention_dp=True,
            torch_compile_enabled=False,
            is_multimodal=False,
            spec_config=SimpleNamespace(
                max_draft_len=3,
                block_size=3,
                decoding_type="DSpark",
                draft_is_embedded_in_target=True,
                use_rejection_sampling=False,
            ),
            prefill_cuda_graph_backend=SimpleNamespace(name="BREAKABLE"),
            prefill_cuda_graph_num_tokens=runtime.BODY_BUCKETS,
        )
        generations = [_request(rank * 1000 + i + 1) for i in range(64)]
        contexts = [_request(99999, context=True)] if rank == 0 and not all_decode else []
        batch = SimpleNamespace(
            context_requests=contexts,
            generation_requests=generations,
            context_requests_last_chunk=contexts,
        )
        planner = DSparkVerifyPlanner(cfg=DSparkScheduleConfig(3), device_windows=True)
        planner._host_buffer = torch.full((64, 3), 5.0)
        planner._host_confidence_stamp = torch.full((64,), 4, dtype=torch.int32)
        planner._snapshot_valid, planner._copy_event = True, _Event()
        slots = {r.py_request_id: i for i, r in enumerate(generations)}
        incarnations = {r.py_request_id: i + 1 for i, r in enumerate(generations)}
        worker = SimpleNamespace(
            _draft_seq_host=5,
            _req_to_slot=slots,
            confidence_row_for=slots.__getitem__,
            confidence_incarnation_for=incarnations.get,
        )
        planner._mixed_current_snapshot_meta = dict(
            buffer=planner._host_buffer,
            event=planner._copy_event,
            owners=slots.copy(),
            staged_iteration=9,
            staging_sequence=5,
            confidence_stamp=planner._host_confidence_stamp,
            producer_attempts={r.py_request_id: (r, i, i + 1) for i, r in enumerate(generations)},
        )
        if stale and rank == 1:
            incarnations[generations[0].py_request_id] += 100
        runner = SimpleNamespace(
            _config=config,
            mapping=SimpleNamespace(tp_size=8, pp_size=1, cp_size=1),
            _dspark_confidence_enabled=True,
            _dspark_trims_submitted_tokens=True,
            use_beam_search=False,
            _dspark_device_budget=10,
            cuda_graph_runner=SimpleNamespace(
                agreed_ragged_bucket=192,
                ragged_pad_verify_len=1,
                ragged_zero_real_high_rows=0,
            ),
            breakable_cuda_graph_runner=SimpleNamespace(
                is_capturing=False, is_warming_up=False, has_graph=lambda _bucket: True
            ),
        )
        group.append((runtime.MixedRuntime(runner, costs), batch, worker, planner, incarnations))
    votes = [
        mixed.local_vote(batch, worker, planner, 10, rank, executor_eligible=True)
        for rank, (mixed, batch, worker, planner, _) in enumerate(group)
    ]
    if not all_decode:
        index = runtime.BODY_BUCKETS.index(192)
        geometry = MixedGeometry(
            tuple(tuple(v[14:26]) for v in votes),
            192,
            tuple(v[28 + index * 3 + 2] for v in votes),
            tuple(v[27] for v in votes),
        )
        costs = replace(costs, cells=(MixedCostCell(geometry, 10.0, 5.0, "4" * 64),))
        for mixed, *_ in group:
            mixed.costs = costs
    return group, votes


def test_host_mixed_shortens_decode_peer_without_mutating_context(monkeypatch):
    group, votes = _group(monkeypatch)
    context = copy.deepcopy(group[0][1].context_requests[0].__dict__)
    for mixed, batch, worker, planner, _ in group:
        assert mixed.apply(batch, worker, planner, votes, 10)
        assert mixed.runner._dspark_host_window_step
        assert mixed.runner.cuda_graph_runner._dspark_host_window_batch is batch
        assert mixed.runner._dspark_device_budget is None
        assert any(r.py_verify_len < 3 for r in batch.generation_requests)
    assert group[0][1].context_requests[0].__dict__ == context
    assert sum(r.py_verify_len + 1 for r in group[1][1].generation_requests) == 192


@pytest.mark.parametrize("failure", ["uncovered", "unready", "identity"])
def test_missing_mixed_evidence_stays_native_without_disabling_decode(monkeypatch, failure):
    group, votes = _group(monkeypatch)
    if failure == "uncovered":
        for mixed, *_ in group:
            mixed.costs = replace(mixed.costs, cells=())
    elif failure == "unready":
        mixed, batch, worker, planner, _ = group[1]
        planner._copy_event.ready = False
        votes[1] = mixed.local_vote(batch, worker, planner, 10, 1, executor_eligible=True)
        assert votes[1][5] == 0
    else:
        votes[2][6] ^= 1
    for mixed, batch, worker, planner, _ in group:
        assert mixed.apply(batch, worker, planner, votes, 10)
        assert all(r.py_verify_len == 3 for r in batch.generation_requests)
        assert mixed.runner._dspark_confidence_enabled


def test_stale_owner_keeps_full_width_without_discarding_other_owners(monkeypatch):
    group, votes = _group(monkeypatch, stale=True)
    mixed, batch, worker, planner, _ = group[1]
    assert votes[1][22] == 63
    mixed.apply(batch, worker, planner, votes, 10)
    assert batch.generation_requests[0].py_verify_len == 3
    assert any(r.py_verify_len < 3 for r in batch.generation_requests[1:])


@pytest.mark.parametrize("change", ["incarnation", "context", "first_chunk"])
def test_ticket_rechecked_before_publishing_lengths(monkeypatch, change):
    group, votes = _group(monkeypatch)
    rank = 1 if change == "incarnation" else 0
    mixed, batch, worker, planner, incarnations = group[rank]
    if change == "incarnation":
        incarnations[batch.generation_requests[0].py_request_id] += 1
    elif change == "context":
        batch.context_requests[0].context_current_position += 1
    else:
        batch.context_requests[0].is_first_context_chunk = True
    with pytest.raises(ValueError, match="changed"):
        mixed.apply(batch, worker, planner, votes, 10)
    assert all(r.py_verify_len is None for r in batch.generation_requests)


@pytest.mark.parametrize("unresolved", ["first_chunk", "token_budget"])
def test_unfinalized_geometry_declines_group_before_compact_publication(monkeypatch, unresolved):
    group, votes = _group(monkeypatch)
    mixed, batch, worker, planner, _ = group[0]
    if unresolved == "first_chunk":
        batch.context_requests[0].is_first_context_chunk = True
    else:
        mixed.runner._config.max_num_tokens = 319
    votes[0] = mixed.local_vote(batch, worker, planner, 10, 0, executor_eligible=True)
    assert votes[0][5] == 0
    assert all(r.py_verify_len is None for r in batch.generation_requests)
    for mixed, batch, worker, planner, _ in group:
        assert mixed.apply(batch, worker, planner, votes, 10)
        assert all(r.py_verify_len == 3 for r in batch.generation_requests)
        assert mixed.runner._dspark_host_window_step


def test_first_chunk_stays_native_across_real_python_resource_preparation(monkeypatch):
    group, votes = _group(monkeypatch)
    mixed, batch, worker, planner, _ = group[0]

    class _NativeRequestModel(SimpleNamespace):
        # Models the native first-chunk predicate; no C++ execution is claimed.
        @property
        def is_first_context_chunk(self):
            return self.context_current_position == self.prepopulated_prompt_len

    values = batch.context_requests[0].__dict__.copy()
    values.pop("is_first_context_chunk")
    context = _NativeRequestModel(**values)
    context.context_chunk_size, context.context_current_position = 256, 0
    context.prepopulated_prompt_len = 0
    batch.context_requests[:] = [context]
    batch.context_requests_last_chunk[:] = [context]
    votes[0] = mixed.local_vote(batch, worker, planner, 10, 0, executor_eligible=True)
    assert votes[0][5] == 0
    for mixed, member, worker, planner, _ in group:
        mixed.apply(member, worker, planner, votes, 10)
        assert all(r.py_verify_len == 3 for r in member.generation_requests)
    phases = []

    def model_prefix_cache_setter(_batch):
        phases.append("prepare")
        context.prepopulated_prompt_len = context.context_current_position = 128
        context.context_chunk_size = 128

    manager = SimpleNamespace(
        resource_managers={
            ResourceManagerType.KV_CACHE_MANAGER: SimpleNamespace(
                prepare_resources=model_prefix_cache_setter,
                report_batch_to_connector=lambda _batch: phases.append("report"),
            )
        },
        maybe_fit_token_budget=lambda _batch: phases.append("trim"),
    )
    ResourceManager.prepare_resources(manager, batch)
    assert phases == ["prepare", "trim", "report"]
    assert context.context_chunk_size == 128
    assert all(r.py_verify_len == 3 for r in batch.generation_requests)


def test_all_decode_keeps_original_policy_and_device_mode(monkeypatch):
    group, votes = _group(monkeypatch, all_decode=True)
    for mixed, batch, worker, planner, _ in group:
        before = [r.__dict__.copy() for r in batch.generation_requests]
        assert not mixed.apply(batch, worker, planner, votes, 10)
        assert [r.__dict__ for r in batch.generation_requests] == before
        assert planner.device_windows


def test_optional_suffix_preserves_legacy_payload(monkeypatch):
    group, votes = _group(monkeypatch)
    mixed = group[0][0]
    legacy, suffix = mixed.split_votes([[1, 2, 3] + vote for vote in votes], 3)
    assert legacy == [[1, 2, 3]] * 8
    assert suffix == votes
    with pytest.raises(ValueError, match="shape differs"):
        mixed.split_votes([[1, 2, 3] + vote[:-1] for vote in votes], 3)


def test_unsupported_backend_declines_before_reading_snapshot(monkeypatch):
    group, _ = _group(monkeypatch)
    mixed, batch, worker, planner, _ = group[1]
    mixed.runner._config.prefill_cuda_graph_backend = SimpleNamespace(name="PIECEWISE")
    planner._mixed_current_snapshot_meta = None
    vote = mixed.local_vote(batch, worker, planner, 10, 1, executor_eligible=True)
    assert len(vote) == runtime.WIRE_WORDS
    assert vote[5] == 0


def test_host_metadata_skips_device_prologue_and_local_full_graph():
    runner = SimpleNamespace(
        _dspark_trims_submitted_tokens=True,
        _dspark_device_windows=True,
        cuda_graph_runner=SimpleNamespace(agreed_ragged_bucket=192),
        get_runtime_tokens_per_gen_step=lambda k: k + 1,
        _ragged_token_lens=DecoderRunner._ragged_token_lens,
    )
    metadata = SimpleNamespace(
        runtime_tokens_per_gen_step=None, ragged_verify_lens=None, device_windows_mode=False
    )
    requests = [SimpleNamespace(py_verify_len=1), SimpleNamespace(py_verify_len=3)]
    DecoderRunner._publish_gen_token_layout(
        runner, metadata, requests, runtime_draft_len=3, host_authoritative=True
    )
    assert not metadata.device_windows_mode
    assert metadata.ragged_verify_lens == [2, 4]
    runner._dspark_host_window_step, runner._dspark_device_budget = True, 5
    assert not DecoderRunner._apply_device_window_prologue(runner, {}, None)
    assert runner._dspark_device_budget is None
    batch = SimpleNamespace(can_run_cuda_graph=True)
    graph = SimpleNamespace(
        _dspark_host_window_batch=batch,
        is_encoder_decoder=False,
        enable_encoder_decoder_mixed_cuda_graph=False,
    )
    assert not CUDAGraphRunner._can_run_cuda_graph_batch(graph, batch)


def test_producer_stamp_copy_uses_same_event_and_validates_before_rotation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    planner = DSparkVerifyPlanner(cfg=DSparkScheduleConfig(3), device_windows=True)
    planner.stage_confidence(torch.ones(2, 3), torch.tensor([1, 2], dtype=torch.int32))
    first = (planner._host_buffer, planner._host_confidence_stamp, planner._copy_event)
    with pytest.raises(ValueError, match="one per slot"):
        planner.stage_confidence(torch.zeros(2, 3), torch.ones(3, dtype=torch.int32))
    assert planner._host_buffer is first[0]
    assert planner._host_confidence_stamp is first[1]
    planner.stage_confidence(torch.zeros(2, 3), torch.tensor([3, 4], dtype=torch.int32))
    assert planner._prev_buffer is first[0]
    assert planner._prev_confidence_stamp is first[1]
    assert planner._prev_event is first[2]
    assert planner._host_confidence_stamp.tolist() == [3, 4]


def test_forward_attempt_alone_is_not_confidence_publication(monkeypatch):
    group, _ = _group(monkeypatch)
    _, batch, worker, planner, _ = group[1]
    runtime.stage_mixed_ownership(planner, worker, 10, batch.generation_requests)
    assert planner._mixed_current_snapshot_meta["producer_attempts"] == {}
    worker._draft_seq_host += 1
    runtime.stage_mixed_ownership(planner, worker, 11, batch.generation_requests)
    assert len(planner._mixed_current_snapshot_meta["producer_attempts"]) == 64
    assert (
        planner._mixed_current_snapshot_meta["confidence_stamp"] is planner._host_confidence_stamp
    )
