# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal-package CPU source-path regressions; no AST/source extraction.

Run only against a fresh matching installed package containing this candidate.
The KV allocator/native serializer and CUDA-only reducer are explicit boundaries;
these tests do not assert native allocation, numerical accuracy or model passes.
"""

import pickle
from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import llm_request as request_module
from tensorrt_llm._torch.pyexecutor.context_logits_plan import store_context_logits
from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ContextLogitsOrigin,
    LlmResponse,
    LlmResult,
    PyResult,
)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm._torch.speculative import spec_sampler_base as spec_module
from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.result import LogProbsResult, compute_prompt_logprobs

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def cpu_storage(monkeypatch):
    # CPU-only copies exercise the genuine storage; no fake tensors/modules.
    monkeypatch.setattr(request_module, "prefer_pinned", lambda: False)


def result(prompt_length=4):
    return PyResult(
        prompt_len=prompt_length,
        max_new_tokens=1,
        use_device_memory=False,
        return_context_logits=True,
        use_chunked_generation_logits=False,
    )


def origin(start, end, prompt_length=4, client_id=17):
    return ContextLogitsOrigin(client_id, prompt_length, start, end)


def full_result(prompt_length=4):
    value = result(prompt_length)
    rows = torch.arange(prompt_length * 7, dtype=torch.float32).reshape(prompt_length, 7)
    value.append_context_logits(rows, origin(0, prompt_length, prompt_length))
    return value, rows


def test_positioned_storage_replaces_replayed_rows_without_duplicating():
    value = result()
    value.append_context_logits(torch.ones(3, 7), origin(0, 3))
    replacement = torch.full((3, 7), 9.0)
    value.append_context_logits(replacement, origin(1, 4))
    expected = torch.cat((torch.ones(1, 7), replacement))
    torch.testing.assert_close(value.context_logits, expected)
    assert value.context_logits_origins == (origin(0, 4),)
    assert len(value.context_logits) == 4


def test_out_of_order_exact_positions_are_not_shape_inferred():
    value = result()
    value.append_context_logits(torch.full((2, 7), 3.0), origin(2, 4))
    with pytest.raises(ValueError, match="full original prompt"):
        _ = value.context_logits
    value.append_context_logits(torch.full((2, 7), 8.0), origin(0, 2))
    torch.testing.assert_close(value.context_logits[:2], torch.full((2, 7), 8.0))
    torch.testing.assert_close(value.context_logits[2:], torch.full((2, 7), 3.0))


@pytest.mark.parametrize("start", [173, 203])
def test_observed_suffix_dimensions_cannot_be_reduced_as_full_prompt(start):
    value = result(204)
    value.append_context_logits(torch.ones(204 - start, 7), origin(start, 204, 204))
    with pytest.raises(ValueError, match="full original prompt"):
        _ = value.context_logits


@pytest.mark.parametrize(
    "bad",
    [
        ContextLogitsOrigin(17, 5, 0, 2),
        ContextLogitsOrigin(17, 4, -1, 1),
        ContextLogitsOrigin(17, 4, 3, 5),
        ContextLogitsOrigin(17, 4, 0, 3),
        ContextLogitsOrigin(True, 4, 0, 2),
    ],
)
def test_invalid_origin_is_rejected_without_initializing_storage(bad):
    value = result()
    with pytest.raises(ValueError):
        value.append_context_logits(torch.ones(2, 7), bad)
    assert value.context_logits_origins is None
    assert value.context_logits is None


def test_identity_cannot_change_between_fragments():
    value = result()
    value.append_context_logits(torch.ones(2, 7), origin(0, 2))
    with pytest.raises(ValueError, match="identity changed"):
        value.append_context_logits(torch.ones(2, 7), origin(2, 4, client_id=18))


def test_missing_origin_cannot_join_indexed_decoder_rows():
    value = result()
    value.append_context_logits(torch.ones(2, 7), origin(0, 2))
    with pytest.raises(ValueError, match="missing their origin"):
        value.append_context_logits(torch.ones(2, 7))


def test_encoder_only_unindexed_storage_keeps_existing_behavior():
    value = result(1)
    value.append_context_logits(torch.tensor([[1.0, 2.0, 3.0]]))
    assert value.context_logits_origins is None
    torch.testing.assert_close(value.context_logits, torch.tensor([[1.0, 2.0, 3.0]]))


def test_diff_pickle_and_replay_preserve_explicit_coordinate_writes():
    source, target = result(), result()
    source.append_context_logits(torch.ones(3, 7), origin(0, 3))
    target.apply_diff(pickle.loads(pickle.dumps(source.get_diff())))
    source.reset_diff()
    source.append_context_logits(torch.full((3, 7), 5.0), origin(1, 4))
    target.apply_diff(pickle.loads(pickle.dumps(source.get_diff())))
    torch.testing.assert_close(target.context_logits, source.context_logits)
    assert target.context_logits_origins == source.context_logits_origins
    assert target.get_diff().context_logits_list == []


def test_mixed_transport_schema_is_rejected_before_partial_application():
    target = result()
    diff = PyResult.Diff(context_logits_list=[torch.ones(4, 7)])
    with pytest.raises(ValueError, match="diff lengths differ"):
        target.apply_diff(diff)
    assert target.context_logits is None


def response_for(value, client_id=17):
    wire_result = LlmResult(b"", value, is_final=True)
    # The real native serializer is a boundary, not mocked into a PASS claim.
    wire_result.get_result = lambda: SimpleNamespace(output_token_ids=[[99]])
    return LlmResponse(request_id=80, client_id=client_id, result=wire_result)


def generation(prompt_length=4, streaming=False):
    return SimpleNamespace(
        _generation_request=SimpleNamespace(
            prompt_token_ids=list(range(100, 100 + prompt_length)), id=17
        ),
        _streaming=streaming,
        _logprob_params=SimpleNamespace(prompt_logprobs=0, prompt_logprobs_simple_format=False),
    )


def test_consumer_selects_all_original_targets_and_actual_first_generation(monkeypatch):
    value, rows = full_result(204)
    captured = []
    returned = LogProbsResult(prompt=[{99: "recorded"}], generation=None)

    def record_reduction(*args, **kwargs):
        captured.append((args, kwargs))
        return returned

    monkeypatch.setattr(base_worker, "compute_prompt_logprobs", record_reduction)
    observed = base_worker._compute_pytorch_prompt_logprobs(generation(204), response_for(value))
    assert observed is returned
    assert len(captured) == 1
    assert captured[0][0][2] == list(range(101, 304)) + [99]
    torch.testing.assert_close(captured[0][0][1], rows)
    assert captured[0][0][0] == 0


@pytest.mark.parametrize("failure", ["client", "missing", "prompt_length"])
def test_consumer_rejects_bad_binding_before_reduction(monkeypatch, failure):
    value, _ = full_result()
    response = response_for(value, client_id=18 if failure == "client" else 17)
    if failure == "missing":
        value._context_logits_client_id = None
    current_generation = generation(5 if failure == "prompt_length" else 4)
    monkeypatch.setattr(
        base_worker,
        "compute_prompt_logprobs",
        lambda *a, **k: pytest.fail("reduction must not run"),
    )
    with pytest.raises(
        base_worker.IncompleteContextLogitsError, match="original prompt-row origins"
    ):
        base_worker._compute_pytorch_prompt_logprobs(current_generation, response)


def test_streaming_cached_result_keeps_early_return(monkeypatch):
    current_generation = generation(streaming=True)
    cached = [{101: "already computed"}]
    current_generation._cached_prompt_logprobs = cached
    monkeypatch.setattr(
        base_worker,
        "compute_prompt_logprobs",
        lambda *a, **k: pytest.fail("cached result must not reduce"),
    )
    assert (
        base_worker._compute_pytorch_prompt_logprobs(current_generation, SimpleNamespace()).prompt
        is cached
    )


def test_clear_context_drops_origin_with_storage():
    value, _ = full_result()
    response = response_for(value)
    response.clear_context_logits()
    assert response.result.context_logits is None
    assert response.result.context_logits_origins is None


def request_for_store(value, prompt_length=4, start=0, chunk=4):
    return SimpleNamespace(
        py_result=value,
        py_return_context_logits=True,
        py_context_logits_explicit=True,
        py_prompt_logprobs=None,
        py_return_generation_logits=False,
        py_return_log_probs=False,
        py_use_chunked_generation_logits=False,
        py_client_id=17,
        prompt_len=prompt_length,
        orig_prompt_len=prompt_length,
        prepopulated_prompt_len=0,
        context_current_position=start,
        context_chunk_size=chunk,
    )


def test_normal_producer_stages_native_positions_and_original_values():
    value = result()
    current_request = request_for_store(value, start=0, chunk=2)
    handler = HandleLogits()
    first = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    handler([current_request], [], first, 1, [0, 2], True)
    current_request.context_current_position = 2
    second = first + 50
    handler([current_request], [], second, 1, [0, 2], True)
    torch.testing.assert_close(value.context_logits, torch.cat((first, second)))
    assert value.context_logits_origins == (origin(0, 4),)


def test_speculative_producer_stages_only_requested_context_rows():
    value = result()
    current_request = request_for_store(value)
    scheduled = SimpleNamespace(context_requests=[current_request])
    hidden = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    sampler = SimpleNamespace(logit_projection=lambda states: states + 12)
    outputs = {
        spec_module.HIDDEN_STATES_KEY: hidden,
        spec_module.GATHER_IDS_KEY: torch.empty(0, dtype=torch.int64),
    }
    observed = spec_module.SpecSampler._stage_requested_logits(sampler, scheduled, outputs, [], 3)
    assert observed == (None, [])
    torch.testing.assert_close(value.context_logits, hidden + 12)
    assert value.context_logits_origins == (origin(0, 4),)


def manager(full_rows, name):
    """Real manager methods with an explicit fake allocator boundary."""
    current_request = SimpleNamespace(
        py_return_context_logits=full_rows,
        py_request_id=80,
        cache_salt="caller-isolation",
        lora_task_id=None,
        prompt_len=204,
        is_first_context_chunk=True,
        is_dummy=False,
        is_dummy_request=False,
        kv_cache_retention_config=None,
        return_perf_metrics=False,
        is_disagg_generation_init_state=False,
        is_generation_only_request=False,
        py_served_prefix_end=0,
        context_current_position=0,
        context_remaining_length=204,
        context_chunk_size=204,
    )
    current_request.set_prepopulated_prompt_len = lambda depth, block: setattr(
        current_request, "context_current_position", depth
    )
    seen = []
    instance = SimpleNamespace(
        enable_block_reuse=True,
        kv_cache_map={},
        conversation_manager=None,
        _has_cp_helix=False,
        _stream=SimpleNamespace(cuda_stream=1),
        is_estimating_kv_cache=False,
        is_draft=name == "draft",
        _spec_recompute_tail=0,
        tokens_per_block=32,
        reuse_match_backoff=3,
        kv_connector_manager=SimpleNamespace(capacity_only=False),
    )
    instance._reuse_token_source = lambda req: list(range(req.prompt_len))
    instance._augment_tokens_for_block_reuse = lambda tokens, req, end: tokens[:end]
    instance._context_reuse_tokens = MethodType(KVCacheManagerV2._context_reuse_tokens, instance)
    instance._derive_reuse_salt = KVCacheManagerV2._derive_reuse_salt
    instance._connector_may_serve = MethodType(KVCacheManagerV2._connector_may_serve, instance)
    instance._record_branch_snapshot_point = lambda *args: None
    instance._resume_context_cache = lambda *args: True
    # Stream placement and allocation remain explicit CPU-test boundaries.
    instance._place_on_allocation_stream = lambda *args: None
    instance._cache_stream = lambda request_id: 1
    instance._retention_config_for_request = MethodType(
        KVCacheManagerV2._retention_config_for_request, instance
    )
    instance._prepare_connector_prefix_reservation = lambda *args: None

    def allocate(request_id, lora_id, tokens, **kwargs):
        seen.append((name, "claim", list(tokens), kwargs["cache_salt"]))
        cache = SimpleNamespace(num_committed_tokens=0)
        instance.kv_cache_map[request_id] = cache
        return cache

    instance._create_kv_cache = allocate
    instance.prepare_context_cache = MethodType(KVCacheManagerV2.prepare_context_cache, instance)
    instance.prepare_context = MethodType(KVCacheManagerV2.prepare_context, instance)

    def probe(scope, tokens):
        seen.append((name, "probe", list(tokens), current_request.cache_salt))
        return 0

    instance.impl = SimpleNamespace(probe_reuse=probe)
    instance.probe_context_reuse = MethodType(KVCacheManagerV2.probe_context_reuse, instance)
    return instance, current_request, seen


@pytest.mark.parametrize("limit", [None, 0, 32, 160])
def test_full_row_policy_suppresses_lookup_evidence_without_changing_salt(limit):
    instance, current_request, _ = manager(True, "target")
    assert instance._context_reuse_tokens(current_request, limit) == []
    assert current_request.cache_salt == "caller-isolation"


def test_ordinary_request_retains_reuse_evidence_and_connector_eligibility():
    instance, current_request, _ = manager(False, "target")
    assert instance._context_reuse_tokens(current_request, 32) == list(range(35))
    assert instance._connector_may_serve(current_request)
    current_request.py_return_context_logits = True
    assert not instance._connector_may_serve(current_request)


def test_joint_target_draft_both_probe_and_claim_empty_before_settle():
    target, current_request, target_seen = manager(True, "target")
    draft, _, draft_seen = manager(True, "draft")
    scheduler = SimpleNamespace(kv_cache_manager=target, _joint_draft_manager=draft)
    scheduler._reuse_claim_limit = MethodType(KVCacheV2Scheduler._reuse_claim_limit, scheduler)
    scheduler._free_kv_caches = lambda req: pytest.fail("valid claims must not roll back")
    assert KVCacheV2Scheduler._prepare_context_pair(scheduler, current_request)
    assert {entry[1] for entry in target_seen} == {"probe", "claim"}
    assert {entry[1] for entry in draft_seen} == {"probe", "claim"}
    assert all(entry[2] == [] for entry in target_seen + draft_seen)
    assert all(entry[3] == "caller-isolation" for entry in target_seen + draft_seen)
    assert current_request.context_current_position == 0


def test_same_request_cache_resume_does_not_discard_gathered_rows_or_reclaim():
    instance, current_request, seen = manager(True, "target")
    own_cache = SimpleNamespace(num_committed_tokens=64)
    instance.kv_cache_map[current_request.py_request_id] = own_cache
    current_request.is_first_context_chunk = False
    assert instance.prepare_context_cache(current_request) == 64
    assert seen == []
    assert instance.kv_cache_map[current_request.py_request_id] is own_cache


@pytest.mark.parametrize("producer", ["normal", "speculative"])
def test_post_generation_recompute_keeps_only_original_prompt_rows(producer):
    value, _ = full_result()
    current_request = request_for_store(value, start=0, chunk=6)
    current_request.prompt_len = 6
    # Native orig_prompt_len is immutable across pause/recompute; the two
    # generated history tokens are not original context-output rows.
    current_request.orig_prompt_len = 4
    rows = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    if producer == "normal":
        HandleLogits()([current_request], [], rows, 1, [0, 6], True)
    else:
        sampler = SimpleNamespace(logit_projection=lambda states: states)
        outputs = {
            spec_module.HIDDEN_STATES_KEY: rows,
            spec_module.GATHER_IDS_KEY: torch.empty(0, dtype=torch.int64),
        }
        scheduled = SimpleNamespace(context_requests=[current_request])
        assert spec_module.SpecSampler._stage_requested_logits(
            sampler, scheduled, outputs, [], 3
        ) == (None, [])
    torch.testing.assert_close(value.context_logits, rows[:4])
    assert value.context_logits_origins == (origin(0, 4),)


@pytest.mark.parametrize("producer", ["normal", "speculative"])
def test_recompute_chunk_entirely_in_generated_history_preserves_context(producer):
    value, original_rows = full_result()
    current_request = request_for_store(value, start=4, chunk=2)
    current_request.prompt_len = 6
    current_request.orig_prompt_len = 4
    rows = torch.full((2, 7), 999.0)
    if producer == "normal":
        HandleLogits()([current_request], [], rows, 1, [0, 2], True)
    else:
        sampler = SimpleNamespace(
            logit_projection=lambda states: pytest.fail("no original rows to project")
        )
        outputs = {
            spec_module.HIDDEN_STATES_KEY: rows,
            spec_module.GATHER_IDS_KEY: torch.empty(0, dtype=torch.int64),
        }
        scheduled = SimpleNamespace(context_requests=[current_request])
        assert spec_module.SpecSampler._stage_requested_logits(
            sampler, scheduled, outputs, [], 3
        ) == (None, [])
    torch.testing.assert_close(value.context_logits, original_rows)
    assert value.context_logits_origins == (origin(0, 4),)


def test_cancelled_before_execution_returns_no_invented_prompt_scores(monkeypatch) -> None:
    cancelled = SimpleNamespace(
        is_sequence_final=True, finish_reasons=[base_worker.tllm.FinishReason.CANCELLED]
    )
    response = SimpleNamespace(
        result=SimpleNamespace(
            context_logits=None, context_prompt_logprob_rows=None, get_result=lambda: cancelled
        )
    )
    monkeypatch.setattr(
        base_worker,
        "compute_prompt_logprobs",
        lambda *args, **kwargs: pytest.fail("queued cancellation must not reduce"),
    )
    assert base_worker._compute_pytorch_prompt_logprobs(generation(), response) is None


def compact_request(value, tokens, *, start=0, chunk=None, client_id=17):
    return SimpleNamespace(
        py_result=value,
        py_return_context_logits=True,
        py_context_logits_explicit=False,
        py_prompt_logprobs=0,
        py_client_id=client_id,
        orig_prompt_len=len(tokens),
        context_current_position=start,
        context_chunk_size=len(tokens) if chunk is None else chunk,
        prepopulated_prompt_len=0,
        get_tokens=lambda beam: tokens,
    )


def compact_response(value, tokens):
    response = response_for(value)
    response.result.get_result = lambda: SimpleNamespace(output_token_ids=[[3]])
    current = generation(len(tokens))
    current._generation_request.prompt_token_ids = tokens
    return current, response


def test_compact_replay_and_pickle_keep_exact_original_target_positions():
    source, target = result(), result()
    tokens = [0, 2, 4, 1]
    rows = torch.tensor(
        [
            [3.0, 1.0, 4.0, 2.0, 0.0, 5.0, -1.0],
            [2.0, -1.0, 5.0, 0.0, 3.0, 1.0, 4.0],
            [-2.0, 3.0, 0.0, 5.0, 1.0, 4.0, 2.0],
            [1.0, 0.0, -3.0, 4.0, 2.0, 5.0, 3.0],
        ]
    )
    request = compact_request(source, tokens, chunk=3)
    store_context_logits(request, rows[:3])
    target.apply_diff(pickle.loads(pickle.dumps(source.get_diff())))
    source.reset_diff()
    replacement = rows[1:] + torch.tensor([0.0, 1.0, -1.0, 2.0, 0.0, -2.0, 3.0])
    request.context_current_position = 1
    store_context_logits(request, replacement)
    target.apply_diff(pickle.loads(pickle.dumps(source.get_diff())))
    expected_logits = torch.cat([rows[:1], replacement])
    expected = compute_prompt_logprobs(0, expected_logits, tokens[1:] + [3])
    for value in (source, target):
        assert value.context_logits is None
        assert value._context_logits._storage is None
        assert value.context_logits_origins == (origin(0, 4),)
        assert torch.cat(value.context_prompt_logprob_rows).shape == (3, 2)
        current, response = compact_response(value, tokens)
        assert base_worker._compute_pytorch_prompt_logprobs(current, response) == expected
    assert target.get_diff().prompt_logprob_rows == []


def test_compact_single_token_prompt_keeps_its_final_prediction_row():
    value = result(1)
    tokens = [2]
    rows = torch.tensor([[3.0, 1.0, 4.0, 2.0, 0.0, 5.0, -1.0]])
    store_context_logits(compact_request(value, tokens), rows)
    assert value.context_logits_origins == (origin(0, 1, 1),)
    assert torch.cat(value.context_prompt_logprob_rows).shape == (0, 2)
    current, response = compact_response(value, tokens)
    expected = compute_prompt_logprobs(0, rows, [3])
    assert base_worker._compute_pytorch_prompt_logprobs(current, response) == expected


def test_compact_suffix_is_rejected_before_result_reduction(monkeypatch):
    value = result()
    tokens = [0, 2, 4, 1]
    store_context_logits(compact_request(value, tokens, start=2, chunk=2), torch.ones(2, 7))
    current, response = compact_response(value, tokens)
    monkeypatch.setattr(
        base_worker,
        "reduce_prompt_logprobs",
        lambda *args, **kwargs: pytest.fail("incomplete original rows must not be reduced"),
    )
    with pytest.raises(base_worker.IncompleteContextLogitsError, match="full original prompt"):
        base_worker._compute_pytorch_prompt_logprobs(current, response)


def test_compact_identity_change_is_rejected_before_device_reduction(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import context_logits_plan

    value = result()
    tokens = [0, 2, 4, 1]
    request = compact_request(value, tokens, chunk=2)
    store_context_logits(request, torch.ones(2, 7))
    request.py_client_id = 18
    request.context_current_position = 2
    monkeypatch.setattr(
        context_logits_plan,
        "reduce_prompt_logprobs",
        lambda *args, **kwargs: pytest.fail("misbound rows must not be reduced"),
    )
    with pytest.raises(ValueError, match="identity changed"):
        store_context_logits(request, torch.ones(2, 7))


def test_compact_recompute_excludes_generated_history_and_replaces_tail():
    value = result()
    tokens = [0, 2, 4, 1]
    rows = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    request = compact_request(value, tokens, chunk=6)
    store_context_logits(request, rows)
    assert value.context_logits_origins == (origin(0, 4),)
    current, response = compact_response(value, tokens)
    expected = compute_prompt_logprobs(0, rows[:4], tokens[1:] + [3])
    assert base_worker._compute_pytorch_prompt_logprobs(current, response) == expected
    original = torch.cat(value.context_prompt_logprob_rows).clone()
    request.context_current_position = 4
    request.context_chunk_size = 2
    store_context_logits(request, rows[4:])
    torch.testing.assert_close(torch.cat(value.context_prompt_logprob_rows), original)


def test_compact_diff_requires_one_origin_per_fragment():
    value = result()
    diff = PyResult.Diff(prompt_logprob_rows=[torch.ones(3, 2, dtype=torch.float64)])
    with pytest.raises(ValueError, match="prompt logprobs and origin diff lengths"):
        value.apply_diff(diff)
    assert value.context_prompt_logprob_rows is None


def test_reusable_context_scratch_is_owned_by_storage_and_transport():
    value = result()
    scratch = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    expected = scratch.clone()
    value.append_context_logits(scratch, origin(0, 4), reusable=True)
    scratch.fill_(-100)
    torch.testing.assert_close(value.context_logits, expected)
    torch.testing.assert_close(value.get_diff().context_logits_list[0], expected)


def test_incomplete_origins_fail_only_their_request():
    from unittest.mock import Mock

    value = result()
    tokens = [0, 2, 4, 1]
    store_context_logits(compact_request(value, tokens, start=2, chunk=2), torch.ones(2, 7))
    current, response = compact_response(value, tokens)
    response.result.is_final = False
    worker = SimpleNamespace(
        _results={17: current}, _refused_client_ids=set(), abort_request=Mock()
    )
    wrapped = base_worker._maybe_wrap_response(worker, response)
    assert isinstance(wrapped, base_worker.ErrorResponse)
    assert "full original prompt" in wrapped.error_msg
    worker.abort_request.assert_called_once_with(17)
    assert worker._refused_client_ids == {17}


def test_full_logits_cannot_mask_missing_compact_prefix_rows():
    value, _ = full_result()
    tokens = [0, 2, 4, 1]
    store_context_logits(compact_request(value, tokens, start=2, chunk=2), torch.ones(2, 7))
    assert value.context_logits_origins == (origin(0, 4),)
    current, response = compact_response(value, tokens)
    with pytest.raises(base_worker.IncompleteContextLogitsError, match="full original prompt"):
        base_worker._compute_pytorch_prompt_logprobs(current, response)
