# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated handoff of the Beam NVFP4 target with its DFlash 2 drafter.

Host-side checks, against expectations written out here rather than derived from the code
under test:

* the auxiliary slot's fixed-width handoff record round-trips, and survives a copy through
  the same per-peer auxiliary transfer layout the NIXL write uses;
* the peer registrar's cache-identity gate accepts equal declarations (or none on both sides)
  and names the differing parts otherwise;
* the Beam model's validation of a received record: identity, transfer id, prompt length and
  full draft-context coverage, each fault refused, and its import into the DFlash worker;
* the worker recipes' transceiver setting refuses a transfer timeout that is not positive;
* the coordinator publishes a receive only once the admission event of the target's KV cache
  manager, which holds the draft layers too, completes;
* the executor ends generation requests no batch reaches yet: an abort of a pending receive
  only once the transceiver reports it safe to free, without a decode; an expired receive is
  failed, also when it completed after its deadline; a received request whose context token
  already ends it (budget, end id) is finished with that token instead of being scheduled.

CUDA checks (one device, no model files): the transfer page table of one ``KVCacheManagerV2``
holding target and draft layers (the draft layers after the target's, in their own pool group);
the V2 scheduler's generation-init admission of both together (the incoming prompt declared as
history, one readiness event, SKIP leaving no active cache when the cache refuses, admission
once capacity frees); and the DFlash worker binding an imported draft-context cursor to a fresh
slot at the first generation step, without a second seed from positions when the draft layers
share the target's cache.
"""

from __future__ import annotations

import ctypes
import hashlib
import types
from collections import deque

import numpy as np
import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (  # noqa: E501
    DISAGG_HANDOFF_VERSION,
    ModelingV2BeamFullModelNvfp4Sm103Tp4,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import (
    HANDOFF_RECORD_WORDS,
    AuxBuffer,
    build_aux_transfer_layout,
    decode_handoff_record,
    encode_handoff_record,
)
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

PROMPT = 4097
TRANSFER_ID = 905
IDENTITY = {"version": 1, "parts": {"target_manifest": "1" * 64, "kv_cache_scales": "2" * 64}}


def _digest(identity: dict) -> str:
    import json

    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record(**overrides) -> dict:
    record = {
        "version": DISAGG_HANDOFF_VERSION,
        "identity": _digest(IDENTITY),
        "transfer_id": TRANSFER_ID,
        "prompt_len": PROMPT,
        "draft": {"draft_ctx_len": PROMPT, "next_position": PROMPT},
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------------- record


def test_handoff_record_round_trips_through_its_words():
    record = _record(transfer_id=(1 << 40) + 7)
    words = encode_handoff_record(record)
    assert len(words) == HANDOFF_RECORD_WORDS
    assert words[0] == 1 and words[2] == (1 << 40) + 7 and words[3:6] == [PROMPT, PROMPT, PROMPT]
    assert decode_handoff_record(words) == {**record, "identity": record["identity"][:32]}
    assert decode_handoff_record(encode_handoff_record(_record(draft=None)))["draft"] is None
    assert decode_handoff_record(encode_handoff_record(None)) is None
    with pytest.raises(ValueError, match="128 bits"):
        encode_handoff_record(_record(identity="abc"))
    with pytest.raises(ValueError, match="words"):
        decode_handoff_record([0, 1, 2])


def _context_request(record):
    return types.SimpleNamespace(
        get_last_tokens=lambda: [17],
        py_draft_tokens=[0] * 7,
        py_disaggregated_params=None,
        prompt_len=PROMPT,
        cached_tokens=0,
        py_disagg_handoff=record,
    )


def test_aux_slot_carries_the_record_through_the_transfer_layout():
    """The record written into a sender slot reads back from the receiver slot after a copy of
    exactly the items the auxiliary transfer layout names, as the NIXL write moves them."""
    src = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=7)
    dst = AuxBuffer(max_slot_num=4, beam_width=1, max_draft_len=7)
    assert len(src.meta.ptrs) == 5 and int(src.meta.item_sizes[4]) == 8 * HANDOFF_RECORD_WORDS
    layout = build_aux_transfer_layout(src.meta, dst.meta)
    src_slot, dst_slot = src.alloc_slot().id, dst.alloc_slot().id
    dst_slot = dst.alloc_slot().id  # a different slot index on the receiver
    record = _record()
    src.fill_slot(src_slot, _context_request(record))
    for base_s, base_d, size_s in zip(
        layout.src_base_ptrs, layout.dst_base_ptrs, layout.src_item_sizes
    ):
        ctypes.memmove(
            int(base_d) + dst_slot * int(size_s), int(base_s) + src_slot * int(size_s), int(size_s)
        )
    assert dst.get_slot_handoff(dst_slot) == {**record, "identity": record["identity"][:32]}
    assert dst.get_slot_tokens(dst_slot) == ([17], [0] * 7)
    # A slot filled without a record reads back as none.
    empty = src.alloc_slot().id
    src.fill_slot(empty, _context_request(None))
    assert src.get_slot_handoff(empty) is None


# ---------------------------------------------------------------------------------- identity


def _rank_info(identity):
    return RankInfo(
        instance_name="i",
        instance_rank=0,
        tp_size=4,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[57],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        cache_identity=identity,
    )


def test_identity_gate_accepts_equal_or_absent_declarations():
    PeerRegistrar.validate_cache_identity_compatible(_rank_info(None), _rank_info(None))
    PeerRegistrar.validate_cache_identity_compatible(_rank_info(IDENTITY), _rank_info(IDENTITY))
    # The serialized form round-trips the declaration unchanged.
    assert RankInfo.from_bytes(_rank_info(IDENTITY).to_bytes()).cache_identity == IDENTITY


@pytest.mark.parametrize(
    "mine,theirs,match",
    [
        (IDENTITY, None, "one side only"),
        (None, IDENTITY, "one side only"),
        (IDENTITY, {**IDENTITY, "version": 2}, "versions differ"),
        (
            IDENTITY,
            {"version": 1, "parts": {**IDENTITY["parts"], "kv_cache_scales": "3" * 64}},
            r"\['kv_cache_scales'\]",
        ),
        (
            IDENTITY,
            {"version": 1, "parts": {**IDENTITY["parts"], "drafter_manifest": "4" * 64}},
            r"\['drafter_manifest'\]",
        ),
    ],
)
def test_identity_gate_names_what_differs(mine, theirs, match):
    with pytest.raises(ValueError, match=match):
        PeerRegistrar.validate_cache_identity_compatible(_rank_info(mine), _rank_info(theirs))


# ------------------------------------------------------------------------------ Beam handoff


class _StubWorker:
    def __init__(self, export=None):
        self.export = export
        self.imports = []
        self.released = []

    def export_disagg_context(self, request_id):
        return self.export

    def import_disagg_context(self, request_id, draft_ctx_len, next_position):
        self.imports.append((request_id, draft_ctx_len, next_position))

    def release_request_state(self, request_id):
        self.released.append(request_id)


def _bare_model(worker=None):
    model = ModelingV2BeamFullModelNvfp4Sm103Tp4.__new__(ModelingV2BeamFullModelNvfp4Sm103Tp4)
    nn.Module.__init__(model)
    model.spec_worker = worker
    model.use_separate_draft_kv_cache = worker is not None
    model._disagg_identity = IDENTITY
    return model


def _generation_request(record):
    decoded = None if record is None else decode_handoff_record(encode_handoff_record(record))
    return types.SimpleNamespace(
        py_request_id=11,
        prompt_len=PROMPT,
        context_phase_params=types.SimpleNamespace(req_id=TRANSFER_ID),
        py_disagg_handoff=decoded,
    )


def test_a_matching_record_is_accepted_and_imported():
    worker = _StubWorker()
    model = _bare_model(worker)
    assert model.transfers_draft_kv_cache
    req = _generation_request(_record())
    assert model.validate_disagg_generation_request(req) is None
    model.prepare_disagg_generation_request(req)
    assert worker.imports == [(11, PROMPT, PROMPT)]
    model.release_request_state(11)
    assert worker.released == [11]


def test_the_exported_record_validates_after_the_aux_encoding():
    worker = _StubWorker(export={"draft_ctx_len": PROMPT, "next_position": PROMPT})
    model = _bare_model(worker)
    ctx = types.SimpleNamespace(
        py_disaggregated_params=None,
        request_id=TRANSFER_ID,
        py_request_id=TRANSFER_ID,
        prompt_len=PROMPT,
    )
    exported = model.export_disagg_context_request(ctx)
    assert exported["identity"] == _digest(IDENTITY) and exported["transfer_id"] == TRANSFER_ID
    assert model.validate_disagg_generation_request(_generation_request(exported)) is None


@pytest.mark.parametrize(
    "record,match",
    [
        (None, "no context handoff record"),
        (_record(version=DISAGG_HANDOFF_VERSION + 1), "version"),
        (_record(identity="f" * 64), "identity differs"),
        (_record(transfer_id=TRANSFER_ID + 1), "belongs to transfer"),
        (_record(prompt_len=PROMPT - 1), "4096-token prompt"),
        (_record(draft=None), "stored no draft context"),
        (_record(draft={"draft_ctx_len": PROMPT - 1, "next_position": PROMPT}), "covers 4096"),
        (_record(draft={"draft_ctx_len": PROMPT, "next_position": PROMPT + 1}), "next position"),
    ],
)
def test_a_record_that_does_not_fit_is_refused(record, match):
    error = _bare_model(_StubWorker()).validate_disagg_generation_request(
        _generation_request(record)
    )
    assert error is not None and match in error, error


def test_an_ordinary_worker_refuses_a_drafting_peer_and_accepts_its_own():
    model = _bare_model(None)
    assert not model.transfers_draft_kv_cache
    assert "has no drafter" in model.validate_disagg_generation_request(
        _generation_request(_record())
    )
    assert (
        model.validate_disagg_generation_request(_generation_request(_record(draft=None))) is None
    )
    model.prepare_disagg_generation_request(_generation_request(_record(draft=None)))  # no-op


def test_a_worker_transfer_timeout_must_be_positive():
    """The worker recipes' transceiver setting; a transfer that cannot complete is refused."""
    from pydantic import ValidationError

    from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

    config = CacheTransceiverConfig(
        backend="NIXL", transceiver_runtime="PYTHON", kv_transfer_timeout_ms=60000
    )
    assert (config.backend, config.transceiver_runtime, config.kv_transfer_timeout_ms) == (
        "NIXL",
        "PYTHON",
        60000,
    )
    for timeout in (0, -1):
        with pytest.raises(ValidationError, match="greater than 0"):
            CacheTransceiverConfig(
                backend="NIXL", transceiver_runtime="PYTHON", kv_transfer_timeout_ms=timeout
            )


# ------------------------------------------------------------------- receive publication


class _Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done


class _Manager:
    def __init__(self, event):
        self.event = event

    def take_disagg_receive_ready(self, req):
        event, self.event = self.event, None
        return event


def _coordinator(target_event):
    published = []
    transceiver = types.SimpleNamespace(
        kv_transfer_timeout_ms=None,
        request_and_receive_async=published.append,
    )
    coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=None,
        kv_cache_manager=_Manager(target_event),
        dist=None,
        effects=None,
        registry=None,
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
        draft_kv_cache_manager=None,
        is_kv_manager_v2=True,
        defer_unready_gen_receives=True,
    )
    return coordinator, published


def test_a_receive_waits_for_the_one_admission_event():
    """The target's KV cache manager admits the target and draft layers together, so one
    readiness event gates the receive."""
    target = _Event(done=False)
    coordinator, published = _coordinator(target)
    req = types.SimpleNamespace(py_request_id=4, state=LlmRequestState.DISAGG_GENERATION_INIT)
    fences = coordinator._take_receive_fences([req])
    assert fences[4] is target
    coordinator._unpublished_gen_receives[4] = (req, fences[4])
    coordinator._publish_ready_gen_receives()
    assert published == []
    target.done = True
    coordinator._publish_ready_gen_receives()
    assert published == [req]


# ------------------------------------------------------------------------- CUDA: managers

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
PAGE = 32


#: Target layers 0 (full attention) and 1 (512-token window), then five draft layers 2..6
#: (1024-token window), all FP8 with 2 KV heads of 128: one manager, three lifecycles.
TARGET_WINDOWS = (None, 512)
DRAFT_LAYERS = 5
DRAFT_WINDOW = 1024


def _unified_manager(max_tokens, max_seq_len=2048):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        KVCacheLayerOwner,
        KVCacheLayerSpec,
        KVCacheManagerV2,
        Role,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    def spec(layer_id, owner, window):
        return KVCacheLayerSpec(
            global_layer_id=layer_id,
            owner=owner,
            dtype=DataType.FP8,
            num_kv_heads=2,
            head_dim=128,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=window,
        )

    specs = [
        spec(layer_id, KVCacheLayerOwner.TARGET, window)
        for layer_id, window in enumerate(TARGET_WINDOWS)
    ] + [
        spec(len(TARGET_WINDOWS) + i, KVCacheLayerOwner.DRAFT, DRAFT_WINDOW)
        for i in range(DRAFT_LAYERS)
    ]
    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=True, enable_partial_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=len(TARGET_WINDOWS),
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=PAGE,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8,
        is_disagg=True,
        layer_specs=specs,
    )


@cuda
def test_the_transfer_page_table_carries_the_draft_layers_after_the_target():
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
        build_page_table_from_manager,
    )

    mgr = _unified_manager(64 * PAGE)
    try:
        num_target = len(TARGET_WINDOWS)
        assert mgr.draft_layer_ids == tuple(range(num_target, num_target + DRAFT_LAYERS))
        assert mgr.get_draft_kv_cache_view() is mgr
        table = build_page_table_from_manager(mgr)
        ids = sorted(ll.global_layer_id for lg in table.layer_groups for ll in lg.local_layers)
        assert ids == list(range(num_target + DRAFT_LAYERS))
        groups = {
            ll.global_layer_id: lg.pool_group_idx
            for lg in table.layer_groups
            for ll in lg.local_layers
        }
        draft_groups = {groups[i] for i in mgr.draft_layer_ids}
        assert len(draft_groups) == 1, "the drafter's uniform window is one lifecycle"
        assert draft_groups.isdisjoint(groups[i] for i in range(num_target))
    finally:
        mgr.shutdown()


def _generation_only_request(rid, prompt_len):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.bindings import executor as tllme
    from tensorrt_llm.disaggregated_params import DisaggregatedParams

    params = tllme.ContextPhaseParams([7], rid + 1000, None, [0] * 7, 0, "tcp://127.0.0.1:1")
    req = LlmRequest(
        request_id=rid,
        max_new_tokens=8,
        input_tokens=[(rid * 131 + j) % 1000 + 1 for j in range(prompt_len)],
        sampling_config=tllme.SamplingConfig(),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=params,
    )
    req.py_disaggregated_params = DisaggregatedParams(
        request_type="generation_only",
        ctx_request_id=rid + 1000,
        ctx_dp_rank=0,
        first_gen_tokens=[7],
        draft_tokens=[0] * 7,
    )
    return req


@cuda
def test_a_generation_init_admits_target_and_draft_layers_together():
    """One admission sizes the request's cache for the target and draft layers, declares the
    incoming prompt once and records one readiness event; a refused request keeps no active
    cache and no event, and is admitted once capacity frees."""
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import (
        KVCacheV2Scheduler,
        ScheduleAction,
    )
    from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy

    prompt = 600
    pages = -(-prompt // PAGE)
    mgr = _unified_manager(3 * pages * PAGE + PAGE // 2)
    try:
        scheduler = KVCacheV2Scheduler(
            max_batch_size=4,
            max_num_tokens=8192,
            kv_cache_manager=mgr,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            enable_recompute_pause=False,
        )
        admitted, refused = [], None
        for rid in range(1, 9):
            req = _generation_only_request(rid, prompt)
            assert req.state == LlmRequestState.DISAGG_GENERATION_INIT
            action, tokens = scheduler._try_schedule_disagg_gen_init(req, None)
            assert tokens == 0
            if action is ScheduleAction.SKIP:
                refused = req
                break
            assert action is ScheduleAction.SCHEDULED
            admitted.append(req)
        assert admitted, "no admission succeeded"
        assert refused is not None, "the cache never refused"
        for req in admitted:
            assert mgr.get_history_length(req) == prompt
            assert mgr.take_disagg_receive_ready(req) is not None
            assert mgr.take_disagg_receive_ready(req) is None
            assert req.context_current_position == 0  # cold: nothing reused
        refused_cache = mgr.kv_cache_map.get(refused.py_request_id)
        assert refused_cache is None or not refused_cache.is_active
        assert mgr.take_disagg_receive_ready(refused) is None
        # Releasing one admitted request makes the refused one admissible.
        mgr.free_resources(admitted[-1])
        action, _ = scheduler._try_schedule_disagg_gen_init(refused, None)
        assert action is ScheduleAction.SCHEDULED
        assert mgr.get_history_length(refused) == prompt
        for req in admitted[:-1] + [refused]:
            mgr.free_resources(req)
    finally:
        mgr.shutdown()


# --------------------------------------------------------------------------- CUDA: DFlash


def _bare_dflash_worker(num_slots):
    from tensorrt_llm._device_copy import DeviceCopyBuffer
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._utils import prefer_pinned

    worker = DFlashWorker.__new__(DFlashWorker)
    nn.Module.__init__(worker)
    worker._ctx_buf_inited = True
    worker._ctx_len = torch.zeros(num_slots + 1, dtype=torch.long, device="cuda")
    worker._ctx_len_host = [0] * (num_slots + 1)
    worker._free_slots = deque(range(num_slots))
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._dummy_slot = num_slots
    worker._graph_dummy_id_floor = 1 << 62
    worker._pending_disagg_imports = {}
    worker._ctx_unified_cache = False
    worker._max_ctx = 1 << 20
    worker._batch_to_slot_buffer = DeviceCopyBuffer(
        torch.zeros(num_slots, dtype=torch.long, device="cuda"), pin_memory=prefer_pinned()
    )
    return worker


@cuda
def test_an_imported_cursor_is_bound_at_the_first_generation_step():
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

    worker = _bare_dflash_worker(4)
    # A slot left over from another request must not leak its length into the import.
    worker._write_ctx_len({0: 123})
    metadata = DFlashSpecMetadata(
        max_num_requests=4,
        max_draft_len=7,
        max_total_draft_tokens=7,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
    )
    metadata._dflash_worker = worker
    worker.import_disagg_context(21, PROMPT, PROMPT)
    worker.import_disagg_context(99, 50, 50)
    worker.release_request_state(99)  # released before its first step
    metadata.request_ids = [21, 22]
    metadata.num_generations = 2
    metadata.prepare()
    torch.cuda.synchronize()
    imported, plain = worker._req_to_slot[21], worker._req_to_slot[22]
    assert int(worker._ctx_len[imported]) == worker._ctx_len_host[imported] == PROMPT
    assert worker._req_ctx_pos[21] == PROMPT
    assert int(worker._ctx_len[plain]) == worker._ctx_len_host[plain] == 0
    assert 22 not in worker._req_ctx_pos
    assert worker._pending_disagg_imports == {}
    assert worker._batch_to_slot_buffer.tensor.tolist()[:2] == [imported, plain]
    # The context side's export reads the same cursor back.
    assert worker.export_disagg_context(21) == {"draft_ctx_len": PROMPT, "next_position": PROMPT}
    assert worker.export_disagg_context(77) is None
    np.testing.assert_array_equal(worker._ctx_len.cpu().numpy()[[imported, plain]], [PROMPT, 0])


@cuda
def test_shared_draft_cache_seeds_from_the_cursor_once():
    """Draft layers in the target's KV cache manager: a received cursor binds as before and is
    not also seeded from positions; a request without one starts at its first position."""
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

    worker = _bare_dflash_worker(4)
    worker._ctx_unified_cache = True
    metadata = DFlashSpecMetadata(
        max_num_requests=4,
        max_draft_len=7,
        max_total_draft_tokens=7,
        spec_dec_mode=SpeculativeDecodingMode.DFLASH,
    )
    metadata._dflash_worker = worker
    worker.import_disagg_context(21, PROMPT, PROMPT)
    metadata.request_ids = [21, 22]
    metadata.num_generations = 2
    metadata.seq_lens = [8, 8]
    # Request 21's first position is not its cursor, so a second (position) seed would show.
    metadata.host_position_ids = torch.tensor(
        list(range(PROMPT + 5, PROMPT + 13)) + list(range(300, 308)), dtype=torch.int32
    )
    metadata.prepare()
    torch.cuda.synchronize()
    imported, plain = worker._req_to_slot[21], worker._req_to_slot[22]
    assert int(worker._ctx_len[imported]) == worker._ctx_len_host[imported] == PROMPT
    assert worker._req_ctx_pos[21] == PROMPT
    assert int(worker._ctx_len[plain]) == worker._ctx_len_host[plain] == 300
    assert worker._pending_disagg_imports == {}


# ------------------------------------------------- executor: requests no batch reaches yet

END_ID = 151674


def _gen_request(rid, max_new_tokens, first_token, state):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
    from tensorrt_llm.bindings import executor as tllme

    req = LlmRequest(
        request_id=rid,
        max_new_tokens=max_new_tokens,
        input_tokens=list(range(1, 18)),
        sampling_config=tllme.SamplingConfig(),
        is_streaming=False,
        end_id=END_ID,
        # As the LLM API sends it: responses carry generated tokens only.
        exclude_input_from_output=True,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=tllme.ContextPhaseParams(
            [first_token], 9000 + rid, None, [], 0, "tcp://127.0.0.1:1"
        ),
    )
    req.state = state
    return req


class _Transceiver:
    """A Python transceiver whose receive cancellation is safe once ``drained`` holds."""

    kv_transfer_timeout_ms = 60000

    def __init__(self, drained):
        self.drained = set(drained)
        self.cancel_calls = []

    def cancel_request(self, req):
        self.cancel_calls.append(req.py_request_id)
        return req.py_request_id in self.drained

    def has_inflight_transfer(self, req):
        return req.py_request_id not in self.drained

    def commit_blocks_for_reuse(self, req):
        pass


def _executor(requests, transceiver, canceled=()):
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    failures = []
    effects = types.SimpleNamespace(
        fail_requests=lambda msg, reqs, charge_budget: failures.append(
            (msg, [r.py_request_id for r in reqs])
        )
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.kv_cache_transceiver = transceiver
    executor.active_requests = list(requests)
    executor.canceled_req_ids = list(canceled)
    executor.max_seq_len = 8192
    executor.iter_counter = 0
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    # PyExecutor.disagg builds this coordinator on first use; the shell supplies it.
    executor._disagg_coordinator = DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=None,
        kv_cache_manager=None,
        dist=None,
        effects=effects,
        registry=None,
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )
    return executor, failures


def test_a_pending_receive_is_aborted_without_a_decode_once_safe_to_free():
    from tensorrt_llm.bindings import executor as tllme

    req = _gen_request(1, 16, 77, LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)
    transceiver = _Transceiver(drained=())
    executor, failures = _executor([req], transceiver, canceled=[1])
    # The sender may still be writing: nothing ends, and the abort stays pending for a retry.
    executor._cancel_pending_gen_receives()
    assert executor.active_requests == [req] and executor.canceled_req_ids == [1]
    assert executor._pending_transfer_responses == []
    transceiver.drained.add(1)
    executor._cancel_pending_gen_receives()
    assert transceiver.cancel_calls == [1, 1]
    assert executor.active_requests == [] and executor.canceled_req_ids == []
    assert executor._pending_response_terminations == [req]
    ((rid, response),) = executor._pending_transfer_responses
    response.result.deserialize()
    assert rid == 1 and response.result.is_final
    assert response.result.output_token_ids == [[]], "the aborted request decoded"
    assert response.result.finish_reasons == [tllme.FinishReason.CANCELLED]
    assert failures == []


def test_an_expired_receive_fails_and_a_late_completion_is_not_decoded():
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState as State

    in_flight = _gen_request(2, 16, 77, State.DISAGG_GENERATION_TRANS_IN_PROGRESS)
    late = _gen_request(3, 16, 78, State.DISAGG_GENERATION_TRANS_COMPLETE)
    on_time = _gen_request(4, 16, 79, State.DISAGG_GENERATION_TRANS_COMPLETE)
    for req in (in_flight, late):
        req.py_kv_transfer_timed_out = True
    transceiver = _Transceiver(drained=(3, 4))
    executor, failures = _executor([in_flight, late, on_time], transceiver)
    executor._cancel_pending_gen_receives()
    # The completed-late receive fails; the in-flight one waits until its writer is fenced.
    assert failures == [("Request 3 timed out", [3])]
    assert in_flight.state == State.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert late.state != State.DISAGG_GENERATION_TRANS_COMPLETE
    assert on_time.state == State.DISAGG_GENERATION_TRANS_COMPLETE
    transceiver.drained.add(2)
    executor._cancel_pending_gen_receives()
    assert failures[-1] == ("Request 2 timed out", [2])
    assert in_flight.state == State.DISAGG_TRANS_ERROR
    assert transceiver.cancel_calls == [2, 3, 2]


@pytest.mark.parametrize(
    "max_new_tokens,first_token,reason,finished",
    [
        (1, 77, "LENGTH", True),
        (2, 77, None, False),
        (8, END_ID, "END_ID", True),
    ],
)
def test_a_receive_whose_first_token_ends_the_request_is_not_scheduled(
    max_new_tokens, first_token, reason, finished
):
    from tensorrt_llm.bindings import executor as tllme

    req = _gen_request(
        5, max_new_tokens, first_token, LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    )
    executor, _ = _executor([req], _Transceiver(drained=(5,)))
    executor._finish_exhausted_gen_receives()
    if not finished:
        assert executor.active_requests == [req]
        assert req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        assert executor._pending_transfer_responses == []
        return
    assert executor.active_requests == [] and executor._pending_response_terminations == [req]
    ((rid, response),) = executor._pending_transfer_responses
    response.result.deserialize()
    assert rid == 5 and response.result.is_final
    assert response.result.output_token_ids == [[first_token]]
    assert response.result.finish_reasons == [getattr(tllme.FinishReason, reason)]
