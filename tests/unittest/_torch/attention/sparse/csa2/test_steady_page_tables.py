# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 steady decode: reused page tables and steady metadata match full preparation."""

import math
import random
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import cache_manager, kernel
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    BAD_PAGE_INDEX,
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    Role,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequest,
    LlmRequestState,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import is_blackwell_datacenter
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import ReusePolicy

DRAFT_LEN = 5
BLOCK = 128
MAX_BATCH = 8

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)


def _manager(attention_dp, disaggregated, scratch, block=BLOCK, *, draft_stages=0):
    return CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=512 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=scratch,
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=4,
        tokens_per_block=block,
        max_seq_len=4096,
        max_batch_size=MAX_BATCH,
        max_num_tokens=4096,
        mapping=Mapping(enable_attention_dp=attention_dp),
        is_disagg=disaggregated,
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3), window_size=128, index_topk=4),
        spec_config=DSparkDecodingConfig(max_draft_len=DRAFT_LEN),
        layer_specs=[
            KVCacheLayerSpec(
                global_layer_id=4 + stage,
                owner=KVCacheLayerOwner.DRAFT,
                dtype=DataType.BF16,
                num_kv_heads=1,
                head_dim=512,
                buffer_roles=(Role.KEY,),
                sliding_window_size=129,
                reuse_policy=ReusePolicy.PRIVATE,
            )
            for stage in range(draft_stages)
        ]
        or None,
    )


def _metadata(manager):
    return CSA2TrtllmMetadata(
        max_num_requests=manager.max_batch_size,
        max_num_tokens=manager.max_num_tokens,
        kv_cache_manager=manager,
    )


def _admit(manager, request_id, prompt_len):
    """Prefill one request in its own context batch and move it to generation."""
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=4096,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.prompt_len)
    metadata = _metadata(manager)
    metadata.request_ids = [request_id]
    metadata.num_contexts = 1
    metadata.seq_lens = torch.tensor([request.context_chunk_size], dtype=torch.int32)
    metadata.prompt_lens = [request.context_chunk_size]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[request.context_current_position]
    )
    metadata.prepare()
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [request]
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    # Under overlap, the context step's sampled token is not appended yet.
    request.py_draft_tokens = [0] * DRAFT_LEN
    return request


def _dummy(manager, request_id):
    (request,) = manager.add_dummy_requests(
        [request_id], token_nums=[1], is_gen=True, max_num_draft_tokens=DRAFT_LEN
    )
    return request


def _accept(manager, requests, accepted):
    batch = ScheduledRequests()
    batch.generation_requests = list(requests)
    for request, count in zip(requests, accepted):
        request.py_num_accepted_draft_tokens = count
        request.py_rewind_len = DRAFT_LEN - count
        for _ in range(count + 1):
            request.add_new_token(1, 0)
    manager.update_resources(batch)


def _generation_batch(manager, active):
    for request in active:
        assert manager.try_allocate_generation(request)
    batch = list(active)
    return batch, [r.max_beam_num_tokens + DRAFT_LEN for r in batch]


def _accept_pending(manager, active, pending):
    # Under overlap scheduling, the previous step's acceptance lands after prepare.
    requests = [r for r in active if r.py_request_id in pending]
    if requests:
        _accept(manager, requests, [pending.pop(r.py_request_id) for r in requests])


def _prepare(metadata, batch, starts):
    metadata.request_ids = [request.py_request_id for request in batch]
    metadata.num_contexts = 0
    metadata.seq_lens = torch.full((len(batch),), DRAFT_LEN + 1, dtype=torch.int32)
    metadata.prompt_lens = [request.py_prompt_len for request in batch]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    metadata.prepare()


def _prepare_steady_entry(metadata, batch, starts, repeated):
    """prepare_steady_decode_step where it may stand in for prepare() (``repeated``: the previous step
    prepared this metadata for these requests), after writing kv_lens_cuda as the input staging does."""
    if not (repeated and metadata.is_cuda_graph and metadata.supports_steady_decode_step):
        _prepare(metadata, batch, starts)
        return
    kv_lens = [start + DRAFT_LEN + 1 for start in starts]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    metadata.kv_lens_cuda[: len(kv_lens)].copy_(torch.tensor(kv_lens, dtype=torch.int32))
    metadata.prepare_steady_decode_step(kv_lens)


def _forward_inputs(metadata, count):
    """Every page table and slot a decode forward reads."""
    metadata._csa2_swa_resolved = False
    metadata._ensure_swa_slots()
    outputs = {
        "block_offsets": metadata.kv_cache_block_offsets[:, :count],
        "positions": metadata.csa2_positions,
    }
    draft = metadata.kv_cache_manager.draft_page_table()
    if draft is not None:
        outputs["draft_pages"] = draft[:count]
    for name in (
        "csa2_global_page_tables",
        "_csa2_swa_page_tables",
        "csa2_swa_indices",
        "csa2_swa_write_slots",
        "csa2_visible_lengths",
        "csa2_main_write_slots",
        "_csa2_compressed_positions",
    ):
        outputs.update({(name, key): value for key, value in getattr(metadata, name).items()})
    for owner, batch in metadata._csa2_compression.items():
        for name in (
            "kv_page_table",
            "score_page_table",
            "start_positions",
            "kv_lengths",
            "cu_compressed_lengths",
        ):
            outputs[owner, name] = getattr(batch, name)
    return outputs


def _clone(outputs):
    return {name: tensor.clone() for name, tensor in outputs.items()}


def _batch_tables(manager):
    return (
        manager._batch_page_tables.clone(),
        manager._batch_rows.copy(),
        None if manager._batch_scratch is None else manager._batch_scratch.copy(),
    )


@contextmanager
def _always_convert(manager):
    """Convert every call over the full table width, as before the reuse; the reusing
    executor keeps its record of its own previous conversion."""
    convert = manager.compute_batch_page_tables

    def full(request_ids, num_contexts, swa_out=None, **kwargs):
        previous = manager._batch_conversion_inputs
        live = manager._table_live.copy(), manager._swa_table_live.copy()
        manager._batch_conversion_inputs = None
        manager._table_live[:] = manager.host_kv_cache_block_offsets.shape[-1]
        manager._swa_table_live[:] = manager.host_kv_cache_block_offsets.shape[-1]
        written = convert(request_ids, num_contexts, swa_out=swa_out, **kwargs)
        manager._batch_conversion_inputs = previous
        np.maximum(manager._table_live, live[0], out=manager._table_live)
        np.maximum(manager._swa_table_live, live[1], out=manager._swa_table_live)
        return written

    manager.compute_batch_page_tables = full
    try:
        yield
    finally:
        del manager.compute_batch_page_tables


class _Graph:
    """Graph metadata of one batch size with a captured consumer of its forward inputs."""

    def __init__(self, base, count):
        self.metadata = base.create_cuda_graph_metadata(count)
        self.count = count
        self.graph = None

    def _pointers(self):
        return {
            name: tensor.data_ptr()
            for name, tensor in _forward_inputs(self.metadata, self.count).items()
        }

    def run(self):
        if self.graph is None:
            for _ in range(2):
                _forward_inputs(self.metadata, self.count)
            self.pointers = self._pointers()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = _clone(_forward_inputs(self.metadata, self.count))
            # As a decode graph's query-tile capture records it: no replay geometry.
            self.metadata._csa2_replay_capture_signature = None
        self.graph.replay()
        observed = _clone(self.outputs)
        # The replayed consumer reads the buffers captured at the first step.
        assert self.pointers == self._pointers()
        return observed


class _Executor:
    """One eager metadata and graph metadata per batch size, as an executor holds them."""

    def __init__(self, manager, steady_entry=False, *, force_full_prepare=False):
        self.manager = manager
        self.eager = _metadata(manager)
        self.graphs = {}
        self.steady_entry = steady_entry
        self.force_full_prepare = force_full_prepare
        self.previous = None

    def _prepare_step(self, metadata, batch, starts, repeated=False):
        if self.force_full_prepare:
            with patch.object(metadata, "_steady_generation_step", return_value=None):
                _prepare(metadata, batch, starts)
        elif self.steady_entry:
            _prepare_steady_entry(metadata, batch, starts, repeated)
        else:
            _prepare(metadata, batch, starts)

    def step(self, batch, starts, use_graph):
        """Prepare one decode step; return the batch tables and every forward input."""
        count = len(batch)
        if use_graph:
            graph = self.graphs.get(count)
            if graph is None:
                graph = self.graphs[count] = _Graph(self.eager, count)
            ids = [r.py_request_id for r in batch]
            repeated = (
                self.previous is not None
                and self.previous[0] is graph.metadata
                and self.previous[1] == ids
            )
            self.previous = (graph.metadata, ids)
            self._prepare_step(graph.metadata, batch, starts, repeated)
            return _batch_tables(self.manager), graph.run()
        self.previous = None
        self._prepare_step(self.eager, batch, starts)
        return _batch_tables(self.manager), _clone(_forward_inputs(self.eager, count))


def _assert_identical(observed, expected, step):
    assert observed.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(observed[name], tensor), f"step {step}: {name} differs"


def _run_mixed_adp_preparation(rank: int, rendezvous: str) -> None:
    from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config, input_preparer
    from _torch.pyexecutor.engine_test_utils import DummyModel

    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputBuffers
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
    from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
    from tensorrt_llm.llmapi import MTPDecodingConfig

    torch.cuda.set_device(rank)
    dist = torch.distributed
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    exchanges = []

    def allgather(values):
        exchanges.append(tuple(values))
        local = torch.tensor(values, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        return torch.stack(gathered).numpy()

    manager = _manager(True, False, False)
    active = []
    try:
        active.extend(_admit(manager, i, 20 * i + rank) for i in (1, 2, 3))
        for slot, request in enumerate(active, 1):
            request.py_seq_slot = request.py_batch_idx = slot
            assert manager.try_allocate_generation(request)
        padding = _dummy(manager, CUDA_GRAPH_DUMMY_REQUEST_ID)
        padding.is_cuda_graph_dummy = True
        active.append(padding)
        batch = ScheduledRequests()
        batch.generation_requests = [*active[:2], padding, padding]
        count = len(batch.generation_requests)
        t = DRAFT_LEN + 1
        bound = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: manager})
        config = decoder_config(
            spec_config=MTPDecodingConfig(max_draft_len=DRAFT_LEN),
            enable_attention_dp=True,
            steady_decode_prepare=True,
        )
        mapping = Mapping(world_size=2, rank=rank, tp_size=2, enable_attention_dp=True)
        model = DummyModel(config.dtype)
        tensors = SampleStateTensorsSpec(
            new_tokens=torch.arange(t * MAX_BATCH, device="cuda", dtype=torch.int32).view(
                t, MAX_BATCH, 1
            ),
            new_tokens_lens=torch.ones(MAX_BATCH, device="cuda", dtype=torch.int32),
            next_draft_tokens=torch.arange(
                MAX_BATCH * DRAFT_LEN, device="cuda", dtype=torch.int32
            ).view(MAX_BATCH, DRAFT_LEN),
        )

        def prepare(preparer, metadata, spec):
            return preparer.prepare_inputs(
                batch,
                manager,
                metadata,
                spec,
                tensors,
                resource_manager=bound,
                enable_spec_decode=True,
                runtime_draft_len=DRAFT_LEN,
                is_dummy=False,
            )

        arms = []
        for _ in range(2):
            buffers = DecoderInputBuffers.for_config(config, False)
            preparer = input_preparer(config, model=model, buffers=buffers, mapping=mapping)
            preparer.dist = SimpleNamespace(tp_cp_allgather_int64=allgather)
            graph_runner = SimpleNamespace(
                enabled=True,
                shared_static_tensors={},
                take_agreed_replay_key=lambda _: None,
            )
            preparer.attach_cuda_graph_runners(graph_runner, None)
            metadata = _metadata(manager).create_cuda_graph_metadata(
                count, max_draft_tokens=DRAFT_LEN
            )
            spec = Mock(
                context_prompt_lookahead_tokens=None,
                supports_steady_prepare=False,
                runtime_draft_len=DRAFT_LEN,
            )
            spec.dp_num_tokens = lambda spec=spec: spec.num_tokens
            prepare(preparer, metadata, spec)
            assert preparer._steady_mtp.record.attn_steady
            _forward_inputs(metadata, count)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _clone(_forward_inputs(metadata, count))
                output["input_ids"] = buffers.input_ids_cuda[: count * t].clone()
            arms.append((preparer, metadata, spec, graph, output))

        batch.generation_requests = [active[0], active[1 + rank], padding, padding]
        outputs = []
        for index, (preparer, metadata, spec, graph, output) in enumerate(arms):
            if rank:
                # A joining request alongside an existing request takes the full pass.
                active[2].py_batch_idx = None
            if index:
                preparer._steady_mtp.invalidate()
            with patch.object(
                preparer,
                "_apply_steady_mtp_fast_prepare",
                wraps=preparer._apply_steady_mtp_fast_prepare,
            ) as steady:
                prepare(preparer, metadata, spec)
            assert steady.called == (rank == 0 and index == 0)
            graph.replay()
            outputs.append(_clone(output))
        _assert_identical(*outputs, "mixed attention-DP replay")
        assert len(exchanges) == 4 and exchanges[-2] == exchanges[-1]
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()
        dist.destroy_process_group()


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_mixed_attention_dp_steady_and_full_replay(tmp_path):
    torch.multiprocessing.spawn(
        _run_mixed_adp_preparation, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2
    )


def _counted_conversions(monkeypatch):
    """Record each conversion launch: True when it converted only some rows."""
    conversions = []
    convert = kernel.convert_page_tables
    monkeypatch.setattr(
        kernel,
        "convert_page_tables",
        lambda *args, **kwargs: (
            conversions.append(kwargs.get("dst_rows") is not None) or convert(*args, **kwargs)
        ),
    )
    return conversions


@skip_pre_blackwell
@pytest.mark.parametrize(
    "block,seed,attention_dp,disaggregated,scratch",
    [(BLOCK, 0, True, True, False), (2 * BLOCK, 1, False, True, True)],
    ids=["128-0-adp_disagg-no_scratch", "256-1-tep_disagg-scratch"],
)
@torch.inference_mode()
def test_steady_decode_reuses_converted_tables(
    monkeypatch, attention_dp, disaggregated, scratch, seed, block
):
    rng = random.Random(seed)
    conversions = _counted_conversions(monkeypatch)
    manager = _manager(
        attention_dp, disaggregated, scratch, block, draft_stages=3 if attention_dp else 0
    )
    reusing, converting = _Executor(manager), _Executor(manager)
    active, pending, padding = [], {}, None
    next_id, previous_ids = 1, None
    stats = dict(steps=0, skipped=0, crossings=0, partial=0)

    def accept(requests):
        requests = [r for r in requests if r.py_request_id in pending]
        if requests:
            _accept(manager, requests, [pending.pop(r.py_request_id) for r in requests])

    try:
        for step in range(160):
            if len(active) < 5 and (not active or rng.random() < 0.06):
                active.append(_admit(manager, next_id, rng.randrange(block - 8, 5 * block + 60)))
                next_id += 1
            elif len(active) > 1 and rng.random() < 0.04:
                done = active.pop(rng.randrange(len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            if len(active) > 1 and rng.random() < 0.08:
                rng.shuffle(active)
            transient = None
            if attention_dp and rng.random() < 0.1:
                # An attention-DP rank without requests decodes a dummy for one step.
                accept(active)
                transient = _dummy(manager, ATTENTION_DP_DUMMY_REQUEST_ID)
                batch, starts = [transient], [0]
            else:
                for request in active:
                    assert manager.try_allocate_generation(request)
                batch = list(active)
                starts = [r.max_beam_num_tokens - 1 + DRAFT_LEN + 1 for r in batch]
                if rng.random() < 0.2:
                    # CUDA graph padding keeps one dummy for the whole run.
                    padding = padding or _dummy(manager, CUDA_GRAPH_DUMMY_REQUEST_ID)
                    batch.append(padding)
                    starts.append(0)
            use_graph = rng.random() < 0.75
            before = len(conversions)
            previous_state = manager.draft_page_table_state()
            tables, observed = reusing.step(batch, starts, use_graph)
            converted = len(conversions) > before
            state = manager.draft_page_table_state()
            assert state is not None
            assert (state is previous_state) == (not converted)
            stats["partial"] += any(conversions[before:])
            with _always_convert(manager):
                expected_tables, expected = converting.step(batch, starts, use_graph)
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            assert np.array_equal(tables[1], expected_tables[1])
            assert (tables[2] is None) == (expected_tables[2] is None)
            if tables[2] is not None:
                assert np.array_equal(tables[2], expected_tables[2])
            _assert_identical(observed, expected, step)
            if "draft_pages" in observed and transient is None:
                pages = manager.get_batch_cache_indices(
                    [batch[0].py_request_id], manager.draft_layer_ids[0]
                )[0]
                mapped = next(column for column, page in enumerate(pages) if page >= 0)
                unmapped = next(
                    (column for column, page in enumerate(pages) if page < 0), len(pages)
                )
                assert observed["draft_pages"][0, mapped].item() == pages[mapped]
                assert observed["draft_pages"][0, unmapped].item() == -1
            ids = [r.py_request_id for r in batch]
            stats["steps"] += 1
            stats["skipped"] += not converted
            stats["crossings"] += converted and ids == previous_ids
            previous_ids = ids
            if transient is not None:
                manager.free_resources(transient)
                continue
            # Overlap scheduling: acceptance lands after the next step is prepared.
            accept(active)
            pending.update((r.py_request_id, rng.randrange(DRAFT_LEN + 1)) for r in active)
        assert stats["steps"] == 160, stats
        if not scratch:
            # Scratch-reuse caches publish every resize, so only runs without scratch reuse.
            assert stats["crossings"], stats
            # A crossing of one request converts that request's rows only.
            assert stats["partial"], stats
            # Switching between graph and eager destinations converts again.
            assert stats["steps"] // 8 < stats["skipped"] < stats["steps"], stats
    finally:
        for request in [*active, *([padding] if padding else [])]:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@torch.inference_mode()
def test_draft_conversion_keeps_holes_with_scratch_enabled():
    manager = _manager(False, True, True, 2 * BLOCK, draft_stages=3)
    request = _admit(manager, 1, 300)
    try:
        params = manager._device_batch_page_params
        host = params.cpu()
        draft = manager._draft_page_table_index
        scratch_table = next(index for index, row in enumerate(host) if row[4] > 0)
        base = torch.full((int(host[:, 0].max()) + 1, 1, 3), -1, dtype=torch.int32, device="cuda")
        base[:, :, 0] = 7
        rows = torch.zeros(1, dtype=torch.int32, device="cuda")
        ranges = torch.tensor(
            [[[1, 2]]] * (int(host[:, 3].max()) + 1), dtype=torch.int32, device="cuda"
        )
        slots = torch.full((*ranges.shape[:2], 1), 23, dtype=torch.int32, device="cuda")
        out = torch.empty((len(host), 1, 3), dtype=torch.int32, device="cuda")
        kernel.convert_page_tables(base, rows, params, out, -1, ranges, slots)
        assert host[draft, 4] == 0
        assert out[draft, 0, 1].item() == -1
        assert (
            out[scratch_table, 0, 1].item() == 23 * host[scratch_table, 1] + host[scratch_table, 2]
        )
    finally:
        manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize(
    "block,seed,scratch",
    [(BLOCK, 0, False), (2 * BLOCK, 1, True)],
    ids=["128-0-no_scratch", "256-1-scratch"],
)
@torch.inference_mode()
def test_steady_spec_steps_match_full_prepare(monkeypatch, scratch, seed, block):
    """Steady DSpark verification steps read exactly what a full preparation writes, also when a
    new request takes a finished request's row."""
    rng = random.Random(seed)
    hits, joins = [], []
    predicate = CSA2TrtllmMetadata._steady_generation_step

    def counted(self, snapshot):
        hit = predicate(self, snapshot)
        hits.append(hit is not None)
        joins.append(isinstance(hit, np.ndarray) and bool((hit < 0).any()))
        return hit

    monkeypatch.setattr(CSA2TrtllmMetadata, "_steady_generation_step", counted)
    manager = _manager(False, False, scratch, block)
    steady, full = _Executor(manager), _Executor(manager, force_full_prepare=True)
    active, pending = [], {}
    next_id = 1

    try:
        for step in range(200):
            if len(active) < 3 and (not active or rng.random() < 0.03):
                active.append(_admit(manager, next_id, rng.randrange(block - 8, 5 * block + 60)))
                next_id += 1
            elif len(active) > 1 and rng.random() < 0.02:
                done = active.pop(rng.randrange(len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            elif active and rng.random() < 0.05:
                # A finished request leaves and a new one takes its row in the same step.
                row = rng.randrange(len(active))
                pending.pop(active[row].py_request_id, None)
                manager.free_resources(active[row])
                active[row] = _admit(manager, next_id, rng.randrange(block - 8, 5 * block + 60))
                next_id += 1
            if len(active) > 1 and rng.random() < 0.03:
                rng.shuffle(active)
            batch, starts = _generation_batch(manager, active)
            use_graph = rng.random() < 0.75
            try:
                tables, observed = steady.step(batch, starts, use_graph)
            except AssertionError as error:
                raise AssertionError(f"steady executor, step {step}, graph {use_graph}") from error
            try:
                expected_tables, expected = full.step(batch, starts, use_graph)
            except AssertionError as error:
                raise AssertionError(f"full executor, step {step}, graph {use_graph}") from error
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            _assert_identical(observed, expected, step)
            _accept_pending(manager, active, pending)
            # A step may also commit nothing.
            pending.update(
                (r.py_request_id, rng.randrange(DRAFT_LEN + 1))
                for r in active
                if rng.random() < 0.97
            )
        assert sum(hits) > len(hits) // 2 and any(joins), (sum(hits), len(hits), sum(joins))
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@torch.inference_mode()
def test_graph_clone_first_prepare_is_full():
    """A CUDA-graph clone's first prepare leaves its source's forward inputs untouched."""
    manager = _manager(False, False, False)
    request = _admit(manager, 1, 300)
    try:
        base = _metadata(manager)
        assert manager.try_allocate_generation(request)
        _prepare(base, [request], [request.max_beam_num_tokens - 1 + DRAFT_LEN + 1])
        clone = base.create_cuda_graph_metadata(1)
        # Every draft accepted: the next start advances by the full query length.
        _accept(manager, [request], [DRAFT_LEN])
        assert manager.try_allocate_generation(request)
        source = _clone(_forward_inputs(base, 1))
        _prepare(clone, [request], [request.max_beam_num_tokens - 1 + DRAFT_LEN + 1])
        _assert_identical(_forward_inputs(base, 1), source, "source")
    finally:
        manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("seed", [0])
@torch.inference_mode()
def test_block_offsets_refill_only_stale_entries(seed):
    """A destination the manager wrote before is refilled only where earlier batches left pages,
    and equals a destination filled in full: across growing, shrinking and reordered batches,
    unchanged batches and a second view of its memory."""
    rng = random.Random(seed)
    manager = _manager(False, False, False)
    shape = (manager.num_attention_op_pools, manager.max_batch_size, 2, manager.max_blocks_per_seq)
    memory = torch.zeros(2 * math.prod(shape), dtype=torch.int32, device="cuda")
    tracked = memory[: math.prod(shape)].view(shape)
    alias = memory[: math.prod(shape)].view(shape)
    active, pending, next_id = [], {}, 1
    try:
        for step in range(150):
            if len(active) < MAX_BATCH and (not active or rng.random() < 0.15):
                active.append(_admit(manager, next_id, rng.randrange(BLOCK - 8, 5 * BLOCK + 60)))
                next_id += 1
            elif len(active) > 1 and rng.random() < 0.12:
                done = active.pop(rng.randrange(len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            if len(active) > 1 and rng.random() < 0.1:
                rng.shuffle(active)
            for request in active:
                assert manager.try_allocate_generation(request)
            ids = [r.py_request_id for r in active]
            destination = alias if rng.random() < 0.05 else tracked
            for _ in range(2 if rng.random() < 0.3 else 1):
                # A repeated call has unchanged inputs and keeps the converted tables.
                manager.copy_batch_block_offsets(destination, ids, 1, 0, len(ids))
            full = torch.zeros(shape, dtype=torch.int32, device="cuda")
            manager.copy_batch_block_offsets(full, ids, 1, 0, len(ids))
            assert torch.equal(destination, full), f"step {step}"
            requests = [r for r in active if r.py_request_id in pending]
            if requests:
                _accept(manager, requests, [pending.pop(r.py_request_id) for r in requests])
            pending.update((r.py_request_id, rng.randrange(DRAFT_LEN + 1)) for r in active)
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("scratch", [False, True], ids=["no_scratch", "scratch"])
@torch.inference_mode()
def test_unchanged_page_storage_exit_matches_full_prepare(monkeypatch, scratch):
    """Steady-entry steps that keep growth blocks and the last conversion of unchanged page storage
    read what a full preparation writes over growth, retirements and admissions."""
    rng = random.Random(0)
    manager = _manager(False, False, scratch)
    manager.keep_growth_blocks()
    manager._unchanged_page_exit = True
    exits = []
    check = manager._unchanged_page_storage
    monkeypatch.setattr(
        manager, "_unchanged_page_storage", lambda *args: exits.append(check(*args)) or exits[-1]
    )
    steady = _Executor(manager, steady_entry=True)
    full = _Executor(manager, force_full_prepare=True)
    steady_calls = 0
    prepare_steady = CSA2TrtllmMetadata.prepare_steady_decode_step

    def counted_prepare(metadata, *args, **kwargs):
        nonlocal steady_calls
        steady_calls += 1
        return prepare_steady(metadata, *args, **kwargs)

    monkeypatch.setattr(CSA2TrtllmMetadata, "prepare_steady_decode_step", counted_prepare)
    active, pending, next_id = [], {}, 1
    previous, taken = None, 0
    try:
        for step in range(200):
            if len(active) < 3 and (not active or rng.random() < 0.05):
                active.append(_admit(manager, next_id, rng.randrange(BLOCK - 8, 3 * BLOCK)))
                next_id += 1
            elif len(active) > 1 and rng.random() < 0.04:
                done = active.pop(rng.randrange(len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            batch, starts = _generation_batch(manager, active)
            state = (
                [r.py_request_id for r in batch],
                [manager.kv_cache_map[r.py_request_id].page_storage_version for r in batch],
            )
            before = len(exits)
            tables, observed = steady.step(batch, starts, True)
            if any(exits[before:]):
                assert state == previous, f"step {step}: exit with changed caches or pages"
                taken += 1
            with _always_convert(manager):
                expected_tables, expected = full.step(batch, starts, True)
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            _assert_identical(observed, expected, step)
            previous = state
            _accept_pending(manager, active, pending)
            pending.update((r.py_request_id, rng.randrange(DRAFT_LEN + 1)) for r in active)
        assert taken > 200 // 4, taken
        assert steady_calls > 200 // 4, steady_calls
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@torch.inference_mode()
def test_overlapping_block_offset_views_convert_only_live_columns(monkeypatch):
    manager = _manager(False, False, False)
    active = []
    try:
        active.extend(_admit(manager, i, 10 * i) for i in (1, 2))
        widths = []
        convert = manager._convert_all_rows

        def capture_width(count, width, *args):
            widths.append(width)
            return convert(count, width, *args)

        monkeypatch.setattr(manager, "_convert_all_rows", capture_width)
        max_blocks = manager.max_blocks_per_seq
        shape = (len(manager.pp_layers), MAX_BATCH, 2, max_blocks)
        storage = torch.full(shape, 123, dtype=torch.int32, device="cuda")
        for ids in ([1], [2, 1], [1]):
            view = storage[:, : len(ids)]
            manager.copy_batch_block_offsets(view, ids, 1, 0, len(ids))
            live = max(manager.batch_table_live())
            assert 0 < live <= widths[-1] < max_blocks
            assert (view[..., live:] == BAD_PAGE_INDEX).all()
            assert (view[..., 0, :live] != BAD_PAGE_INDEX).any()
        assert len(widths) == 3
        expected = view.clone()
        other_stream = torch.cuda.Stream()
        other_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other_stream):
            manager.copy_batch_block_offsets(view, [], 1, 0, 0)
        manager.copy_batch_block_offsets(view, [1], 1, 0, 1)
        assert torch.equal(view, expected)

        # Direct metadata copies can write a wider row into a previously published view.
        # Switching to another destination and back must clear the earlier wide row.
        active.append(_admit(manager, 3, 7 * BLOCK))
        first = storage[:, :2]
        second = torch.empty_like(first)
        manager.copy_batch_block_offsets(first, [3, 1], 1, 0, 2)
        reference = first.clone()
        metadata = _metadata(manager)
        for layer, pages in zip(manager.pp_layers, first[:, :, 0]):
            metadata._bind_swa_pages(layer, pages, pages, [])
        other_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other_stream):
            manager.compute_batch_page_tables([1, 3], 0)
            copies = []
            for layer in manager.pp_layers:
                pages = manager.batch_page_table((layer, CSA2CacheRole.SWA), max_blocks)
                metadata._bind_swa_pages(layer, pages, None, copies)
            assert copies
            targets, sources = zip(*copies)
            with manager._write_swa_page_tables(targets):
                torch._foreach_copy_(targets, sources, non_blocking=True)
        manager.copy_batch_block_offsets(second, [3, 1], 1, 0, 2)
        manager.copy_batch_block_offsets(first, [3, 1], 1, 0, 2)
        assert torch.equal(second, reference)
        assert torch.equal(first, reference)
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@torch.inference_mode()
def test_full_conversions_order_shared_row_upload_across_streams(monkeypatch):
    manager = _manager(False, False, False)
    active = []
    try:
        active.extend(_admit(manager, i, BLOCK * i) for i in (1, 2))
        manager.compute_batch_page_tables([1, 2], 0)
        expected = manager._batch_page_tables.clone()
        manager._batch_conversion_inputs = None
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        for stream in (first, second):
            stream.wait_stream(torch.cuda.current_stream())
        convert = kernel.convert_page_tables
        snapshots = []

        def take_snapshot(base, rows, params, out, *args, **kwargs):
            convert(base, rows, params, out, *args, **kwargs)
            if torch.cuda.current_stream() == first:
                snapshots.append(out.clone())

        monkeypatch.setattr(kernel, "convert_page_tables", take_snapshot)
        with torch.cuda.stream(first):
            manager.compute_batch_page_tables([1, 2], 0)
        first_read_done = manager._base_rows_read[manager._base_rows_slot]
        operations = Mock()
        with (
            torch.cuda.stream(second),
            patch.object(
                torch.cuda.Stream,
                "wait_event",
                autospec=True,
                side_effect=torch.cuda.Stream.wait_event,
            ) as wait_event,
            patch.object(
                cache_manager, "copy_host_to_device", wraps=cache_manager.copy_host_to_device
            ) as upload,
        ):
            operations.attach_mock(wait_event, "wait_event")
            operations.attach_mock(upload, "upload")
            manager.compute_batch_page_tables([2, 1], 0)
        handoff = [
            name
            for name, args, _ in operations.mock_calls
            if (name == "wait_event" and args == (second, first_read_done))
            or (name == "upload" and args[1] == "batch_rows")
        ]
        torch.cuda.current_stream().wait_stream(first)
        torch.cuda.current_stream().wait_stream(second)
        assert len(snapshots) == 1
        torch.testing.assert_close(snapshots[0], expected[..., : snapshots[0].shape[-1]])
        assert handoff == ["wait_event", "upload"]
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
def test_kept_growth_slack_is_returned_after_suspend_resume():
    manager = _manager(False, False, False)
    request = None
    try:
        manager.keep_growth_blocks()
        request = _admit(manager, 1, BLOCK - 16)
        cache = manager.kv_cache_map[1]
        assert cache.resize(BLOCK + 2)
        assert cache.resize(manager._kept_generation_update(1, cache.capacity, 4, 3))
        assert cache.capacity == BLOCK + 1 and manager._kept_growth_slack[1] == 3
        manager.suspend_request(request)
        assert not cache.is_active
        assert manager.try_allocate_generation(request)
        assert manager._kept_growth_slack[1] == 3
        assert manager._kept_generation_update(1, cache.capacity, 12, DRAFT_LEN) == (
            cache.capacity - 12 - 3
        )
        assert 1 not in manager._kept_growth_slack
    finally:
        if request is not None:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("attention_dp", [False, True], ids=["tep", "adp"])
@torch.inference_mode()
def test_row_bounded_page_tables_match_full_width(monkeypatch, attention_dp):
    """Page tables converted and copied per row's live columns, with one long request among short
    ones, read what full-width conversions and copies write in every eager and graph step."""
    rng = random.Random(0)
    narrowed = {"convert": 0, "copy": 0}
    convert, copy_tables = kernel.convert_page_tables, kernel.copy_tables_

    def narrower(kind, columns, width):
        # Some row stops well short of the launch width.
        if columns is not None and int(columns.min()) < width // 4:
            narrowed[kind] += 1

    def counted_convert(base, rows, params, out, *args, **kwargs):
        narrower("convert", kwargs.get("row_columns"), out.shape[-1])
        return convert(base, rows, params, out, *args, **kwargs)

    def counted_copy(descriptors, max_rows, max_columns, row_columns=None):
        narrower("copy", row_columns, max_columns)
        return copy_tables(descriptors, max_rows, max_columns, row_columns)

    monkeypatch.setattr(kernel, "convert_page_tables", counted_convert)
    monkeypatch.setattr(kernel, "copy_tables_", counted_copy)
    manager = _manager(attention_dp, False, False)
    bounded, full = _Executor(manager), _Executor(manager, force_full_prepare=True)
    long_len = 20 * BLOCK + 40
    active = [_admit(manager, 1, long_len)]
    pending, next_id = {}, 2
    try:
        for step in range(120):
            if step == 60:
                # The long request leaves; short ones take its rows.
                done = active.pop(0)
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            if step == 80 or (len(active) < 6 and rng.random() < 0.2):
                prompt = long_len if step == 80 else rng.randrange(BLOCK - 8, 3 * BLOCK)
                active.append(_admit(manager, next_id, prompt))
                next_id += 1
            elif len(active) > 2 and rng.random() < 0.1:
                done = active.pop(rng.randrange(1, len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            if len(active) > 1 and rng.random() < 0.1:
                rng.shuffle(active)
            batch, starts = _generation_batch(manager, active)
            use_graph = rng.random() < 0.75
            tables, observed = bounded.step(batch, starts, use_graph)
            with (
                _always_convert(manager),
                patch.object(CSA2TrtllmMetadata, "_table_copy_bounds", return_value=None),
            ):
                expected_tables, expected = full.step(batch, starts, use_graph)
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            _assert_identical(observed, expected, step)
            _accept_pending(manager, active, pending)
            pending.update((r.py_request_id, rng.randrange(DRAFT_LEN + 1)) for r in active)
        if is_blackwell_datacenter():
            assert narrowed["convert"] > 20 and narrowed["copy"] > 20, narrowed
    finally:
        for request in active:
            manager.free_resources(request)
        manager.shutdown()
