# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 steady decode: reused page tables and steady metadata match full preparation."""

import random
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID,
    LlmRequest,
    LlmRequestState,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

DRAFT_LEN = 5
BLOCK = 128
MAX_BATCH = 8

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)


def _manager(attention_dp, disaggregated, scratch, block=BLOCK):
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


def _prepare(metadata, batch, starts):
    metadata.request_ids = [request.py_request_id for request in batch]
    metadata.num_contexts = 0
    metadata.seq_lens = torch.full((len(batch),), DRAFT_LEN + 1, dtype=torch.int32)
    metadata.prompt_lens = [request.py_prompt_len for request in batch]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    metadata.prepare()


def _forward_inputs(metadata, count):
    """Every page table and slot a decode forward reads."""
    metadata._csa2_swa_resolved = False
    metadata._ensure_swa_slots()
    outputs = {
        "block_offsets": metadata.kv_cache_block_offsets[:, :count],
        "positions": metadata.csa2_positions,
    }
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
        convert(request_ids, num_contexts, swa_out=swa_out, **kwargs)
        manager._batch_conversion_inputs = previous
        np.maximum(manager._table_live, live[0], out=manager._table_live)
        np.maximum(manager._swa_table_live, live[1], out=manager._swa_table_live)

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
        self.graph.replay()
        observed = _clone(self.outputs)
        # The replayed consumer reads the buffers captured at the first step.
        assert self.pointers == self._pointers()
        return observed


class _Executor:
    """One eager metadata and graph metadata per batch size, as an executor holds them."""

    def __init__(self, manager):
        self.manager = manager
        self.eager = _metadata(manager)
        self.graphs = {}

    def step(self, batch, starts, use_graph):
        """Prepare one decode step; return the batch tables and every forward input."""
        count = len(batch)
        if use_graph:
            graph = self.graphs.get(count)
            if graph is None:
                graph = self.graphs[count] = _Graph(self.eager, count)
            _prepare(graph.metadata, batch, starts)
            return _batch_tables(self.manager), graph.run()
        _prepare(self.eager, batch, starts)
        return _batch_tables(self.manager), _clone(_forward_inputs(self.eager, count))


def _assert_identical(observed, expected, step):
    assert observed.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(observed[name], tensor), f"step {step}: {name} differs"


def _counted_conversions(monkeypatch):
    conversions = []
    convert = kernel.convert_page_tables
    monkeypatch.setattr(
        kernel,
        "convert_page_tables",
        lambda *args, **kwargs: conversions.append(1) or convert(*args, **kwargs),
    )
    return conversions


@skip_pre_blackwell
@pytest.mark.parametrize("scratch", [False, True], ids=["no_scratch", "scratch"])
@pytest.mark.parametrize(
    "attention_dp,disaggregated",
    [(False, False), (False, True), (True, True)],
    ids=["tep", "tep_disagg", "adp_disagg"],
)
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("block", [BLOCK, 2 * BLOCK])
@torch.inference_mode()
def test_steady_decode_reuses_converted_tables(
    monkeypatch, attention_dp, disaggregated, scratch, seed, block
):
    rng = random.Random(seed)
    conversions = _counted_conversions(monkeypatch)
    manager = _manager(attention_dp, disaggregated, scratch, block)
    reusing, converting = _Executor(manager), _Executor(manager)
    active, pending, padding = [], {}, None
    next_id, previous_ids = 1, None
    stats = dict(steps=0, skipped=0, crossings=0)

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
            tables, observed = reusing.step(batch, starts, use_graph)
            converted = len(conversions) > before
            with _always_convert(manager):
                expected_tables, expected = converting.step(batch, starts, use_graph)
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            assert np.array_equal(tables[1], expected_tables[1])
            assert (tables[2] is None) == (expected_tables[2] is None)
            if tables[2] is not None:
                assert np.array_equal(tables[2], expected_tables[2])
            _assert_identical(observed, expected, step)
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
            # Switching between graph and eager destinations converts again.
            assert stats["steps"] // 8 < stats["skipped"] < stats["steps"], stats
    finally:
        for request in [*active, *([padding] if padding else [])]:
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.parametrize("scratch", [False, True], ids=["no_scratch", "scratch"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("block", [BLOCK, 2 * BLOCK])
@torch.inference_mode()
def test_steady_spec_steps_match_full_prepare(monkeypatch, scratch, seed, block):
    """Steady DSpark verification steps read exactly what a full preparation writes."""
    rng = random.Random(seed)
    hits = []
    predicate = CSA2TrtllmMetadata._steady_generation_step

    def counted(self, snapshot):
        hit = predicate(self, snapshot)
        if not getattr(self, "_full_reference", False):
            hits.append(hit is not None)
        return hit

    monkeypatch.setattr(CSA2TrtllmMetadata, "_steady_generation_step", counted)
    manager = _manager(False, False, scratch, block)
    steady, full = _Executor(manager), _Executor(manager)
    active, pending = [], {}
    next_id = 1

    def accept(requests):
        requests = [r for r in requests if r.py_request_id in pending]
        if requests:
            _accept(manager, requests, [pending.pop(r.py_request_id) for r in requests])

    try:
        for step in range(200):
            if len(active) < 3 and (not active or rng.random() < 0.03):
                active.append(_admit(manager, next_id, rng.randrange(block - 8, 5 * block + 60)))
                next_id += 1
            elif len(active) > 1 and rng.random() < 0.02:
                done = active.pop(rng.randrange(len(active)))
                pending.pop(done.py_request_id, None)
                manager.free_resources(done)
            if len(active) > 1 and rng.random() < 0.03:
                rng.shuffle(active)
            for request in active:
                assert manager.try_allocate_generation(request)
            batch = list(active)
            starts = [r.max_beam_num_tokens - 1 + DRAFT_LEN + 1 for r in batch]
            use_graph = rng.random() < 0.75
            try:
                tables, observed = steady.step(batch, starts, use_graph)
            except AssertionError as error:
                raise AssertionError(f"steady executor, step {step}, graph {use_graph}") from error
            for metadata in (full.eager, *(graph.metadata for graph in full.graphs.values())):
                metadata._csa2_steady_snapshot = None
                metadata._full_reference = True
            if use_graph and len(batch) not in full.graphs:
                full.graphs[len(batch)] = _Graph(full.eager, len(batch))
                full.graphs[len(batch)].metadata._full_reference = True
            try:
                expected_tables, expected = full.step(batch, starts, use_graph)
            except AssertionError as error:
                raise AssertionError(f"full executor, step {step}, graph {use_graph}") from error
            assert torch.equal(tables[0], expected_tables[0]), f"step {step}: batch tables differ"
            _assert_identical(observed, expected, step)
            accept(active)
            # A step may also commit nothing.
            pending.update(
                (r.py_request_id, rng.randrange(DRAFT_LEN + 1))
                for r in active
                if rng.random() < 0.97
            )
        assert sum(hits) > len(hits) // 2, (sum(hits), len(hits))
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
