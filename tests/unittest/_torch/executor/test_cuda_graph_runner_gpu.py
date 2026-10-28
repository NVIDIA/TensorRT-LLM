# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock

import torch
from _torch.executor.engine_test_utils import DummyModel, _create_request
from _torch.executor.engine_test_utils import kv_cache_manager as managed_kv_cache

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2


@contextlib.contextmanager
def cuda_graph_runner_and_kv_cache() -> Iterator[tuple[CUDAGraphRunner, KVCacheManagerV2]]:
    runner = CUDAGraphRunner(
        CUDAGraphRunnerConfig(
            use_cuda_graph=True,
            cuda_graph_padding_enabled=True,
            cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
            max_cuda_graph_batch_size=128,
            max_beam_width=1,
            max_num_tokens=258,
            spec_config=None,
            cuda_graph_mem_pool=None,
            use_mrope=False,
            original_max_total_draft_tokens=0,
            enable_attention_dp=False,
            is_encoder_decoder=False,
            batch_size=13,
            mapping=Mapping(),
            dist=None,
            kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        )
    )
    with managed_kv_cache(model=DummyModel(torch.half)) as manager:
        try:
            yield runner, manager
        finally:
            runner.clear()


class CudaGraphPaddingTestCase(unittest.TestCase):
    def test_pad_generation_requests(self) -> None:
        (model_engine, kv_cache_manager) = self.enterContext(cuda_graph_runner_and_kv_cache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        def free_blocks() -> int:
            return kv_cache_manager.get_kv_cache_stats().free_num_blocks

        batch_sizes_and_seqlens = [
            (5, 1),
            (13, 1),
            (5, 25),
        ]
        for batch_size, max_seq_len in batch_sizes_and_seqlens:
            requests = [_create_request(max_seq_len, i) for i in range(batch_size)]
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = requests

            pages_before = free_blocks()
            with model_engine.pad_batch(batch, resource_manager) as padded_batch:
                # No padding for prefill
                self.assertIs(batch, padded_batch)
            self.assertEqual(free_blocks(), pages_before)

            batch = ScheduledRequests()
            batch.generation_requests = requests
            pages_before = free_blocks()
            new_dummy_block = 1 if not model_engine.padding_dummy_requests else 0
            with model_engine.pad_batch(batch, resource_manager) as padded_batch:
                if batch_size < 8 and max_seq_len < 25:
                    self.assertEqual(len(padded_batch.generation_requests) % 8, 0)
                else:
                    # No padding if it would create too many concurrent requests.
                    # This requirement is not strictly required, but we should probably
                    # respect the requirement?
                    # The seqlen check makes sure we don't exceed the KV cache memory
                    # budget.
                    self.assertIs(batch, padded_batch)
            self.assertEqual(free_blocks() + new_dummy_block, pages_before)

    def test_pad_batch_strips_cudagraph_dummies_on_clean_exit(self) -> None:
        # Regression guard for the invariant that CUDAGraphRunner.pad_batch's
        # `finally` strips every is_cuda_graph_dummy=True entry from
        # scheduled_requests.generation_requests before the `with` block
        # exits. Downstream consumers of scheduled_batch.generation_requests
        # — including the per-iteration stats populate block in
        # PyExecutor._update_iter_stats — rely on never observing
        # cudagraph dummies.
        (model_engine, kv_cache_manager) = self.enterContext(cuda_graph_runner_and_kv_cache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # batch_size=5 rounds up to 8 (nearest captured graph size in the
        # fixture config) -> padding_size=3, deterministically.
        real_batch_size = 5
        max_seq_len = 1
        real_requests = [_create_request(max_seq_len, i) for i in range(real_batch_size)]
        real_ids = [id(r) for r in real_requests]

        batch = ScheduledRequests()
        batch.generation_requests = list(real_requests)

        with model_engine.pad_batch(batch, resource_manager) as padded_batch:
            # Positive assertion that padding actually fired — guards
            # against a vacuous pass where padding was a no-op.
            self.assertGreater(
                len(padded_batch.generation_requests),
                real_batch_size,
                "padding did not fire; fixture config may have drifted "
                "so that 5 no longer rounds up to 8",
            )
            # Every appended entry past the original count is a
            # cudagraph-flagged dummy.
            for req in padded_batch.generation_requests[real_batch_size:]:
                self.assertTrue(
                    getattr(req, "is_cuda_graph_dummy", False),
                    "pad_batch appended a request without is_cuda_graph_dummy=True",
                )
            # Real requests' identities and order are untouched.
            self.assertEqual(
                [id(r) for r in padded_batch.generation_requests[:real_batch_size]], real_ids
            )

        # After the with-block: finally must have sliced off the padding.
        self.assertEqual(
            len(batch.generation_requests),
            real_batch_size,
            "pad_batch.finally did not strip cudagraph dummies — "
            "downstream consumers of scheduled_batch.generation_requests "
            "would observe the leaked dummies",
        )
        for req in batch.generation_requests:
            self.assertFalse(
                getattr(req, "is_cuda_graph_dummy", False),
                "cudagraph dummy leaked out of pad_batch's finally",
            )

    def test_pad_batch_strips_cudagraph_dummies_on_exception(self) -> None:
        # The strip must fire even when the body raises. This is the
        # critical property of `finally` vs. a plain trailing statement —
        # it guards the invariant on the error path. A refactor that
        # accidentally dropped the `finally` would be caught here but not
        # by the clean-exit variant.
        (model_engine, kv_cache_manager) = self.enterContext(cuda_graph_runner_and_kv_cache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        real_batch_size = 5
        real_requests = [_create_request(1, i) for i in range(real_batch_size)]

        batch = ScheduledRequests()
        batch.generation_requests = list(real_requests)

        class _ForwardBoom(Exception):
            pass

        with self.assertRaises(_ForwardBoom):
            with model_engine.pad_batch(batch, resource_manager) as padded_batch:
                self.assertGreater(len(padded_batch.generation_requests), real_batch_size)
                raise _ForwardBoom()

        self.assertEqual(len(batch.generation_requests), real_batch_size)
        for req in batch.generation_requests:
            self.assertFalse(getattr(req, "is_cuda_graph_dummy", False))

    def test_preallocate_padding_dummies_uses_captured_draft_lens(self) -> None:
        # Speculative engines pad batches with the runtime draft length, so
        # the preallocation must create dummies for the draft lengths of the
        # captured graphs — not draft_len 0 unconditionally (an extra
        # draft_len-0 dummy would permanently hold KV blocks that runtime
        # padding never uses).
        (model_engine, kv_cache_manager) = self.enterContext(cuda_graph_runner_and_kv_cache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        runner = model_engine
        # Simulate a spec-decode capture: graphs keyed (batch_size, draft_len,
        # short_seq_len_mode, is_all_greedy_sample) exist only for draft_len 2.
        runner.graphs[(8, 2, False, True)] = Mock()
        try:
            runner.preallocate_padding_dummies(resource_manager)

            self.assertEqual([2], sorted(runner.padding_dummy_requests.keys()))
        finally:
            runner.graphs.clear()
            for dummy in runner.padding_dummy_requests.values():
                kv_cache_manager.free_resources(dummy)

    def test_release_padding_dummy_covers_every_manager(self) -> None:
        # A padding dummy's request ID is spread across several managers, so
        # releasing only the main KV cache manager leaves the others holding
        # it — and re-creation reuses the same ID.
        (model_engine, kv_cache_manager) = self.enterContext(cuda_graph_runner_and_kv_cache())
        spec_manager = Mock()
        cross_manager = Mock()
        resource_manager = ResourceManager(
            {
                ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
                ResourceManagerType.SPEC_RESOURCE_MANAGER: spec_manager,
                ResourceManagerType.CROSS_KV_CACHE_MANAGER: cross_manager,
            }
        )

        runner = model_engine
        try:
            self.assertIsNotNone(runner._get_or_create_padding_dummy(resource_manager, 0))
            dummy = runner.padding_dummy_requests[0]

            self.assertTrue(runner.release_padding_dummy(resource_manager, 0))

            # Dropped from the runner so the lazy path re-creates it...
            self.assertEqual({}, runner.padding_dummy_requests)
            # ...and the spec resource manager slot is released too, not just
            # the main KV cache manager.
            spec_manager.free_resources.assert_called_once_with(dummy)
            # The cross-KV manager is only involved for encoder-decoder, which
            # this engine is not.
            self.assertFalse(runner.is_encoder_decoder)
            cross_manager.free_resources.assert_not_called()

            # Releasing again is a no-op rather than a double free.
            self.assertFalse(runner.release_padding_dummy(resource_manager, 0))
            spec_manager.free_resources.assert_called_once()
        finally:
            for dummy in runner.padding_dummy_requests.values():
                kv_cache_manager.free_resources(dummy)
            runner.padding_dummy_requests.clear()


class ExecutionStreamTestCase(unittest.TestCase):
    def test_cuda_graph_replay_observes_execution_stream_dependency(self) -> None:
        """A graph replay on the KV manager stream waits for restored KV data."""
        execution_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        kv_cache_manager = self.enterContext(
            managed_kv_cache(model=DummyModel(torch.half), execution_stream=execution_stream)
        )

        source = torch.zeros(1, dtype=torch.int32, device="cuda")
        observed = torch.zeros_like(source)
        graph = torch.cuda.CUDAGraph()
        self.addCleanup(graph.reset)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=execution_stream):
            observed.copy_(source)

        ready = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            # Model the async host-to-device restore completed by the local
            # offload manager. Recording the event after the write is the same
            # dependency shape that refreshBlocks/resume installs.
            source.fill_(7)
            ready.record()

        manager_stream = torch.cuda.ExternalStream(kv_cache_manager._stream.cuda_stream)
        with torch.cuda.stream(manager_stream):
            manager_stream.wait_event(ready)
            graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(observed.item(), 7)
