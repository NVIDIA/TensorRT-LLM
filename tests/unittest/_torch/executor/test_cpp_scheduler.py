# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PIPELINE_AWARE chunking policy used in Chunked Pipeline Parallelism."""

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
    ChunkingPolicy,
    ContextChunkingConfig,
    PyMicroBatchScheduler,
)


def _make_context_request(request_id: int, num_tokens: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=5,
        input_tokens=list(range(num_tokens)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


class TestPipelineAwareChunkingPolicy:
    """Verify that PIPELINE_AWARE splits context into pp_size-aligned chunks."""

    def test_single_request_pp2_even_split(self):
        """A 512-token request with pp_size=2 should be split into ~256-token chunks."""
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 512)
        inflight = set()

        ctx_reqs, gen_reqs = scheduler.schedule([req], inflight)
        assert len(gen_reqs) == 0
        assert len(ctx_reqs) == 1
        chunk_size = ctx_reqs[0].context_chunk_size
        assert chunk_size == 256
        assert chunk_size % unit_size == 0

    def test_single_request_pp4(self):
        """A 1024-token request with pp_size=4 should yield ~256-token chunks."""
        pp_size = 4
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 1024)
        inflight = set()

        ctx_reqs, gen_reqs = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 1
        chunk_size = ctx_reqs[0].context_chunk_size
        assert chunk_size == 256
        assert chunk_size % unit_size == 0

    def test_multi_chunk_iteration_consumes_full_context(self):
        """Repeatedly scheduling a request should consume the entire context."""
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        total_tokens = 512
        req = _make_context_request(1, total_tokens)
        inflight = set()

        consumed = 0
        chunks = []
        while req.context_remaining_length > 0:
            ctx_reqs, _ = scheduler.schedule([req], inflight)
            assert len(ctx_reqs) == 1
            chunk_size = ctx_reqs[0].context_chunk_size
            assert chunk_size > 0
            chunks.append(chunk_size)
            consumed += chunk_size
            req.move_to_next_context_chunk()

        assert consumed == total_tokens
        assert len(chunks) == pp_size

    def test_non_final_chunks_not_last(self):
        """Non-final chunks should report is_last_context_chunk=False."""
        pp_size = 4
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 1024)
        inflight = set()
        chunk_count = 0

        while req.context_remaining_length > 0:
            ctx_reqs, _ = scheduler.schedule([req], inflight)
            assert len(ctx_reqs) == 1
            chunk_size = ctx_reqs[0].context_chunk_size
            chunk_count += 1
            if req.context_remaining_length > chunk_size:
                assert not ctx_reqs[0].is_last_context_chunk
            else:
                assert ctx_reqs[0].is_last_context_chunk
            req.move_to_next_context_chunk()

        assert chunk_count == pp_size

    def test_chunk_alignment_to_unit_size(self):
        """All non-final chunks must be aligned to chunk_unit_size."""
        pp_size = 3
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 500)
        inflight = set()
        chunks = []

        while req.context_remaining_length > 0:
            ctx_reqs, _ = scheduler.schedule([req], inflight)
            chunk_size = ctx_reqs[0].context_chunk_size
            chunks.append(chunk_size)
            if not ctx_reqs[0].is_last_context_chunk:
                assert chunk_size % unit_size == 0, (
                    f"Non-final chunk size {chunk_size} not aligned to {unit_size}"
                )
            req.move_to_next_context_chunk()

        assert sum(chunks) == 500

    def test_respects_max_num_tokens_capacity(self):
        """Chunk size should not exceed max_num_tokens budget."""
        pp_size = 2
        unit_size = 64
        max_tokens = 128
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=max_tokens,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 512)
        inflight = set()

        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 1
        assert ctx_reqs[0].context_chunk_size <= max_tokens

    def test_small_context_single_chunk(self):
        """A context smaller than target chunk size should be scheduled in one chunk."""
        pp_size = 4
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 100)
        inflight = set()

        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 1
        assert ctx_reqs[0].context_chunk_size == 100
        assert ctx_reqs[0].is_last_context_chunk

    def test_multiple_requests_pipeline_aware(self):
        """Multiple context requests should each get pipeline-aware chunks."""
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )

        req1 = _make_context_request(1, 512)
        req2 = _make_context_request(2, 256)
        inflight = set()

        ctx_reqs, _ = scheduler.schedule([req1, req2], inflight)
        assert len(ctx_reqs) == 2

        assert ctx_reqs[0].context_chunk_size == 256
        assert ctx_reqs[1].context_chunk_size == 128

    def test_inflight_requests_skipped(self):
        """Requests in the inflight set should be skipped by the scheduler."""
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 512)
        inflight = {1}

        ctx_reqs, gen_reqs = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 0
        assert len(gen_reqs) == 0


class TestCppNumChunks:
    """Verify that cpp_num_chunks overrides pp_size for chunk count."""

    def test_more_chunks_than_pp_stages(self):
        """With pp_size=2 and cpp_num_chunks=4, a 1024-token request should
        produce ~256-token chunks (1024/4) rather than ~512 (1024/2)."""
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=2,
            cpp_num_chunks=4,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )
        req = _make_context_request(1, 1024)
        ctx_reqs, _ = scheduler.schedule([req], set())
        assert ctx_reqs[0].context_chunk_size == 256

    def test_cpp_num_chunks_full_consumption(self):
        """Repeatedly scheduling should produce exactly cpp_num_chunks chunks."""
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=2,
            cpp_num_chunks=8,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )
        total_tokens = 1024
        req = _make_context_request(1, total_tokens)
        chunks = []
        while req.context_remaining_length > 0:
            ctx_reqs, _ = scheduler.schedule([req], set())
            assert len(ctx_reqs) == 1
            chunks.append(ctx_reqs[0].context_chunk_size)
            req.move_to_next_context_chunk()

        assert sum(chunks) == total_tokens
        assert len(chunks) == 8

    def test_cpp_num_chunks_none_falls_back_to_pp_size(self):
        """When cpp_num_chunks is None, behaviour matches pp_size."""
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=4,
            cpp_num_chunks=None,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )
        req = _make_context_request(1, 1024)
        ctx_reqs, _ = scheduler.schedule([req], set())
        assert ctx_reqs[0].context_chunk_size == 256

    def test_cpp_num_chunks_capped_by_sequence_length(self):
        """cpp_num_chunks should be capped when the sequence is too short."""
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=2,
            cpp_num_chunks=16,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=4096,
            ctx_chunk_config=config,
        )
        req = _make_context_request(1, 200)
        chunks = []
        while req.context_remaining_length > 0:
            ctx_reqs, _ = scheduler.schedule([req], set())
            chunks.append(ctx_reqs[0].context_chunk_size)
            req.move_to_next_context_chunk()

        assert sum(chunks) == 200
        assert len(chunks) <= 200 // 64 + 1


class TestPipelineAwareChunkingSimulation:
    """Simulate CPP scheduling across multiple iterations, verifying that
    chunks of the same request can be pipelined through PP stages."""

    def test_full_pipeline_fill_simulation(self):
        """Simulate a 2-stage pipeline processing a request in 2 chunks.

        Iteration 0: scheduler picks chunk 0, stage 0 processes it
        Iteration 1: scheduler picks chunk 1 (chunk 0 not in inflight since non-final),
                     stage 0 processes chunk 1, stage 1 processes chunk 0
        This demonstrates pipeline overlap.
        """
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 512)
        inflight = set()
        pipeline_trace = []

        # Iteration 0: chunk 0
        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 1
        chunk0_size = ctx_reqs[0].context_chunk_size
        assert chunk0_size == 256
        assert not ctx_reqs[0].is_last_context_chunk
        pipeline_trace.append(("iter0", "chunk0", chunk0_size))
        req.move_to_next_context_chunk()

        # Iteration 1: chunk 1 (non-final chunk 0 NOT added to inflight)
        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 1
        chunk1_size = ctx_reqs[0].context_chunk_size
        assert chunk1_size == 256
        assert ctx_reqs[0].is_last_context_chunk
        pipeline_trace.append(("iter1", "chunk1", chunk1_size))
        req.move_to_next_context_chunk()

        assert req.context_remaining_length == 0
        assert len(pipeline_trace) == pp_size
        assert sum(t[2] for t in pipeline_trace) == 512

    def test_final_chunk_added_to_inflight(self):
        """Only the final chunk's request should be added to the inflight set,
        mimicking the executor's _add_inflight_ids behavior."""
        pp_size = 2
        unit_size = 64
        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=unit_size,
            pp_size=pp_size,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )

        req = _make_context_request(1, 512)
        inflight = set()

        # Chunk 0 - non-final, should NOT be added to inflight
        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert not ctx_reqs[0].is_last_context_chunk
        req.move_to_next_context_chunk()

        # Chunk 1 - final, should be added to inflight
        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert ctx_reqs[0].is_last_context_chunk
        inflight.add(req.request_id)
        req.move_to_next_context_chunk()

        # After adding to inflight, scheduler should skip this request
        ctx_reqs, _ = scheduler.schedule([req], inflight)
        assert len(ctx_reqs) == 0
