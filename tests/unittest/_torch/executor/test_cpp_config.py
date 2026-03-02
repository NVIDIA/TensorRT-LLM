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

"""Tests for Chunked Pipeline Parallelism configuration validation."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy, TorchLlmArgs


class TestContextChunkingPolicyEnum:

    def test_pipeline_aware_exists(self):
        assert hasattr(ContextChunkingPolicy, "PIPELINE_AWARE")
        assert ContextChunkingPolicy.PIPELINE_AWARE == "PIPELINE_AWARE"

    def test_pipeline_aware_pybind_returns_none(self):
        """PIPELINE_AWARE is Python-only and has no C++ counterpart."""
        policy = ContextChunkingPolicy.PIPELINE_AWARE
        assert policy._to_pybind() is None

    def test_existing_policies_unchanged(self):
        assert ContextChunkingPolicy.FIRST_COME_FIRST_SERVED._to_pybind() is not None
        assert ContextChunkingPolicy.EQUAL_PROGRESS._to_pybind() is not None


class TestChunkingPolicyInScheduler:

    def test_pipeline_aware_chunking_policy_enum(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
        )

        assert hasattr(ChunkingPolicy, "PIPELINE_AWARE")
        assert ChunkingPolicy.PIPELINE_AWARE.value == 3

    def test_context_chunking_config_pp_size_default(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
            ContextChunkingConfig,
        )

        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
        )
        assert config.pp_size == 1

    def test_context_chunking_config_pp_size_custom(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
            ContextChunkingConfig,
        )

        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=4,
        )
        assert config.pp_size == 4


class TestSimpleUnifiedSchedulerCppWiring:
    """Test that PIPELINE_AWARE policy is correctly wired through
    SimpleUnifiedScheduler's ctx_chunk_config tuple."""

    def test_pipeline_aware_policy_from_tuple(self):
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
            ContextChunkingConfig,
            PyMicroBatchScheduler,
        )

        config = ContextChunkingConfig(
            chunking_policy=ChunkingPolicy.PIPELINE_AWARE,
            chunk_unit_size=64,
            pp_size=2,
        )
        scheduler = PyMicroBatchScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            ctx_chunk_config=config,
        )
        assert scheduler.ctx_chunk_config.chunking_policy == ChunkingPolicy.PIPELINE_AWARE
        assert scheduler.ctx_chunk_config.pp_size == 2

    def test_string_policy_matching(self):
        """Verify that SimpleUnifiedScheduler correctly matches
        PIPELINE_AWARE from string representation."""
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
            SimpleUnifiedScheduler,
        )

        ctx_chunk_config_tuple = (
            ContextChunkingPolicy.PIPELINE_AWARE, 64, 4)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            kv_cache_manager=None,
            peft_cache_manager=None,
            scheduler_policy="GUARANTEED_NO_EVICT",
            ctx_chunk_config=ctx_chunk_config_tuple,
        )
        micro_sched = scheduler.micro_batch_scheduler
        assert micro_sched.ctx_chunk_config.chunking_policy == ChunkingPolicy.PIPELINE_AWARE
        assert micro_sched.ctx_chunk_config.pp_size == 4

    def test_backward_compat_2tuple(self):
        """Existing 2-tuple (policy, chunk_size) should still work
        with pp_size defaulting to 1."""
        from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (
            ChunkingPolicy,
            SimpleUnifiedScheduler,
        )

        ctx_chunk_config_tuple = (
            ContextChunkingPolicy.FIRST_COME_FIRST_SERVED, 64)
        scheduler = SimpleUnifiedScheduler(
            max_batch_size=8,
            max_num_tokens=2048,
            kv_cache_manager=None,
            peft_cache_manager=None,
            scheduler_policy="GUARANTEED_NO_EVICT",
            ctx_chunk_config=ctx_chunk_config_tuple,
        )
        micro_sched = scheduler.micro_batch_scheduler
        assert micro_sched.ctx_chunk_config.chunking_policy == ChunkingPolicy.FIRST_COME_FIRST_SERVED
        assert micro_sched.ctx_chunk_config.pp_size == 1


class TestCppLlmArgsValidation:
    """Test that CPP fields on LlmArgs are validated correctly."""

    def test_cpp_requires_pp_size_gt_1(self):
        with pytest.raises(ValidationError, match="pipeline_parallel_size > 1"):
            TorchLlmArgs(
                model="/tmp/dummy_model",
                enable_chunked_pipeline_parallelism=True,
                pipeline_parallel_size=1,
            )

    def test_cpp_num_chunks_must_be_gte_pp_size(self):
        with pytest.raises(ValidationError, match="cpp_num_chunks.*must be >="):
            TorchLlmArgs(
                model="/tmp/dummy_model",
                enable_chunked_pipeline_parallelism=True,
                pipeline_parallel_size=4,
                cpp_num_chunks=2,
            )

    def test_cpp_auto_enables_chunked_prefill(self):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            enable_chunked_pipeline_parallelism=True,
            pipeline_parallel_size=2,
        )
        assert args.enable_chunked_prefill is True

    def test_cpp_valid_config(self):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            enable_chunked_pipeline_parallelism=True,
            pipeline_parallel_size=4,
            cpp_num_chunks=8,
        )
        assert args.enable_chunked_pipeline_parallelism is True
        assert args.cpp_num_chunks == 8
        assert args.enable_chunked_prefill is True

    def test_cpp_disabled_no_validation(self):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            enable_chunked_pipeline_parallelism=False,
            pipeline_parallel_size=1,
        )
        assert args.enable_chunked_pipeline_parallelism is False

    def test_cpp_num_chunks_none_defaults(self):
        args = TorchLlmArgs(
            model="/tmp/dummy_model",
            enable_chunked_pipeline_parallelism=True,
            pipeline_parallel_size=2,
        )
        assert args.cpp_num_chunks is None
