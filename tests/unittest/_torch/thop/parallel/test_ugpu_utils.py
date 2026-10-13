# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Unit tests for UGPU utilities in tensorrt_llm._torch.ugpu_utils.

Tests cover:
- UGPU support detection
- Resource initialization
- Stream and mempool retrieval
- Error handling and edge cases
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import tensorrt_llm._torch.ugpu.runtime as ugpu_runtime
import tensorrt_llm._torch.ugpu_utils as ugpu_utils
from tensorrt_llm._torch.autotuner import OptimizationProfile, TunableRunner, TuningConfig
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    _copy_to_new_cuda_allocation,
)
from tensorrt_llm._torch.ugpu.autotune import UgpuConcurrentTunableRunner, tune_ugpu_concurrent
from tensorrt_llm._torch.ugpu.runtime import UgpuRuntime
from tensorrt_llm._torch.ugpu_utils import (
    get_reserved_remainder_stream,
    get_ugpu_compute_sm_counts,
    get_ugpu_mempool,
    get_ugpu_stream,
    initialize_ugpu_resources,
    is_ugpu_enabled,
    is_ugpu_supported,
    node_local_max_active_clusters,
)
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams


@pytest.fixture(scope="module")
def check_ugpu_support():
    """Check if UGPU is supported and skip tests if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if not is_ugpu_supported():
        pytest.skip("UGPU localization is not supported on this system")


class TestUgpuSupport:
    """Tests for UGPU support detection."""

    def test_is_ugpu_supported_returns_bool(self):
        """Test that is_ugpu_supported returns a boolean value."""
        result = is_ugpu_supported()
        assert isinstance(result, bool)

    def test_is_ugpu_enabled_requires_rubin(self):
        is_ugpu_enabled.cache_clear()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("tensorrt_llm._torch.ugpu_utils.get_sm_version", return_value=100),
            patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_supported", return_value=True),
        ):
            assert not is_ugpu_enabled()
        is_ugpu_enabled.cache_clear()

    def test_is_ugpu_enabled_allows_rubin_when_supported(self):
        is_ugpu_enabled.cache_clear()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("tensorrt_llm._torch.ugpu_utils.get_sm_version", return_value=107),
            patch("tensorrt_llm._torch.ugpu_utils.is_ugpu_supported", return_value=True),
        ):
            assert is_ugpu_enabled()
        is_ugpu_enabled.cache_clear()


class TestUgpuComputeTopology:
    """Pure mocked tests for compute topology and grid sizing."""

    @pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
    def test_cluster_occupancy_cache_keeps_topology_scaling_dynamic(self, monkeypatch):
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

        hardware_queries = []
        current_ugpu = None

        class FakeHardwareInfo:
            def __init__(self, device_id):
                self.device_id = device_id

            def get_max_active_clusters(self, cluster_size):
                hardware_queries.append((self.device_id, cluster_size))
                return 106

        occupancy_cache = cute_dsl_custom_ops._get_full_device_max_active_clusters
        occupancy_cache.cache_clear()
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(cute_dsl_custom_ops.cutlass.utils, "HardwareInfo", FakeHardwareInfo)
        monkeypatch.setattr(cute_dsl_custom_ops, "get_current_ugpu", lambda: current_ugpu)
        monkeypatch.setattr(
            cute_dsl_custom_ops,
            "node_local_max_active_clusters",
            lambda full_device_limit: full_device_limit * 100 // 212,
        )

        try:
            assert cute_dsl_custom_ops.get_max_activate_clusters(2) == 106
            assert cute_dsl_custom_ops.get_max_activate_clusters(2) == 106
            current_ugpu = 0
            assert cute_dsl_custom_ops.get_max_activate_clusters(2) == 50
            assert hardware_queries == [(3, 2)]
        finally:
            occupancy_cache.cache_clear()

    @pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="cutlass-dsl is not available")
    def test_cluster_occupancy_cache_is_scoped_by_device_and_cluster(self, monkeypatch):
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

        hardware_queries = []
        current_device = 0

        class FakeHardwareInfo:
            def __init__(self, device_id):
                self.device_id = device_id

            def get_max_active_clusters(self, cluster_size):
                hardware_queries.append((self.device_id, cluster_size))
                return 100 + self.device_id + cluster_size

        occupancy_cache = cute_dsl_custom_ops._get_full_device_max_active_clusters
        occupancy_cache.cache_clear()
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
        monkeypatch.setattr(cute_dsl_custom_ops.cutlass.utils, "HardwareInfo", FakeHardwareInfo)
        monkeypatch.setattr(cute_dsl_custom_ops, "get_current_ugpu", lambda: None)

        try:
            assert cute_dsl_custom_ops.get_max_activate_clusters(1) == 101
            assert cute_dsl_custom_ops.get_max_activate_clusters(1) == 101
            assert cute_dsl_custom_ops.get_max_activate_clusters(2) == 102
            current_device = 1
            assert cute_dsl_custom_ops.get_max_activate_clusters(1) == 102
            assert hardware_queries == [(0, 1), (0, 2), (1, 1)]
        finally:
            occupancy_cache.cache_clear()

    def test_compute_sm_counts_are_read_from_current_device_cache(self, monkeypatch):
        class Manager:
            compute_sm_counts = {(3, 0): (100, 212)}

            @staticmethod
            def is_initialized(device_id):
                return device_id == 3

        monkeypatch.setattr(ugpu_utils, "get_ugpu_resource_manager", Manager)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(
            ugpu_utils,
            "initialize_ugpu_resources",
            lambda: pytest.fail("cached topology unexpectedly reinitialized resources"),
        )

        assert get_ugpu_compute_sm_counts(0) == (100, 212)

    @pytest.mark.parametrize(
        ("ugpu_id", "sm_counts", "max_active_full_device", "expected"),
        [
            pytest.param(0, (100, 212), 106, 50, id="strict-half-limit"),
            pytest.param(0, (106, 212), 106, 53, id="balanced-half-limit"),
            pytest.param(0, (100, 212), 212, 100, id="full-limit"),
            pytest.param(None, (100, 212), 106, None, id="outside-partition"),
            pytest.param(0, None, 106, None, id="missing-topology"),
            pytest.param(0, (0, 212), 106, None, id="empty-partition"),
            pytest.param(0, (213, 212), 106, None, id="oversized-partition"),
            pytest.param(0, (100, 212), 0, None, id="invalid-full-device-limit"),
        ],
    )
    def test_node_local_max_active_clusters(
        self, monkeypatch, ugpu_id, sm_counts, max_active_full_device, expected
    ):
        monkeypatch.setattr(ugpu_utils, "get_current_ugpu", lambda: ugpu_id)
        monkeypatch.setattr(
            ugpu_utils,
            "get_ugpu_compute_sm_counts",
            lambda queried_id: sm_counts if queried_id == ugpu_id else None,
        )

        assert node_local_max_active_clusters(max_active_full_device) == expected

    def test_runtime_topology_identity(self, monkeypatch):
        sm_counts = {0: (100, 212), 1: (106, 212)}
        monkeypatch.setattr(ugpu_runtime, "initialize_ugpu_resources", lambda: None)
        monkeypatch.setattr(
            ugpu_runtime,
            "get_ugpu_compute_sm_counts",
            lambda partition_id: sm_counts[partition_id],
        )

        identity = UgpuRuntime().topology_identity()
        expected = ((100, 212), (106, 212))
        assert identity == expected
        assert hash(identity) == hash(expected)

    @pytest.mark.parametrize("ugpu_id", [-1, 2])
    def test_compute_sm_counts_reject_invalid_partition(self, ugpu_id):
        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_compute_sm_counts(ugpu_id)


class TestUgpuConcurrentTunableRunner:
    class FakeRunner(TunableRunner):
        def unique_id(self):
            return ("fake",)

        def get_valid_tactics(self, inputs, profile, **kwargs):
            return [("candidate",)]

        def forward(self, inputs, tactic=-1, **kwargs):
            raise AssertionError("the concurrent wrapper must use its launch callback")

    class FakeRuntime:
        def __init__(self, num_partitions=2):
            self.num_partitions = num_partitions
            self.events = []

        def topology_identity(self):
            return tuple((100, 212) for _ in range(self.num_partitions))

        def fork(self):
            self.events.append("fork")

        @contextmanager
        def partition_context(self, partition_id):
            self.events.append(("enter", partition_id))
            yield
            self.events.append(("exit", partition_id))

        def join(self):
            self.events.append("join")

    def test_rejects_invalid_or_mismatched_partition_count(self):
        runtime = self.FakeRuntime()

        def launch(partition_id, inputs, tactic):
            return None

        with pytest.raises(ValueError, match="must be positive"):
            UgpuConcurrentTunableRunner(self.FakeRunner(), runtime, 0, launch)
        with pytest.raises(ValueError, match="does not match"):
            UgpuConcurrentTunableRunner(self.FakeRunner(), runtime, 3, launch)

    def test_disables_subprocess_and_joins_after_launch_failure(self):
        runtime = self.FakeRuntime()

        def launch(partition_id, inputs, tactic):
            runtime.events.append(("launch", partition_id, tactic))
            if partition_id == 1:
                raise RuntimeError("injected launch failure")

        runner = UgpuConcurrentTunableRunner(
            self.FakeRunner(), runtime, runtime.num_partitions, launch
        )
        assert runner.unique_id()[1] == runtime.num_partitions
        assert not runner.should_profile_tactic_in_subprocess(
            "fake", [torch.empty(1)], ("candidate",), TuningConfig()
        )

        with pytest.raises(RuntimeError, match="injected launch failure"):
            runner([torch.empty(1)], tactic=("candidate",))

        assert runtime.events[-1] == "join"
        assert runner.get_valid_tactics([torch.empty(1)], OptimizationProfile()) == [("candidate",)]

    def test_concurrent_tuning_disables_cold_l2_without_mutating_config(self):
        runtime = self.FakeRuntime()
        captured = {}

        class FakeTuner:
            def choose_one(self, op_name, runners, tuning_config, inputs, **kwargs):
                captured["op_name"] = op_name
                captured["tuning_config"] = tuning_config
                return runners[0], ("candidate",)

        tuning_config = TuningConfig(use_cold_l2_cache=True)
        with patch(
            "tensorrt_llm._torch.ugpu.autotune.AutoTuner.get",
            return_value=FakeTuner(),
        ):
            _, tactic = tune_ugpu_concurrent(
                "fake",
                self.FakeRunner(),
                runtime,
                runtime.num_partitions,
                lambda partition_id, inputs, tactic: None,
                [torch.empty(1)],
                tuning_config,
            )

        assert tactic == ("candidate",)
        assert captured["op_name"] == "fake::ugpu_concurrent"
        assert tuning_config.use_cold_l2_cache
        assert not captured["tuning_config"].use_cold_l2_cache


class TestUgpuLinearRouting:
    @pytest.mark.parametrize(
        "fusion_op",
        [AllReduceFusionOp.NONE, AllReduceFusionOp.RESIDUAL_RMS_NORM],
    )
    def test_shards_bypass_full_weight_paths(self, fusion_op):
        input_tensor = torch.ones(2, 4)
        local_output = torch.full((2, 3), 2.0)
        reduced_output = torch.full((2, 3), 4.0)
        bias = torch.ones(3)
        residual = (
            torch.zeros_like(local_output)
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
            else None
        )
        all_reduce_params = AllReduceParams(fusion_op=fusion_op, residual=residual)

        linear = MagicMock()
        linear.tp_mode = TensorParallelMode.ROW
        linear.tp_size = 2
        linear.tp_rank = 0
        linear.bias = bias
        linear.reduce_output = True
        linear.use_fused_gemm_allreduce = True
        linear._ugpu_weight_shards = [object(), object()]
        linear.partition_plan = SimpleNamespace(enabled=True)
        linear.weight = torch.empty(0)
        linear.weight_scale = torch.empty(0)
        linear.lora = None

        route = MagicMock()
        linear.apply_linear = route.apply_linear
        linear.apply_linear.return_value = local_output
        linear.all_reduce = route.all_reduce
        linear.all_reduce.return_value = reduced_output
        linear.all_reduce.uses_nccl_symmetric_memory_window.return_value = True
        linear.apply_linear_allreduce = MagicMock()
        linear.quant_method = MagicMock()
        linear.quant_method.supports_nccl_symmetric_memory_window_output = True
        linear._maybe_fuse_bias_into_allreduce = Linear._maybe_fuse_bias_into_allreduce.__get__(
            linear, Linear
        )

        output = Linear.forward(
            linear,
            input_tensor,
            all_reduce_params=all_reduce_params,
            layer_idx=7,
        )

        assert output is reduced_output
        assert [call[0] for call in route.mock_calls] == ["apply_linear", "all_reduce"]
        apply_args = linear.apply_linear.call_args.args
        assert apply_args[0] is input_tensor
        expected_bias = bias if fusion_op == AllReduceFusionOp.NONE else None
        assert apply_args[1] is expected_bias
        assert apply_args[2:] == (None, 7)
        all_reduce_args = linear.all_reduce.call_args
        assert all_reduce_args.args[0] is local_output
        assert all_reduce_args.kwargs["all_reduce_params"] is all_reduce_params
        if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
            assert all_reduce_params.bias is bias
        else:
            assert all_reduce_params.bias is None
        linear.apply_linear_allreduce.assert_not_called()
        linear.quant_method.apply.assert_not_called()
        linear.all_reduce.uses_nccl_symmetric_memory_window.assert_not_called()


class TestUgpuInitialization:
    """Tests for UGPU resource initialization."""

    def test_initialize_ugpu_resources(self, check_ugpu_support):
        """Test initializing UGPU resources for current device."""
        # Should not raise any exception
        initialize_ugpu_resources()

    def test_initialize_ugpu_resources_idempotent(self, check_ugpu_support):
        """Test that multiple initializations are safe."""
        # Initialize multiple times - should not raise
        initialize_ugpu_resources()
        initialize_ugpu_resources()
        initialize_ugpu_resources()


class TestUgpuStream:
    """Tests for UGPU stream retrieval."""

    def test_get_ugpu_stream_ugpu0(self, check_ugpu_support):
        """Test getting stream for uGPU0."""
        ugpu_id = 0
        stream = get_ugpu_stream(ugpu_id)

        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_get_ugpu_stream_ugpu1(self, check_ugpu_support):
        """Test getting stream for uGPU1."""
        ugpu_id = 1
        stream = get_ugpu_stream(ugpu_id)

        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_get_ugpu_stream_different_streams(self, check_ugpu_support):
        """Test that uGPU0 and uGPU1 have different streams."""
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        # The streams should be different objects
        assert stream0 is not stream1

    def test_reserved_remainder_stream_matches_configured_mode(self, check_ugpu_support):
        method = os.environ.get("TLLM_UGPU_STREAM_CREATE_METHOD", "").strip().lower()
        remainder = get_reserved_remainder_stream()
        if method == "balanced":
            assert remainder is None
        else:
            assert isinstance(remainder, torch.cuda.Stream)
            assert remainder.cuda_stream not in {
                get_ugpu_stream(0).cuda_stream,
                get_ugpu_stream(1).cuda_stream,
            }

    def test_get_ugpu_stream_invalid_ugpu_id(self, check_ugpu_support):
        """Test that invalid ugpu_id raises ValueError."""
        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_stream(2)

        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_stream(-1)

    def test_get_ugpu_stream_lazy_initialization(self, check_ugpu_support):
        """Test that stream retrieval triggers lazy initialization."""
        ugpu_id = 0

        # First call should initialize
        stream1 = get_ugpu_stream(ugpu_id)
        # Second call should return the same stream
        stream2 = get_ugpu_stream(ugpu_id)

        assert stream1 is stream2


class TestUgpuMempool:
    """Tests for UGPU memory pool retrieval."""

    def test_get_ugpu_mempool_ugpu0(self, check_ugpu_support):
        """Test getting mempool for uGPU0."""
        ugpu_id = 0

        try:
            mempool = get_ugpu_mempool(ugpu_id)
            assert mempool is not None
            assert isinstance(mempool, torch.cuda.MemPool)
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_ugpu1(self, check_ugpu_support):
        """Test getting mempool for uGPU1."""
        ugpu_id = 1

        try:
            mempool = get_ugpu_mempool(ugpu_id)
            assert mempool is not None
            assert isinstance(mempool, torch.cuda.MemPool)
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_different_pools(self, check_ugpu_support):
        """Test that uGPU0 and uGPU1 have different mempools."""
        try:
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)

            # The mempools should be different objects
            assert mempool0 is not mempool1
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise

    def test_get_ugpu_mempool_invalid_ugpu_id(self, check_ugpu_support):
        """Test that invalid ugpu_id raises ValueError."""
        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_mempool(2)

        with pytest.raises(ValueError, match="ugpu_id must be 0 or 1"):
            get_ugpu_mempool(-1)

    def test_get_ugpu_mempool_lazy_initialization(self, check_ugpu_support):
        """Test that mempool retrieval triggers lazy initialization."""
        ugpu_id = 0

        try:
            # First call should initialize
            mempool1 = get_ugpu_mempool(ugpu_id)
            # Second call should return the same mempool
            mempool2 = get_ugpu_mempool(ugpu_id)

            assert mempool1 is mempool2
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                pytest.skip(f"UGPU mempool not available: {e}")
            raise


class TestUgpuIntegration:
    """Integration tests for UGPU utilities."""

    def test_all_resources_initialized_together(self, check_ugpu_support):
        """Test that initializing creates all resources (streams and mempools)."""
        # Initialize once
        initialize_ugpu_resources()

        # Core resources should always be available
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        assert stream0 is not None
        assert stream1 is not None

        # Mempools may or may not be available depending on system support
        try:
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)
            assert mempool0 is not None
            assert mempool1 is not None
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                # This is acceptable - mempools are optional
                pytest.skip(f"UGPU mempool not available: {e}")
            else:
                raise

    def test_resources_persistent_across_calls(self, check_ugpu_support):
        """Test that resources are persistent and reused."""
        # Get resources multiple times
        stream0_1 = get_ugpu_stream(0)
        stream0_2 = get_ugpu_stream(0)

        # Should be the same objects
        assert stream0_1 is stream0_2

        # Test mempool persistence if available
        try:
            mempool1_1 = get_ugpu_mempool(1)
            mempool1_2 = get_ugpu_mempool(1)
            assert mempool1_1 is mempool1_2
        except RuntimeError as e:
            if "allocator" in str(e).lower() and "not available" in str(e).lower():
                # This is acceptable - mempools are optional
                pass
            else:
                raise

    def test_stream_can_be_used_for_operations(self, check_ugpu_support):
        """Test that UGPU streams can be used for CUDA operations."""
        ugpu_id = 0
        stream = get_ugpu_stream(ugpu_id)

        # Create a tensor and perform an operation on the stream
        device_id = torch.cuda.current_device()
        with torch.cuda.stream(stream):
            tensor = torch.randn(10, 10, device=f"cuda:{device_id}")
            result = tensor * 2

            # Wait for stream to complete
            stream.synchronize()

            assert result.shape == (10, 10)
            assert result.device.type == "cuda"

    def test_concurrent_stream_operations(self, check_ugpu_support):
        """Test that operations on different UGPU streams can execute concurrently."""
        stream0 = get_ugpu_stream(0)
        stream1 = get_ugpu_stream(1)

        device_id = torch.cuda.current_device()

        # Launch operations on both streams
        with torch.cuda.stream(stream0):
            tensor0 = torch.randn(100, 100, device=f"cuda:{device_id}")
            result0 = tensor0 @ tensor0.T

        with torch.cuda.stream(stream1):
            tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
            result1 = tensor1 @ tensor1.T

        # Synchronize both streams
        stream0.synchronize()
        stream1.synchronize()

        # Verify results
        assert result0.shape == (100, 100)
        assert result1.shape == (100, 100)
        assert not torch.allclose(result0, result1)  # Different results


class TestUgpuMempoolAllocation:
    """Tests for UGPU memory pool allocation and deallocation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_copy_to_new_cuda_allocation_does_not_alias_contiguous_input(self):
        source_storage = torch.arange(32, device="cuda")
        source = source_storage[:16]
        assert source.is_contiguous()
        assert source.untyped_storage().data_ptr() == source_storage.untyped_storage().data_ptr()

        copied = _copy_to_new_cuda_allocation(source)

        assert copied.is_contiguous()
        assert copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        torch.testing.assert_close(copied, source)

    def test_allocate_tensor_with_mempool(self, check_ugpu_support):
        """Test allocating a tensor using UGPU mempool."""
        try:
            ugpu_id = 0
            mempool = get_ugpu_mempool(ugpu_id)
            stream = get_ugpu_stream(ugpu_id)

            device_id = torch.cuda.current_device()

            # Allocate tensor using the mempool
            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate tensor using the UGPU mempool
                    tensor = torch.randn(100, 100, device=f"cuda:{device_id}")

                    # Perform operation
                    result = tensor * 2.0

                    # Synchronize
                    stream.synchronize()

                    # Verify
                    assert result.shape == (100, 100)
                    assert result.device.type == "cuda"

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool allocation not supported: {e}")
            raise

    def test_mempool_with_stream_context(self, check_ugpu_support):
        """Test using mempool within its corresponding stream context."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            # Use mempool within stream context
            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate multiple tensors
                    tensors = []
                    for i in range(5):
                        tensor = torch.randn(50, 50, device=f"cuda:{device_id}")
                        tensors.append(tensor)

                    # Perform operations
                    results = [t @ t.T for t in tensors]

                    # Synchronize
                    stream.synchronize()

                    # Verify all results
                    for result in results:
                        assert result.shape == (50, 50)
                        assert result.device.type == "cuda"

            # Cleanup - delete tensors to free memory
            del tensors
            del results
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_large_allocation_with_mempool(self, check_ugpu_support):
        """Test allocating large tensors using UGPU mempool."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate a large tensor (100MB)
                    large_tensor = torch.randn(5000, 5000, device=f"cuda:{device_id}")

                    # Perform operation to ensure it's accessible
                    result = large_tensor.sum()

                    # Synchronize
                    stream.synchronize()

                    # Verify
                    assert result.device.type == "cuda"
                    assert large_tensor.shape == (5000, 5000)

            # Cleanup
            del large_tensor
            del result
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Not enough memory for large allocation test: {e}")
            elif (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_mempool_reuse_after_free(self, check_ugpu_support):
        """Test that mempool can reuse freed memory."""
        try:
            ugpu_id = 0
            stream = get_ugpu_stream(ugpu_id)
            mempool = get_ugpu_mempool(ugpu_id)

            device_id = torch.cuda.current_device()

            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(mempool):
                    # Allocate tensor
                    tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result1 = tensor1.sum()
                    stream.synchronize()

                    # Free tensor
                    del tensor1
                    torch.cuda.empty_cache()

                    # Allocate another tensor of same size
                    tensor2 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result2 = tensor2.sum()
                    stream.synchronize()

                    # Verify both operations succeeded
                    assert result1.device.type == "cuda"
                    assert result2.device.type == "cuda"

            # Cleanup
            del tensor2
            del result1, result2
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool operation not supported: {e}")
            raise

    def test_mempool_across_different_ugpus(self, check_ugpu_support):
        """Test allocating memory on different UGPU mempools."""
        try:
            stream0 = get_ugpu_stream(0)
            stream1 = get_ugpu_stream(1)
            mempool0 = get_ugpu_mempool(0)
            mempool1 = get_ugpu_mempool(1)

            device_id = torch.cuda.current_device()

            # Allocate on uGPU0
            with torch.cuda.stream(stream0):
                with torch.cuda.use_mem_pool(mempool0):
                    tensor0 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result0 = tensor0 @ tensor0.T

            # Allocate on uGPU1
            with torch.cuda.stream(stream1):
                with torch.cuda.use_mem_pool(mempool1):
                    tensor1 = torch.randn(100, 100, device=f"cuda:{device_id}")
                    result1 = tensor1 @ tensor1.T

            # Synchronize both
            stream0.synchronize()
            stream1.synchronize()

            # Verify both allocations succeeded
            assert result0.shape == (100, 100)
            assert result1.shape == (100, 100)
            assert not torch.allclose(result0, result1)

            # Cleanup
            del tensor0, tensor1, result0, result1
            torch.cuda.empty_cache()

        except (RuntimeError, AttributeError) as e:
            if (
                "allocator" in str(e).lower()
                or "mempool" in str(e).lower()
                or "use_mem_pool" in str(e).lower()
            ):
                pytest.skip(f"UGPU mempool not available: {e}")
            raise
