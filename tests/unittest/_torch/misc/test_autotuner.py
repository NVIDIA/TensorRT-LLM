import itertools
import json
import os
import pickle
import sys
import tempfile
from typing import Any, List

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm._torch.autotuner as autotuner
from tensorrt_llm._torch.autotuner import (AutoTuner, DistributedTuningStrategy,
                                           DynamicDim, DynamicTensorSpec,
                                           FakeTensor, OptimizationProfile,
                                           StaticDim, TunableRunner,
                                           TuningConfig, autotune)
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.utils import (get_power_of_2_num_tokens_buckets,
                                       next_positive_power_of_2)
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def test_multi_dynamic_dims():
    tuner = autotuner.AutoTuner()
    x = torch.rand([5, 1024])
    w = torch.rand([7, 9])
    dynamic_tensor_specs = (
        DynamicTensorSpec(0, 0, [1, 3, 5]),
        DynamicTensorSpec(0, 1, [16, 24, 1024]),
        # map_to_tuning_buckets is only applied at runtime, not during tuning
        DynamicTensorSpec(1,
                          1, [3, 7, 9],
                          map_to_tuning_buckets=lambda x: x // 2),
    )

    profiles = tuner._optimization_profiles(
        tuning_config=TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs),
        inputs=[x, w])
    # choice(0, 0) * choice(0, 1) * choice(1, 1)
    # 3 * 3 * 3 = 27, input value 9 is already inside the bucket
    assert len(profiles) == 27
    sample_0 = OptimizationProfile(shapes=[[
        DynamicDim(min=1, opt=1, max=3),
        DynamicDim(min=16, opt=16, max=24)
    ], [StaticDim(val=7), DynamicDim(min=3, opt=3, max=7)]])
    sample_26 = OptimizationProfile(shapes=[[
        DynamicDim(min=5, opt=5, max=float('inf')),
        DynamicDim(min=1024, opt=1024, max=float('inf'))
    ], [StaticDim(
        val=7), DynamicDim(min=9, opt=9, max=float('inf'))]])

    assert sample_0 == profiles[0]
    assert sample_26 == profiles[-1]


# For cache testing
"""
tactic 0 is better when x.shape[0] <= M // 2
tactic 1 is better when x.shape[0] > M // 2
"""
M = 32


# add sleep to simulate bad perf
def gemm_0(x, w):
    if x.shape[0] > M // 2:
        delay_kernel(100, torch.cuda.current_stream())
    return x @ w


def gemm_1(x, w):
    if x.shape[0] <= M // 2:
        delay_kernel(100, torch.cuda.current_stream())
    return x @ w


def gemm_fallback(x, w) -> torch.Tensor:
    # always the slowest
    delay_kernel(500, torch.cuda.current_stream())
    return x @ w


def check_gemm_tactic_valid(tactic: int, m: int) -> bool:
    # TODO: CI is not stable for this test. delay_kernel can not guarantee the profiling result.
    # We need to find a more determinist way to test this.
    if m <= M // 2:
        if tactic != 0:
            logger.warning(
                f"Expect tactic 0 but got {tactic} when m ({m}) is small.")
    elif m <= M:
        if tactic != 1:
            logger.warning(
                f"Expect tactic 1 but got {tactic} when m ({m}) is large.")
    else:
        if tactic != -1:
            logger.warning(
                f"Expect fallback tactic (-1) but got {tactic} when m ({m}) > {M}."
            )


class GemmRunner(TunableRunner):

    def get_valid_tactics(self, inputs: List[FakeTensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        # The simulated delay is not deterministic, so we need to return specific tactics here
        return [-1, 0, 1]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1,
                **kwargs) -> torch.Tensor:
        assert tactic in [-1, 0, 1]
        return [gemm_0, gemm_1, gemm_fallback][tactic](*inputs)


@torch.library.custom_op("autotuner_test::get_best_gemm_tactic",
                         mutates_args=())
def get_best_gemm_tactic(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    runners = [GemmRunner()]
    tuner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )
    runner, tactic = tuner.choose_one(
        "autotuner_test::get_best_gemm_tactic",
        runners,
        tuning_config,
        [x, w],
    )
    return torch.tensor(tactic)


@get_best_gemm_tactic.register_fake
def _(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty(1)


def test_autotuner_cache_basic():
    w = torch.randn(64, 128)

    # tuning with largest M
    AutoTuner.get().clear_cache()
    with autotune():
        torch.ops.autotuner_test.get_best_gemm_tactic(torch.randn(M, 64), w)

    # This tests the logic of print_profiling_cache and print_statistics
    AutoTuner.get().print_profiling_cache()
    AutoTuner.get().print_statistics()

    m = M * 2
    while m >= 1:
        best_tactic = torch.ops.autotuner_test.get_best_gemm_tactic(
            torch.randn(m, 64), w)
        check_gemm_tactic_valid(best_tactic, m)
        m //= 2


def test_bucket_mapping():
    """Test that map_to_tuning_buckets correctly maps runtime sizes to tuning buckets.

    This test demonstrates the single mapper approach:
    - During tuning: NO mapper is applied, raw bucket values are used as cache keys
    - During runtime: map_to_tuning_buckets is applied to map buffer size to actual work size

    With sparsity=0.25, the buffer contains 25% actual work:
    - Tuning stores buckets: 1, 2, 4, 8, 16, 32 as raw cache keys
    - Runtime buffer 4 -> maps to bucket int(4 * 0.25) = 1
    - Runtime buffer 16 -> maps to bucket int(16 * 0.25) = 4

    In MoE EP, the input buffer is allocated for worst-case but sparsely filled.
    Using map_to_tuning_buckets allows us to map buffer size to actual work size at runtime.
    """
    w = torch.randn(64, 128)
    tuner = AutoTuner.get()
    tuner.clear_cache()

    # Sparsity indicates the fraction of buffer containing valid work
    sparsity = 0.25

    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets(M),
        map_to_tuning_buckets=lambda x: int(x * sparsity)), ), )

    with autotune():
        tuner.choose_one("test_bucket_mapping", [GemmRunner()], tuning_config,
                         [torch.randn(1, 64), w])

    # Verify cache entries use raw tuning bucket values
    cache_entries = tuner.profiling_cache.get_specific_custom_op(
        "test_bucket_mapping")

    # Extract the first dimension of the first input shape from each cache key
    assert len(cache_entries) == len(tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets), \
        f"Expected {len(tuning_config.dynamic_tensor_specs[0].gen_tuning_buckets)} cache entries, got {len(cache_entries)}"

    # Test runtime mapping: buffer size is mapped via map_to_runtime_buckets
    # to find the correct tuning bucket based on actual work size
    test_cases = [
        # size 4 -> valid work size (4*0.25)=1, tactic 0 since 1 <= M//2
        (4, 1, 0),
        # size 8 -> valid work size (8*0.25)=2, tactic 0 since 2 <= M//2
        (8, 2, 0),
        # size 16 -> valid work size (16*0.25)=4, tactic 0 since 4 <= M//2
        (16, 4, 0),
        # size 32 -> valid work size (32*0.25)=8, tactic 0 since 8 <= M//2
        (32, 8, 0),
        # size 64 -> valid work size (64*0.25)=16, tactic 0 since 16 <= M//2
        (64, 16, 0),
        # size 128 -> valid work size (128*0.25)=32, tactic 1 since 32 > M//2
        (128, 32, 1),
        # size 256 -> valid work size (256*0.25)=64, tactic -1 since 64 > M
        (256, 64, -1),
    ]

    for buffer_size, valid_size, expected_tactic in test_cases:
        # Verify cache lookup succeeds with the mapped bucket
        x = torch.randn(buffer_size, 64)
        runner, tactic = tuner.choose_one("test_bucket_mapping", [GemmRunner()],
                                          tuning_config, [x, w])
        assert (
            tactic == expected_tactic
        ), f"buffer size={buffer_size} -> valid work size={valid_size}, expected tactic {expected_tactic} but got {tactic}"


def test_autotuner_try_block():

    class PartialCrashedRunner(TunableRunner):

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [-1, 0, 1]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1) -> torch.Tensor:
            assert tactic in [-1, 0, 1]
            if tactic == 1:
                raise Exception(
                    "For profiling try block test: Tactic 1 is not suitable. Crash happens."
                )
            return [gemm_0, gemm_1, gemm_fallback][tactic](*inputs)

    x, w = torch.randn(M, 64), torch.randn(64, 128)
    runners = [PartialCrashedRunner()]
    tuner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )
    with autotune():
        runner, tactic = tuner.choose_one("test_autotuner_try_block", runners,
                                          tuning_config, [x, w])

    m = M // 2
    while m >= 1:
        _, tactic = tuner.choose_one("test_autotuner_try_block", runners,
                                     tuning_config, [torch.randn(m, 64), w])
        assert tactic in [
            -1, 0
        ], f"Expect only tactic -1, 0 being chosen, but got tactic {tactic}."
        m //= 2


@torch.library.custom_op("autotuner_test::recursive_get_best_gemm_tactic",
                         mutates_args=())
def recursive_get_best_gemm_tactic(x: torch.Tensor, w1: torch.Tensor,
                                   w2: torch.Tensor) -> torch.Tensor:
    # Only the first custom_op is tuned, the second one uses the tuned result in cache
    tactic_1 = get_best_gemm_tactic(x, w1)
    tactic_2 = get_best_gemm_tactic(x, w2)
    return torch.stack([tactic_1, tactic_2])


@recursive_get_best_gemm_tactic.register_fake
def _(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return torch.empty(2)


def test_recursive_autotuner():
    x, w1, w2 = torch.randn(M, 64), torch.randn(64, 128), torch.randn(64, 128)
    AutoTuner.get().clear_cache()
    with autotune():
        torch.ops.autotuner_test.recursive_get_best_gemm_tactic(
            torch.randn(M, 64), w1, w2)

    m = M * 2
    while m >= 1:
        t1, t2 = torch.ops.autotuner_test.recursive_get_best_gemm_tactic(
            torch.randn(m, 64), w1, w2)
        check_gemm_tactic_valid(t1, m)
        check_gemm_tactic_valid(t2, m)
        m //= 2


class GemmRunnerWithAttributes(TunableRunner):

    def __init__(self, block_size: int, num_warps: int):
        self.block_size = block_size
        self.num_warps = num_warps

    def get_valid_tactics(self, inputs: List[FakeTensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        return [-1, 0, 1]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1) -> torch.Tensor:
        assert tactic in [-1, 0, 1]
        return [gemm_0, gemm_1, gemm_fallback][tactic](*inputs)


def test_multiple_runners_different_attributes():
    """Test that runners with different attributes get different cache entries"""
    x, w = torch.randn(16, 64), torch.randn(64, 128)

    # Create runners with different attributes
    runner_0 = GemmRunnerWithAttributes(block_size=128, num_warps=4)
    runner_1 = GemmRunnerWithAttributes(block_size=256, num_warps=8)
    runners = [runner_0, runner_1]

    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )

    # Do tuning
    with autotune():
        tuner = AutoTuner.get()
        runner_a, tactic_a = tuner.choose_one("test_multiple_runners", runners,
                                              tuning_config, [x, w])

        # Verify different cache keys are generated
        shapes = (x.shape, w.shape)
        cache_key_0 = tuner.profiling_cache.get_cache_key(
            custom_op="test_multiple_runners",
            input_shapes=shapes,
            runner=runner_0,
            tuning_config=tuning_config,
        )
        cache_key_1 = tuner.profiling_cache.get_cache_key(
            custom_op="test_multiple_runners",
            input_shapes=shapes,
            runner=runner_1,
            tuning_config=tuning_config,
        )

        assert cache_key_0 != cache_key_1, "Runners with different attributes should have different cache keys"


def test_multiple_dynamic_shapes_cache():
    """Test that different dynamic shape combinations are properly cached"""
    w = torch.randn(64, 128)
    runners = [GemmRunner()]

    # Define dynamic ranges for both dimensions
    tuning_config = TuningConfig(dynamic_tensor_specs=(
        DynamicTensorSpec(input_idx=0,
                          dim_idx=0,
                          gen_tuning_buckets=(3, 4, 5),
                          map_to_tuning_buckets=lambda x: x),
        DynamicTensorSpec(input_idx=1,
                          dim_idx=1,
                          gen_tuning_buckets=(64, 128, 256, 512),
                          map_to_tuning_buckets=lambda x: x),
    ), )

    # Do tuning with a sample input
    x = torch.randn(3, 64)
    temp_dir = tempfile.TemporaryDirectory()
    cache_path = os.path.join(temp_dir.name,
                              "test_multiple_dynamic_shapes.json")
    with autotune(cache_path=cache_path):
        tuner = AutoTuner.get()
        runner, tactic = tuner.choose_one("test_multiple_dynamic_shapes",
                                          runners, tuning_config, [x, w])

    cache_entries = tuner.profiling_cache.get_specific_custom_op(
        "test_multiple_dynamic_shapes")
    assert len(cache_entries) == 12, \
        f"Expected 12 cache entries for 3x4 shape combinations, got {len(cache_entries)}"
    # Verify cache size - should have 12 entries (3x4 combinations)
    # We also test the cache serialization and deserialization here.
    AutoTuner.get().profiling_cache.clear()
    AutoTuner.get().profiling_cache.load_cache(cache_path, rank=0)
    cache_entries = tuner.profiling_cache.get_specific_custom_op(
        "test_multiple_dynamic_shapes")

    assert len(cache_entries) == 12, \
        f"Expected 12 cache entries for 3x4 shape combinations, got {len(cache_entries)}"


class GemmRunnerComplexTuningConfigs(TunableRunner):

    # test serialization of different types of tactics
    valid_tactic_ids = [-1, 0, 1]
    valid_tile_sizes = [(128, 128), (256, 256)]
    valid_cluster_sizes = [[1, 1, 1], [2, 2, 1]]

    tune_max_num_tokens = 32

    def get_valid_tactics(
        self,
        inputs: List[FakeTensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Any]:
        # During the tuning process, we verify if the tuning config behaves as expected
        assert inputs[0].shape[0] <= self.tune_max_num_tokens, \
            f"Input shape {inputs[0].shape[0]} is larger than the max num tokens {self.tune_max_num_tokens}"

        assert inputs[0][-1, 0] == inputs[0].shape[0], \
            f"Input shape {inputs[0].shape[0]} is not set through the pre_hook correctly"

        return [{
            "int_tactic_id": tactic_id,
            "tuple_tile_size": tile_size,
            "list_cluster_size": cluster_size,
        } for tactic_id, tile_size, cluster_size in itertools.product(
            self.valid_tactic_ids,
            self.valid_tile_sizes,
            self.valid_cluster_sizes,
        )]

    def forward(
        self,
        /,
        inputs: List[torch.Tensor],
        *,
        tactic: Any = -1,
    ) -> torch.Tensor:
        # Notice that in fallback case tactic is -1
        if tactic == -1:
            # assign default configs for fallback case
            tactic_id, tile_size, cluster_size = -1, (128, 256), [1, 1, 1]
        else:
            tactic_id, tile_size, cluster_size = tactic[
                "int_tactic_id"], tactic["tuple_tile_size"], tactic[
                    "list_cluster_size"]

        assert isinstance(tactic_id, int) and tactic_id in self.valid_tactic_ids
        assert isinstance(tile_size, tuple) and len(tile_size) == 2 \
            and tile_size in self.valid_tile_sizes
        assert isinstance(cluster_size, list) and len(cluster_size) == 3 \
            and cluster_size in self.valid_cluster_sizes
        return [gemm_0, gemm_1, gemm_fallback][tactic_id](*inputs)

    @staticmethod
    def inputs_pre_hook(inputs: List[torch.Tensor]):
        # always set the first element to be the number of tokens in x
        x, w = inputs
        x_hooked = torch.zeros_like(x)
        x_hooked[-1, 0] = x.shape[0]
        return [x_hooked, w]


def test_autotuner_tuning_configs():
    runner_0 = GemmRunnerComplexTuningConfigs()
    runners = [runner_0]
    x, w = torch.randn(64, 64), torch.randn(64, 128)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            input_idx=0,
            dim_idx=0,
            gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
            map_to_tuning_buckets=next_positive_power_of_2,
        ), ),
        # Test if the number of tuning tokens is clipped to 32
        tune_max_num_tokens=GemmRunnerComplexTuningConfigs.tune_max_num_tokens,
        inputs_pre_hook=GemmRunnerComplexTuningConfigs.inputs_pre_hook,
        use_cold_l2_cache=True,
        use_cuda_graph=False,
    )
    temp_dir = tempfile.TemporaryDirectory()
    cache_path = os.path.join(temp_dir.name,
                              "test_autotuner_tactic_configs.json")
    with autotune(cache_path=cache_path):
        tuner = AutoTuner.get()
        runner, best_tactic = tuner.choose_one("test_autotuner_tactic_configs",
                                               runners, tuning_config, [x, w])

    runner_0([x, w], tactic=best_tactic)

    # Test if the tactic can be loaded from cache correctly
    AutoTuner.get().profiling_cache.clear()
    AutoTuner.get().profiling_cache.load_cache(cache_path, rank=0)

    # No further tuning should be performed.
    runner, deserialized_tactic = tuner.choose_one(
        "test_autotuner_tactic_configs", runners, tuning_config, [x, w])
    assert best_tactic == deserialized_tactic, "Tactic should be the same after deserialization"

    runner_0([x, w], tactic=deserialized_tactic)


def test_kernel_testing_single_context():
    """Test kernel testing with a single choose_one context"""
    x, w = torch.randn(16, 64), torch.randn(64, 128)
    runners = [GemmRunner()]
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )

    tuner = AutoTuner.get()
    tuner.clear_cache()

    # First, do tuning to populate cache
    with autotune():
        runner, tactic = tuner.choose_one("test_kernel_testing_single", runners,
                                          tuning_config, [x, w])

    # Capture execution context
    with tuner.capture() as all_tactics:
        runner, tactic = tuner.choose_one("test_kernel_testing_single", runners,
                                          tuning_config, [x, w])
        reference_output = runner([x, w], tactic=tactic)

    # Test all tactics
    tested_tactics = []
    for (runner, tactic), in all_tactics:
        tested_tactics.append((runner, tactic))
        with tuner.replay(((runner, tactic), )):
            runner_ret, tactic_ret = tuner.choose_one(
                "test_kernel_testing_single", runners, tuning_config, [x, w])
            output = runner_ret([x, w], tactic=tactic_ret)
            # Verify output matches reference
            torch.testing.assert_close(output, reference_output)
            assert runner == runner_ret and tactic == tactic_ret, \
                f"Runner and tactic mismatch: expected ({runner, tactic}), got ({runner_ret, tactic_ret})"

    # Should have tested 3 tactics ([-1, 0, 1])
    assert len(tested_tactics) == len(GemmRunner().get_valid_tactics([x, w], OptimizationProfile([[]]))), \
        f"Expected 3 tactics to be tested, got {len(tested_tactics)}"


class MultiContextRunner(TunableRunner):

    def get_valid_tactics(self, inputs: List[FakeTensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        gemm_idx = kwargs.get("gemm_idx", 0)
        # Different gemm_idx have different number of tactics
        if gemm_idx == 0:
            return [0, 1]
        else:
            return [0, 1, 2]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1,
                **kwargs) -> torch.Tensor:
        gemm_idx = kwargs.get("gemm_idx", 0)
        # Analogous to CUTLASS MoE trtllm::fused_moe FC1
        if gemm_idx == 0:
            return [gemm_0, gemm_1][tactic](inputs[0], inputs[1])
        # Analogous to CUTLASS MoE trtllm::fused_moe FC2
        else:
            return [gemm_0, gemm_1, gemm_fallback][tactic](inputs[1].T,
                                                           inputs[0].T)


def test_kernel_testing_multiple_contexts():
    """
    Test kernel testing with multiple choose_one contexts
    (e.g., CUTLASS MoE trtllm::fused_moe)
    """

    x, w = torch.randn(16, 64), torch.randn(64, 128)
    runners = [MultiContextRunner()]
    tuning_config = TuningConfig()

    tuner = AutoTuner.get()
    tuner.clear_cache()

    # First, do tuning to populate cache
    with autotune():
        runner, _ = tuner.choose_one("test_multi_context",
                                     runners,
                                     tuning_config, [x, w],
                                     gemm_idx=0)
        runner, _ = tuner.choose_one("test_multi_context",
                                     runners,
                                     tuning_config, [x, w],
                                     gemm_idx=1)

    # Capture execution context (captures both choose_one calls)
    with tuner.capture() as all_tactics:
        runner_0, tactic_0 = tuner.choose_one("test_multi_context",
                                              runners,
                                              tuning_config, [x, w],
                                              gemm_idx=0)
        runner_1, tactic_1 = tuner.choose_one("test_multi_context",
                                              runners,
                                              tuning_config, [x, w],
                                              gemm_idx=1)
        ref_output_0 = runner_0([x, w], tactic=tactic_0, gemm_idx=0)
        ref_output_1 = runner_1([x, w], tactic=tactic_1, gemm_idx=1)

    # Test all tactic combinations (cartesian product)
    tested_tactics = []
    for tactic in all_tactics:
        tested_tactics.append(tactic)
        # Each tactic is ((runner_0, tactic_0), (runner_1, tactic_1))
        assert len(tactic) == 2, f"Expected 2 contexts, got {len(tactic)}"

        with tuner.replay(tactic):
            # Make the same calls in the same order
            runner_0, tactic_0 = tuner.choose_one("test_multi_context",
                                                  runners,
                                                  tuning_config, [x, w],
                                                  gemm_idx=0)
            runner_1, tactic_1 = tuner.choose_one("test_multi_context",
                                                  runners,
                                                  tuning_config, [x, w],
                                                  gemm_idx=1)

            output_0 = runner_0([x, w], tactic=tactic_0, gemm_idx=0)
            output_1 = runner_1([x, w], tactic=tactic_1, gemm_idx=1)

            # Verify each context independently
            # Since we're testing different tactics, outputs will differ
            # Just verify they don't crash and have correct shapes
            assert output_0.shape == ref_output_0.shape
            assert output_1.shape == ref_output_1.shape

    # Should have tested 2*3 = 6 combinations
    num_tactics_for_gemm_idx = lambda gemm_idx: len(runners[
        0].get_valid_tactics([x, w], OptimizationProfile(), gemm_idx=gemm_idx))
    assert len(tested_tactics) == num_tactics_for_gemm_idx(0) * num_tactics_for_gemm_idx(1), \
        f"Expected 6 tactic combinations (2*3), got {len(tested_tactics)}"


def test_kernel_testing_mismatched_ops():
    """
    Correctly raise and capture the exception when captured context != operation performed
    """
    x, w = torch.randn(16, 64), torch.randn(64, 128)
    runners = [GemmRunner()]
    tuning_config = TuningConfig()

    tuner = AutoTuner.get()
    tuner.clear_cache()

    # Capture execution context for operation A
    with tuner.capture() as all_tactics:
        _ = tuner.choose_one("test_op_A", runners, tuning_config, [x, w])

    # Try to test with operation B (should raise RuntimeError)
    try:
        for (runner, tactic), in all_tactics:
            with tuner.replay(((runner, tactic), )):
                # This should raise RuntimeError because custom_op doesn't match
                _ = tuner.choose_one("test_op_B", runners, tuning_config,
                                     [x, w])
        assert False, "Expected RuntimeError for mismatched custom_op, but none was raised"
    except RuntimeError as e:
        # Verify the error message contains useful information
        error_msg = str(e)
        assert "Custom op mismatch" in error_msg, f"Expected 'Custom op mismatch' in error message, got: {error_msg}"
        assert "test_op_A" in error_msg, f"Expected 'test_op_A' in error message, got: {error_msg}"
        assert "test_op_B" in error_msg, f"Expected 'test_op_B' in error message, got: {error_msg}"


class DistributedGemmRunner(TunableRunner):

    def __init__(self, prefer_tactics: List[int] = [0, 1]):
        self.prefer_tactics = prefer_tactics

    def get_valid_tactics(self, inputs, profile, **kwargs):
        # Return all tactics so merge strategy can choose between them
        return self.prefer_tactics

    def forward(self, inputs, *, tactic=-1, **kwargs):
        # tactic 0 is slower
        if tactic % 2 == 0:
            for _ in range(5):
                inputs[0] @ inputs[1]
        return inputs[0] @ inputs[1]

    def unique_id(self):
        return ()


def _distributed_worker_function(world_size, strategy):
    """Worker function to run on each MPI rank."""
    rank = tensorrt_llm.mpi_rank()
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      tp_size=world_size,
                      pp_size=1)
    dist = Distributed.get(mapping)

    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.setup_distributed_state(mapping)

    x = torch.randn(16, 32, device='cuda')
    w = torch.randn(32, 64, device='cuda')
    inputs = [x, w]

    if strategy == DistributedTuningStrategy.PARALLEL:
        # All ranks get the same set of tactics
        prefer_tactics = [0, 1, 2, 3]
    else:
        # Each rank prefers different tactics
        prefer_tactics = [rank]
    runner = DistributedGemmRunner(prefer_tactics=prefer_tactics)
    runner_independent = DistributedGemmRunner()
    config = TuningConfig(distributed_tuning_strategy=strategy)
    config_independent = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.INDEPENDENT)

    # Keep temp_dir in function scope to prevent premature garbage collection
    temp_dir = None
    if rank == 0:
        temp_dir = tempfile.TemporaryDirectory()
        # rank 0 should broadcast the cache path to all ranks
        cache_path = os.path.join(temp_dir.name, "test_distributed_tuning.json")
        dist.broadcast(cache_path, root=0)
    else:
        cache_path = dist.broadcast(None, root=0)

    with autotune(cache_path=cache_path):
        tuner.choose_one(custom_op=f"test_distributed_{strategy.value}",
                         runners=[runner],
                         tuning_config=config,
                         inputs=inputs)
        # run another normal gemm with INDEPENDENT strategy
        tuner.choose_one(custom_op=f"test_distributed_normal_gemm",
                         runners=[runner_independent],
                         tuning_config=config_independent,
                         inputs=inputs)

    # Check only one file is created in the cache path
    assert len(os.listdir(os.path.dirname(
        cache_path))) == 1, "Only one rank file should be created"

    dist.barrier()

    # Check cache for distributed tuning
    AutoTuner.get().profiling_cache.clear()
    AutoTuner.get().profiling_cache.load_cache(cache_path, rank)

    selected_runner, best_tactic = tuner.choose_one(
        custom_op=f"test_distributed_{strategy.value}",
        runners=[runner],
        tuning_config=config,
        inputs=inputs)

    # Verify cache file structure based on distributed strategy
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    # Helper to check if an op name appears in any cache key string
    def has_op_in_section(section_data: dict, op_name: str) -> bool:
        return any(op_name in key_str for key_str in section_data.keys())

    assert 'metadata' in cache_data, "Metadata should be present"
    assert f'rank_{rank}' in cache_data, f"rank {rank} should be present"

    # The INDEPENDENT op "test_distributed_normal_gemm" should always be in rank-specific sections
    assert has_op_in_section(cache_data[f'rank_{rank}'], 'test_distributed_normal_gemm'), \
        f"rank {rank} should have test_distributed_normal_gemm"

    if strategy == DistributedTuningStrategy.INDEPENDENT:
        # Both ops use INDEPENDENT strategy, so no shared section
        assert 'shared' not in cache_data or len(cache_data.get('shared', {})) == 0, \
            "shared should not be present or be empty for INDEPENDENT strategy"
        # Each rank should have 2 entries (the parameterized op + normal_gemm)
        assert len(cache_data[f'rank_{rank}']) == 2, \
            f"rank {rank} should have 2 entries, got {len(cache_data[f'rank_{rank}'])}"
        assert has_op_in_section(cache_data[f'rank_{rank}'], f'test_distributed_{strategy.value}'), \
            f"rank {rank} should have test_distributed_{strategy.value}"

        assert len(
            AutoTuner.get().profiling_cache.independent_op
        ) == 0, f"Non-INDEPENDENT ops should not be present in the cache"
    else:
        # Non-INDEPENDENT ops go to shared section
        assert 'shared' in cache_data, "shared section should be present"
        # Each rank should have only 1 entry (the normal_gemm with INDEPENDENT strategy)
        assert len(cache_data[f'rank_{rank}']) == 1, \
            f"rank {rank} should have 1 entry, got {len(cache_data[f'rank_{rank}'])}"
        # The parameterized op should NOT be in rank-specific section
        assert not has_op_in_section(cache_data[f'rank_{rank}'], f'test_distributed_{strategy.value}'), \
            f"rank {rank} should not have test_distributed_{strategy.value}"
        # The parameterized op should be in shared section
        assert has_op_in_section(cache_data['shared'], f'test_distributed_{strategy.value}'), \
            f"shared should have test_distributed_{strategy.value}"

        assert "test_distributed_normal_gemm" not in AutoTuner.get().profiling_cache.independent_op and \
            f"test_distributed_{strategy.value}" in AutoTuner.get().profiling_cache.independent_op, \
            f"Distributed tuning strategy is not recovered correctly from cache"

    if strategy == DistributedTuningStrategy.BROADCAST:
        # All ranks should select tactic 0
        assert best_tactic == 0, f"Rank {rank} with {strategy} should select tactic 0, got {best_tactic}"
    elif strategy == DistributedTuningStrategy.INDEPENDENT:
        # Each rank should select the tactic it prefers
        assert best_tactic == rank, f"Rank {rank} with {strategy} should select tactic {rank}, got {best_tactic}"
    elif strategy == DistributedTuningStrategy.MERGE:
        # Because tactic 0 is slower, two ranks should always select tactic 1
        assert best_tactic == 1, f"Rank {rank} with {strategy} should select tactic 1, got {best_tactic}"
    elif strategy == DistributedTuningStrategy.PARALLEL:
        # Tactic 1 or 3 should be selected since they are faster.
        # TODO: This might not cover the case that rank1 tunes nothing
        assert best_tactic % 2 == 1, f"Rank {rank} with {strategy} should select tactic 1, got {best_tactic}"
    else:
        assert False, f"Rank {rank} got unknown strategy: {strategy}"

    dist.barrier()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize(
    "strategy",
    [
        DistributedTuningStrategy.BROADCAST,
        DistributedTuningStrategy.INDEPENDENT,
        DistributedTuningStrategy.MERGE,
        DistributedTuningStrategy.PARALLEL,
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_autotuner_distributed_strategy(strategy, mpi_pool_executor):
    world_size = 2
    # Use MPIPoolExecutor to run distributed test
    results = mpi_pool_executor.map(
        _distributed_worker_function,
        *zip(*[(
            world_size,
            strategy,
        )] * world_size),
    )
    for r in results:
        assert r is True
