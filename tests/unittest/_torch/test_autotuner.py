from typing import List

import torch

import tensorrt_llm._torch.autotuner as autotuner
from tensorrt_llm._torch.autotuner import (AutoTuner, DynamicDim, FakeTensor,
                                           OptimizationProfile, StaticDim,
                                           TunableRunner, TuningConfig,
                                           autotune)
from tensorrt_llm._torch.utils import (get_power_of_2_num_tokens_buckets,
                                       next_positive_power_of_2)
from tensorrt_llm.bindings.internal.runtime import delay_kernel
from tensorrt_llm.logger import logger


def test_multi_dynamic_dims():
    tuner = autotuner.AutoTuner()
    x = torch.rand([5, 1024])
    w = torch.rand([7, 19])
    dynamic_tensors = {
        0: {
            0: ([1, 3, 5], lambda x: x // 2),
            1: ([16, 24, 1024], lambda x: x // 2),
        },
        1: {
            1: ([3, 7, 9], lambda x: x // 2)
        }
    }
    profiles = tuner._optimization_profiles(dynamic_tensors,
                                            constraints={},
                                            inputs=[x, w])
    assert len(profiles) == 27
    sample_0 = OptimizationProfile(shapes=[[
        DynamicDim(min=0, opt=1, max=5),
        DynamicDim(min=0, opt=16, max=1024)
    ], [StaticDim(val=7), DynamicDim(min=0, opt=3, max=19)]])
    sample_26 = OptimizationProfile(shapes=[[
        DynamicDim(min=0, opt=5, max=5),
        DynamicDim(min=0, opt=1024, max=1024)
    ], [StaticDim(val=7), DynamicDim(min=0, opt=9, max=19)]])
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
        delay_kernel(1000, torch.cuda.current_stream())
    return x @ w


def gemm_1(x, w):
    if x.shape[0] <= M // 2:
        delay_kernel(1000, torch.cuda.current_stream())
    return x @ w


def gemm_fallback(x, w) -> torch.Tensor:
    # always the slowest
    delay_kernel(100000, torch.cuda.current_stream())
    return x @ w


def check_gemm_tactic_valid(tactic: int, m: int) -> bool:
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

    def get_valid_tactics(self, inputs: List[FakeTensor]) -> List[int]:
        # The simulated delay is not deterministic, so we need to return specific tactics here
        return [-1, 0, 1]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1) -> torch.Tensor:
        assert tactic in [-1, 0, 1]
        return [gemm_0, gemm_1, gemm_fallback][tactic](*inputs)


@torch.library.custom_op("autotuner_test::get_best_gemm_tactic",
                         mutates_args=())
def get_best_gemm_tactic(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    runners = [GemmRunner()]
    tunner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensors={
        0: {
            0: (get_power_of_2_num_tokens_buckets, next_positive_power_of_2),
        },
    })
    runner, tactic = tunner.choose_one(
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
    with autotune():
        torch.ops.autotuner_test.get_best_gemm_tactic(torch.randn(M, 64), w)

    m = M * 2
    while m >= 1:
        best_tactic = torch.ops.autotuner_test.get_best_gemm_tactic(
            torch.randn(m, 64), w)
        check_gemm_tactic_valid(best_tactic, m)
        m //= 2


def test_autotuner_try_block():

    class PartialCrashedRunner(TunableRunner):

        def get_valid_tactics(self, inputs: List[FakeTensor]) -> List[int]:
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
    tunner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensors={
        0: {
            0: (get_power_of_2_num_tokens_buckets, next_positive_power_of_2),
        },
    })
    with autotune():
        runner, tactic = tunner.choose_one("test_autotuner_try_block", runners,
                                           tuning_config, [x, w])

    m = M // 2
    while m >= 1:
        _, tactic = tunner.choose_one("test_autotuner_try_block", runners,
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

    def get_valid_tactics(self, inputs: List[FakeTensor]) -> List[int]:
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

    tuning_config = TuningConfig(dynamic_tensors={
        0: {
            0: (get_power_of_2_num_tokens_buckets, next_positive_power_of_2),
        },
    })

    # Do tuning
    with autotune():
        tuner = AutoTuner.get()
        runner_a, tactic_a = tuner.choose_one("test_multiple_runners", runners,
                                              tuning_config, [x, w])

        # Verify different cache keys are generated
        cache_key_0 = tuner.get_cache_key(
            "test_multiple_runners", runner_0,
            tuner._find_nearest_profile(tuning_config.dynamic_tensors, {},
                                        [x, w]))
        cache_key_1 = tuner.get_cache_key(
            "test_multiple_runners", runner_1,
            tuner._find_nearest_profile(tuning_config.dynamic_tensors, {},
                                        [x, w]))

        assert cache_key_0 != cache_key_1, "Runners with different attributes should have different cache keys"


def test_multiple_dynamic_shapes_cache():
    """Test that different dynamic shape combinations are properly cached"""
    w = torch.randn(64, 128)
    runners = [GemmRunner()]

    # Define dynamic ranges for both dimensions
    tuning_config = TuningConfig(
        dynamic_tensors={
            0: {
                0: ([3, 4, 5], lambda x: x),  # First dim: 3 values
            },
            1: {
                1: ([64, 128, 256, 512], lambda x: x),  # Second dim: 4 values
            }
        })

    # Do tuning with a sample input
    x = torch.randn(3, 64)
    with autotune():
        tuner = AutoTuner.get()
        runner, tactic = tuner.choose_one("test_multiple_dynamic_shapes",
                                          runners, tuning_config, [x, w])

        # Verify cache size - should have 12 entries (3x4 combinations)
        cache_entries = [
            k for k in tuner.profiling_cache.keys()
            if k[0] == "test_multiple_dynamic_shapes"
        ]
        assert len(cache_entries) == 12, \
            f"Expected 12 cache entries for 3x4 shape combinations, got {len(cache_entries)}"


def test_autotuner_statistics():
    """Test that AutoTuner properly collects and reports statistics"""
    # Reset statistics before test
    AutoTuner.get().reset_statistics()

    # Setup test data
    w = torch.randn(64, 128)
    x_large = torch.randn(M * 2, 64)  # Will use fallback
    x_medium = torch.randn(M, 64)  # Will use tactic 1

    # First do tuning with largest input
    with autotune():
        # Only size <= M will be tuned
        torch.ops.autotuner_test.get_best_gemm_tactic(x_medium, w)

    # Generate a cache miss
    torch.ops.autotuner_test.get_best_gemm_tactic(x_large, w)

    # Get statistics
    stats = AutoTuner.get().stats

    # Check cache misses during tuning
    assert stats.cache_misses == 1, "Should have exact one cache misses"

    # Check that we collected profile configs
    op_name = "autotuner_test::get_best_gemm_tactic"
    assert op_name in stats.cache_miss_config_collection, "Should have collected configs for the operation"
    assert len(stats.cache_miss_config_collection[op_name]
               ) == 1, "Should have exactly one profile config"
    assert next(iter(stats.cache_miss_config_collection[op_name])) == (
        x_large.shape, w.shape), "Should have the correct missed profile config"

    # Reset and verify statistics are cleared
    AutoTuner.get().reset_statistics()
    stats = AutoTuner.get().stats
    assert stats.cache_misses == 0, "Statistics should be reset"
    assert len(stats.cache_miss_config_collection
               ) == 0, "Config collection should be empty after reset"
    assert len(stats.tuned_op_total_configs
               ) == 0, "Operation statistics should be reset"
