import os
import tempfile
from typing import Dict, List

import torch

import tensorrt_llm._torch.autotuner as autotuner
from tensorrt_llm._torch.autotuner import (AutoTuner, DynamicDim,
                                           DynamicTensorSpec, FakeTensor,
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
    dynamic_tensor_specs = (
        DynamicTensorSpec(0, 0, [1, 3, 5]),
        DynamicTensorSpec(0, 1, [16, 24, 1024]),
        DynamicTensorSpec(1, 1, [3, 7, 9], lambda x: x // 2),
    )

    profiles = tuner._optimization_profiles(
        tuning_config=TuningConfig(dynamic_tensor_specs=dynamic_tensor_specs),
        inputs=[x, w])
    # choice(0, 0) * choice(0, 1) * choice(1, 1)
    # 3 * 3 * 3 = 27, because 19 is mapped to 9 and already inside the bucket
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
        delay_kernel(10000, torch.cuda.current_stream())
    return x @ w


def gemm_1(x, w):
    if x.shape[0] <= M // 2:
        delay_kernel(10000, torch.cuda.current_stream())
    return x @ w


def gemm_fallback(x, w) -> torch.Tensor:
    # always the slowest
    delay_kernel(100000, torch.cuda.current_stream())
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
    tunner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )
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
    AutoTuner.get().clear_cache()
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
    tunner = AutoTuner.get()
    tuning_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        input_idx=0,
        dim_idx=0,
        gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
        map_to_tuning_buckets=next_positive_power_of_2), ), )
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
    with autotune(cache_path=os.path.join(temp_dir.name,
                                          "test_multiple_dynamic_shapes.json")):
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
    AutoTuner.get().profiling_cache.load_cache(
        os.path.join(temp_dir.name, "test_multiple_dynamic_shapes.rank0.json"))
    cache_entries = tuner.profiling_cache.get_specific_custom_op(
        "test_multiple_dynamic_shapes")

    assert len(cache_entries) == 12, \
        f"Expected 12 cache entries for 3x4 shape combinations, got {len(cache_entries)}"


class GemmRunnerComplexTuningConfigs(TunableRunner):
    valid_tactic_ids = [-1, 0, 1]
    tune_max_num_tokens = 32

    def get_valid_tactics(
        self,
        inputs: List[FakeTensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Dict[str, int]]:
        # During the tuning process, we verify if the tuning config behaves as expected

        assert inputs[0].shape[0] <= self.tune_max_num_tokens, \
            f"Input shape {inputs[0].shape[0]} is larger than the max num tokens {self.tune_max_num_tokens}"

        assert inputs[0][-1, 0] == inputs[0].shape[0], \
            f"Input shape {inputs[0].shape[0]} is not set through the pre_hook correctly"

        # The simulated delay is not deterministic, so we need to return specific tactics here
        return [{
            "block_size": block_size,
            "tactic_id": tactic_id
        } for tactic_id in self.valid_tactic_ids for block_size in [128, 256]]

    def forward(
        self,
        /,
        inputs: List[torch.Tensor],
        *,
        tactic: dict = {},
    ) -> torch.Tensor:
        # Notice that in fallback case tactic is -1
        if tactic == -1:
            # assign default configs for fallback case
            block_size, tactic_id = 128, -1
        else:
            block_size, tactic_id = tactic["block_size"], tactic["tactic_id"]
        assert tactic_id in self.valid_tactic_ids
        return [gemm_0, gemm_1, gemm_fallback][tactic_id](*inputs)

    @staticmethod
    def inputs_pre_hook(inputs: List[torch.Tensor]):
        # always set the first element to bo iota in x
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
    )
    with autotune():
        tuner = AutoTuner.get()
        runner, tactic = tuner.choose_one("test_autotuner_tactic_configs",
                                          runners, tuning_config, [x, w])

    runner_0.forward(inputs=[x, w], tactic=tactic)


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
