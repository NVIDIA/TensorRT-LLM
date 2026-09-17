# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import itertools
import json
import math
import os
import pickle
import statistics
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

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture
def nvmmh_config_guard():
    """Restore the process-wide NVMMH policy after each test that changes it."""
    tuner = AutoTuner.get()
    previous = tuner.nvmmh_config
    try:
        yield tuner
    finally:
        tuner.configure_nvmmh(previous)


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


def test_load_cache_skips_non_literal_tactic():
    """Regression: a non-literal tactic repr must be skipped on load, not crash it.

    ``_deserialize_cache_data`` reconstructs tactics with ``ast.literal_eval``,
    which raises ``SyntaxError`` on non-literal reprs (e.g. enum tactic reprs,
    until #16782 serializes enums by value). It must skip such entries -- once
    ``SyntaxError`` was uncaught and had no ``continue``, crashing the load.
    """
    import ast

    class _NonLiteralTactic:

        def __repr__(self):
            return "<_NonLiteralTactic object nvfp4>"

    poisoned_repr = repr(_NonLiteralTactic())  # non-literal object repr
    # Precondition: confirm this repr really does raise SyntaxError.
    with pytest.raises(SyntaxError):
        ast.literal_eval(poisoned_repr)

    cache = AutoTuner.get().profiling_cache
    cache.clear()
    good_key = "('op_good', 'R', '0', ((1, 128),))"
    bad_key = "('op_bad', 'R', '0', ((2, 128),))"
    doc = {
        "metadata": cache._serialize_metadata(),
        "shared": {},
        "rank_0": {
            good_key: {
                "runner_id": 0,
                "tactic": "7",
                "min_time": 0.001
            },
            bad_key: {
                "runner_id": 1,
                "tactic": poisoned_repr,
                "min_time": 0.002
            },
        },
    }
    temp_dir = tempfile.TemporaryDirectory()
    cache_path = os.path.join(temp_dir.name, "poisoned_cache.json")
    with open(cache_path, "w") as f:
        json.dump(doc, f)

    # Must not raise (previously raised SyntaxError out of load_cache).
    cache.load_cache(cache_path, rank=0)

    # The literal-safe entry survived with its exact tactic ...
    good = ("op_good", "R", "0", ((1, 128), ))
    assert good in cache.cache
    assert cache.cache[good][1] == 7
    # ... and the non-literal entry was skipped, not silently mis-decoded.
    bad = ("op_bad", "R", "0", ((2, 128), ))
    assert bad not in cache.cache


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


def test_kernel_testing_uses_runner_valid_tactics():
    """Verify captured tactic combinations use the runner-provided candidate list."""

    class CaptureRunner(TunableRunner):

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            """Provide a fixed candidate list for capture and replay assertions."""
            return [1, 2]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1,
                    **kwargs) -> torch.Tensor:
            """Return the input unchanged while the test observes tactic selection."""
            assert tactic in [-1, 0, 1, 2]
            return inputs[0]

    x = torch.randn(4, 4)
    runners = [CaptureRunner()]
    tuning_config = TuningConfig()
    tuner = AutoTuner.get()
    tuner.clear_cache()

    with tuner.capture() as all_tactics:
        tuner.choose_one("test_runner_valid_capture", runners, tuning_config,
                         [x])

    captured = list(all_tactics)
    assert [tactic for ((_, tactic), ) in captured] == [1, 2]

    for ((runner, tactic), ) in captured:
        with tuner.replay(((runner, tactic), )):
            replay_runner, replay_tactic = tuner.choose_one(
                "test_runner_valid_capture", runners, tuning_config, [x])
            assert replay_runner is runner
            assert replay_tactic == tactic


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

    # Check only one cache file is created in the cache path.
    # The sibling ".lock" file is an implementation artifact of
    # _exclusive_cache_lock (see tensorrt_llm/_torch/autotuner.py) and is not
    # a per-rank cache file.
    cache_dir = os.path.dirname(cache_path)
    cache_files = [f for f in os.listdir(cache_dir) if not f.endswith(".lock")]
    assert len(cache_files) == 1, "Only one rank file should be created"

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


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_global_timer_vs_cuda_event(use_cuda_graph, monkeypatch):
    """Verify globaltimer and cuda-event backends are statistically indistinguishable."""

    class PureGemmRunner(TunableRunner):

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [0]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = 0,
                    **kwargs) -> torch.Tensor:
            assert tactic == 0
            return inputs[0] @ inputs[1]

    # Keep full profiling repeats enabled to reduce measurement noise.
    monkeypatch.setenv("TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE", "1")

    gemm_shapes = [
        (256, 4096, 11008),
        (512, 8192, 8192),
    ]
    num_trials = 6
    rel_tol = 0.05
    stat_zscore = 3.0
    abs_tol_ms = 0.01

    runner = PureGemmRunner()
    tuning_config = TuningConfig(use_cuda_graph=use_cuda_graph)
    tuner = AutoTuner()
    trial_rows = []

    for m, k, n in gemm_shapes:
        x = torch.randn(m, k, device='cuda', dtype=torch.float16)
        w = torch.randn(k, n, device='cuda', dtype=torch.float16)

        event_times = []
        gt_times = []

        # Interleave both backends to avoid drift effects from neighboring load.
        for _ in range(num_trials):
            tuner._use_global_timer = False
            event_times.append(
                tuner._profile_single_kernel(
                    runner=runner,
                    inputs=[x, w],
                    tactic=0,
                    tuning_config=tuning_config,
                    use_cuda_graph=use_cuda_graph,
                ))

            tuner._use_global_timer = True
            gt_times.append(
                tuner._profile_single_kernel(
                    runner=runner,
                    inputs=[x, w],
                    tactic=0,
                    tuning_config=tuning_config,
                    use_cuda_graph=use_cuda_graph,
                ))

            event_ms = event_times[-1]
            gt_ms = gt_times[-1]
            abs_diff = abs(gt_ms - event_ms)
            rel_diff = abs_diff / event_ms if event_ms > 0 else float('inf')
            trial_rows.append((m, k, n, len(event_times), event_ms, gt_ms,
                               abs_diff, rel_diff))

        event_mean = statistics.fmean(event_times)
        gt_mean = statistics.fmean(gt_times)
        event_var = statistics.variance(event_times)
        gt_var = statistics.variance(gt_times)
        mean_diff = abs(gt_mean - event_mean)
        rel_diff = mean_diff / event_mean

        # Two-sample mean delta should be small vs a fixed tolerance and
        # indistinguishable within sampling noise.
        combined_sem = math.sqrt(event_var / num_trials + gt_var / num_trials)
        allowed_diff = max(abs_tol_ms, rel_tol * event_mean,
                           stat_zscore * combined_sem)

        assert event_mean > 0, (
            f"({m},{k},{n}): cuda event mean should be positive, got {event_mean}"
        )
        assert gt_mean > 0, (
            f"({m},{k},{n}): globaltimer mean should be positive, got {gt_mean}"
        )
        assert mean_diff <= allowed_diff, (
            f"({m},{k},{n}): timing backends are distinguishable "
            f"(cuda_event_mean={event_mean:.4f}ms, "
            f"globaltimer_mean={gt_mean:.4f}ms, "
            f"relative_diff={rel_diff * 100:.2f}%, "
            f"allowed_diff={allowed_diff:.4f}ms, "
            f"combined_sem={combined_sem:.4f}ms, "
            f"event_samples={event_times}, gt_samples={gt_times})")

    # Visible with `pytest -s`; otherwise captured by pytest.
    print("\nGlobaltimer vs cuda-event trial comparison")
    print(f"cuda_graph={use_cuda_graph}, trials_per_shape={num_trials}")
    print("-" * 102)
    print(
        f"{'shape (M,K,N)':>21} {'trial':>5} {'cuda_event (ms)':>16} "
        f"{'globaltimer (ms)':>17} {'abs diff (ms)':>14} {'rel diff (%)':>13}")
    print("-" * 102)
    for m, k, n, trial, event_ms, gt_ms, abs_diff, rel_diff in trial_rows:
        print(f"{f'({m},{k},{n})':>21} {trial:>5d} "
              f"{event_ms:>16.4f} {gt_ms:>17.4f} "
              f"{abs_diff:>14.4f} {rel_diff * 100:>13.2f}")
    print("-" * 102)


def _make_shapes(*sizes):
    """Convert size-tuples into Tuple[torch.Size, ...] for _find_nearest_profile."""
    return tuple(torch.Size(s) for s in sizes)


class TestSpecBoundsChecking:
    """Bounds-checking in AutoTuner._find_nearest_profile and AutoTuner._optimization_profiles."""

    def setup_method(self):
        AutoTuner._find_nearest_profile.cache_clear()

    @pytest.mark.parametrize("entry", ["find_nearest", "optimization_profiles"])
    @pytest.mark.parametrize(
        "spec_class,input_idx,dim_idx",
        [
            pytest.param("dynamic", 5, 0, id="dynamic_input_idx_out_of_range"),
            pytest.param("dynamic", 0, 10, id="dynamic_dim_idx_out_of_range"),
            pytest.param("dynamic", -1, 0, id="dynamic_negative_input_idx"),
            pytest.param("dynamic", 0, -1, id="dynamic_negative_dim_idx"),
            pytest.param(
                "constraint", 3, 0, id="constraint_input_idx_out_of_range"),
            pytest.param(
                "constraint", 0, 7, id="constraint_dim_idx_out_of_range"),
            pytest.param(
                "constraint", -1, 0, id="constraint_negative_input_idx"),
            pytest.param("constraint", 0, -1, id="constraint_negative_dim_idx"),
        ],
    )
    def test_oob_spec_skipped(self, entry, spec_class, input_idx, dim_idx):
        from tensorrt_llm._torch.autotuner import ConstraintSpec
        if spec_class == "dynamic":
            spec = DynamicTensorSpec(input_idx=input_idx,
                                     dim_idx=dim_idx,
                                     gen_tuning_buckets=(1, 2))
            dyn_specs = (spec, )
            con_specs = ()
        else:
            spec = ConstraintSpec(input_idx=input_idx,
                                  dim_idx=dim_idx,
                                  infer_shape=lambda shapes: 1)
            dyn_specs = ()
            con_specs = (spec, )

        if entry == "find_nearest":
            shapes = _make_shapes([4, 8])
            result = AutoTuner._find_nearest_profile(
                shapes,
                dynamic_tensor_specs=dyn_specs,
                constraint_specs=con_specs)
            assert result == ((4, 8), )
        else:
            tuner = AutoTuner()
            x = torch.rand([4, 8])
            # Constraint-only configs need a dynamic spec to drive the cartesian product.
            if not dyn_specs:
                dyn_specs = (DynamicTensorSpec(input_idx=0,
                                               dim_idx=0,
                                               gen_tuning_buckets=(1, )), )
            config = TuningConfig(dynamic_tensor_specs=dyn_specs,
                                  constraint_specs=con_specs)
            profiles = tuner._optimization_profiles(config, [x])
            # OOB spec skipped — profile generation still produces at least one profile.
            assert len(profiles) >= 1


def test_single_pair_shortcut(monkeypatch):
    """Single (runner, tactic) candidate must bypass the timed profile loop.

    When ``_profile_runners`` sees exactly one (runner, tactic) pair, it
    must (1) skip ``_profile_single_kernel`` entirely, (2) still fire the
    ``do_preparation`` hook for runners that opt in, (3) fire exactly one
    ``forward()`` to drive any JIT side effect, and (4) record the pair
    in the profiling cache. Multi-tactic ops in the same fixture must
    still use the timed path.
    """

    profile_calls: List[Any] = []

    def _track(self, runner, inputs, tactic, tuning_config, **kwargs):
        profile_calls.append(tactic)
        return 1.0 + len(profile_calls) * 0.01

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", _track)

    forward_calls: List[tuple] = []

    class PrepRunner(TunableRunner):

        def unique_id(self):
            return ()

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [0]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1,
                    do_preparation: bool = False,
                    **kwargs) -> torch.Tensor:
            forward_calls.append((tactic, do_preparation))
            if do_preparation:
                return None
            x, w = inputs
            return x @ w

    tuner = AutoTuner.get()
    tuner.clear_cache()
    x = torch.randn(M, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")

    # Single (runner, tactic): shortcut must fire.
    op_single = "autotuner_test::single_pair_shortcut"
    with autotune():
        _, tactic = tuner.choose_one(op_single, [PrepRunner()], TuningConfig(),
                                     [x, w])
    assert tactic == 0
    assert profile_calls == [], (
        f"_profile_single_kernel must not be called for single-pair op; "
        f"got {profile_calls}")
    assert forward_calls == [
        (-1, True), (0, False)
    ], (f"Expected do_preparation then exactly one forward(tactic=0); "
        f"got {forward_calls}")
    assert len(tuner.profiling_cache.get_specific_custom_op(op_single)) == 1, (
        "single-pair shortcut must still record the (runner, tactic) entry")

    # Multi-tactic on the same fixture: timed profile path must still run.
    forward_calls.clear()
    op_multi = "autotuner_test::single_pair_shortcut_multi"
    with autotune():
        tuner.choose_one(op_multi, [GemmRunner()], TuningConfig(), [x, w])
    # GemmRunner exposes 3 tactics -> 3 profile calls.
    assert len(profile_calls) == 3, (
        f"Multi-tactic op must hit _profile_single_kernel per tactic; "
        f"got {len(profile_calls)} ({profile_calls})")


_CUPTI_PREFLIGHT_STATE = {}
_CUTE_DSL_NVMMH_TEST_MNK = (16, 256, 7168)


def _skip_if_cupti_unavailable(error, context):
    """Skip CUPTI-only perf tests without hiding kernel/runtime failures."""
    if not AutoTuner._is_torch_profiler_unavailable_error(error):
        raise error
    reason = f"{context} requires working torch.profiler/CUPTI: {error}"
    _CUPTI_PREFLIGHT_STATE[torch.cuda.current_device()] = reason
    pytest.skip(reason)


def _require_cupti_cuda_activity():
    """Require torch.profiler to report positive CUDA activity on this GPU."""
    device_index = torch.cuda.current_device()
    if AutoTuner._torch_profiler_unavailable:
        pytest.skip(
            "CUPTI-only performance comparison cannot use the process-wide "
            "CUDA-event fallback")

    cached_state = _CUPTI_PREFLIGHT_STATE.get(device_index)
    if cached_state is True:
        return
    if isinstance(cached_state, str):
        pytest.skip(cached_state)

    # Warm CUDA before starting the profiler so the probe tests CUPTI activity,
    # not first-use CUDA initialization. A CUDA/kernel failure here is real and
    # intentionally remains a test failure.
    probe = torch.ones(1, device="cuda")
    probe.add_(1)
    torch.cuda.synchronize()
    try:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                acc_events=True,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
        ) as prof:
            probe.add_(1)
            torch.cuda.synchronize()
        AutoTuner._torch_profiler_elapsed_time_ms(prof, repeat=1)
    except Exception as error:
        _skip_if_cupti_unavailable(error, "CuTe DSL performance test")

    _CUPTI_PREFLIGHT_STATE[device_index] = True


def _require_cupti_autotuner_entry(tuner, name, expected_candidates):
    """Return a measured cache entry only when it was timed with CUPTI."""
    assert len(expected_candidates) > 1, (
        "CUPTI cache-time comparison requires multiple candidates; the "
        "AutoTuner single-pair shortcut records an unmeasured 0.0 entry")
    cache_entries = tuner.profiling_cache.get_specific_custom_op(name)
    assert len(cache_entries) == 1
    cache_key, cache_value = next(iter(cache_entries.items()))
    _, cached_tactic, profile_ms = cache_value
    timer_key = cache_key[-1]
    if (AutoTuner._torch_profiler_unavailable
            or timer_key == "cuda_event_fallback"):
        pytest.skip(
            f"CUPTI became unavailable while profiling {name}; refusing to "
            "compare CUDA-event fallback timings")
    assert timer_key == "torch_profiler", (
        f"Expected CUPTI timing for {name}, got timer key {timer_key!r}")
    assert math.isfinite(profile_ms) and profile_ms > 0.0, (
        f"Expected a measured positive CUPTI time for {name}, got {profile_ms}")
    return cached_tactic, profile_ms * 1000.0


def _choose_cupti_autotuner_tactic(tuner, name, runner, tuning_config, inputs,
                                   expected_candidates):
    """Choose one tactic and return its total CUPTI time in microseconds."""
    assert len(expected_candidates) > 1, (
        "performance comparison requires a measured tactic sweep")
    tuner.clear_cache()
    with autotune(skip_dynamic_tuning_buckets=True):
        _, tactic = tuner.choose_one(name, [runner], tuning_config, inputs)
    cached_tactic, profile_us = _require_cupti_autotuner_entry(
        tuner, name, expected_candidates)
    assert cached_tactic == tactic
    return tactic, profile_us


def _profile_cupti_total_us(fn, context, warmup=20, iterations=200, trials=5):
    """Return median per-call device time, including every CUDA activity."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        try:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    acc_events=True,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
            ) as prof:
                for _ in range(iterations):
                    fn()
                torch.cuda.synchronize()
            elapsed_ms = AutoTuner._torch_profiler_elapsed_time_ms(
                prof, iterations)
        except Exception as error:
            _skip_if_cupti_unavailable(error, context)
        samples.append(elapsed_ms * 1000.0)
    return statistics.median(samples)


def _nvmmh_tactic_family(tactic):
    """Strip model-selected scheduler fields before comparing structural families."""
    # Scheduler-only guidance annotates validated families with model-selected
    # raster/swizzle values; the resulting tuples need not occur in the sweep.
    if tactic[0] == "mixed_clusters":
        return tactic[:7]
    if tactic[0] == "base":
        return tactic[:5] if isinstance(tactic[1], bool) else tactic[:6]
    return tactic[:5]


def _run_cute_dsl_bf16_heuristic_comparison(monkeypatch, tuner, nvmmh):
    """Compare BF16 full sweep, scheduler-only NVMMH, and cuBLASLt."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    m, n, k = _CUTE_DSL_NVMMH_TEST_MNK
    torch.manual_seed(2028)
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    output = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    inputs = [act, weight, output]

    runner = cute_dsl_custom_ops.CuteDSLBf16RubinGemmRunner(
        use_tvm_ffi=True, output_dtype=torch.bfloat16)
    runner.__class__.kernel_cache.clear()
    runner.__class__.split_k_gemm_cache.clear()
    tuning_config = runner.__class__.tuning_config
    monkeypatch.setattr(tuning_config, "use_cuda_graph", False)

    scheduler_fields = ("swizzle", "cta_order", "split_k")
    tuner.configure_nvmmh(enabled=False, fields=scheduler_fields, max_tactics=5)
    baseline_tactics = runner.get_valid_tactics(inputs, None)
    assert baseline_tactics
    sweep_tactic, sweep_us = _choose_cupti_autotuner_tactic(
        tuner,
        "test::cute_dsl_bf16_exact_full_sweep",
        runner,
        tuning_config,
        inputs,
        baseline_tactics,
    )

    tuner.configure_nvmmh(enabled=True, fields=scheduler_fields, max_tactics=5)
    heuristic_candidates = runner.get_valid_tactics(inputs, None)
    assert 0 < len(heuristic_candidates) <= len(baseline_tactics), (
        "nvMMH did not prune SM107 BF16 tactics: "
        f"{len(heuristic_candidates)} vs {len(baseline_tactics)}")
    assert {_nvmmh_tactic_family(t)
            for t in heuristic_candidates
            }.issubset({_nvmmh_tactic_family(t)
                        for t in baseline_tactics})

    model_configs = nvmmh.rank_configs(
        m,
        n,
        k,
        nvmmh.BF16_PRECISION,
        max(tuner.nvmmh_config.max_tactics * 16, 64),
        layout_name=nvmmh.BF16_LAYOUT,
    )
    assert model_configs
    native_rank1_split_k = int(model_configs[0].split_k)

    def _split_k(tactic):
        """Read the tactic split factor, defaulting unsplit variants to one."""
        if (isinstance(tactic, tuple) and tactic and tactic[0] == "base"
                and len(tactic) >= 6):
            return int(tactic[5])
        return 1

    candidate_splits = {_split_k(tactic) for tactic in heuristic_candidates}
    expected_local_splits = {
        split_k
        for split_k in (1, 2, 4, 8) if nvmmh.is_sm107_nvmmh_split_k_eligible(
            k, runner.nvmmh_split_k_cta_k, split_k)
    }
    assert expected_local_splits.issubset(candidate_splits), (
        "SM107 BF16 scheduler-only filtering removed locally admitted "
        f"split-K candidates: expected={expected_local_splits}, "
        f"actual={candidate_splits}")
    assert all(
        nvmmh.is_sm107_nvmmh_split_k_eligible(k, runner.nvmmh_split_k_cta_k,
                                              split_k)
        for split_k in candidate_splits), (
            f"SM107 BF16 retained an ineligible split: {candidate_splits}")
    baseline_base_families = {
        tactic[:5]
        for tactic in baseline_tactics
        if isinstance(tactic, tuple) and tactic and tactic[0] == "base"
    }
    heuristic_base_families = {
        tactic[:5]
        for tactic in heuristic_candidates
        if isinstance(tactic, tuple) and tactic and tactic[0] == "base"
    }
    assert heuristic_base_families == baseline_base_families, (
        "scheduler-only BF16 NVMMH must keep every base tile/cluster family")
    heuristic_tactic, heuristic_us = _choose_cupti_autotuner_tactic(
        tuner,
        "test::cute_dsl_bf16_scheduler_only_heuristic",
        runner,
        tuning_config,
        inputs,
        heuristic_candidates,
    )

    def _cublaslt_call():
        """Execute the cuBLASLt reference used for BF16 timing comparisons."""
        return torch.ops.trtllm.cublas_mm(
            act,
            weight.t(),
            bias=None,
            out_dtype=None,
        )

    output.fill_(torch.nan)
    runner(inputs, tactic=sweep_tactic)
    sweep_output = output.clone()
    output.fill_(torch.nan)
    runner(inputs, tactic=heuristic_tactic)
    heuristic_output = output.clone()
    cublaslt_output = _cublaslt_call()
    torch.cuda.synchronize()
    reference = act.float() @ weight.t().float()
    torch.testing.assert_close(sweep_output.float(),
                               reference,
                               rtol=1e-2,
                               atol=1.0)
    torch.testing.assert_close(heuristic_output.float(),
                               reference,
                               rtol=1e-2,
                               atol=1.0)
    torch.testing.assert_close(cublaslt_output.float(),
                               reference,
                               rtol=1e-2,
                               atol=1.0)

    # Use one CUPTI whole-call timer for the three selected implementations.
    # This counts the reduction kernel when a CuTe DSL tactic uses split-K.
    sweep_cupti_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=sweep_tactic),
        "BF16 full-sweep tactic comparison",
    )
    heuristic_cupti_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=heuristic_tactic),
        "BF16 scheduler-only nvMMH tactic comparison",
    )
    cublaslt_cupti_us = _profile_cupti_total_us(
        _cublaslt_call,
        "BF16 cuBLASLt comparison",
    )

    print(f"\nSM107 BF16 [{m},{k}] x [{k},{n}] -> [{m},{n}]")
    print(f"full sweep: candidates={len(baseline_tactics)}, "
          f"tactic={sweep_tactic}, autotuner_cupti_time={sweep_us:.3f} us, "
          f"comparison_cupti_time={sweep_cupti_us:.3f} us")
    print(f"scheduler-only heuristic: "
          f"candidates={len(heuristic_candidates)}, "
          f"native_rank1_split_k={native_rank1_split_k}, "
          f"candidate_splits={sorted(candidate_splits)}, "
          f"tactic={heuristic_tactic}, "
          f"autotuner_cupti_time={heuristic_us:.3f} us, "
          f"comparison_cupti_time={heuristic_cupti_us:.3f} us")
    print(f"cuBLASLt: comparison_cupti_time={cublaslt_cupti_us:.3f} us")

    tolerance = 1.1
    cublas_tolerance = 1.1
    assert heuristic_us <= sweep_us * tolerance, (
        f"BF16 heuristic tactic {heuristic_tactic} ({heuristic_us:.3f} us) "
        f"is >{tolerance:.2f}x slower than full-sweep tactic "
        f"{sweep_tactic} ({sweep_us:.3f} us)")
    assert heuristic_cupti_us <= sweep_cupti_us * tolerance, (
        f"BF16 heuristic tactic {heuristic_tactic} "
        f"({heuristic_cupti_us:.3f} us CUPTI) is >{tolerance:.2f}x slower "
        f"than full-sweep tactic {sweep_tactic} "
        f"({sweep_cupti_us:.3f} us CUPTI)")
    assert heuristic_cupti_us <= cublaslt_cupti_us * cublas_tolerance, (
        f"BF16 CuTe DSL heuristic tactic {heuristic_tactic} "
        f"({heuristic_cupti_us:.3f} us CUPTI) is "
        f">{cublas_tolerance:.2f}x slower "
        f"than cuBLASLt ({cublaslt_cupti_us:.3f} us CUPTI)")


def _run_cute_dsl_mxfp8_heuristic_comparison(monkeypatch, tuner, nvmmh):
    """Compare MXFP8 full sweep with scheduler-only NVMMH on SM107."""
    from _torch.helpers import calc_diff, per_block_cast_to_fp8_e8m0

    import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    m, n, k = _CUTE_DSL_NVMMH_TEST_MNK
    torch.manual_seed(2029)
    act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    weight = (torch.randn(n, k, dtype=torch.bfloat16, device="cuda") /
              math.sqrt(k))
    act_fp8, act_sf = \
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(act)
    weight_fp8, weight_sf_k128 = per_block_cast_to_fp8_e8m0(weight)
    weight_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(
        weight_sf_k128, mn=n, k=k)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [act_fp8, weight_fp8, act_sf, weight_sf, alpha]

    runner = cute_dsl_custom_ops.CuteDSLMXFP8RubinLinear(
        output_dtype=torch.bfloat16, use_tvm_ffi=True)
    runner.__class__.kernel_cache.clear()
    tuning_config = runner.__class__.tuning_config
    monkeypatch.setattr(tuning_config, "use_cuda_graph", False)

    scheduler_fields = ("swizzle", "cta_order", "split_k")
    tuner.configure_nvmmh(enabled=False, fields=scheduler_fields, max_tactics=5)
    baseline_tactics = runner.get_valid_tactics(inputs, None)
    assert baseline_tactics
    sweep_tactic, sweep_us = _choose_cupti_autotuner_tactic(
        tuner,
        "test::cute_dsl_mxfp8_exact_full_sweep",
        runner,
        tuning_config,
        inputs,
        baseline_tactics,
    )

    tuner.configure_nvmmh(enabled=True, fields=scheduler_fields, max_tactics=5)
    heuristic_candidates = runner.get_valid_tactics(inputs, None)
    assert 0 < len(heuristic_candidates) <= len(baseline_tactics), (
        "nvMMH did not prune SM107 MXFP8 tactics: "
        f"{len(heuristic_candidates)} vs {len(baseline_tactics)}")

    assert {_nvmmh_tactic_family(t)
            for t in heuristic_candidates
            }.issubset({_nvmmh_tactic_family(t)
                        for t in baseline_tactics})

    def _split_k(tactic):
        """Read the tactic split factor, defaulting unsplit variants to one."""
        if (isinstance(tactic, tuple) and tactic and tactic[0] == "base"
                and len(tactic) >= 9):
            return int(tactic[8])
        return 1

    candidate_splits = {_split_k(tactic) for tactic in heuristic_candidates}
    expected_local_splits = {
        split_k
        for split_k in runner.split_k_candidates
        if nvmmh.is_sm107_nvmmh_split_k_eligible(k, runner.mma_tiler_k, split_k)
    }
    assert expected_local_splits.issubset(candidate_splits), (
        "SM107 MXFP8 scheduler-only filtering removed locally admitted "
        f"split-K candidates: expected={expected_local_splits}, "
        f"actual={candidate_splits}")
    assert all(
        nvmmh.is_sm107_nvmmh_split_k_eligible(k, runner.mma_tiler_k, split_k)
        for split_k in candidate_splits), (
            f"SM107 MXFP8 retained an ineligible split: {candidate_splits}")
    assert 8 in candidate_splits
    baseline_base_families = {
        tactic[:7]
        for tactic in baseline_tactics
        if isinstance(tactic, tuple) and tactic and tactic[0] == "base"
    }
    heuristic_base_families = {
        tactic[:7]
        for tactic in heuristic_candidates
        if isinstance(tactic, tuple) and tactic and tactic[0] == "base"
    }
    assert heuristic_base_families == baseline_base_families, (
        "scheduler-only MXFP8 NVMMH must keep every base tile/cluster family")

    heuristic_tactic, heuristic_us = _choose_cupti_autotuner_tactic(
        tuner,
        "test::cute_dsl_mxfp8_scheduler_only_heuristic",
        runner,
        tuning_config,
        inputs,
        heuristic_candidates,
    )

    def _cublaslt_call():
        """Execute the cuBLASLt reference for the MXFP8 comparison inputs."""
        # PyTorch's Rubin MXFP8 reference dispatches this block-scaled call to
        # cuBLASLt. The CuTe DSL inputs already use its flat R128c4 SF layout.
        return torch._scaled_mm(
            act_fp8,
            weight_fp8.t(),
            scale_a=act_sf.view(torch.float8_e8m0fnu),
            scale_b=weight_sf.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )

    sweep_output = runner(inputs, tactic=sweep_tactic)
    heuristic_output = runner(inputs, tactic=heuristic_tactic)
    cublaslt_output = _cublaslt_call()
    torch.cuda.synchronize()
    reference = act @ weight.t()
    assert calc_diff(sweep_output, reference) < 1e-3
    assert calc_diff(heuristic_output, reference) < 1e-3
    assert calc_diff(cublaslt_output, reference) < 1e-3

    sweep_cupti_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=sweep_tactic),
        "MXFP8 full-sweep tactic comparison",
    )
    heuristic_cupti_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=heuristic_tactic),
        "MXFP8 scheduler-only nvMMH tactic comparison",
    )
    cublaslt_cupti_us = _profile_cupti_total_us(
        _cublaslt_call,
        "MXFP8 cuBLASLt comparison",
    )

    print(f"\nSM107 MXFP8 [{m},{k}] x [{k},{n}] -> [{m},{n}]")
    print(f"full sweep: candidates={len(baseline_tactics)}, "
          f"tactic={sweep_tactic}, autotuner_cupti_time={sweep_us:.3f} us, "
          f"comparison_cupti_time={sweep_cupti_us:.3f} us")
    print(f"scheduler-only heuristic: "
          f"candidates={len(heuristic_candidates)}, "
          f"candidate_splits={sorted(candidate_splits)}, "
          f"tactic={heuristic_tactic}, "
          f"autotuner_cupti_time={heuristic_us:.3f} us, "
          f"comparison_cupti_time={heuristic_cupti_us:.3f} us")
    print(f"cuBLASLt: comparison_cupti_time={cublaslt_cupti_us:.3f} us")

    tolerance = 1.1
    cublas_tolerance = 1.1
    assert heuristic_us <= sweep_us * tolerance, (
        f"MXFP8 heuristic tactic {heuristic_tactic} ({heuristic_us:.3f} us) "
        f"is >{tolerance:.2f}x slower than full-sweep tactic "
        f"{sweep_tactic} ({sweep_us:.3f} us)")
    assert heuristic_cupti_us <= sweep_cupti_us * tolerance, (
        f"MXFP8 heuristic tactic {heuristic_tactic} "
        f"({heuristic_cupti_us:.3f} us CUPTI) is >{tolerance:.2f}x slower "
        f"than full-sweep tactic {sweep_tactic} "
        f"({sweep_cupti_us:.3f} us CUPTI)")
    assert heuristic_cupti_us <= cublaslt_cupti_us * cublas_tolerance, (
        f"MXFP8 CuTe DSL heuristic tactic {heuristic_tactic} "
        f"({heuristic_cupti_us:.3f} us CUPTI) is "
        f">{cublas_tolerance:.2f}x slower "
        f"than cuBLASLt ({cublaslt_cupti_us:.3f} us CUPTI)")


def _run_cute_dsl_nvfp4_heuristic_comparison(monkeypatch, tuner, nvmmh,
                                             sm_version):
    """Compare NVFP4 full sweep, NVMMH pruning, and cuBLASLt."""
    from _torch.helpers import calc_diff

    if sm_version == 107:
        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
            CuteDSLNVFP4RubinLinear as NVFP4Runner
    else:
        from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
            CuteDSLNVFP4BlackwellRunner as NVFP4Runner

    # Keep the NVFP4 operands aligned with the BF16 and MXFP8 subcases. FP4
    # packing / scale-factor layout follows
    # shmoo_nvfp4_cutedsl_heuristics.py::_quantize_inputs.
    m, n, k = _CUTE_DSL_NVMMH_TEST_MNK
    dtype = torch.bfloat16
    sf_vec_size = 16
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtype).cuda()
    w = torch.randn((n, k), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    w_sf_global = (448 * 6) / w.abs().max().float()
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, x_sf_global, sf_vec_size,
                                                False)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_sf_global, sf_vec_size,
                                                False)
    alpha = (1.0 / (x_sf_global * w_sf_global)).reshape(1)
    inputs = [x_fp4, w_fp4, x_sf, w_sf, alpha]

    runner = NVFP4Runner(output_dtype=dtype)
    tuning_config = runner.__class__.tuning_config
    monkeypatch.setattr(tuning_config, "use_cuda_graph", False)
    nvfp4_fields = (("tile", "cluster", "split_k") if sm_version == 107 else
                    ("tile", "cluster"))
    tuner.configure_nvmmh(
        enabled=False,
        fields=nvfp4_fields,
        max_tactics=5,
    )

    def _best_tactic(name, expected_candidates):
        """Tune the supplied candidates and require a CUPTI-ranked cache entry."""
        assert len(expected_candidates) > 1, (
            "NVFP4 performance comparison requires a measured tactic sweep")
        tuner.clear_cache()
        with autotune(skip_dynamic_tuning_buckets=True):
            _, tactic = tuner.choose_one(
                name,
                [runner],
                tuning_config,
                inputs,
            )
        cached_tactic, _ = _require_cupti_autotuner_entry(
            tuner, name, expected_candidates)
        assert cached_tactic == tactic
        return tactic

    baseline_tactics = runner.get_valid_tactics(inputs, None)
    assert baseline_tactics

    # Full sweep: heuristics disabled.
    sweep_tactic = _best_tactic("test::cute_dsl_nvfp4_exact_full_sweep",
                                baseline_tactics)

    # Pruned: nvMatmulHeuristics drives the (coupled) tile+cluster candidates.
    tuner.configure_nvmmh(enabled=True)
    heuristic_candidates = runner.get_valid_tactics(inputs, None)
    assert 0 < len(heuristic_candidates) <= len(baseline_tactics), (
        f"nvMMH did not prune SM{sm_version} NVFP4 tactics: "
        f"{len(heuristic_candidates)} vs {len(baseline_tactics)}")
    assert {_nvmmh_tactic_family(t)
            for t in heuristic_candidates
            }.issubset({_nvmmh_tactic_family(t)
                        for t in baseline_tactics})
    if sm_version == 107:

        def _nvfp4_split_k(tactic):
            """Read a base NVFP4 split factor, treating other families as unsplit."""
            if (isinstance(tactic, tuple) and tactic and tactic[0] == "base"
                    and len(tactic) >= 9):
                return int(tactic[8])
            return 1

        candidate_splits = {
            _nvfp4_split_k(tactic)
            for tactic in heuristic_candidates
        }
        expected_local_splits = {
            split_k
            for split_k in runner.split_k_candidates
            if nvmmh.is_sm107_nvmmh_split_k_eligible(k, runner.mma_tiler_k,
                                                     split_k)
        }
        assert expected_local_splits.issubset(candidate_splits)
        assert all(
            nvmmh.is_sm107_nvmmh_split_k_eligible(
                k, runner.mma_tiler_k, split_k) for split_k in candidate_splits)
    heuristic_tactic = _best_tactic("test::cute_dsl_nvfp4_exact_heuristic",
                                    heuristic_candidates)

    # cuBLASLt runs its own heuristic auto-tuning; warm it under autotune().
    def _cublas_call():
        return torch.ops.trtllm.nvfp4_gemm_cublaslt(x_fp4, w_fp4, x_sf, w_sf,
                                                    alpha, dtype)

    with autotune():
        _cublas_call()
    torch.cuda.synchronize()

    sweep_output = runner(inputs, tactic=sweep_tactic)
    heuristic_output = runner(inputs, tactic=heuristic_tactic)
    cublas_output = _cublas_call()
    torch.cuda.synchronize()
    assert calc_diff(sweep_output, cublas_output) < 1e-3
    assert calc_diff(heuristic_output, cublas_output) < 1e-3

    # Use whole-call CUPTI time so in-place split-K's required output zeroing is
    # included alongside the GEMM, matching the AutoTuner's timing semantics.
    sweep_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=sweep_tactic),
        "NVFP4 full-sweep tactic comparison",
    )
    heuristic_us = _profile_cupti_total_us(
        lambda: runner(inputs, tactic=heuristic_tactic),
        "NVFP4 heuristic tactic comparison",
    )
    cublas_us = _profile_cupti_total_us(
        _cublas_call,
        "NVFP4 cuBLASLt comparison",
    )
    print(f"\nSM{sm_version} NVFP4 [{m},{k}] x [{k},{n}] -> [{m},{n}]")
    print(f"full sweep: candidates={len(baseline_tactics)}, "
          f"tactic={sweep_tactic}, time={sweep_us:.3f} us")
    print(f"heuristic: candidates={len(heuristic_candidates)}, "
          f"tactic={heuristic_tactic}, time={heuristic_us:.3f} us")
    print(f"cuBLASLt: time={cublas_us:.3f} us")

    # Pruning must not degrade the achieved kernel runtime beyond this tolerance.
    # With the default MAX_TACTICS=5 the heuristic set includes the empirical
    # best tile, but its cluster ranking can be slightly off across Rubin nodes,
    # so a 10% bound catches gross regressions while allowing that variance.
    tolerance = 1.1
    assert heuristic_us <= sweep_us * tolerance, (
        f"heuristic-pruned tactic {heuristic_tactic} ({heuristic_us:.2f} us) is "
        f">{tolerance:.2f}x slower than full-sweep tactic {sweep_tactic} "
        f"({sweep_us:.2f} us) for M={m}, N={n}, K={k}")

    # The heuristic CuteDSL kernel should beat cuBLAS or be within tolerance.
    # This is a cross-library comparison (CuteDSL vs cuBLASLt) at ~28us per call
    # profiled over only 20 iterations; run-to-run jitter easily reaches a few
    # percent from DVFS, L2 residency, and interleaving with cuBLAS autotune
    # warmup, so keep this bound looser than the intra-CuteDSL one above.
    cublas_tolerance = 1.10
    assert heuristic_us <= cublas_us * cublas_tolerance, (
        f"CuteDSL heuristic kernel ({heuristic_us:.2f} us) is "
        f">{cublas_tolerance:.2f}x slower than cuBLASLt NVFP4 "
        f"({cublas_us:.2f} us) for M={m}, N={n}, K={k}")


@pytest.mark.parametrize(
    "precision,supported_sms",
    [
        pytest.param("nvfp4", (100, 103, 107), id="nvfp4"),
        pytest.param("bf16", (107, ), id="bf16"),
        pytest.param("mxfp8", (107, ), id="mxfp8"),
    ],
)
def test_cute_dsl_nvmmh_matches_full_sweep(precision, supported_sms,
                                           monkeypatch, nvmmh_config_guard):
    """Compare CuTe DSL full-sweep and NVMMH-selected performance.

    Each precision uses the same representative M=16, N=256, K=7168 problem
    and compares the NVMMH-selected tactic with both the full CuTe DSL sweep
    and cuBLASLt using torch.profiler/CUPTI. NVFP4 runs on SM100/SM103/SM107;
    BF16 and MXFP8 run on SM107.

    The heuristic path is a strict validated subset of the sweep, so the exact
    winning tactics may differ. The achieved runtime is the test invariant.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")

    try:
        from tensorrt_llm._utils import get_sm_version
        sm_version = get_sm_version()
    except Exception:
        sm_version = None
    if sm_version not in supported_sms:
        supported_sm_names = ", ".join(f"SM{sm}" for sm in supported_sms)
        pytest.skip(f"CuTe DSL {precision} comparison requires "
                    f"{supported_sm_names}")

    from tensorrt_llm._torch.custom_ops import \
        cutedsl_matmul_heuristics as nvmmh
    if not nvmmh.IS_NVMMH_AVAILABLE:
        pytest.skip("nvMatmulHeuristics library not installed")
    if sm_version == 107:
        from tensorrt_llm._torch.cute_dsl_utils import \
            IS_CUTLASS_DSL_RUBIN_AVAILABLE
        if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
            pytest.skip("CuTe DSL Rubin support is not available")

    _require_cupti_cuda_activity()

    comparison, args = {
        "nvfp4": (_run_cute_dsl_nvfp4_heuristic_comparison,
                  (monkeypatch, nvmmh_config_guard, nvmmh, sm_version)),
        "bf16": (_run_cute_dsl_bf16_heuristic_comparison,
                 (monkeypatch, nvmmh_config_guard, nvmmh)),
        "mxfp8": (_run_cute_dsl_mxfp8_heuristic_comparison,
                  (monkeypatch, nvmmh_config_guard, nvmmh)),
    }[precision]
    comparison(*args)


@pytest.mark.parametrize("distribution", ["random", "balanced"])
def test_trtllm_gen_moe_dummy_topk_local_experts_less_than_topk(
        distribution, monkeypatch):
    """NVBugs 6457853: autotuner warmup must not fail on EP shards where
    local_num_experts < top_k (e.g. gpt-oss-120b: 128 experts, top_k=4,
    EP64 -> 2 local experts per rank, attention-DP => use_dp=True).
    Dummy rows keep the production shape: top_k distinct ids per row, all
    local experts present, remaining slots padded with out-of-shard ids."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import \
        prepare_dummy_topk_and_hook

    monkeypatch.setenv("TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION",
                       distribution)
    num_tokens, top_k = 8, 4
    num_experts, local_num_experts, local_expert_offset = 128, 2, 6
    hidden_states = torch.randn(num_tokens,
                                64,
                                dtype=torch.bfloat16,
                                device="cuda")
    topk_ids = torch.randint(0,
                             num_experts, (num_tokens, top_k),
                             dtype=torch.int32,
                             device="cuda")
    topk_weights = torch.ones(num_tokens,
                              top_k,
                              dtype=torch.bfloat16,
                              device="cuda")

    with autotune():
        _, dummy_weights, dummy_ids, _ = prepare_dummy_topk_and_hook(
            topk_weights,
            topk_ids,
            hidden_states,
            None,
            1,
            TuningConfig(),
            top_k,
            num_experts,
            local_num_experts,
            None,
            None,
            None,
            local_expert_offset=local_expert_offset,
            use_dp=True)

    assert dummy_ids.shape == (num_tokens, top_k)
    assert dummy_ids.dtype == torch.int32
    assert dummy_weights.shape == (num_tokens, top_k)
    shard = range(local_expert_offset, local_expert_offset + local_num_experts)
    for row in dummy_ids.tolist():
        assert len(set(row)) == top_k, f"duplicate ids in row {row}"
        assert sum(x in shard for x in row) == local_num_experts, (
            f"expected all {local_num_experts} local experts in row {row}")
        assert all(0 <= x < num_experts for x in row), row


def test_post_tune_merge_tactics_min_time_and_subset_kept():
    tuner = autotuner.AutoTuner()
    tuner.mapping = Mapping(world_size=2, rank=0, tp_size=2)

    r0 = {
        ("gemm", "X"): (0, ("cutlass", 10), 1.0),
        ("q", "A"): (0, ("trtllm", ), 1.0)
    }
    r1 = {
        ("gemm", "X"): (0, ("cublaslt", 0), 0.5),
        ("vae", "B"): (0, ("cutlass", 2), 1.0)
    }

    class _FakeDist:

        def tp_cp_allgather(self, obj):
            return [r0, r1]

    tuner._dist = _FakeDist()
    tuner.profiling_cache.cache = dict(r0)
    tuner.post_tune_merge_tactics()

    cache = tuner.profiling_cache.cache
    assert cache[("gemm", "X")] == (0, ("cublaslt", 0), 0.5)
    assert cache[("q", "A")] == r0[("q", "A")]
    assert cache[("vae", "B")] == r1[("vae", "B")]


def test_post_tune_merge_tactics_single_rank_noop():
    tuner = autotuner.AutoTuner()
    tuner.mapping = Mapping(world_size=1, rank=0, tp_size=1)
    tuner._dist = None
    original = {("gemm", "X"): (0, ("cutlass", 10), 1.0)}
    tuner.profiling_cache.cache = dict(original)
    tuner.post_tune_merge_tactics()
    assert tuner.profiling_cache.cache == original


def test_profiling_cache_enum_tactic_roundtrip(tmp_path):
    # An enum tactic must survive save -> load: repr("<Enum.X: v>") isn't
    # ast.literal_eval-parsable, so the cache serializes tactic.value instead.
    class _Tac(enum.IntEnum):
        TRTLLM = -1

    key = ("op::x", "Runner", "(1,)")
    src = autotuner.AutoTuner().profiling_cache
    src.cache[key] = (0, _Tac.TRTLLM, 1.0)
    path = str(tmp_path / "cache.json")
    src.save_cache(path, rank=0)  # must not raise

    dst = autotuner.AutoTuner().profiling_cache
    dst.load_cache(path, rank=0)
    assert dst.cache[key][1] == _Tac.TRTLLM  # IntEnum compares equal to -1


def test_autotune_post_tune_merge_before_save(tmp_path):
    # autotune(post_tune_merge_dist=...) must merge across ranks and persist the
    # merged winner, then restore the singleton's prior distributed state.
    tuner = AutoTuner.get()
    tuner.clear_cache()
    r0 = {("gemm", "X"): (0, ("cutlass", 10), 1.0)}
    r1 = {("gemm", "X"): (0, ("cublaslt", 0), 0.5)}
    tuner.profiling_cache.cache = dict(r0)

    class _FakeDist:
        mapping = Mapping(world_size=2, rank=0, tp_size=2)

        def tp_cp_allgather(self, obj):
            return [r0, r1]

    prev_mapping, prev_dist = tuner.mapping, tuner._dist
    path = str(tmp_path / "cache.json")
    with autotune(cache_path=path, post_tune_merge_dist=_FakeDist()):
        pass

    # Merged in memory (fastest tactic won) and persisted before the context
    # exits (the saved file carries the merged winner).
    assert tuner.profiling_cache.cache[("gemm", "X")] == (0, ("cublaslt", 0),
                                                          0.5)
    persisted = autotuner.AutoTuner().profiling_cache
    persisted.load_cache(path, rank=0)
    assert persisted.cache[("gemm", "X")][1] == ("cublaslt", 0)
    # The temporary full-world distributed state was restored.
    assert tuner.mapping is prev_mapping
    assert tuner._dist is prev_dist


def _post_tune_merge_worker(world_size):
    """Run on each MPI rank: seed a distinct cache, then merge for real."""
    rank = tensorrt_llm.mpi_rank()
    mapping = Mapping(world_size=world_size,
                      rank=rank,
                      tp_size=world_size,
                      pp_size=1)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.setup_distributed_state(mapping)

    # Shared key: rank 1's tactic is faster (should win). Plus a rank-unique key.
    shared = ("gemm", "X")
    tuner.profiling_cache.cache = {
        shared: (0, ("cutlass", 10), 1.0) if rank == 0 else
        (0, ("cublaslt", 0), 0.5),
        (f"only_rank{rank}", "K"): (0, ("trtllm", ), 1.0),
    }
    tuner.post_tune_merge_tactics()

    cache = tuner.profiling_cache.cache
    return {
        "shared": cache[shared],
        "has_r0": ("only_rank0", "K") in cache,
        "has_r1": ("only_rank1", "K") in cache,
    }


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_post_tune_merge_tactics_multi_rank(mpi_pool_executor):
    # Real 2-rank merge over an actual process group (not a FakeDist): every
    # rank must converge to the same cache — the min-time winner for the shared
    # key and both ranks' unique keys kept.
    world_size = 2
    results = list(
        mpi_pool_executor.map(_post_tune_merge_worker,
                              *zip(*[(world_size, )] * world_size)))
    assert len(results) == world_size
    for r in results:
        assert r["shared"] == (0, ("cublaslt", 0), 0.5)
        assert r["has_r0"] and r["has_r1"]


def _autotune_dist_allgather_worker(world_size):
    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen.mapping import _VisualGenAutotuneDist
    rank = tensorrt_llm.mpi_rank()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo",
                                world_size=world_size,
                                rank=rank)
    # VG's real mesh has tp_size=1; the communicator must gather over the whole
    # world regardless, so the mapping's tp axis is irrelevant to the gather.
    d = _VisualGenAutotuneDist(
        Mapping(world_size=world_size, rank=rank, tp_size=world_size))
    return d.tp_cp_allgather(f"rank{rank}")


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_visual_gen_autotune_dist_world_allgather(mpi_pool_executor):
    # _VisualGenAutotuneDist.tp_cp_allgather must gather every rank's object over
    # the default world group (not the single-rank tp subgroup), and construct
    # without running TorchDist.__init__.
    world_size = 2
    results = list(
        mpi_pool_executor.map(_autotune_dist_allgather_worker,
                              *zip(*[(world_size, )] * world_size)))
    for got in results:
        assert got == ["rank0", "rank1"]


@pytest.mark.parametrize("conflicting_config",
                         [None, {
                             "fields": ("tile", ),
                             "max_tactics": 2
                         }])
def test_nvmmh_engine_policy_lifetime(nvmmh_config_guard, monkeypatch,
                                      conflicting_config):
    """Engine wiring pins non-default policy until the last owner is cleaned up."""
    from tensorrt_llm._torch.pyexecutor import model_engine
    from tensorrt_llm.llmapi.llm_args import AutoTunerNvMMHConfig, TorchLlmArgs

    class EngineOwner:
        """Minimal engine resources for exercising the real cleanup path."""
        _cleanup_done = False
        model_loader = None

        def _release_cuda_graphs(self):
            """No graph resources are allocated by this wiring test."""

    monkeypatch.setattr(model_engine, "release_gc", lambda: None)
    args = TorchLlmArgs.model_construct(
        autotuner_nvmmh_config=AutoTunerNvMMHConfig(
            fields=("swizzle", "cta_order"), max_tactics=3))
    first, second, rejected = EngineOwner(), EngineOwner(), EngineOwner()
    tuner = nvmmh_config_guard
    try:
        model_engine._configure_autotuner_nvmmh(args, first)
        expected = autotuner.NvMMHConfig(enabled=True,
                                         fields=("swizzle", "cta_order"),
                                         max_tactics=3)
        assert tuner.nvmmh_config == expected
        installed = tuner.nvmmh_config
        model_engine._configure_autotuner_nvmmh(args, second)
        assert tuner.nvmmh_config is installed
        other_args = TorchLlmArgs.model_construct(autotuner_nvmmh_config=(
            None if conflicting_config is None else AutoTunerNvMMHConfig(
                **conflicting_config)))
        with pytest.raises(ValueError, match="active model engines"):
            model_engine._configure_autotuner_nvmmh(other_args, rejected)
        assert tuner.nvmmh_config == expected
        with pytest.raises(ValueError, match="active model engines"):
            tuner.configure_nvmmh(enabled=False)
        assert tuner.nvmmh_config == expected

        model_engine.PyTorchModelEngine.cleanup(first)
        model_engine.PyTorchModelEngine.cleanup(first)  # Idempotent release.
        with pytest.raises(ValueError, match="active model engines"):
            model_engine._configure_autotuner_nvmmh(other_args, rejected)
        model_engine.PyTorchModelEngine.cleanup(second)
        model_engine._configure_autotuner_nvmmh(other_args, rejected)
        assert tuner.nvmmh_config != expected
        model_engine.PyTorchModelEngine.cleanup(rejected)
    finally:
        tuner._release_nvmmh_policy(first)
        tuner._release_nvmmh_policy(second)
        tuner._release_nvmmh_policy(rejected)


def test_nvmmh_policy_owner_collection(nvmmh_config_guard):
    """An abandoned or partially initialized engine cannot pin policy forever."""
    import gc
    import weakref

    class Owner:
        """Weak-referenceable stand-in for an engine that failed to initialize."""

    owner = Owner()
    tuner = nvmmh_config_guard
    tuner._acquire_nvmmh_policy(owner, autotuner.NvMMHConfig(enabled=True))
    ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert ref() is None
    tuner.configure_nvmmh(enabled=False)
    assert not tuner.nvmmh_config.enabled


def test_nvmmh_unmatched_swap_preserves_split_k_admission(
        nvmmh_config_guard, monkeypatch):
    """A model miss for one swap must not resurrect rejected split-K factors."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
    from tensorrt_llm._torch.custom_ops import \
        cutedsl_matmul_heuristics as heuristics

    if not hasattr(ops, "CuteDSLMXFP8RubinLinear"):
        pytest.skip("Rubin CuTe DSL is unavailable")
    monkeypatch.setattr(ops, "IS_NVMMH_AVAILABLE", True)
    runner = ops.CuteDSLMXFP8RubinLinear(output_dtype=torch.bfloat16)
    nvmmh_config_guard.configure_nvmmh(enabled=True,
                                       fields=("tile", "cluster", "split_k"))
    tactics = [("base", (128, 128, 128), (128, 128, 64), (1, 1), swap, False,
                "static", "m", split) for swap in (False, True)
               for split in (1, 2, 4, 8)]

    def rank_one_orientation(m, n, *args, **kwargs):
        """Only the unswapped model problem has a representable result."""
        return ([heuristics.HeuristicConfig((128, 128),
                                            (1, 1), 1, 0)] if m == 128 else [])

    monkeypatch.setattr(ops, "rank_configs", rank_one_orientation)
    selected = runner._rank_prune_tactics(tactics, 128, 256, 1024)
    admitted = [
        t for t in tactics if heuristics.is_sm107_nvmmh_split_k_eligible(
            1024, runner.mma_tiler_k, t[8])
    ]
    assert len(admitted) < len(tactics)
    assert {t[4] for t in selected} == {False, True}
    assert set(selected) == set(admitted)
