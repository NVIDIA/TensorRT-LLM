# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import enum
import errno
import itertools
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, List

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm._torch.autotuner as autotuner
from tensorrt_llm import _jit_share as jit_share
from tensorrt_llm._torch.autotuner import (AutoTuner, DistributedTuningStrategy,
                                           DynamicDim, DynamicTensorSpec,
                                           FakeTensor, NvMMHConfig,
                                           OptimizationProfile, StaticDim,
                                           TunableRunner, TuningConfig,
                                           autotune)
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.utils import (deep_gemm_jit_warmup_buckets,
                                       get_power_of_2_num_tokens_buckets,
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


class JitGemmRunner(GemmRunner):
    """GemmRunner that compiles per tactic, like the CuTe DSL runners."""
    kernel_cache: dict[int, bool] = {}
    calls: list[tuple[tuple[int, ...], int]] = []

    def forward(self,
                /,
                inputs: list[torch.Tensor],
                *,
                tactic: int = -1,
                **kwargs) -> torch.Tensor:
        type(self).kernel_cache.setdefault(tactic, True)
        type(self).calls.append((tuple(inputs[0].shape), tactic))
        return super().forward(inputs, tactic=tactic, **kwargs)


@pytest.mark.parametrize("strategy", [
    DistributedTuningStrategy.PARALLEL,
    DistributedTuningStrategy.MERGE,
])
def test_prime_cached_tactics_on_cache_hit(monkeypatch, strategy) -> None:
    """Prime local cached winners without starting collective profiling."""
    monkeypatch.setenv("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", "1")
    w = torch.randn(64, 128)
    x = torch.randn(3, 64)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(input_idx=0,
                              dim_idx=0,
                              gen_tuning_buckets=(3, 4, 5),
                              map_to_tuning_buckets=lambda value: value),
            DynamicTensorSpec(input_idx=1,
                              dim_idx=1,
                              gen_tuning_buckets=(64, 128, 256, 512),
                              map_to_tuning_buckets=lambda value: value),
        ),
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL)
    op = "test_prime_cached_tactics"
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_path = os.path.join(temp_dir, "prime.json")
        with autotune(cache_path=cache_path):
            tuner.choose_one(op, [JitGemmRunner()], tuning_config, [x, w])
        winners = {
            value[1]
            for value in tuner.profiling_cache.get_specific_custom_op(
                op).values()
        }
        assert winners

        # Simulate a fresh process that only has the persistent tactic cache.
        tuning_config.distributed_tuning_strategy = strategy
        tuner.profiling_cache.clear()
        tuner._primed_cached_tactics.clear()
        JitGemmRunner.kernel_cache.clear()
        JitGemmRunner.calls.clear()
        if strategy == DistributedTuningStrategy.MERGE:

            def unexpected_prepare(*args, **kwargs) -> None:
                pytest.fail("A collective cache hit must not prepare inputs")

            monkeypatch.setattr(tuner, "_prepare_input_tensors",
                                unexpected_prepare)
        with autotune(cache_path=cache_path):
            tuner.choose_one(op, [JitGemmRunner()], tuning_config, [x, w])
        expected = winners if strategy != DistributedTuningStrategy.MERGE else set(
        )
        assert set(JitGemmRunner.kernel_cache) == expected
        assert len(JitGemmRunner.calls) == len(expected)

        with autotune(cache_path=cache_path):
            tuner.choose_one(op, [JitGemmRunner()], tuning_config, [x, w])
        assert len(JitGemmRunner.calls) == len(expected)

        # A kernel cache emptied in place is filled again.
        JitGemmRunner.kernel_cache.clear()
        with autotune(cache_path=cache_path):
            tuner.choose_one(op, [JitGemmRunner()], tuning_config, [x, w])
        assert set(JitGemmRunner.kernel_cache) == expected

        monkeypatch.setenv("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", "0")
        tuner._primed_cached_tactics.clear()
        JitGemmRunner.calls.clear()
        with autotune(cache_path=cache_path):
            tuner.choose_one(op, [JitGemmRunner()], tuning_config, [x, w])
        assert JitGemmRunner.calls == []


def test_prime_cached_tactics_after_parallel_cache_merge(monkeypatch) -> None:
    """Prime winners selected by another rank after a cold profile."""
    monkeypatch.setenv("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", "1")
    x = torch.randn(20, 64)
    w = torch.randn(64, 128)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(
            input_idx=0,
            dim_idx=0,
            gen_tuning_buckets=(3, 20),
            map_to_tuning_buckets=lambda value: value,
        ), ),
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
    )
    op = "test_prime_cached_tactics_after_parallel_cache_merge"
    tuner = AutoTuner.get()
    runner = JitGemmRunner()
    tuner.clear_cache()
    tuner._primed_cached_tactics.clear()
    JitGemmRunner.kernel_cache.clear()
    JitGemmRunner.calls.clear()

    def merge_remote_winners(strategy, custom_op) -> None:
        assert strategy == DistributedTuningStrategy.PARALLEL
        for m, tactic in ((3, 0), (20, 1)):
            cache_key = tuner.profiling_cache.get_cache_key(
                custom_op,
                runner,
                ((m, 64), (64, 128)),
                tuning_config,
                apply_map_to_tuning_buckets=False,
            )
            tuner.profiling_cache[cache_key] = (0, tactic, 0.5)

    monkeypatch.setattr(tuner, "is_tuning_mode", True)
    # Distributed synchronization is covered separately.
    monkeypatch.setattr(tuner, "_should_current_rank_tune", lambda _: False)
    monkeypatch.setattr(tuner, "cache_pp_recv", lambda: None)
    monkeypatch.setattr(tuner, "_maybe_sync_cache_data", merge_remote_winners)

    selected_runner, tactic = tuner.choose_one(op, [runner], tuning_config,
                                               [x, w])

    assert selected_runner is runner
    assert tactic == 1
    assert set(JitGemmRunner.kernel_cache) == {0, 1}
    assert JitGemmRunner.calls == [((3, 64), 0), ((20, 64), 1)]


def test_failed_cached_tactic_prime_is_retried(monkeypatch) -> None:
    """Do not memoize a cached winner until its prime launch succeeds."""
    monkeypatch.setenv("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", "1")
    x = torch.randn(3, 64)
    w = torch.randn(64, 128)
    tuning_config = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL)
    op = "test_failed_cached_tactic_prime_is_retried"
    tuner = AutoTuner.get()
    runner = JitGemmRunner()
    tuner.clear_cache()
    tuner._primed_cached_tactics.clear()
    JitGemmRunner.kernel_cache.clear()
    JitGemmRunner.calls.clear()
    profile = tuner._optimization_profiles(tuning_config, [x, w])[0]
    cache_key = tuner.profiling_cache.get_cache_key(
        op,
        runner,
        profile.get_opt_shapes(),
        tuning_config,
        apply_map_to_tuning_buckets=False,
    )
    tuner.profiling_cache[cache_key] = (0, 0, 1.0)

    original_forward = runner.forward
    attempts = 0

    def fail_once(self, inputs, *, tactic=-1, **kwargs) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected prime failure")
        return original_forward(inputs, tactic=tactic, **kwargs)

    monkeypatch.setattr(JitGemmRunner, "forward", fail_once)
    tuner._prime_cached_tactics(op, [runner], tuning_config, [x, w])
    tuner._prime_cached_tactics(op, [runner], tuning_config, [x, w])

    assert attempts == 2
    assert set(JitGemmRunner.kernel_cache) == {0}
    assert JitGemmRunner.calls == [((3, 64), 0)]


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


def _capture_autotuner_debug_lines(monkeypatch) -> List[str]:
    """Collect what the tuner writes through ``_debug_logger`` during a tuning run.

    The lines are the only record of what the tuner considered: the profiling
    cache keeps just the winner, so a tactic choice can otherwise be explained
    only by re-running the tuner with tactics removed. Patching the instance
    attribute rather than the logger keeps other loggers out of the capture --
    ``_debug_logger`` is bound once in ``__init__`` to ``logger.info`` or
    ``logger.debug`` depending on TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO, and
    this test must not depend on which.
    """
    lines: List[str] = []
    monkeypatch.setattr(AutoTuner.get(), "_debug_logger", lines.append)
    return lines


def test_every_candidate_is_logged_and_the_winner_is_the_best_of_them(
        monkeypatch):
    """Each profiled (runner, tactic) is logged, and Selected agrees with them.

    Two separate things, both needed to read a tactic choice off a run's log:

    1. Candidates the tuner measured appear, not just the winner.
    2. The Selected line is in the same family as the candidates -- it used to
       be the one [Autotuner] line on logger.debug while the rest went through
       _debug_logger, so promoting the family to INFO gave a log with every
       candidate and no winner, which is precisely the log a tuning
       investigation asks for. Capturing through _debug_logger would not see it
       at all if that regressed.

    The agreement check is what makes this more than a formatting test: each
    Selected time must equal the minimum candidate time of some profile, so a
    candidate line that reported a different measurement from the one the
    comparison used would fail here.
    """
    w = torch.randn(64, 128)
    AutoTuner.get().clear_cache()

    lines = _capture_autotuner_debug_lines(monkeypatch)
    with autotune():
        torch.ops.autotuner_test.get_best_gemm_tactic(torch.randn(M, 64), w)

    op = "autotuner_test::get_best_gemm_tactic"
    # Parsed with a regex, not a split on ", ": the shapes field is a tuple of
    # tuples and carries commas of its own.
    candidate_re = re.compile(
        rf"^\[Autotuner\] Candidate: custom_op={re.escape(op)}, runner=.*?, "
        r"tactic=(?P<tactic>-?\d+), shapes=(?P<shapes>.*), "
        r"time=(?P<time>\S+)ms$")
    selected_re = re.compile(
        rf"^\[Autotuner\] Selected: custom_op={re.escape(op)}, runner=.*?, "
        r"tactic=(?P<tactic>-?\d+), time=(?P<time>\S+)ms, fine_grained=")

    candidates = [m for m in map(candidate_re.match, lines) if m]
    selected = [m for m in map(selected_re.match, lines) if m]

    assert candidates, f"no candidate lines were logged; got {lines}"
    assert selected, f"no Selected line was logged; got {lines}"

    best_per_profile = {}
    for m in candidates:
        shapes, time = m["shapes"], m["time"]
        best = best_per_profile.get(shapes)
        if best is None or float(time) < float(best):
            best_per_profile[shapes] = time

    # String equality, deliberately: both lines format through the same
    # "{:.3f}", so the winner's time is the best candidate's time character for
    # character unless the two were computed from different measurements.
    winners = set(best_per_profile.values())
    for m in selected:
        assert m["time"] in winners, (
            f"Selected reports {m['time']}ms but no profile's best candidate "
            f"matches; candidate minima were {sorted(winners)}")


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


def _rank_disagreeing_cache_worker(world_size, strategy, seeded_rank):
    """Only ``seeded_rank`` starts with the cache entry, and the ranks' token counts build different
    profile lists, as under attention DP; tuning must not deadlock."""
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
    runner = DistributedGemmRunner(prefer_tactics=[0, 1, 2, 3])
    config = TuningConfig(distributed_tuning_strategy=strategy)
    custom_op = f"test_rank_disagreeing_cache_{strategy.value}"
    if rank == seeded_rank:
        key = tuner.profiling_cache.get_cache_key(custom_op, runner,
                                                  (x.size(), w.size()), config)
        tuner.profiling_cache[key] = (0, 2, 1e-9)
    bucketed_config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, get_power_of_2_num_tokens_buckets, next_positive_power_of_2), ),
                                   distributed_tuning_strategy=strategy)
    bucketed_op = f"{custom_op}_bucketed"

    with autotune():
        _, tactic = tuner.choose_one(custom_op=custom_op,
                                     runners=[runner],
                                     tuning_config=config,
                                     inputs=[x, w])
        tuner.choose_one(
            custom_op=bucketed_op,
            runners=[runner],
            tuning_config=bucketed_config,
            inputs=[torch.randn(16 + 8 * rank, 32, device='cuda'), w])
    # The 16-row bucket, which every rank profiled.
    _, bucketed_tactic = tuner.choose_one(custom_op=bucketed_op,
                                          runners=[runner],
                                          tuning_config=bucketed_config,
                                          inputs=[x, w])
    tactics = dist.tp_cp_allgather(obj=(tactic, bucketed_tactic))
    dist.barrier()
    return rank, tactics


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs for this test")
@pytest.mark.parametrize("seeded_rank", [1])
@pytest.mark.parametrize(
    "strategy",
    [
        DistributedTuningStrategy.BROADCAST,
        DistributedTuningStrategy.PARALLEL,
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_rank_disagreeing_cache_does_not_deadlock(strategy, seeded_rank,
                                                  mpi_pool_executor):
    world_size = 2
    results = list(
        mpi_pool_executor.map(_rank_disagreeing_cache_worker,
                              *zip(*[(world_size, strategy, seeded_rank)] *
                                   world_size),
                              timeout=300))
    assert len(results) == world_size
    for _, tactics in results:
        assert len(set(tactics)) == 1, f"ranks disagree: {tactics}"


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
    lines = _capture_autotuner_debug_lines(monkeypatch)
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
    # The shortcut bypasses the timed loop but must not bypass the candidate
    # log: 0.000 is the recorded-without-profiling marker, and the format is
    # the timed path's, so one regex reads a run's log whichever path ran.
    assert any(
        line.startswith(f"[Autotuner] Candidate: custom_op={op_single}, ")
        and line.endswith("time=0.000ms") for line in lines), (
            f"single-pair shortcut must log its candidate; got {lines}")

    # Multi-tactic on the same fixture: timed profile path must still run.
    forward_calls.clear()
    op_multi = "autotuner_test::single_pair_shortcut_multi"
    with autotune():
        tuner.choose_one(op_multi, [GemmRunner()], TuningConfig(), [x, w])
    # GemmRunner exposes 3 tactics -> 3 profile calls.
    assert len(profile_calls) == 3, (
        f"Multi-tactic op must hit _profile_single_kernel per tactic; "
        f"got {len(profile_calls)} ({profile_calls})")


def test_single_pair_shortcut_failure_is_logged_as_inf(monkeypatch):
    """A failing single-pair run still gets a Candidate record, as inf.

    The timed path lists failed candidates as inf next to the ones that ran;
    the shortcut must match, or a run whose only candidate crashed would show
    neither a Candidate nor a Selected line and read as if the op was never
    tuned. choose_one itself must not raise: the fallback (runner 0, tactic
    -1) covers the op, same as an all-failed timed loop."""

    class FailingRunner(TunableRunner):

        def unique_id(self):
            return ()

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [0]

        # No do_preparation in the signature: the shortcut fires that hook
        # OUTSIDE its try (same as the timed path), so a runner that raises
        # there crashes choose_one before the candidate is ever attempted --
        # this test is about the attempted-and-failed candidate.
        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1,
                    **kwargs) -> torch.Tensor:
            raise RuntimeError("single-pair candidate crash")

    tuner = AutoTuner.get()
    tuner.clear_cache()
    lines = _capture_autotuner_debug_lines(monkeypatch)
    x = torch.randn(M, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")

    op = "autotuner_test::single_pair_shortcut_failure"
    with autotune():
        runner, tactic = tuner.choose_one(op, [FailingRunner()], TuningConfig(),
                                          [x, w])
    assert tactic == -1, "an all-failed op must fall back to tactic -1"
    assert any(
        line.startswith(f"[Autotuner] Candidate: custom_op={op}, ")
        and line.endswith("time=infms") for line in lines
    ), (f"a failed single-pair candidate must be logged as inf; got {lines}")


def test_cutedsl_nvfp4_heuristic_matches_full_sweep(monkeypatch):
    """End-to-end guard for the nvMatmulHeuristics tactic generation.

    For one representative problem size, the tactic the AutoTuner selects when
    ``AutoTuner.nvmmh_config`` builds the candidates from nvMatmulHeuristics'
    top configurations must be no slower (up to a small tolerance) than the
    tactic it selects from the full CuteDSL NVFP4 tactic sweep. This validates
    that the nvMMH guidance does not cost performance.

    It additionally compares the heuristic-chosen CuteDSL kernel against the
    cuBLASLt NVFP4 GEMM on the same fp4 inputs: the CuteDSL kernel-only device
    time must be within a small tolerance of cuBLAS. Both kernel symbol names
    are logged for manual inspection -- cuBLAS exposes no API for its selected
    kernel's CTA tile / cluster shape, so that is not asserted.

    Note: the sweep and heuristic paths do NOT profile identical candidate sets
    -- nvMMH may also pick rasterization orders or swizzles the sweep does not
    enumerate -- so the exact winning tactic can differ. Only the achieved
    runtime is a meaningful invariant, hence the tolerance comparisons.

    Blackwell only (SM100/SM103) and requires the nvMatmulHeuristics library;
    skipped otherwise.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    try:
        from tensorrt_llm._utils import get_sm_version
        sm_version = get_sm_version()
    except Exception:
        sm_version = None
    if sm_version not in (100, 103):
        pytest.skip("CuteDSL NVFP4 requires SM100 (B200) / SM103 (B300)")

    from tensorrt_llm._torch.custom_ops import \
        cutedsl_matmul_heuristics as nvmmh
    if not nvmmh.IS_NVMMH_AVAILABLE:
        pytest.skip("nvMatmulHeuristics library not installed")
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        CuteDSLNVFP4BlackwellRunner

    # One representative square problem. fp4 packing / scale-factor layout
    # follows shmoo_nvfp4_cutedsl_heuristics.py::_quantize_inputs.
    m = n = k = 4096
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
    alpha = torch.tensor([1.0], device="cuda")
    inputs = [x_fp4, w_fp4, x_sf, w_sf, alpha]

    runner = CuteDSLNVFP4BlackwellRunner(output_dtype=dtype)
    tuning_config = runner.__class__.tuning_config

    def _best_tactic():
        tuner = AutoTuner.get()
        tuner.clear_cache()
        with autotune():
            _, tactic = tuner.choose_one(
                "test::cutedsl_nvfp4_heuristic_match",
                [runner],
                tuning_config,
                inputs,
            )
        return tactic

    def _dominant_kernel(fn, iters=20):
        """(name, per-call device-us) of the longest-running CUDA kernel that
        fn launches, isolating kernel time from host/op overhead."""
        from torch.profiler import ProfilerActivity, profile
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        best_name, best_total, best_count = "<none>", -1.0, 1
        for e in prof.key_averages():
            t = (getattr(e, "self_device_time_total", 0)
                 or getattr(e, "self_cuda_time_total", 0))
            if t and t > best_total:
                best_name, best_total, best_count = e.key, t, max(1, e.count)
        return best_name, best_total / best_count

    # Full sweep: no nvMMH guidance.
    tuner = AutoTuner.get()
    monkeypatch.setattr(tuner, "nvmmh_config", None)
    sweep_tactic = _best_tactic()

    # nvMatmulHeuristics builds the candidates from its top 5 configurations.
    monkeypatch.setattr(tuner, "nvmmh_config", NvMMHConfig(max_tactics=5))
    heuristic_tactic = _best_tactic()

    # cuBLASLt runs its own heuristic auto-tuning; warm it under autotune().
    def _cublas_call():
        return torch.ops.trtllm.nvfp4_gemm_cublaslt(x_fp4, w_fp4, x_sf, w_sf,
                                                    alpha, dtype)

    with autotune():
        _cublas_call()
    torch.cuda.synchronize()

    # All comparisons use kernel-only device time (isolates the GEMM kernel from
    # host/op dispatch overhead), measured via the CUDA profiler.
    _, sweep_us = _dominant_kernel(lambda: runner(inputs, tactic=sweep_tactic))
    _, heuristic_us = _dominant_kernel(
        lambda: runner(inputs, tactic=heuristic_tactic))
    _, cublas_us = _dominant_kernel(_cublas_call)

    # The guidance must not degrade the achieved kernel runtime beyond this
    # tolerance. With the default max_tactics=5 the nvMMH set includes the
    # empirical best tile (its cluster ranking can still be slightly off, ~2-3%
    # on square 4096), so a tight bound catches gross regressions.
    tolerance = 1.05
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
        f">{cublas_tolerance:.2f}x slower than cuBLAS NVFP4 "
        f"({cublas_us:.2f} us) for M={m}, N={n}, K={k}")


def _nvmmh_problem(case, sm_version):
    """(runner, inputs, run, reference) of one nvMMH-aware CuTe DSL runner of
    this GPU, or None if it has none for the case; run(tactic) returns that
    tactic's output."""
    from _torch.helpers import per_block_cast_to_fp8, per_block_cast_to_fp8_e8m0

    import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    torch.manual_seed(0)
    bf16 = torch.bfloat16
    rubin = sm_version == 107

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=bf16)

    def into(runner, inputs):

        def run(tactic):
            inputs[-1].fill_(float("nan"))
            runner(inputs, tactic=tactic)
            return inputs[-1]

        return run

    def returned(runner, inputs):
        return lambda tactic: runner(inputs, tactic=tactic)

    if case in ("bf16_gemm", "bf16_gemm_fp32_out"):
        # nvMMH splits K of the FP32-output router GEMM and tiles the
        # BF16-output decode GEMM narrowly.
        fp32_out = case == "bf16_gemm_fp32_out"
        m, n, k = (64, 256, 7168) if fp32_out else (64, 2112, 7168)
        c_dtype = torch.float32 if fp32_out else bf16
        a, b = randn(m, k), randn(n, k)
        runner_class = (ops.CuteDSLBf16RubinGemmRunner
                        if rubin else ops.CuteDSLBf16BlackwellGemmRunner)
        runner = runner_class(output_dtype=c_dtype)
        inputs = [a, b, torch.empty(m, n, device="cuda", dtype=c_dtype)]
        return runner, inputs, into(runner, inputs), a.float() @ b.float().t()
    if case in ("bf16_bmm", "bf16_bmm_fp8_out"):
        fp8_out = case == "bf16_bmm_fp8_out"
        if fp8_out and rubin:
            return None
        batch, m, n, k = 8, 300, 256, 512
        a, b = randn(batch, m, k), randn(batch, n, k)
        if rubin:
            runner = ops.CuteDSLBf16RubinBmmRunner()
        elif fp8_out:
            runner = ops.CuteDSLBf16BlackwellBmmFp8OutRunner()
        else:
            runner = ops.CuteDSLBf16BlackwellBmmRunner()
        c_dtype = torch.float8_e4m3fn if fp8_out else bf16
        inputs = [a, b, torch.empty(batch, m, n, device="cuda", dtype=c_dtype)]
        reference = torch.bmm(a.float(), b.float().transpose(1, 2))
        if fp8_out:
            reference = reference.to(c_dtype).float()
        return runner, inputs, into(runner, inputs), reference
    if case == "fp8_gemm":
        m, n, k = 64, 2048, 7168
        a, b = randn(m, k) / k, randn(n, k) / k
        a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128(a)
        b_fp8, b_sf = per_block_cast_to_fp8(b)
        runner = (ops.CuteDSLFp8RubinGemmRunner()
                  if rubin else ops.CuteDSLFp8BlackwellRunner())
        inputs = [a_fp8, b_fp8, a_sf, b_sf]
        reference = a.float() @ b.float().t()
        return runner, inputs, returned(runner, inputs), reference
    if case == "fp8_bmm":
        # On SM100 nvMMH splits the 64 K tiles in 3 (unequal slices).
        batch, m, n, k = 4, 64, 512, 8192
        a, b = randn(m, batch, k) / k, randn(batch, n, k) / k
        a_fp8, a_sf = torch.ops.trtllm.fp8_batched_quantize_1x128_permute102(a)
        b_fp8 = torch.empty_like(b, dtype=torch.float8_e4m3fn)
        b_sf = torch.empty(batch,
                           n // 128,
                           k // 128,
                           device="cuda",
                           dtype=torch.float32)
        for group in range(batch):
            b_fp8[group], b_sf[group] = per_block_cast_to_fp8(b[group])
        runner = (ops.CuteDSLFp8RubinBmmRunner()
                  if rubin else ops.CuteDSLFp8BlackwellBmmRunner())
        inputs = [
            a_fp8, b_fp8, a_sf, b_sf,
            torch.empty(batch, m, n, device="cuda", dtype=bf16)
        ]
        reference = torch.einsum("mgk,gnk->gmn", a.float(), b.float())
        return runner, inputs, into(runner, inputs), reference
    if case == "mxfp8_gemm":
        if not rubin:
            return None
        # A DeepSeek-V4 decode projection; nvMMH splits its 56 K tiles.
        m, n, k = 32, 2048, 7168
        a, b = randn(m, k) / k, randn(n, k) / k
        a_fp8, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(a)
        b_fp8, b_sf_k128 = per_block_cast_to_fp8_e8m0(b)
        b_sf = fp8_utils.transform_k128_scales_to_cutedsl_mxfp8_layout(
            b_sf_k128, mn=n, k=k)
        alpha = ops._get_mxfp8_gemm_alpha(a.device)
        runner = ops.CuteDSLMXFP8RubinLinear(output_dtype=bf16)
        inputs = [a_fp8, b_fp8, a_sf, b_sf, alpha]
        reference = a.float() @ b.float().t()
        return runner, inputs, returned(runner, inputs), reference
    assert case == "nvfp4_gemm"
    # nvMMH splits the 64 K blocks in 2 and 3 on SM100.
    m, n, k = 64, 2048, 16384
    x, w = randn(m, k), randn(n, k)
    x_scale = 2688 / x.abs().max().float()
    w_scale = 2688 / w.abs().max().float()
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, x_scale, 16, False)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_scale, 16, False)
    alpha = (1 / (x_scale * w_scale)).reshape(1)
    inputs = [x_fp4, w_fp4, x_sf, w_sf, alpha]
    runner = (ops.CuteDSLNVFP4RubinLinear(bf16)
              if rubin else ops.CuteDSLNVFP4BlackwellRunner(bf16))
    reference = torch.ops.trtllm.nvfp4_gemm_cutlass(*inputs, bf16).float()
    return runner, inputs, returned(runner, inputs), reference


@pytest.mark.parametrize(
    "case",
    [
        "bf16_gemm",
        "bf16_gemm_fp32_out",
        "bf16_bmm",
        "bf16_bmm_fp8_out",
        "fp8_gemm",
        "fp8_bmm",
        "mxfp8_gemm",
        "nvfp4_gemm",
    ],
)
def test_nvmmh_tactics_compute_the_product(case, monkeypatch):
    """Every nvMMH-guided tactic of the SM100/SM103 or SM107 CuTe DSL
    GEMM/BMM runners is correct, including tiles, rasters and split-K
    factors outside their full sweeps."""
    from tensorrt_llm._torch.custom_ops import \
        cutedsl_matmul_heuristics as nvmmh
    from tensorrt_llm._torch.cute_dsl_utils import (
        IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    from tensorrt_llm._utils import get_sm_version

    sm_version = get_sm_version()
    if not IS_CUTLASS_DSL_AVAILABLE or sm_version not in (100, 103, 107):
        pytest.skip("The test requires SM100/SM103/SM107 and CuTe DSL.")
    if sm_version == 107 and not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("The test requires the SM107 CuTe DSL helpers.")
    if not nvmmh.IS_NVMMH_AVAILABLE:
        pytest.skip("nvMatmulHeuristics is not installed")
    from _torch.helpers import calc_diff

    problem = _nvmmh_problem(case, sm_version)
    if problem is None:
        pytest.skip(f"No nvMMH-aware {case} runner on SM{sm_version}.")
    runner, inputs, run, reference = problem
    selected = []
    select_tactics = nvmmh.select_tactics

    def recording_select_tactics(ranked, max_tactics):
        selected.append(select_tactics(ranked, max_tactics))
        return selected[-1]

    monkeypatch.setattr(nvmmh, "select_tactics", recording_select_tactics)
    monkeypatch.setattr(AutoTuner.get(), "nvmmh_config",
                        NvMMHConfig(max_tactics=3))
    tactics = runner.get_valid_tactics(inputs, None)
    # The tactics come from nvMMH, not from the full-sweep fallback.
    assert tactics and tactics == selected[-1]
    # FP8 inputs are quantized from the BF16 reference operands.
    max_diff = 1e-3 if case.startswith(("fp8", "mxfp8")) else 1e-4
    for tactic in tactics:
        output = run(tactic).float()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), tactic
        assert calc_diff(output, reference) < max_diff, tactic


class _TacticSpaceRunner(TunableRunner):
    """Runner whose tactic space key is set by the test."""

    def __init__(self, space_key=None):
        self.space_key = space_key

    def unique_id(self):
        return ()

    def tactic_space_key(self):
        return self.space_key

    def get_valid_tactics(self, inputs: List[FakeTensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        return [0]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1,
                **kwargs) -> torch.Tensor:
        return inputs[0]


_TACTIC_SPACE_SHAPES = (torch.Size([16, 64]), )


def _tactic_space_cache_key(cache, runner):
    return cache.get_cache_key("test_tactic_space", runner,
                               _TACTIC_SPACE_SHAPES, TuningConfig())


def test_tactic_space_key_extends_the_cache_key(monkeypatch):
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
    cache = autotuner.AutoTuner().profiling_cache
    default = _tactic_space_cache_key(cache, _TacticSpaceRunner())
    assert default == ("test_tactic_space", "_TacticSpaceRunner", "()",
                       ((16, 64), ),
                       autotuner._key_fingerprint(
                           _TacticSpaceRunner,
                           autotuner._runner_fingerprint(_TacticSpaceRunner)))
    guided = _tactic_space_cache_key(cache, _TacticSpaceRunner(
        ("nvmmh", "x", 3)))
    assert guided == default + (("nvmmh", "x", 3), )
    assert _tactic_space_cache_key(cache, _TacticSpaceRunner(
        ("nvmmh", "x", 5))) != guided


_FP_BASE = """
from tensorrt_llm._torch.autotuner import TunableRunner


class Base(TunableRunner):
    def get_valid_tactics(self, inputs, profile, **kwargs):
        return [0]

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0]
"""


@pytest.fixture
def fp_modules(tmp_path, monkeypatch):
    """Modules written to tmp_path; those named fp_* stand in for tensorrt_llm
    ones, and fp_base holds a TunableRunner base."""
    import textwrap
    import types

    monkeypatch.setattr(autotuner, "_tracked",
                        lambda module: module.startswith("fp_"))
    # These tests exercise the code walk (TLLM_AUTOTUNER_CACHE_KEY=code).
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "code")
    # Rewritten sources must never run from a stale .pyc (one written within
    # the same second for a source of the same size).
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    # The modules are written after this process started; only
    # test_code_fingerprint_marks_sources_edited_after_start wants that seen.
    monkeypatch.setattr(autotuner, "_EPOCH_NS", float("inf"))
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(name, text):
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(text))

    def load(name, text):
        write(name, text)
        path = tmp_path / f"{name}.py"
        module = types.ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
        # Compiled from the source just written, never from bytecode.
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        return module

    def edit(name, old, new):
        path = tmp_path / f"{name}.py"
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))

    def fingerprints(*runner_classes):
        # As a new process would: an edit that keeps a file's size can share
        # its mtime with the previous write, so the source index is dropped.
        autotuner._SOURCE_INDEXES.clear()
        autotuner._code_fingerprint.cache_clear()
        autotuner._candidates_fingerprint.cache_clear()
        return [autotuner._code_fingerprint(k) for k in runner_classes]

    before = set(sys.modules)
    load("fp_base", _FP_BASE)
    yield SimpleNamespace(root=tmp_path,
                          write=write,
                          load=load,
                          edit=edit,
                          fingerprints=fingerprints)
    autotuner._candidates_fingerprint.cache_clear()
    # Drop the fp_* modules a fingerprint imported.
    for name in set(sys.modules) - before:
        if name.startswith("fp_"):
            del sys.modules[name]


def _changed(before, after):
    return [b != a for b, a in zip(before, after)]


def test_code_fingerprint_follows_the_definitions_a_runner_uses(fp_modules):
    fp_modules.load(
        "fp_util", """
        def used():
            return 0


        def unused():
            return 0
        """)
    runners = fp_modules.load(
        "fp_runners", """
        import fp_util
        from fp_base import Base


        class UsesUtil(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return fp_util.used()


        class Other(Base):
            pass
        """)
    classes = (runners.UsesUtil, runners.Other)

    before = fp_modules.fingerprints(*classes)
    fp_modules.edit("fp_util", "def unused():\n    return 0",
                    "def unused():\n    return 42")
    assert fp_modules.fingerprints(*classes) == before
    fp_modules.edit("fp_util", "def used():\n    return 0",
                    "def used():\n    return 42")
    assert _changed(before, fp_modules.fingerprints(*classes)) == [True, False]


def test_code_fingerprint_follows_what_values_are_built_from(fp_modules):
    fp_modules.load(
        "fp_util", """
        BLOCK = 128
        TILE = BLOCK * 2
        TABLE = {"small": 1}
        OVERRIDE = None


        def build_tactics():
            return [0]


        TACTICS = build_tactics()

        # Tables and classes completed after they are bound.
        LAUNCH = {100: (2048, 4)}
        LAUNCH[103] = LAUNCH[100]
        LAUNCH.update({90: (512, 2)})


        class Config:
            tile = 1


        Config.tile = 2

        # Tables and values decided by a loop or a branch around them.
        REGS = {}
        for atom in range(9, 17):
            REGS[atom] = 16

        if BLOCK > 64:
            MODE = 1
        else:
            MODE = 2
        """)
    runners = fp_modules.load(
        "fp_runners", """
        import fp_util
        from fp_base import Base


        def tactics(limit=fp_util.BLOCK):
            return [limit]


        class UsesDerivedConstant(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.TILE]


        class UsesTable(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return list(fp_util.TABLE)


        class UsesClassAttribute(Base):
            block = fp_util.BLOCK + 0


        class UsesDefaultArgument(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return tactics()


        class UsesBuiltTable(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return fp_util.TACTICS


        class UsesNoneConstant(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.OVERRIDE or 0]


        class ImportsNoneConstant(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                from fp_util import OVERRIDE
                return [OVERRIDE or 0]


        class UsesCompletedTable(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.LAUNCH[103]]


        class UsesPatchedClass(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.Config.tile]


        class UsesLoopTable(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.REGS.get(7, 0)]


        class UsesBranchConstant(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [fp_util.MODE]
        """)
    names = ("UsesDerivedConstant", "UsesTable", "UsesClassAttribute",
             "UsesDefaultArgument", "UsesBuiltTable", "UsesNoneConstant",
             "ImportsNoneConstant", "UsesCompletedTable", "UsesPatchedClass",
             "UsesLoopTable", "UsesBranchConstant")
    classes = [getattr(runners, name) for name in names]

    # Each edit, and the runners it must re-key (no other runner may).
    edits = [
        (("BLOCK = 128", "BLOCK = 1024"), {
            "UsesDerivedConstant", "UsesClassAttribute", "UsesDefaultArgument",
            "UsesBranchConstant"
        }),
        (('{"small": 1}', '{"small": 1, "large": 2}'), {"UsesTable"}),
        (("return [0]", "return [0, 1]"), {"UsesBuiltTable"}),
        (("OVERRIDE = None", "OVERRIDE = 1024"),
         {"UsesNoneConstant", "ImportsNoneConstant"}),
        # The repository's _SILU_AND_MUL_2IN_LAUNCH_PARAMS is completed so.
        (("LAUNCH[103] = LAUNCH[100]", "LAUNCH[103] = (1024, 8)"),
         {"UsesCompletedTable"}),
        (("{90: (512, 2)}", "{90: (512, 32)}"), {"UsesCompletedTable"}),
        (("Config.tile = 2", "Config.tile = 32"), {"UsesPatchedClass"}),
        # Only the loop's range changes (as for fp4_paged_mqa_logits' tables).
        (("range(9, 17)", "range(7, 17)"), {"UsesLoopTable"}),
        (("if BLOCK > 64:", "if BLOCK > 512:"), {"UsesBranchConstant"}),
    ]
    before = fp_modules.fingerprints(*classes)
    for (old, new), expected in edits:
        fp_modules.edit("fp_util", old, new)
        after = fp_modules.fingerprints(*classes)
        changed = {name for name, b, a in zip(names, before, after) if b != a}
        assert changed == expected, old
        before = after


def test_code_fingerprint_follows_kernel_objects_and_imports(
        fp_modules, monkeypatch):
    # jitlib stands in for a kernel DSL outside tensorrt_llm (as triton.jit):
    # its kernels are callable objects, not functions.
    fp_modules.load(
        "jitlib", """
        class jit:
            def __init__(self, fn):
                self.fn = fn
                self.__module__ = fn.__module__
                self.__qualname__ = fn.__qualname__

            def __call__(self, *args):
                return self.fn(*args)
        """)
    fp_modules.load(
        "fp_kernels", """
        import jitlib


        @jitlib.jit
        def helper():
            return 0


        @jitlib.jit
        def kernel():
            return helper()
        """)
    # A key must not import modules outside tensorrt_llm: once imported,
    # this one would stay in sys.modules.
    fp_modules.write("side_effect_mod", "IMPORTED = True\n")
    # Nor look up attributes of one, which can import (PEP 562).
    lazy = fp_modules.load(
        "lazypkg", """
        CALLS = []


        def __getattr__(name):
            CALLS.append(name)
            raise AttributeError(name)
        """)
    (fp_modules.root / "fp_pkg").mkdir()
    fp_modules.write("fp_pkg/__init__", "")
    fp_modules.write("fp_pkg/mod", "def run():\n    return 0\n")
    (fp_modules.root / "fp_pkg" / "sub").mkdir()
    fp_modules.write("fp_pkg/sub/__init__", "")
    fp_modules.write("fp_pkg/sub/deep", "def run():\n    return 0\n")
    runners = fp_modules.load(
        "fp_runners", """
        import lazypkg
        from fp_base import Base
        from fp_kernels import kernel


        class UsesKernel(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return kernel()


        class ImportsInForward(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import side_effect_mod
                from fp_kernels import helper
                return helper(), side_effect_mod


        class ImportsAliasedModule(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import fp_pkg.mod as mod
                return mod.run()


        class ImportsDeepAliasedModule(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import fp_pkg.sub.deep as deep
                return deep.run()


        class UsesLazyPackage(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return lazypkg.backend.run()
        """)
    versions = {"side_effect_mod": "1.0"}
    monkeypatch.setattr(autotuner, "_package_versions",
                        lambda top: versions.get(top, ""))
    classes = (runners.UsesKernel, runners.ImportsInForward,
               runners.ImportsAliasedModule, runners.ImportsDeepAliasedModule,
               runners.UsesLazyPackage)

    # Each change, and which of the runners above it must re-key.
    def bump_side_effect_mod():
        versions["side_effect_mod"] = "2.0"

    changes = [
        (lambda: fp_modules.edit("fp_kernels", "def helper():\n    return 0",
                                 "def helper():\n    return 42"),
         [1, 1, 0, 0, 0]),
        (lambda: fp_modules.edit("fp_pkg/mod", "return 0", "return 42"),
         [0, 0, 1, 0, 0]),
        (lambda: fp_modules.edit("fp_pkg/sub/deep", "return 0", "return 42"),
         [0, 0, 0, 1, 0]),
        (bump_side_effect_mod, [0, 1, 0, 0, 0]),
    ]
    before = fp_modules.fingerprints(*classes)
    assert "side_effect_mod" not in sys.modules
    for change, expected in changes:
        change()
        after = fp_modules.fingerprints(*classes)
        assert _changed(before, after) == [bool(e) for e in expected]
        before = after
    assert "side_effect_mod" not in sys.modules
    assert lazy.CALLS == []


def test_code_fingerprint_survives_a_module_that_fails_to_import(fp_modules):
    # A runner may import, in a branch it never takes here, a module that
    # cannot load on this system; the key must still compute.
    fp_modules.write("fp_broken", "raise ValueError('unsupported system')\n")
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class ImportsBrokenModule(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                if tactic == 99:
                    from fp_broken import kernel
                    return kernel()
                return inputs[0]
        """)
    [fingerprint] = fp_modules.fingerprints(runners.ImportsBrokenModule)
    assert fp_modules.fingerprints(runners.ImportsBrokenModule) == [fingerprint]


def test_code_fingerprint_follows_which_definition_a_name_binds(fp_modules):
    fp_modules.load(
        "fp_impls", """
        def primary():
            return 0


        def alternate():
            return 1
        """)
    fp_modules.load("fp_impl_a", "def run():\n    return 0\n")
    fp_modules.load("fp_impl_b", "def run():\n    return 1\n")
    # Each runner source, and the swap of the two definitions its names bind.
    cases = [
        ("""
        from fp_base import Base
        from fp_impls import primary as first, alternate as second


        class Runner(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return (first if tactic == 0 else second)()
        """, "primary as first, alternate as second",
         "alternate as first, primary as second"),
        ("""
        from fp_base import Base
        import fp_impl_a as first, fp_impl_b as second


        class Runner(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return first.run() if tactic == 0 else second.run()
        """, "fp_impl_a as first, fp_impl_b as second",
         "fp_impl_b as first, fp_impl_a as second"),
    ]
    for source, bindings, swapped_bindings in cases:
        [before] = fp_modules.fingerprints(
            fp_modules.load("fp_runners", source).Runner)
        assert fp_modules.fingerprints(
            fp_modules.load("fp_runners", source).Runner) == [before]
        swapped = source.replace(bindings, swapped_bindings)
        assert fp_modules.fingerprints(
            fp_modules.load("fp_runners", swapped).Runner) != [before]


def test_code_fingerprint_follows_modules_imported_in_a_function(fp_modules):
    fp_modules.load("fp_helpers", "def helper():\n    return 0\n")
    fp_modules.load("fp_backend_a", "def run():\n    return 0\n")
    fp_modules.load("fp_backend_b", "def run():\n    return 1\n")
    fp_modules.load("fp_impl", "def select():\n    return 0\n")
    # Only re-exports what another module defines.
    fp_modules.load("fp_catalog", "from fp_impl import select\n")
    (fp_modules.root / "fp_moe").mkdir()
    fp_modules.write("fp_moe/__init__", "")
    fp_modules.write(
        "fp_moe/decode", """
        from fp_helpers import helper


        def select():
            return helper()


        def unused():
            return 0
        """)
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        def run_with(module):
            return module.select()


        class UsesAttribute(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                from fp_moe import decode
                return decode.select()


        class PassesModule(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                from fp_moe import decode
                return run_with(decode)


        class PicksBackend(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                if tactic == 0:
                    import fp_backend_a as backend
                else:
                    import fp_backend_b as backend
                return backend.run()


        class RebindsLocal(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import fp_backend_a as backend
                first = backend.run()
                import fp_backend_b as backend
                return first + backend.run()


        class PassesCatalog(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import fp_catalog
                return run_with(fp_catalog)
        """)
    classes = (runners.UsesAttribute, runners.PassesModule,
               runners.PicksBackend, runners.RebindsLocal,
               runners.PassesCatalog)

    before = fp_modules.fingerprints(*classes)
    # What the module's code uses counts, in another file too.
    fp_modules.edit("fp_helpers", "return 0", "return 42")
    after_helper = fp_modules.fingerprints(*classes)
    assert _changed(before, after_helper) == [True, True, False, False, False]
    # A module used only through its attributes counts by those; one passed
    # on as a whole counts whole.
    fp_modules.edit("fp_moe/decode", "def unused():\n    return 0",
                    "def unused():\n    return 42")
    after_unused = fp_modules.fingerprints(*classes)
    assert _changed(after_helper,
                    after_unused) == [False, True, False, False, False]
    # A local that two imports bind, in two branches or one after the
    # other, counts both modules.
    for backend in ("fp_backend_a", "fp_backend_b"):
        previous = fp_modules.fingerprints(*classes)
        fp_modules.edit(backend, "return ", "return 4")
        assert _changed(previous, fp_modules.fingerprints(*classes)) == [
            False, False, True, True, False
        ]
    # A module counted whole counts what it re-exports.
    previous = fp_modules.fingerprints(*classes)
    fp_modules.edit("fp_impl", "return 0", "return 42")
    assert _changed(previous, fp_modules.fingerprints(*classes)) == [
        False, False, False, False, True
    ]


def test_code_fingerprint_follows_operators(fp_modules, monkeypatch):
    import uuid

    op = f"op_{uuid.uuid4().hex[:12]}"
    fp_modules.load(
        "fp_ops", f"""
        import torch
        from tensorrt_llm._torch.custom_ops.fast_custom_op import fast_custom_op

        _lib = torch.library.Library("fp_test", "FRAGMENT")
        _lib.define("native_{op}(Tensor x) -> Tensor")


        @torch.library.custom_op("fp_test::python_{op}", mutates_args=())
        def python_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()


        @python_op.register_kernel("cpu")
        def python_op_cpu(x):
            return x.mul(1)


        # Registered through Library.define / impl, not torch.library.custom_op.
        @fast_custom_op("fp_test::fast_{op}", mutates_args=(), device_types="CPU")
        def fast_op(x: torch.Tensor) -> torch.Tensor:
            return x.add(1)
        """)
    runners = fp_modules.load(
        "fp_runners", f"""
        import torch
        import fp_ops
        from fp_base import Base


        class UsesPythonOp(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return torch.ops.fp_test.python_{op}(inputs[0])


        class UsesNativeOp(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return torch.ops.fp_test.native_{op}(inputs[0])


        class UsesNativeClass(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                runner = torch.classes.fp_test.Runner_{op}()
                return list(range(runner.get_num_configs()))


        class ImportsClassesLocally(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                from torch import classes
                runner = classes.fp_test.Runner_{op}()
                return list(range(runner.get_num_configs()))


        class ImportsTorchLocally(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                import torch as t
                return t.ops.fp_test.native_{op}(inputs[0])


        class UsesFastOp(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return torch.ops.fp_test.fast_{op}(inputs[0])
        """)
    build = {"id": "a"}
    monkeypatch.setattr(autotuner, "_native_build_id", lambda: build["id"])
    classes = (runners.UsesPythonOp, runners.UsesNativeOp,
               runners.UsesNativeClass, runners.ImportsClassesLocally,
               runners.ImportsTorchLocally, runners.UsesFastOp)

    # Each change, and which of the runners above it must re-key.
    def rebuild():
        build["id"] = "b"

    changes = [
        (lambda: fp_modules.edit("fp_ops", "return x.clone()",
                                 "return x.clone() + 0"), [1, 0, 0, 0, 0, 0]),
        (lambda: fp_modules.edit("fp_ops", "x.mul(1)", "x.mul(10)"),
         [1, 0, 0, 0, 0, 0]),
        (lambda: fp_modules.edit("fp_ops", "x.add(1)", "x.add(10)"),
         [0, 0, 0, 0, 0, 1]),
        (rebuild, [0, 1, 1, 1, 1, 0]),
    ]
    before = fp_modules.fingerprints(*classes)
    for change, expected in changes:
        change()
        after = fp_modules.fingerprints(*classes)
        assert _changed(before, after) == [bool(e) for e in expected]
        before = after


_CANDIDATE_RUNNERS = """
    import fp_util
    from fp_base import Base


    class Winner(Base):
        pass


    class Loser(Base):
        def forward(self, inputs, tactic=-1, **kwargs):
            return fp_util.helper()
    """


def test_candidates_fingerprint_covers_every_candidate(fp_modules):
    fp_modules.load("fp_util", "def helper():\n    return 0\n")
    runners = fp_modules.load("fp_runners", _CANDIDATE_RUNNERS)

    def fingerprints():
        [own] = fp_modules.fingerprints(runners.Winner)
        candidates = autotuner._candidates_fingerprint
        return (own, candidates(
            (runners.Winner, )), candidates((runners.Winner, runners.Loser)),
                candidates((runners.Loser, runners.Winner)))

    before = fingerprints()
    own, alone, pair, pair_reversed = before
    assert alone == own and pair == pair_reversed and pair != alone
    # A change to the runner that lost re-keys entries tuned against it.
    fp_modules.edit("fp_util", "return 0", "return 42")
    assert _changed(before, fingerprints()) == [False, False, True, True]


def test_code_fingerprint_marks_sources_edited_after_start(
        fp_modules, monkeypatch):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Runner(Base):
            pass
        """)
    [unchanged_since_start] = fp_modules.fingerprints(runners.Runner)
    warnings = []
    monkeypatch.setattr(autotuner.logger,
                        "warning_once",
                        lambda message, key=None: warnings.append(key))
    monkeypatch.setattr(autotuner, "_EPOCH_NS", 0)
    # The process may run other code than the disk holds: its key matches no
    # entry a process that loaded the edited file writes, and is not saved.
    [edited_since_start] = fp_modules.fingerprints(runners.Runner)
    assert edited_since_start == autotuner._EDITED_PREFIX + unchanged_since_start
    assert autotuner._edited_key(("op", "Runner", "()", (), edited_since_start))
    assert any(key and key[0] == "autotuner_source_edited" for key in warnings)


_FACTORY_RUNNERS = """
    from fp_base import Base


    class Narrow(Base):
        def get_valid_tactics(self, inputs, profile, **kwargs):
            return [0]


    class Wide(Base):
        def get_valid_tactics(self, inputs, profile, **kwargs):
            return [0, 1]


    def factory(base):
        class Runner(base):
            pass

        return Runner
    """


def test_code_fingerprint_tells_factory_built_runner_classes_apart(fp_modules):
    runners = fp_modules.load("fp_runners", _FACTORY_RUNNERS)
    narrow = runners.factory(runners.Narrow)
    wide = runners.factory(runners.Wide)
    assert narrow.__qualname__ == wide.__qualname__
    fp_modules.fingerprints()
    cached = [autotuner._code_fingerprint(k) for k in (narrow, wide)]
    walked = [autotuner._FingerprintWalk(k).digest() for k in (narrow, wide)]
    assert cached == walked and cached[0] != cached[1]
    assert autotuner._candidates_fingerprint(
        (narrow, )) != autotuner._candidates_fingerprint((wide, ))
    # Built again on the same base, it is the same runner.
    assert autotuner._code_fingerprint(runners.factory(
        runners.Narrow)) == cached[0]


# One factory whose runners differ only in the implementation they capture.
_CLOSURE_RUNNERS = """
    from fp_base import Base


    def narrow():
        return [0]


    def wide():
        return [0, 1]


    def factory(impl):
        class Runner(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return impl()

        return Runner
    """


def test_code_fingerprint_follows_captured_implementations(fp_modules):
    runners = fp_modules.load("fp_runners", _CLOSURE_RUNNERS)
    narrow = runners.factory(runners.narrow)
    wide = runners.factory(runners.wide)
    fp_modules.fingerprints()
    cached = [autotuner._code_fingerprint(k) for k in (narrow, wide)]
    walked = [autotuner._FingerprintWalk(k).digest() for k in (narrow, wide)]
    assert cached == walked and cached[0] != cached[1]
    pair = autotuner._candidates_fingerprint((narrow, wide))
    assert autotuner._key_fingerprint(narrow,
                                      pair) != autotuner._key_fingerprint(
                                          wide, pair)
    # Built again around the same implementation, it is the same runner.
    assert autotuner._code_fingerprint(runners.factory(
        runners.narrow)) == cached[0]
    # An edit to the captured implementation re-keys only its runner.
    fp_modules.edit("fp_runners", "return [0, 1]", "return [0, 1, 2]")
    assert _changed(cached, fp_modules.fingerprints(narrow,
                                                    wide)) == [False, True]


def test_candidates_fingerprint_follows_the_build(fp_modules, monkeypatch):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Winner(Base):
            pass


        class Loser(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return 0
        """)
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
    build = {"id": "a"}
    monkeypatch.setattr(autotuner, "_native_build_id", lambda: build["id"])

    def fingerprints():
        autotuner._candidates_fingerprint.cache_clear()
        return [
            autotuner._candidates_fingerprint(classes)
            for classes in ((runners.Winner, ), (runners.Loser, ),
                            (runners.Winner, runners.Loser))
        ]

    before = fingerprints()
    assert len(set(before)) == 3
    # The same on every process of an installation, whatever the code does.
    fp_modules.edit("fp_runners", "return 0", "return 42")
    assert fingerprints() == before
    # A rebuild re-keys every runner.
    build["id"] = "b"
    assert _changed(before, fingerprints()) == [True, True, True]


def test_candidates_fingerprint_covers_the_losers_tactic_spaces(fp_modules):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Spaced(Base):
            def __init__(self, space=None, uid=()):
                self.space = space
                self.uid = uid

            def unique_id(self):
                return self.uid

            def tactic_space_key(self):
                return self.space


        class Winner(Base):
            pass
        """)
    winner = runners.Winner()
    classes = (runners.Winner, runners.Spaced)
    set_fingerprint = autotuner._candidates_fingerprint(classes)

    def key(loser):
        spaces = autotuner._tactic_spaces([winner, loser])
        return autotuner._key_fingerprint(runners.Winner, set_fingerprint,
                                          spaces)

    # A loser that widens its tactic space re-opens the comparison, and so
    # does one that changes its unique_id without a tactic space key.
    assert key(runners.Spaced(("limited", ))) != key(runners.Spaced(("all", )))
    assert key(runners.Spaced(uid=(1, ))) != key(runners.Spaced(uid=(2, )))

    # So does one of the same class as the winner, which it tells apart by
    # unique_id.
    alike = autotuner._candidates_fingerprint((runners.Spaced, ))

    def alike_key(winner_space, loser_space):
        candidates = [
            runners.Spaced(winner_space, uid=(0, )),
            runners.Spaced(loser_space, uid=(1, ))
        ]
        return autotuner._key_fingerprint(runners.Spaced, alike,
                                          autotuner._tactic_spaces(candidates))

    assert alike_key(("all", ), ("limited", )) != alike_key(("all", ),
                                                            ("all", ))
    assert alike_key(("all", ), ("limited", )) != alike_key(("limited", ),
                                                            ("all", ))


def test_code_fingerprint_of_a_factory_runner_that_names_itself(fp_modules):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        def factory():
            class Runner(Base):
                tactics = [0]

                def get_valid_tactics(self, inputs, profile, **kwargs):
                    return Runner.tactics

            return Runner
        """)
    fp_modules.fingerprints()
    first, second = runners.factory(), runners.factory()
    assert autotuner._code_fingerprint(first) == autotuner._code_fingerprint(
        second)


def test_candidates_fingerprint_keys_same_named_runners_apart(fp_modules):
    runners = fp_modules.load("fp_runners", _FACTORY_RUNNERS)
    narrow = runners.factory(runners.Narrow)
    wide = runners.factory(runners.Wide)
    fp_modules.fingerprints()
    pair = autotuner._candidates_fingerprint((narrow, wide))
    # Same class name and unique_id: only their own fingerprints differ.
    assert autotuner._key_fingerprint(narrow,
                                      pair) != autotuner._key_fingerprint(
                                          wide, pair)
    # A runner tuned alone is keyed by its own fingerprint (and the GPU's).
    alone = autotuner._candidates_fingerprint((narrow, ))
    assert autotuner._own_fingerprint(autotuner._key_fingerprint(
        narrow, alone))[0] == alone


def test_candidates_fingerprint_tells_factory_runners_apart_by_build(
        fp_modules, monkeypatch):
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
    for source, narrow_impl, wide_impl in (
        (_FACTORY_RUNNERS, "Narrow", "Wide"),
        (_CLOSURE_RUNNERS, "narrow", "wide"),
    ):
        runners = fp_modules.load("fp_runners", source)
        narrow, wide, again = (runners.factory(getattr(runners, impl))
                               for impl in (narrow_impl, wide_impl,
                                            narrow_impl))
        fp_modules.fingerprints()
        # Built around different bases or implementations: their names differ.
        assert autotuner._runner_fingerprint(
            narrow) != autotuner._runner_fingerprint(wide)
        # Built alike: one key, in every process.
        assert autotuner._runner_fingerprint(
            narrow) == autotuner._runner_fingerprint(again)
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        def narrow():
            return [0]


        def wide():
            return [0, 1]


        def factory(impl):
            class Runner(Base):
                def get_valid_tactics(self, inputs, profile, kernel=impl,
                                      **kwargs):
                    return kernel()

            return Runner
        """)
    fp_modules.fingerprints()
    # An implementation bound as a default argument counts too.
    assert autotuner._runner_fingerprint(runners.factory(
        runners.narrow)) != autotuner._runner_fingerprint(
            runners.factory(runners.wide))


def test_code_fingerprint_of_a_class_another_thread_identifies(fp_modules):
    import threading

    runners = fp_modules.load("fp_runners", _CLOSURE_RUNNERS)
    narrow = runners.factory(runners.narrow)
    started, finish = threading.Event(), threading.Event()

    def identify_elsewhere():
        # Another thread in the middle of identifying the same class.
        autotuner._CLASS_IDENTITIES_IN_PROGRESS.classes = {narrow}
        started.set()
        finish.wait()

    thread = threading.Thread(target=identify_elsewhere)
    thread.start()
    try:
        started.wait()
        assert autotuner._class_identity(narrow) != autotuner._class_name(
            narrow)
    finally:
        finish.set()
        thread.join()


def test_build_fingerprint_skips_unreadable_package_metadata(monkeypatch):
    import importlib.metadata

    class Broken:

        @property
        def metadata(self):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

    class Good:
        metadata = {"Name": "good-dist"}
        files = None

        def read_text(self, name):
            return "good\n" if name == "top_level.txt" else None

    monkeypatch.setattr(importlib.metadata, "distributions",
                        lambda: [Broken(), Good()])
    autotuner._packages_distributions.cache_clear()
    try:
        assert autotuner._packages_distributions() == {"good": ["good-dist"]}
    finally:
        autotuner._packages_distributions.cache_clear()


def test_build_fingerprint_follows_what_factory_runners_hold(monkeypatch):
    import functools

    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
    key = autotuner._build_class_key

    def narrow():
        return [0]

    def wide():
        return [0, 1]

    def pick(first, second):
        return first()

    def make_base(impl):

        class Base:

            def get_valid_tactics(self):
                return impl()

        return Base

    def make(base):

        class Runner(base):
            pass

        return Runner

    def holding(impl):

        class Runner:

            def get_valid_tactics(self):
                return impl()

        return Runner

    # A class built on bases that one factory builds differently.
    assert key(make(make_base(narrow))) != key(make(make_base(wide)))
    assert key(make(make_base(narrow))) == key(make(make_base(narrow)))
    # A partial counts by its keyword names, not only its values' order.
    assert key(holding(functools.partial(
        pick, first=narrow, second=wide))) != key(
            holding(functools.partial(pick, second=narrow, first=wide)))
    # Constants a factory captures count.
    assert key(holding((0, ))) != key(holding((0, 1)))

    # A function wrapping a partial, and a partial that holds itself, are
    # named without failing the lookup.
    def wrapper():
        return pick(narrow, wide)

    wrapped = functools.wraps(functools.partial(pick, narrow, wide))(wrapper)
    assert key(holding(wrapped)) != key(holding(narrow))
    cyclic = functools.partial(pick, first=narrow, second=None)
    cyclic.keywords["second"] = cyclic
    assert key(holding(cyclic))


def test_build_fingerprint_follows_loaded_libraries(monkeypatch, tmp_path):
    library = tmp_path / "libcublasLt.so.13"
    library.write_bytes(b"one")
    native = tmp_path / "libth_common.so"
    native.write_bytes(b"native")
    loaded = {"paths": (str(library), )}
    monkeypatch.setattr(autotuner, "_native_build_id", lambda: "")
    monkeypatch.setattr(autotuner, "_native_libraries", lambda: [native])
    monkeypatch.setattr(autotuner, "_NATIVE_FILES",
                        {str(native): autotuner._file_identity(native)})
    monkeypatch.setattr(autotuner, "_loaded_libraries", lambda: loaded["paths"])
    monkeypatch.setattr(autotuner, "_EPOCH_NS", float("inf"))

    def build():
        autotuner._build_fingerprint.cache_clear()
        autotuner._native_replaced.cache_clear()
        return autotuner._build_fingerprint()

    try:
        first = build()
        assert not first.startswith(autotuner._EDITED_PREFIX)
        library.write_bytes(b"two")  # Another cuBLAS build.
        second = build()
        assert second != first
        # Fingerprints taken before a library loads (cuBLAS once CUDA is
        # initialized) are taken again once it has.
        monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
        before = autotuner._runner_fingerprint(_TacticSpaceRunner)
        loaded["paths"] = (str(library), str(tmp_path / "libcublas.so.13"))
        assert autotuner._runner_fingerprint(_TacticSpaceRunner) != before
        loaded["paths"] = (str(library), )
        # A library replaced on disk after this process loaded it: its
        # mapping is deleted, it is newer than the process, or (renamed over,
        # keeping the old mtime) it is not the file found on import.
        loaded["paths"] = (str(library) + " (deleted)", )
        assert build().startswith(autotuner._EDITED_PREFIX)
        loaded["paths"] = (str(library), )
        assert not build().startswith(autotuner._EDITED_PREFIX)
        replacement = tmp_path / "replacement.so"
        replacement.write_bytes(b"native")
        os.replace(replacement, native)
        assert build().startswith(autotuner._EDITED_PREFIX)
        monkeypatch.setattr(autotuner, "_NATIVE_FILES",
                            {str(native): autotuner._file_identity(native)})
        assert not build().startswith(autotuner._EDITED_PREFIX)
        monkeypatch.setattr(autotuner, "_EPOCH_NS", 0)
        assert build().startswith(autotuner._EDITED_PREFIX)
    finally:
        autotuner._clear_fingerprints()


def test_candidates_fingerprint_keys_follow_the_current_gpu(monkeypatch):
    """The GPU architecture is looked up at each key, not memoized with the
    fingerprints: a process that switches GPUs does not serve the tactics
    tuned on the other one."""
    device = {"capability": "sm107"}
    monkeypatch.setattr(autotuner, "_device_capability",
                        lambda: device["capability"])
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")

    def key():
        return autotuner._key_fingerprint(
            _TacticSpaceRunner,
            autotuner._candidates_fingerprint((_TacticSpaceRunner, )))

    on_sm107 = key()
    device["capability"] = "sm100"
    assert key() != on_sm107
    assert autotuner._own_fingerprint(key()) == (on_sm107.partition("@")[0],
                                                 "sm100")


@pytest.mark.parametrize("mode", ["code", "build"])
def test_candidates_fingerprint_follows_the_runner_a_wrapper_holds(
        fp_modules, monkeypatch, mode):
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", mode)
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Narrow(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [0]


        class Wide(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [0, 1]


        class Wrapper(Base):
            def __init__(self, op_runner):
                self._op_runner = op_runner

            def get_valid_tactics(self, inputs, profile, **kwargs):
                return self._op_runner.get_valid_tactics(inputs, profile)
        """)
    fp_modules.fingerprints()

    def fingerprint(runner):
        return autotuner._candidates_fingerprint(
            (autotuner._runner_kind(runner), ))

    narrow = runners.Wrapper(runners.Narrow())
    assert fingerprint(narrow) != fingerprint(runners.Wrapper(runners.Wide()))
    assert fingerprint(narrow) == fingerprint(runners.Wrapper(runners.Narrow()))


def test_candidates_fingerprint_binds_tactic_spaces_to_runners(fp_modules):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Narrow(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [0]


        class Wide(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return [0, 1]


        def factory(base):
            class Runner(base):
                def __init__(self, space):
                    self.space = space

                def unique_id(self):
                    return ()

                def tactic_space_key(self):
                    return self.space

            return Runner


        class Winner(Base):
            pass
        """)
    fp_modules.fingerprints()
    narrow = runners.factory(runners.Narrow)
    wide = runners.factory(runners.Wide)

    def spaces(narrow_space, wide_space):
        return autotuner._tactic_spaces(
            [runners.Winner(),
             narrow(narrow_space),
             wide(wide_space)])

    # Same class name and unique_id: which runner has which space counts.
    assert spaces(("limited", ), ("all", )) != spaces(("all", ), ("limited", ))


def test_code_fingerprint_follows_what_factory_runners_hold(fp_modules):
    import functools

    r = fp_modules.load(
        "fp_runners", """
        import functools

        from fp_base import Base


        def narrow():
            return [0]


        def wide():
            return [0, 1]


        def pick(first, second):
            return first()


        def layer(fn):
            def wrapper():
                return fn()

            return wrapper


        def by_default(impl):
            class Runner(Base):
                def get_valid_tactics(self, inputs, profile, kernel=impl,
                                      **kwargs):
                    return kernel()

            return Runner


        def holding(impl):
            class Runner(Base):
                def get_valid_tactics(self, inputs, profile, **kwargs):
                    return impl()

            return Runner


        def wrapped(impl):
            for _ in range(5):
                impl = layer(impl)
            return holding(impl)


        def ordered(preferred, fallback):
            class Runner(Base):
                def get_valid_tactics(self, inputs, profile, **kwargs):
                    return preferred() or fallback()

            return Runner


        def constant(tactics):
            class Runner(Base):
                def get_valid_tactics(self, inputs, profile, **kwargs):
                    return list(tactics)

            return Runner


        def flagged(wide):
            return [0, 1] if wide else [0]


        def deep(impl):
            for _ in range(16):
                impl = layer(impl)
            return holding(impl)


        def classed(tactics):
            class Runner(Base):
                TACTICS = tactics

                def get_valid_tactics(self, inputs, profile, **kwargs):
                    return list(self.TACTICS)

            return Runner


        def limited(limit):
            def decorate(fn):
                @functools.wraps(fn)
                def call():
                    return fn()[:limit]

                return call

            return decorate
        """)
    fp_modules.fingerprints()
    fp = autotuner._code_fingerprint
    partial = functools.partial
    pairs = [
        (r.by_default(r.narrow), r.by_default(r.wide)),
        (r.holding(partial(r.narrow)), r.holding(partial(r.wide))),
        (r.holding(partial(r.pick, first=r.narrow, second=r.wide)),
         r.holding(partial(r.pick, second=r.narrow, first=r.wide))),
        (r.wrapped(r.narrow), r.wrapped(r.wide)),
        (r.ordered(r.narrow, r.wide), r.ordered(r.wide, r.narrow)),
        (r.constant((0, )), r.constant((0, 1))),
        (r.holding(partial(r.flagged,
                           False)), r.holding(partial(r.flagged, True))),
        (r.deep(r.narrow), r.deep(r.wide)),
        (r.classed((0, )), r.classed((0, 1))),
        (r.holding(r.limited(1)(r.wide)), r.holding(r.limited(2)(r.wide))),
    ]
    assert [fp(a) != fp(b) for a, b in pairs] == [True] * len(pairs)
    # Built alike, one fingerprint.
    assert fp(r.by_default(r.narrow)) == fp(r.by_default(r.narrow))
    # A partial that holds itself is walked once.
    looped = partial(r.pick, first=r.narrow)
    looped.keywords["second"] = looped
    assert fp(r.holding(looped)) != fp(r.holding(partial(r.narrow)))


def test_code_fingerprint_follows_fallback_imports_and_modules_passed_whole(
        fp_modules):
    fp_modules.load("fp_primary", "TACTICS = [0, 1]\n")
    fp_modules.load("fp_backup", "TACTICS = [0]\n")
    fp_modules.load("fp_impl", "def tactics():\n    return [0, 1]\n")
    fp_modules.load(
        "fp_backend",
        "import fp_impl\n\ndef valid():\n    return fp_impl.tactics()\n")
    fp_modules.load("fp_bundle", "import fp_backend\n")
    runners = fp_modules.load(
        "fp_runners", """
        try:
            from fp_primary import TACTICS
        except ImportError:
            from fp_backup import TACTICS
        import fp_bundle
        from fp_base import Base


        def query(bundle):
            return bundle.fp_backend.valid()


        class Imported(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return TACTICS


        class Passed(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return query(fp_bundle)
        """)
    classes = (runners.Imported, runners.Passed)
    before = fp_modules.fingerprints(*classes)
    fp_modules.edit("fp_primary", "[0, 1]", "[0, 2]")
    # A module the bundle's module imports whole uses another in turn.
    fp_modules.edit("fp_impl", "[0, 1]", "[0, 2]")
    assert _changed(before, fp_modules.fingerprints(*classes)) == [True, True]


def test_code_fingerprint_follows_tables_changed_through_an_alias_or_a_call(
        fp_modules):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base

        TACTICS = [0]
        ALIAS = TACTICS
        SECOND = ALIAS
        SECOND.append(1)
        OTHER = [0]


        def init(table):
            table.append(1)


        init(OTHER)


        class ByAlias(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return TACTICS


        class ByCall(Base):
            def get_valid_tactics(self, inputs, profile, **kwargs):
                return OTHER
        """)
    classes = (runners.ByAlias, runners.ByCall)
    before = fp_modules.fingerprints(*classes)
    fp_modules.edit("fp_runners", "SECOND.append(1)", "SECOND.append(2)")
    fp_modules.edit("fp_runners", "table.append(1)", "table.append(2)")
    assert _changed(before, fp_modules.fingerprints(*classes)) == [True, True]


_HELPER_RUNNER = """
    from fp_base import Base
    from fp_util import helper


    class Runner(Base):
        def get_valid_tactics(self, inputs, profile, **kwargs):
            return helper()
    """


def test_code_fingerprint_follows_the_code_a_process_runs(fp_modules):
    fp_modules.load("fp_util", "def helper():\n    return [0, 1]\n")
    runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
    # A process that still runs [0, 1] (from a stale .pyc) while the file
    # says [0, 2], and one that runs what the file says.
    fp_modules.edit("fp_util", "[0, 1]", "[0, 2]")
    stale = fp_modules.fingerprints(runners.Runner)
    fp_modules.load("fp_util", "def helper():\n    return [0, 2]\n")
    runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
    assert fp_modules.fingerprints(runners.Runner) != stale
    # Nor when the stale code is a module-level table the helper returns.
    table = "T = (0, 1)\n\ndef helper():\n    return T\n"
    fp_modules.load("fp_util", table)
    runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
    fp_modules.edit("fp_util", "(0, 1)", "(0, 2)")
    stale = fp_modules.fingerprints(runners.Runner)
    fp_modules.load("fp_util", table.replace("(0, 1)", "(0, 2)"))
    runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
    assert fp_modules.fingerprints(runners.Runner) != stale
    # A helper whose source can no longer be read counts by its code, and by
    # what that code calls.
    unreadable = []
    for tactics in ("[0, 1]", "[0, 3]"):
        fp_modules.load("fp_util", f"def helper():\n    return {tactics}\n")
        runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
        (fp_modules.root / "fp_util.py").unlink()
        unreadable += fp_modules.fingerprints(runners.Runner)
    assert unreadable[0] != unreadable[1]
    fp_modules.load("fp_impl", "def tactics():\n    return [0, 1]\n")
    fp_modules.load(
        "fp_util",
        "from fp_impl import tactics\n\ndef helper():\n    return tactics()\n")
    runners = fp_modules.load("fp_runners", _HELPER_RUNNER)
    (fp_modules.root / "fp_util.py").unlink()
    before = fp_modules.fingerprints(runners.Runner)
    fp_modules.edit("fp_impl", "[0, 1]", "[0, 2]")
    assert fp_modules.fingerprints(runners.Runner) != before


def test_code_fingerprint_keeps_no_object_a_global_held_alive(fp_modules):
    import gc
    import weakref

    util = fp_modules.load(
        "fp_util", """
        class Weights:
            pass


        CURRENT = None
        """)
    runners = fp_modules.load(
        "fp_runners", """
        import fp_util
        from fp_base import Base


        class Runner(Base):
            def forward(self, inputs, tactic=-1, **kwargs):
                return fp_util.CURRENT
        """)
    util.CURRENT = util.Weights()  # As a layer sets its weights for one call.
    alive = weakref.ref(util.CURRENT)
    fp_modules.fingerprints(runners.Runner)
    util.CURRENT = None
    gc.collect()
    assert alive() is None


def test_code_fingerprint_walks_a_runner_class_built_per_call_once(
        fp_modules, monkeypatch):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        def make_runner():
            class Runner(Base):
                pass

            return Runner
        """)
    walks = []

    class _CountingWalk(autotuner._FingerprintWalk):

        def digest(self):
            walks.append(self)
            return super().digest()

    monkeypatch.setattr(autotuner, "_FingerprintWalk", _CountingWalk)
    first, second = runners.make_runner(), runners.make_runner()
    assert first is not second
    fp_modules.fingerprints()
    assert autotuner._code_fingerprint(first) == autotuner._code_fingerprint(
        second)
    assert len(walks) == 1


def test_code_fingerprint_falls_back_to_the_build_when_the_walk_fails(
        fp_modules, monkeypatch):
    runners = fp_modules.load(
        "fp_runners", """
        from fp_base import Base


        class Runner(Base):
            pass
        """)

    def fail(self):
        raise RuntimeError("walk failed")

    warnings = []
    monkeypatch.setattr(autotuner._FingerprintWalk, "digest", fail)
    monkeypatch.setattr(autotuner, "_native_build_id", lambda: "build")
    monkeypatch.setattr(autotuner.logger,
                        "warning_once",
                        lambda message, key=None: warnings.append(key))
    [fingerprint] = fp_modules.fingerprints(runners.Runner)
    assert fingerprint == autotuner._build_runner_fingerprint(runners.Runner)
    assert any(key and key[0] == "autotuner_fingerprint_failed"
               for key in warnings)
    # As under "build", factory runners around different implementations
    # keep their keys apart, and a replaced native library stays local.
    factory = fp_modules.load("fp_factory", _CLOSURE_RUNNERS)
    narrow = factory.factory(factory.narrow)
    wide = factory.factory(factory.wide)
    assert fp_modules.fingerprints(narrow) != fp_modules.fingerprints(wide)
    import functools
    monkeypatch.setattr(autotuner, "_native_replaced",
                        functools.lru_cache(maxsize=None)(lambda: True))
    [fingerprint] = fp_modules.fingerprints(runners.Runner)
    assert fingerprint.startswith(autotuner._EDITED_PREFIX)


def test_code_fingerprint_of_a_runner_in_a_fileless_main(
        fp_modules, monkeypatch):
    import types

    main = types.ModuleType("__main__")  # As under `python -c`, or a notebook.
    monkeypatch.setitem(sys.modules, "__main__", main)
    exec("from fp_base import Base\n\n\nclass Runner(Base):\n    pass\n",
         main.__dict__)
    assert main.Runner.__module__ == "__main__"
    assert fp_modules.fingerprints(main.Runner)[0]


def test_code_fingerprint_package_versions_skip_dists_without_metadata(
        monkeypatch):
    # An interrupted install can leave a dist-info without METADATA.
    autotuner._package_versions.cache_clear()
    monkeypatch.setattr(autotuner, "_packages_distributions",
                        lambda: {"fp_pkg": ["fp-dist", None]})
    monkeypatch.setattr(autotuner.importlib.metadata, "version",
                        lambda dist: {"fp-dist": "1.0"}[dist])
    try:
        assert autotuner._package_versions("fp_pkg") == "fp-dist==1.0"
    finally:
        autotuner._package_versions.cache_clear()


def test_native_library_id_is_the_elf_build_id(tmp_path):
    import hashlib
    import struct

    build_id = bytes(range(20))
    note = struct.pack("<III", 4, len(build_id), 3) + b"GNU\0" + build_id
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"  # 64-bit, little-endian
    struct.pack_into("<Q", header, 0x28, 64 + len(note))  # section headers
    struct.pack_into("<HH", header, 0x3A, 64, 2)
    note_section = bytearray(64)
    struct.pack_into("<I", note_section, 4, 7)  # SHT_NOTE
    struct.pack_into("<QQ", note_section, 0x18, 64, len(note))
    library = tmp_path / "libfp.so"
    library.write_bytes(bytes(header) + note + bytes(64) + note_section)
    assert autotuner._library_id(library) == build_id.hex()

    no_sections = bytearray(header)
    for entry_size, count in ((0, 2), (64, 0)):
        struct.pack_into("<HH", no_sections, 0x3A, entry_size, count)
        stripped = tmp_path / f"libstripped_{entry_size}_{count}.so"
        stripped.write_bytes(bytes(no_sections))
        assert autotuner._library_id(stripped) == hashlib.sha256(
            bytes(no_sections)).hexdigest()

    other = tmp_path / "libother.so"
    other.write_bytes(b"not an ELF file")
    assert autotuner._library_id(other) == hashlib.sha256(
        b"not an ELF file").hexdigest()


def test_search_cache_reopens_the_comparison_when_a_loser_changes(fp_modules):
    fp_modules.load("fp_util", "def helper():\n    return 0\n")
    runners = fp_modules.load("fp_runners", _CANDIDATE_RUNNERS)
    cache = autotuner.AutoTuner().profiling_cache
    winner, loser = runners.Winner(), runners.Loser()
    key = cache.get_cache_key("test_candidates",
                              winner,
                              _TACTIC_SPACE_SHAPES,
                              TuningConfig(),
                              candidates=[winner, loser])
    cache.cache[key] = (0, 7, 1.0)

    def search(candidates):
        return cache.search_cache("test_candidates", candidates,
                                  _TACTIC_SPACE_SHAPES, TuningConfig())

    assert search([winner, loser]) == (True, 0, 7, 1.0)
    assert not search([winner])[0]  # Tuned among other candidates.
    fp_modules.edit("fp_util", "return 0", "return 42")
    fp_modules.fingerprints()  # A new process: fingerprints recomputed.
    assert not search([winner, loser])[0]


def test_ranks_that_fingerprint_differently_raise(fp_modules):
    fp_modules.load("fp_util", "def helper():\n    return 0\n")
    runners = fp_modules.load("fp_runners", _CANDIDATE_RUNNERS)
    candidates = [runners.Winner(), runners.Loser()]

    def agree(gather):
        tuner = SimpleNamespace(_dist=SimpleNamespace(tp_cp_allgather=gather))
        return autotuner.AutoTuner._agree_on_cache_hit(tuner, candidates, True,
                                                       (0, "1"))

    def rank(obj, **changes):
        fields = ("hit", "fingerprint", "signature", "choice", "groups", "uids",
                  "profiles", "profile_choices")
        values = dict(zip(fields, obj))
        values.update(changes)
        return tuple(values[f] for f in fields)

    def edited(obj):
        return rank(obj,
                    fingerprint=autotuner._EDITED_PREFIX + "0123456789abcdef")

    assert agree(lambda obj: [obj, obj])
    assert not agree(lambda obj: [obj, rank(obj, hit=False)])
    # Ranks whose entries chose differently tune again together.
    assert not agree(lambda obj: [obj, rank(obj, choice=(1, "2"))])
    # Ranks that may run different code must not tune or reuse together.
    with pytest.raises(RuntimeError, match="fingerprint the code"):
        agree(lambda obj: [rank(obj, fingerprint="other-fingerprint"), obj])
    # Nor ranks whose sources changed after they started, even when their
    # fingerprints match: what each loaded is no longer on disk.
    with pytest.raises(RuntimeError, match="changed after"):
        agree(lambda obj: [edited(obj), edited(obj)])
    # Nor ranks that pass the candidates in another order (PARALLEL deals
    # tactics by position) or with other tactic spaces.
    reordered = autotuner._rank_signature(list(reversed(candidates)))
    with pytest.raises(RuntimeError, match="different orders"):
        agree(lambda obj: [rank(obj, signature=reordered), obj])


class _Strategy(enum.IntEnum):
    ONE = -1


def test_ranks_of_one_communicator_agree_on_every_profile():
    """AllReduce runners name their communicator (rank_group): ranks of one
    agree on unique_id and on the tactic of every profile, ranks of another
    (a TP subgroup under Helix CP) may choose differently, and a tactic and
    its reloaded form (an Enum and its value) are one choice."""

    class _Grouped(_TacticSpaceRunner):

        def __init__(self, group, window=False):
            super().__init__()
            self.group, self.window = group, window

        def unique_id(self):
            return (self.group, self.window)

        def rank_group(self):
            return self.group

    profiles = [
        SimpleNamespace(get_opt_shapes=lambda shapes=shapes: shapes)
        for shapes in (((8, ), ), ((16, ), ))
    ]

    def agree(*ranks):
        """`ranks`: (runner, choice, choice per profile) of each rank; the
        result of the first."""
        objs = []
        for runner, choice, choices in ranks:
            # Collect each rank's part of the all-gather.
            collect = SimpleNamespace(_dist=SimpleNamespace(
                tp_cp_allgather=lambda obj: [objs.append(obj) or obj]))
            autotuner.AutoTuner._agree_on_cache_hit(collect, [runner], True,
                                                    choice, profiles, choices)
        tuner = SimpleNamespace(_dist=SimpleNamespace(
            tp_cp_allgather=lambda obj: objs))
        runner, choice, choices = ranks[0]
        return autotuner.AutoTuner._agree_on_cache_hit(tuner, [runner], True,
                                                       choice, profiles,
                                                       choices)

    nccl, symmetric = (0, "0"), (0, "8")
    # Helix TP2 x CP2: subgroups [0, 2] and [1, 3] keep their own winners.
    assert agree((_Grouped((0, 2)), nccl, [nccl, nccl]), (_Grouped(
        (1, 3)), symmetric, [symmetric, symmetric]), (_Grouped(
            (0, 2)), nccl, [nccl, nccl]), (_Grouped(
                (1, 3)), symmetric, [symmetric, symmetric]))
    # Within one, another profile chosen differently is tuned again, even
    # when this call's profile agrees: native caches take every profile.
    assert not agree((_Grouped((0, 2)), nccl, [nccl, nccl]), (_Grouped(
        (0, 2)), nccl, [nccl, symmetric]))
    # Ranks of one communicator that build the runner differently raise.
    with pytest.raises(RuntimeError, match="communicator"):
        agree((_Grouped((0, 2)), nccl, [nccl, nccl]), (_Grouped(
            (0, 2), window=True), nccl, [nccl, nccl]))
    # A tactic and the value a cache file reloads it as are one choice.
    assert autotuner._tactic_repr(_Strategy.ONE) == autotuner._tactic_repr(-1)


def test_rank_signature_leaves_out_rank_local_unique_ids():
    """Under Helix CP an AllReduce runner's unique_id names its rank's TP
    subgroup; ranks of one tuning group still agree."""

    class _Grouped(_TacticSpaceRunner):

        def __init__(self, group):
            super().__init__()
            self.group = group

        def unique_id(self):
            return (self.group, )

    assert autotuner._rank_signature([_Grouped(
        (0, 2))]) == autotuner._rank_signature([_Grouped((1, 3))])
    assert autotuner._rank_signature([_TacticSpaceRunner(
        ("limited", ))]) != autotuner._rank_signature(
            [_TacticSpaceRunner(("all", ))])


def test_rank_signature_tells_factory_runners_apart(fp_modules, monkeypatch):
    """Runners one factory builds share a class name: ranks that pass them in
    different orders cannot tune them together (PARALLEL deals by
    position)."""
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "build")
    runners = fp_modules.load("fp_runners", _CLOSURE_RUNNERS)
    narrow = runners.factory(runners.narrow)()
    wide = runners.factory(runners.wide)()
    assert autotuner._rank_signature(
        [narrow, wide]) != (autotuner._rank_signature([wide, narrow]))


def test_profiles_whose_entries_chose_differently_are_tuned_again():
    profiles = [
        SimpleNamespace(get_opt_shapes=lambda shapes=shapes: shapes)
        for shapes in (((1, ), ), ((2, ), ), ((3, ), ))
    ]
    scope = ("unique_id", ("()", ))

    def hits(mine, theirs, their_scope=scope):
        tuner = SimpleNamespace(_dist=SimpleNamespace(
            tp_cp_allgather=lambda obj: [obj, (their_scope, obj[1], theirs)]))
        return autotuner.AutoTuner._profile_hits_on_all_ranks(
            tuner, profiles, mine, scope)

    mine = [(0, "1"), (0, "1"), None]
    theirs = [(0, "1"), (0, "2"), (0, "1")]
    assert hits(mine, theirs) == [True, False, False]
    # A rank of another scope (another communicator) may choose differently,
    # but a profile one rank misses is still tuned by all.
    assert hits(mine, theirs, ("group", ((1, 3), ))) == [True, True, False]


def test_parallel_deals_tactics_in_one_order_on_every_rank():
    """Ranks whose get_valid_tactics lists the same tactics in different
    orders still deal each to exactly one rank."""
    shares = []
    for rank, tactics in enumerate(([1, 2], [2, 1])):
        tuner = SimpleNamespace(_is_distributed=lambda: True,
                                _tuning_group=lambda rank=rank: (2, rank))
        shares += autotuner.AutoTuner._maybe_parallelize_tactics(
            tuner, tactics, DistributedTuningStrategy.PARALLEL)
    assert sorted(shares) == [1, 2]


@pytest.mark.parametrize("mode", ["code", "build"])
@pytest.mark.parametrize("source,narrow_impl,wide_impl", [
    (_FACTORY_RUNNERS, "Narrow", "Wide"),
    (_CLOSURE_RUNNERS, "narrow", "wide"),
],
                         ids=["bases", "captured"])
def test_search_cache_keeps_same_named_candidates_apart(fp_modules, monkeypatch,
                                                        source, narrow_impl,
                                                        wide_impl, mode):
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", mode)
    runners = fp_modules.load("fp_runners", source)
    narrow = runners.factory(getattr(runners, narrow_impl))()
    wide = runners.factory(getattr(runners, wide_impl))()
    candidates = [narrow, wide]
    cache = autotuner.AutoTuner().profiling_cache
    cache.cache = {
        cache.get_cache_key("test_same_named",
                            wide,
                            _TACTIC_SPACE_SHAPES,
                            TuningConfig(),
                            candidates=candidates): (1, 1, 1.0)
    }
    # Wide's tactic 1 stays Wide's: Narrow only runs tactic 0.
    assert cache.search_cache("test_same_named", candidates,
                              _TACTIC_SPACE_SHAPES,
                              TuningConfig()) == (True, 1, 1, 1.0)


def test_save_cache_leaves_out_entries_of_edited_sources(tmp_path):
    src = autotuner.AutoTuner().profiling_cache
    current = ("test_edited", "R", "()", ((16, 64), ), "0123456789abcdef")
    edited = ("test_edited", "R", "()", ((32, 64), ),
              autotuner._EDITED_PREFIX + "0123456789abcdef")
    src.cache = {current: (0, 1, 1.0), edited: (0, 2, 1.0)}
    # Nor do they reach another rank (merge, broadcast, pipeline send).
    assert src.shareable(src.cache) == {current: (0, 1, 1.0)}
    path = str(tmp_path / "cache.json")
    src.save_cache(path, rank=0)
    dst = autotuner.AutoTuner().profiling_cache
    dst.load_cache(path, rank=0)
    assert dst.cache == {current: (0, 1, 1.0)}


def test_save_cache_drops_entries_another_build_keyed(tmp_path):
    path = str(tmp_path / "cache.json")
    old = ("test_superseded", "R", "()", ((16, 64), ), "aaaaaaaaaaaaaaaa")
    other_shape = ("test_superseded", "R", "()", ((32, 64), ),
                   "aaaaaaaaaaaaaaaa")
    src = autotuner.AutoTuner().profiling_cache
    src.cache = {old: (0, 1, 1.0), other_shape: (0, 1, 1.0)}
    src.save_cache(path, rank=0)
    dst = autotuner.AutoTuner().profiling_cache
    dst.load_cache(path, rank=0)
    # Tuned again here under another build: the entry it replaces goes.
    # Entries of the same runner build tuned among other candidates, or on
    # another GPU architecture, stay.
    new = (*old[:4], "bbbbbbbbbbbbbbbb")
    among_others = (*old[:4], "cccccccccccccccc/bbbbbbbbbbbbbbbb/dddddddd")
    other_gpu = (*old[:4], "aaaaaaaaaaaaaaaa@sm100")
    dst.cache.update({among_others: (1, 3, 1.0), other_gpu: (0, 4, 1.0)})
    dst[new] = (0, 2, 1.0)
    # Nor are the replaced entries exchanged with other ranks.
    assert old not in dst.shareable(dst.cache)
    dst.save_cache(path, rank=0)
    final = autotuner.AutoTuner().profiling_cache
    final.load_cache(path, rank=0)
    assert final.cache == {
        new: (0, 2, 1.0),
        other_shape: (0, 1, 1.0),
        among_others: (1, 3, 1.0),
        other_gpu: (0, 4, 1.0)
    }


def test_cache_key_mode_none_leaves_the_fingerprint_out(monkeypatch):
    monkeypatch.setattr(autotuner, "_CACHE_KEY_MODE", "none")
    cache = autotuner.AutoTuner().profiling_cache
    key = _tactic_space_cache_key(cache, _TacticSpaceRunner())
    assert key == ("test_tactic_space", "_TacticSpaceRunner", "()", ((16,
                                                                      64), ))
    cache.cache = {key: (0, 7, 1.0)}
    assert list(cache.entries_of("test_tactic_space", _TacticSpaceRunner)) == [
        ("()", ((16, 64), ), (0, 7, 1.0))
    ]


def test_choose_one_round_trip_among_two_runner_classes():
    """Entries tuned among two runner classes are found when serving: every
    store and lookup keys them by both classes' code."""
    x, w = torch.randn(M, 64), torch.randn(64, 128)
    runners = [
        GemmRunner(),
        GemmRunnerWithAttributes(block_size=128, num_warps=4)
    ]
    config = TuningConfig()
    op = "test_two_runner_classes"
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune():
        tuned = tuner.choose_one(op, runners, config, [x, w])
    assert tuner.profiling_cache.search_cache(op, runners, (x.shape, w.shape),
                                              config)[0]
    assert tuner.choose_one(op, runners, config, [x, w]) == tuned


def test_entries_of_skips_older_key_layouts_and_fingerprints():
    cache = autotuner.AutoTuner().profiling_cache
    current = _tactic_space_cache_key(cache, _TacticSpaceRunner())
    op, name, unique_id = current[:3]
    cache.cache = {
        current: (0, 7, 1.0),
        (op, name, unique_id, ((32, 64), )): (0, 8, 1.0),  # Older layout.
        (op, name, unique_id, ((48, 64), ), "0" * 16): (0, 9, 1.0),
        ("other_op", *current[1:]): (0, 10, 1.0),
    }
    assert list(cache.entries_of(op, _TacticSpaceRunner)) == [
        (unique_id, current[3], (0, 7, 1.0))
    ]


def test_mxfp8_sync_registers_only_entries_for_the_current_code(monkeypatch):
    from tensorrt_llm._torch.custom_ops import torch_custom_ops as ops

    registered = []

    class _NativeRunner:

        def register_tactic(self, m, n, k, tactic):
            registered.append((m, n, k, tactic))

    runner_cls = ops.MXFP8GemmRunner
    monkeypatch.setattr(runner_cls, "runner_dict", {"weights": _NativeRunner()})
    tuner = autotuner.AutoTuner()
    current = autotuner._key_fingerprint(
        runner_cls, autotuner._candidates_fingerprint((runner_cls, )))

    def key(m, *fingerprint):
        return (ops._MXFP8_AUTOTUNED_OP, runner_cls.__name__, "weights",
                ((m, 64), (m, 2), (32, 64)), *fingerprint)

    tuner.profiling_cache.cache = {
        key(16, current): (0, 3, 1.0),
        key(32): (0, 4, 1.0),  # The key layout before code fingerprints.
        key(48, "0" * 16): (0, 5, 1.0),  # Tuned against other code.
    }
    runner_cls.sync_all_tactic_caches(tuner)
    assert registered == [(16, 32, 64, 3)]


def test_search_cache_misses_across_tactic_spaces():
    cache = autotuner.AutoTuner().profiling_cache
    guided = _TacticSpaceRunner(("nvmmh", "x", 3))
    cache.cache[_tactic_space_cache_key(cache, guided)] = (0, 7, 1.0)

    def search(runner):
        return cache.search_cache("test_tactic_space", [runner],
                                  _TACTIC_SPACE_SHAPES, TuningConfig())

    assert search(guided) == (True, 0, 7, 1.0)
    assert not search(_TacticSpaceRunner())[0]
    assert not search(_TacticSpaceRunner(("nvmmh", "x", 5)))[0]


def test_tactic_space_key_survives_cache_save_and_load(tmp_path):
    src = autotuner.AutoTuner().profiling_cache
    key = _tactic_space_cache_key(src,
                                  _TacticSpaceRunner(("nvmmh", "0.1.0.27", 3)))
    src.cache[key] = (0, 7, 1.0)
    path = str(tmp_path / "cache.json")
    src.save_cache(path, rank=0)

    dst = autotuner.AutoTuner().profiling_cache
    dst.load_cache(path, rank=0)
    assert dst.cache[key][1] == 7


def test_nvmmh_tactic_space_key_follows_the_policy(monkeypatch):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    tuner = AutoTuner.get()
    monkeypatch.setattr(ops.cutedsl_nvmmh, "IS_NVMMH_AVAILABLE", True)
    monkeypatch.setattr(ops.cutedsl_nvmmh, "library_version", lambda: "x")
    monkeypatch.setattr(tuner, "nvmmh_config", None)
    assert ops._nvmmh_tactic_space_key() is None
    monkeypatch.setattr(tuner, "nvmmh_config", NvMMHConfig(max_tactics=3))
    assert ops._nvmmh_tactic_space_key() == ("nvmmh", "x", 3)
    monkeypatch.setattr(ops.cutedsl_nvmmh, "IS_NVMMH_AVAILABLE", False)
    assert ops._nvmmh_tactic_space_key() is None


def test_nvfp4_unified_runner_keys_only_the_cutedsl_tactic_space(monkeypatch):
    from tensorrt_llm._torch.custom_ops import torch_custom_ops

    monkeypatch.setattr(torch_custom_ops, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(torch_custom_ops,
                        "_nvmmh_tactic_space_key",
                        lambda: ("nvmmh", "x", 3),
                        raising=False)
    runner = torch_custom_ops.NVFP4GemmUnifiedRunner
    assert runner(0, torch.bfloat16,
                  ["cutlass", "cutedsl"]).tactic_space_key() == ("nvmmh", "x",
                                                                 3)
    assert runner(0, torch.bfloat16,
                  ["cutlass", "cublaslt"]).tactic_space_key() is None


def test_locality_domain_runner_forwards_the_tactic_space_key():
    from tensorrt_llm._torch.locality_domain.autotune import \
        LocalityDomainConcurrentTunableRunner

    inner = _TacticSpaceRunner()
    wrapper = LocalityDomainConcurrentTunableRunner(
        inner, SimpleNamespace(num_partitions=2), 2, lambda *args: None)
    assert wrapper.tactic_space_key() is None
    inner.space_key = ("nvmmh", "x", 3)
    assert wrapper.tactic_space_key() == ("nvmmh", "x", 3)


def test_configure_autotuner_nvmmh_installs_the_engine_policy(monkeypatch):
    from tensorrt_llm._torch.pyexecutor.engine.model_setup import \
        configure_autotuner_nvmmh
    from tensorrt_llm.llmapi.llm_args import AutoTunerNvMMHConfig

    tuner = AutoTuner.get()
    monkeypatch.setattr(tuner, "nvmmh_config", None)
    configure_autotuner_nvmmh(
        SimpleNamespace(autotuner_nvmmh_config=AutoTunerNvMMHConfig(
            max_tactics=7)))
    assert tuner.nvmmh_config == NvMMHConfig(max_tactics=7)
    configure_autotuner_nvmmh(SimpleNamespace(autotuner_nvmmh_config=None))
    assert tuner.nvmmh_config is None


def test_merge_nvmmh_rankings_picks_the_lowest_scoring_head():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import \
        _merge_nvmmh_rankings

    def ranking(name, *scores):
        return [(SimpleNamespace(score=score), f"{name}{i}")
                for i, score in enumerate(scores)]

    def merge(*rankings):
        return [name for _, name in _merge_nvmmh_rankings(list(rankings))]

    # Each ranking keeps its own order, even where it is not sorted by score.
    assert merge(ranking("a", 1, 4, 2),
                 ranking("b", 3, 5)) == ["a0", "b0", "a1", "a2", "b1"]
    # Ties go to the earlier ranking.
    assert merge(ranking("a", 1, 1), ranking("b", 1)) == ["a0", "a1", "b0"]
    assert merge(ranking("a"), ranking("b", 2)) == ["b0"]
    assert merge() == []


def test_select_tactics_counts_configs_with_new_tactics():
    from tensorrt_llm._torch.custom_ops.cutedsl_matmul_heuristics import \
        select_tactics

    ranked = [(None, ["a", "b"]), (None, ["b"]), (None, ["b", "c"]),
              (None, ["d"])]
    # The second config only repeats "b", so it is not counted.
    assert select_tactics(ranked, 1) == ["a", "b"]
    assert select_tactics(ranked, 2) == ["a", "b", "c"]
    assert select_tactics(ranked, 10) == ["a", "b", "c", "d"]
    assert select_tactics([], 3) == []


@pytest.mark.parametrize("max_tactics", [True, 0, -1, "5", 2.0])
def test_nvmmh_config_rejects_non_positive_integers(max_tactics):
    with pytest.raises(ValueError, match="positive integer"):
        NvMMHConfig(max_tactics=max_tactics)


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


def test_trtllm_gen_moe_dummy_topk_is_redrawn_at_every_bucket():
    """Buckets larger than the tuned call get their own dummy routing, not the
    tuned call's rows repeated."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
        FP8BlockScaleMoERunner, prepare_dummy_topk_and_hook)

    num_tokens, top_k, hidden, num_experts = 8, 10, 256, 512
    hidden_states = torch.zeros(num_tokens, hidden, device="cuda")
    topk_ids = torch.randn(num_tokens, num_experts,
                           device="cuda").topk(top_k).indices.to(torch.int32)
    topk_weights = torch.ones(num_tokens, top_k, device="cuda")
    tuner = AutoTuner.get()
    with autotune():
        logits, weights, ids, config = prepare_dummy_topk_and_hook(
            topk_weights, topk_ids, hidden_states, None, 1,
            FP8BlockScaleMoERunner.get_tuning_config(1), top_k, num_experts,
            num_experts, None, None, None)
    # FP8 block-scale runner input layout; the expert weights keep their shape.
    scale = torch.ones(hidden // 128, num_tokens, device="cuda")
    weights_stack = [torch.zeros(1, device="cuda")] * 4
    inputs = [logits, None, hidden_states, scale, *weights_stack, weights, ids]
    larger = 0
    for profile in tuner._optimization_profiles(config, inputs):
        tensors = config.inputs_pre_hook(
            tuner._prepare_input_tensors(profile, inputs))
        rows = tensors[2].shape[0]
        assert tensors[-1].shape == tensors[-2].shape == (rows, top_k)
        if rows > num_tokens:
            larger += 1
            distinct = len({tuple(r) for r in tensors[-1].tolist()})
            assert distinct > num_tokens, (
                f"{rows}-row bucket profiled with {distinct} distinct routing "
                f"rows (the tuned call had {num_tokens})")
    assert larger


def test_fp4_moe_tunes_on_dummy_topk_and_runs_on_runtime_topk(monkeypatch):
    """With precomputed top-k (attention DP), the tuner gets the local-shard dummy top-k and the
    kernel the runtime top-k: restoring the runtime tensors must not reach the tuner's inputs."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import \
        FP4BlockScaleMoERunner

    seen = {}

    def fake_choose_one(self, custom_op, runners, tuning_config, inputs,
                        **kwargs):
        seen["tuner"] = list(inputs)
        return runners[0], -1

    def fake_forward(self,
                     inputs,
                     tactic,
                     output=None,
                     prerouted=None,
                     lamport_out=None):
        seen["kernel"] = list(inputs)
        return [torch.zeros(1, device="cuda")]

    monkeypatch.setattr(AutoTuner, "choose_one", fake_choose_one)
    monkeypatch.setattr(FP4BlockScaleMoERunner, "forward", fake_forward)
    num_tokens, top_k = 8, 4
    num_experts, local_num_experts, local_expert_offset = 32, 8, 8
    hidden_states = torch.zeros(num_tokens,
                                64,
                                dtype=torch.uint8,
                                device="cuda")
    # Expert 0 lies outside this rank's shard; the dummy never picks it.
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device="cuda")
    topk_weights = torch.ones(num_tokens,
                              top_k,
                              dtype=torch.bfloat16,
                              device="cuda")
    # Weights, scales and biases: the op only passes them on.
    passthrough = [torch.zeros(1, device="cuda")] * 13
    scalars = (num_experts, top_k, None, None, 128, local_expert_offset,
               local_num_experts, None, 1, True)

    with autotune():
        torch.ops.trtllm.fp4_block_scale_moe_runner(None,
                                                    None,
                                                    hidden_states,
                                                    *passthrough,
                                                    *scalars,
                                                    topk_weights=topk_weights,
                                                    topk_ids=topk_ids,
                                                    use_dp=True)

    tuner_ids = seen["tuner"][-1]
    assert tuner_ids is not topk_ids and seen["tuner"][0] is None
    assert ((tuner_ids >= local_expert_offset) &
            (tuner_ids < local_expert_offset + local_num_experts)).all()
    assert seen["kernel"][-1] is topk_ids
    assert seen["kernel"][-2] is topk_weights


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


@pytest.mark.parametrize("tp_size,cp_size", [(1, 4), (2, 2)])
def test_parallel_tactics_split_across_helix_cp_ranks(tp_size, cp_size):
    """Helix CP ranks share the PARALLEL tactic sweep with the TP ranks."""
    from tensorrt_llm.mapping import CpType

    world_size = tp_size * cp_size
    tactics = list(range(11))
    shares = []
    for rank in range(world_size):
        tuner = autotuner.AutoTuner()
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=tp_size,
                          cp_size=cp_size,
                          cp_config={"cp_type": CpType.HELIX})
        tuner.setup_distributed_state(mapping, dist=object())
        assert tuner._is_distributed()
        assert tuner._tuning_group() == (world_size, rank)
        shares.append(
            tuner._maybe_parallelize_tactics(
                tactics, DistributedTuningStrategy.PARALLEL))
        assert tuner._maybe_parallelize_tactics(
            tactics, DistributedTuningStrategy.MERGE) == tactics

    # Every tactic is tuned on exactly one rank.
    assert sorted(t for share in shares for t in share) == tactics


def test_parallel_tactics_split_skips_non_helix_cp_ranks():
    """Non-Helix CP keeps the TP-only tuning group."""
    from tensorrt_llm.mapping import CpType

    tuner = autotuner.AutoTuner()
    mapping = Mapping(world_size=4,
                      rank=3,
                      tp_size=1,
                      cp_size=4,
                      cp_config={"cp_type": CpType.ULYSSES})
    tuner.setup_distributed_state(mapping, dist=object())
    assert tuner._tuning_group() == (1, 0)
    assert not tuner._is_distributed()
    tactics = list(range(5))
    assert tuner._maybe_parallelize_tactics(
        tactics, DistributedTuningStrategy.PARALLEL) == tactics


@pytest.mark.parametrize("num_tactics", [0, 1, 3, 15, 16, 17])
@pytest.mark.parametrize("offset", [0, 1, 7, 15])
def test_parallel_tactics_short_list_partitions_without_duplicates(
        num_tactics: int, offset: int) -> None:
    from types import SimpleNamespace

    tactics = list(range(num_tactics))
    shards = []
    for rank in range(16):
        tuner = SimpleNamespace(_is_distributed=lambda: True,
                                _tuning_group=lambda: (16, rank))
        selected = autotuner.AutoTuner._maybe_parallelize_tactics(
            tuner, tactics, DistributedTuningStrategy.PARALLEL, offset=offset)
        shards.append(selected)
    assert sorted(tactic for shard in shards for tactic in shard) == tactics
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert sum(not shard for shard in shards) == max(0, 16 - num_tactics)


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


def _autotune_dist_allgather_worker(world_size, init_file):
    import torch.distributed as dist

    from tensorrt_llm._torch.visual_gen.mapping import _VisualGenAutotuneDist
    rank = tensorrt_llm.mpi_rank()
    dist.init_process_group(backend="gloo",
                            init_method=f"file://{init_file}",
                            world_size=world_size,
                            rank=rank)
    try:
        # A TP-only gather would return one rank instead of the whole world.
        d = _VisualGenAutotuneDist(
            Mapping(world_size=world_size,
                    rank=rank,
                    tp_size=1,
                    pp_size=world_size))
        return d.tp_cp_allgather(f"rank{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_visual_gen_autotune_dist_world_allgather(mpi_pool_executor, tmp_path):
    # _VisualGenAutotuneDist.tp_cp_allgather must gather every rank's object over
    # the default world group (not the single-rank tp subgroup), and construct
    # without running TorchDist.__init__.
    world_size = 2
    init_file = str(tmp_path / "gloo-store")
    results = list(
        mpi_pool_executor.map(_autotune_dist_allgather_worker,
                              [world_size] * world_size,
                              [init_file] * world_size))
    for got in results:
        assert got == ["rank0", "rank1"]


def test_deep_gemm_jit_warmup_buckets():
    # A bucket left out leaves a DeepGemm layout uncompiled, so it compiles
    # under traffic while holding the GIL. The layout is constant on each
    # 16-aligned window of M, so every window must hold a bucket.
    for max_m in (4096, 6144, 8192):
        b = deep_gemm_jit_warmup_buckets(max_m)
        assert list(b) == sorted(set(b)), "must be sorted and duplicate-free"
        assert b[-1] == max_m, "the bound itself must be a bucket"

        assert all(hi - lo <= 16 for lo, hi in zip(b, b[1:])), \
            "no gap may exceed 16: it would skip a 16-aligned window of M"
        windows = {(m - 1) // 16 for m in b}
        missing = [j for j in range((max_m - 1) // 16 + 1) if j not in windows]
        assert not missing, \
            f"windows with no bucket (M from {[16 * j + 1 for j in missing]})"

    # Clamping keeps a small bound from skipping warmup for larger M,
    # and a huge one from exploding the bucket count.
    assert deep_gemm_jit_warmup_buckets(256) == deep_gemm_jit_warmup_buckets(
        4096)
    # But a decode-only worker (M = batch x MTP) stays in the low band.
    assert max(deep_gemm_jit_warmup_buckets(32)) < 128
    assert deep_gemm_jit_warmup_buckets(16384) == deep_gemm_jit_warmup_buckets(
        8192)


def _record_profile_graph_pools(monkeypatch):
    """Record the pool of every profiling capture and of every owner release / abandon."""
    events = {"capture": [], "release": [], "abandon": []}
    real_capture = autotuner.nccl_window_graph_capture
    real_release = autotuner.release_nccl_window_graph_owner
    real_abandon = autotuner.abandon_nccl_window_graph_owner

    @contextlib.contextmanager
    def capture(graph, pool, **kwargs):
        events["capture"].append(pool)
        with real_capture(graph, pool, **kwargs):
            yield

    def release(pool):
        events["release"].append(pool)
        real_release(pool)

    def abandon(pool):
        events["abandon"].append(pool)
        real_abandon(pool)

    monkeypatch.setattr(autotuner, "nccl_window_graph_capture", capture)
    monkeypatch.setattr(autotuner, "release_nccl_window_graph_owner", release)
    monkeypatch.setattr(autotuner, "abandon_nccl_window_graph_owner", abandon)
    return events


def test_profiling_captures_share_one_graph_pool_per_tuning_pass(monkeypatch):
    """All profiling captures of one choose_one() tuning pass capture into one
    CUDA graph memory pool, each capture still releases its NCCL-window owner,
    and the pool is dropped when the pass ends."""
    events = _record_profile_graph_pools(monkeypatch)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    x = torch.randn(M, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(0, 0, (4, 8, 16, 32)), ))

    pools_per_pass = []
    for op in ("autotuner_test::shared_pool_a",
               "autotuner_test::shared_pool_b"):
        for v in events.values():
            v.clear()
        with autotune():
            tuner.choose_one(op, [GemmRunner()], config, [x, w])
        # 4 profiles x 3 tactics, at least one capture each.
        assert len(events["capture"]) >= 12, events["capture"]
        assert len(set(events["capture"])) == 1, (
            f"one tuning pass must use one pool; got {set(events['capture'])}")
        assert events["release"] == events["capture"]
        assert events["abandon"] == []
        assert tuner._profile_graph_pool is None
        assert tuner._profile_graph_pool_depth == 0
        pools_per_pass.append(events["capture"][0])
    assert pools_per_pass[0] != pools_per_pass[1], (
        "a new tuning pass must not reuse the dropped pool")


def test_failed_profiling_capture_moves_to_a_fresh_graph_pool(monkeypatch):
    """A capture that fails abandons its owner, and the captures after it use a
    new pool rather than the one the failed capture used."""

    class CaptureFailingRunner(GemmRunner):

        def forward(self, /, inputs, *, tactic=-1, **kwargs):
            if tactic == 0 and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("tactic 0 fails under capture")
            return super().forward(inputs, tactic=tactic, **kwargs)

    events = _record_profile_graph_pools(monkeypatch)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    x = torch.randn(M, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    with autotune():
        tuner.choose_one("autotuner_test::fresh_pool_after_failure",
                         [CaptureFailingRunner()], TuningConfig(), [x, w])

    captures = events["capture"]
    assert len(events["abandon"]) == 1, events
    failed_pool = events["abandon"][0]
    # The failed capture is the last one in its pool.
    failed_at = len(captures) - 1 - captures[::-1].index(failed_pool)
    assert captures[:failed_at + 1] == [failed_pool] * (failed_at + 1)
    after = captures[failed_at + 1:]
    assert after, "tactic 1 must still be profiled after tactic 0 fails"
    assert len(set(after)) == 1 and after[0] != failed_pool, captures
    assert events["release"] == captures[:failed_at] + after
    assert tuner._profile_graph_pool is None


def test_parallel_split_deals_pairs_across_runners(monkeypatch):
    """PARALLEL shares the (runner, tactic) pairs of all runners evenly.

    Each rank of the group profiles only its share of a profile's pairs.
    Splitting each runner's tactics on its own would put the first tactic of
    every runner on group rank 0, so ranks with low indices would build and
    time most of the candidates while the others wait at the merge. With fewer
    pairs than ranks, a rank dealt none logs no warning: the merge brings it
    the winner.
    """

    class ListedTacticsRunner(TunableRunner):

        def __init__(self, name: str, num_tactics: int):
            self.name, self.num_tactics = name, num_tactics

        def unique_id(self):
            return (self.name, )

        def get_valid_tactics(self, inputs, profile, **kwargs) -> List[int]:
            return list(range(self.num_tactics))

        def forward(self, /, inputs, *, tactic: int = -1, **kwargs):
            return inputs[0]

    timed = {}

    def record_pair(self, runner, inputs, tactic, **kwargs):
        timed.setdefault(self._tuning_group()[1], []).append(
            (runner.name, tactic))
        return 1.0

    warnings = []
    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", record_pair)
    monkeypatch.setattr(autotuner.logger, "warning_once",
                        lambda *args, **kwargs: warnings.append(args))
    group_size = 4
    op = "autotuner_test::parallel_deal"
    config = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL)
    profile = OptimizationProfile([[StaticDim(8), StaticDim(64)]])
    x = torch.zeros(8, 64)

    def deal(runners):
        timed.clear()
        for rank in range(group_size):
            tuner = autotuner.AutoTuner()
            tuner.setup_distributed_state(Mapping(world_size=group_size,
                                                  rank=rank,
                                                  tp_size=group_size),
                                          dist=object())
            # choose_one sets these up before it profiles.
            tuner.stats.tuned_op_profiled_configs[op] = 0
            tuner.stats.failed_profiling_count[op] = set()
            tuner._profile_runners_impl(op, runners, [x], profile, config)
        return tuner, [timed.get(rank, []) for rank in range(group_size)]

    runners = [
        ListedTacticsRunner("cublas", 1),
        ListedTacticsRunner("direct", 7),
        ListedTacticsRunner("splitk", 3)
    ]
    tuner, shares = deal(runners)
    # Every pair is timed on exactly one rank, and the shares differ by at most one.
    assert sorted(pair for share in shares for pair in share) == sorted(
        (r.name, t) for r in runners for t in range(r.num_tactics))
    assert max(map(len, shares)) - min(map(len, shares)) <= 1, shares
    # The deal starts at the profile's start rank.
    start = tuner._parallel_deal_start(op, profile,
                                       DistributedTuningStrategy.PARALLEL)
    assert shares[start][0] == ("cublas", 0), (start, shares)
    # Two pairs on four ranks: two ranks are dealt none and do not warn.
    _, shares = deal([ListedTacticsRunner("direct", 2)])
    assert sorted(map(len, shares)) == [0, 0, 1, 1], shares
    assert warnings == [], warnings


@pytest.mark.parametrize("group_size", [2, 4, 16])
def test_parallel_rank_dealt_no_pair_runs_the_merged_winner(
        monkeypatch: pytest.MonkeyPatch, group_size: int) -> None:
    """choose_one on a PARALLEL rank that profiles nothing returns the winner
    the group found: with one (runner, tactic) pair, all ranks the deal skips
    get it from the merge without warning."""

    class SingleTacticRunner(TunableRunner):

        def get_valid_tactics(self, inputs, profile, **kwargs) -> List[int]:
            return [7]

        def forward(self, /, inputs, *, tactic: int = -1, **kwargs):
            return inputs[0]

    profiled = []

    def record_pair(self, runner, inputs, tactic, **kwargs):
        profiled.append((self._tuning_group()[1], tactic))
        return 1.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", record_pair)
    warnings = []
    monkeypatch.setattr(autotuner.logger, "warning_once",
                        lambda *args, **kwargs: warnings.append(args))
    gathered = {}

    class _AllgatherDist:
        """tp_cp_allgather of the ranks run so far, in rank order, matched call by call."""

        def __init__(self, rank):
            self.rank = rank

        def tp_cp_allgather(self, obj):
            calls = gathered.setdefault(self.rank, [])
            calls.append(obj)
            n = len(calls) - 1
            return [
                gathered[r][n] for r in sorted(gathered) if len(gathered[r]) > n
            ]

    op = "autotuner_test::parallel_rank_without_pairs"
    config = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL)
    runner = SingleTacticRunner()
    inputs = [torch.zeros(8, 64), torch.zeros(64, 32)]

    def tuner_for(rank):
        tuner = autotuner.AutoTuner()
        tuner.setup_distributed_state(Mapping(world_size=group_size,
                                              rank=rank,
                                              tp_size=group_size),
                                      dist=_AllgatherDist(rank))
        tuner.is_tuning_mode = True
        return tuner

    profile = tuner_for(0)._optimization_profiles(config, inputs)[0]
    profiling_rank = tuner_for(0)._parallel_deal_start(
        op, profile, DistributedTuningStrategy.PARALLEL)
    results = {}
    ranks = [profiling_rank
             ] + [rank for rank in range(group_size) if rank != profiling_rank]
    for rank in ranks:
        results[rank] = tuner_for(rank).choose_one(op, [runner], config, inputs)
    assert profiled == [(profiling_rank, 7)], profiled
    assert all(result == (runner, 7) for result in results.values()), results
    assert warnings == [], warnings


class _NamedTactics(TunableRunner):
    """A runner the test names, with the tactics it lists, optionally run by
    the communicator `group`."""

    def __init__(self, name, tactics, group=None):
        self.name, self.tactics, self.group = name, tactics, group

    def unique_id(self):
        return (self.name, self.group)

    def rank_group(self):
        return self.group

    def get_valid_tactics(self, inputs, profile, **kwargs) -> List[int]:
        return list(self.tactics)

    def forward(self, /, inputs, *, tactic: int = -1, **kwargs):
        return inputs[0]


class _ThreadedRanks:
    """The ranks of one tuning group, run as threads that exchange
    tp_cp_allgather and tp_cp_broadcast through a barrier."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots = [None] * size

    def dist(self, rank):

        def allgather(obj):
            self.slots[rank] = obj
            self.barrier.wait()
            gathered = list(self.slots)
            self.barrier.wait()
            return gathered

        return SimpleNamespace(
            tp_cp_allgather=allgather,
            tp_cp_broadcast=lambda obj, root: allgather(obj)[root])

    def run(self, fn):
        """fn(rank) on every rank at once; the results in rank order."""
        results, errors = [None] * self.size, []

        def target(rank):
            try:
                results[rank] = fn(rank)
            except BaseException as e:  # noqa: BLE001 - raised below
                errors.append(e)
                self.barrier.abort()

        threads = [
            threading.Thread(target=target, args=(rank, ))
            for rank in range(self.size)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return results


_GROUP_OP = "autotuner_test::group"


def _choose_on_ranks(monkeypatch,
                     config,
                     runners_of,
                     entries_of,
                     times,
                     inputs=None,
                     calls=1):
    """choose_one `calls` times on each rank of a group of len(entries_of),
    the ranks running together. `runners_of(rank)`: the rank's candidates;
    `entries_of[rank]`: {(runner index, opt shapes): (tactic, ms)} its cache
    holds first (loaded from a file); `times[(runner name, tactic)]`: what
    profiling measures. Returns each rank's (runner name, tactic), the
    (rank, runner name, tactic) of each profiling, and the tuners."""
    inputs = inputs or [torch.zeros(8, 64)]
    profiled = []
    lock = threading.Lock()

    def measure(self, runner, inputs, tactic, **kwargs):
        with lock:
            profiled.append((self._tuning_group()[1], runner.name, tactic))
        return times[(runner.name, tactic)]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", measure)
    size = len(entries_of)
    group = _ThreadedRanks(size)
    tuners = []
    for rank in range(size):
        tuner = autotuner.AutoTuner()
        tuner.setup_distributed_state(Mapping(world_size=size,
                                              rank=rank,
                                              tp_size=size),
                                      dist=group.dist(rank))
        tuner.is_tuning_mode = True
        runners = runners_of(rank)
        for (index, shapes), (tactic, ms) in entries_of[rank].items():
            tuner.profiling_cache.cache[tuner.profiling_cache.get_cache_key(
                _GROUP_OP,
                runners[index],
                shapes,
                config,
                apply_map_to_tuning_buckets=False,
                candidates=runners)] = (index, tactic, ms)
        tuners.append((tuner, runners))

    def choose(rank):
        tuner, runners = tuners[rank]
        for _ in range(calls):
            runner, tactic = tuner.choose_one(_GROUP_OP, runners, config,
                                              inputs)
        return runner.name, tactic

    return group.run(choose), profiled, tuners


_SHAPES = ((8, 64), )


def _two_runners(rank):
    return [_NamedTactics("a", [1]), _NamedTactics("b", [7])]


def test_broadcast_peers_take_the_roots_winner(monkeypatch):
    """A peer that held another candidate's entry for a profile drops it
    when the root's arrives: the lookup would find it first."""
    config = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.BROADCAST)
    for root_entries in ({(1, _SHAPES): (7, 0.5)}, {}):
        results, _, _ = _choose_on_ranks(
            monkeypatch, config, _two_runners,
            [root_entries, {
                (0, _SHAPES): (1, 0.5)
            }], {
                ("a", 1): 2.0,
                ("b", 7): 1.0
            })
        assert results == [("b", 7)] * 2, (root_entries, results)


@pytest.mark.parametrize("strategy", [
    DistributedTuningStrategy.MERGE,
    DistributedTuningStrategy.PARALLEL,
])
def test_profiles_tuned_again_compare_only_this_rounds_measurements(
        monkeypatch, strategy):
    """An older entry (another candidate's, or one with a faster old time on
    a rank the PARALLEL deal leaves idle) does not win over what the ranks
    measure when they tune the profile again."""
    config = TuningConfig(distributed_tuning_strategy=strategy)
    stale = {(0, _SHAPES): (1, 0.01)}
    results, profiled, _ = _choose_on_ranks(monkeypatch, config, _two_runners,
                                            [{}, stale, stale, stale], {
                                                ("a", 1): 2.0,
                                                ("b", 7): 1.0
                                            })
    assert profiled
    assert results == [("b", 7)] * 4, results


def test_communicators_keep_their_own_winners(monkeypatch):
    """Ranks of different communicators (TP subgroups under Helix CP) may
    hold different winners: a warm cache is not tuned again."""
    config = TuningConfig(
        distributed_tuning_strategy=DistributedTuningStrategy.MERGE)
    groups = [(0, 2), (1, 3), (0, 2), (1, 3)]

    def runners_of(rank):
        return [_NamedTactics("ar", [0, 8], groups[rank])]

    entries = [{
        (0, _SHAPES): (0 if rank % 2 == 0 else 8, 1.0)
    } for rank in range(4)]
    results, profiled, _ = _choose_on_ranks(monkeypatch,
                                            config,
                                            runners_of,
                                            entries, {},
                                            calls=2)
    assert profiled == []
    assert results == [("ar", 0), ("ar", 8), ("ar", 0), ("ar", 8)]


def test_a_cache_hit_still_agrees_on_the_other_profiles(monkeypatch):
    """A hit on this call's profile does not skip a profile the ranks chose
    differently: ops such as AllReduce register every profile's entry in a
    native cache."""
    config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(0, 0, (8, 16), lambda m: 8
                                                if m <= 8 else 16), ),
        distributed_tuning_strategy=DistributedTuningStrategy.MERGE)
    small, large = ((8, 64), ), ((16, 64), )

    def runners_of(rank):
        return [_NamedTactics("ar", [0, 8])]

    _, profiled, tuners = _choose_on_ranks(monkeypatch, config, runners_of, [{
        (0, small): (0, 1.0),
        (0, large): (0, 1.0)
    }, {
        (0, small): (0, 1.0),
        (0, large): (8, 1.0)
    }], {
        ("ar", 0): 2.0,
        ("ar", 8): 1.0
    })
    assert sorted(t for _, _, t in profiled) == [0, 0, 8, 8]
    chosen = [
        tuner.profiling_cache.search_cache(_GROUP_OP, runners, large, config,
                                           False)[:3]
        for tuner, runners in tuners
    ]
    assert chosen == [(True, 0, 8)] * 2, chosen


def test_cache_excluded_ops_profile_every_profile_in_process(monkeypatch):
    """An op whose tuning runs JIT side effects (exclude_from_cache) profiles
    every profile in the process, though entries loaded from a file hit."""
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, (8, 16), lambda m: 8 if m <= 8 else 16), ),
                          exclude_from_cache=True)
    profiled = []
    monkeypatch.setattr(
        AutoTuner, "_profile_single_kernel", lambda self, runner, inputs,
        tactic, **kwargs: profiled.append(tactic) or float(tactic))
    tuner = autotuner.AutoTuner()
    tuner.is_tuning_mode = True
    runners = [_NamedTactics("jit", [1, 2])]
    op = "autotuner_test::excluded"
    for shapes in (((8, 64), ), ((16, 64), )):
        tuner.profiling_cache.cache[tuner.profiling_cache.get_cache_key(
            op, runners[0], shapes, config,
            apply_map_to_tuning_buckets=False)] = (0, 2, 1.0)
    tuner.choose_one(op, runners, config, [torch.zeros(8, 64)])
    assert sorted(profiled) == [1, 1, 2, 2]


def test_parallel_search_cache_picks_fastest_runner():
    """After a PARALLEL merge a shape can hold one entry per runner.

    The lookup must return the fastest of them; other strategies keep one
    entry per shape and return the first runner found.
    """
    runners = [
        GemmRunner(),
        GemmRunnerWithAttributes(block_size=128, num_warps=4)
    ]
    shapes = (torch.Size([M, 64]), torch.Size([64, 128]))
    for strategy, expected in ((DistributedTuningStrategy.PARALLEL, 1),
                               (DistributedTuningStrategy.INDEPENDENT, 0)):
        config = TuningConfig(distributed_tuning_strategy=strategy)
        cache = autotuner.AutoTunerProfilingCache()
        for runner, tactic, ms in ((runners[0], 0, 2.0), (runners[1], 1, 1.0)):
            cache[cache.get_cache_key("autotuner_test::parallel_lookup",
                                      runner,
                                      shapes,
                                      config,
                                      candidates=runners)] = (0, tactic, ms)
        hit, runner_id, tactic, _ = cache.search_cache(
            "autotuner_test::parallel_lookup", runners, shapes, config)
        assert hit and runner_id == expected, (strategy, runner_id)


def test_parallel_deal_start_is_stable_and_spread():
    """Each profile's PARALLEL deal starts at a rank derived from the op and
    the shapes: the same on every rank and in every process, and spread over
    the group so a runner's costly tactics do not land on the same ranks."""
    group_size = 16
    starts = []
    for m in range(1, 129):
        profile = autotuner.OptimizationProfile(
            [[autotuner.StaticDim(m),
              autotuner.StaticDim(7168)],
             [autotuner.StaticDim(7168),
              autotuner.StaticDim(2048)]])
        per_rank = set()
        for rank in range(group_size):
            tuner = autotuner.AutoTuner()
            tuner.setup_distributed_state(Mapping(world_size=group_size,
                                                  rank=rank,
                                                  tp_size=group_size),
                                          dist=object())
            per_rank.add(
                tuner._parallel_deal_start("autotuner_test::deal", profile,
                                           DistributedTuningStrategy.PARALLEL))
        assert len(per_rank) == 1, (m, per_rank)
        starts.append(per_rank.pop())
    assert set(starts) == set(range(group_size)), sorted(set(starts))


def test_share_jit_builds_sets_aside_profiles_built_elsewhere(
        monkeypatch, tmp_path):
    """With DeepGEMM's builds shared across ranks, a rank runs the profiles it
    claims first. One whose kernel another rank is building is set aside rather
    than failed and runs after the others; a profile another rank claimed runs
    last, once that rank has run it."""
    calls: List[int] = []

    class SharedBuildRunner(TunableRunner):

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
                    **kwargs) -> torch.Tensor:
            m = inputs[0].shape[0]
            calls.append(m)
            if m in (2, 4) and calls.count(m) == 1:
                raise RuntimeError(
                    f"command failed with exit code 75:\n{jit_share.BUSY_MARK} 0"
                )
            return inputs[0]

    monkeypatch.setattr(jit_share, "enabled", lambda: True)
    monkeypatch.setattr(jit_share, "start_fraction", lambda: 0.0)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    tuner = AutoTuner.get()
    tuner.clear_cache()
    op = "autotuner_test::share_jit_builds"
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, (1, 2, 4, 8)), ),
                          share_jit_builds=True)
    real_claim = jit_share.SharedSweep.claim
    # Another rank holds the claim on the last profile (M = 8) and has run it.
    monkeypatch.setattr(jit_share.SharedSweep, "claim",
                        lambda self, i: i != 3 and real_claim(self, i))
    monkeypatch.setattr(jit_share.SharedSweep, "wait_done",
                        lambda self, i: calls.append(-i))
    with autotune():
        tuner.choose_one(op, [SharedBuildRunner()], config,
                         [torch.zeros(8, 64, device="cuda")])
    assert calls == [1, 2, 4, 2, 4, -3, 8], calls
    assert not tuner.stats.failed_profiling_count[op]
    assert len(tuner.profiling_cache.get_specific_custom_op(op)) == 4
    assert jit_share._DEFER_ENV not in os.environ
    assert jit_share._DEADLINE_ENV not in os.environ


def test_share_jit_builds_sets_aside_a_timed_profile_whose_kernel_is_busy(
        monkeypatch, tmp_path):
    """The ranks split a sweep by profile when its first profile has one tactic.
    A later profile with several tactics is timed, and a tactic whose kernel
    another rank is building does not fail there either: the profile is set
    aside and timed again, every tactic, after the others."""
    calls: List[tuple] = []

    class MixedTacticsRunner(TunableRunner):

        def unique_id(self):
            return ()

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [0] if inputs[0].shape[0] == 1 else [0, 1]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1,
                    **kwargs) -> torch.Tensor:
            m = inputs[0].shape[0]
            calls.append((m, tactic))
            if (m, tactic) == (2, 0) and calls.count((2, 0)) == 1:
                raise RuntimeError(
                    f"command failed with exit code 75:\n{jit_share.BUSY_MARK} 0"
                )
            if tactic == 1:
                torch.cuda._sleep(1_000_000)  # The slower tactic.
            return inputs[0] + 1

    monkeypatch.setattr(jit_share, "enabled", lambda: True)
    monkeypatch.setattr(jit_share, "start_fraction", lambda: 0.0)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    tuner = AutoTuner.get()
    tuner.clear_cache()
    op = "autotuner_test::share_jit_builds_timed_busy"
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, (1, 2, 4)), ),
                          share_jit_builds=True)
    runners = [MixedTacticsRunner()]
    with autotune():
        tuner.choose_one(op, runners, config,
                         [torch.zeros(4, 64, device="cuda")])
    assert not tuner.stats.failed_profiling_count[op]
    after_m4 = max(i for i, (m, _) in enumerate(calls) if m == 4)
    assert {t for m, t in calls[after_m4:] if m == 2} == {0, 1}, calls
    assert len(tuner.profiling_cache.get_specific_custom_op(op)) == 3
    _, tactic = tuner.choose_one(op, runners, config,
                                 [torch.zeros(2, 64, device="cuda")])
    assert tactic == 0
    tuner.clear_cache()


def test_deep_gemm_jit_share_builds_each_kernel_once(tmp_path, monkeypatch):
    """DeepGEMM's compiler wrapper builds a kernel in the process that claims
    it. Others copy the published cubin, report busy while it is being built if
    asked to defer, and build it themselves when the claimant does not publish
    it within their wait, which an enclosing deadline cuts short."""
    # The keys computed here must match the wrapper's, which runs without the salt.
    monkeypatch.delenv(jit_share._SALT_ENV, raising=False)
    calls = tmp_path / "calls"
    compiler = tmp_path / "fake_nvcc"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        f"open({str(calls)!r}, 'a').write(' '.join(args) + '\\n')\n"
        "if args == ['--version']:\n"
        "    print('Cuda compilation tools, release 13.0')\n"
        "    sys.exit(0)\n"
        "out = args[args.index('--output-file') + 1]\n"
        "open(out, 'wb').write(open(args[0], 'rb').read()[::-1])\n")
    compiler.chmod(0o755)
    kernels = tmp_path / "share" / "kernels"
    env = {
        k: v
        for k, v in os.environ.items() if not k.startswith("TLLM_JIT_SHARE_")
    }
    env.update({
        jit_share.SHARE_DIR_ENV: str(tmp_path / "share"),
        jit_share._REAL_COMPILER_ENV: str(compiler),
    })

    def kernel(name: str, source: str) -> List[str]:
        work = tmp_path / name
        work.mkdir()
        (work / "kernel.cu").write_text(source)
        return [
            str(work / "kernel.cu"), "--cubin", "--output-file",
            str(work / "kernel.cubin"), "-O3"
        ]

    def run(argv: List[str], **extra_env) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-S", jit_share.__file__] +
                              argv,
                              env=dict(env, **extra_env),
                              capture_output=True,
                              text=True)

    def builds() -> int:
        return calls.read_text().count("--cubin") if calls.exists() else 0

    def claim(argv: List[str]) -> None:
        kernels.mkdir(parents=True, exist_ok=True)
        path = kernels / f"{jit_share._kernel_key(argv[0], argv)}.claim"
        path.write_text("another process\n")

    assert "release 13.0" in run(["--version"]).stdout

    for name in ("a", "b"):  # built once, then copied
        argv = kernel(name, "kernel one")
        assert run(argv).returncode == 0
        assert (tmp_path / name / "kernel.cubin").read_text() == "eno lenrek"
    assert builds() == 1

    argv = kernel("c", "kernel two")
    claim(argv)
    busy = run(argv, **{jit_share._DEFER_ENV: "1"})
    assert busy.returncode == jit_share.BUSY_EXIT_CODE
    assert jit_share.BUSY_MARK in busy.stdout and builds() == 1
    assert run(argv, **{jit_share._WAIT_ENV: "0.2"}).returncode == 0
    assert builds() == 2

    argv = kernel("d", "kernel three")
    claim(argv)
    past = {jit_share._WAIT_ENV: "600", jit_share._DEADLINE_ENV: repr(0.0)}
    assert run(argv, **past).returncode == 0
    assert builds() == 3


def test_deep_gemm_jit_share_builds_here_when_the_directory_fails(
        tmp_path, monkeypatch):
    """Where the share directory cannot be used (here its kernels directory is
    a file), DeepGEMM's compiler wrapper runs the real compiler, as without
    sharing. A kernel built here that cannot be published (the quota is full)
    is still this build's result, and is marked failed for the others."""
    calls = tmp_path / "calls"
    compiler = tmp_path / "fake_nvcc"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        f"open({str(calls)!r}, 'a').write(' '.join(args) + '\\n')\n"
        "out = args[args.index('--output-file') + 1]\n"
        "open(out, 'wb').write(open(args[0], 'rb').read()[::-1])\n")
    compiler.chmod(0o755)
    share = tmp_path / "share"
    share.mkdir()
    (share / "kernels").write_text("not a directory")
    work = tmp_path / "work"
    work.mkdir()
    (work / "kernel.cu").write_text("kernel one")
    output = work / "kernel.cubin"
    argv = [str(work / "kernel.cu"), "--cubin", "--output-file", str(output)]
    env = {
        k: v
        for k, v in os.environ.items() if not k.startswith("TLLM_JIT_SHARE_")
    }
    env.update({
        jit_share.SHARE_DIR_ENV: str(share),
        jit_share._REAL_COMPILER_ENV: str(compiler),
    })
    run = subprocess.run([sys.executable, "-I", "-S", jit_share.__file__] +
                         argv,
                         env=env,
                         capture_output=True,
                         text=True)
    assert run.returncode == 0, run.stderr
    assert output.read_text() == "eno lenrek"
    assert calls.read_text().count("--cubin") == 1

    def quota_full(output: str, cubin: str) -> None:
        raise OSError(errno.EDQUOT, os.strerror(errno.EDQUOT))

    monkeypatch.setattr(jit_share, "_publish", quota_full)
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    assert jit_share._build(str(compiler), argv, str(output),
                            str(kernels / "k.cubin"),
                            str(kernels / "k.failed")) == 0
    assert (kernels / "k.failed").exists()
    assert calls.read_text().count("--cubin") == 2


def test_jit_share_wrapper_needs_a_private_directory(tmp_path):
    """DeepGEMM runs the compiler wrapper found in its directory, so the wrapper
    is written to and used from a directory only this user can write to: one
    another user could write to, or a symlink, is refused."""
    if os.statvfs(tmp_path).f_flag & os.ST_NOEXEC:
        pytest.skip("the temporary directory does not allow executing files")
    private = tmp_path / "private"
    wrapper = jit_share._write_wrapper(str(private))
    assert wrapper is not None and os.access(wrapper, os.X_OK)
    assert os.stat(private).st_mode & 0o077 == 0
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    assert jit_share._write_wrapper(str(shared)) is None
    link = tmp_path / "link"
    link.symlink_to(private)
    assert jit_share._write_wrapper(str(link)) is None


def test_jit_share_wrapper_stays_where_only_this_user_can_replace_it(
        tmp_path, monkeypatch):
    """DeepGEMM runs the compiler wrapper at its path for the life of the
    process, so the wrapper goes where no other user can rename a directory on
    that path: this user's directory in a sticky (or private) temporary
    directory first, the share directory only if no one else can write to it."""
    sticky = tmp_path / "tmp"
    sticky.mkdir()
    sticky.chmod(0o1777)
    monkeypatch.setattr(tempfile, "tempdir", str(sticky))
    mine = str(sticky / f"tllm_jit_share_{os.getuid()}")
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    assert jit_share._wrapper_dirs(str(shared)) == [mine]
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o700)
    assert jit_share._wrapper_dirs(str(private)) == [mine, str(private / "bin")]
    sticky.chmod(0o777)  # others could rename this user's directory in it
    assert jit_share._wrapper_dirs(str(private)) == [str(private / "bin")]


def test_jit_share_install_moves_past_an_unusable_wrapper_directory(
        tmp_path, monkeypatch):
    """install() writes DeepGEMM's compiler wrapper to the first of its
    directories that can hold it and makes that wrapper DeepGEMM's compiler."""
    if os.statvfs(tmp_path).f_flag & os.ST_NOEXEC:
        pytest.skip("the temporary directory does not allow executing files")
    bad, good = str(tmp_path / "bad"), str(tmp_path / "good")
    write = jit_share._write_wrapper
    monkeypatch.setattr(jit_share, "_wrapper_dirs",
                        lambda share_dir: [bad, good])
    monkeypatch.setattr(jit_share, "_write_wrapper", lambda d: None
                        if d == bad else write(d))
    monkeypatch.setattr(jit_share, "_installed", False)
    compiler = tmp_path / "nvcc"
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o700)
    # install() sets these; registering them first makes the test restore them.
    for name in ("DG_JIT_NVCC_COMPILER", "DG_JIT_CACHE_DIR",
                 jit_share._SALT_ENV, jit_share._PROCESS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path / "share"))
    monkeypatch.setenv(jit_share._REAL_COMPILER_ENV, str(compiler))
    assert jit_share.install()
    assert os.path.dirname(os.environ["DG_JIT_NVCC_COMPILER"]) == good
    assert os.environ[jit_share._REAL_COMPILER_ENV] == str(compiler)


def test_jit_share_install_needs_a_directory_other_users_cannot_write(
        tmp_path, monkeypatch):
    """Every process runs the kernels it finds in the share directory, so
    install() shares builds only when the directory, and the ones it holds
    kernels in, belong to this user and other users cannot write to them."""
    if os.statvfs(tmp_path).f_flag & os.ST_NOEXEC:
        pytest.skip("the temporary directory does not allow executing files")
    monkeypatch.setattr(jit_share, "_wrapper_dirs",
                        lambda share_dir: [str(tmp_path / "bin")])
    monkeypatch.setattr(jit_share, "_installed", False)
    compiler = tmp_path / "nvcc"
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o700)
    # install() sets these; registering them first makes the test restore them.
    for name in ("DG_JIT_NVCC_COMPILER", "DG_JIT_CACHE_DIR",
                 jit_share._SALT_ENV, jit_share._PROCESS_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    share = tmp_path / "share"
    share.mkdir()
    share.chmod(0o777)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(share))
    monkeypatch.setenv(jit_share._REAL_COMPILER_ENV, str(compiler))
    with pytest.warns(RuntimeWarning, match="other users cannot write"):
        assert not jit_share.install()
    assert "DG_JIT_NVCC_COMPILER" not in os.environ
    share.chmod(0o755)
    (share / "cute").chmod(0o777)
    with pytest.warns(RuntimeWarning, match="cute"):
        assert not jit_share.install()
    (share / "cute").chmod(0o755)
    assert jit_share.install()
    assert os.environ["DG_JIT_CACHE_DIR"] == str(share / "deep_gemm")


def test_shared_kernel_cache_exports_and_loads(monkeypatch, tmp_path):
    """A CuTe DSL kernel stored in a shared runner cache is exported, and the
    cache of another process (a second instance here) loads it instead of
    compiling it. A kernel claimed but not yet exported is busy inside
    defer_busy_builds, and a miss once the wait runs out otherwise."""
    exported = {}

    def export(kernel, obj, name, meta):
        exported[name] = kernel
        for path in (meta, obj):
            with open(path, "w") as f:
                f.write(name)

    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(jit_share._WAIT_ENV, "0.1")
    monkeypatch.setattr(jit_share, "_cute_identity", lambda: "test")
    monkeypatch.setattr(jit_share, "_export_kernel", export)
    monkeypatch.setattr(jit_share, "_load_kernel", lambda obj, name, meta:
                        ("loaded", exported[name]))
    first, second = (jit_share.SharedKernelCache("Runner") for _ in range(2))

    def claims() -> int:
        return len(list((tmp_path / "cute").glob("*.claim")))

    with jit_share.sharing_kernels():
        assert (1, "tile") not in first  # claimed: this process compiles it
        first[(1, "tile")] = "kernel"
        assert (1, "tile") in second
        assert second[(1, "tile")] == ("loaded", "kernel")
        # Another process claimed (2, "tile") and has not exported it yet.
        open(second._paths((2, "tile"))[2], "w").close()
        with jit_share.defer_busy_builds(), pytest.raises(jit_share.KernelBusy):
            _ = (2, "tile") in second
        assert (2, "tile") not in second
        before = claims()
        assert object() not in first  # its repr holds an address: not shared
        assert claims() == before
    assert (1, "tile") not in jit_share.SharedKernelCache("Runner")


def test_shared_kernel_cache_marks_a_raised_compile_failed(
        monkeypatch, tmp_path):
    """A claimed kernel that is never stored (its compile raised) is marked
    failed at the thread's next lookup, or when tuning ends: this process
    compiles it again without waiting for its own claim, and another process
    (a second instance here) stops waiting for it."""
    waits = []
    wait = jit_share._wait

    def record_wait(obj, failed):
        waits.append(os.path.exists(failed))
        return wait(obj, failed)

    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(jit_share._WAIT_ENV, "600")
    monkeypatch.setattr(jit_share, "_cute_identity", lambda: "test")
    monkeypatch.setattr(jit_share, "_wait", record_wait)
    here, other = (jit_share.SharedKernelCache("Runner") for _ in range(2))
    with jit_share.sharing_kernels():
        assert (1, "tile") not in here  # claimed; its compile raises
        assert (1, "tile") not in here  # compiled here again
        assert (1, "tile") not in other
        assert (2, "tile") not in here  # claimed; tuning ends before a store
    assert waits == [True]
    assert len(list((tmp_path / "cute").glob("*.failed"))) == 2


def test_jit_share_waits_end_at_the_deadline(monkeypatch, tmp_path):
    """A wait for another process's sweep item or kernel ends at the deadline
    of an enclosing wait_until, however long the wait setting, so that a
    claimant that hangs delays the others by one wait at most."""
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(jit_share._WAIT_ENV, "600")
    monkeypatch.setattr(jit_share, "_cute_identity", lambda: "test")
    sweep = jit_share.SharedSweep("sweep")
    assert sweep.claim(0) and not sweep.claim(0)
    cache = jit_share.SharedKernelCache("Runner")
    (tmp_path / "cute").mkdir()
    open(cache._paths((1, "tile"))[2], "w").close()  # claimed elsewhere
    with jit_share.sharing_kernels(), jit_share.wait_until(0.0):
        assert not sweep.wait_done(0)
        assert (1, "tile") not in cache
    assert jit_share._DEADLINE_ENV not in os.environ
    sweep.done(0)
    assert sweep.wait_done(0)


def test_jit_share_builds_here_when_the_directory_fails(monkeypatch, tmp_path):
    """Where the share directory cannot be used (here the directories in it are
    files, so that nothing can be created in them), a process builds each
    kernel itself, as without sharing: a pass starts at the front, a sweep's
    items are all this process's, and a CuTe DSL lookup is a miss whose store
    keeps the kernel here."""
    for name in ("slots", "sweeps", "cute"):
        (tmp_path / name).write_text("not a directory")
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(jit_share, "_slot", None)
    monkeypatch.setattr(jit_share, "_cute_identity", lambda: "test")
    assert jit_share.start_fraction() == 0.0
    sweep = jit_share.SharedSweep("sweep")
    assert sweep.claim(0) and sweep.claim(0)
    sweep.done(0)
    assert sweep.wait_done(0)
    cache = jit_share.SharedKernelCache("Runner")
    with jit_share.sharing_kernels():
        assert (1, "tile") not in cache
        cache[(1, "tile")] = "kernel"  # its export fails
        assert (1, "tile") in cache and cache[(1, "tile")] == "kernel"


def test_build_across_ranks_runs_only_claimed_items(monkeypatch, tmp_path):
    """With several tactics under INDEPENDENT tuning, a build pass runs each
    (profile, runner, tactic) this process claims once, before profiling, and
    leaves the items other processes claimed to them; profiling then times
    every pair."""
    ran: List[tuple] = []
    timed: List[tuple] = []

    class TwoTacticRunner(TunableRunner):

        def unique_id(self):
            return ()

        def get_valid_tactics(self, inputs: List[FakeTensor],
                              profile: OptimizationProfile,
                              **kwargs) -> List[int]:
            return [0, 1]

        def forward(self,
                    /,
                    inputs: List[torch.Tensor],
                    *,
                    tactic: int = -1,
                    **kwargs) -> torch.Tensor:
            ran.append((inputs[0].shape[0], tactic))
            return inputs[0]

    def time_pair(self, runner, inputs, tactic, tuning_config, **kwargs):
        timed.append((inputs[0].shape[0], tactic))
        return 1.0 + tactic

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", time_pair)
    monkeypatch.setattr(jit_share, "enabled", lambda: True)
    monkeypatch.setattr(jit_share, "start_fraction", lambda: 0.0)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    real_claim = jit_share.SharedSweep.claim
    # Other processes hold the odd items.
    monkeypatch.setattr(jit_share.SharedSweep, "claim",
                        lambda self, k: k % 2 == 0 and real_claim(self, k))
    tuner = AutoTuner.get()
    tuner.clear_cache()
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(0, 0,
                                                                  (2, 4)), ),
                          share_jit_builds=True)
    with autotune():
        tuner.choose_one("autotuner_test::build_across_ranks",
                         [TwoTacticRunner()], config,
                         [torch.zeros(4, 64, device="cuda")])
    # Items: (2, 0), (2, 1), (4, 0), (4, 1); this process holds 0 and 2.
    assert ran == [(2, 0), (4, 0)], ran
    assert sorted(timed) == [(2, 0), (2, 1), (4, 0), (4, 1)], timed


def test_jit_share_salt_ignores_file_times(tmp_path):
    """Copies of the same files installed at different times give the same
    salt, as each node installs its own copy; different contents do not."""
    for name, stamp in (("a", 1_000_000), ("b", 2_000_000)):
        (tmp_path / name).mkdir()
        compiler = tmp_path / name / "nvcc"
        compiler.write_text("compiler")
        os.utime(compiler, (stamp, stamp))
    salt_a = jit_share._salt(str(tmp_path / "a" / "nvcc"))
    assert salt_a == jit_share._salt(str(tmp_path / "b" / "nvcc"))
    (tmp_path / "b" / "nvcc").write_text("another compiler")
    assert salt_a != jit_share._salt(str(tmp_path / "b" / "nvcc"))


def test_excluded_op_reuses_the_sweep_profiled_in_this_process(monkeypatch):
    """Once this process has profiled an exclude_from_cache op's sweep, later
    tuning-mode calls hit those entries instead of walking the sweep again, and
    input tensors are only built for the profiles that missed the cache."""
    tuner = AutoTuner.get()
    tuner.clear_cache()
    counts = {"sweeps": 0, "inputs": 0}
    real_profiles = tuner._optimization_profiles
    real_inputs = tuner._prepare_input_tensors

    def profiles(*args, **kwargs):
        counts["sweeps"] += 1
        return real_profiles(*args, **kwargs)

    def inputs(*args, **kwargs):
        counts["inputs"] += 1
        return real_inputs(*args, **kwargs)

    monkeypatch.setattr(tuner, "_optimization_profiles", profiles)
    monkeypatch.setattr(tuner, "_prepare_input_tensors", inputs)
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, (4, 8, 16, 32)), ),
                          exclude_from_cache=True)
    w = torch.randn(64, 128, device="cuda")
    op = "autotuner_test::excluded_op_sweep_reuse"
    with autotune():
        tuner.choose_one(op, [GemmRunner()], config,
                         [torch.randn(M, 64, device="cuda"), w])
        assert counts == {"sweeps": 1, "inputs": 4}
        # The same op in later layers, at swept and unswept shapes.
        for m in (M, M // 2, M):
            tuner.choose_one(op, [GemmRunner()], config,
                             [torch.randn(m, 64, device="cuda"), w])
    assert counts == {"sweeps": 1, "inputs": 4}

    # An entry this process did not profile (e.g. received from a peer) does
    # not prove the op's JIT ran here, so the call walks the sweep again.
    tuner._keys_profiled_in_process.clear()
    with autotune():
        tuner.choose_one(op, [GemmRunner()], config,
                         [torch.randn(M, 64, device="cuda"), w])
    assert counts["sweeps"] == 2
    tuner.clear_cache()


class ThreadRecordingRunner(TunableRunner):
    """One tactic. Records the thread that ran each shape, as a runner whose
    first call per shape JIT-compiles a kernel would compile it there."""
    calls: list[tuple[int, int]] = []

    def get_valid_tactics(self, inputs: List[FakeTensor],
                          profile: OptimizationProfile, **kwargs) -> List[int]:
        return [0]

    def forward(self,
                /,
                inputs: List[torch.Tensor],
                *,
                tactic: int = -1,
                **kwargs) -> torch.Tensor:
        type(self).calls.append((inputs[0].shape[0], threading.get_ident()))
        return inputs[0] @ inputs[1]


@pytest.mark.parametrize("threads, env, expect_pool", [
    (0, None, False),
    (4, None, True),
    (4, "0", False),
])
def test_parallel_jit_warmup_runs_pending_profiles_on_a_thread_pool(
        monkeypatch, threads, env, expect_pool):
    """With parallel_jit_warmup, every not-yet-tuned profile runs once on a
    thread pool before the profiling loop, which still profiles every profile
    on the calling thread. TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP overrides it."""
    if env is None:
        monkeypatch.delenv("TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP", raising=False)
    else:
        monkeypatch.setenv("TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP", env)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    ThreadRecordingRunner.calls.clear()
    buckets = (4, 8, 16, 32, 64, 128)
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, buckets), ),
                          parallel_jit_warmup=threads)
    x = torch.randn(128, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    caller = threading.get_ident()
    with autotune():
        tuner.choose_one(f"autotuner_test::parallel_jit_warmup_{threads}_{env}",
                         [ThreadRecordingRunner()], config, [x, w])

    on_caller = {m for m, t in ThreadRecordingRunner.calls if t == caller}
    on_pool = sorted(m for m, t in ThreadRecordingRunner.calls if t != caller)
    assert on_caller == set(buckets)
    assert on_pool == (sorted(buckets) if expect_pool else [])
    tuner.clear_cache()


def test_parallel_jit_warmup_worker_failure_is_left_to_profiling(monkeypatch):
    """A profile that fails on the warmup pool does not fail choose_one: the
    profiling loop still runs every profile on the calling thread."""
    monkeypatch.delenv("TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP", raising=False)
    caller = threading.get_ident()

    class WorkerFailingRunner(ThreadRecordingRunner):
        calls: list[tuple[int, int]] = []

        def forward(self, /, inputs, *, tactic=-1, **kwargs):
            if threading.get_ident() != caller:
                raise RuntimeError("fails on warmup threads only")
            return super().forward(inputs, tactic=tactic, **kwargs)

    tuner = AutoTuner.get()
    tuner.clear_cache()
    buckets = (4, 8, 16, 32, 64, 128)
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, buckets), ),
                          parallel_jit_warmup=4)
    x = torch.randn(128, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    with autotune():
        tuner.choose_one("autotuner_test::parallel_jit_warmup_worker_failure",
                         [WorkerFailingRunner()], config, [x, w])

    assert {m for m, _ in WorkerFailingRunner.calls} == set(buckets)
    assert all(t == caller for _, t in WorkerFailingRunner.calls)
    assert not tuner.stats.failed_profiling_count.get(
        "autotuner_test::parallel_jit_warmup_worker_failure")
    tuner.clear_cache()


@pytest.mark.parametrize(
    "host_cpus, affinity, local_ranks, env_local, mpi_off, expected", [
        (144, 144, 4, None, False, 32),
        (64, 64, 8, None, False, 8),
        (64, 8, 8, None, False, 8),
        (64, 64, 1, "16", True, 4),
        (64, 64, RuntimeError("no node split"), None, False, 32),
    ])
def test_default_parallel_jit_warmup_threads(monkeypatch, host_cpus, affinity,
                                             local_ranks, env_local, mpi_off,
                                             expected):
    """parallel_jit_warmup=-1 uses the CPUs this rank may run on: unpinned ranks
    share the host's CPUs with the other ranks on the node, a rank pinned to a
    subset has it to itself, and the result is capped at 32."""

    def node_ranks():
        if isinstance(local_ranks, Exception):
            raise local_ranks
        return local_ranks

    monkeypatch.setattr(os, "cpu_count", lambda: host_cpus)
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda pid: set(range(affinity)))
    monkeypatch.setattr(autotuner, "local_mpi_size", node_ranks)
    monkeypatch.setattr(autotuner, "mpi_disabled", lambda: mpi_off)
    if env_local is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", env_local)
    assert autotuner._default_parallel_jit_warmup_threads() == expected


@pytest.mark.parametrize("sharing", [False, True])
def test_shared_jit_builds_replace_the_warmup_pool(monkeypatch, tmp_path,
                                                   sharing):
    """An op with both parallel_jit_warmup and share_jit_builds runs its sweep on
    the warmup pool only when the processes do not share their JIT builds. When
    they do, the claims split the builds and every profile runs on the calling
    thread."""
    monkeypatch.delenv("TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP", raising=False)
    monkeypatch.setattr(jit_share, "enabled", lambda: sharing)
    monkeypatch.setattr(jit_share, "start_fraction", lambda: 0.0)
    monkeypatch.setenv(jit_share.SHARE_DIR_ENV, str(tmp_path))
    tuner = AutoTuner.get()
    tuner.clear_cache()
    ThreadRecordingRunner.calls.clear()
    buckets = (4, 8, 16, 32, 64, 128)
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(
        0, 0, buckets), ),
                          parallel_jit_warmup=4,
                          share_jit_builds=True)
    x = torch.randn(128, 64, device="cuda")
    w = torch.randn(64, 128, device="cuda")
    caller = threading.get_ident()
    with autotune():
        tuner.choose_one(f"autotuner_test::shared_builds_or_pool_{sharing}",
                         [ThreadRecordingRunner()], config, [x, w])

    on_caller = {m for m, t in ThreadRecordingRunner.calls if t == caller}
    on_pool = sorted(m for m, t in ThreadRecordingRunner.calls if t != caller)
    assert on_caller == set(buckets)
    assert on_pool == ([] if sharing else sorted(buckets))
    tuner.clear_cache()
