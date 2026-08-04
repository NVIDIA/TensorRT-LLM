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
"""Coverage tests for deep_gemm_gen_tuning_buckets (nvbug 6550749).

The bucket list exists for exactly one purpose: to make DeepGEMM JIT-compile
every kernel config it will need *during autotuning at startup*, so that no
live iteration ever pays an ~2.2s nvcc stall. A bucket list with holes silently
defeats that -- the workload still runs, just with a multi-second spike on
whichever iteration first lands in an unwarmed band. On
minimax_m2.5_fp8 128,128 gpus:4 (GB300) that spike was 2316.6ms on a single
iteration, 101.4% of a 55.9% end-to-end regression, and it presented as
*instability* because whether a run pays it depends on request packing.

So the property under test is coverage, not a literal bucket list: for every M
the runtime can produce, some bucket must select the same DeepGEMM config.
Config selection steps only at M % 16 == 1 (see the derivation next to
DEEP_GEMM_BLOCK_M_QUANTUM in tensorrt_llm/_torch/utils.py), which makes that
property checkable in pure Python with no GPU.
"""

import pytest

from tensorrt_llm._torch.utils import DEEP_GEMM_BLOCK_M_QUANTUM, deep_gemm_gen_tuning_buckets


# Mirrors csrc/jit_kernels/heuristics/sm100.hpp:62-63 and :230: the band index
# of M under a given BLOCK_M. Two M values sharing a band select byte-identical
# kernels, so warming either one warms both.
def _band(m: int, block_m: int) -> int:
    return -(-m // block_m)  # ceil_div


# Every BLOCK_M DeepGEMM can choose on SM100. swap_ab walks
# lcm(16, block_m_multiple_of=1)..256 step 16 (256 is dropped by the tmem check
# at :123-128, but including it here only makes the test stricter); non-swap_ab
# picks from {32, 64, 128}.
_BLOCK_M_CANDIDATES = tuple(range(16, 257, 16)) + (32, 64, 128)


@pytest.mark.parametrize("max_num_tokens", [128, 512, 2048, 4096, 8192])
def test_every_reachable_m_shares_a_band_with_some_bucket(max_num_tokens):
    """The coverage property, checked exhaustively at stride 1.

    For each candidate BLOCK_M, every reachable M must land in a band that at
    least one bucket also lands in. This is what "the warmup is complete"
    means; it is not a statement about bucket count or spacing.
    """
    buckets = deep_gemm_gen_tuning_buckets(max_num_tokens)

    for block_m in _BLOCK_M_CANDIDATES:
        warmed = {_band(b, block_m) for b in buckets}
        holes = [m for m in range(1, max_num_tokens + 1) if _band(m, block_m) not in warmed]
        assert not holes, (
            f"max_num_tokens={max_num_tokens} BLOCK_M={block_m}: "
            f"{len(holes)} M values fall in bands no bucket warms, e.g. "
            f"{holes[:8]}. Each is an ~2.2s in-window nvcc stall (nvbug "
            f"6550749). If the stride grew past {DEEP_GEMM_BLOCK_M_QUANTUM}, "
            f"that is the cause."
        )


def test_stride_matches_the_block_m_quantum():
    """Guard the specific constant, so a future widening is a deliberate act.

    A stride of 128 (the pre-fix value) leaves ~1 band in 8 cold. Because the
    bands are 16 wide, only a stride that divides 16 can be complete.
    """
    buckets = deep_gemm_gen_tuning_buckets(4096)
    high = [b for b in buckets if b >= 128]
    strides = {b - a for a, b in zip(high, high[1:])}

    assert strides == {DEEP_GEMM_BLOCK_M_QUANTUM}, (
        f"expected a uniform stride of {DEEP_GEMM_BLOCK_M_QUANTUM} above 128, got {sorted(strides)}"
    )
    assert 16 % DEEP_GEMM_BLOCK_M_QUANTUM == 0, (
        f"DeepGEMM config bands are 16 wide; a stride of "
        f"{DEEP_GEMM_BLOCK_M_QUANTUM} cannot sample every band"
    )


def test_lower_clamp_survives_a_small_first_call():
    """The max(x, 4096) floor must stay -- it is not dead code.

    fp8SwapABGemmRunner leaves tune_max_num_tokens unset, so the autotuner
    passes the *current input size* here, not a maximum. Without the floor a
    small first call would warm nothing above 120 and every larger M would JIT
    mid-iteration -- the same bug in a different disguise.
    """
    for first_call_m in (1, 8, 64, 129, 540, 2048):
        buckets = deep_gemm_gen_tuning_buckets(first_call_m)
        if first_call_m >= 128:
            assert max(buckets) >= 4096, (
                f"first call M={first_call_m} warmed only up to "
                f"{max(buckets)}; the lower clamp is gone"
            )


@pytest.mark.parametrize("max_num_tokens", [4096, 5000, 5001, 6144, 8000, 8191, 8192, 9000])
def test_the_band_containing_max_num_tokens_is_warmed(max_num_tokens):
    """The top band must be covered -- it is the most-visited M, not an edge.

    Every full batch runs at M == max_num_tokens, so of all the bands this one
    is the likeliest to be hit. A half-open range(128, x, 16) covers none of it:
    the only multiple of 16 in a band [16k+1, 16k+16] is its *top*, so the list
    must reach a bucket >= x. Pre-fix this same hole was 128 wide.
    """
    buckets = deep_gemm_gen_tuning_buckets(max_num_tokens)
    effective = min(max(max_num_tokens, 4096), 8192)
    band_start = ((effective - 1) // DEEP_GEMM_BLOCK_M_QUANTUM) * DEEP_GEMM_BLOCK_M_QUANTUM + 1

    assert any(band_start <= b < band_start + DEEP_GEMM_BLOCK_M_QUANTUM for b in buckets), (
        f"max_num_tokens={max_num_tokens} (effective {effective}) "
        f"lives in band [{band_start}, "
        f"{band_start + DEEP_GEMM_BLOCK_M_QUANTUM - 1}] but the "
        f"highest bucket is {max(buckets)}: a full batch JITs "
        f"mid-iteration"
    )


def test_low_buckets_are_unchanged():
    """M < 128 was already covered; keep it byte-identical."""
    buckets = deep_gemm_gen_tuning_buckets(4096)
    assert [b for b in buckets if b < 128] == list(range(8, 128, 8))


def test_buckets_are_sorted_and_unique():
    """The autotuner de-dupes into a set, but duplicates would waste profiling
    iterations and mask a generator bug."""
    for max_num_tokens in (128, 512, 2048, 4096, 8192):
        buckets = deep_gemm_gen_tuning_buckets(max_num_tokens)
        assert list(buckets) == sorted(buckets)
        assert len(set(buckets)) == len(buckets)


# The M values measured compiling on a *warmed* GB300 cache before the fix,
# with the BLOCK_M the heuristic actually chose for each (job 2815335, shapes
# N=8192,K=3072 and N=3072,K=6144). M=540 is the one that landed inside the
# measured window and caused the 2316.6ms spike on iter 140; the other seven
# are the full residual set the stride-1 sweep found.
#
# These are pinned as *band starts*, not as bucket membership: what makes a
# band cold is that no bucket falls anywhere inside it, and each band here
# begins at its listed M and runs 16 wide.
_MEASURED_COLD_BANDS = (
    (540, 144),
    (129, 80),
    (161, 96),
    (193, 112),
    (257, 144),
    (385, 128),
    (401, 80),
    (1729, 128),
)


@pytest.mark.parametrize("m,block_m", _MEASURED_COLD_BANDS)
def test_measured_cold_band_now_has_a_bucket_inside_it(m, block_m):
    """Each band measured cold on GB300 must now contain a bucket.

    Note the assertion is "a bucket lands in [16k+1, 16k+16] around M", not
    "some bucket shares ceil_div(M, block_m)". The weaker form is vacuous: at
    BLOCK_M=144, ceil_div(512, 144) == ceil_div(540, 144) == 4, so the old
    stride-128 buckets appear to cover M=540 -- yet 540 demonstrably compiled a
    fresh kernel, because the heuristic picks a *different* BLOCK_M at 512 than
    at 540. Only a bucket physically inside the band warms it.
    """
    del block_m  # documents the measured selection; not needed by the check
    band_start = ((m - 1) // 16) * 16 + 1
    band = range(band_start, band_start + 16)
    buckets = deep_gemm_gen_tuning_buckets(2048)

    assert any(b in band for b in buckets), (
        f"no bucket inside [{band.start}, {band.stop - 1}], the band "
        f"containing M={m}: that band JIT-compiles on a live iteration "
        f"(nvbug 6550749)"
    )
