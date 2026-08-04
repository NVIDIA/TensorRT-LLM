# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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
"""Host-side routing for the GVR top-k decode family.

One kernel, three assist tiers selected by which emission inputs the
indexer epilogue produced for this step:

  * ``list``   - bucketed candidate list + packed seed row (v5): the
                 top-k pass never re-reads the row on a hit.
  * ``counts`` - packed seed row only ([rows, 8]: three lines + three
                 counts): one filtered row pass, no in-kernel counting.
  * ``rungs``  - closed-loop lines only (no emission): in-kernel
                 multi-line count, then the stock collect.
  * ``none``   - stock kernel (v1 path).

``plan_emission`` decides which tier the epilogue should emit for the
NEXT step (the emission tax is shape-dependent); ``pick_config`` maps
(tier, B, N, K) to concrete launch knobs for THIS step.

All thresholds are measured on B200 (f15 layer-complete grid,
2026-07-27, cold-L2 kernel-only protocol, validated on both the shared
grid dataset and first-party captures). They are deployment defaults,
not universal truths - keep them in one place so retuning is a
constant edit, not a logic edit.
"""

from dataclasses import dataclass
from typing import Optional

# ---- measured thresholds (B200) ------------------------------------------
#
# UNITS: every threshold here is fitted on KERNEL-ONLY time for both the
# indexer and the top-k. An earlier fit compared a wall-clock indexer
# (which carries a ~22us launch floor) against kernel-only top-k times;
# that inflated the apparent emission budget at small shapes and is what
# put the counts tier behind a batch gate it never needed.

# The kernel's fixed cost does not shrink with N, so once the stock
# kernel finishes under that floor no tier can win.
ASSIST_MIN_N_COMP = 2048

# Block-skip prefix pays for the counts tier from 65536 up, but not for
# the zero-emission rungs tier below 131072. Measured on captured V4
# rows (5 layers x all decode steps, nsys kernel-only, flash n_comp
# 65537): counts 16.1us with the prefix vs 17.3 without at batch
# 4/8/16, rungs 18.5 with vs 17.9 without. A synthetic-Gaussian A/B put
# the counts break-even a doubling later - the block-max distribution
# of real rows decides the pass rate, so set this from captures only.
SKIP_MIN_N_COUNTS = 65536  # va: attach block_max from here up
SKIP_MIN_N_RUNGS_FLASH = 131072  # vb (flash): bm pays from here
# Cluster split is a loss for the assist tiers below this point,
# block_max or not: at n_comp 65537 rungs measures 17.9 / 19.0 / 20.4us
# at cs 1 / 4 / 8 (batch 4) and cs8 spills to 31.5us at batch 16.
SKIP_CS_MIN_N_RUNGS = 196608  # vb: cluster split from here up

# Emission cost measured on the FP4 indexer KERNEL (nsys kernel-only,
# ABBA-interleaved NVTX blocks, batch 1..64 x ctx 32k..512k) as the delta
# of the same kernel with the emission outputs attached:
#
#   batch                1     2     4     8    16    64
#   counts (us)        +5.6  +4.4  +2.9  +2.0  +0.4   0.0
#   list, bucketed     +6.2  +6.2  +6.8 +10.5 +25.0  +63.1  (ctx 32k)
#                     +10.3 +14.7 +25.6 +44.5 +85.7 +417.8  (ctx 512k)
#
# The counts cost is a fixed reduction-latency chain that depends on
# BATCH ONLY and hides once there are enough rows to overlap it - it is
# not a fraction of the indexer, so it cannot price the tier out at
# scale. An earlier fit modelled it as 3% of a WALL-clock indexer time
# (which carries a ~22us launch floor) and so charged 12us at batch 64 /
# 512k, where the true cost is zero; that is what put the counts tier
# behind a batch gate. The list cost is real per-emitted-entry work and
# grows with batch and context alike. Parking the two tight lines above
# the score range (see gvr_ext.LIST_PARK_LINE) drops it 2.7-5x - every
# entry then lands in the one segment that claims through a per-warp
# window instead of an exact ballot - at top-k parity, which is what
# makes the list tier affordable past a single row.
LIST_EMIT_MIN_N = 65536  # shorter rows: the emission outweighs the saving
LIST_EMIT_MAX_B = 4  # past four rows the list stops repaying its emission
COUNTS_MIN_TOKENS = 524288  # B * raw length; below this the counts
# latency chain is exposed and the zero-emission rungs tier wins
RUNGS_ONLY_MIN_N = 16384  # short-row band where rungs also beats counts
RUNGS_ONLY_MAX_N = 49152

# Mid-row weak band: rows long enough that the stock kernel splits them
# across a cluster, but short enough (and at a small enough batch) that
# its split grid still fits one wave. The assist tiers cannot follow -
# splitting a row costs them more than the scan it saves, with or without
# the block-skip prefix - so the stock kernel wins outright and the
# epilogue should emit nothing. Narrowed after phase 4 got its coarse
# search and its boundary-class repair back: the tiers gained about
# 1.5us there, which is enough to take the upper half of the band back
# off the stock kernel (2 cells fall through now, down from 4).
ASSIST_WEAK_MIN_N = 49152
ASSIST_WEAK_MAX_N_SMALL_K = 98304  # k <= ASSIST_WEAK_K
ASSIST_WEAK_MAX_N_LARGE_K = 98304
ASSIST_WEAK_K = 512
ASSIST_WEAK_MAX_B = 8

# rungs-tier block_max pays only at small K: with K=1024 the tight-line
# pass rate runs too high and the prefix read is pure overhead.
RUNGS_BM_MAX_K = 512

# 512-thread build wins for list-hit rows at small K (work is O(list)).
SMALL_K_LIST_THREADS = 512
SMALL_K_MAX = 512

# GPC packing: cs=8 only while all row-clusters fit half the device
# (B=16 x cs8 wave-spill regression); cs4/2 keep a 10% headroom.
CS8_HALF_DEVICE = 2
CS_HEADROOM_NUM = 9
CS_HEADROOM_DEN = 10


@dataclass
class TopkRoute:
    """Launch knobs for one decode step of the GVR top-k kernel."""

    tier: str  # list | counts | rungs | none
    cluster_size: int = 1
    num_threads: Optional[int] = None  # None = runner heuristic
    attach_block_max: bool = False


def plan_emission(
    batch: int, n_comp: int, k: int, have_epilogue: bool, compress_ratio: int = 4
) -> str:
    """Which assist tier the indexer epilogue should emit this step.

    ``n_comp``: compressed row length (post compress_ratio) - the
    top-k kernel's N. Returns the tier name; the epilogue emits the
    matching buffers and the next top-k launch routes on them.
    """
    if n_comp < ASSIST_MIN_N_COMP:
        # short rows: the stock kernel is already under our fixed cost,
        # and this holds for the zero-emission rungs tier too - it is
        # the same kernel, so the floor is the same
        return "none"
    if have_epilogue and n_comp >= LIST_EMIT_MIN_N and batch <= LIST_EMIT_MAX_B:
        # checked before the weak band below: that band is about the stock
        # kernel out-scanning us, and a list hit never scans the row
        return "list"
    weak_max = ASSIST_WEAK_MAX_N_SMALL_K if k <= ASSIST_WEAK_K else ASSIST_WEAK_MAX_N_LARGE_K
    if batch <= ASSIST_WEAK_MAX_B and ASSIST_WEAK_MIN_N <= n_comp < weak_max:
        return "none"  # stock's split grid wins this band outright
    if (
        have_epilogue
        and batch * n_comp * compress_ratio >= COUNTS_MIN_TOKENS
        and not (RUNGS_ONLY_MIN_N <= n_comp < RUNGS_ONLY_MAX_N)
    ):
        return "counts"
    return "rungs"  # closed-loop lines cost nothing to carry


def pick_config(tier: str, batch: int, n_comp: int, k: int, num_sms: int) -> TopkRoute:
    """Map (tier, B, N, K) to launch knobs. Pure function of shape."""
    r = TopkRoute(tier=tier)
    if tier == "none":
        return r
    if tier == "list":
        if k <= SMALL_K_MAX:
            r.num_threads = SMALL_K_LIST_THREADS
        # list + block_max: miss rows fall back to a skip-walk instead
        # of a dense re-scan (measured -19% on pro long chains).
        r.attach_block_max = n_comp >= SKIP_MIN_N_COUNTS
        return r
    if tier == "counts":
        r.attach_block_max = n_comp >= SKIP_MIN_N_COUNTS
        return r
    # rungs (vb)
    if k <= RUNGS_BM_MAX_K and n_comp >= SKIP_MIN_N_RUNGS_FLASH:
        r.attach_block_max = True
    if n_comp >= SKIP_CS_MIN_N_RUNGS:
        if batch * 8 <= num_sms // CS8_HALF_DEVICE:
            r.cluster_size = 8
        elif batch * 4 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 4
        elif batch * 2 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 2
    return r
