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

# ---- measured thresholds (B200, f15 grid) --------------------------------

# Below this compressed length no assist tier can pay for itself. The
# kernel has a fixed-cost floor (~12us at K=1024, ~15us at K=512) that
# does not shrink with N, and by n_comp/K ~ 1 the stock kernel finishes
# under that floor - there is nothing left to win.
#
# f17 worst-step grid (2026-07-30, 162 cells, one B200, emission tax
# charged) grouped by selectivity n_comp/K:
#
#   n_comp/K   cells  geomean  worst  losing
#   1.0-1.5        9    0.793  0.717     9/9   <- pro 4k: 1027 candidates
#   1.5-3         18    1.004  0.908    8/18      for K=1024, a top-k that
#   3-6           18    1.031  0.910    2/18      selects nearly everything
#   6-20          36    1.094  0.943    5/36
#   20-80         36    1.174  0.941    2/36
#   80+           45    1.721  0.880    2/45
#
# Gating here costs flash 4k/8k (geomean 1.03, inside the +-3%
# run-to-run band) and takes the grid from geomean 1.218 / worst 0.717
# / 28 losing cells to 1.231 / 0.880 / 10.
ASSIST_MIN_N_COMP = 4096  # ~8k raw context at compress_ratio 4

# Block-skip prefix pays only when whole-row reads dominate.
SKIP_MIN_N_COUNTS = 65536  # va: attach block_max unconditionally here up
SKIP_MIN_N_RUNGS_FLASH = 131072  # vb (flash): bm pays from here
SKIP_CS_MIN_N_RUNGS = 196608  # vb: cluster split on top of bm from here

# Emission tax ~ B*N on the GEMM side: the candidate list is only worth
# emitting for latency-bound shapes (small B, long rows).
LIST_EMIT_MAX_B = 16
LIST_EMIT_MIN_N = 65536

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


def plan_emission(batch: int, n_comp: int, k: int, have_epilogue: bool) -> str:
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
    if not have_epilogue:
        return "rungs"  # closed-loop lines cost nothing to carry
    if batch <= LIST_EMIT_MAX_B and n_comp >= LIST_EMIT_MIN_N:
        return "list"  # latency-bound long rows: list pays big
    return "counts"  # near-free tax, wins almost everywhere


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
    if n_comp >= 65536:
        if batch * 8 <= num_sms // CS8_HALF_DEVICE:
            r.cluster_size = 8
        elif batch * 4 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 4
        elif batch * 2 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 2
    if r.attach_block_max and n_comp < SKIP_CS_MIN_N_RUNGS:
        r.cluster_size = 1  # bm without cs below the split point
    return r
