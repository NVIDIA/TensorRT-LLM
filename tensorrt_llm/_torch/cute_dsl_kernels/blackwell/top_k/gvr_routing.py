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

All thresholds are B200 deployment defaults, kept in one place so
retuning is a constant edit, not a logic edit.
"""

from dataclasses import dataclass
from typing import Optional

# ---- thresholds (B200 deployment defaults) --------------------------------
#
# UNITS: every threshold is fitted on KERNEL-ONLY time for both the
# indexer and the top-k; do not mix in wall-clock numbers when retuning.

# Below this the stock kernel already runs at the fixed-cost floor.
ASSIST_MIN_N_COMP = 2048

# Block-skip prefix break-evens. Retune these from real captures only:
# the real block-max distribution decides the pass rate.
SKIP_MIN_N_COUNTS = 65536  # va: attach block_max from here up
SKIP_MIN_N_RUNGS_FLASH = 131072  # vb (flash): bm pays from here
SKIP_CS_MIN_N_RUNGS = 196608  # vb: cluster split from here up

# Emission-cost model: counts emission is a batch-only latency chain
# (hides at large batch); list emission grows with batch and context.
LIST_EMIT_MIN_N = 65536  # shorter rows: the emission outweighs the saving
LIST_EMIT_MAX_B = 4  # past four rows the list stops repaying its emission
COUNTS_MIN_TOKENS = 524288  # B * raw length; below this the rungs tier wins
RUNGS_ONLY_MIN_N = 16384  # short-row band where rungs also beats counts
RUNGS_ONLY_MAX_N = 49152

# Mid-row weak band: the stock kernel's split grid fits one wave here
# and out-scans every assist tier, so the epilogue emits nothing.
ASSIST_WEAK_MIN_N = 49152
# The band interior is unmeasured; it stays on the stock kernel.
ASSIST_WEAK_MAX_N = 65536
ASSIST_WEAK_MAX_B = 8

# rungs block_max pays only at small K; at large K the prefix is overhead.
RUNGS_BM_MAX_K = 512

# 512-thread build wins for list-hit rows at small K (work is O(list)).
SMALL_K_LIST_THREADS = 512
SMALL_K_MAX = 512

# GPC packing: cs=8 only while all row-clusters fit half the device;
# cs4/2 keep a 10% headroom.
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
        # short rows: the stock kernel is already under the fixed cost
        return "none"
    if have_epilogue and n_comp >= LIST_EMIT_MIN_N and batch <= LIST_EMIT_MAX_B:
        # must stay ahead of the weak-band gate: a list hit never scans the row
        return "list"
    if batch <= ASSIST_WEAK_MAX_B and ASSIST_WEAK_MIN_N <= n_comp < ASSIST_WEAK_MAX_N:
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
        # block_max: miss rows take a skip-walk instead of a dense re-scan
        r.attach_block_max = n_comp >= SKIP_MIN_N_COUNTS
        return r
    if tier == "counts":
        r.attach_block_max = n_comp >= SKIP_MIN_N_COUNTS
        return r
    # rungs (vb); the skip prefix and cluster split are mutually exclusive
    if n_comp >= SKIP_CS_MIN_N_RUNGS:
        if batch * 8 <= num_sms // CS8_HALF_DEVICE:
            r.cluster_size = 8
        elif batch * 4 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 4
        elif batch * 2 <= (num_sms * CS_HEADROOM_NUM) // CS_HEADROOM_DEN:
            r.cluster_size = 2
    if r.cluster_size == 1 and k <= RUNGS_BM_MAX_K and n_comp >= SKIP_MIN_N_RUNGS_FLASH:
        r.attach_block_max = True
    return r
