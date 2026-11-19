# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sparse MLA policy: choose a schedule, size buffers, then fill the SM grid.

``rows`` is planned B*Q, ``query_length`` is planned Q, and ``capacity`` counts
selected slots (each source rounded to 128, at least 256 in total), not raw KV
context. Live lengths and indices remain runtime inputs. A query/head grid has
rows*ceil(heads/tile_Q) work tiles; a 2CTA tile consumes two CTAs.

Selection tries qualified schedules before the small-grid defaults, then
sizes buffers and fills the remaining grid with KV/V partitions. Row-to-SM
ratios estimate available work, not achieved occupancy. Head/capacity/Q and
wave cutoffs are measured GB300 crossovers, not hardware or model limits;
resource and layout eligibility is checked separately by the kernel config.
"""

from dataclasses import dataclass, replace
from math import ceil


@dataclass(frozen=True)
class SparseMlaProfile:
    """Kernel schedule; zero tile/split/V-partition values are resolved below."""

    family: str
    tile_size_q: int = 0
    split_kv: int = 1
    scheduler: str = "nonpersistent"
    head_dim_ctas: int = 1
    gather_issue_warps: int = 4
    balanced_registers: bool = False
    reuse_kv: bool = False
    reuse_kv_stages: int = 0
    uniform_offset_cache: bool = False
    kv_pipeline_stages: int = 0


def base_fp8_profile(rows, heads, capacity, sm_count, rope_dim):
    """Small-grid tile and split selection for FP8.

    ``rope_dim`` is the QK-only RoPE width (64 for D576). Rules gated on
    ``rope_dim == 64`` are D576 crossovers measured on GB300 with top-k 2048;
    D512 plans use the remaining rules.
    """
    # M64 shares KV across more heads but can leave a small grid underfilled.
    # M16 exposes more tasks at the cost of repeating KV work for head tiles.
    throughput_h32 = heads == 32 and rows * 2 >= sm_count and (capacity >= 512)
    long_list = capacity >= 8192
    m16_work = rows * ((heads + 15) // 16)
    m16_one_step_work = m16_work * ((capacity + 255) // 256)
    use_keep = heads >= 64 and (
        capacity > 1024 and m16_one_step_work > sm_count or m16_work >= sm_count // 2
    )
    use_keep = use_keep or (long_list and heads >= 16 and (rows >= 8))
    use_keep = use_keep or (capacity <= 256 and heads >= 16)
    use_keep = use_keep or throughput_h32
    base_clusters = rows * ((heads + 127) // 128)
    keep_work = rows * ((heads + 63) // 64)
    kv_tiles = (capacity + 127) // 128
    split_limit = min(kv_tiles, max(1, sm_count // 4 // keep_work))
    kv_steps = (kv_tiles + split_limit - 1) // split_limit
    keep_splits = (kv_tiles + kv_steps - 1) // kv_steps
    prefer_wide_q = (
        heads >= 64
        and capacity >= 512
        and (kv_steps <= 2)
        and (keep_work * keep_splits * 4 >= sm_count // 2)
    )
    # Partition V when the wide tile alone cannot fill a resident wave.
    if prefer_wide_q:
        return SparseMlaProfile(family="keep", split_kv=keep_splits, head_dim_ctas=4)
    if heads >= 128 and capacity >= 512 and (base_clusters * 2 >= sm_count // 2):
        return SparseMlaProfile(family="2cta", gather_issue_warps=1)
    return SparseMlaProfile(
        family="keep" if use_keep else "swap",
        split_kv=1 if throughput_h32 or (use_keep and capacity <= 1024) else 0,
        head_dim_ctas=1 if throughput_h32 else 0,
        balanced_registers=not use_keep,
        reuse_kv=capacity >= 512 and use_keep,
        reuse_kv_stages=10 if capacity >= 512 and use_keep else 0,
    )


def choose_fp8_profile(rows, heads, query_length, capacity, sm_count, rope_dim):
    """Prefer resident split grids or sustained-throughput schedules when eligible.

    Falls back to :func:`base_fp8_profile`; ``rope_dim`` has the same meaning.
    """
    # These short-Q regions favor one KV stream with retained tiles and a
    # split grid near one device wave. The Q<=8 cutoff is empirical; the
    # kernel also supports prefill. Powers of two are policy, not a KV ABI.
    if (
        heads == 128
        and query_length <= 8
        and (capacity >= 512)
        and ((8 if capacity <= 6 * 128 else 16) <= rows <= sm_count // 4)
    ):
        # Two CTAs cover H128 without repeating KV loads across M64 tiles.
        # Up to six KV tiles, avoiding the separate split-KV merge wins from
        # eight query rows. Longer lists keep the measured sixteen-row
        # crossover; these are GB300 latency crossovers, not kernel limits.
        # Four splits help only the small, long-list grid measured here.
        small_long_grid = capacity >= 2048 and rows <= sm_count // 8
        return SparseMlaProfile(
            family="2cta", split_kv=4 if small_long_grid else 1, gather_issue_warps=1
        )
    if heads == 128 and rows * 2 >= sm_count:
        # Planned capacity is at least 256. Once the two-CTA grid fills a
        # device wave, share KV across H128 even for short selected lists.
        # BF16's CLC choice below is a separately qualified schedule.
        return SparseMlaProfile(family="2cta", scheduler="static", gather_issue_warps=1)
    return base_fp8_profile(rows, heads, capacity, sm_count, rope_dim)


def size_profile_buffers(profile, *, rows, heads, query_length, capacity, dtype, sm_count):
    """Set register-budget and KV/page stage counts for a chosen schedule."""
    # These are coupled measured budgets, not independent universal wins:
    # issuer/cache registers compete with softmax/correction, and extra KV
    # stages compete with Q, P, metadata and scheduler SMEM.
    if (
        dtype == "fp8"
        and heads == 128
        and (capacity >= 2048)
        and (query_length <= 8)
        and (rows >= 4 * sm_count)
    ):
        profile = replace(profile, balanced_registers=True)
    return profile


QUALIFIED_POLICY_HEADS = (8, 16, 32, 64, 128)


def sparse_policy_heads(heads, *, dtype, rope_dim):
    """Return the head count whose measured schedule serves ``heads``.

    The crossovers are measured for H8/16/32/64/128. For D576 FP8-MMA plans,
    other counts (e.g. 12/24/48/96) use the next qualified count's schedule:
    its Q tile already pads that count and the kernel masks the extra rows.
    Matching them on their raw count instead falls through to a default M16
    Swap-AB grid that repeats every KV tile once per 16-head tile (three times
    for H48, six for H96).
    BF16 and D512 plans keep their raw count until measured the same way.
    """
    return next((h for h in QUALIFIED_POLICY_HEADS if heads <= h), heads)


def select_sparse_mla_profile(
    *,
    rows,
    heads,
    query_length,
    capacity,
    dtype,
    sm_count,
    rope_dim=0,
):
    """Resolve one complete schedule from the validated public plan bounds.

    ``heads`` is first mapped by :func:`sparse_policy_heads`.
    """
    kwargs = {
        "rows": rows,
        "query_length": query_length,
        "capacity": capacity,
        "dtype": dtype,
        "sm_count": sm_count,
        "rope_dim": rope_dim,
    }
    policy_heads = sparse_policy_heads(heads, dtype=dtype, rope_dim=rope_dim)
    profile = resolve_sparse_mla_profile(heads=policy_heads, **kwargs)
    return profile


def resolve_sparse_mla_profile(
    *,
    rows,
    heads,
    query_length,
    capacity,
    dtype,
    sm_count,
    rope_dim=0,
):
    """Choose, size and grid-fill the schedule for exactly ``heads`` heads.

    :func:`select_sparse_mla_profile` is the planning entry point; it maps
    unmeasured head counts before calling this.
    """
    if dtype == "fp8":
        profile = choose_fp8_profile(rows, heads, query_length, capacity, sm_count, rope_dim)
    profile = size_profile_buffers(
        profile,
        rows=rows,
        heads=heads,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
    )
    if rope_dim and profile.reuse_kv:
        retained_tiles = 2 if dtype == "fp8" else 1
        profile = replace(
            profile,
            reuse_kv_stages=max(profile.reuse_kv_stages, retained_tiles * 5),
        )
    tile = profile.tile_size_q or {"swap": 16, "keep": 64, "2cta": 128}[profile.family]
    work = rows * ceil(heads / tile)
    splits = profile.split_kv
    if splits == 0:
        # Fill a resident wave using instruction-aligned KV partitions.
        # More splits shorten each CTA's work but increase partial-O/LSE merge
        # traffic. Two-CTA work targets half as many cluster tasks as SMs.
        step = 128 if tile >= 64 else 256
        steps = max(1, ceil(capacity / step))
        target = sm_count // 2 if profile.family == "2cta" else sm_count
        partitions = min(steps, max(1, target // work), 128)
        splits = ceil(steps / ceil(steps / partitions))
    v_ctas = profile.head_dim_ctas
    if v_ctas == 0:
        # V partitioning exposes more tasks but repeats QK/softmax across them.
        # Prefer it only while the query/KV grid still lacks enough work.
        remaining = work * splits
        v_ctas = (
            1
            if profile.family == "2cta" or remaining * 2 > sm_count
            else 4
            if remaining * 4 <= sm_count
            else 2
        )
    # Retaining K through PV requires a single owner of the V dimension.
    reuse = profile.reuse_kv and v_ctas == 1
    return replace(
        profile,
        tile_size_q=tile,
        split_kv=splits,
        head_dim_ctas=v_ctas,
        reuse_kv=reuse,
        reuse_kv_stages=profile.reuse_kv_stages if reuse else 0,
    )
