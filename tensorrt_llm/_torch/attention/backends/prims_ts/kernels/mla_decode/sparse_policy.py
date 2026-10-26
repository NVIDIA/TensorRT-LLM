# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
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
    reduction: str = "gmem_separate"
    head_dim_ctas: int = 1
    gather_issue_warps: int = 4
    balanced_registers: bool = False
    reuse_kv: bool = False
    reuse_kv_stages: int = 0
    paired_correction: bool = False
    single_kv_stream: bool = False
    defer_max_update: bool = False
    uniform_offset_cache: bool = False
    kv_pipeline_stages: int = 0
    kv_tile_size: int = 128
    page_pipeline_stages: int = 0


def _base_fp8(rows, heads, capacity, sm_count):
    """Small-grid tile and split selection for FP8."""
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
    cluster_splits = 1 << ((capacity + 255) // 256 - 1).bit_length()
    m8_work = rows * ((heads + 7) // 8)
    one_tile_splits = (capacity + 127) // 128
    if heads == 32 and capacity >= 512 and rows >= sm_count:
        # Enough query rows amortize padded M64 work. Retaining two FP8 K
        # tiles saves V reloads; uniform coordinates amortize repeated D slices.
        return SparseMlaProfile(
            family="keep",
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            scheduler="clc",
        )
    if (
        heads >= 64
        and 512 <= capacity <= 1024
        and (sm_count // 2 <= keep_work * one_tile_splits <= sm_count)
    ):
        # One K tile per split fills about half to one device wave. Further
        # splitting cannot shorten the tile and only increases merge traffic.
        return SparseMlaProfile(
            family="keep", split_kv=one_tile_splits, reuse_kv=True, reuse_kv_stages=10
        )
    if heads == 32 and 32 <= rows <= sm_count // 2 and (capacity >= 512):
        # This intermediate grid favors M64 reuse, with a second split only
        # for longer lists; additional splits lose to their merge overhead.
        return SparseMlaProfile(
            family="keep",
            split_kv=1 if capacity <= 1024 else 2,
            reuse_kv=True,
            reuse_kv_stages=10,
        )
    if heads <= 16 and rows * 4 >= sm_count and (capacity >= 512):
        # Narrow Q tiles avoid padded heads; twelve stages retain two FP8
        # tiles plus four prefetched D slices (four slices per D512 tile).
        return SparseMlaProfile(
            family="swap",
            tile_size_q=8 if heads <= 8 else 16,
            split_kv=2 if capacity > 1024 and rows * 2 <= sm_count else 1,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=12,
        )
    if capacity <= 256 and heads >= 16 and (m8_work * 4 <= sm_count // 2):
        # Short lists offer little KV parallelism. V partitions expose work
        # despite repeating QK/softmax, while M8 limits padded arithmetic.
        return SparseMlaProfile(
            family="swap", tile_size_q=8, head_dim_ctas=4, balanced_registers=True
        )
    # Partition V when the wide tile alone cannot fill a resident wave.
    if prefer_wide_q:
        return SparseMlaProfile(family="keep", split_kv=keep_splits, head_dim_ctas=4)
    elif (
        1 < cluster_splits <= 4
        and heads % 16 == 0
        and (m16_work * cluster_splits <= sm_count // 2)
    ):
        # A small two/four-split cluster avoids a standalone GMEM finisher.
        # Its coordination/residency cost loses for larger grids or split sets.
        return SparseMlaProfile(
            family="swap",
            split_kv=cluster_splits,
            reduction="cluster",
            head_dim_ctas=0,
            balanced_registers=True,
        )
    if (
        long_list
        and heads >= 128
        and (rows >= 8)
        and (base_clusters * 2 <= sm_count // 2)
    ):
        # Wide heads amortize the 2CTA layout; long lists provide enough KV
        # work to split when the query grid alone is small.
        return SparseMlaProfile(family="2cta", split_kv=0, gather_issue_warps=1)
    elif heads >= 128 and capacity >= 512 and (base_clusters * 2 >= sm_count // 2):
        return SparseMlaProfile(family="2cta", gather_issue_warps=1)
    return SparseMlaProfile(
        family="keep" if use_keep else "swap",
        split_kv=1 if throughput_h32 or (use_keep and capacity <= 1024) else 0,
        head_dim_ctas=1 if throughput_h32 else 0,
        balanced_registers=not use_keep,
        reuse_kv=capacity >= 512 and use_keep,
        reuse_kv_stages=10 if capacity >= 512 and use_keep else 0,
    )


def _base_bf16(rows, heads, capacity, sm_count):
    """Small-grid tile and split selection for BF16."""
    # BF16 BK128 cannot retain two complete D512 tiles in SMEM. A narrow
    # grid favors more small-Q tasks; a large grid amortizes M64's padding
    # and benefits from retaining one tile instead of reloading V.
    throughput_h32 = heads == 32 and rows * 2 >= sm_count and capacity >= 512
    throughput_bf16 = (
        heads >= 32 and (rows >= sm_count or throughput_h32) and (capacity >= 512)
    )
    short_list = capacity <= 256 and heads <= 64
    tile = (8 if heads <= 16 else 16) if short_list else 16 if heads <= 32 else 64
    if throughput_bf16:
        tile = 64
    profile = SparseMlaProfile(
        family="swap" if tile < 64 else "keep",
        tile_size_q=tile,
        split_kv=1 if short_list or throughput_bf16 else 0,
        head_dim_ctas=1 if throughput_bf16 else 0,
        balanced_registers=tile < 64,
        reuse_kv=throughput_bf16,
        reuse_kv_stages=4 if throughput_bf16 else 0,
        uniform_offset_cache=throughput_bf16
        and rows >= sm_count
        and (
            capacity >= 2048
            or (capacity >= 1024 and rows >= 2 * sm_count)
            or rows >= 8 * sm_count
        ),
    )
    cluster_splits = 1 << ((capacity + 255) // 256 - 1).bit_length()
    m16_work = rows * ((heads + 15) // 16)
    if (
        not short_list
        and tile == 16
        and (1 < cluster_splits <= 4)
        and (heads % 16 == 0)
        and (m16_work * cluster_splits <= sm_count // 2)
    ):
        # Same small-grid tradeoff as FP8: cluster fusion saves a launch, but
        # only the qualified two/four-split M16 layouts justify its overhead.
        profile = replace(profile, split_kv=cluster_splits, reduction="cluster")
    return profile


def _choose_fp8(rows, heads, query_length, capacity, sm_count):
    """Prefer resident split grids or sustained-throughput schedules when eligible."""
    # These short-Q regions favor one KV stream with retained tiles and a
    # split grid near one device wave. The Q<=8 cutoff is empirical; the
    # kernel also supports prefill. Powers of two are policy, not a KV ABI.
    if (
        query_length <= 8
        and capacity >= 512
        and (
            heads <= 16
            and 32 <= rows <= sm_count
            and (not (heads <= 8 and capacity >= 2048 and (rows * 2 >= sm_count)))
            or (heads == 32 and 16 <= rows <= sm_count // 8 and (capacity >= 2048))
        )
    ):
        base_ctas = rows * ((heads + 15) // 16)
        wave_splits = max(1, sm_count // base_ctas)
        wave_splits = 1 << (wave_splits.bit_length() - 1)
        return SparseMlaProfile(
            family="swap",
            split_kv=wave_splits,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=10,
            single_kv_stream=True,
        )
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
    if heads == 64 and capacity >= 512 and (rows * 2 >= sm_count):
        # Sustained M64 work amortizes eight issuers, expanded coordinate
        # registers and CLC setup. Paired correction reduces stats traffic.
        return SparseMlaProfile(
            family="keep",
            gather_issue_warps=8,
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            scheduler="clc",
            paired_correction=True,
        )
    if heads == 128 and rows * 2 >= sm_count:
        # Planned capacity is at least 256. Once the two-CTA grid fills a
        # device wave, share KV across H128 even for short selected lists.
        # BF16's CLC choice below is a separately qualified schedule.
        return SparseMlaProfile(family="2cta", scheduler="static", gather_issue_warps=1)
    if (
        heads <= 16
        and capacity >= 512
        and (
            capacity <= 1024
            and rows >= (sm_count if heads > 8 else 2 * sm_count)
            or rows >= 8 * sm_count
        )
    ):
        # At high row counts, saved KV traffic outweighs M64 head padding.
        # Smaller grids retain M8/M16 to preserve independent work.
        return SparseMlaProfile(
            family="keep",
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            scheduler="clc",
        )
    return _base_fp8(rows, heads, capacity, sm_count)


def _choose_bf16(rows, heads, query_length, capacity, sm_count):
    """Prefer resident split grids or sustained-throughput schedules when eligible."""
    # Small-head workloads can retain one BK128 tile with five/six 32-KiB
    # stages: four hold D512, the rest prefetch. PV must release that tile
    # before next QK; this favors a single stream in the measured regions.
    if (
        heads <= 16
        and capacity >= 512
        and (rows * 2 >= sm_count)
        and (
            query_length <= 8
            or (
                heads <= 8
                and capacity <= 1024
                and (query_length >= 128)
                and (rows >= 8 * sm_count)
            )
        )
    ):
        return SparseMlaProfile(
            family="swap",
            tile_size_q=8 if heads <= 8 else 16,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=6
            if heads > 8 or (query_length <= 8 and rows >= 4 * sm_count)
            else 5,
            single_kv_stream=True,
            defer_max_update=True,
            scheduler="clc",
        )
    if (
        heads in (8, 16, 32, 64)
        and capacity >= 512
        and (query_length >= 128)
        and (rows >= 8 * sm_count)
    ):
        # Long-Q grids have ample tasks. Static persistence amortizes CTA
        # setup here; wider heads favor M64 reuse, narrow heads avoid padding.
        keep = heads >= 32
        return SparseMlaProfile(
            family="keep" if keep else "swap",
            tile_size_q=64 if keep else 8 if heads <= 8 else 16,
            balanced_registers=not keep,
            reuse_kv=True,
            reuse_kv_stages=5 if keep else 6,
            uniform_offset_cache=keep,
            scheduler="static",
        )
    if (
        heads in (32, 64)
        and rows * 2 >= sm_count
        and (query_length <= 8)
        and (capacity >= 512)
    ):
        # Decode with enough rows favors M64 reuse. Buffer sizing below
        # supplies its qualified gather/cache budget and scheduler crossover.
        return SparseMlaProfile(
            family="keep",
            reuse_kv=True,
            reuse_kv_stages=5,
            uniform_offset_cache=capacity >= 2048
            and rows >= sm_count
            or (capacity >= 1024 and rows >= 4 * sm_count)
            or rows >= 8 * sm_count,
        )
    if heads <= 16 and rows * 2 >= sm_count and (capacity >= 512):
        return SparseMlaProfile(
            family="swap",
            tile_size_q=8 if heads <= 8 else 16,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=6,
        )
    if heads == 128 and 2 <= sm_count // (2 * rows) <= 4 and (capacity >= 512):
        # Two M64 head tiles with BK64 reuse and a few KV splits fill this
        # small H128 grid. Larger grids favor the wide 2CTA path below.
        wave_splits = max(1, sm_count // 2 // rows)
        return SparseMlaProfile(
            family="keep",
            split_kv=wave_splits,
            gather_issue_warps=8,
            uniform_offset_cache=True,
            reuse_kv=True,
            reuse_kv_stages=10,
            kv_tile_size=64,
            page_pipeline_stages=2,
            defer_max_update=True,
        )
    if (
        capacity >= 512
        and rows * 4 >= sm_count
        and (heads == 128 or (heads in (32, 64) and rows * 2 <= sm_count))
    ):
        # BF16 2CTA amortizes wide-head work once enough clusters are present.
        # More K/V stages overlap gathers; coordinate caching costs registers,
        # so it is enabled only for the qualified H128 configuration.
        return SparseMlaProfile(
            family="2cta",
            kv_pipeline_stages=8,
            uniform_offset_cache=heads == 128,
            scheduler="clc" if heads == 128 and rows >= sm_count else "nonpersistent",
        )
    return _base_bf16(rows, heads, capacity, sm_count)


def _size_buffers(profile, *, rows, heads, query_length, capacity, dtype, sm_count):
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
    if (
        dtype == "bf16"
        and profile.family == "keep"
        and (profile.reuse_kv_stages == 5)
        and (heads in (32, 64))
    ):
        # Both prefill and decode wide-BF16 reuse schedules share this budget.
        profile = replace(
            profile,
            gather_issue_warps=8,
            uniform_offset_cache=True,
            defer_max_update=True,
        )
        if query_length <= 8 and rows >= 2 * sm_count:
            profile = replace(profile, scheduler="clc")
        if (
            heads == 32
            and query_length <= 8
            and (rows >= 4 * sm_count)
            and (capacity >= 1024)
        ):
            # BK64 halves each BF16 stage to 16 KiB: ten stages retain two
            # four-slice K tiles plus two slices ahead. This restores QK/PV
            # overlap, at the cost of twice as many token-tile iterations.
            profile = replace(
                profile, kv_tile_size=64, reuse_kv_stages=10, page_pipeline_stages=2
            )
    if (
        dtype == "bf16"
        and profile.family == "keep"
        and (heads == 64)
        and (rows >= sm_count)
        and (capacity >= 512)
        and (profile.split_kv == 1)
    ):
        # The qualified H64 BK64 schedule serves both prefill and decode.
        # Two metadata stages let preparation advance while issuers reuse
        # current coordinates. The bounded maximum update is BF16-only.
        profile = replace(
            profile,
            kv_tile_size=64,
            reuse_kv=True,
            reuse_kv_stages=10,
            page_pipeline_stages=2,
            gather_issue_warps=8,
            uniform_offset_cache=True,
            scheduler="clc",
            defer_max_update=True,
        )
    return profile


def select_sparse_mla_profile(*, rows, heads, query_length, capacity, dtype, sm_count):
    """Resolve one complete schedule from the validated public plan bounds."""
    choose = _choose_fp8 if dtype == "fp8" else _choose_bf16
    profile = choose(rows, heads, query_length, capacity, sm_count)
    profile = _size_buffers(
        profile,
        rows=rows,
        heads=heads,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
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
