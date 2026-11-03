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
    whole_kv_tile: bool = False


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
    cluster_splits = 1 << ((capacity + 255) // 256 - 1).bit_length()
    m8_work = rows * ((heads + 7) // 8)
    one_tile_splits = (capacity + 127) // 128
    if heads == 32 and capacity >= 512 and rows >= sm_count:
        # Enough query rows amortize padded M64 work. Retaining two FP8 K
        # tiles saves V reloads; uniform coordinates amortize repeated D slices.
        # D576 prefill-sized grids over long lists use the H64 issuer budget
        # (GB300 top-k 2048, H24 prefill 8k: 1669 -> 1511 us).
        prefill = rope_dim == 64 and capacity >= 2048 and rows >= 8 * sm_count
        return SparseMlaProfile(
            family="keep",
            gather_issue_warps=8 if prefill else 4,
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            paired_correction=prefill,
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
        if rope_dim != 64 or capacity <= 1024:
            return SparseMlaProfile(
                family="keep",
                split_kv=1 if capacity <= 1024 else 2,
                reuse_kv=True,
                reuse_kv_stages=10,
            )
        # D576 long lists split KV up to one resident wave: four splits at
        # 32-37 rows (GB300 top-k 2048, 29.6 -> 21.8 us), otherwise two. They
        # use eight issuers; four splits finish their short K loops faster by
        # streaming V than by retaining K (22.9 -> 20.7 us), two-split grids
        # retain K with uniform coordinates (37.8 -> 36.3 us).
        wave_splits = 1 << ((sm_count // rows).bit_length() - 1)
        splits = min(4, max(2, wave_splits))
        retain = splits < 4
        return SparseMlaProfile(
            family="keep",
            split_kv=splits,
            gather_issue_warps=8,
            reuse_kv=retain,
            reuse_kv_stages=10 if retain else 0,
            uniform_offset_cache=retain,
        )
    if rope_dim == 64 and heads == 32 and 8 <= rows < 16 and capacity >= 2048:
        # Few D576 rows: M16 Swap-AB with resident-wave KV splits; retaining K
        # through PV beats reloading V (GB300 top-k 2048, B1 Sq8: 13.1 ->
        # 12.2 us).
        return SparseMlaProfile(
            family="swap", split_kv=0, reuse_kv=True, reuse_kv_stages=10
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


def base_bf16_profile(rows, heads, capacity, sm_count):
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


def choose_fp8_profile(rows, heads, query_length, capacity, sm_count, rope_dim):
    """Prefer resident split grids or sustained-throughput schedules when eligible.

    Falls back to :func:`base_fp8_profile`; ``rope_dim`` has the same meaning.
    """
    if (
        rope_dim == 64
        and heads == 32
        and query_length <= 8
        and capacity >= 2048
        and 16 <= rows <= sm_count // 8
    ):
        # D576: one M64 tile per row with a resident wave of KV splits beats
        # the one-stream M16 split below (GB300 top-k 2048, B4 Sq4: 17.2 ->
        # 15.6 us).
        wave_splits = 1 << ((sm_count // rows).bit_length() - 1)
        return SparseMlaProfile(
            family="keep",
            split_kv=min(wave_splits, capacity // 256),
            reuse_kv=True,
            reuse_kv_stages=10,
            paired_correction=True,
            page_pipeline_stages=2,
        )
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
    if rope_dim == 64 and 8 < heads <= 16 and capacity >= 2048 and rows >= 8 * sm_count:
        # Long D576 lists keep M16 Swap-AB on prefill-sized grids: eight
        # issuers, twelve retained stages and CLC persistence gather fast
        # enough that M64 head padding no longer pays (GB300 top-k 2048, H12
        # prefill 8k: 1637 -> 1374 us).
        return SparseMlaProfile(
            family="swap",
            split_kv=1,
            scheduler="clc",
            gather_issue_warps=8,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=12,
        )
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
    return base_fp8_profile(rows, heads, capacity, sm_count, rope_dim)


def choose_bf16_profile(rows, heads, query_length, capacity, sm_count):
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
    return base_bf16_profile(rows, heads, capacity, sm_count)


def size_profile_buffers(
    profile, *, rows, heads, query_length, capacity, dtype, sm_count
):
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
    if dtype != "fp8" or rope_dim != 64:
        return heads
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
    kv_cache_format="native",
):
    """Resolve one complete schedule from the validated public plan bounds.

    ``heads`` is first mapped by :func:`sparse_policy_heads`. A mapped
    schedule with cluster reduction is replaced by the raw-count schedule when
    ``heads`` is not a multiple of 16: the cluster finisher needs whole M16
    head tiles, a kernel constraint rather than a measured crossover.
    """
    kwargs = dict(
        rows=rows,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
        rope_dim=rope_dim,
        kv_cache_format=kv_cache_format,
    )
    policy_heads = sparse_policy_heads(heads, dtype=dtype, rope_dim=rope_dim)
    profile = resolve_sparse_mla_profile(heads=policy_heads, **kwargs)
    if policy_heads != heads and profile.reduction == "cluster" and heads % 16:
        profile = resolve_sparse_mla_profile(heads=heads, **kwargs)
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
    kv_cache_format="native",
):
    """Choose, size and grid-fill the schedule for exactly ``heads`` heads.

    :func:`select_sparse_mla_profile` is the planning entry point; it maps
    unmeasured head counts before calling this.
    """
    if (
        kv_cache_format in ("nvfp4", "nvfp4.fp8rope", "fp8")
        and dtype == "fp8"
        and rope_dim in (0, 64)
        and heads <= 32
        and capacity >= 512
        and (rows * 2 >= sm_count or heads >= 16)
    ):
        # Retain complete converted tiles even when the query grid is small.
        # Fill one resident wave with power-of-two KV partitions, preserving
        # at least two tiles per split for the eight-issuer steady-state loop.
        # Persistent 1CTA queues do not enumerate KV splits, so small grids
        # use the same reuse resources with a nonpersistent split schedule.
        persistent = rows * 2 >= sm_count
        split_limit = min(capacity // 128 // 2, max(1, sm_count // rows))
        splits = 1 if persistent else 1 << (split_limit.bit_length() - 1)
        # Above sixteen heads, WS M64 wins on GB300: its N256 PV consumes
        # Dv512 in two passes, whereas the narrow Swap-AB path needs four.
        keep = heads > 16
        return SparseMlaProfile(
            family="keep" if keep else "swap",
            tile_size_q=64 if keep else 8 if heads <= 8 else 16,
            split_kv=splits,
            gather_issue_warps=8,
            reuse_kv=True,
            whole_kv_tile=True,
            single_kv_stream=not keep,
            uniform_offset_cache=keep,
            paired_correction=keep,
            scheduler="clc" if persistent else "nonpersistent",
        )
    if dtype == "fp8":
        profile = choose_fp8_profile(
            rows, heads, query_length, capacity, sm_count, rope_dim
        )
    else:
        profile = choose_bf16_profile(rows, heads, query_length, capacity, sm_count)
    # D576 NVFP4 keeps this schedule down to one row: the general small-grid
    # choice partitions V over four CTAs that each repeat the K conversion
    # (GB300 H64 top-k 2048, B1 Sq4: 32.3 -> 13.1 us; Sq8: 18.0 -> 14.2 us).
    if (
        kv_cache_format != "native"
        and dtype == "fp8"
        and query_length <= 8
        and capacity >= 2048
        and rows >= (1 if rope_dim == 64 else 16)
        and rows * 2 < sm_count
    ):
        if (
            heads == 64
            or (heads == 128 and kv_cache_format == "fp8" and rows * 4 < sm_count)
            # D576 NVFP4 H128 below 16 rows: two whole-tile M64 heads beat the
            # 2CTA split (GB300 top-k 2048, H96 B1 Sq8: 27.3 -> 16.8 us).
            or (heads == 128 and rope_dim == 64 and rows < 16)
        ):
            # Small packed grids still benefit from whole-record gathers and
            # retained WS tiles. At low row counts, two M64 head tiles also
            # beat group-2 mixed-pool startup despite repeating KV conversion.
            # Power-of-two splits fill most of a resident wave without adding
            # a short split for a source-boundary tile.
            work = rows * ceil(heads / 64)
            split_limit = min(capacity // 128, max(1, sm_count // work))
            wave_splits = 1 << (split_limit.bit_length() - 1)
            profile = SparseMlaProfile(
                family="keep",
                split_kv=wave_splits,
                gather_issue_warps=8,
                reuse_kv=True,
                reuse_kv_stages=10,
                whole_kv_tile=True,
                uniform_offset_cache=True,
                paired_correction=True,
            )
        elif profile.family == "2cta":
            # Conversion makes longer per-cluster loops costly. Use the same
            # resident-wave split sizing below, with two SMs per cluster.
            profile = replace(profile, split_kv=0)
    profile = size_profile_buffers(
        profile,
        rows=rows,
        heads=heads,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
    )
    if (
        rope_dim
        and dtype == "bf16"
        and profile.family == "2cta"
        and profile.kv_pipeline_stages
    ):
        # Each CTA's Q RoPE tail needs 64*rope_dim BF16 elements. A combined
        # K/V stage holds 64*128, so trade whole prefetch stages for that storage.
        profile = replace(
            profile,
            kv_pipeline_stages=max(
                4, profile.kv_pipeline_stages - ceil(rope_dim / 128)
            ),
        )
    if rope_dim and profile.reuse_kv:
        if dtype == "bf16" and profile.family == "keep":
            # A D576 M64 query plus a retained BK128 BF16 tile exceeds SMEM.
            # BK64 retains one five-slice K tile plus one prefetched slice;
            # its pipeline serializes PV before the next complete QK tile.
            profile = replace(profile, kv_tile_size=64, reuse_kv_stages=6)
        else:
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
    if profile.family == "2cta" and kv_cache_format != "native":
        # Convert each CTA's64 K rows once, then exchange the peer's V
        # features through DSMEM. Buffering depends on the QK width.
        profile = replace(profile, balanced_registers=True)
    return replace(
        profile,
        tile_size_q=tile,
        split_kv=splits,
        head_dim_ctas=v_ctas,
        reuse_kv=reuse,
        reuse_kv_stages=profile.reuse_kv_stages if reuse else 0,
    )


# D576 (DSA) sparse MLA: one CTA per (query token, 64 heads) on the 1CTA
# keeps-MMA-AB q64 schedule with four Gather4 issue warps and the uniform quad
# cache (fixed by the D576 1CTA config). Only the split varies with the plan:
# below D576_SPLIT_KV_MIN_CTAS CTAs most SMs would idle, so every token's top-k
# list is split into S contiguous tile ranges (one CTA each) and the 1CTA GMEM
# reducer merges the partials, following trtllm-gen's MLA generation rule.
D576_SPLIT_KV_MIN_CTAS = 96
D576_SPLIT_KV_CHOICES = (16, 8, 4, 2)


def d576_v_split(total_ctas: int, sm_count: int) -> int:
    """V head-dim split of a split launch of ``total_ctas`` CTAs: 4 / 2 while the
    grid fills at most a quarter / half of the SMs, else 1."""
    if total_ctas * 4 <= sm_count:
        return 4
    if total_ctas * 2 <= sm_count:
        return 2
    return 1


def select_sparse_mla_d576_split(
    num_ctas: int, sparse_mla_top_k: int, sm_count: int
) -> tuple[int, int]:
    """Return ``(split_kv, head_dim_split)`` for an E4M3 launch of ``num_ctas`` CTAs.

    trtllm-gen's MLA generation rule: split the top-k list so every CTA keeps
    at least two 128-key tiles within one wave (a top-k of one tile pair:
    one tile, if the V split then applies), then, while fewer than half the
    SMs have work, split the 512 V columns over 4 (or 2) CTAs.  Each V
    slice repeats the QK / softmax of its tiles, but a one- or two-tile CTA is
    bound by its fixed latency (Q load, first gather, the single QK-softmax-PV
    chain, the partial-O store), which the slice cuts.  B200, K=2048: B1 is
    8 x 4 = 32 CTAs instead of 16 x 1.
    """
    if num_ctas >= D576_SPLIT_KV_MIN_CTAS:
        return 1, 1
    pairs = -(-sparse_mla_top_k // 256)
    limit = min(pairs, sm_count // num_ctas)
    split = next((s for s in D576_SPLIT_KV_CHOICES if s <= limit), 1)
    if split == 1 and sparse_mla_top_k > 128 and num_ctas * 4 <= sm_count:
        # A single tile pair (top-k <= 256): one tile per CTA, so the V split
        # below can apply.  B200, top-k 256: B1 9.77 -> 7.52 us (trtllm-gen
        # 8.13), B32 9.78 -> 9.35; without the V split (B64) one-tile CTAs
        # are slower (10.6 -> 12.0), hence the quarter-wave bound.
        split = 2
    if split == 1:
        return 1, 1
    return split, d576_v_split(num_ctas * split, sm_count)


def prefer_sparse_d576_reload(
    *, rows, heads, query_length, capacity, sm_count, general_profile
):
    """Choose native FP8 reload versus the general retained-KV/2CTA pipeline.

    ``capacity`` is the primary top-k slot count and ``rows`` is B*Q. H128
    benefits from sharing KV in 2CTA beyond the measured four-or-more native
    KV-partition region. For H64, small grids and short lists favor early K
    release. Longer queries increase within-request reuse; measured GB300
    crossovers keep q<=16 on reload, q<=32 through four SM-sized row groups,
    and larger query grids on retained KV. These are schedule crossovers, not
    supported-shape limits. ``general_profile`` is the existing alternative:
    its narrow-Q/V partitions and cluster finish beat a separate native merge
    on short lists. The native reload launch alone uses grid-z per row.
    """
    if rows > 65535:
        return False
    if capacity <= 512 and general_profile.family == "swap":
        return False
    if heads == 128:
        return select_sparse_mla_d576_split(rows * 2, capacity, sm_count)[0] >= 4
    if heads != 64:
        return False
    return (
        capacity <= 256
        or rows <= sm_count
        or query_length <= 16
        or (query_length <= 32 and rows <= 4 * sm_count)
    )
