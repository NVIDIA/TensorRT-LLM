# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_NORMQ ring boundary geometry and engage rules (ctx_boundary_plan.py), CPU only, no torch.

The plan mirrors ctxBoundaryNormQuantSm107.cuh's launch math: every row of the padded scale tiles is visited by exactly
one quad slot, the producer loads the consumer's rows in the consumer's order, the quad's 16-byte scale-group stores put
every byte where get_sf_out_offset_128x4 (the served layout) puts it, and two CTAs' shared memory fit an SM. The engage
rules of DEP2CTX_NORMQ_STAGE / _DROP_NORM decline for the right first reason. The launch choice and ring depth, and
CBNQ_P2's chunk map, reduction tree (bit for bit the one-chunk kernel's, exact FP32 emulation), scale staging and
launch form follow the C++ launcher and its defaults (P2 at 3 stages, the one-chunk ring at 3 stages with no
carveout). The module is loaded from its file so the test also runs where torch is not installed.
"""

import importlib.util
import itertools
import pathlib
import random
import struct

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[7]
_PLAN = (
    _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/"
    "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/ctx_boundary_plan.py"
)
_spec = importlib.util.spec_from_file_location("ctx_boundary_plan", _PLAN)
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

pytestmark = pytest.mark.cpu_only

H = 6144
SMEM_PER_SM = 228 * 1024  # sm_100 / sm_107
SMEM_RESERVED_PER_CTA = 1024
STATIC_SMEM = 32 * 4  # the block reduction's float[32]


def test_supported_hidden_sizes():
    assert plan.supported(6144) and plan.supported(4096) and plan.supported(256)
    assert not plan.supported(8192)  # 1024 threads: above the two-CTA launch bound
    assert not plan.supported(6016) and not plan.supported(128)


def test_shared_memory_fits_two_ctas_per_sm():
    add = plan.smem_bytes(plan.STAGES, True, H)
    assert add == 3 * 2 * H * 2 + 2 * (H // 128) * 16 + H // 16 + 3 * 8 == 75672
    assert plan.MIN_BLOCKS * (add + SMEM_RESERVED_PER_CTA + STATIC_SMEM) <= SMEM_PER_SM
    assert (
        plan.smem_bytes(plan.STAGES, False, H) == 3 * H * 2 + 2 * (H // 128) * 16 + H // 16 + 3 * 8
    )
    # every piece starts 16-byte aligned (ring rows, two quad buffers, the NVFP4 scale row, the barriers)
    for hidden in (256, 1024, 4096, 6144):
        assert (plan.STAGES * 2 * hidden * 2) % 16 == 0 and (2 * (hidden // 128) * 16) % 16 == 0


@pytest.mark.parametrize("rows", [1, 31, 127, 128, 129, 1000, 1024, 4097, 8192, 16384])
def test_quads_cover_the_padded_rows_once(rows):
    padded = (rows + 127) // 128 * 128
    seen = [plan.quad_row(q, j) for q in range(plan.num_quads(rows)) for j in range(plan.QUAD_ROWS)]
    assert sorted(seen) == list(range(padded))


@pytest.mark.parametrize("rows", [1, 129, 1000, 8192, 16384])
@pytest.mark.parametrize("grid_size", [1, 7, 148, 296, 320])
def test_the_producer_loads_the_consumer_rows_in_order(rows, grid_size):
    g = min(grid_size, plan.num_quads(rows))
    every = []
    for cta in range(g):
        consumed = [r for _, _, r in plan.cta_rows(cta, g, rows) if r < rows]
        assert plan.producer_rows(cta, g, rows) == consumed
        every += consumed
    assert sorted(every) == list(range(rows))


def test_ring_stage_and_parity():
    slots = plan.ring_slots(8, 3)
    assert slots == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 0), (1, 0)]


@pytest.mark.parametrize("hidden", [256, 1024, 6144])
@pytest.mark.parametrize("rows", [1, 100, 128, 300])
def test_quad_scale_stores_match_the_served_layout(hidden, rows):
    ks = range(hidden // 32)
    for q in range(plan.num_quads(rows)):
        for j in range(plan.QUAD_ROWS):
            row = plan.quad_row(q, j)
            for k in ks:
                assert plan.quad_sf_offset(q, j, k, hidden) == plan.served_sf_offset(
                    row, k, rows, hidden
                )


def test_every_scale_group_is_stored_once():
    rows, hidden = 300, 1024
    groups = {}
    for q in range(plan.num_quads(rows)):
        for kt in range(hidden // 128):
            base = plan.quad_sf_offset(q, 0, kt * 4, hidden)
            assert base % 16 == 0
            groups[base] = groups.get(base, 0) + 1
    total = (rows + 127) // 128 * 128 * (hidden // 32)
    assert sorted(groups) == list(range(0, total, 16)) and set(groups.values()) == {1}


def test_grid():
    assert plan.grid(8192, 2, 148) == 296
    assert plan.grid(1, 2, 148) == 32  # one tile: 32 quads
    assert plan.grid(0, 2, 148) == 1


def test_the_price_formula():
    # ms/1K = delta_us x launches / rows; 57 pre-MoE launches per chunk
    assert plan.ms_per_1k(10.0, 57, 8192) == pytest.approx(57 * 10.0 / 8192)


_OK = dict(
    enabled=True,
    ops_built=True,
    sm_major=10,
    hidden=H,
    seat_api=True,
    router_f32=True,
    lora=False,
    rows=8192,
    all_rank_num_tokens=[8192, 7000],
    ep_rank=0,
    moe_max_num_tokens=16384,
    may_fuse=True,
    appended_columns=0,
)


def test_stage_engages_on_the_served_shape():
    assert plan.stage_decline_reason(**_OK) is None
    assert plan.stage_decline_reason(**dict(_OK, all_rank_num_tokens=None)) is None
    assert plan.chunk_max_tokens([8192, 7000]) == 8192 and plan.chunk_max_tokens(None) is None


@pytest.mark.parametrize(
    "change, reason",
    [
        (dict(enabled=False), "switched off"),
        (dict(ops_built=False), "not built"),
        (dict(sm_major=12), "sm_12x"),
        (dict(hidden=8192), "hidden 8192"),
        (dict(seat_api=False), "MegaMoE-local seat"),
        (dict(router_f32=False), "FP32"),
        (dict(lora=True), "LoRA"),
        (dict(all_rank_num_tokens=[8000, 8192]), "live rows 8000"),
        (dict(ep_rank=3), "EP rank 3"),
        (dict(all_rank_num_tokens=[8192, 20000]), "chunked"),
        (dict(all_rank_num_tokens=None, rows=20000), "chunked"),
        (dict(may_fuse=False), "scheduler"),
        (dict(appended_columns=1), "appended"),
    ],
)
def test_stage_declines_name_the_first_failing_check(change, reason):
    got = plan.stage_decline_reason(**dict(_OK, **change))
    assert got is not None and reason in got


def test_the_first_failing_check_wins():
    got = plan.stage_decline_reason(**dict(_OK, enabled=False, ops_built=False, lora=True))
    assert got == "switched off"


@pytest.mark.parametrize(
    "enabled, shared, lora, dropped",
    [
        (True, True, False, True),
        (True, None, False, True),  # no shared expert: router + stage are the only readers
        (False, True, False, False),
        (True, False, False, False),
        (True, True, True, False),
    ],
)
def test_drop_norm_rule(enabled, shared, lora, dropped):
    got = plan.drop_norm_decline_reason(enabled=enabled, shared_takes_mxfp8=shared, lora=lora)
    assert (got is None) == dropped


def test_drop_norm_rule_router_bf16():
    """DEP2CTX_ROUTER_BF16S: a router that reads the BF16 norm keeps it written, named before any other reason."""
    kw = dict(enabled=True, shared_takes_mxfp8=True, lora=False)
    assert plan.drop_norm_decline_reason(**kw) is None
    assert plan.drop_norm_decline_reason(**kw, router_reads_bf16=False) is None
    for over in ({}, dict(enabled=False), dict(lora=True)):
        got = plan.drop_norm_decline_reason(**dict(kw, **over), router_reads_bf16=True)
        assert got == "the router reads the BF16 norm (DEP2CTX_ROUTER_BF16S)"


def test_quad_rows_partition_one_tile():
    tile = list(
        itertools.chain.from_iterable([plan.quad_row(q, j) for j in range(4)] for q in range(32))
    )
    assert sorted(tile) == list(range(128))
    assert [plan.quad_row(5, j) for j in range(4)] == [5, 37, 69, 101]


# ---- the ring depth at one CTA per SM ---------------------------------------------------------------------------
OPTIN_SM10X = 232448  # cudaDevAttrMaxSharedMemoryPerBlockOptin, sm_100 / sm_107
SMEM_PER_SM_SM10X = 233472


def test_the_default_depths():
    """The one-chunk ring at 3 stages (no carveout) and P2 at 3 stages, mirroring kCbnqL6DefaultStages /
    kCbnqP2DefaultStages."""
    assert plan.L6_STAGES == plan.STAGES == 3
    assert plan.P2_STAGES == 3
    assert plan.launch_choice() == (3, False)
    assert plan.p2_choice() == (True, 3)
    assert plan.topk_dual_choice() is True


_CU = _PLAN.parent / "csrc" / "ctxBoundaryNormQuantSm107.cu"


def test_cpp_defaults_match_the_plan():
    """The launcher's default depths and carveout rule are the plan's (the GPU UT checks the launch form; this pins
    the source where no GPU runs)."""
    src = _CU.read_text()
    assert f"constexpr int kCbnqL6DefaultStages = {plan.L6_STAGES};" in src
    assert f"constexpr int kCbnqP2DefaultStages = {plan.P2_STAGES};" in src
    assert (
        "return CbnqLaunchChoice{kCbnqL6DefaultStages, false, true, kCbnqP2DefaultStages};" in src
    )


SM107 = (1024, SMEM_PER_SM_SM10X)  # threads, shared memory per SM: one 768-thread CTA
SM100 = (2048, SMEM_PER_SM_SM10X)  # sm_100 / sm_103: two 768-thread CTAs


def test_l6_launch_stages_step_down_to_fit_the_optin():
    assert plan.launch_stages(6, True, H, OPTIN_SM10X, *SM107) == 6
    assert plan.launch_stages(8, True, H, OPTIN_SM10X, *SM107) == 8
    assert plan.smem_bytes(8, True, H) + plan.STATIC_SMEM_BOUND <= OPTIN_SM10X
    assert plan.launch_stages(4, True, H, OPTIN_SM10X, *SM107) == 4
    assert plan.launch_stages(3, True, H, OPTIN_SM10X, *SM107) == 3
    # a smaller opt-in steps 8 -> 6 -> 4 -> 3
    six = plan.smem_bytes(6, True, H) + plan.STATIC_SMEM_BOUND
    assert plan.launch_stages(8, True, H, six, *SM107) == 6
    assert plan.launch_stages(8, True, H, six - 1, *SM107) == 4
    assert plan.launch_stages(8, True, H, 101376, *SM107) == 3  # sm_8x-class 99 KB
    # norm only (one row per stage): 8 stages fit easily
    assert plan.launch_stages(8, False, H, OPTIN_SM10X, *SM107) == 8


def _resident(stages, add, hidden, threads, smem_per_sm):
    """What cudaOccupancyMaxActiveBlocksPerMultiprocessor answers for the one-chunk kernel, bounded by the launch
    bound: the thread limit and the shared memory (dynamic + static bound + the per-CTA reservation)."""
    per_cta = (
        plan.smem_bytes(stages, add, hidden) + plan.STATIC_SMEM_BOUND + plan.RESERVED_SMEM_PER_CTA
    )
    return min(plan.MIN_BLOCKS, threads // (hidden // plan.PACKED), smem_per_sm // per_cta)


@pytest.mark.parametrize("device", [SM107, SM100], ids=["sm_107", "sm_100"])
@pytest.mark.parametrize("hidden", [1024, 4096, 6144])
@pytest.mark.parametrize("add", [True, False])
@pytest.mark.parametrize("requested", [3, 4, 6, 8])
def test_l6_depth_never_costs_a_resident_cta(device, hidden, add, requested):
    """The ring depth the launcher picks keeps the CTAs per SM the thread limit admits (ctas_per_sm, which the launch
    shape reports): on sm_100 / sm_103 two 6-stage CTAs at hidden 6144 do not fit 228 KB, so the depth stays at 4
    there."""
    s = plan.launch_stages(requested, add, hidden, OPTIN_SM10X, *device)
    assert s <= max(requested, plan.STAGES)
    assert _resident(s, add, hidden, *device) == plan.ctas_per_sm(device[0], hidden)
    # and it is the deepest such depth
    for deeper in (d for d in (4, 6, 8) if s < d <= requested):
        fits = plan.smem_bytes(deeper, add, hidden) + plan.STATIC_SMEM_BOUND <= OPTIN_SM10X
        assert not fits or _resident(deeper, add, hidden, *device) < plan.ctas_per_sm(
            device[0], hidden
        )


def test_l6_depth_on_sm100_at_hidden_6144():
    assert plan.launch_stages(6, True, H, OPTIN_SM10X, *SM100) == 4
    assert _resident(6, True, H, *SM100) == 1  # the 6-stage ring would halve the resident CTAs
    assert plan.launch_stages(8, False, H, OPTIN_SM10X, *SM100) == 8
    assert plan.launch_form(H, True, 2048, OPTIN_SM10X, SMEM_PER_SM_SM10X) == (1, 2, 3)


def test_l6_ctas_per_sm_is_the_thread_limit():
    assert plan.ctas_per_sm(1024, H) == 1  # sm_107: one 768-thread CTA
    assert plan.ctas_per_sm(2048, H) == 2  # sm_100 / sm_103: the launch bound
    assert plan.ctas_per_sm(1024, 4096) == 2


# ---- CBNQ_P2: two chunks per thread, two CTAs per SM -----------------------------------------------------------------
def test_p2_chunks_cover_the_row_once():
    for hidden in (512, 4096, 6144):
        block = plan.p2_block(hidden)
        chunks = [plan.p2_chunk(t, h, hidden) for t in range(block) for h in range(plan.P2_CHUNKS)]
        assert sorted(chunks) == list(range(hidden // plan.PACKED))
    assert plan.p2_block(H) == 384 and plan.p2_virtual_warps(H) == 24 == H // 8 // 32


def test_p2_mx_groups_match_today():
    """The quantize helpers shuffle within lane groups of 4 (MXFP8) / 2 (NVFP4): for every (warp, half, group) the
    group's lanes hold the chunks of one 32-element block (NVFP4: one 16-element block), as today's warp does, and
    the scale-writing lane (chunk % 4 == 0 / % 2 == 0) is the group's first lane."""
    block = plan.p2_block(H)
    for p in range(block // 32):
        for h in range(plan.P2_CHUNKS):
            vw = plan.p2_virtual_warp(p, h, H)
            for lane in range(32):
                c = plan.p2_chunk(p * 32 + lane, h, H)
                assert c == vw * 32 + lane, "virtual warp vw, virtual lane = lane"
            for g in range(8):
                cs = [plan.p2_chunk(p * 32 + 4 * g + i, h, H) for i in range(4)]
                assert cs == list(range(cs[0], cs[0] + 4)) and cs[0] % 4 == 0
                assert len({c * 8 // 32 for c in cs}) == 1  # one MX block
                for pair in (cs[:2], cs[2:]):
                    assert (
                        pair[0] % 2 == 0 and len({c * 8 // 16 for c in pair}) == 1
                    )  # one NVFP4 block


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bf16(x):
    b = _bits(_f32(x))
    return struct.unpack("<f", struct.pack("<I", (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000))[0]


def _warp_xor(vals):
    v = list(vals)
    for off in (16, 8, 4, 2, 1):
        v = [_f32(v[i] + v[i ^ off]) for i in range(32)]
    return v


def _thread_acc(row, c):
    # cbnqAccumulate: acc += v * v over the chunk's 8 elements (v BF16: v * v is exact in FP32, so fma == mul + add)
    acc = 0.0
    for e in row[c * 8 : c * 8 + 8]:
        acc = _f32(acc + _f32(e * e))
    return acc


def _reduce_today(row, hidden):
    """cbnqBlockReduceSum at hidden / 8 threads: warp xor per warp, smem[warp], final xor over warpNum lanes."""
    threads = hidden // 8
    accs = [_thread_acc(row, t) for t in range(threads)]
    warps = threads // 32
    smem = [_warp_xor(accs[32 * w : 32 * w + 32])[0] for w in range(warps)]
    return _warp_xor([smem[lane] if lane < warps else 0.0 for lane in range(32)])


def _reduce_p2(row, hidden):
    """cbnqBlockReduceSumP2: per physical warp p two xor trees (halves), stored at their virtual warps' slots
    (p2_virtual_warp), final xor over the virtual warp count (2 x physical warps)."""
    block = plan.p2_block(hidden)
    phys = block // 32
    smem = [0.0] * 32
    for p in range(phys):
        for h in range(plan.P2_CHUNKS):
            lanes = [
                _thread_acc(row, plan.p2_chunk(p * 32 + lane, h, hidden)) for lane in range(32)
            ]
            smem[plan.p2_virtual_warp(p, h, hidden)] = _warp_xor(lanes)[0]
    virt = plan.p2_virtual_warps(hidden)
    return _warp_xor([smem[lane] if lane < virt else 0.0 for lane in range(32)])


def test_p2_reduce_bits_equal_today():
    rng = random.Random(20261104)
    rows = []
    for band in (-20, -8, 0, 4, 12, 15):
        for _ in range(10):
            rows.append(
                [_bf16(rng.gauss(0.0, 1.0) * 2.0 ** (band + rng.randint(-3, 3))) for _ in range(H)]
            )
    rows.append([0.0] * H)
    single = [0.0] * H
    single[4097] = 3.0
    rows.append(single)
    mixed = [_bf16(rng.gauss(0.0, 1.0) * 2.0 ** rng.randint(-30, 15)) for _ in range(H)]
    rows += [mixed, list(reversed(mixed))]
    assert len(rows) == 64
    for row in rows:
        today, p2 = _reduce_today(row, H), _reduce_p2(row, H)
        assert [_bits(v) for v in p2] == [_bits(v) for v in today]
        assert len({_bits(v) for v in today}) == 1, "every lane ends with the same sum"


@pytest.mark.parametrize("rows", [1, 129, 1000])
def test_p2_sf_offsets_equal_served(rows):
    """Every MXFP8 scale byte of a row is staged once, by the half whose chunk owns it, where the served layout puts it
    (and the NVFP4 linear scale bytes once each)."""
    block = plan.p2_block(H)
    writers = [plan.p2_chunk(t, h, H) for t in range(block) for h in range(plan.P2_CHUNKS)]
    ks = sorted(c // 4 for c in writers if c % 4 == 0)
    assert ks == list(range(H // 32))
    assert sorted(c // 2 for c in writers if c % 2 == 0) == list(range(H // 16))
    for q in range(plan.num_quads(rows)):
        for j in range(plan.QUAD_ROWS):
            row = plan.quad_row(q, j)
            for k in ks:
                assert plan.quad_sf_offset(q, j, k, H) == plan.served_sf_offset(row, k, rows, H)


def test_p2_launch_plan():
    assert plan.p2_supported(H) and plan.p2_supported(4096) and plan.p2_supported(512)
    assert (
        not plan.p2_supported(8192)
        and not plan.p2_supported(6144 - 256)
        and not plan.p2_supported(256)
    )
    # two P2 CTAs fit an sm_107 SM by threads and by shared memory at 3 and 4 stages, not at 6
    for s in (3, 4):
        assert plan.p2_ctas_per_sm(s, True, H, 1024, SMEM_PER_SM_SM10X) == 2
    assert plan.p2_ctas_per_sm(6, True, H, 1024, SMEM_PER_SM_SM10X) == 1
    vr = dict(max_threads_per_sm=1024, smem_optin=OPTIN_SM10X, smem_per_sm=SMEM_PER_SM_SM10X)
    # the default: P2 at 3 stages
    assert plan.launch_form(H, True, **vr) == (2, 2, 3)
    assert plan.launch_form(H, False, **vr) == (2, 2, 3)
    # a shared-memory budget for one P2 pair at 3 stages keeps P2; one byte less takes the one-chunk launch
    tight = 2 * (plan.smem_bytes(3, True, H) + plan.STATIC_SMEM_BOUND + 1024)
    assert plan.launch_form(H, True, 1024, OPTIN_SM10X, tight) == (2, 2, 3)
    assert plan.launch_form(H, True, 1024, OPTIN_SM10X, tight - 1) == (1, 1, 3)
    # hidden 4096 (512 threads): the one-chunk kernel already runs two CTAs per sm_107 SM, so it keeps that form
    assert plan.launch_form(4096, True, **vr) == (1, 2, 3)
    # sm_100 / sm_103 (2048 threads per SM): the one-chunk kernel runs two CTAs per SM; P2 is not taken
    assert plan.launch_form(H, True, 2048, OPTIN_SM10X, SMEM_PER_SM_SM10X) == (1, 2, 3)
    # P2 is bitwise the one-chunk kernel only with whole warps per half; hidden 8192 is not served at all
    assert plan.launch_form(6144 - 512, True, **vr)[0] == 2


@pytest.mark.parametrize("rows", [1, 1000, 1160, 8192, 16317, 16384])
def test_p2_quads_cover_rows_with_double_grid(rows):
    g = plan.grid(rows, 2, 212)
    assert g == min(plan.num_quads(rows), 424)
    every, slots = [], []
    for cta in range(g):
        visits = plan.cta_rows(cta, g, rows)
        slots += [r for _, _, r in visits]
        consumed = [r for _, _, r in visits if r < rows]
        assert plan.producer_rows(cta, g, rows) == consumed
        every += consumed
    padded = (rows + 127) // 128 * 128
    assert sorted(slots) == list(range(padded)) and sorted(every) == list(range(rows))
