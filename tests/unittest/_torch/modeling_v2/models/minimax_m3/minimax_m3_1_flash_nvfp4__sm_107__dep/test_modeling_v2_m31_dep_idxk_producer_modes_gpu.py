# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP target's horizontal producers' index-K modes, chosen by the index_k_cache dtype.

Mode 0 (float8_e4m3fn) writes E4M3 rows, mode 1 (uint8) quantizes the index-K straight into NVFP4 index-K pages and
mode 2 (bfloat16, tests only) writes the bf16-rounded rows mode 1 quantizes. On identical inputs every other output
(q, index-Q, main K/V and their scales) is bitwise the same in all modes, and mode 1's pages equal the NVFP4 index-K
writer on mode 2's rows, for both producers (FP8 and NVFP4 main KV). Mode 0 of the FP8 producer is bitwise the
built-in trtllm::minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
    nvfp4_index_k,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAGE = 128
DIM = 128
BYTES_PER_TOKEN = DIM // 2 + DIM // 16  # 72
PAGE_BYTES = PAGE * BYTES_PER_TOKEN  # 9216
DATA_BYTES = PAGE * DIM // 2  # 8192
NUM_PAGES = 4
# Two requests of 8 query tokens (K7 rows): page edges (0, 127, 128, 255), a padded row (-1), and a slot past the
# last page (page >= num_pages), which every cache must leave untouched.
SLOTS = (0, 1, 127, 128, 129, 255, -1, 300, 301, 302, 303, 304, NUM_PAGES * PAGE, 510, 511, 256)
POSITIONS = tuple(range(100, 108)) + tuple(range(7, 15))
E4M3_448 = 0x7E
E4M3_2_POW_MINUS_9 = 0x01


def _require_sm107():
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the NVFP4 index-K pages need SM107")


def _rope_cache(max_positions, rotary_dim=64, base=5_000_000.0):
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    inverse_frequency = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    frequency = torch.outer(positions, inverse_frequency)
    return torch.stack((frequency.cos(), frequency.sin()), dim=1).contiguous()


def _main_caches(main_kv, num_kv_heads):
    """Zeroed main K/V caches (and their scales for NVFP4) with a guard page on each side; returns (pools, views)."""
    if main_kv == "fp8":
        pool = torch.zeros(
            NUM_PAGES + 2, 2, num_kv_heads, PAGE, DIM, dtype=torch.float8_e4m3fn, device="cuda"
        )
        return (pool,), (pool[1:-1],)
    data = torch.zeros(NUM_PAGES + 2, 2, num_kv_heads, PAGE, 64, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(NUM_PAGES + 2, 2, num_kv_heads, PAGE, 8, dtype=torch.uint8, device="cuda")
    return (data, scales), (data[1:-1], scales[1:-1])


def _index_cache(mode, pages_shape="4d"):
    """The zeroed index-K cache of a mode with a guard page on each side; returns (pool, view)."""
    if mode == 0:
        pool = torch.zeros(NUM_PAGES + 2, 1, PAGE, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    elif mode == 1:
        pool = torch.zeros(
            NUM_PAGES + 2, 1, PAGE, BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda"
        )
        if pages_shape == "2d":
            pool = pool.view(NUM_PAGES + 2, PAGE_BYTES)
    else:
        pool = torch.zeros(NUM_PAGES + 2, 1, PAGE, DIM, dtype=torch.bfloat16, device="cuda")
    return pool, pool[1:-1]


def _produce(main_kv, inputs, main_views, index_view, builtin=False):
    packed, slots, positions, weights, rope, inv_scales, num_heads_q, num_kv_heads = inputs
    if main_kv == "fp8":
        op = (
            torch.ops.trtllm.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert
            if builtin
            else native_ops.minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert
        )
        return op(
            packed,
            main_views[0],
            index_view,
            slots,
            num_heads_q,
            num_kv_heads,
            num_kv_heads,
            DIM,
            64,
            1e-5,
            *weights,
            rope,
            positions,
        )
    return native_ops.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
        packed,
        main_views[0],
        main_views[1],
        index_view,
        slots,
        inv_scales,
        num_heads_q,
        num_kv_heads,
        num_kv_heads,
        DIM,
        64,
        1e-5,
        *weights,
        rope,
        positions,
    )


def _inputs(num_heads_q, num_kv_heads, stress="random", seed=1234):
    torch.manual_seed(seed)
    num_tokens = len(SLOTS)
    total_heads = num_heads_q + 3 * num_kv_heads + 1
    packed = torch.randn(num_tokens, total_heads * DIM, dtype=torch.bfloat16, device="cuda")
    weights = [torch.randn(DIM, dtype=torch.bfloat16, device="cuda") * 0.5 for _ in range(4)]
    positions = torch.tensor(POSITIONS, dtype=torch.int32, device="cuda")
    index_k = packed[:, -DIM:]
    if stress == "zeros":
        # A zero row normalizes to zero: scale 2^-9, every code +0.
        index_k.zero_()
    elif stress == "group_scale":
        # Group amax spread over 2^-12..2^12 (position 0: RoPE is the identity, so the groups stay apart).
        exponents = torch.tensor([-12, -8, -4, 0, 4, 8, 12, -12], device="cuda")
        index_k.mul_(torch.exp2(exponents.float()).repeat_interleave(16).to(torch.bfloat16))
        positions.zero_()
    elif stress == "saturate":
        # (1 + w) = 1025 drives amax / 6 past 448: the scale saturates and the codes clip at +-6.
        weights[3] = torch.full((DIM,), 1024.0, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor(SLOTS, dtype=torch.int32, device="cuda")
    inv_scales = torch.tensor([1.0, 1.75, 0.625], dtype=torch.float32, device="cuda")
    return (
        packed,
        slots,
        positions,
        weights,
        _rope_cache(512),
        inv_scales,
        num_heads_q,
        num_kv_heads,
    )


def _live(slots):
    return (slots >= 0) & (slots < NUM_PAGES * PAGE)


def _writer_pages(tap_view, slots):
    """The NVFP4 index-K writer's pages (with guard pages) on the bf16 tap's rows at the live slots."""
    pool, view = _index_cache(1)
    live = _live(slots)
    rows = tap_view.reshape(-1, DIM)[slots.clamp(0, NUM_PAGES * PAGE - 1).long()]
    nvfp4_index_k.write(rows, torch.where(live, slots, -1), view)
    return pool


def _bytes(tensor):
    return tensor.view(torch.uint8)


def _run_modes(main_kv, inputs, pages_shape="4d"):
    """Run the producer once per mode on fresh caches; returns {mode: (outputs, main pools, index pool)}."""
    runs = {}
    for mode in (0, 1, 2):
        main_pools, main_views = _main_caches(main_kv, inputs[-1])
        index_pool, index_view = _index_cache(mode, pages_shape)
        outputs = _produce(main_kv, inputs, main_views, index_view)
        runs[mode] = (outputs, main_pools, index_pool)
    torch.cuda.synchronize()
    return runs


def _assert_modes_agree(runs):
    """q, index-Q and the main caches (with guard pages) bitwise equal across modes."""
    (q0, iq0), main0, _ = runs[0]
    for mode in (1, 2):
        (q, iq), main, _ = runs[mode]
        assert torch.equal(_bytes(q), _bytes(q0))
        assert torch.equal(_bytes(iq), _bytes(iq0))
        for got, want in zip(main, main0):
            assert torch.equal(_bytes(got), _bytes(want))


def _assert_pages_equal_writer(runs, slots):
    pages_pool = runs[1][2].reshape(NUM_PAGES + 2, PAGE_BYTES)
    tap_pool = runs[2][2]
    expected = _writer_pages(tap_pool[1:-1], slots).reshape(NUM_PAGES + 2, PAGE_BYTES)
    # Data and scale bytes, every page: guard pages and the pages of skipped slots stay zero.
    assert torch.equal(pages_pool[:, :DATA_BYTES], expected[:, :DATA_BYTES])
    assert torch.equal(pages_pool[:, DATA_BYTES:], expected[:, DATA_BYTES:])
    for pool in (pages_pool, _bytes(tap_pool), _bytes(runs[0][2])):
        assert torch.count_nonzero(pool[0]).item() == 0
        assert torch.count_nonzero(pool[-1]).item() == 0
    return pages_pool


def _token_bytes(pages_pool, slots):
    """Per live slot: (64 data bytes, 8 scale bytes) of the token."""
    view = pages_pool[1:-1]
    live = slots[_live(slots)].long()
    page, t = live // PAGE, live % PAGE
    data_index = (t * (DIM // 2)).unsqueeze(1) + torch.arange(DIM // 2, device="cuda")
    scale_index = (
        DATA_BYTES + (t * (DIM // 16)).unsqueeze(1) + torch.arange(DIM // 16, device="cuda")
    )
    return view[page.unsqueeze(1), data_index], view[page.unsqueeze(1), scale_index]


def _report_vs_sink_pack(runs, slots, label):
    """Report (do not assert) the fraction of code bytes that differ from the E4M3 sink + pack."""
    sink = runs[0][2][1:-1]
    pool, view = _index_cache(1)
    nvfp4_index_k.write_from_sink(sink, torch.where(_live(slots), slots, -1), slots.numel(), view)
    direct_data, direct_scale = _token_bytes(runs[1][2].reshape(NUM_PAGES + 2, PAGE_BYTES), slots)
    pack_data, pack_scale = _token_bytes(pool.reshape(NUM_PAGES + 2, PAGE_BYTES), slots)
    print(
        f"[idxk producer nvfp4] {label}: data bytes differing vs sink+pack "
        f"{(direct_data != pack_data).float().mean().item():.4f}, scale bytes "
        f"{(direct_scale != pack_scale).float().mean().item():.4f}"
    )


@pytest.mark.parametrize("main_kv", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "num_heads_q,num_kv_heads",
    [pytest.param(8, 1, id="tp8-hq8-hkv1"), pytest.param(64, 4, id="dep-hq64-hkv4")],
)
@pytest.mark.parametrize("pages_shape", ["4d", "2d"])
@torch.inference_mode()
def test_nvfp4_index_k_mode_matches_the_writer_on_the_bf16_tap(
    main_kv, num_heads_q, num_kv_heads, pages_shape
):
    _require_sm107()
    inputs = _inputs(num_heads_q, num_kv_heads)
    runs = _run_modes(main_kv, inputs, pages_shape)
    _assert_modes_agree(runs)
    _assert_pages_equal_writer(runs, inputs[1])
    _report_vs_sink_pack(runs, inputs[1], f"{main_kv} hq{num_heads_q} hkv{num_kv_heads}")


@pytest.mark.parametrize("main_kv", ["fp8", "nvfp4"])
@pytest.mark.parametrize("stress", ["zeros", "group_scale", "saturate"])
@torch.inference_mode()
def test_nvfp4_index_k_mode_stress_rows(main_kv, stress):
    _require_sm107()
    inputs = _inputs(8, 1, stress=stress)
    slots = inputs[1]
    runs = _run_modes(main_kv, inputs)
    _assert_modes_agree(runs)
    pages_pool = _assert_pages_equal_writer(runs, slots)
    data, scale = _token_bytes(pages_pool, slots)
    if stress == "zeros":
        assert torch.count_nonzero(data).item() == 0
        assert bool((scale == E4M3_2_POW_MINUS_9).all())
    elif stress == "group_scale":
        # Groups 4..7 (unrotated) keep their 2^4..2^12 / 2^-12 spread: distinct scale bytes per token.
        assert bool((scale[:, 4:].amax(dim=1) > scale[:, 4:].amin(dim=1)).all())
    else:
        assert bool((scale == E4M3_448).any())
    _report_vs_sink_pack(runs, slots, f"{main_kv} {stress}")


@pytest.mark.parametrize("main_kv", ["fp8", "nvfp4"])
@torch.inference_mode()
def test_nvfp4_index_k_mode_cuda_graph_replay(main_kv):
    """Captured once, replayed on new packed rows, positions and slots: the pages equal an eager run on them."""
    _require_sm107()
    inputs = _inputs(8, 1, seed=4321)
    packed, slots, positions = inputs[:3]
    main_pools, main_views = _main_caches(main_kv, 1)
    index_pool, index_view = _index_cache(1)
    _produce(main_kv, inputs, main_views, index_view)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_q, graph_index_q = _produce(main_kv, inputs, main_views, index_view)

    packed.copy_(torch.randn_like(packed))
    positions.copy_(torch.arange(len(SLOTS), dtype=torch.int32, device="cuda") * 7 + 3)
    slots.copy_(slots.roll(3))
    for pool in (*main_pools, index_pool):
        pool.zero_()
    graph.replay()
    torch.cuda.synchronize()

    runs = _run_modes(main_kv, inputs)
    (q, index_q), eager_main, eager_pages = runs[1]
    assert torch.equal(_bytes(graph_q), _bytes(q))
    assert torch.equal(_bytes(graph_index_q), _bytes(index_q))
    for got, want in zip(main_pools, eager_main):
        assert torch.equal(_bytes(got), _bytes(want))
    assert torch.equal(index_pool, eager_pages)
    _assert_pages_equal_writer(runs, slots)


@pytest.mark.parametrize("main_kv", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "invalid,error",
    [
        ("page_stride_alignment", "16-byte aligned"),
        ("base_alignment", "16-byte aligned"),
        ("page_count", "same number of pages"),
        ("page_bytes", "must hold 9216 bytes"),
        ("page_not_contiguous", "page must be contiguous"),
        ("tap_page_count", "same number of pages"),
        ("tap_alignment", "8-byte store alignment"),
        ("float32", "torch.float8_e4m3fn"),
    ],
)
def test_direct_index_k_modes_reject_invalid_caches(main_kv, invalid, error):
    inputs = _inputs(8, 1)
    _, main_views = _main_caches(main_kv, 1)
    if invalid == "page_stride_alignment":
        stride = PAGE_BYTES + 8
        index = torch.zeros(NUM_PAGES * stride, dtype=torch.uint8, device="cuda").as_strided(
            (NUM_PAGES, 1, PAGE, BYTES_PER_TOKEN), (stride, PAGE_BYTES, BYTES_PER_TOKEN, 1)
        )
    elif invalid == "base_alignment":
        index = torch.zeros(NUM_PAGES * PAGE_BYTES + 8, dtype=torch.uint8, device="cuda")[8:].view(
            NUM_PAGES, 1, PAGE, BYTES_PER_TOKEN
        )
    elif invalid == "page_count":
        index = _index_cache(1)[0]
    elif invalid == "page_bytes":
        index = torch.zeros(NUM_PAGES, 1, PAGE, 64, dtype=torch.uint8, device="cuda")
    elif invalid == "page_not_contiguous":
        index = torch.zeros(NUM_PAGES, 1, PAGE, 80, dtype=torch.uint8, device="cuda")[
            ..., :BYTES_PER_TOKEN
        ]
    elif invalid == "tap_page_count":
        index = _index_cache(2)[0]
    elif invalid == "tap_alignment":
        index = torch.zeros(NUM_PAGES * PAGE * DIM + 2, dtype=torch.bfloat16, device="cuda")[
            2:
        ].view(NUM_PAGES, 1, PAGE, DIM)
    else:
        index = torch.zeros(NUM_PAGES, 1, PAGE, DIM, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match=error):
        _produce(main_kv, inputs, main_views, index)


@pytest.mark.parametrize(
    "num_heads_q,num_kv_heads",
    [pytest.param(8, 1, id="tp8-hq8-hkv1"), pytest.param(64, 4, id="dep-hq64-hkv4")],
)
@torch.inference_mode()
def test_fp8_e4m3_mode_is_bitwise_the_builtin_producer(num_heads_q, num_kv_heads):
    """Mode 0 of the target's FP8 producer writes every output the built-in op writes, bit for bit."""
    inputs = _inputs(num_heads_q, num_kv_heads)
    got, want = [], []
    for builtin, sink in ((False, got), (True, want)):
        main_pools, main_views = _main_caches("fp8", num_kv_heads)
        index_pool, index_view = _index_cache(0)
        outputs = _produce("fp8", inputs, main_views, index_view, builtin=builtin)
        sink.extend((*outputs, *main_pools, index_pool))
    torch.cuda.synchronize()
    for a, b in zip(got, want):
        assert torch.equal(_bytes(a), _bytes(b))
