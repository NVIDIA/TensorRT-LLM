# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the sharded cold-page codec's kernels and for its encode / decode round trip on one rank.

The copy kernel, the page checksum kernel and the page-id matcher are checked bit for bit against PyTorch references
(the checksum also when launched on several streams at once and when its output held stale values); the codec's restore
must reproduce the evicted pages byte for byte. The multi-rank paths (the all-gather, the TP agreement, the
cross-rank page-id match and its fatal cases) need a TP group and are covered by the 8-rank microbench."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.common.match_page_ids import (  # noqa: E501
    match_page_ids,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.common.page_checksums import (  # noqa: E501
    page_checksums,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.common.paged_segment_copy import (  # noqa: E501
    SEG_COLS,
    paged_segment_copy,
)

assert torch.cuda.is_available(), "the sharded cold-page codec requires a CUDA device"


def _seg(rows):
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, SEG_COLS).cuda()


@pytest.mark.parametrize("n,q_blocks", ((1, 1), (7, 1), (129, 3), (64, 8)))
def test_paged_segment_copy_matches_reference(n: int, q_blocks: int) -> None:
    """Indexed segment copies with per-rank-block source positions equal the indexed torch copy."""
    g = torch.Generator(device="cuda").manual_seed(n * 31 + q_blocks)
    slots, src_stride, dst_stride = 256, 4096, 40960
    src = torch.randint(
        0, 256, (q_blocks, slots, src_stride), dtype=torch.uint8, device="cuda", generator=g
    )
    dst = torch.zeros(slots, dst_stride, dtype=torch.uint8, device="cuda")
    seg_bytes = [1024, 48, 3008]  # multiples of 16, as the kernel requires
    rows, off_src, off_dst = [], 16, 0
    for q in range(q_blocks):
        for nb in seg_bytes:
            rows.append(
                [src.data_ptr(), src_stride, off_src, dst.data_ptr(), dst_stride, off_dst, nb, q]
            )
            off_dst += nb
    segs = _seg(rows)
    perm = torch.randperm(slots, generator=torch.Generator().manual_seed(n))
    src_idx = torch.randint(
        0, slots, (q_blocks, n), dtype=torch.int32, generator=torch.Generator().manual_seed(7)
    )
    dst_idx = perm[:n].to(torch.int32)
    paged_segment_copy(
        segs,
        src_idx.cuda().reshape(-1),
        dst_idx.cuda(),
        q_stride=slots * src_stride,
        idx_q_stride=n,
        max_segment_bytes=max(seg_bytes),
    )
    torch.cuda.synchronize()
    ref = torch.zeros_like(dst)
    off_dst = 0
    for q in range(q_blocks):
        for nb in seg_bytes:
            for i in range(n):
                ref[dst_idx[i], off_dst : off_dst + nb] = src[q, src_idx[q, i], 16 : 16 + nb]
            off_dst += nb
    assert torch.equal(dst, ref)


def test_paged_segment_copy_large_segment() -> None:
    """A segment longer than one CTA block (16 KiB) is split over the grid's z extent."""
    nbytes = 110592
    src = torch.randint(0, 256, (8, nbytes), dtype=torch.uint8, device="cuda")
    dst = torch.zeros(8, nbytes, dtype=torch.uint8, device="cuda")
    segs = _seg([[src.data_ptr(), nbytes, 0, dst.data_ptr(), nbytes, 0, nbytes, 0]])
    idx = torch.tensor([3, 0, 7], dtype=torch.int32, device="cuda")
    out = torch.tensor([1, 2, 5], dtype=torch.int32, device="cuda")
    paged_segment_copy(segs, idx, out, max_segment_bytes=nbytes)
    torch.cuda.synchronize()
    assert torch.equal(dst[out.long()], src[idx.long()])


def test_match_page_ids_reorder_and_missing() -> None:
    """Each page of this rank is found in every rank block (same position first, else the first equal id); a page
    some block lacks, or a wrong tag, counts as a mismatch and keeps its own position."""
    world, n, cpb, toff, rank, magic = 4, 6, 64, 48, 1, 0x53484344
    ids = np.arange(100, 100 + n, dtype=np.int64)
    blocks = [ids.copy() for _ in range(world)]
    blocks[2] = ids[[1, 0, 2, 3, 5, 4]]  # rank 2 lists the same pages in another order
    blocks[3] = ids.copy()
    blocks[3][4] = 999  # rank 3 lacks page 4 of this rank
    staging = torch.zeros(world, n, cpb, dtype=torch.uint8, device="cuda")
    trailer = torch.zeros(world, n, 2, dtype=torch.int64)
    for q in range(world):
        trailer[q, :, 0] = torch.from_numpy(blocks[q])
        trailer[q, :, 1] = (magic << 32) | q
    trailer[0, 5, 1] = (magic << 32) | 3  # wrong tag on rank 0's page 5
    staging[:, :, toff : toff + 16] = trailer.view(torch.uint8).view(world, n, 16).cuda()
    source = torch.full((world * n,), -1, dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int64, device="cuda")
    match_page_ids(
        source,
        staging=staging.data_ptr(),
        counters=counters,
        n=n,
        world=world,
        rank=rank,
        cold_page_bytes=cpb,
        trailer_off=toff,
        magic=magic,
    )
    torch.cuda.synchronize()
    got = source.view(world, n).cpu().tolist()
    assert got[0] == [0, 1, 2, 3, 4, 5]
    assert got[1] == [0, 1, 2, 3, 4, 5]
    assert got[2] == [1, 0, 2, 3, 5, 4]
    assert got[3] == [0, 1, 2, 3, 4, 5]  # page 4 missing: own position, counted below
    bad, reordered = counters.tolist()
    assert bad == 2  # rank 3 page 4 missing, rank 0 page 5 wrong tag
    assert reordered == 4  # rank 2 positions 0, 1, 4, 5


def _checksum_plan(slots: int = 256, layers: int = 4, key_b: int = 36864, idx_b: int = 8448):
    """A single-rank codec plan over two interleaved pools (KEY and INDEX_KEY per layer), and those pools."""
    sc, provider = _single_rank_provider()
    g = torch.Generator(device="cuda").manual_seed(5)
    key_pool = torch.randint(
        -(2**62), 2**62, (slots, layers * key_b // 8), dtype=torch.int64, device="cuda", generator=g
    )
    idx_pool = torch.randint(
        -(2**62), 2**62, (slots, layers * idx_b // 8), dtype=torch.int64, device="cuda", generator=g
    )

    def hot(base, slot, n):
        return SimpleNamespace(raw_base=base, raw_slot_bytes=slot, raw_bytes=n)

    lifecycle = SimpleNamespace(
        layers={
            layer: {
                "key": hot(key_pool.data_ptr() + layer * key_b, layers * key_b, key_b),
                "index_key": hot(idx_pool.data_ptr() + layer * idx_b, layers * idx_b, idx_b),
            }
            for layer in range(layers)
        }
    )
    plan = provider._plan(lifecycle)
    provider.close()
    return plan, (key_pool, idx_pool)


def _expression_checksums(plan, slots: torch.Tensor) -> torch.Tensor:
    """The checksum as the codec computed it before the kernel: gather each page, then (words * weights).sum(dim=1)."""
    n = slots.shape[0]
    vbuf = torch.empty(n * plan.page_bytes, dtype=torch.uint8, device="cuda")
    paged_segment_copy(
        plan.gather,
        slots,
        torch.arange(n, dtype=torch.int32, device="cuda"),
        dst_add=vbuf.data_ptr(),
        max_segment_bytes=max(r.nbytes for r in plan.regions),
    )
    words = vbuf.view(torch.int64).view(n, plan.page_bytes // 8)
    return (words * plan.weights).sum(dim=1)


def _launch(plan, slots, out) -> None:
    page_checksums(
        plan.gather,
        slots,
        plan.weights,
        out,
        max_segment_bytes=max(r.nbytes for r in plan.regions),
    )


@pytest.mark.parametrize("n", (1, 37, 127, 128))
def test_page_checksums_equal_expression(n: int) -> None:
    """The kernel's checksums equal the gather + (words * weights).sum(dim=1) expression bit for bit, with slots read
    through a strided index (a column of the codec's (dst, src) pairs) into a strided output (the trailer column)."""
    plan, pools = _checksum_plan()
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(n))[:n]
    pairs = torch.stack([torch.zeros(n, dtype=torch.int32), perm.to(torch.int32)], dim=1).cuda()
    slots = pairs[:, 1]
    trailers = torch.full((128, 2), -7, dtype=torch.int64, device="cuda")
    _launch(plan, slots, trailers[:n, 0])
    torch.cuda.synchronize()
    assert torch.equal(trailers[:n, 0], _expression_checksums(plan, slots.contiguous()))
    assert torch.all(trailers[:n, 1] == -7) and torch.all(trailers[n:] == -7)
    del pools


def test_page_checksums_concurrent_streams() -> None:
    """Eight launches in flight on eight streams at once, each into its own output, equal a serial reference."""
    plan, pools = _checksum_plan()
    streams = [torch.cuda.Stream() for _ in range(8)]
    sizes = [1, 37, 127, 128, 64, 5, 128, 100]
    slot_sets, outs = [], []
    for k, n in enumerate(sizes):
        slot_sets.append(
            torch.randperm(256, generator=torch.Generator().manual_seed(100 + k))[:n]
            .to(torch.int32)
            .cuda()
        )
        outs.append(torch.full((n,), 12345, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()
    for stream, slots, out in zip(streams, slot_sets, outs):
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch(plan, slots, out)
    torch.cuda.synchronize()
    for slots, out in zip(slot_sets, outs):
        assert torch.equal(out, _expression_checksums(plan, slots))
    del pools


def test_page_checksums_no_stale_sum() -> None:
    """Back-to-back launches on different streams into the same output (ordered by an event, as the codec orders its
    calls), starting from garbage: each result is the checksum of its own pages only."""
    plan, pools = _checksum_plan()
    out = torch.full((128,), -(2**61) + 3, dtype=torch.int64, device="cuda")
    a = torch.randperm(256, generator=torch.Generator().manual_seed(1))[:128].to(torch.int32).cuda()
    b = torch.randperm(256, generator=torch.Generator().manual_seed(2))[:37].to(torch.int32).cuda()
    s1, s2, done = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Event()
    torch.cuda.synchronize()
    with torch.cuda.stream(s1):
        _launch(plan, a, out[:128])
        done.record()
    with torch.cuda.stream(s2):
        s2.wait_event(done)
        _launch(plan, b, out[:37])
    torch.cuda.synchronize()
    assert torch.equal(out[:37], _expression_checksums(plan, b))
    assert torch.equal(out[37:], _expression_checksums(plan, a)[37:])
    del pools


def test_codec_kernels_run_on_the_current_stream() -> None:
    """The codec's kernels launch on torch's current stream (the KV-V2 copy stream the codec makes current), not on the
    default stream: with the default stream still busy, each result is visible through a copy on the side stream."""
    plan, pools = _checksum_plan()
    side, n = torch.cuda.Stream(), 8
    slots = torch.arange(n, dtype=torch.int32, device="cuda")
    sums = torch.zeros(n, dtype=torch.int64, device="cuda")
    dst = torch.zeros(n, plan.page_bytes, dtype=torch.uint8, device="cuda")
    world, cpb, toff = 2, 64, 48
    staging = torch.zeros(world, n, cpb, dtype=torch.uint8, device="cuda")
    trailer = torch.zeros(world, n, 2, dtype=torch.int64)
    trailer[:, :, 0] = torch.arange(100, 100 + n)
    trailer[:, :, 1] = torch.arange(world)[:, None]
    staging[:, :, toff : toff + 16] = trailer.view(torch.uint8).view(world, n, 16).cuda()
    source = torch.zeros(world * n, dtype=torch.int32, device="cuda")
    counters = torch.zeros(2, dtype=torch.int64, device="cuda")
    launches = (
        (sums, lambda: _launch(plan, slots, sums), _expression_checksums(plan, slots)),
        (
            dst,
            lambda: paged_segment_copy(
                plan.gather,
                slots,
                slots,
                dst_add=dst.data_ptr(),
                max_segment_bytes=max(r.nbytes for r in plan.regions),
            ),
            None,
        ),
        (
            source,
            lambda: match_page_ids(
                source,
                staging=staging.data_ptr(),
                counters=counters,
                n=n,
                world=world,
                rank=0,
                cold_page_bytes=cpb,
                trailer_off=toff,
                magic=0,
            ),
            torch.arange(n, dtype=torch.int32, device="cuda").repeat(world),
        ),
    )
    for out, launch, expected in launches:
        launch()  # compile outside the timed part
        torch.cuda.synchronize()
        if expected is None:
            expected = out.clone()
        host = torch.empty(out.shape, dtype=out.dtype, pin_memory=True)
        torch.cuda._sleep(1_000_000_000)
        done = torch.cuda.Event()
        with torch.cuda.stream(side):
            out.fill_(0 if out.dtype == torch.uint8 else -1)
            launch()
            host.copy_(out, non_blocking=True)
            done.record()
        done.synchronize()
        default_busy = not torch.cuda.default_stream().query()
        torch.cuda.synchronize()
        assert default_busy, "the default stream finished first; the check needs a longer sleep"
        assert torch.equal(host, expected.cpu())
    del pools


def _single_rank_provider():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.common import (  # noqa: E501
        shard_cold_page_codec as sc,
    )
    from tensorrt_llm.mapping import Mapping

    return sc, sc.ShardColdPageProvider(
        Mapping(world_size=1, rank=0, tp_size=1), abort_on_error=False
    )


def test_codec_round_trip_single_rank() -> None:
    """encode -> wipe -> decode into other slots reproduces the pages byte for byte; a corrupted trailer is reported
    loudly (raised here, since the test turns the abort off)."""
    sc, provider = _single_rank_provider()
    try:
        _round_trip(sc, provider)
    finally:
        provider.close()


def _round_trip(sc, provider) -> None:
    layers, key_b, idx_b, slots = 4, 36864, 8448, 64
    g = torch.Generator(device="cuda").manual_seed(11)
    key_pool = torch.randint(
        0, 256, (slots, layers * key_b), dtype=torch.uint8, device="cuda", generator=g
    )
    idx_pool = torch.randint(
        0, 256, (slots, layers * idx_b), dtype=torch.uint8, device="cuda", generator=g
    )
    ref_key, ref_idx = key_pool.clone(), idx_pool.clone()

    def hot(base, slot, n):
        return SimpleNamespace(raw_base=base, raw_slot_bytes=slot, raw_bytes=n)

    lifecycle = SimpleNamespace(
        layers={
            layer: {
                "key": hot(key_pool.data_ptr() + layer * key_b, layers * key_b, key_b),
                "index_key": hot(idx_pool.data_ptr() + layer * idx_b, layers * idx_b, idx_b),
            }
            for layer in range(layers)
        }
    )
    state = sc._CodecState(layer_ids=tuple(range(layers)))
    provider.configure(state, [lifecycle])
    cpb = state.plans[0].cold_page_bytes
    assert cpb == layers * (key_b + idx_b) + sc.TRAILER_BYTES
    cold = torch.empty((32, cpb), dtype=torch.uint8, pin_memory=True)
    stream = torch.cuda.Stream()

    def pairs(dst, src):
        return np.ascontiguousarray(
            np.stack([np.asarray(dst, np.int32), np.asarray(src, np.int32)], axis=1)
        )

    hot_src, cold_slots, hot_dst = [3, 9, 20, 41, 5], [7, 0, 31, 12, 4], [50, 51, 60, 2, 33]
    enc = pairs(cold_slots, hot_src)
    provider.encode_cold_pages(
        state, 0, cold.data_ptr(), enc.ctypes.data, len(hot_src), stream.cuda_stream
    )
    stream.synchronize()
    with torch.cuda.stream(stream):
        key_pool[torch.tensor(hot_dst, device="cuda")] = 0
        idx_pool[torch.tensor(hot_dst, device="cuda")] = 0
    dec = pairs(hot_dst, cold_slots)
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), dec.ctypes.data, len(hot_dst), stream.cuda_stream
    )
    stream.synchronize()
    provider._poll_match(blocking=True)
    d, s = torch.tensor(hot_dst, device="cuda"), torch.tensor(hot_src, device="cuda")
    assert torch.equal(key_pool[d], ref_key[s]) and torch.equal(idx_pool[d], ref_idx[s])
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[d] = False
    assert torch.equal(key_pool[untouched], ref_key[untouched])
    assert torch.equal(idx_pool[untouched], ref_idx[untouched])

    # A cold page whose trailer tag is wrong cannot be matched: the next check raises.
    trailer_off = state.plans[0].trailer_off
    cold[cold_slots[2], trailer_off + 12 : trailer_off + 16] = 0
    provider.decode_cold_pages(
        state, 0, cold.data_ptr(), dec.ctypes.data, len(hot_dst), stream.cuda_stream
    )
    stream.synchronize()
    with pytest.raises(RuntimeError, match="content mismatch"):
        provider._poll_match(blocking=True)


@pytest.mark.parametrize("reuse", [True, False], ids=["reuse-on", "reuse-off"])
def test_provider_follows_block_reuse(monkeypatch, reuse: bool) -> None:
    """A TEP server gets the provider exactly when its block reuse is on; without reuse, without a KV config, with
    TLLM_K31_SHARD_COLD_PAGES=0, or on a layout that does not replicate the attention pools it gets
    None and KV-V2 keeps its lossless codec."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.common import (  # noqa: E501
        shard_cold_page_codec as sc,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    monkeypatch.delenv(sc.SHARD_COLD_PAGES_ENV, raising=False)
    tep8 = Mapping(world_size=8, rank=0, tp_size=8, gpus_per_node=4)
    config = KvCacheConfig(enable_block_reuse=reuse)
    provider = sc.create_shard_cold_page_provider(tep8, config)
    assert (provider is not None) == reuse
    assert sc.create_shard_cold_page_provider(tep8, None) is None
    assert (
        sc.create_shard_cold_page_provider(Mapping(world_size=1, rank=0, tp_size=1), config) is None
    )
    dep8 = Mapping(world_size=8, rank=0, tp_size=8, enable_attention_dp=True, gpus_per_node=4)
    assert sc.create_shard_cold_page_provider(dep8, config) is None
    monkeypatch.setenv(sc.SHARD_COLD_PAGES_ENV, "0")
    assert sc.create_shard_cold_page_provider(tep8, config) is None
