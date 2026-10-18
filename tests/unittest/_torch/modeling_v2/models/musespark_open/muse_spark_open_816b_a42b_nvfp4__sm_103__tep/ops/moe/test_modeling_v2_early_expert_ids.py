# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``early_expert_ids``: the persistent expert kernel's early-ids variant, on one GPU.

The variant reads its ids from tagged words that its same-stream predecessor
publishes, starts its weight stream before its dependency wait, and waits only
before it touches the activations, the outputs and the hand-off workspace.
Here the predecessor is a test kernel (``_early_ids_kernels.late_producer``) that
triggers its dependents at entry, waits, publishes the ids, then spins and only
then writes the expert kernel's activations and plain ids. Claims:

* the early variant's routed rows and row map are bitwise the default
  kernel's on the same inputs (rows 1..8, any number of active experts), with
  the live input buffers poisoned until the producer writes them and the id
  words holding stale tags (the previous epoch) and wrong ids beforehand; also
  over the 96-expert window (three ring windows);
* a missing publication costs the deadline, never a wrong value: every polling
  thread falls back to the plain ids after its wait, and the fallback counter
  says so -- a pinned host word the kernel adds to directly, which the host
  reads without a device sync (``fallbacks_seen``);
* the persistent kernel's fast route (its TMA warp reads the ids from global memory) is
  refused with the early variant;
* a captured graph of producer + early kernel replays bitwise with new ids
  and activations, and across the 32-bit epoch wrap (epoch 0 skipped on both
  sides);
* the host pairing: a publication arms the buffer for one ``local_ids``
  tensor, only a launch with exactly that tensor runs early, and every launch
  disarms it.
"""

import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    early_expert_ids as ei_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    persistent_experts as pe_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.persistent_experts import (  # noqa: E501
    HIDDEN,
    INTER,
    TOP_K,
    persistent_experts,
)

__extra_import_path__ = ["."]
from _early_ids_kernels import late_producer  # noqa: E402

assert torch.cuda.is_available(), "the expert kernels require a CUDA device"

EXPERTS = 256
SV = 16
WINDOW = 64


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _e4m3(shape, lo_exp: int, hi_exp: int, gen: torch.Generator) -> torch.Tensor:
    e = torch.randint(lo_exp, hi_exp + 1, shape, device="cuda", generator=gen).float()
    m = 1.0 + torch.rand(shape, device="cuda", generator=gen)
    return (torch.exp2(e) * m).clamp(max=448.0).to(torch.float8_e4m3fn)


def _scalars(n: int, gen: torch.Generator):
    e = torch.randint(-7, -4, (n,), device="cuda", generator=gen).float()
    return torch.exp2(e) * (1.0 + torch.rand(n, device="cuda", generator=gen))


def _stack(size: int, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return dict(
        w1=torch.randint(0, 256, (size, 2 * INTER, HIDDEN // 2), dtype=torch.uint8,
                         device="cuda", generator=gen),
        s1=_e4m3((size, 2 * INTER, HIDDEN // SV), -3, 0, gen),
        w2=torch.randint(0, 256, (size, HIDDEN, INTER // 2), dtype=torch.uint8,
                         device="cuda", generator=gen),
        s2=_e4m3((size, HIDDEN, INTER // SV), -3, 0, gen),
        scale_c=_scalars(size, gen),
        alpha1=_scalars(size, gen),
        alpha2=_scalars(size, gen),
    )  # fmt: skip


@pytest.fixture(scope="module")
def window():
    w = _stack(WINDOW, 2029)
    device = torch.device("cuda")
    early = pe_mod.early_expert_ids(device)
    persistent_experts.prepare(w["w1"], w["s1"], w["w2"], w["s2"], early=early)
    num_ctas, options = pe_mod._config(device)
    ws = pe_mod._workspace(device, num_ctas, options["handoff"])
    torch.cuda.synchronize()
    return w, early, ws


def _ids(rows: int, active: int, rng: random.Random, size: int = WINDOW) -> torch.Tensor:
    """``[rows, 8]`` window ids with ``active`` distinct local experts, distinct per row."""
    local = rng.sample(range(size), active)
    assign = {t: [] for t in range(rows)}
    for j, g in enumerate(local):
        assign[j % rows].append(g)
    for g in local:
        for t in rng.sample(range(rows), rng.randint(0, rows - 1)):
            if len(assign[t]) < TOP_K and g not in assign[t]:
                assign[t].append(g)
    out = []
    for t in range(rows):
        row = assign[t][:TOP_K]
        while len(row) < TOP_K:
            c = rng.randrange(size, EXPERTS)
            if c not in row:
                row.append(c)
        rng.shuffle(row)
        out.append(row)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def _inputs(rows: int, gen: torch.Generator):
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=gen)
    x_sf = _e4m3((rows * HIDDEN // SV,), -4, 0, gen)
    wts = (torch.rand(rows, TOP_K, device="cuda", generator=gen) * 0.2 + 0.05).to(torch.bfloat16)
    return x, x_sf, wts


def _call(w, x, x_sf, ids, wts):
    return persistent_experts(
        x, x_sf, ids, wts, w["w1"], w["s1"], w["w2"], w["s2"],
        w["scale_c"], w["alpha1"], w["alpha2"],
    )  # fmt: skip


def _routed(rows_out: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The rows a finalize reads (the map's valid entries), as raw bits."""
    sel = index.reshape(-1)
    return rows_out[sel[sel >= 0].long()].view(torch.int16)


def _poison_words(early, ws, rng: random.Random) -> None:
    """Stale tags (the current counter: the previous launch's epoch) over wrong ids."""
    torch.cuda.synchronize()
    stale = int(ws.epoch[0].item()) & 0xFFFFFFFF
    wrong = torch.tensor([rng.randrange(0, WINDOW) for _ in range(ei_mod.WORDS)], dtype=torch.int64)
    early.words.copy_(((torch.full_like(wrong, stale) << 32) | wrong).to("cuda"))


def _early_chain(w, early, live, staged, delay_ns: int, publish: bool = True):
    """Producer (publishes ``staged`` ids, then writes ``live``) + the early kernel on ``live``."""
    x_l, sf_l, ids_l, wts = live
    x_s, sf_s, ids_s = staged
    late_producer(
        [(x_s, x_l), (sf_s.view(torch.uint8), sf_l.view(torch.uint8)), (ids_s, ids_l)],
        delay_ns=delay_ns,
        early=early if publish else None,
        ids=ids_s,
    )
    early.arm(ids_l)
    return _call(w, x_l, sf_l, ids_l, wts)


@pytest.mark.parametrize("delay_ns", [0, 5_000, 50_000])
def test_early_variant_is_bitwise_the_default(window, delay_ns):
    w, early, ws = window
    rng = random.Random(_seed(f"bitwise-{delay_ns}"))
    gen = torch.Generator(device="cuda").manual_seed(_seed(f"bitwise-x-{delay_ns}"))
    torch.cuda.synchronize()
    fallbacks = early.fallback_count()
    calls = 0
    for i in range(16):
        m = 1 + i % 8
        active = rng.randint(1, min(WINDOW, TOP_K * m))
        x, x_sf, wts = _inputs(m, gen)
        ids = _ids(m, active, rng)
        want_rows, _, want_index = _call(w, x, x_sf, ids, wts)
        # live buffers poisoned with other values until the producer writes them
        x_p, sf_p, _ = _inputs(m, gen)
        live = (x_p, sf_p, _ids(m, rng.randint(1, min(WINDOW, TOP_K * m)), rng), wts)
        _poison_words(early, ws, rng)
        got_rows, _, got_index = _early_chain(w, early, live, (x, x_sf, ids), delay_ns)
        torch.cuda.synchronize()
        assert early.armed_for is None
        assert torch.equal(got_index, want_index), f"call {i} (rows {m}): row map differs"
        bad = int((_routed(got_rows, got_index) != _routed(want_rows, want_index)).sum())
        assert bad == 0, f"call {i} (rows {m}, {active} active, delay {delay_ns}): {bad} differ"
        calls += 1
    assert early.fallback_count() == fallbacks, "a published word was missed"
    print(f"{calls} early calls bitwise the default kernel, delay {delay_ns} ns, 0 fallbacks")


@pytest.mark.parametrize("delay_ns", [0, 50_000])
def test_early_variant_over_the_96_expert_window_is_bitwise_the_default(window, delay_ns):
    """Three ring windows: the early variant compiled for 96 local experts (three bit-set
    words, 96 slot entries) on ids anywhere in the window."""
    _, early, ws = window
    w = _stack(3 * 32, 2033)
    persistent_experts.prepare(w["w1"], w["s1"], w["w2"], w["s2"], early=early)
    rng = random.Random(_seed(f"bitwise96-{delay_ns}"))
    gen = torch.Generator(device="cuda").manual_seed(_seed(f"bitwise96-x-{delay_ns}"))
    torch.cuda.synchronize()
    fallbacks = early.fallback_count()
    third = 0
    for i in range(16):
        m = 1 + i % 8
        active = rng.randint(1, min(96, TOP_K * m))
        x, x_sf, wts = _inputs(m, gen)
        ids = _ids(m, active, rng, 96)
        third += int((ids >= 64).logical_and(ids < 96).sum())
        want_rows, _, want_index = _call(w, x, x_sf, ids, wts)
        x_p, sf_p, _ = _inputs(m, gen)
        live = (x_p, sf_p, _ids(m, rng.randint(1, min(96, TOP_K * m)), rng, 96), wts)
        _poison_words(early, ws, rng)
        got_rows, _, got_index = _early_chain(w, early, live, (x, x_sf, ids), delay_ns)
        torch.cuda.synchronize()
        assert early.armed_for is None
        assert torch.equal(got_index, want_index), f"call {i} (rows {m}): row map differs"
        bad = int((_routed(got_rows, got_index) != _routed(want_rows, want_index)).sum())
        assert bad == 0, f"call {i} (rows {m}, {active} active, delay {delay_ns}): {bad} differ"
    assert third > 0, "no id fell in the third window"
    assert early.fallback_count() == fallbacks, "a published word was missed"
    print(f"16 early calls over 96 experts bitwise the default ({third} third-window ids)")


def test_missing_publication_falls_back(window):
    """No publication: every polling thread waits out the deadline, then reads the
    plain ids after its dependency wait; the values are the default kernel's."""
    w, early, ws = window
    saved = early.deadline_ns
    early.deadline_ns = 20_000
    try:
        persistent_experts.prepare(w["w1"], w["s1"], w["w2"], w["s2"], early=early)
        rng = random.Random(_seed("fallback"))
        gen = torch.Generator(device="cuda").manual_seed(_seed("fallback-x"))
        num_ctas, _ = pe_mod._config(torch.device("cuda"))
        for m in (8, 3, 1):
            x, x_sf, wts = _inputs(m, gen)
            ids = _ids(m, rng.randint(1, TOP_K * m), rng)
            want_rows, _, want_index = _call(w, x, x_sf, ids, wts)
            x_p, sf_p, _ = _inputs(m, gen)
            live = (x_p, sf_p, _ids(m, 1, rng), wts)
            _poison_words(early, ws, rng)
            before = early.fallback_count()
            got_rows, _, got_index = _early_chain(
                w, early, live, (x, x_sf, ids), 1_000, publish=False
            )
            torch.cuda.synchronize()
            assert torch.equal(got_index, want_index)
            assert int((_routed(got_rows, got_index) != _routed(want_rows, want_index)).sum()) == 0
            assert early.fallback_count() - before == num_ctas * TOP_K * m
            # The pinned host counters read the same without a device sync once the
            # launches are done.
            synced = sum(buffer.fallback_count() for buffer in ei_mod._BUFFERS)
            assert ei_mod.fallbacks_seen() == synced >= early.fallback_count()
    finally:
        early.deadline_ns = saved


def test_the_fallback_counter_is_pinned_host_memory(window):
    _, early, _ = window
    assert early.fallbacks.device.type == "cpu" and early.fallbacks.is_pinned()
    assert early.fallbacks_address == early.fallbacks.data_ptr()
    assert early in ei_mod._BUFFERS


def test_the_fast_route_is_refused(window, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe._persistent_experts.kernel import (  # noqa: E501
        PersistentExpertsKernel,
    )

    monkeypatch.setitem(pe_mod.OPTIONS, "fastroute", 1)
    with pytest.raises(ValueError, match="fastroute 0"):
        pe_mod.early_expert_ids(torch.device("cuda"))
    with pytest.raises(ValueError, match="fastroute 0"):
        PersistentExpertsKernel(148, 64, fastroute=1, early_ids=0x1000, early_fallbacks=0x2000)
    with pytest.raises(ValueError, match="fallback counter"):
        PersistentExpertsKernel(148, 64, early_ids=0x1000, early_fallbacks=0)


def test_graph_replay_across_the_epoch_wrap(window):
    """Producer + early kernel captured once, replayed with new ids and activations,
    the workspace counter starting at 0xFFFFFFFD so the launches cross the wrap."""
    w, early, ws = window
    rng = random.Random(_seed("graph"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("graph-x"))
    m = 8
    x_s, sf_s, wts = _inputs(m, gen)
    ids_s = _ids(m, 5, rng)
    x_l, sf_l, _ = _inputs(m, gen)
    ids_l = _ids(m, 3, rng)
    live = (x_l, sf_l, ids_l, wts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _early_chain(w, early, live, (x_s, sf_s, ids_s), 2_000)
        with torch.cuda.graph(graph, stream=stream):
            out = _early_chain(w, early, live, (x_s, sf_s, ids_s), 2_000)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    ws.images.zero_()
    ws.flags.zero_()
    ws.epoch.fill_(-3)
    fallbacks = early.fallback_count()
    for trial in range(6):
        x_s.copy_(_inputs(m, gen)[0])
        ids_s.copy_(_ids(m, [4, 6, 1, 8, 12, 2][trial], rng))
        x_l.copy_(_inputs(m, gen)[0])
        _poison_words(early, ws, rng)
        graph.replay()
        torch.cuda.synchronize()
        want_rows, _, want_index = _call(w, x_s, sf_s, ids_s, wts)
        torch.cuda.synchronize()
        got_rows, _, got_index = out
        assert torch.equal(got_index, want_index), f"replay {trial}: row map differs"
        bad = int((_routed(got_rows, got_index) != _routed(want_rows, want_index)).sum())
        assert bad == 0, f"replay {trial}: {bad} differ"
    assert early.fallback_count() == fallbacks
    epoch = int(ws.epoch[0].item()) & 0xFFFFFFFF
    # 12 launches from 0xFFFFFFFD: 0xFFFFFFFE, 0xFFFFFFFF, then 1..10 (0 skipped)
    assert epoch == 10, f"6 replays + 6 default launches from 0xFFFFFFFD: at {epoch:#x}"


def test_host_pairing(window):
    w, early, _ = window
    a = torch.zeros(8, TOP_K, dtype=torch.int32, device="cuda")
    b = torch.zeros(8, TOP_K, dtype=torch.int32, device="cuda")
    assert early.may_publish()
    early.arm(a)
    assert not early.may_publish()
    with pytest.raises(AssertionError):
        early.arm(b)
    assert not ei_mod.launches_early(early, b)  # another tensor: default, and disarmed
    assert early.may_publish()
    early.arm(a)
    assert ei_mod.launches_early(early, a)
    assert not ei_mod.launches_early(early, a)  # consumed
    # every expert launch on the workspace disarms the buffer
    gen = torch.Generator(device="cuda").manual_seed(_seed("pairing"))
    x, x_sf, wts = _inputs(8, gen)
    early.arm(a)
    _call(w, x, x_sf, b, wts)
    assert early.armed_for is None
    torch.cuda.synchronize()
