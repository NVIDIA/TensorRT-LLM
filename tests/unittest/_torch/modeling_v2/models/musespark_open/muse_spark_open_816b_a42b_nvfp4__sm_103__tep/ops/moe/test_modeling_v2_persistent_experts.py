# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``persistent_experts``: the persistent NVFP4 expert kernel against TRTLLM-gen, row for row.

Claims gated here:

* Every local slot's FC2 row is bitwise the row ``fp4_block_scale_moe_runner``
  computes for that slot from the same bytes (the oracle), and the row map is
  ``t * 8 + k`` for local slots and ``-1`` otherwise: on every certified cell,
  and on randomized calls covering every row count 1..8, all three windows
  (32, 64 and the three-window 96), and
  block scales and per-expert FP32 scalars with arbitrary mantissas, redrawn
  for every call. Bitwise rows need TRTLLM-gen's FC1 epilogue in its exact
  operation order. Short-mantissa scalars rarely put an FP32 value on a
  rounding midpoint, so they cannot tell that order from a reassociated one;
  arbitrary scalars do it in about 1 % of calls, so the randomized test runs
  enough calls to see it (``MODELING_V2_PERSISTENT_EXPERTS_CALLS`` raises the count).
* A token that repeats a local expert routes it once, through the first such
  slot, exactly as TRTLLM-gen does; the dropped slots are ``-1``, so no
  finalize reads a row the op did not write.
* The plain-torch FP64 ``reference`` agrees with the op within the band below,
  and the op's error against it is no larger than TRTLLM-gen's (bitwise rows
  make the two errors equal; the comparison is kept for the fallback budget).
* CUDA-graph replay with new ids and activations reproduces eager launches.
* The kernel's delay sites (late CTAs, late FC1 producers, late FC2 consumers)
  change no bit.
* Epochs: launches across the 32-bit wrap stay bitwise (epoch 0 is never
  used), and the rotation keeps every slot's hand-off tags within 65 launches
  of the current epoch.
* A grid larger than the resident SMs is refused instead of launched, and a
  call under graph capture refuses to compile or to create the workspace.

The weights are random bytes: both kernels read the same stock layout, so
whatever function the bytes define is the same function for both.
"""

import os
import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as moe_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.persistent_experts import (  # noqa: E501
    rows as expert_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    persistent_experts as pe_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.persistent_experts import (  # noqa: E501
    HIDDEN,
    INTER,
    TOP_K,
    PersistentExperts,
    persistent_experts,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "the expert kernels require a CUDA device"

EXPERTS = 256
SV = 16
#: FC2-row band of the op against the FP64 reference, in BF16 ulp of each
#: row's largest magnitude (element) and of the rows' RMS: the reference
#: divides exactly where the kernels use rcp.approx, so a few FP4 codes of
#: the intermediate land on the neighbouring grid point.
ELEMENT_ULP, RMS_ULP = 64.0, 4.0
#: Randomized bitwise calls (a reassociated FC1 epilogue fails about 1 % of
#: calls on this data, so the default sees it with near certainty).
RANDOM_CALLS = int(os.environ.get("MODELING_V2_PERSISTENT_EXPERTS_CALLS") or 1200)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _e4m3(shape, lo_exp: int, hi_exp: int, gen: torch.Generator) -> torch.Tensor:
    """E4M3 values ``2^U{lo..hi} * U(1, 2)``: arbitrary 3-bit mantissas, not powers of two."""
    e = torch.randint(lo_exp, hi_exp + 1, shape, device="cuda", generator=gen).float()
    m = 1.0 + torch.rand(shape, device="cuda", generator=gen)
    return (torch.exp2(e) * m).clamp(max=448.0).to(torch.float8_e4m3fn)


def _scalars(num_local: int, gen: torch.Generator, lo: int = -7, hi: int = -5):
    """Per-expert FP32 scalars ``2^U{lo..hi} * U(1, 2)`` with full random mantissas."""
    e = torch.randint(lo, hi + 1, (num_local,), device="cuda", generator=gen).float()
    return torch.exp2(e) * (1.0 + torch.rand(num_local, device="cuda", generator=gen))


def _window(num_local: int, gen: torch.Generator):
    """``num_local`` random NVFP4 experts with arbitrary block scales and scalars."""
    w1 = torch.randint(
        0, 256, (num_local, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2 = torch.randint(
        0, 256, (num_local, HIDDEN, INTER // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    return dict(
        w1=w1,
        s1=_e4m3((num_local, 2 * INTER, HIDDEN // SV), -3, 0, gen),
        w2=w2,
        s2=_e4m3((num_local, HIDDEN, INTER // SV), -3, 0, gen),
        scale_c=_scalars(num_local, gen),
        alpha1=_scalars(num_local, gen),
        alpha2=_scalars(num_local, gen),
    )


@pytest.fixture(scope="module")
def windows():
    gen = torch.Generator(device="cuda").manual_seed(2026)
    out = {64: _window(64, gen), 32: _window(32, gen)}
    # The three-window 96-expert window (drawn last, so the other two keep their bytes).
    out[96] = _window(96, gen)
    return out


def _ids(rows: int, active: int, num_local: int, rng: random.Random) -> torch.Tensor:
    """``[rows, 8]`` window ids: exactly ``active`` distinct local experts, distinct per row."""
    local = rng.sample(range(num_local), active)
    assign = {t: [] for t in range(rows)}
    for j, g in enumerate(local):  # every local expert at least once
        assign[j % rows].append(g)
    for g in local:  # and some on more rows
        for t in rng.sample(range(rows), rng.randint(0, rows - 1)):
            if len(assign[t]) < TOP_K and g not in assign[t]:
                assign[t].append(g)
    out = []
    for t in range(rows):
        row = assign[t][:TOP_K]
        while len(row) < TOP_K:
            c = rng.randrange(num_local, EXPERTS)
            if c not in row:
                row.append(c)
        rng.shuffle(row)
        out.append(row)
    ids = torch.tensor(out, dtype=torch.int32, device="cuda")
    assert len({v for v in ids.reshape(-1).tolist() if v < num_local}) == active
    return ids


def _drop_some_nonlocal(ids: torch.Tensor, num_local: int, rng: random.Random) -> torch.Tensor:
    """The same routing with about half of the non-local ids written as ``-1``."""
    out = ids.clone()
    for t, k in (ids >= num_local).nonzero().tolist():
        if rng.random() < 0.5:
            out[t, k] = -1
    return out


def _inputs(rows: int, gen: torch.Generator):
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=gen)
    x_sf = _e4m3((rows * HIDDEN // SV,), -4, 0, gen)
    w = (torch.rand(rows, TOP_K, device="cuda", generator=gen) * 0.2 + 0.05).to(torch.bfloat16)
    return x, x_sf, w


def _args(w, x, x_sf, ids, wts):
    return (
        x, x_sf, ids, wts,
        w["w1"], w["s1"], w["w2"], w["s2"],
        w["scale_c"], w["alpha1"], w["alpha2"],
    )  # fmt: skip


def _oracle(w, x, x_sf, ids, wts, num_local):
    gemm2, _, index = moe_op(
        None, None, x, x_sf, w["w1"], w["s1"], None, None, None, None, w["w2"], w["s2"], None,
        w["scale_c"], w["alpha1"], w["alpha2"], EXPERTS, TOP_K, None, None, INTER, 0, num_local,
        None, 1, False, 0, topk_weights=wts, topk_ids=ids, output=None,
        tune_max_num_tokens=8192, use_dp=False,
    )  # fmt: skip
    return gemm2, index


def _check_rows(rows, index, gemm2, oracle_index, ids, num_local):
    """Bitwise per local slot; returns the number of local slots.

    Also through the finalize's side of the contract (``expert_rows``): the
    rows a finalize selects from the op's triple equal, bit for bit, the rows
    it selects from TRTLLM-gen's.
    """
    m = ids.shape[0]
    local = expert_rows.first_local_slots(ids, num_local)
    expect = expert_rows.slot_row_map(ids, num_local)
    assert torch.equal(index, expect), "row map must be t * 8 + k for local slots, -1 otherwise"
    assert torch.equal(oracle_index.reshape(-1)[: m * TOP_K].reshape(m, TOP_K) >= 0, local), (
        "oracle disagrees on locality"
    )
    ours_sel, ours_valid = expert_rows.gather_routed_rows(rows, index, m, HIDDEN)
    theirs_sel, theirs_valid = expert_rows.gather_routed_rows(gemm2, oracle_index, m, HIDDEN)
    assert torch.equal(ours_valid, theirs_valid) and torch.equal(ours_valid, local)
    assert torch.equal(ours_sel.view(torch.int32), theirs_sel.view(torch.int32)), (
        "the finalize would read different rows from the two producers"
    )
    n = 0
    for t, k in local.nonzero().tolist():
        ours = rows[t * TOP_K + k]
        theirs = gemm2[int(oracle_index.reshape(m, TOP_K)[t, k]), :HIDDEN]
        diff = (ours.view(torch.int16) != theirs.view(torch.int16)).sum().item()
        assert diff == 0, f"slot ({t}, {k}) expert {int(ids[t, k])}: {diff} of {HIDDEN} bits differ"
        n += 1
    return n


def _differing(rows, index, gemm2, oracle_index, ids, num_local):
    """(local slots, differing BF16 elements over them, row map correct), vectorized."""
    m = ids.shape[0]
    local = expert_rows.first_local_slots(ids, num_local)
    expect = expert_rows.slot_row_map(ids, num_local)
    oi = oracle_index.reshape(-1)[: m * TOP_K].reshape(m, TOP_K)
    map_ok = torch.equal(index, expect) and torch.equal(oi >= 0, local)
    sel = local.reshape(-1).nonzero().squeeze(1)
    if sel.numel() == 0:
        return 0, 0, map_ok
    ours = rows[sel].view(torch.int16)
    theirs = gemm2[oi.reshape(-1)[sel].long(), :HIDDEN].view(torch.int16)
    return int(sel.numel()), int((ours != theirs).sum().item()), map_ok


def _band(rows, ref_rows, index):
    sel = index.reshape(-1) >= 0
    out, ref = rows[sel].float(), ref_rows[sel].float()
    scale = ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    elem = ((out - ref).abs() / scale).max().item() / 2**-8
    rms = ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()
    return elem, rms / 2**-8


@pytest.mark.parametrize("cell", PersistentExperts.CELLS, ids=lambda c: c.why[:48])
def test_cells_bitwise_vs_trtllm_gen(windows, certifying, cell):  # noqa: F811 (fixture by name)
    certifying(persistent_experts)
    spec = cell.spec
    rows, num_local, active = spec["rows"], spec["window"], spec["active"]
    rng = random.Random(_seed(f"{rows}-{num_local}-{active}"))
    gen = torch.Generator(device="cuda").manual_seed(_seed(f"x-{rows}-{num_local}-{active}"))
    w = windows[num_local]
    x, x_sf, wts = _inputs(rows, gen)
    ids = _ids(rows, active, num_local, rng)
    out_rows, out_w, index = persistent_experts(*_args(w, x, x_sf, ids, wts))
    torch.cuda.synchronize()
    assert out_w is wts
    gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, num_local)
    n = _check_rows(out_rows, index, gemm2, oracle_index, ids, num_local)
    print(f"cell rows={rows} window={num_local} active={active}: {n} local slots bitwise")
    ref_rows, _, ref_index = persistent_experts.reference(*_args(w, x, x_sf, ids, wts))
    assert torch.equal(ref_index, index)
    if n:
        elem, rms = _band(out_rows, ref_rows, index)
        oracle_rows = torch.zeros_like(out_rows)
        oi = oracle_index.reshape(-1)
        oracle_rows[oi >= 0] = gemm2[oi[oi >= 0].long(), :HIDDEN]
        o_elem, o_rms = _band(oracle_rows, ref_rows, index)
        print(
            f"  vs FP64 reference: op {elem:.2f} ulp element / {rms:.3f} ulp RMS; "
            f"TRTLLM-gen {o_elem:.2f} / {o_rms:.3f}"
        )
        assert elem <= ELEMENT_ULP and rms <= RMS_ULP
        assert rms <= 1.10 * o_rms + 1e-12


def test_randomized_arbitrary_scalars_bitwise(windows):
    """``RANDOM_CALLS`` calls, every one bitwise: rows 1..8 in turn, all windows,
    any number of active experts, per-expert scalars redrawn per call (every
    16th call over 2^-12..2^4), half the calls with ``-1`` non-local ids."""
    rng = random.Random(_seed("random"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("random-x"))
    failures, slots, rows_seen, windows_seen = [], 0, set(), set()
    for i in range(RANDOM_CALLS):
        m = 1 + i % 8
        num_local = {3: 32, 1: 96}.get((i // 8) % 4, 64)
        active = rng.randint(1, min(num_local, TOP_K * m))
        lo, hi = (-12, 4) if i % 16 == 15 else (-7, -5)
        w = dict(windows[num_local])
        for name in ("scale_c", "alpha1", "alpha2"):
            w[name] = _scalars(num_local, gen, lo, hi)
        x, x_sf, wts = _inputs(m, gen)
        ids = _ids(m, active, num_local, rng)
        ids_op = _drop_some_nonlocal(ids, num_local, rng) if i % 2 else ids
        out_rows, _, index = persistent_experts(*_args(w, x, x_sf, ids_op, wts))
        gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, num_local)
        n, bad, map_ok = _differing(out_rows, index, gemm2, oracle_index, ids, num_local)
        slots += n
        rows_seen.add(m)
        windows_seen.add(num_local)
        if bad or not map_ok:
            failures.append(dict(call=i, rows=m, window=num_local, active=active, bad=bad))
    print(f"{RANDOM_CALLS} randomized calls, {slots} local slots: {len(failures)} not bitwise")
    assert rows_seen == set(range(1, 9))
    assert windows_seen == {32, 64, 96}
    assert not failures, f"{len(failures)} calls differ from TRTLLM-gen, first: {failures[:5]}"


#: Rows that repeat a local expert inside one token (``L`` = a local id, ``N``
#: a non-local id, ``-`` = -1): two adjacent slots first, then a repeat at the
#: far end, a triple, two pairs, and all eight slots on one expert.
REPEAT_ROWS = (
    "L0 L0 N N N N N N",
    "N L0 N N N N N L0",
    "L0 N L1 L0 N L0 L1 -",
    "L0 L1 L1 L0 L2 L3 L2 N",
    "L0 L0 L0 L0 L0 L0 L0 L0",
)


def _repeat_row(pattern: str, num_local: int, rng: random.Random):
    local = rng.sample(range(num_local), 4)
    row = []
    for tok in pattern.split():
        if tok == "-":
            row.append(-1)
        elif tok == "N":
            row.append(rng.randrange(num_local, EXPERTS))
        else:
            row.append(local[int(tok[1:])])
    return row


def test_repeated_local_ids_route_once(windows):
    """A token that names one local expert in several slots: the first slot gets
    the row and the later ones are ``-1``, as TRTLLM-gen lays them out, and the
    rows a finalize would select are bitwise TRTLLM-gen's. Rows the map drops
    are filled with NaN before each call, so a finalize reading one would show.

    The served routing never repeats an id (a top-8 over distinct experts,
    remapped by ``ring_buddy_assign`` to distinct window ids); this pins the
    op as total, so a later producer of ids cannot hand the finalize an
    unwritten row.
    """
    rng = random.Random(_seed("repeat"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("repeat-x"))
    calls = slots = dropped = 0
    for num_local in (64, 32):
        w = windows[num_local]
        for m in (1, 2, 3, 8):
            for trial in range(6):
                rows_ids = []
                for t in range(m):
                    if trial < len(REPEAT_ROWS):
                        row = _repeat_row(
                            REPEAT_ROWS[(trial + t) % len(REPEAT_ROWS)], num_local, rng
                        )
                    else:  # random slots drawn mostly from a pool of 1..3 local ids
                        pool = rng.sample(range(num_local), rng.randint(1, 3))
                        row = [
                            rng.choice(pool) if rng.random() < 0.7 else rng.randrange(-1, EXPERTS)
                            for _ in range(TOP_K)
                        ]
                    rows_ids.append(row)
                ids = torch.tensor(rows_ids, dtype=torch.int32, device="cuda")
                x, x_sf, wts = _inputs(m, gen)
                poison = torch.full(
                    (m * TOP_K, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                del poison  # the op's ``rows`` reuses this block
                out_rows, _, index = persistent_experts(*_args(w, x, x_sf, ids, wts))
                gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, num_local)
                slots += _check_rows(out_rows, index, gemm2, oracle_index, ids, num_local)
                selected, valid = expert_rows.gather_routed_rows(out_rows, index, m, HIDDEN)
                assert not torch.isnan(selected).any(), "a selected row was never written"
                _, _, ref_index = persistent_experts.reference(*_args(w, x, x_sf, ids, wts))
                assert torch.equal(ref_index, index)
                local_any = (ids >= 0) & (ids < num_local)
                dropped += int((local_any & ~valid).sum())
                calls += 1
    print(f"{calls} calls with repeated local ids: {slots} routed slots bitwise, {dropped} dropped")
    assert dropped > 0


def test_repeated_launches_and_graph_replay(windows):
    """Epochs across many launches, layer-to-layer reuse of one workspace, and
    a captured graph replayed with new ids and activations in place."""
    w = windows[64]
    rng = random.Random(_seed("graph"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("graph-x"))
    x, x_sf, wts = _inputs(8, gen)
    ids = _ids(8, 5, 64, rng)
    for _ in range(5):  # eager launches advance every CTA's epoch
        persistent_experts(*_args(w, x, x_sf, ids, wts))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        persistent_experts(*_args(w, x, x_sf, ids, wts))  # warm the allocator on this stream
        with torch.cuda.graph(graph, stream=stream):
            outs = [persistent_experts(*_args(w, x, x_sf, ids, wts)) for _ in range(3)]
    torch.cuda.current_stream().wait_stream(stream)
    for trial in range(4):
        active = [4, 6, 1, 8][trial]
        x.copy_(_inputs(8, gen)[0])
        ids.copy_(_ids(8, active, 64, rng))
        graph.replay()
        torch.cuda.synchronize()
        gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, 64)
        for out_rows, _, index in outs:
            _check_rows(out_rows, index, gemm2, oracle_index, ids, 64)


def test_delay_sites_bitwise(windows, monkeypatch):
    """A re-drawn third of the CTAs late at entry, before publishing FC1 tiles
    and before fetching FC2 operands (``OPTIONS["delay"] = 7``), eager
    and in a replayed graph: still bitwise."""
    monkeypatch.setitem(pe_mod.OPTIONS, "delay", 7)
    monkeypatch.setitem(pe_mod.OPTIONS, "delay_ns", 1000)
    rng = random.Random(_seed("delay"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("delay-x"))
    w = windows[64]
    for i in range(24):
        m = 1 + i % 8
        x, x_sf, wts = _inputs(m, gen)
        ids = _ids(m, rng.randint(1, min(64, TOP_K * m)), 64, rng)
        out_rows, _, index = persistent_experts(*_args(w, x, x_sf, ids, wts))
        gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, 64)
        _, bad, map_ok = _differing(out_rows, index, gemm2, oracle_index, ids, 64)
        assert map_ok and bad == 0, f"delayed call {i} (rows {m}): {bad} elements differ"
    x, x_sf, wts = _inputs(8, gen)
    ids = _ids(8, 5, 64, rng)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        persistent_experts(*_args(w, x, x_sf, ids, wts))
        with torch.cuda.graph(graph, stream=stream):
            outs = [persistent_experts(*_args(w, x, x_sf, ids, wts)) for _ in range(3)]
    torch.cuda.current_stream().wait_stream(stream)
    for active in (3, 8, 5, 12):
        x.copy_(_inputs(8, gen)[0])
        ids.copy_(_ids(8, active, 64, rng))
        graph.replay()
        torch.cuda.synchronize()
        gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, 64)
        for out_rows, _, index in outs:
            _, bad, map_ok = _differing(out_rows, index, gemm2, oracle_index, ids, 64)
            assert map_ok and bad == 0, f"delayed graph replay, {active} active: {bad} differ"


def _workspace(w):
    device = torch.device("cuda")
    persistent_experts.prepare(w["w1"], w["s1"], w["w2"], w["s2"])
    num_ctas, options = pe_mod._config(device)
    return pe_mod._workspace(device, num_ctas, options["handoff"]), options["handoff"]


def _epoch(ws) -> int:
    epochs = ws.epoch[: ws.num_ctas]
    assert bool((epochs == epochs[0]).all()), "every CTA advances its counter once per launch"
    return int(epochs[0].item()) & 0xFFFFFFFF


def test_epoch_wrap_skips_zero(windows):
    """From 0xFFFFFFFE with never-written (zero-tagged) hand-off words: the
    launches run at 0xFFFFFFFF, 1, 2, ... and every one is bitwise."""
    w = windows[64]
    ws, _ = _workspace(w)
    rng = random.Random(_seed("wrap"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("wrap-x"))
    ws.images.zero_()
    ws.flags.zero_()
    ws.epoch.fill_(-2)
    expected = [0xFFFFFFFF, 1, 2, 3, 4, 5, 6]
    for active, want in zip((1, 8, 5, 8, 3, 8, 64), expected):
        x, x_sf, wts = _inputs(8, gen)
        ids = _ids(8, active, 64, rng)
        out_rows, _, index = persistent_experts(*_args(w, x, x_sf, ids, wts))
        gemm2, oracle_index = _oracle(w, x, x_sf, ids, wts, 64)
        _, bad, map_ok = _differing(out_rows, index, gemm2, oracle_index, ids, 64)
        assert _epoch(ws) == want
        assert map_ok and bad == 0, f"launch at epoch {want:#x} ({active} active): {bad} differ"


def test_rotation_bounds_tag_age(windows):
    """70 launches that only ever use slot 0: every other slot's tags are
    still rewritten, so none is older than 64 launches."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe._persistent_experts import (  # noqa: E501
        kernel as K,
    )

    w = windows[64]
    ws, handoff = _workspace(w)
    rng = random.Random(_seed("rotation"))
    gen = torch.Generator(device="cuda").manual_seed(_seed("rotation-x"))
    ws.images.zero_()
    ws.flags.zero_()
    ws.epoch.fill_(1000)
    x, x_sf, wts = _inputs(8, gen)
    ids = _ids(8, 1, 64, rng)
    for _ in range(70):
        persistent_experts(*_args(w, x, x_sf, ids, wts))
    torch.cuda.synchronize()
    now = _epoch(ws)
    assert now == 1070
    if handoff:
        words = ws.images.view(torch.int64)[: K.MAX_SLOTS * K.TAG_SLOT_BYTES // 8]
        tags = (words >> 32) & 0xFFFFFFFF
    else:
        tags = ws.flags.to(torch.int64) & 0xFFFFFFFF
    assert int(tags.min()) >= now - 63 and int(tags.max()) == now, (
        f"tags span {int(tags.min())}..{int(tags.max())} at epoch {now}"
    )


def test_capture_refuses_lazy_setup(windows, monkeypatch):
    """A configuration first seen under graph capture is refused: compiling or
    zero-filling a workspace inside a graph would replay as an epoch reset."""
    w = windows[64]
    gen = torch.Generator(device="cuda").manual_seed(_seed("capture-x"))
    x, x_sf, wts = _inputs(8, gen)
    ids = _ids(8, 5, 64, random.Random(_seed("capture")))
    monkeypatch.setitem(pe_mod.OPTIONS, "delay_ns", 1234)  # never compiled in this process
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before CUDA-graph capture"):
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            persistent_experts(*_args(w, x, x_sf, ids, wts))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
