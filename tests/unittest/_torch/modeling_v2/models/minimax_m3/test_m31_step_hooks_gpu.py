# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 family step hooks (``minimax_m3.ops.step_hooks``) match the engine / metadata torch lines they replace bit
for bit (with attention-DP dummy rows, eagerly and in a replayed CUDA graph), and decline other layouts without
writing."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import step_hooks
from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the step-hook ops run on SM107", allow_module_level=True)


@pytest.fixture(autouse=True)
def _ops_built():
    assert step_hooks.ops_available(), "this build lacks the step-hook ops"


K = 7
T = K + 1
D = T - 1
GENS = (1, 16)
DUMMIES = (0, 3)
SLOTS = 64
CTX_TOKENS = 5


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _ri(lo, hi, shape, g):
    return torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=torch.int32)


def _gather_case(gens, dummies, fresh, seed):
    """One step's operands: ``fresh`` generation rows without a previous batch come first, then the ``gens`` rows of
    the previous batch, then ``dummies`` attention-DP dummy rows; every engine buffer starts with junk."""
    g = _gen(seed)
    rows = fresh + gens + dummies
    src = dict(
        new_tok=_ri(0, 200_000, (T, SLOTS, 1), g),
        next_draft=_ri(0, 200_000, (SLOTS, K), g),
        lens=_ri(1, T + 1, (SLOTS,), g),
    )
    slots = torch.randperm(SLOTS, generator=g, device="cuda")[:gens].to(torch.int32)
    ppi = torch.zeros(SLOTS * T, dtype=torch.int32, device="cuda")
    ppi[: gens * T] = slots.repeat_interleave(
        T
    )  # previous_pos_indices: each token reads its row's slot
    cap = (rows + 4) * T + CTX_TOKENS
    dst = dict(
        ids=_ri(-9, -1, (cap,), g),
        draft=_ri(-9, -1, (cap,), g),
        pos_off=_ri(-9, -1, (cap,), g),
        kv_off=_ri(-9, -1, (SLOTS,), g),
    )
    geo = dict(P=gens, ids0=CTX_TOKENS + fresh * T, d0=fresh * D, b0=fresh)
    return src, slots, ppi, dst, geo


def _engine_gather(src, slots, ppi, dst, geo):
    """decoder runner.py, a step whose fold declined: zero the offsets, then the segment copies."""
    P, ids0, d0, b0 = geo["P"], geo["ids0"], geo["d0"], geo["b0"]
    dst["pos_off"] *= 0
    dst["kv_off"] *= 0
    dst["ids"][ids0 : ids0 + P * T].copy_(src["new_tok"].transpose(0, 1)[slots, :T].flatten())
    dst["draft"][d0 : d0 + P * D].copy_(src["next_draft"][slots, :D].flatten())
    kv_len_offsets = src["lens"] - T
    dst["pos_off"][b0 * T : (b0 + P) * T].copy_(src["lens"][ppi[0 : P * T]])
    dst["kv_off"][b0 : b0 + P].copy_(kv_len_offsets[slots])


def _fold_gather(fold, src, slots, ppi, geo):
    g = geo
    return fold.gather(src["new_tok"], src["next_draft"], src["lens"], slots, ppi, g["P"], T, D, g["ids0"],
                       g["d0"], g["b0"])  # fmt: skip


def _bound_fold(dst):
    fold = step_hooks.M3DSparkStepFold()
    fold.bind_engine_buffers(dst["ids"], dst["draft"], dst["pos_off"], dst["kv_off"], SLOTS)
    return fold


def _clone(d):
    return {k: v.clone() for k, v in d.items()}


def _assert_equal(got, ref, what):
    for name in ref:
        assert torch.equal(got[name], ref[name]), f"{what}: {name}"


@pytest.mark.parametrize("fresh", [0, 2])
@pytest.mark.parametrize("dummies", DUMMIES)
@pytest.mark.parametrize("gens", GENS)
def test_step_gather_writes_the_engine_gathers(gens, dummies, fresh):
    for trial in range(3):
        src, slots, ppi, dst, geo = _gather_case(
            gens, dummies, fresh, 1000 * gens + 10 * trial + fresh
        )
        ref = _clone(dst)
        _engine_gather(src, slots, ppi, ref, geo)
        fold = _bound_fold(dst)
        assert fold.begin_step([int(s) for s in slots.tolist()], gens + dummies) is False
        assert _fold_gather(fold, src, slots, ppi, geo) is True
        torch.cuda.synchronize()
        _assert_equal(dst, ref, f"gens {gens} dummies {dummies} fresh {fresh} trial {trial}")
        assert (fold.taken, fold.declined) == (1, 0)


@pytest.mark.parametrize("gens", GENS)
def test_step_gather_in_a_replayed_graph(gens):
    src, slots, ppi, dst, geo = _gather_case(gens, 3, 1, 7 * gens)
    fold = _bound_fold(dst)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert _fold_gather(fold, src, slots, ppi, geo) is True
    for step in range(4):
        nxt, nslots, nppi, _, _ = _gather_case(gens, 3, 1, 7 * gens + 100 + step)
        for name in src:
            src[name].copy_(nxt[name])
        slots.copy_(nslots)
        ppi.copy_(nppi)
        for name in dst:
            dst[name].fill_(-3 - step)
        ref = _clone(dst)
        _engine_gather(src, slots, ppi, ref, geo)
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(dst, ref, f"replay {step}")


def test_the_fold_declines_what_step_gather_does_not_cover_and_writes_nothing():
    src, slots, ppi, dst, geo = _gather_case(11, 3, 0, 5)
    before = _clone(dst)
    fold = _bound_fold(dst)
    assert _fold_gather(fold, src, slots.long(), ppi, geo) is False  # int64 slots
    bad = dict(src, next_draft=src["next_draft"].long())
    assert _fold_gather(fold, bad, slots, ppi, geo) is False
    wide = dict(src, new_tok=torch.zeros((T, SLOTS, 2), dtype=torch.int32, device="cuda"))
    assert _fold_gather(fold, wide, slots, ppi, geo) is False  # beam width 2
    torch.cuda.synchronize()
    _assert_equal(dst, before, "declined")
    assert (fold.taken, fold.declined) == (0, 3)
    unbound = step_hooks.M3DSparkStepFold()
    assert _fold_gather(unbound, src, slots, ppi, geo) is False


def _offsets_case(gens, dummies, ctx, seed):
    """The decoder runner's slices: position_ids [1, N] past the context tokens, kv_lens of the generation rows, and
    the previous step's offsets (0 for fresh and dummy rows, the accepted-length deltas for the previous batch)."""
    g = _gen(seed)
    rows = gens + dummies
    position_ids = _ri(0, 1 << 20, (1, ctx + rows * T), g)
    kv_lens = _ri(0, 1 << 20, (SLOTS,), g)
    pos_off = torch.zeros(SLOTS * T, dtype=torch.int32, device="cuda")
    kv_off = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    pos_off[: gens * T] = _ri(1, T + 1, (gens * T,), g)
    kv_off[:gens] = _ri(-K, 1, (gens,), g)
    num_ctx_requests = 1 if ctx else 0
    views = (
        lambda: position_ids[0, ctx:],
        lambda: pos_off[: rows * T],
        lambda: kv_lens[num_ctx_requests : num_ctx_requests + rows],
        lambda: kv_off[:rows],
    )
    return position_ids, kv_lens, pos_off, kv_off, views


@pytest.mark.parametrize("ctx", [0, 9])
@pytest.mark.parametrize("dummies", DUMMIES)
@pytest.mark.parametrize("gens", GENS)
def test_spec_offsets_is_the_two_aten_adds(gens, dummies, ctx):
    position_ids, kv_lens, pos_off, kv_off, views = _offsets_case(gens, dummies, ctx, gens + ctx)
    ref_pos, ref_kv = position_ids.clone(), kv_lens.clone()
    pos, po, kv, ko = (v() for v in views)
    ref_pos[0, ctx:] += po
    ref_kv[(1 if ctx else 0) : (1 if ctx else 0) + gens + dummies] += ko
    assert step_hooks.spec_offsets(pos, po, kv, ko) is True
    torch.cuda.synchronize()
    assert torch.equal(position_ids, ref_pos) and torch.equal(kv_lens, ref_kv)


def test_spec_offsets_declines_other_layouts_and_writes_nothing():
    pos = torch.arange(16, dtype=torch.int32, device="cuda")
    kv = torch.arange(4, dtype=torch.int32, device="cuda")
    one = torch.ones(16, dtype=torch.int32, device="cuda")
    before = (pos.clone(), kv.clone())
    assert step_hooks.spec_offsets(pos.long(), one.long(), kv, kv) is False
    assert step_hooks.spec_offsets(pos[::2], one[::2], kv, kv) is False
    assert step_hooks.spec_offsets(pos, one[:8], kv, kv) is False
    assert step_hooks.spec_offsets(pos.view(4, 4), one.view(4, 4), kv, kv) is False
    torch.cuda.synchronize()
    assert torch.equal(pos, before[0]) and torch.equal(kv, before[1])


def _msa_torch(kv_lens, staged, seq_lens, qbr_buf, intra, table, page, batch, total_q, out):
    """msa_backend.MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens past its hook call, line for line."""
    kv_true = torch.minimum(kv_lens[:batch], staged[:batch])
    out["msa_seq_lens"][:batch].copy_(kv_true)
    if out.get("cu_kv") is not None:
        out["cu_kv"][0].zero_()
        torch.cumsum(kv_true, 0, out=out["cu_kv"][1 : batch + 1])
    qbr = qbr_buf[:total_q].to(torch.long)
    qo_dev = seq_lens[:batch]
    pos = kv_true[qbr] - qo_dev[qbr] + intra[:total_q]
    pos_slot = pos.clamp_min(0)
    block_col = torch.div(pos_slot, page, rounding_mode="floor").clamp(max=int(table.shape[1]) - 1)
    slots = table[qbr, block_col.to(torch.long)] * page + torch.remainder(pos_slot, page)
    out["out_cache_loc"][:total_q].copy_(slots)
    n_valid = torch.div((pos + 1).clamp_min(0) + (page - 1), page, rounding_mode="floor")
    if out.get("n_valid") is not None:
        out["n_valid"][:total_q].copy_(n_valid.to(torch.int32))


def _msa_case(gens, dummies, page, seed, cu_kv=True, n_valid=True, dummy_kv=0):
    """K7 verify rows (8 query tokens each, partial acceptance below the staged length, some crossing a page edge)
    then attention-DP dummy rows (kv length ``dummy_kv``); the block table is a row-strided view."""
    g = _gen(seed)
    batch = gens + dummies
    total_q = batch * T
    cols = 24
    staged = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    staged[:gens] = _ri(T, page * (cols - 2), (gens,), g)
    staged[: min(gens, 2)] = page * 3 + T // 2  # the verify window straddles a page edge
    kv_lens = staged.clone()
    kv_lens[:gens] -= _ri(0, T, (gens,), g)  # rejected draft tokens shorten kv_lens
    staged[gens:batch] = dummy_kv
    kv_lens[gens:batch] = dummy_kv
    seq_lens = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    seq_lens[:batch] = T
    qbr = torch.zeros(SLOTS * T, dtype=torch.int32, device="cuda")
    intra = torch.zeros(SLOTS * T, dtype=torch.int32, device="cuda")
    qbr[:total_q] = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(T)
    intra[:total_q] = torch.arange(T, device="cuda", dtype=torch.int32).repeat(batch)
    table = _ri(0, 1 << 16, (SLOTS, cols + 8), g)[:, :cols]
    out = dict(
        msa_seq_lens=_ri(-9, -1, (SLOTS,), g),
        cu_kv=_ri(-9, -1, (SLOTS + 1,), g) if cu_kv else None,
        out_cache_loc=_ri(-9, -1, (SLOTS * T,), g),
        n_valid=_ri(-9, -1, (SLOTS * T,), g) if n_valid else None,
    )
    return (kv_lens, staged, seq_lens, qbr, intra, table, page, batch, total_q), out


def _msa_hook(args, out):
    batch, total_q = args[7], args[8]
    cu_kv, n_valid = out["cu_kv"], out["n_valid"]
    return step_hooks.msa_kv_update(
        *args,
        out["msa_seq_lens"],
        None if cu_kv is None else cu_kv[: batch + 1],
        out["out_cache_loc"],
        None if n_valid is None else n_valid[:total_q],
    )


def _out_clone(out):
    return {k: (None if v is None else v.clone()) for k, v in out.items()}


def _out_equal(got, ref, what):
    for name, t in ref.items():
        assert (t is None) == (got[name] is None), f"{what}: {name}"
        if t is not None:
            assert torch.equal(got[name], t), f"{what}: {name}"


@pytest.mark.parametrize(
    "gens, dummies, dummy_kv, page, cu_kv, n_valid",
    [
        (16, 3, 37, 128, True, True),
        (11, 0, 0, 64, False, True),
        (1, 3, 0, 128, True, False),
    ],
)
def test_msa_kv_update_is_the_metadata_torch_lines(gens, dummies, dummy_kv, page, cu_kv, n_valid):
    args, out = _msa_case(gens, dummies, page, gens * 31 + dummies + page, cu_kv, n_valid, dummy_kv)
    ref = _out_clone(out)
    _msa_torch(*args, ref)
    assert _msa_hook(args, out) is True
    torch.cuda.synchronize()
    _out_equal(out, ref, f"gens {gens} dummies {dummies} page {page}")


@pytest.mark.parametrize("gens", GENS)
def test_msa_kv_update_in_a_replayed_graph(gens):
    args, out = _msa_case(gens, 3, 128, 77 * gens)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert _msa_hook(args, out) is True
    for step in range(4):
        new_args, _ = _msa_case(gens, 3, 128, 77 * gens + 9 + step)
        for buf, new in zip(args[:6], new_args[:6]):
            buf.copy_(new)
        for t in out.values():
            t.fill_(-4 - step)
        ref = _out_clone(out)
        _msa_torch(*args, ref)
        graph.replay()
        torch.cuda.synchronize()
        _out_equal(out, ref, f"replay {step}")


def test_msa_kv_update_declines_other_operands_and_writes_nothing():
    args, out = _msa_case(11, 3, 128, 3)
    before = _out_clone(out)
    kv_lens, staged, seq_lens, qbr, intra, table, page, batch, total_q = args
    assert _msa_hook((kv_lens.long(), *args[1:]), out) is False
    assert _msa_hook((kv_lens, staged, seq_lens, qbr, intra, table.t(), *args[6:]), out) is False
    assert _msa_hook((kv_lens[::2], *args[1:]), out) is False
    assert _msa_hook((kv_lens[:4], *args[1:]), out) is False  # fewer lengths than requests
    torch.cuda.synchronize()
    _out_equal(out, before, "declined")


class _TritonLaunches:
    """Counts Triton kernel launches (JITFunction.run) while active."""

    def __enter__(self):
        from triton.runtime import jit

        self._jit = jit
        self._orig = jit.JITFunction.run
        self.names: list = []
        outer = self

        def run(fn, *args, **kwargs):
            outer.names.append(getattr(fn, "__name__", repr(fn)))
            return outer._orig(fn, *args, **kwargs)

        jit.JITFunction.run = run
        return self

    def __exit__(self, *exc):
        self._jit.JITFunction.run = self._orig


def _arm_pack(arm):
    """native: the family prologue op; stock: the Triton prologue; killed: no prologue op (eager native scatter)."""
    kw = dict(parallel_scatter=True, in_graph=True, early_copy=True, step_epoch=True)
    if arm != "stock":
        kw["cpp_scatter"] = True
    if arm == "native":
        kw["prologue_op"] = step_hooks.step_prologue
    return StepInputPack(torch.device("cuda", torch.cuda.current_device()), **kw)


def _step_buffers(gens):
    rows = gens + 3
    z = lambda *s: torch.zeros(*s, dtype=torch.int32, device="cuda")  # noqa: E731
    return dict(
        position_ids=z(rows * T),
        kv_lens=z(rows),
        slots=z(rows),
        table=z(rows, 64)[:, :40],  # a page table's leading columns: one descriptor per row
        draft_in=z(rows * D),
        mirror=z(rows * T),  # a device copy reading a host copy's destination
        big=z(StepInputPack.GRAPH_WINDOW // 4),
    )


def _step_values(gens, step):
    g = torch.Generator().manual_seed(1000 * gens + step)
    rows = gens + 3
    r = lambda *s: torch.randint(0, 1 << 30, s, generator=g, dtype=torch.int32)  # noqa: E731
    return dict(
        position_ids=r(rows * T),
        kv_lens=r(rows),
        slots=r(rows),
        table=r(rows, 40),
        draft_in=r(rows * D),
    )


def _run_arm(arm, gens, steps=6):
    """Capture a graph with the arm's prologue reading every destination, then run ``steps`` steps: packs in the
    window, one over it (eager), one with nothing staged; returns every replay's output and the final epoch."""
    pack = _arm_pack(arm)
    bufs = _step_buffers(gens)
    out = torch.zeros(sum(t.numel() for t in bufs.values()), dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with step_inputs.active(pack):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            has = step_inputs.capture_prologue() is pack
            out.copy_(torch.cat([bufs[k].reshape(-1) for k in bufs]))
    outs = []
    with _TritonLaunches() as launches:
        for step in range(steps):
            vals = _step_values(gens, step)
            with step_inputs.active(pack):
                if step != 4:  # step 4 replays with nothing staged
                    step_inputs.begin()
                    for name, v in vals.items():
                        step_inputs.copy_(bufs[name], v.pin_memory())
                    step_inputs.copy_(bufs["mirror"], bufs["position_ids"])
                    if step == 2:  # over the graph window: issued eagerly
                        step_inputs.copy_(
                            bufs["big"], torch.full_like(bufs["big"], step, device="cpu")
                        )
                step_inputs.flush(in_graph=True)
            graph.replay()
            torch.cuda.synchronize()
            outs.append(out.clone())
    epoch = None if pack.epoch is None else int(pack.epoch.item())
    return outs, epoch, has, launches.names, (pack.graph_windows, pack.eager_fallbacks)


@pytest.mark.parametrize("gens", GENS)
def test_the_native_prologue_writes_what_the_stock_prologue_writes(gens):
    stock, stock_epoch, stock_has, _, stock_counts = _run_arm("stock", gens)
    native, native_epoch, native_has, native_launches, native_counts = _run_arm("native", gens)
    killed, killed_epoch, killed_has, killed_launches, killed_counts = _run_arm("killed", gens)
    assert stock_has and native_has and not killed_has
    for step, (a, b, c) in enumerate(zip(stock, native, killed)):
        assert torch.equal(a, b), f"native vs stock, step {step}"
        assert torch.equal(a, c), f"killed vs stock, step {step}"
    # The stock arm (and so every arm) reads back the values staged so far, in out's key order.
    exp = {k: torch.zeros_like(v, device="cpu") for k, v in _step_buffers(gens).items()}
    for step in range(6):
        if step != 4:
            for k, v in _step_values(gens, step).items():
                exp[k] = v.reshape(exp[k].shape)
            exp["mirror"] = exp["position_ids"].clone()
            if step == 2:
                exp["big"] = torch.full_like(exp["big"], 2)
        want = torch.cat([t.reshape(-1) for t in exp.values()])
        assert torch.equal(stock[step].cpu(), want), f"stock vs staged, step {step}"
    # The epoch: once at the warm-up, then once per replay, with or without a new window.
    assert native_epoch == stock_epoch == 1 + 6
    assert killed_epoch is None
    assert native_counts == stock_counts == (4, 1)
    assert native_launches == [] and killed_launches == []
