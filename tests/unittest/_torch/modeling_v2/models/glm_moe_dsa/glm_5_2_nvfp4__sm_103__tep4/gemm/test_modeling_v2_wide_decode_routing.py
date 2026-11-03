# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which GEMM the GLM-5.2 target's call sites pick for a step's row count, by engine and phase.

MTP3 / MTP5 decode steps carry 128 / 192 rows (32 sequences x 4 / 6), above the 64 rows that
otherwise tell a decode step from a context one. The target records the phase where it routes
the step (`_select_target` sets `_wide_decode`), and only then sends rows above 64 to the MTP
decode cells. What this pins, on a core built without weights (`object.__new__`) and with every
entry replaced by a recorder:

* no speculation and MTP1 take exactly the paths they took before the MTP cells existed, at
  every row count and phase, even with MTP cells present;
* MTP3 / MTP5 context steps do too;
* MTP3 / MTP5 decode steps keep the numerics of the call's 64-row decode path above 64 rows:
  where that is bf16, the MTP cell at 128 / 192 rows, and at any other count up to 192 (a batch
  of fewer than 32 sequences: 96, 150) the next cell up on the exact rows (`_wide_cell`: a
  re-pinned projection runner, the K-B object itself, the runtime-M indexer split); with
  `_wide_bucket` off bf16 elsewhere (the prefill GEMM, or cuBLAS where a projection has no
  prefill entry); never the MXF8 context GEMM; where it is the decode MXF8 GEMM (the trunk
  o_proj), the wide decode MXF8 GEMM at every row count up to 192 when it is built (bitwise
  the context MXF8 GEMM's output), else the context MXF8 GEMM.

Dispatch only: no kernel runs, so this needs no GPU beyond what importing the target needs.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the GLM-5.2 target imports sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as M,
)

BF16 = torch.bfloat16
#: (max_draft_len, label): no speculation, MTP1, MTP3, MTP5.
ENGINES = (0, 1, 3, 5)
#: The decode-cell edge, the two MTP cells, and a wide decode count below each without a cell.
ROWS = (64, 96, 128, 150, 192)
MTP_ROWS = (128, 192)


def _aligned(rows: int, cols: int, dtype=BF16) -> torch.Tensor:
    """A contiguous `[rows, cols]` tensor at a 128-byte aligned address (what the in-place
    paths check), on the CPU: nothing reads it."""
    item = torch.tensor([], dtype=dtype).element_size()
    flat = torch.zeros(rows * cols + 128 // item, dtype=dtype)
    skip = (-flat.data_ptr() % 128) // item
    return flat[skip : skip + rows * cols].view(rows, cols)


def _runner(tag: str, n: int = 8, out_dtype=BF16, **kw) -> SimpleNamespace:
    return SimpleNamespace(tag=tag, n=n, out_dtype=out_dtype, in_dtype=BF16, max_rows=8192, **kw)


@pytest.fixture
def core_for(monkeypatch):
    """`core_for(max_draft_len)`: a core whose runner tables mirror `post_load_weights` (decode
    cells at 64 rows, prefill entries, MTP cells at 128 / 192 -- present for every engine, so a
    path an engine must not take is there to be wrongly taken), and `calls`, the tags of the
    runners the recorders saw, in order."""
    calls: list[str] = []

    def record(tag_of):
        def entry(*args, **kwargs):
            calls.append(tag_of(*args, **kwargs))
            return args[2] if len(args) > 2 and isinstance(args[2], torch.Tensor) else None

        return entry

    monkeypatch.setattr(M, "cute_dsl_projection", record(lambda x, w, out, r: r.tag))
    # A cell re-pinned to fewer rows (`_projection_op.at_rows`): the cell's tag plus the rows.
    monkeypatch.setattr(M._projection_op, "at_rows", lambda r, rows: _runner(f"{r.tag}@{rows}"))
    monkeypatch.setattr(M, "cute_dsl_projection_mxf8", record(lambda *a, **k: "mxf8"))
    monkeypatch.setattr(
        M, "cute_dsl_projection_mxf8_decode", record(lambda x, w, out, r, pre=False: r.tag)
    )
    monkeypatch.setattr(M, "cublas_mm", lambda x, w, b, d: calls.append("cublas") or x)
    monkeypatch.setattr(M, "cute_dsl_absorbed_bmm", record(lambda a, w, out, r: r.tag))
    monkeypatch.setattr(M, "bmm_out", record(lambda *a: "bmm_out"))
    monkeypatch.setattr(M, "cute_dsl_indexer_pre_proj_split", record(lambda x, w, kb, wq, h: h.tag))
    monkeypatch.setattr(M, "cute_dsl_shared_gate_up_swiglu", record(lambda x, w, o, r: r.tag))
    monkeypatch.setattr(M, "cute_dsl_mtp_norm_shard", record(lambda *a: "norm_shard"))
    monkeypatch.setattr(
        M, "cute_dsl_mtp_eh_proj", lambda local, w, r: calls.append("ksplit" if r is None else r)
    )
    # The K-split re-pinned to fewer rows (`_mtp_eh.at_rows`): tagged with the rows.
    monkeypatch.setattr(M._mtp_eh, "at_rows", lambda r, rows: f"ksplit@{rows}")
    monkeypatch.setattr(M, "cute_dsl_mtp_eh_norm_cat_mxf8", record(lambda *a: "eh_norm_cat8"))
    monkeypatch.setattr(M._eh8_op, "supports", lambda e, h: True)
    monkeypatch.setattr(M._eh8_op, "operand", lambda rows, dev: (None, None))
    # A new contiguous tensor, as the real SwiGLU returns (the down GEMM's paths check it).
    monkeypatch.setattr(
        M, "flashinfer_silu_and_mul", lambda t: t[:, : t.shape[1] // 2].contiguous()
    )

    def build(max_draft_len: int):
        core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
        core.max_draft_len = max_draft_len
        core._wide_decode = False
        core._wide_bucket, core._wide_rag = True, {}
        core.route_hits, core.route_misses = {}, {}
        core._decode, core._prefill = "decode", "prefill"
        core.tp_rank, core.eps = 0, 1e-5
        core.idx_heads, core.idx_dim = 32, 128
        core._rows = SimpleNamespace(swiglu=None, norm=None, copy=None, add=None)
        core._reduce = lambda t: t
        core._rmsnorm = lambda x, w: x
        core._copy_into = lambda dst, src: dst
        # The trunk layers' decode MXF8 GEMMs (taken with a decode weight `wd8`, up to 64 rows),
        # and the trunk o_proj's wide one (MTP3 / MTP5 decode rows 65..192; present for every
        # engine, as a path an engine must not take).
        core._dec8 = {"o_proj": _runner("dec_mxf8"), "q_b": _runner("dec_mxf8")}
        core._dec8w = {"o_proj": _runner("dec_mxf8w")}
        names = ("qkva", "q_b", "o_proj", "ix_qb", "dense_gate_up", "shared_gate_up", "shared_down")
        core._proj = {name: {64: _runner(f"{name}@64")} for name in names}
        core._proj_mtp = {name: {m: _runner(f"{name}@{m}:mtp") for m in MTP_ROWS} for name in names}
        core._proj_pre = {
            name: _runner(f"{name}:prefill") for name in names + ("mtp_eh",) if name != "ix_qb"
        }
        core._proj_mxf8 = {name: _runner("mxf8") for name in ("qkva", "o_proj", "dense_gate_up")}
        core._proj_mxf8["mtp_eh"] = _runner("mxf8")
        core._absorb = {"k_b": {64: _runner("k_b@64")}}
        core._absorb_mtp = {"k_b": {m: _runner(f"k_b@{m}:mtp") for m in MTP_ROWS}}
        core._absorb_pre = {"k_b": _runner("k_b:prefill")}
        # The prefill handle's split launch (`TRTLLM_GLM52_CTX_IX_NOSPLIT=0`): with the
        # no-split read on, its rows read the projection output in place instead. The indexer
        # front picks its q-side launcher before it knows whether the q side runs.
        core._ix_nosplit = False
        core._ix_qside_compiled = None
        core._ix_pre_split = {64: _runner("ix_split@64")}
        core._ix_pre_split_mtp = {m: _runner(f"ix_split@{m}:mtp") for m in MTP_ROWS}
        core._ix_pre_split_pre = _runner("ix_split:prefill")
        core._ix_pre_split_rag = _runner("ix_split:rag", max_m=192)
        core._moe_k = SimpleNamespace(shared_down_pre=_runner("shared_down:shared_pre"))
        core._moe_k.shared_up_pre = _runner("shared_up:prefill")
        core._moe_mid = _aligned(256, 512)
        core._shared_gu_swiglu = {64: _runner("shared_gu@64")}
        core._shared_gu_swiglu_mtp = {m: _runner(f"shared_gu@{m}:mtp") for m in MTP_ROWS}
        core._mtp = dict(enorm=None, hnorm=None, eh_local=None, ehproj=None, norm_shard=None)
        core._mtp["eh"] = torch.zeros(8, 4, dtype=BF16)
        core._mtp_eh8 = SimpleNamespace(norm=None, w8=None)
        return core, calls

    return build


def _step(core, decode: bool) -> None:
    """Route a step the way the engine's forward does: the phase is set at the dispatcher."""
    target = core._select_target(SimpleNamespace(num_contexts=0 if decode else 2))
    assert target == ("decode" if decode else "prefill")


def _wide(max_draft_len: int, decode: bool, rows: int) -> bool:
    return decode and max_draft_len > 1 and rows > 64


def _cell(max_draft_len: int, decode: bool, rows: int, bucket: bool) -> int | None:
    """The MTP cell a step's rows take: their own (128 / 192), else with buckets on the next one
    up (launched on the exact rows); None off a wide decode step or with no cell."""
    if not _wide(max_draft_len, decode, rows):
        return None
    if rows in MTP_ROWS:
        return rows
    return min((m for m in MTP_ROWS if m > rows), default=None) if bucket else None


@pytest.mark.parametrize("wide8", [True, False], ids=["wide_dec8", "no_wide_dec8"])
@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_projections(core_for, max_draft_len, decode, rows, bucket, wide8) -> None:
    """qkva (an MXF8 context path, a bf16 decode cell), the trunk o_proj (an MXF8 context path,
    the decode MXF8 GEMM; with `wide8` its wide decode GEMM built), the MTP layer's o_proj and
    q_b (prefill GEMM above 64) and ix_qb (cuBLAS above 64)."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    if not wide8:
        core._dec8w = {}
    _step(core, decode)
    wide = _wide(max_draft_len, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    x = _aligned(rows, 16)
    w = torch.zeros(4, 16, dtype=BF16).t()
    for name, w8, wd8, context in (
        ("qkva", object(), None, "mxf8"),
        ("o_proj", object(), object(), "mxf8"),
        ("o_proj", None, None, "o_proj:prefill"),
        ("q_b", None, object(), "q_b:prefill"),
        ("q_b", None, None, "q_b:prefill"),
        ("ix_qb", None, None, "cublas"),
    ):
        calls.clear()
        core._project(name, x, w, w8=w8, wd8=wd8)
        if rows <= 64:
            want = "dec_mxf8" if wd8 is not None else f"{name}@64"
        elif wd8 is not None and w8 is not None:
            # The trunk o_proj keeps its decode path's MXF8 numerics: on a wide decode step the
            # wide decode MXF8 GEMM where built (bucket or not: it takes any row count up to
            # 192), else, and on a context step, the MXF8 GEMM. (The trunk q_b has no MXF8 GEMM
            # above 64 rows: bf16, like every other call.)
            want = "dec_mxf8w" if wide and wide8 else "mxf8"
        elif cell == rows:
            want = f"{name}@{rows}:mtp"
        elif cell is not None:
            want = f"{name}@{cell}:mtp@{rows}"
        elif wide:
            want = "cublas" if name == "ix_qb" else f"{name}:prefill"
        else:
            want = context
        assert calls == [want], (name, max_draft_len, decode, rows, calls)


@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_dense_mlp_and_bmm_and_indexer_split(core_for, max_draft_len, decode, rows, bucket) -> None:
    """The dense MLP's MXF8 gate, the absorbed K-B and the indexer's pre-projection split."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    _step(core, decode)
    cell = _cell(max_draft_len, decode, rows, bucket)
    wide = _wide(max_draft_len, decode, rows)
    x = _aligned(rows, 16)
    assert core._mxf8_takes("dense_gate_up", x, object()) == (rows > 64 and not wide)

    calls.clear()
    a = torch.zeros(16, rows, 8, dtype=BF16)
    core._absorbed_bmm("k_b", a, None, None, a)
    # The K-B object of the 128 / 192 cells launches at any row count above 64 itself.
    want = "k_b@64" if rows <= 64 else f"k_b@{cell}:mtp" if cell else "k_b:prefill"
    assert calls == [want], calls

    calls.clear()
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None)
    core._indexer_project(x, None, None, ix, rows, need_q=False)
    if rows <= 64:
        want = "ix_split@64"
    elif cell == rows:
        want = f"ix_split@{rows}:mtp"
    else:
        want = "ix_split:rag" if cell else "ix_split:prefill"
    assert calls == [want], calls


@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("trunk", [True, False], ids=["trunk", "mtp_layer"])
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_shared_expert(core_for, max_draft_len, decode, trunk, rows, bucket) -> None:
    """Gate|up + SwiGLU, then down, for a trunk layer and the MTP layer."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    _step(core, decode)
    cell = _cell(max_draft_len, decode, rows, bucket)
    x = _aligned(rows, 16)
    gu = torch.zeros(16, 1024, dtype=BF16)
    dn = torch.zeros(512, 8, dtype=BF16)
    core._shared_mlp(x, gu, None, dn, trunk=trunk)
    if rows <= 64:
        want = ["shared_gu@64", "shared_down@64"]
    elif cell == rows:
        want = [f"shared_gu@{rows}:mtp", f"shared_down@{rows}:mtp"]
    elif cell is not None:
        # The gate|up cell's split-K kernel takes any row count; the down cell is re-pinned.
        want = [f"shared_gu@{cell}:mtp", f"shared_down@{cell}:mtp@{rows}"]
    elif trunk:
        want = ["shared_up:prefill", "shared_down:shared_pre"]
    else:
        # The MTP layer's two-launch path: the prefill GEMM gate|up, SwiGLU, then down.
        want = ["shared_gate_up:prefill", "shared_down:prefill"]
    assert calls == want, calls


@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", (64, 96, 128, 150, 192))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_mtp_front_eh_proj(core_for, max_draft_len, decode, rows, bucket) -> None:
    """The MTP front's eh_proj: the bf16 K-split at its row counts (192 on an MTP3 / MTP5
    decode step only), and at such a step's other row counts above 64 on the next one up
    (128 / 192) at the exact rows; with buckets off bf16 there (the replicated prefill
    GEMM); the MXF8 norm + GEMM at other context row counts above 64."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    _step(core, decode)
    wide = _wide(max_draft_len, decode, rows)
    e = torch.zeros(rows, 16, dtype=BF16)
    core._mtp_front(e, e, rows)
    if rows in (64, 128) or (rows == 192 and wide):
        want = ["norm_shard", "ksplit"]
    elif wide and bucket:
        want = ["norm_shard", f"ksplit@{rows}"]
    elif wide:
        want = ["mtp_eh:prefill"]
    else:
        want = ["eh_norm_cat8", "mxf8"]
    assert calls == want, calls


@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", (64, 96, 150, 192, 200))
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_logits_rows(core_for, monkeypatch, max_draft_len, rows, bucket) -> None:
    """The lm_head's local GEMM: a row count with its own instance runs it; MTP3 / MTP5's
    other counts below 256 run the next instance up (128 / 192 / 256) at the exact rows
    (no zero rows, no copy, the gather at the exact rows), or with buckets off on zero
    rows past them, sliced after the gather. MTP1 and no speculation keep the stock head
    there."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    lm = SimpleNamespace(weight=None)
    core._lm_head, core.tp_size = lm, 4
    built = (64, 128, 256) + ((192,) if max_draft_len > 1 else ())
    core._lm_head_runners = {m: _runner(f"lm@{m}", n=4) for m in built}
    core._lamport = SimpleNamespace(max_rows=8192, hidden=6144, gather=object())

    def gather(local, out, ws):
        calls.append(f"gather {local.shape[0]}")
        return out

    monkeypatch.setattr(M, "cute_dsl_allgather_widen", gather)
    stock = SimpleNamespace(forward=lambda *a: calls.append("stock") or torch.zeros(rows, 1))
    out = core.logits(_aligned(rows, 16), lm, stock, None)
    assert out.shape[0] == rows
    if rows in built:
        want = [f"lm@{rows}", f"gather {rows}"]
    elif max_draft_len <= 1:
        want = ["stock"]
    else:
        up = min(m for m in built if m > rows)
        want = [f"lm@{up}@{rows}", f"gather {rows}"] if bucket else [f"lm@{up}", f"gather {up}"]
    assert calls == want, calls


def _moe_close_core(max_draft_len: int):
    """A core whose MoE-layer close (`_mlp_and_close` on a trunk MoE layer, `_mtp_close`) runs
    on recorders: every call is logged with the stream it was issued on ("side" / "main")."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.max_draft_len = max_draft_len
    core.route_hits, core.route_misses = {}, {}
    core.num_layers = 1
    core._side_stream = torch.cuda.Stream()
    log = []

    def on(tag):
        def rec(*a, **k):
            where = "side" if torch.cuda.current_stream() == core._side_stream else "main"
            log.append((tag, where))
            rows = next((t.shape[0] for t in a if isinstance(t, torch.Tensor)), 1)
            out = torch.zeros(rows, 8, dtype=BF16)
            if tag == "ar_quant":
                return out, True
            return (out, None) if tag == "ar_x8" else out

        return rec

    attn = SimpleNamespace(o=None, o8=None, o_dec8=None, norm2=None, qkva8=None)
    core._attn = [attn, attn]
    core._mlp = [(None, None), (None, None)]
    core._moe = [SimpleNamespace(g1=None)]
    core._moe_k = SimpleNamespace(topk_pre=object(), pair=object(), merge=None)
    core._shared_w8 = [None, None]
    core._shared_gu_il = [None, None]
    core._next_norm = [None]
    core._mtp_fp4 = object()
    core._mtp_out_normed = False
    core._project = on("o_proj")
    core._pf_fork = lambda *a: None
    core._reduce_norm_quant = on("ar_quant")
    core._reduce_norm_x8 = on("ar_x8")
    core._reduce_norm = on("ar")
    core._moe_begin_layer = lambda rows, exact: torch.zeros(rows, 8, dtype=BF16)
    core._shared_mlp = on("shared")
    core._routed = on("routed")
    core._add = on("add")
    core._reduce = on("ar_close")
    core._rmsnorm = on("norm")
    core._add_rmsnorm = on("add_norm")
    core._mtp = {"head_norm": None}
    real_cut = core._shared_after_collective

    def cut():
        on("cut")()
        real_cut()

    core._shared_after_collective = cut
    return core, log


@pytest.mark.parametrize("rows", (64, 96, 128, 192))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_moe_close_shared_after_collective(max_draft_len, decode, rows) -> None:
    """On an MTP3 / MTP5 decode step's exact rows (`_moe_decode_exact`) a trunk MoE layer and
    the MTP layer put one plain kernel on the side stream right after its fork, ahead of the
    shared expert (`_shared_after_collective`: the shared GEMM then cannot launch off the
    pre-MoE collective's PDL release ahead of the router). Nowhere else: MTP1, no speculation
    and context steps issue exactly the calls they did."""
    exact = decode and max_draft_len > 1 and rows > 64
    for close in ("trunk", "mtp"):
        core, log = _moe_close_core(max_draft_len)
        x = torch.zeros(rows, 8, dtype=BF16)
        if close == "trunk":
            core._mlp_and_close(x, x, 0, rows, decode=decode)
            # The pre-MoE collective writes the routed experts' NVFP4 input at decode widths
            # and on the exact wide rows (`_moe_prequant`), else it is the plain AR + norm.
            prequant = rows <= 64 or exact
            first = [("o_proj", "main"), ("ar_quant" if prequant else "ar_x8", "main")]
        else:
            core._mtp_close(x, x, rows, decode=decode)
            first = [("o_proj", "main"), ("ar", "main")]
        cut = [("cut", "side")] if exact else []
        want = first + cut + [("shared", "side"), ("routed", "main")]
        assert log[: len(want)] == want, (close, log)
        assert core.route_hits.get("moe_shared_after_ar", 0) == (1 if exact else 0), core.route_hits


@pytest.mark.parametrize("rows", (32, 64, 128, 192))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_mtp_close_output_add_and_head_norm(max_draft_len, decode, rows) -> None:
    """The MTP layer's output: routed + shared, the closing all-reduce, then the residual add.
    On an MTP3 / MTP5 decode step (its 1-token draft steps' 32 rows included) the residual
    add and the head norm are one launch (`_add_rmsnorm`, bit for bit the pair: the op's own
    test); those engines' context steps keep the two launches, and MTP1 / no speculation
    return the un-normed sum as before."""
    core, log = _moe_close_core(max_draft_len)
    core._mtp_out_normed = max_draft_len > 1
    x = torch.zeros(rows, 8, dtype=BF16)
    core._mtp_close(x, x, rows, decode=decode)
    exact = decode and max_draft_len > 1 and rows > 64
    cut = [("cut", "side")] if exact else []
    head = [("o_proj", "main"), ("ar", "main")] + cut + [("shared", "side"), ("routed", "main")]
    tail = [("add", "main"), ("ar_close", "main")]
    if max_draft_len > 1 and decode:
        tail += [("add_norm", "main")]
    elif max_draft_len > 1:
        tail += [("add", "main"), ("norm", "main")]
    else:
        tail += [("add", "main")]
    assert log == head + tail, log


@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_mtp_layer_rope_widths(max_draft_len) -> None:
    """Which MTP-layer decode widths take the specialised rope kernel: `mtp_widths` (2, and
    max_draft_len + 1 at MTP3 / MTP5), and at MTP3 / MTP5 also width 1, the 1-token draft
    steps 1..max_draft_len-1 (the stock op there needed the latent packed first). No
    speculation and MTP1 never run a 1-token MTP-layer step and keep exactly their widths."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.max_draft_len = max_draft_len
    core.mtp_widths = (2,) + ((max_draft_len + 1,) if max_draft_len > 1 else ())
    taken = {p for p in range(1, 9) if core._mtp_rope_specialised(p)}
    want = set(core.mtp_widths) | ({1} if max_draft_len > 1 else set())
    assert taken == want, (taken, want)
