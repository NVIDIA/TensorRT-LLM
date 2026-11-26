# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which GEMM the GLM-5.2 target's call sites pick for a step's row count, by engine and phase.

MTP3 / MTP5 decode steps carry 128 / 192 rows (32 sequences x 4 / 6), above the 64 rows that
otherwise tell a decode step from a context one. The model routes such a step to its own
Target (`_select_target(md, rows)`: `WideDecodeTarget`), whose ops run at `Phase(wide=True)`,
and only then send rows above 64 to the MTP decode cells. What this pins, on a core built
without weights (`object.__new__`), its ops (`forward_ops`) called at the phase of the Target
the step routes to, and with every entry replaced by a recorder:

* no speculation and MTP1 take exactly the paths they took before the MTP cells existed, at
  every row count and phase, even with MTP cells present;
* MTP3 / MTP5 context steps do too;
* MTP3 / MTP5 decode steps keep the numerics of the call's 64-row decode path above 64 rows:
  where that is bf16, the MTP cell at 128 / 192 rows, and at any other count up to 192 (a batch
  of fewer than 32 sequences: 96, 150) the next cell up on the exact rows (`Gemm.wide_cell`: a
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

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    decode as DT,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    forward_ops as F,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as M,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    prefill as PT,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    wide_decode as W,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.activation import (  # noqa: E402, E501
    cute_dsl_swiglu_rows as SR,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    mla_rope as MR,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm import (  # noqa: E402, E501
    cute_dsl_allreduce as AR,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.comm import (  # noqa: E402, E501
    cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 as ARQ,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    order_fence as FENCE,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    absorbed_bmm as AB,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    indexer_fold as IF,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    projection as P,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    shared_mlp as SM,
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

    projection = record(lambda x, w, out, r: r.tag)
    monkeypatch.setattr(P, "cute_dsl_projection", projection)
    # The shared expert's prefill down GEMM, and the logits head (on the core), launch the
    # projection kernel themselves.
    monkeypatch.setattr(SM, "cute_dsl_projection", projection)
    monkeypatch.setattr(M, "cute_dsl_projection", projection)
    # A cell re-pinned to fewer rows (`_projection_op.at_rows`): the cell's tag plus the rows.
    # Read when the handles are built (`_build_gemm_handles`, by `_step`).
    monkeypatch.setattr(F._projection_op, "at_rows", lambda r, rows: _runner(f"{r.tag}@{rows}"))
    for mod in (P, F):
        monkeypatch.setattr(mod, "cute_dsl_projection_mxf8", record(lambda *a, **k: "mxf8"))
    monkeypatch.setattr(
        P, "cute_dsl_projection_mxf8_decode", record(lambda x, w, out, r, pre=False: r.tag)
    )
    monkeypatch.setattr(P, "cublas_mm", lambda x, w, b, d: calls.append("cublas") or x)
    monkeypatch.setattr(AB, "cute_dsl_absorbed_bmm", record(lambda a, w, out, r: r.tag))
    monkeypatch.setattr(AB, "bmm_out", record(lambda *a: "bmm_out"))
    # A fold's plain-projection fallback (`indexer_fold`): the fold tests see only the fold.
    monkeypatch.setattr(IF, "projection", lambda *a, **k: None)
    monkeypatch.setattr(F, "cute_dsl_indexer_pre_proj_split", record(lambda x, w, kb, wq, h: h.tag))
    monkeypatch.setattr(SM, "cute_dsl_shared_gate_up_swiglu", record(lambda x, w, o, r: r.tag))
    monkeypatch.setattr(F, "cute_dsl_mtp_norm_shard", record(lambda *a: "norm_shard"))
    monkeypatch.setattr(
        F, "cute_dsl_mtp_eh_proj", lambda local, w, r: calls.append("ksplit" if r is None else r)
    )
    # The K-split re-pinned to fewer rows (`_mtp_eh.at_rows`): tagged with the rows.
    monkeypatch.setattr(F._mtp_eh, "at_rows", lambda r, rows: f"ksplit@{rows}")
    monkeypatch.setattr(F, "cute_dsl_mtp_eh_norm_cat_mxf8", record(lambda *a: "eh_norm_cat8"))
    monkeypatch.setattr(F._eh8_op, "supports", lambda e, h: True)
    monkeypatch.setattr(F._eh8_op, "operand", lambda rows, dev: (None, None))
    # A new contiguous tensor, as the real SwiGLU returns (the down GEMM's paths check it).
    monkeypatch.setattr(
        SR, "flashinfer_silu_and_mul", lambda t: t[:, : t.shape[1] // 2].contiguous()
    )

    def build(max_draft_len: int):
        core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
        core.max_draft_len = max_draft_len
        core.wide_cells = core.max_draft_len > 1
        core._wide_bucket = True
        # TRTLLM_GLM52_DEC_CELL_UP (default on; `dec_cell_up` as `__init__` derives it), and the
        # fused shared-expert cells an MTP3 / MTP5 engine builds at its small-batch decode
        # counts (none here unless a test adds).
        core._dec_cell_up = True
        core.dec_cell_up = max_draft_len > 1
        core._shared_gu_swiglu_up = {}
        core.route_hits, core.route_misses = {}, {}
        core.tp_rank, core.eps = 0, 1e-5
        core.idx_heads, core.idx_dim = 32, 128
        core._rows = core._build_row_ops(compile=False)
        core.op = _ops(core)
        core.op.comm.reduce = lambda t: t
        core.op.rows.rmsnorm = lambda x, w: x
        core.op.rows.copy = lambda dst, src: dst
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
        core._shared_pre = (_runner("shared_up:prefill"), _runner("shared_down:shared_pre"))
        core._shared8 = None
        core._dense_mxf8, core._dense_swiglu = None, None
        core._moe_mid = _aligned(256, 512)
        core._shared_gu_swiglu = {64: _runner("shared_gu@64")}
        core._shared_gu_swiglu_mtp = {m: _runner(f"shared_gu@{m}:mtp") for m in MTP_ROWS}
        core._mtp = dict(enorm=None, hnorm=None, eh_local=None, ehproj=None, norm_shard=None)
        core._mtp["eh"] = torch.zeros(8, 4, dtype=BF16)
        core._mtp_eh8 = SimpleNamespace(norm=None, w8=None)
        return core, calls

    return build


def _ops(core) -> SimpleNamespace:
    """The core's ops, as its `__init__` builds them (`core.op`)."""
    return SimpleNamespace(
        rows=F.Rows(core), comm=F.Comm(core), gemm=F.Gemm(core),
        indexer=F.Indexer(core), moe=F.Moe(core), mtp=F.Mtp(core), ctx=F.Ctx(core),
    )  # fmt: skip


def _absorb(core, kind: str) -> AB.AbsorbedBmmHandle:
    """`kind`'s absorbed-BMM handle over the core's runner tables, as `_build_attn_handles`
    builds it."""
    return AB.build(
        kind,
        cells=core._absorb.get(kind, {}),
        mtp_cells=core._absorb_mtp.get(kind, {}),
        prefill=core._absorb_pre.get(kind),
        wide_bucket=core._wide_bucket,
        count=core._count,
    )


def _qkva_fold(core, ph, x, i: int):
    """`qkva_ix_fold` on layer `i`'s stacked weight over the core's fold cells, as the Targets
    call it: `(qkva, key, head_weights)` where the fold serves, else None (the plain
    projection, stubbed out by `core_for`)."""
    h = IF.build_qkva(core._qkva_ix_mtp, None, core._wide_bucket, core._count)
    qkva, kw = IF.qkva_ix_fold(ph, x, None, None, None, core._ix_fold_w.get(i), h)
    return None if kw is None else (qkva, *kw)


def _qbix_cell(core, ph, i: int, rows: int):
    """The q_b | wq_b fold cell `q_b_ix_wq_b` launches for layer `i`'s `rows`, or None."""
    h = IF.build_q_b(core._qbix_mtp, None, core._wide_bucket, core._count)
    return IF._fold_cell(ph, rows, core._qbix_w.get(i), h, "qb_ixqb")


def _wide(max_draft_len: int, decode: bool, rows: int) -> bool:
    return decode and max_draft_len > 1 and rows > 64


def _step(core, decode: bool, rows: int) -> F.Phase:
    """Route a step of `rows` rows the way the engine's forward does (`_select_target`: a
    context step to prefill, an MTP3 / MTP5 decode step above 64 rows to the wide decode
    Target, any other decode step to the decode Target), and return the `Phase` the chosen
    Target runs its ops at. Builds the core's projection and MLP handles first, from its
    runner tables as the test left them."""
    core._build_gemm_handles()
    core._prefill, core._decode, core._wide = "prefill", "decode", "wide"
    target = core._select_target(SimpleNamespace(num_contexts=0 if decode else 2), rows)
    want = "wide" if _wide(core.max_draft_len, decode, rows) else "decode" if decode else "prefill"
    assert target == want, (core.max_draft_len, decode, rows, target)
    return {"prefill": F.PREFILL, "decode": M.DecodeTarget(core).phase, "wide": W._WIDE}[target]


@pytest.mark.parametrize("wide_env", (True, False))
@pytest.mark.parametrize("rows", (1, 32, 64, 65, 96, 128, 192))
@pytest.mark.parametrize("contexts", (0, 2))
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_select_target(max_draft_len, contexts, rows, wide_env) -> None:
    """Three Targets: a step with context rows on prefill; a decode step above 64 rows on the
    wide decode Target wherever the engine built the wide decode cells (always at MTP3 /
    MTP5; at MTP1 and without speculation unless `TRTLLM_GLM52_WIDE_DECODE=0`, `wide_env`
    False), else on the decode Target. The decode Target's phase takes the next cell up only
    on an MTP3 / MTP5 engine (with `TRTLLM_GLM52_DEC_CELL_UP`); the wide one's maps rows above
    64 onto the MTP decode cells."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.max_draft_len = max_draft_len
    core.wide_cells = core.max_draft_len > 1 or wide_env
    core._prefill, core._decode, core._wide = "prefill", "decode", "wide"
    got = core._select_target(SimpleNamespace(num_contexts=contexts), rows)
    if contexts:
        assert got == "prefill"
    elif (max_draft_len > 1 or wide_env) and rows > 64:
        assert got == "wide"
    else:
        assert got == "decode"
    for cell_up in (True, False):
        core.dec_cell_up = max_draft_len > 1 and cell_up
        want = F.Phase(up="decode") if max_draft_len > 1 and cell_up else F.PREFILL
        assert M.DecodeTarget(core).phase == want
    assert W._WIDE == F.Phase(wide=True)


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
    ph = _step(core, decode, rows)
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
        core.op.gemm.project(ph, name, x, w, w8=w8, wd8=wd8)
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
    ph = _step(core, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    wide = _wide(max_draft_len, decode, rows)
    x = _aligned(rows, 16)
    assert core.op.gemm.mxf8_takes(ph, "dense_gate_up", x, object()) == (rows > 64 and not wide)

    calls.clear()
    a = torch.zeros(16, rows, 8, dtype=BF16)
    AB.absorbed_bmm(ph, a, None, None, a, _absorb(core, "k_b"))
    # The K-B object of the 128 / 192 cells launches at any row count above 64 itself.
    want = "k_b@64" if rows <= 64 else f"k_b@{cell}:mtp" if cell else "k_b:prefill"
    assert calls == [want], calls

    calls.clear()
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None)
    core.op.indexer.project(ph, x, None, None, ix, rows, need_q=False)
    if rows <= 64:
        want = "ix_split@64"
    elif cell == rows:
        want = f"ix_split@{rows}:mtp"
    else:
        want = "ix_split:rag" if cell else "ix_split:prefill"
    assert calls == [want], calls


def _shared(core, ph, x, gu, dn, trunk: bool):
    """The shared expert as the Targets call it: gate|up + SwiGLU, then down."""
    act, prefill = SM.shared_gate_up_swiglu(ph, x, gu, None, None, None, core._shared_h, trunk)
    return SM.shared_down(ph, act, dn, core._shared_h, None, prefill)


@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("trunk", [True, False], ids=["trunk", "mtp_layer"])
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_shared_expert(core_for, max_draft_len, decode, trunk, rows, bucket) -> None:
    """Gate|up + SwiGLU, then down, for a trunk layer and the MTP layer."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    ph = _step(core, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    x = _aligned(rows, 16)
    gu = torch.zeros(16, 1024, dtype=BF16)
    dn = torch.zeros(512, 8, dtype=BF16)
    _shared(core, ph, x, gu, dn, trunk)
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
    ph = _step(core, decode, rows)
    wide = _wide(max_draft_len, decode, rows)
    e = torch.zeros(rows, 16, dtype=BF16)
    core.op.mtp.front(ph, e, e, rows)
    if rows in (64, 128) or (rows == 192 and wide):
        want = ["norm_shard", "ksplit"]
    elif wide and bucket:
        want = ["norm_shard", f"ksplit@{rows}"]
    elif wide:
        want = ["mtp_eh:prefill"]
    else:
        want = ["eh_norm_cat8", "mxf8"]
    assert calls == want, calls


@pytest.mark.parametrize("own_m", [False, True], ids=["runtime_m", "own_m_64"])
@pytest.mark.parametrize("rows", (1, 17, 33, 63, 64))
@pytest.mark.parametrize("draft", [True, False], ids=["draft_eager", "not_draft"])
@pytest.mark.parametrize("max_draft_len", (3, 5))
def test_draft_eager_rows(core_for, monkeypatch, max_draft_len, draft, rows, own_m) -> None:
    """An MTP3 / MTP5 eager draft step (`DRAFT`: `DecodeTarget.mtp_forward` outside a
    CUDA-graph capture, the draft loop of a context or mixed step) at a row count without a
    decode cell runs each
    call on a cell at its exact rows, never cuBLAS: a projection on the smallest decode cell
    above it, or the MTP cell where that one is compiled for its own M only (`at_rows` gives
    None: `own_m_64`); the indexer split on the runtime-M handle; eh_proj on the next K-split
    instance. A count with its own cell, and every count off a draft step, route as before."""
    core, calls = core_for(max_draft_len)
    if own_m:
        monkeypatch.setattr(
            F._projection_op,
            "at_rows",
            lambda r, rows: None if r.tag.endswith("@64") else _runner(f"{r.tag}@{rows}"),
        )
    # Off a draft step: a context step's rows (no cell-up rule).
    core._build_gemm_handles()
    ph = F.DRAFT if draft else _step(core, False, rows)
    eager = draft and rows != 64
    x = _aligned(rows, 160)
    w = torch.zeros(4, 160, dtype=BF16).t()
    for name in ("qkva", "q_b", "o_proj"):
        calls.clear()
        core.op.gemm.project(ph, name, x, w)
        if not eager:
            want = f"{name}@64" if rows == 64 else "cublas"
        else:
            want = f"{name}@128:mtp@{rows}" if own_m else f"{name}@64@{rows}"
        assert calls == [want], (name, rows, calls)

    calls.clear()
    ix = SimpleNamespace(fused_bf16=None, fused=None, kn_w=None, kn_b=None)
    core.op.indexer.project(ph, x, None, None, ix, rows, need_q=False)
    want = "ix_split@64" if rows == 64 else "ix_split:rag" if eager else "cublas"
    assert calls == [want], calls

    calls.clear()
    e = torch.zeros(rows, 16, dtype=BF16)
    core.op.mtp.front(ph, e, e, rows)
    if rows in (1, 64):
        want = ["norm_shard", "ksplit"]
    else:
        want = ["norm_shard", f"ksplit@{rows}"] if draft else ["cublas"]
    assert calls == want, calls
    # One `draft_cell` per eager call: 3 projections, the indexer split, eh_proj (1 row has
    # its own K-split instance).
    assert core.route_hits.get("draft_cell", 0) == (0 if not eager else 4 if rows == 1 else 5)


@pytest.mark.parametrize("cell_up", [True, False], ids=["cell_up", "cell_up_off"])
@pytest.mark.parametrize("rows", (6, 12, 18, 64))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_captured_decode_rows_without_a_cell(
    core_for, monkeypatch, max_draft_len, decode, rows, cell_up
) -> None:
    """An MTP3 / MTP5 decode step (graph-captured, `DecodeTarget` at `Phase(up="decode")`) at a
    row count up to 64 without a decode cell -- MTP5 at 1-3 requests: 6 / 12 / 18 -- takes the
    eager draft step's rule (`projection.cell_up`): each projection the next cell up at its
    exact rows (the MTP cell where the decode cell is compiled for its own M), the indexer split the runtime-M handle
    (not `x.float()` + cuBLAS), eh_proj the next K-split instance, the shared expert its fused
    cell built at that count. 64 (a cell) is the control. No speculation, MTP1, context steps
    and `TRTLLM_GLM52_DEC_CELL_UP=0` keep cuBLAS / the two-launch path there."""
    core, calls = core_for(max_draft_len)
    core._dec_cell_up = cell_up
    core.dec_cell_up = max_draft_len > 1 and cell_up
    # A decode cell at 16 that launches at runtime M for q_b; the rest of the 64-row cells are
    # compiled for their own M (`at_rows` None): those go to the MTP cell (128).
    core._proj["q_b"][16] = _runner("q_b@16")
    monkeypatch.setattr(
        F._projection_op,
        "at_rows",
        lambda r, rows: None if r.tag.endswith("@64") else _runner(f"{r.tag}@{rows}"),
    )
    if max_draft_len > 1:
        core._shared_gu_swiglu_up = {
            m: _runner(f"shared_gu@{m}:up") for m in {(max_draft_len + 1) * b for b in (1, 2, 3)}
        }
    ph = _step(core, decode, rows)
    up = decode and max_draft_len > 1 and cell_up and rows != 64
    x = _aligned(rows, 160)
    w = torch.zeros(4, 160, dtype=BF16).t()
    # dense_down's cells, as the other projections' (the fixture builds none for it).
    core._proj["dense_down"] = {64: _runner("dense_down@64")}
    core._proj_mtp["dense_down"] = {m: _runner(f"dense_down@{m}:mtp") for m in MTP_ROWS}
    core._build_gemm_handles()
    for name in ("qkva", "q_b", "ix_qb", "dense_gate_up", "dense_down"):
        calls.clear()
        core.op.gemm.project(ph, name, x, w)
        if rows == 64:
            want = f"{name}@64"
        elif not up or name in P.CELL_UP_KEEPS_CUBLAS:
            # ix_qb and dense_down keep cuBLAS: there it is faster than their next cell up.
            want = "cublas"
        elif name == "q_b" and rows < 16:
            want = f"q_b@16@{rows}"
        else:
            want = f"{name}@128:mtp@{rows}"
        assert calls == [want], (name, rows, calls)

    calls.clear()
    ix = SimpleNamespace(fused_bf16=None, fused=None, kn_w=None, kn_b=None)
    core.op.indexer.project(ph, x, None, None, ix, rows, need_q=False)
    want = "ix_split@64" if rows == 64 else "ix_split:rag" if up else "cublas"
    assert calls == [want], calls

    calls.clear()
    e = torch.zeros(rows, 16, dtype=BF16)
    core.op.mtp.front(ph, e, e, rows)
    if rows == 64:
        want = ["norm_shard", "ksplit"]
    else:
        want = ["norm_shard", f"ksplit@{rows}"] if up else ["cublas"]
    assert calls == want, calls

    calls.clear()
    xs = _aligned(rows, 16)
    gu = torch.zeros(16, 1024, dtype=BF16)
    dn = torch.zeros(512, 8, dtype=BF16)
    _shared(core, ph, xs, gu, dn, True)
    if rows == 64:
        want = ["shared_gu@64", "shared_down@64"]
    elif up and rows in core._shared_gu_swiglu_up:
        want = [f"shared_gu@{rows}:up", f"shared_down@128:mtp@{rows}"]
    elif up:
        # No fused cell at this count (MTP3 at 6 / 18): the two-launch path on the cells.
        want = [f"shared_gate_up@128:mtp@{rows}", f"shared_down@128:mtp@{rows}"]
    else:
        want = ["cublas", "cublas"]
    assert calls == want, calls
    # The same cells again: kept per (call, rows), frozen for the capture shape.
    calls.clear()
    core.op.gemm.project(ph, "qkva", x, w)
    assert len(calls) == 1
    assert core.route_hits.get("draft_cell", 0) == 0
    if up:
        assert core.route_hits.get("dec_cell_up", 0) > 0
    else:
        assert "dec_cell_up" not in core.route_hits


@pytest.mark.parametrize("cell_up", [True, False], ids=["cell_up", "cell_up_off"])
@pytest.mark.parametrize("rows", (6, 12, 18, 16))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_absorbed_bmm_at_decode_rows_without_a_cell(
    core_for, max_draft_len, decode, rows, cell_up
) -> None:
    """K-B / V-B at an MTP3 / MTP5 decode step's 12 / 18 rows (9..64 without a cell) take the
    M >= 16 group's object, which launches at them, instead of the prefill BMM; so do the fp8
    q-base K-B and the V-B that writes o_proj's MXF8 operand (`decode_cell`). 6 rows is below
    that group (the prefill BMM as before); 16 is a cell (control). No speculation, MTP1,
    context steps and `TRTLLM_GLM52_DEC_CELL_UP=0` keep the prefill BMM there."""
    core, calls = core_for(max_draft_len)
    core._dec_cell_up = cell_up
    core.dec_cell_up = max_draft_len > 1 and cell_up
    group = _runner("k_b@16..64")
    core._absorb = {"k_b": {16: group, 64: group}}
    core._kb8 = {m: _runner("k_b8@16..64") for m in (16, 32, 48, 64)}
    core._vb8 = {m: _runner("v_b8@16..64") for m in (16, 32, 48, 64)}
    ph = _step(core, decode, rows)
    up = decode and max_draft_len > 1 and cell_up and 9 <= rows <= 64
    calls.clear()
    a = torch.zeros(16, rows, 8, dtype=BF16)
    AB.absorbed_bmm(ph, a, None, None, a, _absorb(core, "k_b"))
    want = "k_b@16..64" if rows == 16 or up else "k_b:prefill"
    assert calls == [want], calls
    for cells, tag in ((core._kb8, "k_b8@16..64"), (core._vb8, "v_b8@16..64")):
        got = AB.decode_cell(ph, cells, rows, core._count)
        assert (got.tag if got is not None else None) == (tag if rows == 16 or up else None), (
            tag,
            rows,
            got,
        )
    assert core.route_hits.get("dec_cell_up", 0) == (3 if up and rows != 16 else 0)


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


def _moe_close_core(max_draft_len: int, monkeypatch):
    """A one-MoE-layer core whose Targets run on recorders: every op the MoE-layer close calls
    (o_proj, the collectives, the shared expert, the routed experts, the adds and norms) is
    logged with the stream it was issued on ("side" / "main"). The ops that decide the close's
    path keep their own logic: `allreduce_add_rmsnorm_quant` (which collective, at the window
    the Target passes), `Moe.wide_exact` / `Moe.open_layer` / `Moe.combine`,
    `Mtp.wide_exact` / `Mtp.moe`, and `order_fence` (logged as "cut" where it launches). The
    attention ahead of o_proj is a stub."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.max_draft_len = max_draft_len
    core.wide_cells = core.max_draft_len > 1
    core._dec_cell_up = True
    core.dec_cell_up = max_draft_len > 1
    core.route_hits, core.route_misses = {}, {}
    core.num_layers = 1
    core.heads, core.v_dim, core.kv_lora = 1, 8, 8
    core.idx_topk = 2048
    core.eps = 1e-5
    core.w, core._embedding = {"embed": None}, None
    core._side_stream = torch.cuda.Stream()
    log = []

    def on(tag, ret=lambda out: out):
        def rec(*a, **k):
            where = "side" if torch.cuda.current_stream() == core._side_stream else "main"
            log.append((tag, where))
            rows = next((t.shape[0] for t in a if isinstance(t, torch.Tensor)), 1)
            return ret(torch.zeros(rows, 8, dtype=BF16))

        return rec

    attn = SimpleNamespace(
        **dict.fromkeys(
            ("o", "o8", "o_dec8", "norm1", "norm2", "qkva", "qkva8", "ix_fold_w", "q_norm"),
        ),
        **dict.fromkeys(("kv_norm", "qb", "qbix_w", "qb_dec8", "k_b", "k_b_nk", "v_b_t", "v_b_nk")),
    )
    core._attn = [attn, attn]
    core._idx = [None, None]
    core._pool_layer_id = [0]
    core._mlp = [(None, None), (None, None)]
    core._moe = [SimpleNamespace(g1=None)]
    core._moe_k = SimpleNamespace(topk_pre=object(), pair=object(), merge=None)
    core._dense_w8 = []
    core._shared_w8 = [None, None]
    core._shared_gu_il = [None, None]
    core._next_norm = [None]
    core._mtp_fp4 = object()
    core._mtp_out_normed = False
    core._mtp = {"head_norm": None}
    core.mtp_index_share = False
    # The handles the Targets pass: o_proj's (the recorder logs its name), the collectives'
    # (the quant op's own window logic runs on them), the order fence's, and the handoffs.
    core._proj_h = SimpleNamespace(o_proj=SimpleNamespace(name="o_proj"))
    core._comm = AR.build(SimpleNamespace(max_rows=8192), core._count)
    core._ar_quant = ARQ.build(core._comm, torch.zeros(256, 1), torch.zeros(256, 1))
    core._shared_fence = FENCE.build("cpu", "moe_shared_after_ar", core._count)
    core._dense_h = core._shared_h = None
    core._rows = core._build_row_ops(compile=False)
    core._mlp_x8, core._close_x8 = [False], [False]
    core._qb_pre, core._vb_pre = [False, False], [False, False]
    core._attn_h = SimpleNamespace(
        **dict.fromkeys(("qkva_fold", "norm", "q_b_fold", "fmha", "k_b", "rope", "v_b"))
    )
    core._quant_rows, core._quant_rows_wide = core._prequant_rows(False), core._prequant_rows(True)
    core.op = _ops(core)
    op = core.op
    # The pre-MoE collective's two kernels inside `allreduce_add_rmsnorm_quant`.
    monkeypatch.setattr(ARQ, "cute_dsl_allreduce_add_rmsnorm_quant_nvfp4", on("ar_quant"))
    monkeypatch.setattr(ARQ, "allreduce_add_rmsnorm", on("ar_x8", lambda out: (out, None)))
    real_fence = FENCE.order_fence

    def fence(launch, handle):
        if launch:
            on("cut")()
        real_fence(launch, handle)

    for mod in (DT, W, PT):
        monkeypatch.setattr(mod, "embed", lambda ids, w, runner: ids)
        monkeypatch.setattr(mod, "hidden_rmsnorm", on("norm"))
        monkeypatch.setattr(mod, "hidden_add", on("add"))
        monkeypatch.setattr(mod, "allreduce_add_rmsnorm", on("ar", lambda out: (out, None)))
        monkeypatch.setattr(mod, "allreduce", on("ar_close"))
        monkeypatch.setattr(mod, "projection", lambda ph, x, w, h, *a, **k: on(h.name)(x))
        monkeypatch.setattr(mod, "shared_gate_up_swiglu", on("shared", lambda out: (out, False)))
        monkeypatch.setattr(mod, "shared_down", lambda ph, act, *a, **k: act)
    monkeypatch.setattr(W, "order_fence", fence)
    op.rows.embed = lambda ids: ids
    op.rows.rmsnorm = on("norm")
    op.rows.add = on("add")
    op.rows.add_rmsnorm = on("add_norm")
    op.indexer.topk_buf = lambda *a: None
    # The attention ahead of o_proj: its ops pass `x` through and launch nothing.
    for mod in (DT, W):
        monkeypatch.setattr(mod, "qkva_ix_fold", lambda ph, x, *a: (x, None))
        monkeypatch.setattr(mod, "qkv_a_rmsnorm", lambda qkva, *a: (qkva, qkva, None, False))
        monkeypatch.setattr(mod, "q_b_ix_wq_b", lambda ph, q_a, *a: (q_a, None))
        monkeypatch.setattr(mod, "decode_slots", lambda *a: (None, "global", None))
        monkeypatch.setattr(mod, "convert_req_index_to_global", lambda *a: None)
        monkeypatch.setattr(mod, "k_b_q_base", lambda *a: None)
        monkeypatch.setattr(mod, "mla_rope", lambda *a: (None, None, None, None))
        monkeypatch.setattr(mod, "sparse_mla_decode", lambda *a: None)
        monkeypatch.setattr(mod, "v_b", lambda *a: False)
    op.mtp.front = lambda ph, e, h, rows: h
    op.moe.pf_fork = lambda *a, **k: None
    op.moe.begin_layer = lambda rows, exact=False: torch.zeros(rows, 8, dtype=BF16)
    op.moe.routed = on("routed")
    core._prefill = M.PrefillTarget(core)
    core._decode = M.DecodeTarget(core)
    core._wide = W.WideDecodeTarget(core)
    return core, log


def _close(core, close: str, decode: bool, rows: int) -> None:
    """Run a step's trunk MoE layer (`close == "trunk"`) or MTP layer (`"mtp"`) through the
    Target `_select_target` routes the step to: on a context step its MoE-layer close
    (`PrefillTarget._mlp_and_close` / `_mtp_close`), on a decode step the decode Target's
    whole `forward` / `mtp_forward` (one layer)."""
    x = torch.zeros(rows, 8, dtype=BF16)
    md = SimpleNamespace(num_contexts=0 if decode else 2)
    target = core._select_target(md, rows)
    if not decode:
        assert target is core._prefill
        if close == "trunk":
            core._prefill._mlp_and_close(x, x, 0, rows, None, None)
        else:
            core._prefill._mtp_close(x, x, rows)
        return
    want = core._wide if core.max_draft_len > 1 and rows > 64 else core._decode
    assert target is want
    md = SimpleNamespace(
        num_contexts=0, num_tokens=rows, num_seqs=1, skip_indexer_for_gen_reqs=False,
        req_idx_per_token=torch.zeros(rows, dtype=torch.int32),
        host_kv_cache_pool_mapping=torch.zeros((1, 2), dtype=torch.int32), tokens_per_block=64,
        block_table=None,
        kv_cache_manager=SimpleNamespace(layer_offsets={core.num_layers: 0}),
    )  # fmt: skip
    pos = torch.zeros(rows, dtype=torch.int32)
    if close == "trunk":
        target.forward(md, position_ids=pos, inputs_embeds=x)
    else:
        target.mtp_forward(None, None, pos, x, md, position_ids=pos)


def _from_o_proj(log: list) -> list:
    """The log from o_proj on: the MoE-layer close (the attention ahead of it logs a norm)."""
    return log[log.index(("o_proj", "main")) :]


@pytest.mark.parametrize("rows", (64, 96, 128, 192))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_moe_close_shared_after_collective(max_draft_len, decode, rows, monkeypatch) -> None:
    """On an MTP3 / MTP5 decode step's exact rows (`Moe.decode_exact`, `WideDecodeTarget`) a
    trunk MoE layer and the MTP layer put one plain kernel on the side stream right after its
    fork, ahead of the shared expert (`order_fence`: the shared GEMM then
    cannot launch off the pre-MoE collective's PDL release ahead of the router). Nowhere else:
    MTP1, no speculation and context steps issue exactly the calls they did."""
    exact = decode and max_draft_len > 1 and rows > 64
    for close in ("trunk", "mtp"):
        with monkeypatch.context() as mp:
            core, log = _moe_close_core(max_draft_len, mp)
            _close(core, close, decode, rows)
        if close == "trunk":
            # The pre-MoE collective writes the routed experts' NVFP4 input at decode widths
            # and on the exact wide rows (`_quant_rows_wide`), else the plain AR + norm.
            prequant = rows <= 64 or exact
            first = [("o_proj", "main"), ("ar_quant" if prequant else "ar_x8", "main")]
        else:
            first = [("o_proj", "main"), ("ar", "main")]
        cut = [("cut", "side")] if exact else []
        want = first + cut + [("shared", "side"), ("routed", "main")]
        got = _from_o_proj(log)
        assert got[: len(want)] == want, (close, log)
        assert core.route_hits.get("moe_shared_after_ar", 0) == (1 if exact else 0), core.route_hits


@pytest.mark.parametrize("rows", (32, 64, 128, 192))
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_mtp_close_output_add_and_head_norm(max_draft_len, decode, rows, monkeypatch) -> None:
    """The MTP layer's output: routed + shared, the closing all-reduce, then the residual add.
    On an MTP3 / MTP5 decode step (its 1-token draft steps' 32 rows included) the residual
    add and the head norm are one launch (`Rows.add_rmsnorm`, bit for bit the pair: the op's
    own test); those engines' context steps keep the two launches, and MTP1 / no speculation
    return the un-normed sum as before."""
    core, log = _moe_close_core(max_draft_len, monkeypatch)
    core._mtp_out_normed = max_draft_len > 1
    _close(core, "mtp", decode, rows)
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
    assert _from_o_proj(log) == head + tail, log


@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_mtp_layer_rope_widths(max_draft_len) -> None:
    """Which MTP-layer decode widths take the specialised rope kernel: `mtp_widths` (2, and
    max_draft_len + 1 at MTP3 / MTP5), and at MTP3 / MTP5 also width 1, the 1-token draft
    steps 1..max_draft_len-1 (the stock op there needed the latent packed first). No
    speculation and MTP1 never run a 1-token MTP-layer step and keep exactly their widths."""
    core = object.__new__(M.GlmMoeDsaTep4ModelingV2)
    core.max_draft_len = max_draft_len
    core.wide_cells = core.max_draft_len > 1
    core.mtp_widths = (2,) + ((max_draft_len + 1,) if max_draft_len > 1 else ())
    widths = MR.widths(core.mtp_widths, max_draft_len)
    taken = {p for p in range(1, 9) if widths.takes("mtp", p)}
    assert {p for p in range(1, 9) if widths.takes("trunk", p)} == set(core.mtp_widths)
    assert not any(widths.takes("stock", p) for p in range(1, 9))
    want = set(core.mtp_widths) | ({1} if max_draft_len > 1 else set())
    assert taken == want, (taken, want)


@pytest.mark.parametrize("built", [True, False], ids=["fold", "fold_off"])
@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_qkva_ix_fold(core_for, monkeypatch, max_draft_len, decode, rows, bucket, built) -> None:
    """The qkva + indexer key / head-weight fold (`qkva_ix_fold`) serves exactly where the qkva
    projection would take its bf16 MTP cell -- an MTP3 / MTP5 decode step above 64 rows, at the
    cell or (buckets on) the next cell up on the exact rows -- on a layer whose weights were
    stacked, with the fold cells built (`TRTLLM_GLM52_WIDE_IX_FOLD`, on by default); everywhere
    else it returns None and the two GEMMs run. With the caller's keys and head weights the
    indexer front runs no projection of its own."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    # `post_load_weights` stacks only on an MTP3 / MTP5 engine; layer 0 has an indexer here.
    core._ix_fold_w = {0: torch.zeros(1, dtype=BF16)} if max_draft_len > 1 else {}
    core._qkva_ix_mtp = {m: _runner(f"fold@{m}") for m in MTP_ROWS} if built else {}
    monkeypatch.setattr(
        IF._ix_fold_op,
        "at_rows",
        lambda r, rows: None if rows > r.rows else _runner(f"{r.tag}@{rows}", rows=rows),
    )
    for m, r in core._qkva_ix_mtp.items():
        r.rows = m
    monkeypatch.setattr(
        IF, "cute_dsl_qkva_ix_fold", lambda x, w, out, r: calls.append(r.tag) or (out, None, None)
    )
    ph = _step(core, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    x = _aligned(rows, 16)
    calls.clear()
    got = _qkva_fold(core, ph, x, 0)
    serves = built and max_draft_len > 1 and cell is not None
    if serves:
        assert calls == [f"fold@{cell}" if cell == rows else f"fold@{cell}@{rows}"], calls
        assert got is not None and got[0].shape[0] == rows
    else:
        assert got is None and calls == [], (got, calls)
    # A layer without stacked weights (no indexer, or a stack the build skipped): the two GEMMs.
    calls.clear()
    assert _qkva_fold(core, ph, x, 1) is None and calls == []
    # Keys and head weights from the caller: no split, no fp32 projection.
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None)
    kb = torch.zeros(rows, 128, dtype=BF16)
    q_fp8, weights, k_pre = core.op.indexer.project(
        ph, x, None, None, ix, rows, need_q=False, kw=(kb, None)
    )
    assert calls == [] and k_pre[0] is kb and q_fp8 is None and weights is None


@pytest.mark.parametrize("built", [True, False], ids=["fold", "fold_off"])
@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_q_b_ix_wq_b(core_for, monkeypatch, max_draft_len, decode, rows, bucket, built) -> None:
    """The q_b | wq_b fold (`q_b_ix_wq_b`) serves exactly where q_b and ix_qb would take
    their bf16 MTP cells -- an MTP3 / MTP5 decode step above 64 rows, at the cell or (buckets
    on) the next cell up -- on a layer whose weights were stacked, with the cells built
    (`TRTLLM_GLM52_WIDE_QB_IXQB`); elsewhere None and the two GEMMs run. With the caller's
    indexer q the indexer front runs no wq_b projection."""
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    core._qbix_w = {0: torch.zeros(1, dtype=BF16)} if max_draft_len > 1 else {}
    core._qbix_mtp = {m: _runner(f"qbix@{m}", rows=m) for m in MTP_ROWS} if built else {}
    monkeypatch.setattr(
        IF._qbix_op,
        "at_rows",
        lambda r, rows: None if rows > r.rows else _runner(f"{r.tag}@{rows}", rows=rows),
    )
    ph = _step(core, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    got = _qbix_cell(core, ph, 0, rows)
    if built and max_draft_len > 1 and cell is not None:
        assert got is not None and got.tag == (
            f"qbix@{cell}" if cell == rows else f"qbix@{cell}@{rows}"
        )
    else:
        assert got is None
    assert _qbix_cell(core, ph, 1, rows) is None
    calls.clear()
    x = _aligned(rows, 16)
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None, qb=None)
    monkeypatch.setattr(F, "cute_dsl_indexer_q_side_fused", lambda *a, **k: calls.append("q_side"))
    core._ix_qside_compiled, core._idx_cos_sin, core.idx_weight_scale = None, None, 1.0
    core.indexer_is_neox = False
    kb = torch.zeros(rows, 128, dtype=BF16)
    ixq = torch.zeros(rows, 32 * 128, dtype=BF16)
    core.op.indexer.project(ph, x, x, None, ix, rows, need_q=True, kw=(kb, None), ixq=ixq)
    assert calls == ["q_side"], calls


@pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])
@pytest.mark.parametrize("built", [True, False], ids=["qk", "qk_off"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_indexer_q_side_k_cache(
    core_for, monkeypatch, max_draft_len, decode, rows, built, checked
) -> None:
    """The q side + K-cache update in one launch (`Indexer.qk_handle`, `TRTLLM_GLM52_WIDE_IX_QK`)
    serves only an MTP3 / MTP5 decode step above 64 rows, given the layer's cache, with the
    launchers built and the K update's slots validated once (its first eager call): it takes
    the launcher of the cache's layout, writes the keys (no pending `k_pre`) and counts
    `ix_qk`. Everywhere else the q side runs alone and the keys go to `Indexer.write_k`, as
    with keys the launcher cannot load (not 8-byte aligned)."""
    core, calls = core_for(max_draft_len)
    core._ix_qk_compiled = {c: _runner(f"qk:{c}") for c in (False, True)} if built else {}
    core.op.indexer._k_checked = checked
    ph = _step(core, decode, rows)
    monkeypatch.setattr(F, "cute_dsl_indexer_q_side_fused", lambda *a, **k: calls.append("q_side"))
    monkeypatch.setattr(
        F, "cute_dsl_indexer_q_side_k_cache", lambda *a, compiled, **k: calls.append(compiled.tag)
    )
    core._ix_qside_compiled, core._idx_cos_sin, core.idx_weight_scale = None, None, 1.0
    core.indexer_is_neox = False
    x = _aligned(rows, 16)
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None, qb=None)
    ixq = torch.zeros(rows, 32 * 128, dtype=BF16)
    md = SimpleNamespace(slot_mapping_fp8=None, slot_mapping_scale=None)
    dense = torch.zeros(4, 32, 1, 132, dtype=torch.uint8)
    strided = torch.zeros(4, 2, 32 * 132, dtype=torch.uint8)[:, 0].view(4, 32, 1, 132)

    def project(kb, pool):
        calls.clear()
        return core.op.indexer.project(
            ph, x, x, None, ix, rows, need_q=True, kw=(kb, None), ixq=ixq, kc=(md, pool)
        )

    kb = _aligned(rows, 128)
    fused = built and checked and _wide(max_draft_len, decode, rows)
    _, _, k_pre = project(kb, dense)
    if fused:
        assert calls == ["qk:True"] and k_pre is None and core.route_hits["ix_qk"] == 1
        _, _, k_pre = project(kb, strided)
        assert calls == ["qk:False"] and k_pre is None
    else:
        assert calls == ["q_side"] and k_pre[0] is kb and "ix_qk" not in core.route_hits
    misaligned = torch.zeros(rows * 128 + 1, dtype=BF16)[1:].view(rows, 128)
    _, _, k_pre = project(misaligned, dense)
    assert calls == ["q_side"] and k_pre[0] is misaligned


@pytest.mark.parametrize("nosplit", [True, False], ids=["nosplit", "split"])
@pytest.mark.parametrize("fold", [True, False], ids=["fold", "fold_off"])
@pytest.mark.parametrize("rag", [True, False], ids=["rag", "no_rag"])
@pytest.mark.parametrize("bucket", [True, False], ids=["bucket", "no_bucket"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("decode", [True, False], ids=["decode", "context"])
@pytest.mark.parametrize("max_draft_len", ENGINES)
def test_indexer_nosplit_beside_wide_decode(
    core_for, monkeypatch, max_draft_len, decode, rows, bucket, rag, fold, nosplit
) -> None:
    """The context indexer projection's no-split read (`TRTLLM_GLM52_CTX_IX_NOSPLIT`, default
    on: the prefill handle's fp32 `[T, 160]` output read in place by the pitch-160 q side and
    the K write's fp32-key launchers) beside the MTP3 / MTP5 wide decode indexer front, in the
    order the targets call it (decode: `qkva_ix_fold`, then
    `Indexer.project` with the layer's cache, then `Indexer.write_k` unless the keys were
    written; context and mixed: `Indexer.project`, then `Indexer.write_k`).

    * It applies wherever the prefill handle serves: every engine's context and mixed steps
      above 64 rows (an MTP3 / MTP5 mixed step routes to prefill, not to `WideDecodeTarget`,
      so its rows take it as MTP1's do).
    * A wide decode step's keys come from the fold, the MTP split cell or the runtime-M split
      (buckets on), never from it; the one-launch q side + K update then serves as before.
    * Where none of those serves (fold off and buckets off or no runtime-M handle) the wide
      rows fall back to the prefill handle and, with the read on, take it like a context
      step: pitch-160 q side, fp32 keys to the K write, and not the one-launch q side + K
      update (it loads contiguous bf16 keys). The bytes equal the split's (the no-split
      read's own test); only the launch count differs from the split fallback.
    """
    core, calls = core_for(max_draft_len)
    core._wide_bucket = bucket
    core._ix_nosplit = nosplit
    if not rag:
        core._ix_pre_split_rag = None
    core._ix_fold_w = {0: torch.zeros(1, dtype=BF16)} if max_draft_len > 1 else {}
    core._qkva_ix_mtp = {m: _runner(f"fold@{m}", rows=m) for m in MTP_ROWS} if fold else {}
    core._ix_qk_compiled = {c: _runner(f"qk:{c}") for c in (False, True)}
    core.op.indexer._k_checked = True
    core._ix_qside_compiled = SimpleNamespace(wq_pitch=32)
    core._ix_qside_compiled_p160 = SimpleNamespace(wq_pitch=160)
    core._ix_k_compiled = {False: "k:strided", True: "k:dense"}
    core._idx_cos_sin, core.rope_dim, core.idx_weight_scale = torch.zeros(4, 64), 64, 1.0
    core.indexer_is_neox = False
    x = _aligned(rows, 16)
    fold_kb, fold_wq = _aligned(rows, 128), torch.zeros(rows, 32)

    def unsplit(x, w, handle):
        calls.append(f"unsplit:{handle.tag}")
        full = torch.zeros(int(x.shape[0]), 160)
        return full[:, :128], full[:, 128:]

    monkeypatch.setattr(F._ix_pre_split_op, "project_unsplit", unsplit)
    monkeypatch.setattr(
        IF._ix_fold_op,
        "at_rows",
        lambda r, rows: None if rows > r.rows else _runner(f"{r.tag}@{rows}", rows=rows),
    )
    monkeypatch.setattr(
        IF,
        "cute_dsl_qkva_ix_fold",
        lambda x, w, out, r: calls.append(r.tag) or (out, fold_kb, fold_wq),
    )
    monkeypatch.setattr(
        F,
        "cute_dsl_indexer_q_side_fused",
        lambda q, pos, cs, wq, c, q8, w, *, is_neox, compiled: calls.append(
            f"q_side:{compiled.wq_pitch}:wq{wq.stride(0)}"
        ),
    )
    monkeypatch.setattr(
        F, "cute_dsl_indexer_q_side_k_cache", lambda *a, compiled, **k: calls.append(compiled.tag)
    )
    monkeypatch.setattr(
        F,
        "cute_dsl_indexer_k_cache_update",
        lambda kb, *a, compiled, **k: calls.append(f"k_write:{kb.dtype}:{kb.stride(0)}:{compiled}"),
    )
    ph = _step(core, decode, rows)
    ix = SimpleNamespace(fused_bf16=None, kn_w=None, kn_b=None, qb=None)
    ixq = torch.zeros(rows, 32 * 128, dtype=BF16)
    md = SimpleNamespace(slot_mapping_fp8=None, slot_mapping_scale=None)
    pool = torch.zeros(4, 32, 1, 132, dtype=torch.uint8)
    calls.clear()
    if decode:
        got = _qkva_fold(core, ph, x, 0)
        kw = None if got is None else got[1:]
        _, _, k_pre = core.op.indexer.project(
            ph, x, x, None, ix, rows, True, kw=kw, ixq=ixq, kc=(md, pool)
        )
    else:
        _, _, k_pre = core.op.indexer.project(ph, x, x, None, ix, rows, True, ixq=ixq)
    if k_pre is not None:
        core.op.indexer.write_k(md, pool, k_pre, rows)

    wide = _wide(max_draft_len, decode, rows)
    cell = _cell(max_draft_len, decode, rows, bucket)
    if fold and max_draft_len > 1 and cell is not None:
        front = [f"fold@{cell}" if cell == rows else f"fold@{cell}@{rows}"]
    elif rows <= 64:
        front = ["ix_split@64"]
    elif wide and cell == rows:
        front = [f"ix_split@{rows}:mtp"]
    elif wide and bucket and rag:
        front = ["ix_split:rag"]
    else:
        front = ["unsplit:ix_split:prefill" if nosplit else "ix_split:prefill"]
    in_place = front == ["unsplit:ix_split:prefill"]
    if wide and not in_place:
        back = ["qk:True"]
    elif in_place:
        back = ["q_side:160:wq160", "k_write:torch.float32:160:k:dense"]
    else:
        back = ["q_side:32:wq32", "k_write:torch.bfloat16:128:k:dense"]
    assert calls == front + back, (calls, front + back)
    assert core.route_hits.get("ctx_ix_nosplit", 0) == int(in_place)
