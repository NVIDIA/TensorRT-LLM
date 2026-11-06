# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``rl``: the router logits by linearity through o_proj (no pool composites).

The router logits of a MoE layer are linear in the residual ``r`` and the o_proj partials ``o_r``:
``l = mu * s2 * (W_R (gA * r) + s1 * sum_r D_r o_r)`` with ``D_r = W_R diag(gB) W_o,r`` (``gA = g * alpha``,
``gB = g * beta``, ``g = 1 + w_preffn``, ``s1`` / ``s2`` the two norm scalars of the hop-1 boundary). With both terms
computed before the hop-1 boundary, the logits are complete at the end of ``MhH1``: the router GEMV and the logit hop
leave the chain and the routing / placement / cut overlap the pool GEMV, the pool hop and the quantization.

**Build** (``IFLAG_RL_ON``): the plan does not change. ``MhRouter``'s 64 row-split units become rl units ``i`` (experts
``[4 i, +4)``, all 8 rows) on the same CTAs, regions and issue points; each holds a 32 KiB FP32 slab (``slabs``): the A
part ``W_R diag(gA)`` over the rank's residual columns ``[1024 rank, +1024)`` and the D part ``D_r``, both made in FP64
and rounded to FP32 once. The unit runs after the grid-dependency wait (its residual) and the attention half (its
o_proj input), multiplies in FP32 on the CUDA cores and multicasts the rank's A / B partials into every rank's
workspace (``RlWorkspace``). ``MhH1`` sums the 8 ranks' partials in rank order and forms the logits with its own
``s1``, ``s2`` into its rank's F2' logit area. The experts stage then runs the routing before the quantization
(``EXPERTS_CLEAR`` / ``EXPERTS_OR``: ``MOE_FLAG_ISSUE_AT_CUT``). Precision: FP32 weights (no precision reduction), FP32
products and sums in a fixed order; the residual stream is unchanged, only the router input differs from stock's (it
skips stock's BF16 roundings of the o_proj sum, the gated residual and x_ffn).

**Attribution** (``oracle_plan`` + ``IFLAG_RL_ORACLE_LG``): the ``MhRouter`` units are dropped from the plan
and each ``MhH1`` unit stores its slice of the logits (recorded by a run with ``IFLAG_RL_RECORD`` on identical inputs)
right after its boundary arithmetic. ``IFLAG_RL_ORACLE_POOL`` (``pool=True`` in ``oracle_plan``) also stores the pool
words: the full oracle. Not used in serving.

**Check** (``IFLAG_RL_RECORD`` with a record buffer of ``REC_BYTES`` per layer in the rl block): the layer's F2' areas,
residual and gated o_proj input, for ``reference`` / ``top8_stats``.

**Build modes** (``library.rl_level``, ``msoMegakernel.cu`` ``MSO_MK_RL``): none of this code is compiled without the
define. Bit 1 the build, bit 2 the record, bit 4 the oracle, bit 8 (with the ``MSO_MK_RP`` publisher): the rank's logits
formed once by the experts stage's publisher CTA from the partials and MhH1's own norm scalars (the ``MSO_MK_NB`` one
exchange), so MhH1 never waits for the partials. The layer's pointers reach the kernel through one parameter field,
``Rl``, the address of the layer's rl block (``make_block``: record / oracle buffer, workspace multicast and unicast,
weight slabs).

NUMERICS RULE: the routing input differs from stock's (FP32 composites and sums, no BF16 roundings of the
o_proj sum, the gated residual and x_ffn), so near-tied top-8 choices can differ from stock's. Served switch:
``serving.RL_ENV_VAR`` (unset: the build's rl, on for the served build). The switch is kept because the build is not
bitwise the stock layers.

Pure Python (no GPU) except ``check_layout``, ``RlWorkspace``, ``slabs``, ``make_block``, ``set_fields``, ``apply``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from . import schedule

#: ``kPIFlags`` bits (``msoMegakernel.cu`` kIFlagRl*; bits 52-55).
IFLAG_RL_ORACLE_LG = 1 << 52
IFLAG_RL_RECORD = 1 << 53
IFLAG_RL_ORACLE_POOL = 1 << 54
IFLAG_RL_ON = 1 << 55

RANKS = 8
HIDDEN = 8192
POOL_WIDTH = 4096
EXPERTS = 256
O_COLS = 1024
#: rl units (``stages/moe_hops_layout.cuh`` kRl*): MhRouter's 64 row-split units, 4 experts each.
UNITS = 64
EXPERTS_PER_UNIT = EXPERTS // UNITS
PART_BYTES = O_COLS * EXPERTS_PER_UNIT * 4
UNIT_BYTES = 2 * PART_BYTES
TERM_BYTES = RANKS * 8 * EXPERTS * 4
COPY_BYTES = 2 * TERM_BYTES
#: One layer's recorded F2' areas: pool [2][8][4096] FP32, logits [256][8] FP32, in the hop workspace's layout; then
#: (the record) the residual r FP32 [8][8192] and this rank's gated o_proj input BF16 [8][1024].
POOL_BYTES = 2 * 8 * POOL_WIDTH * 4
LOGIT_BYTES = EXPERTS * 8 * 4
ORACLE_BYTES = POOL_BYTES + LOGIT_BYTES
REC_RES_OFF = ORACLE_BYTES
REC_O_OFF = REC_RES_OFF + 8 * HIDDEN * 4
REC_BYTES = REC_O_OFF + 8 * O_COLS * 2
#: Boundary units (``MhH1`` / ``MhF1``) per row the slices assume (``kRlOk``).
ROW_UNITS = 4

Plan = schedule.Plan


#: The experts stage front orders of an rl layer (``experts_order``).
ORDERS = ("iac", "lpre", "qf")


def experts_order(order: str = "iac") -> Tuple[int, int]:
    """(experts flag bits cleared, bits set) of an rl layer's experts stage front.

    ``iac``: the routing, placement and cut first, the early weight requests at the cut, then the quantization (the
    logits are complete at H1's end, the pool output later); ``lpre``: the served QuantFirst order with the logit
    vectors requested inside shareQuant (``MOE_FLAG_W2_LOGITS_PRE``); ``qf``: the served order unchanged.
    """
    from . import experts

    assert order in ORDERS, order
    if order == "iac":
        return (
            experts.MOE_FLAG_QUANT_FIRST | experts.MOE_FLAG_W2_LOGITS_PRE,
            experts.MOE_FLAG_ISSUE_AT_CUT,
        )
    if order == "lpre":
        return 0, experts.MOE_FLAG_W2_LOGITS_PRE
    return 0, 0


#: The rl block (``msoMegakernel.cu`` kRlBlock*): int64 words.
BLOCK_REC, BLOCK_MC, BLOCK_UC, BLOCK_W = 0, 1, 2, 3
BLOCK_WORDS = 4


def check_layout(lib) -> int:
    """Asserts the mirror above against the build (``msoMkRlLayout``); returns the build's ``MSO_MK_RL`` mask."""
    import ctypes

    from . import library

    fn = lib.lib.msoMkRlLayout
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p]
    out = (ctypes.c_int64 * 32)()
    n = fn(out)
    rec = [int(out[i]) for i in range(n)]
    want = [
        UNITS,
        UNIT_BYTES,
        TERM_BYTES,
        COPY_BYTES,
        REC_BYTES,
        REC_RES_OFF,
        REC_O_OFF,
        ORACLE_BYTES,
        library.PARAM["Rl"],
        BLOCK_WORDS,
        IFLAG_RL_ORACLE_LG,
        IFLAG_RL_RECORD,
        IFLAG_RL_ORACLE_POOL,
        IFLAG_RL_ON,
        1,
    ]
    assert rec[:-1] == want, f"[rl] layout mismatch: {rec[:-1]} vs {want}"
    return rec[-1]


# ---------------------------------------------------------------------------------------------------------------------
# Plans (attribution)
# ---------------------------------------------------------------------------------------------------------------------
def oracle_plan(plan: Plan, pool: bool = False) -> Plan:
    """``plan`` without its ``MhRouter`` units (and its ``MhPool`` units with ``pool``): ``MhH1`` stores their
    outputs."""
    drop = ("MhRouter", "MhPool") if pool else ("MhRouter",)
    return [[r for r in units if r[0] not in drop] for units in plan]


def rl_tables(
    tables: schedule.Tables,
    base: Callable[[str, int], int],
    dynamic: Dict[str, Tuple[int, int]],
    pool: bool = False,
) -> schedule.Tables:
    """``tables``' plan with the oracle edit, placed again."""
    return schedule.build_tables(
        oracle_plan([list(u) for u in tables.plan], pool), base, dynamic=dynamic
    )


def moved_units(old: schedule.Tables, new: schedule.Tables) -> List[Tuple[int, str, int]]:
    """Units present in both tables whose region offset or issue unit changed (``(cta, stage, unit)``); a unit's
    issue point is compared as the ``(stage, unit)`` it waits for (``ISSUE_AT_ENTRY`` stays itself)."""

    def key(t: schedule.Tables, c: int) -> Dict[Tuple[str, int], Tuple[int, object]]:
        out = {}
        for i, su in enumerate(t.plan[c]):
            iss = int(t.issue[c, i])
            at = tuple(t.plan[c][iss]) if 0 <= iss < len(t.plan[c]) else iss
            out[tuple(su)] = (int(t.smem[c, i]), at)
        return out

    moved = []
    for c in range(old.ctas):
        a, b = key(old, c), key(new, c)
        for su, v in b.items():
            if su in a and a[su] != v:
                moved.append((c, su[0], su[1]))
    return moved


def boundary_slices(
    rows: int = 8, units: int = ROW_UNITS, threads: int = 256
) -> Tuple[List[int], List[int]]:
    """Byte offsets (in the oracle record) of every 4-byte word the boundary units store: (pool words, logit words),
    with ``stages/moe_hops.cuh`` ``rlPoolOff`` / ``rlLogitOff``'s formulas."""
    pool, logit = [], []
    for row in range(rows):
        for q in range(units):
            for t in range(threads):
                po = ((t >> 7) * 8 + row) * 4096 + 1024 * q + 8 * (t & 127)
                pool += [4 * (po + k) for k in range(8)]
                if t < 64:
                    logit.append(POOL_BYTES + ((64 * q + t) * 8 + row) * 4)
    return pool, logit


def rl_unit_of(router_unit: int) -> int:
    """The rl unit of a row-split ``MhRouter`` unit code (``stages/moe_hops.cuh`` ``rlUnitOf``)."""
    return (router_unit & 31) + 32 * ((router_unit >> 5) & 1)


def partial_slices(
    rows: int = 8, units: int = ROW_UNITS, threads: int = 256
) -> List[Tuple[int, int, int, int]]:
    """(term, source rank, row, first expert) of every 16-byte vector the boundary units read (``rlPartAddr``)."""
    out = []
    for row in range(rows):
        for q in range(units):
            for t in range(threads):
                out.append((t >> 7, (t >> 4) & 7, row, 64 * q + 4 * (t & 15)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------------------------------
def composites(
    w_o: torch.Tensor,
    w_router: torch.Tensor,
    w_preffn: torch.Tensor,
    alpha1: torch.Tensor,
    beta1: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``(A, D)`` ``[256, 1024]`` of this rank: ``A = W_R diag(g alpha1)[:, 1024 rank : +1024]``,
    ``D = W_R diag(g beta1) W_o,r``, made in FP64 (``g = float(w_preffn) + 1``) and rounded once. ``w_o``: this rank's
    o_proj shard BF16 ``[8192, 1024]``; ``w_router`` FP32 ``[256, 8192]`` (full)."""
    assert tuple(w_o.shape) == (HIDDEN, O_COLS), tuple(w_o.shape)
    assert tuple(w_router.shape) == (EXPERTS, HIDDEN), tuple(w_router.shape)
    g = w_preffn.float().double() + 1.0
    wr = w_router.double()
    ga = g * alpha1.double()
    gb = g * beta1.double()
    c0 = O_COLS * int(rank)
    a = wr[:, c0 : c0 + O_COLS] * ga[c0 : c0 + O_COLS].view(1, -1)
    d = (wr * gb.view(1, -1)) @ w_o.double()
    return a.float().contiguous(), d.float().contiguous()


def pack(m: torch.Tensor) -> torch.Tensor:
    """``m`` FP32 ``[256, 1024]`` as the units' part layout ``[64 units][4 j][256 threads][4]``: thread
    ``t = 4 ks + e`` of unit ``i`` holds columns ``16 ks + 4 j .. + 3`` of expert ``4 i + e``."""
    assert tuple(m.shape) == (EXPERTS, O_COLS)
    v = m.reshape(UNITS, EXPERTS_PER_UNIT, O_COLS // 16, 4, 4)  # [i][e][ks][j][c]
    return v.permute(0, 3, 2, 1, 4).reshape(UNITS, 4, 256, 4).contiguous()


def slabs(a: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The rl units' weight slabs FP32 ``[64][2][4][256][4]`` (A part, then D part; ``UNIT_BYTES`` each)."""
    return torch.stack([pack(a), pack(d)], dim=1).contiguous()


class LayerSlabs:
    """One layer's rl weight slabs on this rank (device)."""

    def __init__(self, layer, rank: int):
        a1, b1 = layer.add_attn.coefficients()
        w_pre = layer.pre_ffn_norm.weight.data.to(torch.bfloat16)
        router = layer.feed_forward.router.weight.data.float()
        assert router.shape[0] == EXPERTS, "[rl] needs the full router"
        a, d = composites(layer.self_attn.o_proj.weight.data, router, w_pre, a1, b1, rank)
        self.slab = slabs(a, d)
        assert self.slab.numel() * 4 == UNITS * UNIT_BYTES


# ---------------------------------------------------------------------------------------------------------------------
# Workspace and parameter fields
# ---------------------------------------------------------------------------------------------------------------------
class RlWorkspace:
    """The rl multicast workspace (``copies`` x ``COPY_BYTES``), every word ``0x80000000``. Collective over the TEP
    group (MPI communicator)."""

    def __init__(self, comm, rank: int, local_rank: int, copies: int):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert comm.Get_size() == RANKS
        self.copies = int(copies)
        total = self.copies * COPY_BYTES
        self.mcast = McastGPUBuffer(total, RANKS, int(rank), int(local_rank), True, comm.py2f())
        self.uc = self.mcast.get_uc_buffer(int(rank), (total // 4,), torch.float32, 0)
        self.mc = self.mcast.get_mc_buffer((total // 4,), torch.float32, 0)
        self.uc.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()

    def reset(self) -> None:
        self.uc.fill_(-0.0)


def make_block(
    device, rec_ptr: int = 0, ws: "RlWorkspace" = None, slab: "LayerSlabs" = None
) -> torch.Tensor:
    """A layer's rl block (int64 [BLOCK_WORDS] on ``device``); the caller keeps it alive while the layer runs."""
    words = [0] * BLOCK_WORDS
    words[BLOCK_REC] = int(rec_ptr)
    if ws is not None:
        words[BLOCK_MC] = ws.mc.data_ptr()
        words[BLOCK_UC] = ws.uc.data_ptr()
    if slab is not None:
        words[BLOCK_W] = slab.slab.data_ptr()
    return torch.tensor(words, dtype=torch.int64, device=device)


def layer_fields(ws: RlWorkspace, slab: "LayerSlabs", rec_ptr: int = 0) -> Dict[str, int]:
    """The ``Rl`` parameter field of an rl layer: its rl block, made once per record buffer and kept on ``slab`` (every
    binding's parameter blocks of the layer point at the same block, so none outlives it)."""
    blocks = slab.__dict__.setdefault("blocks", {})
    if int(rec_ptr) not in blocks:
        blocks[int(rec_ptr)] = make_block(slab.slab.device, rec_ptr, ws, slab)
    return dict(Rl=blocks[int(rec_ptr)].data_ptr())


def set_fields(blocks, iflags: int, block_ptr: int) -> None:
    """OR ``iflags`` into the layer's ``IFlags`` and set ``Rl`` (an rl block's address) in every block (eager)."""
    from . import library

    for block in blocks:
        if block is None:
            continue
        t = block.tensor
        t[library.PARAM["IFlags"]] = int(t[library.PARAM["IFlags"]].item()) | int(iflags)
        t[library.PARAM["Rl"]] = int(block_ptr)


def apply(mk, blocks, fields: Dict[str, int], iflags: int = 0) -> None:
    """Write ``fields`` into every block of a layer made by ``mk`` (a ``library.Megakernel``), OR ``iflags`` into
    ``IFlags``, and fill the blocks' weight tables again (the rl units' rows depend on ``IFlags`` and the rl block's
    weight slabs). Eager."""
    from . import library

    lib = mk.lib
    n = len(library.PARAM_FIELDS) + lib.table_words
    for block in blocks:
        if block is None:
            continue
        t = block.tensor
        for name, v in fields.items():
            t[library.PARAM[name]] = int(v)
        if iflags:
            t[library.PARAM["IFlags"]] = int(t[library.PARAM["IFlags"]].item()) | int(iflags)
        rc = lib.lib.msoMkWeightTable(
            t.data_ptr(), t[n:].data_ptr(), torch.cuda.current_stream().cuda_stream
        )
        assert rc == 0, f"msoMkWeightTable: CUDA error {rc}"
    torch.cuda.current_stream().synchronize()


# ---------------------------------------------------------------------------------------------------------------------
# Check: the record against an FP64 reference
# ---------------------------------------------------------------------------------------------------------------------
def unpack_record(rec: torch.Tensor) -> Dict[str, torch.Tensor]:
    """A layer's record (any dtype, ``REC_BYTES``): logits FP32 [8][256], r FP32 [8][8192], o BF16 [8][1024]."""
    b = rec.view(torch.uint8).view(-1)
    logits = b[POOL_BYTES:ORACLE_BYTES].view(torch.float32).view(EXPERTS, 8).t()
    r = b[REC_RES_OFF:REC_O_OFF].view(torch.float32).view(8, HIDDEN)
    o = b[REC_O_OFF:REC_BYTES].view(torch.bfloat16).view(8, O_COLS)
    return dict(logits=logits.contiguous(), r=r, o=o)


def reference(
    r: torch.Tensor, a_exact: torch.Tensor, alpha1, beta1, w_preffn, w_router, mu: float
) -> torch.Tensor:
    """The FP64 router logits [8, 256] of a layer from its residual ``r`` [8, 8192] and the exact all-reduced o_proj
    output ``a_exact`` [8, 8192] (sum over ranks of ``W_o,r o_r`` in FP64)."""
    r = r.double()
    a = a_exact.double()
    s1 = torch.rsqrt(a.pow(2).mean(dim=1, keepdim=True) + 1e-8)
    rp = alpha1.double() * r + beta1.double() * (s1 * a)
    s2 = torch.rsqrt(rp.pow(2).mean(dim=1, keepdim=True) + 1e-5)
    x = s2 * rp * (w_preffn.double() + 1.0)
    return mu * (x @ w_router.double().t())


def top8_stats(logits: torch.Tensor, ref: torch.Tensor, bias: torch.Tensor) -> Dict[str, object]:
    """Routing agreement of ``logits`` with ``ref`` (both [rows, 256]) under the stock selection (sigmoid + bias,
    top 8): rows whose top-8 sets agree, and the reference's 8th-9th score gap on the rows that differ."""

    def sel(lg):
        s = torch.sigmoid(lg.double()) + bias.double()
        return s, torch.topk(s, 8, dim=1).indices.sort(dim=1).values

    sr, ir = sel(ref)
    _, il = sel(logits)
    eq = (ir == il).all(dim=1)
    srt = sr.sort(dim=1, descending=True).values
    gap = srt[:, 7] - srt[:, 8]
    return dict(
        rows=int(eq.numel()),
        top8_equal=int(eq.sum()),
        gap89_differing=[float(g) for g in gap[~eq].tolist()],
    )


def topk_sets(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The stock selection's sorted top-8 expert ids [rows, 8]."""
    s = torch.sigmoid(logits.double()) + bias.double()
    return torch.topk(s, 8, dim=1).indices.sort(dim=1).values


def flips(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> int:
    """Rows whose top-8 sets differ between logits ``a`` and ``b``."""
    return int((topk_sets(a, bias) != topk_sets(b, bias)).any(dim=1).sum())


def check_record_empty(words: torch.Tensor) -> int:
    """F2' words of a record buffer [..., REC_BYTES / 4] still ``kEmpty`` (the oracle part only)."""
    return int(
        (words[..., : ORACLE_BYTES // 4].contiguous().view(torch.int32) == -(2**31)).sum().item()
    )
