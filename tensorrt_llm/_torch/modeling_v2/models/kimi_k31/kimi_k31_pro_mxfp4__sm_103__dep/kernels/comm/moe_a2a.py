# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 expert-parallel MoE dispatch over NVLink peer mappings (CuTe DSL, SM100 / SM103): its geometry
(``MoeA2ADispatchKernel``; ``kernels/moe/ep_moe_fused.EpDispatchKernel`` is the kernel), and the PTX helpers and
workspace flag layout the fused EP MoE (``kernels/moe/ep_moe_fused``) and the attention-DP rows exchange share.

The ``W`` ranks of the EP group own ``num_experts / W`` consecutive experts each. The dispatch pushes 16-byte vectors
straight into a peer's region through the peer's unicast mapping of one symmetric workspace (``McastGPUBuffer``) and
publishes with a system-scope fence followed by a relaxed flag: token ``t`` of rank ``s`` goes to every rank that
owns one of its ``top_k`` experts (each such rank once) as row ``s * T + t`` of that rank's receive arrays (``T`` =
the step's largest per-rank token count): the activation row, its scale row, the expert ids and the routing weights,
bit for bit. Every other rank gets an expert-id row of ``-1`` at that position (so are the rows of padding tokens
``t >= num_tokens``): receive rows hold either a token's data or ``-1`` ids, which the MoE skips. Each CTA counts
itself done; the grid's last CTA fences at system scope, sets this rank's flag on every peer to the call count and
waits for every peer's flag, so the kernel ends with every row in place.

Per call, the kernel reads its call count from the workspace; flags carry the count, so they never need resetting.
Calls must be made in the same order on every rank with the same ``T``.

`MoeA2AWorkspace` is one EP group's symmetric workspace (a `McastGPUBuffer` of the MNNVL workspace helpers): flags,
the dispatch receive arrays ``[W * T_max, ...]`` (activation, scales, expert ids, routing weights) and the slot-row
region ``[W, T_max, out_bytes]`` the fused EP MoE pushes its expert output rows into; flags plus ``W x T_max x (row
bytes + out_bytes)`` per rank.
"""

from __future__ import annotations

from typing import Dict

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm.mapping import Mapping

#: Largest EP group the flag layout holds.
MAX_WORLD = 32
#: Token chunks (of ``chunk`` tokens) the combine flags hold per sender.
MAX_CHUNKS = 1024
#: Flag words (uint32, hot words on their own 128-byte lines): the dispatch / chunked-combine call counts, the CTA
#: count and work queue of the chunked combine, the dispatch flags (one per sender), the flat-combine call count,
#: the dispatch / flat-combine CTA counts, the flat-combine flags (one per sender) and the chunked-combine
#: (sender, chunk) flags.
DEPOCH, CEPOCH, COUNT, ITEMQ, DFLAGS, FEPOCH, DCOUNT, FCOUNT, FFLAGS, CFLAGS = (
    0,
    32,
    64,
    96,
    128,
    192,
    224,
    256,
    512,
    1024,
)
#: Bytes reserved for the flags.
FLAGS_BYTES = 256 * 1024
#: Most distinct destination ranks per token the combine sums (the engine tree's two groups of eight).
MAX_SOURCES = 16


# ---------------------------------------------------------------------------
# Inline PTX
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _lv(v, *, loc=None, ip=None):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _iv(v, *, loc=None, ip=None):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _v4(text, addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        text,
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    """16 bytes of local memory no kernel writes concurrently."""
    return _v4("ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", addr, loc=loc, ip=ip)


@dsl_user_op
def ld_cg_v4(addr, *, loc=None, ip=None):
    """16 bytes a peer wrote (or a peer's memory), read past L1."""
    return _v4("ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", addr, loc=loc, ip=ip)


@dsl_user_op
def st_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_nc_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_acquire_sys_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_relaxed_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_acq_rel_gpu(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_relaxed_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


def _f32_pair(res, loc, ip):
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


_UNPACK = "".join(
    f"shl.b32 t, ${2 + i}, 16;\nmov.b32 l{i}, t;\nand.b32 t, ${2 + i}, 0xffff0000;\nmov.b32 h{i}, t;\n"
    for i in range(8)
)


@dsl_user_op
def tree8_bf16x2(w0, w1, w2, w3, w4, w5, w6, w7, *, loc=None, ip=None):
    """``((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))`` in FP32 (round to nearest) for the low and the high
    BF16 halves of eight words."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3, w4, w5, w6, w7)],
        "{\n.reg .b32 t;\n.reg .f32 l<8>, h<8>, p<4>, q<4>;\n"
        + _UNPACK
        + "add.rn.f32 p0, l0, l1;\nadd.rn.f32 p1, l2, l3;\nadd.rn.f32 p2, l4, l5;\nadd.rn.f32 p3, l6, l7;\n"
        "add.rn.f32 p0, p0, p1;\nadd.rn.f32 p2, p2, p3;\nadd.rn.f32 $0, p0, p2;\n"
        "add.rn.f32 q0, h0, h1;\nadd.rn.f32 q1, h2, h3;\nadd.rn.f32 q2, h4, h5;\nadd.rn.f32 q3, h6, h7;\n"
        "add.rn.f32 q0, q0, q1;\nadd.rn.f32 q2, q2, q3;\nadd.rn.f32 $1, q0, q2;\n}",
        "=f,=f,r,r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return _f32_pair(res, loc, ip)


@dsl_user_op
def add2_bf16x2(w0, w1, *, loc=None, ip=None):
    """``x0 + x1`` in FP32 for the low and the high BF16 halves of two words."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(w0, loc=loc, ip=ip), _iv(w1, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\n.reg .f32 l<2>, h<2>;\n"
        + "".join(
            f"shl.b32 t, ${2 + i}, 16;\nmov.b32 l{i}, t;\nand.b32 t, ${2 + i}, 0xffff0000;\nmov.b32 h{i}, t;\n"
            for i in range(2)
        )
        + "add.rn.f32 $0, l0, l1;\nadd.rn.f32 $1, h0, h1;\n}",
        "=f,=f,r,r",
        loc=loc,
        ip=ip,
    )
    return _f32_pair(res, loc, ip)


@dsl_user_op
def add_f32(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _rank_mask(
    row: Int64,
    vecs: cutlass.Constexpr,
    num_experts: cutlass.Constexpr,
    experts_per_rank: cutlass.Constexpr,
):
    """Bit ``d`` set when one of the expert ids at ``row`` (``vecs`` 16-byte vectors) lives on rank ``d``."""
    mask = Int32(0)
    for q in cutlass.range_constexpr(vecs):
        w = ld_v4(row + q * 16)
        for k in cutlass.range_constexpr(4):
            s = w[k]
            if s >= 0:
                if s < num_experts:
                    mask = mask | (Int32(1) << (s // experts_per_rank))
    return mask


@cute.jit
def _arrive(
    base: Int64,
    peers: Int64,
    rank: Int32,
    epoch: Int32,
    nblk: Int32,
    tid: Int32,
    sitem: cute.Tensor,
    count_word: cutlass.Constexpr,
    flag_word: cutlass.Constexpr,
    world: cutlass.Constexpr,
):
    """Count this CTA as done (its stores ordered before the count); the grid's last CTA fences at system scope,
    sets word ``flag_word + rank`` of every rank to ``epoch`` and re-arms the count. Returns the CTA's count
    ticket (``nblk - 1`` for the last)."""
    cute.arch.barrier()
    if tid == 0:
        sitem[0] = atom_add_acq_rel_gpu(base + Int64(count_word * 4), Int32(1))
    cute.arch.barrier()
    ticket = sitem[0]
    if ticket == nblk - 1:
        if tid < world:
            fence_acq_rel_sys()
            apeer = ld_nc_u64(peers + Int64(tid) * 8)
            st_relaxed_u32(apeer + Int64(flag_word * 4) + Int64(rank) * 4, epoch)
        if tid == 0:
            st_relaxed_u32(base + Int64(count_word * 4), Int32(0))
    return ticket


@cute.jit
def _wait_flags(
    base: Int64, epoch: Int32, tid: Int32, flag_word: cutlass.Constexpr, world: cutlass.Constexpr
):
    """Threads ``0 .. world - 1`` wait until every rank's flag reads ``epoch``; then the CTA syncs."""
    if tid < world:
        wflag = base + Int64(flag_word * 4) + Int64(tid) * 4
        wgot = ld_acquire_sys_u32(wflag)
        while wgot != epoch:
            wgot = ld_acquire_sys_u32(wflag)
    cute.arch.barrier()


class MoeA2ADispatchKernel:
    """Geometry of the push dispatch of ``[num_tokens, ...]`` rows (activation, optional scales, expert ids,
    optional weights); ``kernels/moe/ep_moe_fused.EpDispatchKernel`` is the kernel.

    Warp items ``(token, group)``: group ``g`` of ``groups`` sends the token to destinations
    ``(rank + 1 + g + k * groups) % W``; the warp loads the rows once and stores them to each destination that owns
    one of the token's experts (an id row of ``-1`` to the others).
    """

    def __init__(
        self,
        world: int,
        top_k: int,
        num_experts: int,
        hidden_bytes: int,
        sf_bytes: int,
        weight_bytes: int,
        groups: int = 1,
        threads: int = 512,
    ):
        if not 2 <= world <= MAX_WORLD or num_experts % world:
            raise ValueError(f"world {world} / experts {num_experts} not served")
        if world % groups:
            raise ValueError(f"groups {groups} must divide world {world}")
        for b in (hidden_bytes, sf_bytes, weight_bytes, top_k * 4):
            if b % 16:
                raise ValueError(f"row bytes {b} not a multiple of 16")
        self.W, self.G, self.threads = world, groups, threads
        self.NE, self.EPR = num_experts, num_experts // world
        self.HB, self.SB, self.IB, self.FB = hidden_bytes, sf_bytes, top_k * 4, weight_bytes
        self.HV, self.SV, self.IV, self.FV = (b // 16 for b in (self.HB, self.SB, self.IB, self.FB))
        if self.SV + self.IV + self.FV > 32:
            raise ValueError("scale, id and weight rows must fit one vector per lane")
        self.HJ = (self.HV + 31) // 32


# ---------------------------------------------------------------------------
# Workspace: one EP group's symmetric buffer (flags, dispatch receive arrays, the fused EP MoE's slot rows)
# ---------------------------------------------------------------------------


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class MoeA2AWorkspace:
    """Symmetric workspace of one EP group (the TP group under attention DP)."""

    def __init__(
        self,
        mapping: Mapping,
        max_tokens_per_rank: int,
        hidden_bytes: int,
        sf_bytes: int,
        top_k: int,
        weight_bytes: int,
        out_bytes: int,
    ):
        world = mapping.moe_ep_size
        if (
            mapping.tp_size != world
            or mapping.moe_tp_size != 1
            or mapping.moe_ep_rank != mapping.tp_rank
            or world > MAX_WORLD
        ):
            raise ValueError("the all-to-all kernels serve pure EP over the TP group")
        self.mapping = mapping
        self.world = world
        self.rank = mapping.tp_rank
        self.max_tokens = max_tokens_per_rank
        self.row_bytes = (hidden_bytes, sf_bytes, top_k * 4, weight_bytes)
        self.out_bytes = out_bytes
        rows = world * max_tokens_per_rank
        offs = [FLAGS_BYTES]
        for b in self.row_bytes + (out_bytes,):
            offs.append(offs[-1] + _round_up(rows * b, 1 << 16))
        self.offsets = offs[:5]
        self.total_bytes = offs[5]
        comm = _dist_ops._get_mnnvl_workspace_comm(mapping)
        self.handle = _dist_ops._make_mnnvl_mcast_buffer(
            comm, self.total_bytes, mapping, mapping.is_multi_node()
        )
        device = torch.device("cuda", _dist_ops._mnnvl_device_index(mapping))
        self.local = self.handle.get_uc_buffer(self.rank, (self.total_bytes,), torch.uint8, 0)
        self.base = self.local.data_ptr()
        peers = [
            self.handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)
        ]
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        self.offs = torch.tensor(self.offsets, dtype=torch.int64, device=device)
        self.local[:FLAGS_BYTES].zero_()
        torch.cuda.synchronize()
        if not _dist_ops._mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("all-to-all workspace setup failed on a rank")
        self._comm = comm
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    def recv(self, i: int, tmax: int, dtype: torch.dtype, width: int) -> torch.Tensor:
        """Receive array ``i`` (0 activation, 1 scales, 2 ids, 3 weights) as ``[W * tmax, width]``."""
        rows = self.world * tmax
        start = self.offsets[i]
        nbytes = rows * width * torch.empty((), dtype=dtype).element_size()
        return self.local[start : start + nbytes].view(dtype).view(rows, width)


_WORKSPACES: Dict[tuple, MoeA2AWorkspace] = {}


def push_all_to_all_serves(mapping: Mapping, num_experts: int, top_k: int) -> bool:
    """Whether the kernels serve this EP group: pure EP over the attention-DP TP group of 2..32 ranks, experts
    split evenly, at most 16 destination ranks per token, on SM100 / SM103."""
    world = mapping.moe_ep_size
    return (
        mapping.enable_attention_dp
        and mapping.moe_tp_size == 1
        and mapping.tp_size == world
        and mapping.moe_ep_rank == mapping.tp_rank
        and 2 <= world <= MAX_WORLD
        and num_experts % world == 0
        and min(world, top_k) <= MAX_SOURCES
        and top_k % 4 == 0
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


def get_workspace(mapping: Mapping, max_tokens_per_rank: int, **row_bytes) -> MoeA2AWorkspace:
    """The group's workspace for these row sizes, allocated on first use (collective)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank, tuple(sorted(row_bytes.items())))
    ws = _WORKSPACES.get(key)
    if ws is None or ws.max_tokens < max_tokens_per_rank:
        ws = MoeA2AWorkspace(mapping, max_tokens_per_rank, **row_bytes)
        _WORKSPACES[key] = ws
    return ws


__all__ = [
    "FLAGS_BYTES",
    "MAX_CHUNKS",
    "MAX_SOURCES",
    "MAX_WORLD",
    "MoeA2ADispatchKernel",
    "MoeA2AWorkspace",
    "get_workspace",
    "push_all_to_all_serves",
]
