# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE-side hops of a decoder layer as megakernel stages over the 8 ranks of a TEP8 group
(``stages/moe_hops.cuh``, plan in ``schedule_moe_hops.py``).

One launch runs, on every rank: the hop1 consumer (the stock ``post_attn_consumer_hop`` arithmetic: rank-ordered sum
of the o_proj partials, post-attention norm, gated residual, pre-FFN norm), the router and pool GEMVs (weights
requested at kernel entry), the F2' all-gather of pool partials and logits, the finalize all-reduce of the experts'
weighted-sum partials, the unpool GEMV and the F1 boundary (all-gather, post-FFN norm, gated residual, next pre-QKV
norm, DFlash capture). The o_proj epilogue and the experts are stubs here (``MhOproj``, ``MhExpStub``): they publish a
given per-rank partial.

``MoeHopWorkspace`` is collective: every rank of the group creates it at the same point of its sequence of multicast
allocations (after every stock MNNVL workspace of the served path). Launches cycle the library workspace's copies
(``copies`` = 3); the stages index this workspace with the same copy.

The F2' contract for the experts stage (``stages/experts.cuh``): in launch copy ``c``, this rank's unicast view holds at
``c * COPY_BYTES + POOL_OFF`` the pool partials ``[2 halves][8 rows][4096]`` FP32 (pooled value = ``bf16(h0 + h1)``) and
at ``c * COPY_BYTES + LOGIT_OFF`` the logits ``[256 experts][8 rows]`` FP32; every word is ``0x80000000`` until it
arrives; any unit may poll any of them any number of times in the launch (``MhClear`` of launch n + 1 empties them
for launch n + 3). The experts publish their BF16 weighted-sum partial ``[8, 4096]`` into slot ``rank`` of every rank's
finalize area (``FIN_OFF``, ``[8 src][8 rows][4096]``) through the multicast address, each element once, words with
``0x80000000`` stored as 0 (``mhExpStub`` shows the code).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from . import library, qkv, schedule_moe_hops

RANKS = 8
HIDDEN = 8192
POOL_WIDTH = 4096
EXPERTS = 256
POST_EPS = 1e-8
PRE_EPS = 1e-5
#: The router's metap multiplier (``(8192 / 256) ** -0.5``).
ROUTER_SCALE = 0.17677669529663687

#: Per-copy layout of the multicast workspace and of the local arena (``moe_hops_layout.cuh``), bytes.
H1_OFF = 0
H1_BYTES = RANKS * 8 * HIDDEN * 2
POOL_OFF = H1_OFF + H1_BYTES
POOL_BYTES = 2 * 8 * POOL_WIDTH * 4
LOGIT_OFF = POOL_OFF + POOL_BYTES
LOGIT_BYTES = EXPERTS * 8 * 4
FIN_OFF = LOGIT_OFF + LOGIT_BYTES
FIN_BYTES = RANKS * 8 * POOL_WIDTH * 2
F1_OFF = FIN_OFF + FIN_BYTES
F1_BYTES = 8 * HIDDEN * 2
COPY_BYTES = F1_OFF + F1_BYTES
AR_X = 0
AR_M = AR_X + 8 * HIDDEN * 2
AR_SSQ = AR_M + 8 * POOL_WIDTH * 2
ARENA_BYTES = AR_SSQ + 4 * 8 * 32 * 4
CLEAR_UNITS = 20
CLEAR_UNIT_BYTES = (ARENA_BYTES + POOL_BYTES + LOGIT_BYTES) // CLEAR_UNITS
# ---- the builds with one-exchange boundaries (``MSO_MK_NB`` != 0) add the slots [2 boundaries][8 rows][32 slots]
# x 16 B after the norm phases, then 128 pad bytes; their arena copy is ARENA_BYTES_NB. ----
AR_ABC = AR_SSQ + 4 * 8 * 32 * 4
ARENA_BYTES_NB = AR_ABC + 2 * 8 * 32 * 16 + 128
CLEAR_UNIT_BYTES_NB = (ARENA_BYTES_NB + POOL_BYTES + LOGIT_BYTES) // CLEAR_UNITS
# ---- end one-exchange boundaries ----
# ---- the builds with two-term consumers at H1 (``MSO_MK_BH`` != 0, on ``MSO_MK_NB`` bit 1) add BF16 [8][8192] term
# planes after the one-exchange slots: rA (bit 2), then the lo planes of rA and xB (bit 4); the copy is padded to 128 B
# with a clear span of whole 16-byte vectors per clear unit (``mhBhArenaBytes``). ----
PLANE_BYTES = 8 * HIDDEN * 2
AR_RA = AR_ABC + 2 * 8 * 32 * 16


def arena_bytes_bh(bits: int) -> int:
    """The arena copy of a build with ``MSO_MK_BH = bits`` (``moe_hops_layout.cuh`` ``kMhArenaBytes``)."""
    planes = 3 if bits & 4 else (1 if bits & 2 else 0)
    a = -(-(AR_RA + planes * PLANE_BYTES) // 128) * 128
    while (a + POOL_BYTES + LOGIT_BYTES) % (CLEAR_UNITS * 16):
        a += 128
    return a


# ---- end two-term consumers ----
# ---- the builds with zero-exchange F1 (``MSO_MK_FZ`` = 1, on ``MSO_MK_NB`` bit 2) add one fz block per copy after the
# room of FZ_MAX_COPIES copies of the multicast workspace (at ``FZ_MAX_COPIES * COPY_BYTES + c * FZ_BYTES``; such a
# workspace holds at most FZ_MAX_COPIES copies): the F1 leaf area [4 reading quarters][8 rows]
# [1024 leaves] x 16 B (S1, A, B, C of the stock thread owning columns [8 leaf, +8), one copy per reading MhF1 quarter),
# then the rank-local r' area [8 rows][1024] FP32 (r' of the rank's unpool columns); every word has one reader per rank,
# which resets it (the copies, the arena and MhClear are unchanged). ----
FZ_LEAF_OFF = 0
FZ_LEAF_BYTES = 4 * 8 * 1024 * 16
FZ_R1_OFF = FZ_LEAF_OFF + FZ_LEAF_BYTES
FZ_R1_BYTES = 8 * (HIDDEN // RANKS) * 4
FZ_BYTES = FZ_R1_OFF + FZ_R1_BYTES
FZ_MAX_COPIES = 8
# ---- end zero-exchange F1 ----


def layout(lib: library.Library) -> List[int]:
    """The C ABI's layout record (``msoMkMhLayout``): offsets, sizes, the FMA mode, the parameter indices."""
    fn = lib.lib.msoMkMhLayout
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p]
    out = (ctypes.c_int64 * 32)()
    n = fn(out)
    return [int(out[i]) for i in range(n)]


def check_layout(lib: library.Library) -> int:
    """Asserts the mirror above; returns the build's FMA mode (``MSO_MK_MH_FMA``)."""
    rec = layout(lib)
    nb = rec[18] if len(rec) > 18 else 0  # MSO_MK_NB
    bh = rec[20] if len(rec) > 20 else 0  # MSO_MK_BH
    fz = rec[22] if len(rec) > 22 else 0  # MSO_MK_FZ
    arena = arena_bytes_bh(bh) if bh else (ARENA_BYTES_NB if nb else ARENA_BYTES)
    want = [
        COPY_BYTES,
        H1_OFF,
        POOL_OFF,
        LOGIT_OFF,
        FIN_OFF,
        F1_OFF,
        arena,
        AR_X,
        AR_M,
        AR_SSQ,
        CLEAR_UNITS,
        (arena + POOL_BYTES + LOGIT_BYTES) // CLEAR_UNITS,
    ]
    assert rec[: len(want)] == want, f"layout mismatch: {rec} vs {want}"
    assert rec[13] == library.PARAM["MhUc"] and rec[14] == library.PARAM["MhOutXattn"], (
        "parameter block mismatch"
    )
    u = rec[15]
    assert rec[16] == schedule_moe_hops.coef_h1_bytes(u) and rec[
        17
    ] == schedule_moe_hops.coef_f1_bytes(u)
    assert len(rec) == (25 if fz else 22 if bh else 20 if nb else 18) and (
        not nb or rec[19] == AR_ABC
    ), f"nb record mismatch: {rec}"
    assert not bh or (nb & 1 and rec[21] == AR_RA), f"bh record mismatch: {rec}"
    assert not fz or (
        nb & 2 and rec[21] == (AR_RA if bh else 0) and rec[23:25] == [FZ_BYTES, FZ_MAX_COPIES]
    ), f"fz record mismatch: {rec}"
    return rec[12]


def row_units(lib: library.Library) -> int:
    """The build's boundary units per row (``MSO_MK_MH_ROW_UNITS``)."""
    return layout(lib)[15]


def nb_bits(lib: library.Library) -> int:
    """The build's one-exchange boundaries (``MSO_MK_NB``: 1 MhH1, 2 MhF1; 0 = two exchanges per boundary)."""
    rec = layout(lib)
    return rec[18] if len(rec) > 18 else 0


def bh_bits(lib: library.Library) -> int:
    """The build's two-term consumer bits at H1 (``MSO_MK_BH``: 1 branch term, 2 residual term, 4 lo planes, 8 pool
    only; 0 = none)."""
    rec = layout(lib)
    return rec[20] if len(rec) > 20 else 0


def fz_bits(lib: library.Library) -> int:
    """The build's zero-exchange F1 (``MSO_MK_FZ``: 1 = MhF1 sums its row's leaves; 0 = the ``MSO_MK_NB`` exchange or
    none)."""
    rec = layout(lib)
    return rec[22] if len(rec) > 22 else 0


def fz_bytes(lib: library.Library) -> int:
    """The build's fz block per copy in bytes (``kMhFzBytes`` with ``MSO_MK_FZ``, else 0)."""
    check_layout(lib)
    rec = layout(lib)
    return rec[23] if len(rec) > 22 and rec[22] else 0


def arena_bytes(lib: library.Library) -> int:
    """The build's local arena copy in bytes (``kMhArenaBytes``: ARENA_BYTES, or ARENA_BYTES_NB with ``MSO_MK_NB``)."""
    check_layout(lib)
    return layout(lib)[6]


def exchange(lib: library.Library) -> Dict[str, int]:
    """The build's row-norm exchange (``MSO_MK_MH_XCHG``: 0 arena, every warp polls; 1 arena, one polling warp; 2
    cluster DSMEM), the cluster dimension its launches use, and whether every F1 unit sums the first norm over the
    whole row itself (``MSO_MK_MH_F1FULL``)."""
    fn = lib.lib.msoMkMhXchg
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p]
    out = (ctypes.c_int * 4)()
    fn(out)
    return dict(xchg=int(out[0]), cluster=int(out[1]), f1full=int(out[2]))


def max_active_clusters(lib: library.Library, dim: int) -> int:
    """``cudaOccupancyMaxActiveClusters`` of the build's megakernel at cluster dimension ``dim`` (the launch needs
    ``lib.ctas // dim``)."""
    fn = lib.lib.msoMkMaxActiveClusters
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
    out = (ctypes.c_int * 1)()
    rc = fn(dim, out)
    assert rc == 0, f"cudaOccupancyMaxActiveClusters({dim}): CUDA error {rc}"
    return int(out[0])


def pack_coef(row_units_: int, *columns: torch.Tensor) -> torch.Tensor:
    """Per boundary unit q (columns ``[q W, +W)``, W = 8192 / U) the bytes of every tensor's slice, in order:
    uint8 ``[U * W * sum(element sizes)]`` -- the unit weights of ``MhH1`` (alpha1, beta1, w_preffn) or ``MhF1``
    (alpha2, beta2, gain_pf, w_next)."""
    w = HIDDEN // row_units_
    parts = []
    for q in range(row_units_):
        for t in columns:
            assert t.numel() == HIDDEN and t.is_contiguous()
            parts.append(t.reshape(-1)[q * w : (q + 1) * w].contiguous().view(torch.uint8))
    return torch.cat(parts).contiguous()


def pack_coef_f1(row_units_: int, alpha2, beta2, gain_pf, w_next) -> torch.Tensor:
    """The ``MhF1`` coefficient blob: the whole row's ``gain_pf`` (BF16 [8192]; the ``MSO_MK_MH_F1FULL`` units take
    it as their second weight source), then ``pack_coef`` of (alpha2, beta2, gain_pf, w_next)."""
    assert gain_pf.dtype == torch.bfloat16 and gain_pf.numel() == HIDDEN and gain_pf.is_contiguous()
    return torch.cat(
        [
            gain_pf.reshape(-1).view(torch.uint8),
            pack_coef(row_units_, alpha2, beta2, gain_pf, w_next),
        ]
    ).contiguous()


class MoeHopWorkspace:
    """The multicast workspace of the hops (``McastGPUBuffer``, ``copies`` x ``COPY_BYTES`` per rank) and the stages'
    local arena (``copies`` x ``ARENA_BYTES``), every word ``0x80000000``. Collective: every rank of the group (MPI
    communicator ``comm``, 8 ranks over 2 nodes) constructs it at the same point.
    """

    def __init__(self, lib: library.Library, comm, rank: int, local_rank: int, copies: int = 3):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert copies == 3, "the F2' / arena clear rule needs three copies"
        assert comm.Get_size() == RANKS, "built for a group of 8"
        self.arena_bytes = arena_bytes(lib)  # the arena copy of the build's ``nb`` level
        self.fz_bytes = fz_bytes(lib)  # the build's ``fz`` block per copy, after the copies
        self.copies = copies
        self.rank = int(rank)
        # The fz blocks follow the room of FZ_MAX_COPIES copies
        total = (
            (FZ_MAX_COPIES * COPY_BYTES + copies * self.fz_bytes)
            if self.fz_bytes
            else copies * COPY_BYTES
        )
        self.total = total
        self.mcast = McastGPUBuffer(total, RANKS, self.rank, int(local_rank), True, comm.py2f())
        self.uc = self.mcast.get_uc_buffer(self.rank, (total // 4,), torch.float32, 0)
        self.mc = self.mcast.get_mc_buffer((total // 4,), torch.float32, 0)
        self.uc.fill_(-0.0)
        dev = self.uc.device
        self.arena = torch.full(
            (copies * self.arena_bytes // 4,), -0.0, dtype=torch.float32, device=dev
        )
        # Every rank's unicast mapping of the same buffer (FLAG_MH_UNICAST publications).
        self.peers = torch.tensor(
            [self.mcast.get_uc_buffer(r, (4,), torch.float32, 0).data_ptr() for r in range(RANKS)],
            dtype=torch.int64,
            device=dev,
        )
        torch.cuda.synchronize()
        comm.Barrier()

    def empty_words(self) -> Dict[str, int]:
        """Words that are not ``0x80000000`` per area of every copy (all zero between launches except the multi-
        reader areas of the last two copies, which the next launches clear)."""
        u = self.uc.view(torch.int32)[: self.copies * COPY_BYTES // 4].view(
            self.copies, COPY_BYTES // 4
        )
        e = -(1 << 31)
        areas = dict(
            hop1=(H1_OFF, H1_BYTES),
            pool=(POOL_OFF, POOL_BYTES),
            logits=(LOGIT_OFF, LOGIT_BYTES),
            fin=(FIN_OFF, FIN_BYTES),
            f1=(F1_OFF, F1_BYTES),
        )
        out = {}
        for c in range(self.copies):
            for name, (off, n) in areas.items():
                out[f"{name}{c}"] = int((u[c, off // 4 : (off + n) // 4] != e).sum())
            a = self.arena.view(torch.int32).view(self.copies, self.arena_bytes // 4)[c]
            out[f"arena{c}"] = int((a != e).sum())
            if self.fz_bytes:  # the copy's fz block (leaves, r')
                z = self.uc.view(torch.int32)[
                    (FZ_MAX_COPIES * COPY_BYTES + c * self.fz_bytes) // 4 :
                ]
                for name, (off, n) in dict(
                    leaf=(FZ_LEAF_OFF, FZ_LEAF_BYTES), r1=(FZ_R1_OFF, FZ_R1_BYTES)
                ).items():
                    out[f"{name}{c}"] = int((z[off // 4 : (off + n) // 4] != e).sum())
        return out


@dataclass
class LayerWeights:
    """One layer's per-rank weights and coefficients for the hop stages.

    ``router`` FP32 ``[32, 8192]`` (rows ``[32 rank, +32)``), ``pool`` BF16 ``[512, 8192]`` (the rank's slice),
    ``unpool`` BF16 ``[1024, 4096]`` (the rank's slice), ``alpha1`` / ``beta1`` / ``alpha2`` / ``beta2`` FP32
    ``[8192]``, ``w_preffn`` / ``w_next`` BF16 ``[8192]`` (Gemma weights, gain ``1 + w``), ``gain_pf`` BF16 ``[8192]``
    (the post-FFN norm's ``w + 1``)."""

    router: torch.Tensor
    pool: torch.Tensor
    unpool: torch.Tensor
    alpha1: torch.Tensor
    beta1: torch.Tensor
    w_preffn: torch.Tensor
    gain_pf: torch.Tensor
    alpha2: torch.Tensor
    beta2: torch.Tensor
    w_next: torch.Tensor
    packed: Dict[str, torch.Tensor] = field(default_factory=dict)

    def pack(self, row_units_: int = schedule_moe_hops.DEFAULT_ROW_UNITS) -> "LayerWeights":
        """The GEMV weights in the kernel's packed order (``qkv.pack_bf16``) and the boundary coefficients per
        boundary unit (``pack_coef``), once per row-unit count."""
        if self.packed.get("row_units") != row_units_:
            assert self.router.dtype == torch.float32 and tuple(self.router.shape) == (32, HIDDEN)
            assert tuple(self.pool.shape) == (512, HIDDEN) and tuple(self.unpool.shape) == (
                1024,
                POOL_WIDTH,
            )
            for t in (self.alpha1, self.beta1, self.alpha2, self.beta2):
                assert t.dtype == torch.float32 and t.numel() == HIDDEN and t.is_contiguous()
            for t in (self.w_preffn, self.gain_pf, self.w_next):
                assert t.dtype == torch.bfloat16 and t.numel() == HIDDEN and t.is_contiguous()
            gemv = (
                self.packed
                if self.packed
                else dict(
                    router=self.router.contiguous(),
                    pool=qkv.pack_bf16(self.pool),
                    unpool=qkv.pack_bf16(self.unpool),
                )
            )
            self.packed = dict(
                router=gemv["router"],
                pool=gemv["pool"],
                unpool=gemv["unpool"],
                coef_h1=pack_coef(row_units_, self.alpha1, self.beta1, self.w_preffn),
                coef_f1=pack_coef_f1(
                    row_units_, self.alpha2, self.beta2, self.gain_pf, self.w_next
                ),
                row_units=row_units_,
            )
        return self


#: Check outputs a layer may write (parameter field -> (shape, dtype)).
OUTPUTS = {
    "MhOutRes1": ((8, HIDDEN), torch.float32),
    "MhOutRes2": ((8, HIDDEN), torch.float32),
    "MhOutXffn": ((8, HIDDEN), torch.bfloat16),
    "MhOutPool": ((8, POOL_WIDTH), torch.bfloat16),
    "MhOutLogits": ((EXPERTS, 8), torch.float32),
    "MhOutM": ((8, POOL_WIDTH), torch.bfloat16),
    "MhOutB": ((8, HIDDEN), torch.bfloat16),
    "MhOutXattn": ((8, HIDDEN), torch.bfloat16),
}


class MoeHopStage:
    """The hop stages of one compile mode on a shared library workspace and hop workspace.

    Args:
        mode: the library's compile mode (``check``, ``lean``, ``check_mhfma0``, ...).
        workspace: the library workspace (copies 3), shared by every mode used in the process so that the copy
            rotation stays one sequence.
        hops: the hop workspace.
    """

    def __init__(
        self,
        mode: str,
        workspace: library.Workspace,
        hops: MoeHopWorkspace,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        self.fma = check_layout(self.lib)
        self.row_units = row_units(self.lib)
        assert workspace.copies == hops.copies == 3
        self.ws = workspace
        self.hops = hops
        self.exchange = exchange(self.lib)
        self.tables = schedule_moe_hops.moe_hops_tables(
            self.lib.ctas, self.row_units, f1_full=bool(self.exchange["f1full"])
        )
        self.mk = library.Megakernel(self.lib, workspace, self.tables, timeout_ms=timeout_ms)

    def make_layer(
        self,
        w: LayerWeights,
        oproj: torch.Tensor,
        expert: torch.Tensor,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        capture: Optional[torch.Tensor] = None,
        stamps: bool = False,
        one_piece: bool = True,
        unicast: bool = False,
    ):
        """One layer's parameter blocks. ``oproj`` BF16 ``[8, 8192]`` / ``expert`` BF16 ``[8, 4096]``: this rank's
        stub inputs (read at every launch); ``outputs``: check outputs by field name (``OUTPUTS``); ``capture``: a
        BF16 ``[8, >= 8192]`` row-strided view for the DFlash capture; ``unicast``: every publication as one store per
        rank's unicast mapping instead of one multicast store."""
        w.pack(self.row_units)
        outputs = outputs or {}
        for name, t in outputs.items():
            shape, dtype = OUTPUTS[name]
            assert tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous(), name
        assert (
            oproj.dtype == torch.bfloat16
            and tuple(oproj.shape) == (8, HIDDEN)
            and oproj.is_contiguous()
        )
        assert expert.dtype == torch.bfloat16 and tuple(expert.shape) == (8, POOL_WIDTH)
        assert expert.is_contiguous()
        fields = dict(
            MhUc=self.hops.uc.data_ptr(),
            MhMc=self.hops.mc.data_ptr(),
            MhArena=self.hops.arena.data_ptr(),
            MhRank=self.hops.rank,
            MhWRouter=w.packed["router"].data_ptr(),
            MhWPool=w.packed["pool"].data_ptr(),
            MhWUnpool=w.packed["unpool"].data_ptr(),
            MhCoefH1=w.packed["coef_h1"].data_ptr(),
            MhCoefF1=w.packed["coef_f1"].data_ptr(),
            MhPeers=self.hops.peers.data_ptr(),
            MhScale=int(torch.tensor(ROUTER_SCALE, dtype=torch.float32).view(torch.int32)),
            MhInOproj=oproj.data_ptr(),
            MhInExpert=expert.data_ptr(),
        )
        if capture is not None:
            assert (
                capture.dtype == torch.bfloat16 and capture.shape[0] == 8 and capture.stride(1) == 1
            )
            assert (
                capture.shape[1] >= HIDDEN
                and capture.data_ptr() % 16 == 0
                and capture.stride(0) % 8 == 0
            )
            fields["MhCapture"] = capture.data_ptr()
            fields["MhCaptureStride"] = capture.stride(0)
        for name, t in outputs.items():
            fields[name] = t.data_ptr()
        flags = (
            library.FLAG_EVICT_FIRST
            | (library.FLAG_ONE_PIECE if one_piece else 0)
            | (library.FLAG_MH_UNICAST if unicast else 0)
        )
        keep = [w.packed[k] for k in ("router", "pool", "unpool", "coef_h1", "coef_f1")]
        keep += [self.hops.peers, oproj, expert, *outputs.values()]
        if capture is not None:
            keep.append(capture)
        return self.mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep)

    def launch(self, layer: Sequence[library.ParamBlock], residual: torch.Tensor, pdl: bool = True):
        """One launch on the current stream; ``residual`` FP32 ``[8, 8192]`` is read and updated in place."""
        assert residual.dtype == torch.float32 and tuple(residual.shape) == (8, HIDDEN)
        assert residual.is_contiguous()
        return self.mk.launch(layer, library.KernelIo(res=residual), pdl=pdl)
