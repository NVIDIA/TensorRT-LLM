# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The megakernel's MoE experts stage on one rank (``stages/experts.cuh``): from the router logits and the gathered pool
output of the 8 verify rows to this rank's ``[8, 4096]`` BF16 partial of the expert output (the finalize all-reduce's
input).

Input contract (the producer, the F2' hop of the whole-layer launch, writes both as Lamport words: a word that equals
``0x80000000`` has not arrived; a payload equal to it is stored as ``+0``):

* ``logits`` FP32 ``[8, 256]``: all router logits of each row (``x_ffn @ W_r^T * metap``; rank ``r``'s box is columns
  ``[32 r, 32 r + 32)``);
* ``pool`` BF16 ``[8, 4096]``: the gathered pool output before the pre-expert norm (rank ``r``'s slice is columns
  ``[512 r, 512 r + 512)``), two BF16 per Lamport word.

The stage reads them after its routing / quantization starts and never clears them (their producer owns their reuse:
a multi-reader buffer, so a C-copy arena or a stream-ordered refill). Output: ``out`` BF16 ``[8, 4096]``, plain
stores by the 64 final units.

Weights are the stock tensors of the three-window stack, read in place: ``w3_w1 [96, 8192, 2048]`` u8 and
``w2 [96, 4096, 2048]`` u8 (TRTLLM-gen row order) through a 2-D tensor map each, their 128x4-interleaved E4M3 scales,
``fc31_alpha`` / ``fc31_scale_c`` / ``fc2_alpha`` FP32 ``[96]``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from . import library, schedule_experts

#: Descriptor word indices (``mso_mk::MoeDesc``).
DESC_FIELDS = (
    "Magic",
    "Logits",
    "Pool",
    "Bias",
    "GainPre",
    "Gs",
    "TmapW13",
    "SfW13",
    "TmapW2",
    "SfW2",
    "Alpha1",
    "ScaleC",
    "Alpha2",
    "GainPost",
    "Ws",
    "Part",
    "Rank",
    "Out",
    "Dbg",
    "Eps",
    "RenormEps",
    "Flags",
    "Rows",
    "Reserved",
)
DESC = {name: i for i, name in enumerate(DESC_FIELDS)}
#: ``mso_mk::MoeFlag``.
MOE_FLAG_TILE_CUT = 1
MOE_FLAG_PIECE0_FIRST = 2
MOE_FLAG_FC2_TAIL_L2 = 4
MOE_FLAG_WARM_TWICE = 8
MOE_FLAG_NO_WARM_TMA = 16
MOE_FLAG_DBG_NO_MMA = 32
MOE_FLAG_ISSUE_FIRST = 64
MOE_FLAG_LOADER_EARLY = 128
MOE_FLAG_SPIN_SLEEP = 256
MOE_FLAG_DBG_NO_SFA = 512
MOE_FLAG_DBG_COPIES_ONLY = 1024
MOE_FLAG_NO_PAIRS = 2048
MOE_FLAG_NO_POOL_COPY = 4096
MOE_FLAG_LOCAL_QUANT = 16384
MOE_FLAG_MH_F2 = 32768  # inputs from the MoE hop areas (layer_mk.MOE_FLAG_MH_F2)
# ---- MoE front (stages/experts_defs.cuh) ----
MOE_FLAG_LOGITS_COOP = 1 << 20
MOE_FLAG_QUANT_FIRST = 1 << 21
MOE_FLAG_FC2_TAIL_EARLY = 1 << 22
MOE_FLAG_EARLY9 = 1 << 23
MOE_FLAG_ISSUE_AT_CUT = 1 << 24
MOE_FLAG_QUANT_AT_CUT = 1 << 25
# ---- end MoE front ----
# ---- FC1 -> h -> FC2 hand-off (stages/experts_defs.cuh) ----
MOE_FLAG_E3_H_WIDE = 1 << 26  # the h loader polls every needed k-block at once
MOE_FLAG_E3_FC2_PAIRS = 1 << 27  # FC2 units issued in pairs by the MMA warp
MOE_FLAG_E3_RING10 = 1 << 28  # 10-stage weight ring, X and h share one B area
MOE_FLAG_E3_FC1_QUADS = 1 << 29  # FC1 units issued in groups of up to 4
MOE_FLAG_E3_FC2_QUADS = 1 << 30  # FC2 units issued in groups of up to 4
MOE_FLAG_E3_EPI_PRE = 1 << 31  # the epilogue's partial / gain loads before the accumulator wait
MOE_FLAG_E3_XSF_PRE = 1 << 32  # all FC1 token-scale copies to TMEM at the MMA warp's start
# ---- end FC1 -> h -> FC2 hand-off ----
# ---- bytes in flight (stages/experts_defs.cuh kMoeCap*; bits 52-59) ----
#: Bits 52-55: early ring requests of the spread-quantization gather (0 = 8, one per token warp).
MOE_CAP_EARLY_SHIFT = 52
#: Bits 56-59: weight units the producer keeps requested and not landed (0 = the ring depth).
MOE_CAP_INFLIGHT_SHIFT = 56
MOE_CAP_MASK = 0xFF << MOE_CAP_EARLY_SHIFT


def moe_cap_flags(early: int = 0, inflight: int = 0) -> int:
    """The experts descriptor bits of the in-flight caps: ``early`` gather requests, ``inflight`` producer units
    (0 = default)."""
    assert 0 <= early < 16 and 0 <= inflight < 16, (early, inflight)
    return (int(early) << MOE_CAP_EARLY_SHIFT) | (int(inflight) << MOE_CAP_INFLIGHT_SHIFT)


#: The served default (``serving.DEFAULT_CAP_MOE``): 3 early requests, 6 units in flight.
MOE_CAP_DEFAULT = moe_cap_flags(3, 6)
# ---- end bytes in flight ----
# ---- MoE front + expert stream (stages/experts_defs.cuh; bits 36-43) ----
MOE_W2_CAP_SHIFT = 36  # bits 36-39: at most N weight units in flight per CTA (0 = the ring depth)
MOE_W2_CAP_MASK = 0xF << MOE_W2_CAP_SHIFT
MOE_FLAG_W2_PLACE_WARP = 1 << 41  # the placement in warp 0 alone (same slots)
# FC2 h-scale copies inside the MMA block (same operations, same order)
MOE_FLAG_W2_FC2_ONE_ELECT = 1 << 42
MOE_FLAG_W2_CAP_FC1_ONLY = 1 << 43  # the in-flight cap on the FC1 units only
# the logit vectors requested inside shareQuant (QuantFirst order; same values)
MOE_FLAG_W2_LOGITS_PRE = 1 << 40


# ---- the table-driven per-unit issue loops (stages/experts_defs.cuh; bits 33-34; same requests, same tensor
# operations, same order: bitwise vs the default loops) ----
# the weight stream on the whole of warp 0, from a lane-parallel unit table
MOE_FLAG_LX_PRODUCER = 1 << 33
# the MMA warp from a lane-parallel per-unit table (needs MOE_FLAG_W2_FC2_ONE_ELECT)
MOE_FLAG_LX_MMA = 1 << 34
MOE_FLAGS_LX = MOE_FLAG_LX_PRODUCER | MOE_FLAG_LX_MMA
# ---- end table-driven issue loops ----
# ---- tile queue (stages/experts_defs.cuh kMoeFlagPxQueue): persistent_experts' static queue of whole 128-row tiles
# instead of the byte cut (its rows bit for bit; schedule_experts.px_tiles is the CPU mirror) ----
MOE_FLAG_PX_QUEUE = 1 << 35
# with the queue: the MMA warp in the stock kernel's loop shape (p4a::mmaLoopPx)
MOE_FLAG_PX_MMA = 1 << 17
# with the queue and PX_MMA: an 8-stage weight ring (the stock depth on sm_107)
MOE_FLAG_PX_RING8 = 1 << 18
#: The tile-queue levels: 1 = the queue, 2 = + the MMA loop, 3 = + the 8-stage ring (served: ``serving.PX_LEVEL``).
MOE_FLAGS_PX_LEVELS = {
    0: 0,
    1: MOE_FLAG_PX_QUEUE,
    2: MOE_FLAG_PX_QUEUE | MOE_FLAG_PX_MMA,
    3: MOE_FLAG_PX_QUEUE | MOE_FLAG_PX_MMA | MOE_FLAG_PX_RING8,
}
# ---- end tile queue ----


def moe_w2_cap(n: int) -> int:
    """The flag bits of an in-flight cap of ``n`` ring units per CTA (0 = the ring depth)."""
    if not 0 <= n <= 15:
        raise ValueError(f"cap {n} out of range")
    return n << MOE_W2_CAP_SHIFT


# ---- end MoE front + expert stream ----

#: Model constants.
EXPERTS = 256
TOP_K = 8
WINDOW = 96
HIDDEN = 4096
PRE_EXPERT_EPS = 1e-5
POST_EXPERT_EPS = 1e-8
RENORM_EPS = 1e-15

#: The check-mode debug block (``kMoeDbg*``).
DBG_SEL, DBG_W, DBG_LOCAL, DBG_LOADS, DBG_SLOTS, DBG_XQ = 0, 256, 512, 768, 832, 1280
DBG_XSF = DBG_XQ + 8 * HIDDEN // 2


def _f32_bits(x: float) -> int:
    return int(torch.tensor([x], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


def lamport_words(t: torch.Tensor) -> torch.Tensor:
    """``t`` (contiguous, a multiple of 4 bytes) with every 32-bit word equal to ``0x80000000`` replaced by ``0``: the
    value a producer publishes (for FP32 that is ``-0 -> +0``; for a BF16 pair ``(+0, -0) -> (+0, +0)``)."""
    w = t.contiguous().view(torch.int32)
    return torch.where(w == -(2**31), torch.zeros_like(w), w).view(t.dtype).view(t.shape)


def _bind_abi(lib: library.Library) -> None:
    c = lib.lib
    if getattr(c, "_p4a_bound", False):
        return
    for name in (
        "msoMkMoeWsBytes",
        "msoMkMoePartBytes",
        "msoMkMoeDbgBytes",
        "msoMkMoeDescWords",
        "msoMkMoeStages",
    ):
        getattr(c, name).restype = ctypes.c_int
        getattr(c, name).argtypes = []
    c.msoMkMoeMagic.restype = ctypes.c_int64
    c.msoMkMoeMagic.argtypes = []
    c.msoMkMoeTmap.restype = ctypes.c_int
    c.msoMkMoeTmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong]
    assert c.msoMkMoeDescWords() == len(DESC_FIELDS), "descriptor mismatch"
    assert c.msoMkMoeStages() == schedule_experts.STAGES, "ring depth mismatch"
    c._p4a_bound = True


def tensor_map(lib: library.Library, stack: torch.Tensor) -> torch.Tensor:
    """The device copy (128 B) of the SW128 tensor map of an expert stack ``[L, R, 2048]`` u8 seen as
    ``[L R, 2048]``."""
    _bind_abi(lib)
    assert stack.dtype == torch.uint8 and stack.is_contiguous() and stack.shape[-1] == 2048
    rows = stack.numel() // 2048
    host = torch.zeros(128, dtype=torch.uint8)
    rc = lib.lib.msoMkMoeTmap(
        ctypes.c_void_p(host.data_ptr()),
        ctypes.c_void_p(stack.data_ptr()),
        ctypes.c_ulonglong(rows),
    )
    assert rc == 0, f"cuTensorMapEncodeTiled: {rc}"
    return host.to(stack.device)


@dataclass
class Window:
    """One layer's three-window expert stack on this rank (the stock tensors, in place)."""

    w13: torch.Tensor  # u8 [96, 8192, 2048]
    sf13: torch.Tensor  # E4M3 (any 1-byte dtype) [96, 8192, 256], 128x4 interleaved
    w2: torch.Tensor  # u8 [96, 4096, 2048]
    sf2: torch.Tensor  # [96, 4096, 256]
    alpha1: torch.Tensor  # FP32 [96] fc31_alpha (gate)
    scale_c: torch.Tensor  # FP32 [96] fc31_scale_c
    alpha2: torch.Tensor  # FP32 [96] fc2_alpha

    def check(self) -> None:
        L = self.w13.shape[0]
        assert L == WINDOW
        assert tuple(self.w13.shape) == (L, 2 * HIDDEN, HIDDEN // 2) and self.w13.is_contiguous()
        assert tuple(self.w2.shape) == (L, HIDDEN, HIDDEN // 2) and self.w2.is_contiguous()
        assert self.sf13.numel() == L * 2 * HIDDEN * HIDDEN // 16 and self.sf13.is_contiguous()
        assert self.sf2.numel() == L * HIDDEN * HIDDEN // 16 and self.sf2.is_contiguous()
        for t in (self.alpha1, self.scale_c, self.alpha2):
            assert t.dtype == torch.float32 and t.numel() == L and t.is_contiguous()


@dataclass
class LayerInputs:
    """Per-layer buffers the descriptor points at (graph-stable)."""

    logits: torch.Tensor  # FP32 [8, 256], Lamport words
    pool: torch.Tensor  # BF16 [8, 4096], Lamport words
    out: torch.Tensor  # BF16 [8, 4096]
    dbg: Optional[torch.Tensor] = None  # u8 [kMoeDbgBytes] (check)
    rows: Optional[torch.Tensor] = None  # BF16 [64, 4096] stock-format FC2 rows (check)


class ExpertsStage:
    """The MoE experts stage on its own: one ``Megakernel`` of ``experts_tables`` and per layer its parameter blocks.

    Args:
        mode: the library's compile mode (``check`` or ``lean``).
        copies: arena / workspace copies (launch n uses copy n % copies).
    """

    def __init__(
        self,
        mode: str = "check",
        copies: int = 3,
        workspace: Optional[library.Workspace] = None,
        timeout_ms: int = 20000,
    ):
        self.lib = library.library(mode)
        _bind_abi(self.lib)
        self.ws = workspace or library.Workspace(self.lib, copies)
        dev = self.ws.arena.device
        c = self.lib.lib
        self.ws_bytes = c.msoMkMoeWsBytes()
        self.dbg_bytes = c.msoMkMoeDbgBytes()
        self.magic = int(c.msoMkMoeMagic())
        empty = lambda nbytes: torch.full((nbytes // 4,), -0.0, dtype=torch.float32, device=dev)  # noqa: E731
        self.moe_ws = empty(self.ws.copies * self.ws_bytes)
        self.part = empty(c.msoMkMoePartBytes())
        self.tables = schedule_experts.experts_tables(self.lib.ctas)
        self.mk = library.Megakernel(self.lib, self.ws, self.tables, timeout_ms=timeout_ms)

    def inputs(self, check: bool = False) -> LayerInputs:
        dev = self.ws.arena.device
        return LayerInputs(
            logits=torch.zeros(8, EXPERTS, dtype=torch.float32, device=dev),
            pool=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=dev),
            out=torch.zeros(8, HIDDEN, dtype=torch.bfloat16, device=dev),
            dbg=torch.zeros(self.dbg_bytes, dtype=torch.uint8, device=dev) if check else None,
            rows=torch.zeros(64, HIDDEN, dtype=torch.bfloat16, device=dev) if check else None,
        )

    def make_layer(
        self,
        window: Window,
        bias: torch.Tensor,
        gain_pre: torch.Tensor,
        global_scale: torch.Tensor,
        w_post: torch.Tensor,
        rank: int,
        io: LayerInputs,
        tile_cut: bool = False,
        piece0_first: bool = False,
        stamps: bool = False,
        extra_flags: int = 0,
    ):
        """One layer's parameter blocks. ``bias`` FP32 ``[256]``; ``gain_pre`` BF16 ``[4096]`` = ``bf16(w + 1)`` of
        ``pre_expert_norm``; ``global_scale`` FP32 ``[1]`` (``fc31_input_scale``); ``w_post`` BF16 ``[4096]``
        (``post_expert_norm_weight``, gain ``w + 1``); ``io`` the per-layer buffers."""
        window.check()
        dev = self.ws.arena.device
        assert bias.dtype == torch.float32 and bias.numel() == EXPERTS and bias.is_contiguous()
        assert (
            gain_pre.dtype == torch.bfloat16
            and gain_pre.numel() == HIDDEN
            and gain_pre.is_contiguous()
        )
        assert (
            w_post.dtype == torch.bfloat16 and w_post.numel() == HIDDEN and w_post.is_contiguous()
        )
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1
        t13 = tensor_map(self.lib, window.w13)
        t2 = tensor_map(self.lib, window.w2)
        flags = (MOE_FLAG_TILE_CUT if tile_cut else 0) | (
            MOE_FLAG_PIECE0_FIRST if piece0_first else 0
        )
        flags |= int(extra_flags)
        vals = dict(
            Magic=self.magic,
            Logits=io.logits.data_ptr(),
            Pool=io.pool.data_ptr(),
            Bias=bias.data_ptr(),
            GainPre=gain_pre.data_ptr(),
            Gs=global_scale.data_ptr(),
            TmapW13=t13.data_ptr(),
            SfW13=window.sf13.data_ptr(),
            TmapW2=t2.data_ptr(),
            SfW2=window.sf2.data_ptr(),
            Alpha1=window.alpha1.data_ptr(),
            ScaleC=window.scale_c.data_ptr(),
            Alpha2=window.alpha2.data_ptr(),
            GainPost=w_post.data_ptr(),
            Ws=self.moe_ws.data_ptr(),
            Part=self.part.data_ptr(),
            Rank=int(rank),
            Out=io.out.data_ptr(),
            Dbg=0 if io.dbg is None else io.dbg.data_ptr(),
            Eps=_f32_bits(PRE_EXPERT_EPS) | (_f32_bits(POST_EXPERT_EPS) << 32),
            RenormEps=_f32_bits(RENORM_EPS),
            Flags=flags,
            Rows=0 if io.rows is None else io.rows.data_ptr(),
            Reserved=0,
        )
        host = torch.zeros(len(DESC_FIELDS), dtype=torch.int64)
        for name, v in vals.items():
            host[DESC[name]] = int(v) if int(v) < 2**63 else int(v) - 2**64
        desc = host.to(dev)
        keep = (desc, t13, t2, window, bias, gain_pre, global_scale, w_post, io)
        return self.mk.make_layer(
            dict(Moe=desc.data_ptr()), flags=library.FLAG_EVICT_FIRST, stamps=stamps, keep=keep
        )

    def __call__(self, layer, pdl: bool = True):
        return self.mk.launch(layer, library.KernelIo(), pdl=pdl)


def debug_view(dbg: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The debug block's fields (host tensors)."""
    d = dbg.cpu()
    i32 = lambda off, n: d[off : off + 4 * n].view(torch.int32).clone()  # noqa: E731
    E = int(i32(DBG_SLOTS, 1)[0])
    return dict(
        selected=i32(DBG_SEL, 64).view(8, 8),
        weights=d[DBG_W : DBG_W + 256].view(torch.float32).clone().view(8, 8),
        local_ids=i32(DBG_LOCAL, 64).view(8, 8),
        loads=i32(DBG_LOADS, 16),
        E=torch.tensor(E),
        slots=i32(DBG_SLOTS + 4, 64)[:E],
        fp4=d[DBG_XQ : DBG_XQ + 8 * HIDDEN // 2].clone().view(8, HIDDEN // 2),
        scale=d[DBG_XSF : DBG_XSF + 8 * HIDDEN // 16].clone().view(8, HIDDEN // 16),
    )


# ---- synthetic inputs ------------------------------------------------------------------------------------------------
def census_routing(loads: Sequence[int], gen: torch.Generator) -> List[List[int]]:
    """Top-8 sets of the 8 rows whose per-block distinct experts are ``loads`` (a recorded verify-step cell): the
    distinct experts are drawn per block, every one is given to at least one row, the remaining slots of each row
    are filled with other experts of the set."""
    experts: List[int] = []
    for b, n in enumerate(loads):
        perm = torch.randperm(32, generator=gen).tolist()
        experts += [32 * b + i for i in perm[:n]]
    D = len(experts)
    assert 8 <= D <= 64, loads
    order = torch.randperm(D, generator=gen).tolist()
    rows: List[List[int]] = [[] for _ in range(8)]
    for j, idx in enumerate(order):
        rows[j % 8].append(experts[idx])
    for t in range(8):
        cand = [e for e in experts if e not in rows[t]]
        perm = torch.randperm(len(cand), generator=gen).tolist()
        for i in perm:
            if len(rows[t]) >= TOP_K:
                break
            rows[t].append(cand[i])
        assert len(rows[t]) == TOP_K and len(set(rows[t])) == TOP_K
    return rows


def logits_for(rows: List[List[int]], gen: torch.Generator, device) -> torch.Tensor:
    """Router logits whose top-8 (with a bias below 0.01 in magnitude) are exactly ``rows``: selected experts in
    [2, 4], the rest in [-6, -1]."""
    out = torch.rand(8, EXPERTS, generator=gen, device="cpu") * -5.0 - 1.0
    for t, sel in enumerate(rows):
        for e in sel:
            out[t, e] = 2.0 + 2.0 * float(torch.rand(1, generator=gen))
    return out.to(device)


def random_window(gen: torch.Generator, device, lo_exp: int = -3, hi_exp: int = 0) -> Window:
    """A random NVFP4 window (random codes; E4M3 block scales 2^U{lo..hi} * U(1, 2); per-expert scalars with full
    mantissas), as the stock kernel's tests draw it."""

    def e4m3(shape):
        e = torch.randint(lo_exp, hi_exp + 1, shape, device=device, generator=gen).float()
        m = 1.0 + torch.rand(shape, device=device, generator=gen)
        return (torch.exp2(e) * m).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)

    def scalars(lo=-7, hi=-5):
        e = torch.randint(lo, hi + 1, (WINDOW,), device=device, generator=gen).float()
        return torch.exp2(e) * (1.0 + torch.rand(WINDOW, device=device, generator=gen))

    return Window(
        w13=torch.randint(
            0,
            256,
            (WINDOW, 2 * HIDDEN, HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
            generator=gen,
        ),
        sf13=e4m3((WINDOW, 2 * HIDDEN, HIDDEN // 16)),
        w2=torch.randint(
            0, 256, (WINDOW, HIDDEN, HIDDEN // 2), dtype=torch.uint8, device=device, generator=gen
        ),
        sf2=e4m3((WINDOW, HIDDEN, HIDDEN // 16)),
        alpha1=scalars(),
        scale_c=scalars(),
        alpha2=scalars(),
    )


def load_census(path: str) -> List[List[int]]:
    """The per-block distinct-expert counts of recorded verify steps (one hex digit per block)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append([int(ch, 16) for ch in line])
    return out


# ---- stock references (single GPU) -----------------------------------------------------------------------------------
def stock_routing(logits: torch.Tensor, bias: torch.Tensor):
    """``(selected int32 [8, 8], weights FP32 [8, 8])`` of the built-in ``_full_routing_kernel``."""
    from tensorrt_llm._torch.models.modeling_musespark_open import _full_routing_kernel

    rows = logits.shape[0]
    selected = torch.empty(rows, TOP_K, dtype=torch.int32, device=logits.device)
    weights = torch.empty(rows, TOP_K, dtype=torch.float32, device=logits.device)
    _full_routing_kernel[(rows,)](
        logits, bias, selected, weights, RENORM_EPS, EXPERTS=256, TORCH_ORDER=False, num_warps=4
    )
    return selected, weights


def stock_placement(selected: torch.Tensor, rank: int) -> torch.Tensor:
    from ..replicated_experts import windows

    flat, _ = windows.place(selected.reshape(-1).tolist(), rank, 3)
    return torch.tensor(flat, dtype=torch.int32).view(selected.shape)


def stock_quant(pool: torch.Tensor, gain_pre: torch.Tensor, global_scale: torch.Tensor):
    """flashinfer ``rmsnorm_fp4quant`` as the built-in ``prepare_expert_input`` calls it: ``(fp4 u8 [8, 2048], scale
    u8 [8, 256])``."""
    from flashinfer.norm import rmsnorm_fp4quant

    fp4, sf = rmsnorm_fp4quant(
        pool,
        gain_pre,
        eps=PRE_EXPERT_EPS,
        global_scale=global_scale.reshape(1),
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=False,
    )
    return fp4.view(torch.uint8).view(8, HIDDEN // 2), sf.view(torch.uint8).reshape(8, HIDDEN // 16)


def stock_experts(fp4, scale, local_ids, weights, window: Window):
    """``persistent_experts`` (the served sm107 kernel) on the same quantized rows: ``(rows [64, 4096], index)``."""
    from ...ops.moe.persistent_experts import persistent_experts

    rows, _, index = persistent_experts(
        fp4.contiguous(),
        scale.contiguous().view(-1),
        local_ids.to(torch.int32).contiguous(),
        weights.to(torch.bfloat16).contiguous(),
        window.w13,
        window.sf13.view(torch.float8_e4m3fn),
        window.w2,
        window.sf2.view(torch.float8_e4m3fn),
        window.scale_c,
        window.alpha1,
        window.alpha2,
    )
    return rows, index


def stock_combine(
    rows: torch.Tensor, index: torch.Tensor, weights: torch.Tensor, w_post: torch.Tensor
):
    """``post_expert_norm_combine`` (FP32) rounded to BF16 with -0 cleared: this rank's partial."""
    from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

    out = post_expert_norm_combine(
        rows, index, weights, w_post, POST_EXPERT_EPS, num_tokens=8, top_k=TOP_K, hidden_size=HIDDEN
    )
    b = out.to(torch.bfloat16)
    return torch.where(b == 0, torch.zeros_like(b), b)


# ---- float64 references between the stock rounding points ------------------------------------------------------------
def quant_exact(
    pool: torch.Tensor, gain_pre: torch.Tensor, eps: float = PRE_EXPERT_EPS
) -> torch.Tensor:
    """The pre-expert norm's exact value ``x (w + 1) / rms(x)`` in float64 (the quantizer's target, before any
    rounding but the BF16 product the stock kernel keeps)."""
    x = pool.double()
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (pool * gain_pre).double() * rstd


def dequant_nvfp4(fp4: torch.Tensor, scale: torch.Tensor, global_scale: float) -> torch.Tensor:
    """NVFP4 rows (linear scales) back to float64, divided by the global scale (the value the quantizer targets)."""
    from ...ops.moe.persistent_experts import _decode_fp4

    sf = scale.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
    return _decode_fp4(fp4).double() * sf / global_scale


def experts_reference(fp4, scale, local_ids, weights, window: Window) -> torch.Tensor:
    """FP64 ``PersistentExperts.reference`` rows (exact division in the requantization)."""
    from ...ops.moe.persistent_experts import persistent_experts

    rows, _, _ = persistent_experts.reference(
        fp4,
        scale.contiguous().view(torch.float8_e4m3fn).view(-1),
        local_ids.to(torch.int32),
        weights.to(torch.bfloat16),
        window.w13,
        window.sf13.view(torch.float8_e4m3fn),
        window.w2,
        window.sf2.view(torch.float8_e4m3fn),
        window.scale_c,
        window.alpha1,
        window.alpha2,
    )
    return rows


def combine_reference(
    rows: torch.Tensor, local_ids: torch.Tensor, weights: torch.Tensor, w_post: torch.Tensor
):
    """The weighted post-norm sum in float64 from given rows: ``sum_k w_k rsqrt(mean(rho_k^2) + eps) rho_k``,
    ``rho = y (w_post + 1)``."""
    gain = w_post.double() + 1.0
    y = rows.double().view(8, TOP_K, HIDDEN) * gain
    valid = (local_ids >= 0) & (local_ids < WINDOW)
    rstd = torch.rsqrt(y.square().mean(-1) + POST_EXPERT_EPS)
    scale = torch.where(valid.to(rows.device), weights.double() * rstd, torch.zeros_like(rstd))
    return (scale[..., None] * y).sum(1)


# ---- one real layer from the checkpoint (open export format) ---------------------------------------------------------
def _shuffled_rows(rows: int, device) -> torch.Tensor:
    """Physical row ``p`` of a TRTLLM-gen 32-row shuffled matrix holds pre-shuffle row ``[p]``."""
    p = torch.arange(rows, device=device)
    return (p // 32) * 32 + 4 * (p % 8) + (p % 32) // 8


def _swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """Linear ``[R, C]`` scale bytes -> the 128x4-interleaved layout (``persistent_experts._unswizzle_sf``'s
    inverse)."""
    rows, cols = sf.shape
    m = torch.arange(rows, device=sf.device)[:, None]
    c = torch.arange(cols, device=sf.device)[None, :]
    off = (
        (m // 128) * (cols // 4) * 512
        + (c // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (c % 4)
    )
    out = torch.empty(rows * cols, dtype=torch.uint8, device=sf.device)
    out[off.reshape(-1)] = sf.reshape(-1)
    return out.view(rows, cols)


def load_real_layer(ckpt: str, layer: int, rank: int, device) -> Dict[str, object]:
    """Layer ``layer``'s MoE tensors for rank ``rank``'s three-window stack, converted to the stock layout.

    The checkpoint holds ``gate_up_proj [256, 8192, 2048]`` (rows ``[0, 4096)`` gate = w1, ``[4096, 8192)`` up =
    w3), linear E4M3 scales, ``weight_scale_2 [256, 2]`` (gate, up) and input scales. The stock layout
    (``persistent_experts`` reference): pre-shuffle row ``2 i`` = up row ``i``, ``2 i + 1`` = gate row ``i``, then the
    32-row shuffle; scales permuted alike and 128x4 interleaved; w2 rows shuffled. Scalars as ``quantization.py``
    forms them, with the input scales' maxima over the window's 96 experts: ``fc31_input_scale = 1 /
    max(w13_input_scale)``, ``fc31_alpha = ws2_gate / fc31_input_scale``, ``fc31_scale_c = fc2_input_scale * ws2_up /
    fc31_input_scale``, ``fc2_alpha = ws2_w2 / fc2_input_scale``."""
    import json
    import os

    from safetensors import safe_open

    pre = f"model.language_model.layers.{layer}.mlp."
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    handles: Dict[str, object] = {}

    def get(name, sl=None):
        fn = wmap[pre + name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(ckpt, fn), framework="pt", device="cpu")
        h = handles[fn]
        return h.get_slice(pre + name)[sl] if sl is not None else h.get_tensor(pre + name)

    blocks = [(rank - j) % 8 for j in range(3)]  # [own | r - 1 | r - 2]
    src13 = _shuffled_rows(2 * HIDDEN, device)
    src2 = _shuffled_rows(HIDDEN, device)
    w13s, sf13s, w2s, sf2s = [], [], [], []
    for b in blocks:
        gu = get("experts.gate_up_proj", slice(32 * b, 32 * b + 32)).to(device)
        gus = (
            get("experts.gate_up_proj_weight_scale", slice(32 * b, 32 * b + 32))
            .view(torch.uint8)
            .to(device)
        )
        dn = get("experts.down_proj", slice(32 * b, 32 * b + 32)).to(device)
        dns = (
            get("experts.down_proj_weight_scale", slice(32 * b, 32 * b + 32))
            .view(torch.uint8)
            .to(device)
        )
        for e in range(32):
            inter = torch.empty_like(gu[e])
            inter[0::2] = gu[e, HIDDEN:]
            inter[1::2] = gu[e, :HIDDEN]
            w13s.append(inter[src13])
            inter_s = torch.empty_like(gus[e])
            inter_s[0::2] = gus[e, HIDDEN:]
            inter_s[1::2] = gus[e, :HIDDEN]
            sf13s.append(_swizzle_128x4(inter_s[src13]))
            w2s.append(dn[e][src2])
            sf2s.append(_swizzle_128x4(dns[e][src2]))
        del gu, gus, dn, dns
    ids = torch.tensor([32 * b + i for b in blocks for i in range(32)])
    ws2 = get("experts.gate_up_proj_weight_scale_2").float()[ids]
    in13 = get("experts.gate_up_proj_input_scale").float()[ids]
    ws2_2 = get("experts.down_proj_weight_scale_2").float()[ids]
    in2 = get("experts.down_proj_input_scale").float()[ids]
    fc31_is = 1.0 / in13.max()
    fc2_is = 1.0 / in2.max()
    window = Window(
        w13=torch.stack(w13s).contiguous(),
        sf13=torch.stack(sf13s).contiguous(),
        w2=torch.stack(w2s).contiguous(),
        sf2=torch.stack(sf2s).contiguous(),
        alpha1=(ws2[:, 0] / fc31_is).to(device).contiguous(),
        scale_c=(fc2_is * ws2[:, 1] / fc31_is).to(device).contiguous(),
        alpha2=(ws2_2 / fc2_is).to(device).contiguous(),
    )
    return dict(
        window=window,
        global_scale=fc31_is.reshape(1).to(device),
        router=get("gate.weight").float().to(device),
        bias=get("gate.e_score_correction_bias").float().to(device).contiguous(),
        pooler=get("pre_expert_proj.weight").to(device),
        gain_pre=(get("pre_expert_norm.weight").float() + 1)
        .to(torch.bfloat16)
        .to(device)
        .contiguous(),
        w_post=get("experts.post_expert_norm.weight").to(torch.bfloat16).to(device).contiguous(),
    )
