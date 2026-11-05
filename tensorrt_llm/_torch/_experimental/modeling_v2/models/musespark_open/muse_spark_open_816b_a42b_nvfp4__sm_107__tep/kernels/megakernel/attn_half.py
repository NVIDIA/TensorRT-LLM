# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention half of a sliding-window decoder layer as one megakernel launch (``stages/attnHalf.cuh``).

One launch per layer and step takes the layer's normed BF16 attention input ``x [8, 8192]`` (what the previous
layer's post-FFN boundary wrote) and stores the o_proj partial BF16 ``[8, 8192]`` into this rank's hop1 slot, in
between: the QKV projection (``qkv.py``'s K-slice runs), the stock prep of every head (weightless Q/K RMSNorm, query
pre-scale, GPT-J RoPE), the FP8 KV append of the 8 rows into the paged pool, the sliding-window attention over the
FP8 pages, the merge, the output norm-gate and the o_proj GEMV. The stock post-attention consumer hop then reads the
slot as it reads the stock o_proj's.

``AttnHalf`` owns the compiled library (a ``*_s4`` mode), the workspace, the attention partial buffer and the role
tables (``schedule_attn``); ``make_layer`` builds a layer's parameter blocks (eagerly, before any capture: the
weights are repacked there and every per-step buffer -- kv lengths, positions, block table, slot -- is bound by
address), and ``__call__`` launches one.

The float64 references below are the check-mode stations of the unit tests (``check.py`` holds the error statistic).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

from . import library, qkv, schedule, schedule_attn

HEAD_DIM = 64
Q_HEADS = 16
KV_HEADS = 2
ROWS = 8
O_COLS = Q_HEADS * HEAD_DIM


#: The packed QKV weight's 64-row blocks (``stages/attnHalf.cuh``): q heads, k heads, v heads, gate heads, as
#: checkpoint blocks (checkpoint order ``[q_0 | g_0 | ... | q_15 | g_15 | k_0 | k_1 | v_0 | v_1]``).
HEADS_FIRST_BLOCKS = (
    [2 * h for h in range(Q_HEADS)]
    + [2 * Q_HEADS + h for h in range(KV_HEADS)]
    + [2 * Q_HEADS + KV_HEADS + h for h in range(KV_HEADS)]
    + [2 * h + 1 for h in range(Q_HEADS)]
)


def pack_qkv_heads_first(w: torch.Tensor) -> torch.Tensor:
    """The QKV weight ``[2304, 8192]`` (checkpoint row order) with its 64-row blocks reordered heads first, packed in
    the K-slice order (``qkv.pack_qkv(..., 4)``)."""
    rows = torch.cat(
        [torch.arange(64 * b, 64 * b + 64, device=w.device) for b in HEADS_FIRST_BLOCKS]
    )
    return qkv.pack_qkv(w[rows].contiguous(), 4)


#: ``kPAhFlags`` bit 0 (``kAhFlagOprojCanary``): o_proj's lane 0 of warp 0 waits on one word of the gated input before
#: the CTA polls its whole copy.
OPROJ_CANARY = 1


def _f32_bits(x: float) -> int:
    return int(torch.tensor(float(x), dtype=torch.float32).view(torch.int32))


class AttnHalf:
    """The attention-half megakernel on this device.

    Args:
        mode: a K-slice compile mode of the library (``lean_s4`` for timing, ``check_s4`` for checks and stamps).
        copies: arena copies (the launches of one replay must be a multiple of it).
        o_replicas: copies of the gated o_proj input in the arena (1-4; the o_proj CTAs read copy ``cta % n``).
        three_tile_ctas: CTAs that take three o_proj tiles (the rest take two; 512 tiles in all).
        late_ctas: the last CTAs of the grid, given only gate work and o_proj (they start last under PDL); 0 places
            by CTA order alone.
        oproj_issue, oproj: see ``schedule_attn.attention_tables``.
    """

    def __init__(
        self,
        mode: str = "lean_s4",
        copies: int = 3,
        timeout_ms: int = 20000,
        o_replicas: int = 1,
        three_tile_ctas: int = schedule_attn.THREE_TILE_CTAS,
        workspace: Optional[library.Workspace] = None,
        late_ctas: int = schedule_attn.LATE_CTAS,
        oproj_issue: str = "early",
        oproj: bool = True,
    ):
        assert 1 <= o_replicas <= 4
        self.lib = library.library(mode)
        assert self.lib.qkv_slices == 4, "the attention half needs a K-slice (_s4) build"
        self.ws = workspace or library.Workspace(self.lib, copies)
        dev = self.ws.arena.device
        # Single-reader partials: every word kEmpty (-0.0) between launches.
        self.part = torch.full(
            (self.lib.ah_part_bytes // 4,), -0.0, dtype=torch.float32, device=dev
        )
        self.o_replicas = int(o_replicas)
        self.tables = schedule_attn.attention_tables(
            self.lib.ctas,
            three_tile_ctas=three_tile_ctas,
            late_ctas=late_ctas,
            oproj_issue=oproj_issue,
            oproj=oproj,
        )
        self.mk = library.Megakernel(self.lib, self.ws, self.tables, timeout_ms=timeout_ms)

    def make_layer(
        self,
        w_qkv: torch.Tensor,
        w_o: torch.Tensor,
        pool: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        slot: torch.Tensor,
        *,
        kv_scale: float,
        q_scale: float,
        prep_eps: float,
        gate_eps: float,
        window: int,
        q_fp8: bool = True,
        check: bool = False,
        stamps: bool = False,
        attn_out: Optional[torch.Tensor] = None,
        evict_first: bool = True,
        l2_deferred: bool = False,
        oproj_canary: bool = False,
        packed_qkv: Optional[torch.Tensor] = None,
    ):
        """One layer's parameter blocks (one per arena copy).

        Args:
            w_qkv: BF16 ``[2304, 8192]`` in the checkpoint row order.
            w_o: BF16 ``[8192, 1024]``.
            pool: the layer's FP8 pool view ``[blocks, 2, 32, 64]`` (any 1-byte dtype, contiguous).
            block_table: int32 ``[2, max_blocks]`` (K, V block ids), unit last stride.
            seq_lens: int32, element 0 = kv length including the step (graph input; bound by address).
            positions: int32 ``[8]`` RoPE positions (graph input; bound by address).
            cos_sin: FP32 ``[positions, 2, 32]`` contiguous.
            slot: BF16 ``[8, 8192]`` output (the hop1 slot's multicast view, or a plain buffer).
            attn_out: BF16 ``[8, 1024]``: the ungated attention output (check runs), or None.
            l2_deferred: at entry, prefetch into L2 the weights of the units whose region is still busy (o_proj).
            oproj_canary: o_proj waits on one word of the gated input before polling all of it (``OPROJ_CANARY``).
            packed_qkv: ``pack_qkv_heads_first(w_qkv)`` made by the caller (e.g. to L2-prefetch the same tensor), or
                None to pack it here.
        """
        assert w_qkv.dtype == torch.bfloat16 and tuple(w_qkv.shape) == (
            schedule.QKV_ROWS,
            schedule.HIDDEN,
        )
        assert w_o.dtype == torch.bfloat16 and tuple(w_o.shape) == (schedule.HIDDEN, O_COLS)
        assert pool.element_size() == 1 and pool.is_contiguous() and pool.dim() == 4
        assert tuple(pool.shape[1:]) == (KV_HEADS, 32, HEAD_DIM)
        assert block_table.dtype == torch.int32 and block_table.dim() == 2
        assert block_table.shape[0] == 2 and block_table.stride(1) == 1
        assert seq_lens.dtype == torch.int32 and positions.dtype == torch.int32
        assert positions.numel() == ROWS and positions.is_contiguous()
        assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
        assert tuple(cos_sin.shape[1:]) == (2, HEAD_DIM // 2)
        assert slot.dtype == torch.bfloat16 and tuple(slot.shape) == (ROWS, schedule.HIDDEN)
        assert slot.is_contiguous()
        if attn_out is not None:
            assert attn_out.dtype == torch.bfloat16 and tuple(attn_out.shape) == (ROWS, O_COLS)
        if packed_qkv is None:
            packed_qkv = pack_qkv_heads_first(w_qkv)
        packed_o = qkv.pack_bf16(w_o.contiguous())
        fields: Dict[str, int] = dict(
            WQkv=packed_qkv.data_ptr(),
            Eps=_f32_bits(qkv.PRE_QKV_EPS),
            AhWOproj=packed_o.data_ptr(),
            AhSlot=slot.data_ptr(),
            AhPool=pool.data_ptr(),
            AhBlockTable=block_table.data_ptr(),
            AhBtStride=block_table.stride(0),
            AhSeqLen=seq_lens.data_ptr(),
            AhPositions=positions.data_ptr(),
            AhCosSin=cos_sin.data_ptr(),
            AhKvScale=_f32_bits(kv_scale),
            AhQScale=_f32_bits(q_scale),
            AhPrepEps=_f32_bits(prep_eps),
            AhGateEps=_f32_bits(gate_eps),
            AhPart=self.part.data_ptr(),
            AhWindow=int(window),
            AhQFp8=int(bool(q_fp8)),
            AhOReplicas=self.o_replicas,
            AhAttnOut=0 if attn_out is None else attn_out.data_ptr(),
            AhFlags=OPROJ_CANARY if oproj_canary else 0,
        )
        flags = (
            (library.FLAG_EVICT_FIRST if evict_first else 0)
            | library.FLAG_ONE_PIECE
            | (library.FLAG_CHECK if check else 0)
            | (library.FLAG_L2_DEFERRED if l2_deferred else 0)
        )
        keep = (
            packed_qkv,
            packed_o,
            pool,
            block_table,
            seq_lens,
            positions,
            cos_sin,
            slot,
            attn_out,
        )
        return self.mk.make_layer(fields, flags=flags, stamps=stamps, keep=keep)

    def __call__(
        self, layer, x: torch.Tensor, qkv_out: Optional[torch.Tensor] = None, pdl: bool = True
    ):
        """Launch one layer on the current stream: ``x`` BF16 ``[8, 8192]``; ``qkv_out`` BF16 ``[8, 2304]`` receives
        the projection (check runs)."""
        assert x.dtype == torch.bfloat16 and tuple(x.shape) == (ROWS, schedule.HIDDEN)
        assert x.is_contiguous()
        return self.mk.launch(layer, library.KernelIo(x=x, qkv=qkv_out), pdl=pdl)

    # ---- arena views of the last launch of copy ``copy`` (tests) ----
    def arena_bytes(self, copy: int) -> torch.Tensor:
        return self.ws.arena_copy(copy).view(torch.uint8)

    def arena_q(self, copy: int) -> torch.Tensor:
        """BF16 ``[8 tokens, 16 heads, 64]`` attention Q."""
        a = self.arena_bytes(copy)[16384 : 16384 + 16384].view(torch.bfloat16)
        return (
            a.view(KV_HEADS, ROWS, Q_HEADS // KV_HEADS, HEAD_DIM)
            .permute(1, 0, 2, 3)
            .reshape(ROWS, Q_HEADS, HEAD_DIM)
        )

    def arena_kv_new(self, copy: int) -> torch.Tensor:
        """uint8 ``[2 (K, V), 2 heads, 8 tokens, 64]`` the step's FP8 rows."""
        a = self.arena_bytes(copy)
        return torch.stack([a[32768:33792], a[33792:34816]]).view(2, KV_HEADS, ROWS, HEAD_DIM)

    def arena_sigmoid(self, copy: int) -> torch.Tensor:
        """BF16 ``[8, 1024]`` bf16(sigmoid(gate))."""
        return self.arena_bytes(copy)[34816 : 34816 + 16384].view(torch.bfloat16).view(ROWS, O_COLS)

    def arena_gated(self, copy: int, replica: int = 0) -> torch.Tensor:
        """BF16 ``[8, 1024]`` the gated o_proj input."""
        off = 51200 + replica * 16384
        return self.arena_bytes(copy)[off : off + 16384].view(torch.bfloat16).view(ROWS, O_COLS)


# ---------------------------------------------------------------------------------------------------------------------
# float64 references (check mode)
# ---------------------------------------------------------------------------------------------------------------------
def kv_rows(pool: torch.Tensor, block_table: torch.Tensor, length: int) -> torch.Tensor:
    """The step's 8 K and V rows in the pool: uint8 ``[2, 2 heads, 8, 64]``."""
    p = pool.view(torch.uint8)
    pos = torch.arange(length - ROWS, length, device=pool.device)
    out = []
    for kv in (0, 1):
        pages = block_table[kv].long()[pos // 32]
        out.append(p[pages, :, pos % 32, :].permute(1, 0, 2))
    return torch.stack(out)


def to_cache_fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """The stock cache store of BF16 values: e4m3(bf16(x * bf16(s))) (RN, satfinite)."""
    s = torch.tensor(scale, dtype=torch.float32).to(torch.bfloat16).float()
    y = (x.float() * s).to(torch.bfloat16).float()
    return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def attention_exact(
    q: torch.Tensor,
    pool: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    window: int,
    kv_scale: float,
) -> torch.Tensor:
    """float64 sliding-window attention ``[8, 16, 64]`` of ``q`` (the kernel's BF16 Q, i.e. the FP8 query values
    times s with q_fp8) over the FP8 pool (which holds the step's rows): token t at position L - 8 + t attends keys
    ``max(L - 8 - window + 1, 0) <= j <= L - 8 + t`` with ``j > L - 8 + t - window``; scores ``q k / 8`` with
    ``k = k8 / s``, values ``v8 / s``."""
    dev = pool.device
    past = length - ROWS
    lo = max(past - window + 1, 0)
    keys = torch.arange(lo, length, device=dev)
    p = pool.view(torch.float8_e4m3fn)
    deq = 1.0 / kv_scale
    k = p[block_table[0].long()[keys // 32], :, keys % 32, :].double() * deq  # [n, 2, 64]
    v = p[block_table[1].long()[keys // 32], :, keys % 32, :].double() * deq
    qd = q.double() * deq  # q holds the FP8 values times s
    out = torch.empty(ROWS, Q_HEADS, HEAD_DIM, dtype=torch.float64, device=dev)
    for t in range(ROWS):
        qp = past + t
        keep = (keys <= qp) & (keys > qp - window)
        kk, vv = k[keep], v[keep]
        for h in range(Q_HEADS):
            kv = h // (Q_HEADS // KV_HEADS)
            s = (kk[:, kv, :] @ qd[t, h]) / math.sqrt(HEAD_DIM)
            w = torch.softmax(s, 0)
            out[t, h] = w @ vv[:, kv, :]
    return out


def norm_gate_exact(attn: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """float64 output norm-gate ``[8, 1024]``: per head ``a rsqrt(mean(a^2) + eps) sigmoid(g)``, no roundings."""
    a = attn.double().reshape(ROWS, Q_HEADS, HEAD_DIM)
    g = gate.double().reshape(ROWS, Q_HEADS, HEAD_DIM)
    n = a * torch.rsqrt(a.square().mean(-1, keepdim=True) + eps)
    return (n * torch.sigmoid(g)).reshape(ROWS, O_COLS)


def raw_gate(qkv_out: torch.Tensor) -> torch.Tensor:
    """The output gate ``[8, 16, 64]`` view of the raw projection (checkpoint order)."""
    return qkv_out[:, : 2 * Q_HEADS * HEAD_DIM].view(ROWS, Q_HEADS, 2 * HEAD_DIM)[..., HEAD_DIM:]


def stamped_units(
    lib: library.Library, tables: schedule.Tables, stamps: torch.Tensor
) -> Sequence[dict]:
    """``library.stamp_report`` of a stamped launch (a convenience re-export)."""
    return library.stamp_report(lib, tables, stamps)


def sass_function_sizes(lib: library.Library) -> Dict[str, int]:
    """SASS instructions per device function of the library (``cuobjdump -sass``), or an empty dict."""
    import os
    import re
    import subprocess

    from torch.utils import cpp_extension

    cuda_home = cpp_extension.CUDA_HOME or "/usr/local/cuda"
    try:
        out = subprocess.run(
            [os.path.join(cuda_home, "bin", "cuobjdump"), "-sass", lib.path],
            capture_output=True,
            text=True,
            timeout=300,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    sizes: Dict[str, int] = {}
    name = None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            sizes[name] = 0
        elif name is not None and re.match(r"\s*/\*[0-9a-f]{4,}\*/", line):
            sizes[name] += 1
    return sizes
