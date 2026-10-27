# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host side of the MiniMax-M3 routed-expert kernel on one TP4 rank (routed and shared intermediate 768 per rank),
over the TRTLLM-Gen NVFP4 buffers read in place.

``m3_moe`` (m3_moe_kernel.py, :class:`M3MoeState`): 1..8 decode tokens (decode batches, spec-decode verify). The call
takes the bf16 hidden states and folds in the router GEMV (bit-exact with m3_router_gemv_fp4) and the activation's
NVFP4 quantization. With ``shared=True`` it also runs the shared expert on dedicated CTAs and returns
bf16(shared + routed); otherwise it returns the rank's routed partial, bit for bit what the stock fused MoE returns.

Usage (build before CUDA-graph capture; every call is capture-safe afterwards)::

    state = M3MoeState(
        device, num_tokens=6, shared=True
    )  # workspace + compiled kernel, shared by all layers
    layer = state.layer(
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        e_score_correction_bias,
        router_weight=router_weight,
        fc31_input_scale=fc31_input_scale,
        shared_weights=(gate_up_w, gate_up_sf, down_w, down_sf),
    )
    out = layer(hidden)  # bf16 [M, 6144]
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch import cute_aot
from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled
from .. import m3_report

HIDDEN = 6144  # capacity: fixed: the checkpoint's hidden size
INTER = 1536  # capacity: fixed: the routed experts' width per tp2 rank
NUM_EXPERTS = 128
SHARED_INTER = 1536  # capacity: fixed: the shared expert's width per tp2 rank
SHARED_CTAS = 24  # capacity: fixed: an SM split, not a size (m3_moe's shared-expert CTAs (beside num_ctas routed CTAs))
TOP_K = 4
U1 = 2 * INTER // 32  # FC1 units per expert
# The router GEMV takes at most 2 experts per CTA (router weights held in registers).
MIN_CTAS = 64  # capacity: fixed: an SM split, not a size

_UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)  # one FC1 unit's weight + scale bytes
_ROW2_BYTES = INTER // 2 + INTER // 16  # one FC2 weight row's bytes (values + scales)
_S1 = 6  # FC1 stages (items) per unit  # capacity: fixed: pipeline stages
_ITEM_BYTES = _UNIT_BYTES // _S1  # capacity: fixed: a ring item's bytes
_modules: Dict[tuple, object] = {}
_compiled: Dict[tuple, object] = {}
# m3_moe trace-time options by token count 1..8 (index num_tokens - 1).
# ``sfa_cp``: 18 KB ring stages, the A scales copied by cp.async into the instruction layout (every token count).
SFA_CP_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
# ``b2img``: FC1 -> FC2 hand-off through per-group global B images + release counters. Off at TP4 (at 4 / 8 tokens
# 0.5 / 0.8 us/call slower than the tagged records in the TP4 sweep).
B2IMG_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0)
# ``ar_pre``: the finalize's all-reduce push with its call parities / descriptor preloaded (all but 2 tokens).
AR_PRE_BY_M: Tuple[int, ...] = (1, 0, 1, 1, 1, 1, 1, 1)
# Ring ``stages`` (0 = the kernel's default, the most that fit). 5 at 8 tokens (with a 50 ns FC2 record poll below)
# measured 0.8 us/call faster than 6 in the TP4 sweep (same MMAs, bit-identical outputs); at 4 tokens the default is
# fastest with heavy_fc2_only.
STAGES_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 5)
# The FC2 B builders' record poll back-off in ns (0 = the kernel's default, 200); 50 at 8 tokens.
B2_POLL_NS_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 50)
# ``early0``: warp 4 streams group 0 right after its own top-k. Off at 8 tokens (0.6 us/layer faster in the chain:
# there it is on every routing warp's critical path and the ring holds only 5 stages).
EARLY0_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 0)
# The plan's ``heavy_fc2_only``: 192 FC2 output blocks over 188 routed CTAs leave 4 CTAs with two blocks, whose
# trailing FC2 after the last hand-off otherwise ends the kernel 3-7 us after the rest; they stream no FC1 instead
# (1.7 / 0.9 us/call faster at 4 / 8 tokens in the TP4 sweep; bit-identical outputs).
HEAVY_FC2_ONLY_BY_M: Tuple[int, ...] = (0, 0, 0, 1, 0, 0, 0, 1)
# ``hb_repl``: copies of the tagged FC1 -> FC2 records (applied when > 1); CTA bx polls copy bx % hb_repl, which
# spreads the polls of the few hot record lines over L2. 7 and 8 tokens hand off through the B images instead.
HB_REPL_BY_M: Tuple[int, ...] = (
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)  # off at TP4 (0.2 us/call slower at 4 tokens)
# Per-token-count overrides of the tables above: at 8 tokens the FC1 -> FC2 hand-off through tagged records instead of
# the B image (ported from the GB300 target: 0.15 us/call faster there, paired 4/4 with bit-identical outputs,
# MOE_CLK64_MR6C_READ; it also leaves warp 6 idle at 8 tokens). Kill switch ``TRTLLM_M31_DISABLE_MOE_M8CFG`` restores
# the tables' values.
M8_CFG_BY_M: Dict[int, Dict[str, int]] = {8: {"b2img": 0}}
# SH20: at SH20_TOKENS the shared expert's 24 work units run on
# SH20_CTAS shared CTAs (CTAs 0 .. 3 run two units, both gate_ups before either down), so the routed grid takes the
# 4 SMs this frees: num_ctas = SMs - SH20_CTAS = 192 gives every routed CTA exactly one of the 192 FC2 32-row blocks
# (188 left 4 CTAs with two: the call's stragglers). Bit-identical outputs (only the CTA -> unit mapping and the
# routed CTA count change). Measured (VR TP8, 8 tokens): -0.94 us/call. Kill switch TRTLLM_M31_DISABLE_MOE_SH20,
# read at import.
# TP2 port: BLOCKED until the TP4/TP2 kernel (SH_SPLIT == 1 shared branch) takes the SH20 unit loop
# (kernels/m3_moe/m3_moe_kernel.py is the TP4 kernel here). Held off; the switch name is kept.
SH20 = False and not m31_disabled("MOE_SH20")
SH20_CTAS = 20  # capacity: fixed: an SM split, not a size
SH20_TOKENS = frozenset(
    {8}
)  # capacity: fixed: the K7 c1 rows (the two-unit CTAs finish off the critical path there)


def shared_ctas_for(num_tokens: int, shared: bool) -> int:
    """The shared-expert CTAs beside the routed ones of an engine built for ``num_tokens`` (0 without the shared
    expert): SH20_CTAS at SH20_TOKENS while SH20 is on, else SHARED_CTAS."""
    if not shared:
        return 0
    return SH20_CTAS if SH20 and num_tokens in SH20_TOKENS else SHARED_CTAS


# ``lg_repl``: replicas of the router logit exchange at up to LG_REPL_MAX_TOKENS tokens (fewer stores before the
# exchange; bit-identical outputs). Above that the kernel's default applies.
LG_REPL = 32
LG_REPL_MAX_TOKENS = 0  # capacity: fixed: measured crossover (off at TP4: flat in the sweep)
# The TP4 FLAT stack: the TP2 FLAT values (no B2 images, 16 router-logit replicas, AR push parities loaded in the
# finalize, PDL release right after the grid wait) over the per-m tables above. Bitwise-identical output (m 1..8 x 4
# routings, eager + graph); standalone m=4 -0.49 us/call (late_trigger). The values are part of the engine's module
# key.
FLAT_STACK = {"b2img": 0, "lg_repl": 16, "ar_pre": 0, "late_trigger": 0}
# FLAT only up to 4 tokens: at 8 tokens the per-m tables are faster (TP2: 2.96 us/call; TP4 in-model at 2 requests
# +175 us/step with FLAT). Bit-identical either way.
FLAT_MAX_TOKENS = 4  # capacity: fixed: measured crossover


def flat_applies(num_tokens: int) -> bool:
    """Whether an engine state for ``num_tokens`` tokens runs the FLAT stack."""
    return num_tokens <= FLAT_MAX_TOKENS


# VRPOLL: 16 replicas instead of the kernel's 32 at 7 and 8 tokens (the K7 c1 rows): the exchange's tail CTAs poll
# fewer lines. Bit-identical outputs. Microbench (VR TP8, CUDA graph, 57 calls): -0.25 us/call at 8 tokens, -0.24 at 7.
# Kill switch TRTLLM_M31_DISABLE_MOE_VRPOLL, read at import.
VRPOLL = not m31_disabled("MOE_VRPOLL")
VRPOLL_LG_REPL = 16  # capacity: fixed: measured optimum (8 / 16 / 64 swept)
VRPOLL_TOKENS = frozenset({7, 8})  # capacity: fixed: the measured rows
# Byte-balanced FC1 ranges below one unit per CTA (:func:`_fine_ranges`) for groups too small to give every CTA a
# whole unit; False gives such groups whole units on the CTAs with the least FC2 work.
FINE_PLAN = True
# HALFPLAN: when every CTA's byte-balanced share of a group is the same and at most half a unit (the partial last FC1
# group with SH20's 192 routed CTAs at 8 tokens), the fine ranges' boundaries sit on half units (S1 // 2 items), so a
# split unit is halved (3 + 3) instead of extended to its end (1 + 5) and no CTA gets more than half a unit; other
# groups keep their ranges. Bit-identical outputs (microbench: -0.18 us per call at m8 random). Kill switch
# TRTLLM_M31_DISABLE_MOE_HALFPLAN, read at import.
HALFPLAN = not m31_disabled("MOE_HALFPLAN")
# m3_moe phase lag: phase p streams FC1 of group p and FC2 of group p - M3_MOE_LAG (the kernel's default lag).
M3_MOE_LAG = 2


_ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_moe_kernel.py")


def _kernel_module(config: dict, path: str):
    key = (os.path.basename(path),) + tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        name = (
            f"{__name__}_"
            + os.path.splitext(os.path.basename(path))[0]
            + "_"
            + "_".join(f"{k}{v}" for k, v in key[1:])
        )
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        mod.M3_CONFIG = dict(config)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _modules[key] = mod
    return mod


def _view(t: torch.Tensor, align: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align)


def is_supported(device: Optional[torch.device] = None) -> tuple:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
        return False, "needs the sm_100 family (tcgen05 block-scaled NVFP4 MMA)"
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    return True, ""


def ar_push_desc(
    mc_ptr: int, flags_ptr: int, rank: int, world: int, max_tokens: int, device
) -> torch.Tensor:
    """The ``arp`` descriptor that makes a call also push its bf16 output into a multicast all-reduce exchange (set A
    of m3_mc_allreduce's workspace at ``mc_ptr``, slot half = the all-reduce CTA's call parity in ``flags_ptr``).
    With it the kernel releases its PDL dependents only after every push; the default (enable 0) keeps the old
    behaviour."""
    return torch.tensor(
        [1, mc_ptr, flags_ptr, rank, world, max_tokens, 0, 0], dtype=torch.int64, device=device
    )


def _shared_views(shared_weights, who):
    """(gate_up, its scales, down, its scales) of the shared expert as flat uint8 views, checked."""
    if shared_weights is None:
        raise ValueError(f"{who} (shared): shared_weights are required")
    gu_w, gu_sf, dn_w, dn_sf = shared_weights
    want = {"gate_up": (gu_w, (2 * SHARED_INTER, HIDDEN)), "down": (dn_w, (HIDDEN, SHARED_INTER))}
    for name, (t, shape) in want.items():
        if t.dtype != torch.float8_e4m3fn or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"{who} (shared): {name} weight must be contiguous e4m3 {shape}")
    if (
        gu_sf.numel() != 2 * SHARED_INTER * HIDDEN // 32
        or dn_sf.numel() != HIDDEN * SHARED_INTER // 32
    ):
        raise ValueError(
            f"{who} (shared): the UE8M0 scales must be the swizzled [N, K / 32] buffers"
        )
    return (
        gu_w.view(torch.uint8).view(-1),
        gu_sf.view(torch.uint8).reshape(-1),
        dn_w.view(torch.uint8).view(-1),
        dn_sf.view(torch.uint8).reshape(-1),
    )


def _pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta: int = 24) -> torch.Tensor:
    """The shared expert's weights as m3_moe's shared CTAs stream them (SH_PRETILE): per CTA s, per stage g, the 16 KB
    shared-memory image (128-byte swizzle: 16-byte chunk j of row r at j ^ (r % 8)) of the stage's 128-row MMA tile and
    its 512-byte instruction scale atom (row R's 4 codes at word 4 (R % 32) + R / 32). CTA s owns P = SHARED_INTER / 24
    = 16 positions in SPLIT = 128 / (2 P) = 4 K slices of GT = 48 / SPLIT = 12 k-tiles. Stages 0..GT-1: MMA row
    R = P O + i (O = 2 q + u: slice q, u 0 gate / 1 up) = weight row u SHARED_INTER + P s + i at k-tile GT q + g; then
    down tile j (rows 256 s + 128 j ..) at k-tile kt, j-major (SHARED_INTER / 128 k-tiles each). Built once per
    layer."""
    dev = gu8.device
    H_, I_ = HIDDEN, SHARED_INTER
    P = I_ // n_cta
    split = 128 // (2 * P)
    gt = H_ // 128 // split
    dkt = I_ // 128
    gu = gu8.view(2 * I_, H_)
    dn = dn8.view(H_, I_)
    gkb, dkb = H_ // 32, I_ // 32

    def sf_bytes(sf, n, b, kbn):  # code of (row n, K block b), broadcast
        return sf[
            ((n >> 7) * (kbn // 4) + (b >> 2)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (b & 3)
        ]

    r = torch.arange(128, device=dev)
    q = r >> 5
    ri = r & 31
    octant = r // P
    chunk = torch.arange(8, device=dev)
    swz = chunk[None, :] ^ (r[:, None] & 7)  # source chunk of image chunk j: j ^ (r % 8)
    out = torch.empty(
        n_cta, gt + 2 * dkt, 16384 + 512, dtype=torch.uint8, device=dev
    )  # capacity: fixed: a workspace of the kernel's fixed geometry
    for s in range(n_cta):
        # gate_up: MMA row R -> weight row, slice
        nrow = (octant & 1) * I_ + P * s + r % P
        sl = octant >> 1
        for g in range(gt):
            kt = sl * gt + g  # per row
            tile = gu[
                nrow[:, None], (kt[:, None] * 128 + torch.arange(128, device=dev)[None, :])
            ]  # [128, 128]
            img = tile.view(128, 8, 16)[r[:, None], swz]  # image chunk j = tile chunk j ^ (r % 8)
            out[s, g, :16384] = img.reshape(-1)
            kb = torch.arange(4, device=dev)
            codes = sf_bytes(gusf8, nrow[:, None], kt[:, None] * 4 + kb[None, :], gkb)  # [128, 4]
            atom = torch.empty(512, dtype=torch.uint8, device=dev)
            atom[(ri[:, None] * 16 + q[:, None] * 4 + kb[None, :]).reshape(-1)] = codes.reshape(-1)
            out[s, g, 16384:] = atom
        for gd in range(2 * dkt):
            jt, kt = gd // dkt, gd % dkt
            row0 = 256 * s + 128 * jt
            tile = dn[row0 : row0 + 128, kt * 128 : (kt + 1) * 128]
            img = tile.reshape(128, 8, 16)[r[:, None], swz]
            out[s, gt + gd, :16384] = img.reshape(-1)
            out[s, gt + gd, 16384:] = dnsf8[
                ((row0 >> 7) * (dkb // 4) + kt) * 512 : ((row0 >> 7) * (dkb // 4) + kt) * 512 + 512
            ]
    return out.view(-1)


# ---------------------------------------------------------------------------------------------------------------------
# m3_moe (m3_moe_kernel.py): 1..8 tokens per call, group-phased weight stream.
# ---------------------------------------------------------------------------------------------------------------------
# Pre-tiled shared weights per layer, keyed by the source buffers (which the engines read in place): every token-count
# state of a layer streams the same copy. An entry holds its sources, so their addresses are not reused while cached.
_PRETILED: Dict[tuple, tuple] = {}


def _pretiled(gu8, gusf8, dn8, dnsf8) -> torch.Tensor:
    key = (gu8.device, gu8.data_ptr(), gusf8.data_ptr(), dn8.data_ptr(), dnsf8.data_ptr())
    hit = _PRETILED.get(key)
    if hit is None:
        hit = _PRETILED[key] = ((gu8, gusf8, dn8, dnsf8), _pretile_shared(gu8, gusf8, dn8, dnsf8))
    return hit[1]


# Set by ``_fine_ranges`` when HALFPLAN's half-unit plan is taken; read (and reset) per engine state build.
_HALFPLAN_TAKEN = [False]


def _fine_ranges(weights, total: int):
    """Contiguous FC1 item ranges [s_c, e_c) over the CTAs, sized as close to ``weights`` (ideal real-valued item
    counts summing to ``total``) as the split-unit protocol allows: every unit (S1 items) has at most one CTA boundary
    strictly inside it (its head and tail parts, summed by the tail's holder), so a range that starts inside a unit
    runs at least to that unit's end. Returns the item counts."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
    half = _S1 // 2
    if HALFPLAN and n and max(weights) == min(weights) and total % half == 0 and total <= half * n:
        _HALFPLAN_TAKEN[0] = True
        # equal shares of at most half a unit: boundaries on half units, so every CTA gets 0 or S1 // 2 items, every
        # split unit is halved and no range needs extending
        for c in range(n):
            acc += weights[c]
            e = total if c == n - 1 else min(total, max(s, half * int(acc / half + 0.5)))
            items[c] = e - s
            s = e
        assert s == total
        return items
    for c in range(n):
        acc += weights[c]
        e = total if c == n - 1 else min(total, max(s, int(acc + 0.5)))
        if s % _S1 and e < total:  # started inside a unit: run to (at least) its end
            e = max(e, min(total, (s // _S1 + 1) * _S1))
        items[c] = e - s
        s = e
    assert s == total
    return items


def make_phase_plan(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    fine: bool = FINE_PLAN,
    heavy_fc2_only: bool = False,
):
    """The work split of ``m3_moe`` for each distinct-expert count G (4 .. 4 m_max): FC2 32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..) a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work, or (``fine``) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit (:func:`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and the largest block count."""
    gcap = TOP_K * m_max
    nb2 = HIDDEN // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    # (heavy_fc2_only) CTAs with more FC2 blocks than the minimum stream no FC1: they run each group's FC2 as soon as
    # its hand-off lands, so their trailing FC2 after the last hand-off matches the others' two lagged groups
    min_nb = min(nbs)
    no_fc1 = [heavy_fc2_only and nbs[c] > min_nb for c in range(num_ctas)]
    BIG_W = 1 << 40
    blk_bytes = 32 * _ROW2_BYTES  # one block, one slot
    for g in range(TOP_K, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = b0s[c] | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * U1 * _S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and not (
                    early0 and p == 0 and m_max >= 2
                ):  # early0: group 0's ranges the same for every G
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            for c in range(num_ctas):
                if no_fc1[c]:
                    fc2[c] += BIG_W
            items = [0] * num_ctas
            if total >= _S1 * num_ctas:

                def counts_for(limit):
                    return [
                        0 if no_fc1[c] else max(_S1, (limit - f) // _ITEM_BYTES)
                        for c, f in enumerate(fc2)
                    ]

                lo, hi = 0, max(f for c, f in enumerate(fc2) if not no_fc1[c]) + total * _ITEM_BYTES
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sum(counts_for(mid)) >= total:
                        hi = mid
                    else:
                        lo = mid + 1
                items = counts_for(lo)
                extra = sum(items) - total
                while extra > 0:
                    c = min(
                        (c for c in range(num_ctas) if items[c] > _S1),
                        key=lambda c: (-(fc2[c] + items[c] * _ITEM_BYTES), c),
                    )
                    items[c] -= 1
                    extra -= 1
            elif fine:
                # water-fill: CTA c streams fc2[c] + w_c item bytes, equal where w_c > 0
                lo_f, hi_f = 0.0, float(max(fc2) + total * _ITEM_BYTES)
                for _ in range(100):
                    mid = 0.5 * (lo_f + hi_f)
                    if sum(max(0.0, (mid - f) / _ITEM_BYTES) for f in fc2) >= total:
                        hi_f = mid
                    else:
                        lo_f = mid
                wts = [max(0.0, (hi_f - f) / _ITEM_BYTES) for f in fc2]
                sw = sum(wts)
                items = _fine_ranges([w * total / sw for w in wts], total)
            else:
                chosen = sorted(sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // _S1])
                for c in chosen:
                    items[c] = _S1
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g, c, p] = i0 | ((i0 + items[c]) << 16)
                i0 += items[c]
            assert i0 == total
    return plan, max(nbs)


class M3MoeState:
    """Workspace + compiled kernel of ``m3_moe`` (1..8 tokens per call) on one device, shared by the layers that run
    on it in one stream order. The call takes the bf16 hidden states and computes the router logits (bit-exact with
    m3_router_gemv_fp4) and the NVFP4 activation itself. ``shared``: the shared expert runs on SHARED_CTAS dedicated
    CTAs beside the ``num_ctas`` routed ones and the call returns bf16(shared + routed). Build before CUDA-graph
    capture; the first call compiles."""

    fused_router = True

    def __init__(
        self,
        device: torch.device,
        num_tokens: int,
        num_ctas: Optional[int] = None,
        shared: bool = False,
        inter: int = INTER,
    ):
        """``inter``: the routed intermediate per rank; this engine is built for INTER (768) only."""
        if inter != INTER:
            raise ValueError(
                f"m3_moe (TP4): routed intermediate {inter} per rank is not supported (only {INTER})"
            )
        self.inter = inter
        logger.info(f"[M3 MoE] Rubin m3_moe engine (tp4 tables): state for {num_tokens} tokens")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3MoeState allocates and compiles: build it before CUDA-graph capture"
            )
        sms_all = torch.cuda.get_device_properties(device).multi_processor_count
        sh_ctas = shared_ctas_for(num_tokens, shared)
        if shared and (num_ctas or sms_all) + sh_ctas > sms_all:
            raise ValueError(
                f"m3_moe (shared): num_ctas + {sh_ctas} shared-expert CTAs must fit the {sms_all} SMs "
                f"(all resident at once)"
            )
        if not 1 <= num_tokens <= 8:
            raise ValueError(f"m3_moe takes 1..8 tokens per call, not {num_tokens}")
        self.device = torch.device(device)
        ok, why = is_supported(self.device)
        if not ok:
            raise RuntimeError(f"m3_moe: {why}")
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.num_ctas = num_ctas or sms
        if not MIN_CTAS <= self.num_ctas <= sms:
            raise ValueError(
                f"m3_moe runs {MIN_CTAS} .. {sms} CTAs (all resident at once), not {self.num_ctas}"
            )
        if self.num_ctas > HIDDEN // 32:
            # a routed CTA that owns no 32-row FC2 output block deadlocks the FC2 hand-off on its second call
            raise ValueError(
                f"m3_moe runs at most {HIDDEN // 32} routed CTAs (one FC2 output block each), not {self.num_ctas}"
            )
        self.m = num_tokens
        # Gate e: the M3.1 items this engine state carries; ``run`` counts each per call.
        self.items = set()
        _HALFPLAN_TAKEN[0] = False
        # One token: the trailing FC2s weigh nothing in the last phase's balance. Group 0's FC1 ranges are the same
        # for every distinct-expert count (early0).
        plan, nb_cap = make_phase_plan(
            self.num_ctas,
            num_tokens,
            M3_MOE_LAG,
            0.0 if num_tokens == 1 else 1.0,
            True,
            heavy_fc2_only=bool(HEAVY_FC2_ONLY_BY_M[num_tokens - 1]),
        )
        self.config = {
            "m_max": num_tokens,
            "num_ctas": self.num_ctas,
            "nb2_cap": nb_cap,
            "fused_router": 1,
            "timing": 0,
            "i_tp": INTER,
        }
        if HB_REPL_BY_M[num_tokens - 1] > 1:
            self.config["hb_repl"] = HB_REPL_BY_M[num_tokens - 1]
        if num_tokens <= LG_REPL_MAX_TOKENS:
            self.config["lg_repl"] = LG_REPL
        elif VRPOLL and num_tokens in VRPOLL_TOKENS:
            self.config["lg_repl"] = VRPOLL_LG_REPL
            self.items.add("MOE_VRPOLL")
            from tensorrt_llm.logger import logger as _vrpoll_log

            _vrpoll_log.info(
                f"MiniMax-M3 MoE engine VRPOLL (TP2): {VRPOLL_LG_REPL} router logit replicas, tokens={num_tokens}"
            )
        if shared:
            self.config["sh_i"] = SHARED_INTER
            self.config["shared"] = 1
            if sh_ctas != SHARED_CTAS:
                # The kernel here (the TP4 build) has no SH20 unit loop and ignores ``sh_ctas``: its grid is always
                # num_ctas + NSH. Never build an engine whose host buffers assume another shared CTA count.
                raise ValueError(
                    f"m3_moe (TP2): SH20 asks for {sh_ctas} shared CTAs, but this kernel runs the shared expert on "
                    f"{SHARED_CTAS} (no SH20 unit loop)"
                )
        self.shared = shared
        # 18 KB ring stages (A scales by cp.async into the instruction layout) per token count, unless configured
        self.config["sfa_cp"] = SFA_CP_BY_M[num_tokens - 1]
        # FC1 -> FC2 hand-off through per-group global B images + release counters, per token count
        self.config["b2img"] = B2IMG_BY_M[num_tokens - 1]
        # the finalize's all-reduce push with its call parities / descriptor preloaded, per token count
        self.config["ar_pre"] = AR_PRE_BY_M[num_tokens - 1]
        if STAGES_BY_M[num_tokens - 1] > 0:
            self.config["stages"] = STAGES_BY_M[num_tokens - 1]
        if B2_POLL_NS_BY_M[num_tokens - 1] > 0:
            self.config["b2_poll_ns"] = B2_POLL_NS_BY_M[num_tokens - 1]
        if not EARLY0_BY_M[num_tokens - 1]:
            self.config["early0"] = 0
        if flat_applies(num_tokens):
            self.config.update(FLAT_STACK)
            # engagement line (every rank; once per engine state = once per token count)
            from tensorrt_llm.logger import logger as _flog

            _flog.info(
                "MiniMax-M3 MoE engine FLAT (TP2): "
                + " ".join(f"{k}={self.config[k]}" for k in FLAT_STACK)
                + f" tokens={num_tokens}"
            )
        else:
            from tensorrt_llm.logger import logger as _flog

            _flog.info(
                f"MiniMax-M3 MoE engine FLAT (TP2) off above {FLAT_MAX_TOKENS} tokens (per-m tables): "
                f"tokens={num_tokens}"
            )
        if num_tokens in M8_CFG_BY_M and not m31_disabled("MOE_M8CFG"):
            self.config.update(M8_CFG_BY_M[num_tokens])
            self.items.add("MOE_M8CFG")
        if _HALFPLAN_TAKEN[0]:
            self.items.add("MOE_HALFPLAN")
        # E0@m4: at 4 tokens the early group-0 FC1 stages are issued by FC2 B builder warp 8 instead of the routing
        # warp 4 (issue order only, bitwise; TP4 -0.19 / TP8 -0.21 us/call).
        if num_tokens == 4:
            from tensorrt_llm.logger import logger as _e0log

            _e0log.info(
                "MiniMax-M3 MoE engine E0M4 (%s): early group-0 stages on warp 8, tokens=4" % "TP2"
            )
        try:
            self.mod = mod = _kernel_module(self.config, _ENGINE_PATH)
        except AssertionError as err:
            raise ValueError(
                f"m3_moe: num_tokens={num_tokens}, num_ctas={self.num_ctas} does not fit the shared "
                f"memory / TMEM budget (fewer CTAs means more FC2 blocks per CTA): {err}"
            ) from err
        # Every CTA waits on others (cross-CTA spins on partials, hand-off records, the shared expert's h and y): the
        # grid must be resident at once, one CTA per SM. The module's shared-memory / TMEM budget assertion makes one
        # CTA fit an SM; the grid must not exceed the SMs. (The kernel also traps after HANG_NS in any such spin.)
        if self.num_ctas + mod.NSH > sms:
            raise ValueError(
                f"m3_moe: a grid of {self.num_ctas} + {mod.NSH} CTAs exceeds the {sms} SMs; its CTAs wait on each "
                "other and must all be resident at once"
            )
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        self.partials = torch.full(
            (mod.G_CAP * mod.U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw
        )
        # the FC1 -> FC2 records; with HB_REPL > 1 copies, HB_STRIDE int32 words apart
        hb_words = mod.G_CAP * num_tokens * mod.U1 * 4
        if mod.HB_REPL > 1:
            hb_words = mod.HB_REPL * mod.HB_STRIDE
        self.hbuf32 = torch.zeros(hb_words, dtype=torch.int32, **kw)
        # One epoch / debug row per CTA of the grid (num_ctas + NSH: the TP4 kernel has no SH20 ``NSHC``).
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, 64, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        # B2IMG: per-group FC2 B images and their unit counters (int32; every group gains CNT_PER_CALL per call)
        self.img32 = torch.zeros(max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.NG_CAP * 8 + 8, dtype=torch.int32, **kw)
        # shared expert: h and y_sh as (call tag << 16 | bf16), tag 0 = never written
        self.hx = torch.zeros(num_tokens * SHARED_INTER, dtype=torch.int32, **kw)
        self.dnx = torch.zeros(num_tokens * HIDDEN, dtype=torch.int32, **kw)
        self._fn = None

    def layer(
        self,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        e_score_correction_bias,
        router_weight=None,
        fc31_input_scale=None,
        shared_weights=None,
    ) -> "M3MoeLayer":
        """One MoE layer's operands (the TRTLLM-Gen NVFP4 expert buffers, read in place, never copied).
        ``shared_weights`` (a ``shared`` state): the shared expert's (gate_up e4m3 [3072, 6144], its swizzled UE8M0
        scales, down e4m3 [6144, 768], its swizzled UE8M0 scales)."""
        layer = M3MoeLayer(
            self,
            w3_w1_weight,
            w3_w1_weight_scale,
            w2_weight,
            w2_weight_scale,
            fc31_alpha,
            fc31_scale_c,
            act_alpha,
            act_beta,
            act_clamp,
            fc2_alpha,
            e_score_correction_bias,
            router_weight,
            fc31_input_scale,
        )
        if self.shared:
            layer.shared_weights = _shared_views(shared_weights, "m3_moe")
            layer.shared_mxfp8 = tuple(shared_weights)
            if self.mod.SH_PRETILE:
                # (gate_up, pre-tiled stages, down, dummy): the kernel streams the pre-tiled copy
                gu8, _, dn8, _ = layer.shared_weights
                layer.shared_weights = (
                    gu8,
                    _pretiled(*layer.shared_weights),
                    dn8,
                    self._dummy.view(torch.uint8),
                )
        return layer

    def run(self, layer, hidden, out=None, arp=None) -> torch.Tensor:
        m = self.m
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != (m, HIDDEN)
            or not hidden.is_contiguous()
        ):
            raise ValueError(
                f"m3_moe (fused router): expects contiguous bf16 hidden states [{m}, {HIDDEN}]"
            )
        fz = (
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1),
            layer.fc31_input_scale.view(-1),
            self.lgbuf,
        )
        lgv, xq, xs = self._dummy.view(torch.float32), self._dummy, self._dummy
        if out is None:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        sh = layer.shared_weights if self.shared else (self._dummy.view(torch.uint8),) * 4
        args = (
            layer.w3_w1_weight,
            layer.w3_w1_weight_scale.view(-1),
            layer.w2_weight,
            layer.w2_weight_scale.view(-1),
            lgv,
            layer.bias,
            xq,
            xs,
            layer.escal,
            self.plan.view(-1),
            self.partials,
            self.hbuf32,
            self.epochs,
            out.view(-1),
            self.route_dbg.view(-1),
            self.tdbg.view(-1),
            *fz,
            *sh,
            self.hx,
            self.dnx,
            self._ar_off if arp is None else arp,
            layer.w3_w1_weight_scale.view(torch.uint8).reshape(-1).view(torch.int32),
            layer.w2_weight_scale.view(torch.uint8).reshape(-1).view(torch.int32),
            self.img32,
            self.cnt,
        )
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("m3_moe compiles on its first call: call it once before capture")
            import cutlass.cute as cute

            aligns = [
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                4,
                16,
                16,
                16,
                16,
                16,
                4,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
            ]
            sig = [_view(t, a) for t, a in zip(args, aligns)]
            # keyed on the config the kernel module was built with (not self.config), so a module built with any other
            # trace-time config can never reuse another build's compiled kernel
            key = (_ENGINE_PATH,) + tuple(sorted(self.mod.M3_CONFIG.items()))
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute_aot.compile_or_load(
                    f"m3_moe {os.path.basename(key[0])} {key[1:]}",
                    lambda: cute.compile(
                        self.mod.m3_moe,
                        *sig,
                        cute.runtime.make_fake_stream(),
                        options="--enable-tvm-ffi",
                    ),
                    sources=_aot_sources(),
                    tvm_ffi=True,
                )
            self._fn = fn
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        for item in sorted(self.items):
            m3_report.work(item, f"MoE engine at {m} tokens, width {INTER}")
        return out


class M3MoeLayer:
    """One MoE layer on a :class:`M3MoeState`: ``layer(hidden)``. The expert buffers are read in
    place, never copied; the per-expert scalars are packed once (layer constants: rebuild the layer after reloading
    weights)."""

    shared_weights = None  # the kernel's view of the shared expert (a shared state)
    shared_mxfp8 = None  # the shared expert's (gate_up, its scales, down, its scales) as given

    def __init__(
        self,
        state,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        bias,
        router_weight=None,
        fc31_input_scale=None,
    ):
        e = NUM_EXPERTS
        expected = {
            "w3_w1_weight": (w3_w1_weight, (e, 2 * INTER, HIDDEN // 2)),
            "w3_w1_weight_scale": (w3_w1_weight_scale, (e, 2 * INTER, HIDDEN // 16)),
            "w2_weight": (w2_weight, (e, HIDDEN, INTER // 2)),
            "w2_weight_scale": (w2_weight_scale, (e, HIDDEN, INTER // 16)),
        }
        for name, (t, shape) in expected.items():
            if t.element_size() != 1 or tuple(t.shape) != shape or not t.is_contiguous():
                raise ValueError(
                    f"m3_moe: {name} is {t.dtype} {tuple(t.shape)}, expected contiguous 1-byte {shape}"
                )
        self.state = state
        self.w3_w1_weight = w3_w1_weight.view(torch.uint8)
        self.w3_w1_weight_scale = w3_w1_weight_scale.view(torch.uint8)
        self.w2_weight = w2_weight.view(torch.uint8)
        self.w2_weight_scale = w2_weight_scale.view(torch.uint8)
        self.escal = torch.stack(
            [fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha]
        ).float()
        self.escal = self.escal.reshape(6, e).contiguous()
        self.bias = bias.float().contiguous()
        if (
            router_weight is None
            or fc31_input_scale is None
            or router_weight.dtype != torch.float32
            or tuple(router_weight.shape) != (e, HIDDEN)
            or not router_weight.is_contiguous()
            or fc31_input_scale.dtype != torch.float32
            or fc31_input_scale.numel() != 1
        ):
            raise ValueError(
                "m3_moe (fused router): needs router_weight f32 [128, 6144] and fc31_input_scale f32 [1]"
            )
        self.router_weight = router_weight
        self.fc31_input_scale = fc31_input_scale

    def __call__(
        self,
        hidden: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``hidden``: contiguous bf16 [M, 6144]. ``arp`` (int64 [8], see :func:`ar_push_desc`): also push the output
        into an all-reduce exchange."""
        return self.state.run(self, hidden, out, arp=arp)


def _aot_sources():
    """The Python files this module's CuTe DSL kernels are traced from (the target package and the shared Blackwell
    kernel helpers): the ``sources`` of their shipped AOT objects."""
    target = Path(__file__).resolve().parents[2]
    shared = Path(cute_aot.__file__).resolve().parent / "cute_dsl_kernels" / "blackwell"
    return cute_aot.py_files(str(target), str(shared))
