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
"""Host side of the fused MoE engine (kernel.py, 1..8 tokens): router, NVFP4 routed experts, MXFP8 shared expert on
extra CTAs, combine and the push into the next layer, in one launch."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

import torch

from .....runtime.knobs import KNOBS as _K
from ...knobs import knob

HIDDEN = 6144
INTER = int(
    knob("routed_inter_per_rank", "768")
)  # the routed intermediate per TP rank (G1: per process)
NUM_EXPERTS = 128
SHARED_INTER = INTER  # the shared expert's intermediate per TP rank (the routed one)
SHARED_CTAS = 24  # m3_moe's shared-expert CTAs (beside num_ctas routed CTAs)
TOP_K = 4
U1 = 2 * INTER // 32  # FC1 units per expert
MIN_CTAS = 64  # the router GEMV takes at most 2 experts per CTA (router weights held in registers)

_UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)  # one FC1 unit's weight + scale bytes
_ROW2_BYTES = INTER // 2 + INTER // 16  # one FC2 weight row's bytes (values + scales)
_S1 = 6  # FC1 stages (items) per unit
_ITEM_BYTES = _UNIT_BYTES // _S1
_modules: Dict[tuple, object] = {}
_compiled: Dict[tuple, object] = {}
# m3_moe trace-time options by token count 1..8 (index num_tokens - 1).
# ``sfa_cp``: 18 KB ring stages, the A scales copied by cp.async into the instruction layout (every token count).
SFA_CP_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
# ``b2img``: FC1 -> FC2 hand-off through per-group global B images + release counters (7 and 8 tokens).
B2IMG_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 1, 1)
# G1 (TP2, I = 1536, sm_107, real layer 10, flush): B images also win at 2, 4, 5, 6 tokens (-6 / -10 / -1 / -1 %
# random routing, -3 / -14 / -6 / -10 % at 9 distinct experts); 1 and 3 tokens stay on tagged records.
if INTER > 768:
    B2IMG_BY_M = (0, 1, 0, 1, 1, 1, 1, 1)
# ``ar_pre``: the finalize's all-reduce push with its call parities / descriptor preloaded (all but 2 tokens).
AR_PRE_BY_M: Tuple[int, ...] = (1, 0, 1, 1, 1, 1, 1, 1)
# Ring ``stages`` (0 = the kernel's default, the most that fit). 6 at 8 tokens measured 0.4-0.5 us faster than the
# default 7 (same MMAs, bit-identical outputs).
STAGES_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 6)
# ``hb_repl``: copies of the tagged FC1 -> FC2 records (applied when > 1); CTA bx polls copy bx % hb_repl, which
# spreads the polls of the few hot record lines over L2. 7 and 8 tokens hand off through the B images instead.
HB_REPL_BY_M: Tuple[int, ...] = (32, 32, 8, 8, 1, 1, 1, 1)
# ``lg_repl``: replicas of the router logit exchange at up to LG_REPL_MAX_TOKENS tokens (fewer stores before the
# exchange; bit-identical outputs). Above that the kernel's default applies.
LG_REPL = 32
LG_REPL_MAX_TOKENS = 4
# Byte-balanced FC1 ranges below one unit per CTA (:func:`_fine_ranges`) for groups too small to give every CTA a
# whole unit; False gives such groups whole units on the CTAs with the least FC2 work.
FINE_PLAN = True
# m3_moe phase lag: phase p streams FC1 of group p and FC2 of group p - M3_MOE_LAG (the kernel's default lag).
M3_MOE_LAG = int(knob("fused_phase_lag", "2"))  # H4: env override (plan and kernel together)


_ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")


def _check_dl_roles(sms: int, ncta: int, nxtra: int) -> None:
    """(V-SIM R11) The kernel's _dl_role must map the SMs one to one onto the grid's roles; it does only for a grid of
    every SM with SMs % 4 == 0 (206 / 210 / 214 pass an evenness check and give one duplicate, one missing role)."""
    roles = []
    for sm in range(sms):
        d, r = (sm >> 1) & 1, (sm >> 2) * 2 + (sm & 1)
        roles.append(
            d * (ncta // 2) + r if r < ncta // 2 else ncta + d * (nxtra // 2) + r - ncta // 2
        )
    if ncta + nxtra != sms or sms % 4 or sorted(roles) != list(range(sms)):
        raise ValueError(
            f"m3_moe dl: roles by SM need a grid of every SM ({ncta} + {nxtra} vs {sms}) and SMs % 4 == 0"
        )


B1X4_MAXNREG = 88  # K-B1X4: the C1 b1_x4 builds' register cap (88 used, 0 spill at M = 4 / 8)


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


@contextlib.contextmanager
def _maxnreg(n: int):
    """K-B1X4 (cfg maxnreg, 0 = off = no attribute): .maxnreg n on the kernel compiled inside (CuTe DSL has no launch
    knob for it; without it ptxas sizes the b1_x4 builds below their need and spills, K-B1X4.md section 4.4)."""
    if not n:
        yield
        return
    from cutlass._mlir import ir
    from cutlass.cutlass_dsl import cutlass as _cd

    real = _cd.CuTeDSL._generate_kernel_attrs

    def gen(self, config):
        ret = real(self, config)
        ret["nvvm.maxnreg"] = ir.Attribute.parse(f"{int(n)} : i32")
        return ret

    _cd.CuTeDSL._generate_kernel_attrs = gen
    try:
        yield
    finally:
        _cd.CuTeDSL._generate_kernel_attrs = real


def is_supported(device: Optional[torch.device] = None) -> tuple:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    cc = torch.cuda.get_device_capability(device)
    # sm_107 (SM107) added for the standalone port; the device code carries no arch guard.
    if cc not in ((10, 0), (10, 3), (10, 7)):
        return (
            False,
            f"needs sm_100/sm_103/sm_107 (tcgen05 block-scaled NVFP4 MMA), got sm_{cc[0]}{cc[1]}",
        )
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


def ar_push_desc_g3(
    bufs_ptrs: torch.Tensor, flags_ptr: int, rank: int, world: int, device
) -> torch.Tensor:
    """G1: ``arp`` for a state built with ``ar_g3=world``: push the finalized rows into G3's unicast boundary slots
    (lanes/G3-comm/API.md section 2). ``bufs_ptrs``: int64 device tensor [world] of every rank's slot-buffer base
    (this process's mapped addresses), ``flags_ptr``: this rank's Flags (call counter at offset 0)."""
    assert bufs_ptrs.dtype == torch.int64 and bufs_ptrs.numel() == world and bufs_ptrs.is_cuda
    return torch.tensor(
        [1, bufs_ptrs.data_ptr(), flags_ptr, rank, world, 48, 0, 0],
        dtype=torch.int64,
        device=device,
    )


def push_params_v3(pp_bytes: torch.Tensor, device):
    """H7: (arp, peers, (cluster, ndst)) from G3's ext.push_params_bytes(ws) (prar::PushParams: int* peers[8],
    const int* flags, int rank, world, maxt, cluster, ndst). Keep ``peers`` alive as long as ``arp``."""
    import struct

    b = bytes(pp_bytes.cpu().numpy().tobytes())
    peers = struct.unpack_from("<8Q", b, 0)
    flags = struct.unpack_from("<Q", b, 64)[0]
    rank, world, maxt, cluster, ndst = struct.unpack_from("<5i", b, 72)
    pt = torch.tensor(list(peers[:ndst]), dtype=torch.int64, device=device)
    arp = torch.tensor(
        [1, pt.data_ptr(), flags, rank, world, maxt, 0, 0], dtype=torch.int64, device=device
    )
    return arp, pt, (cluster, ndst)


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
    dt = H_ // n_cta // 128  # down tiles per shared CTA
    out = torch.empty(n_cta, gt + dt * dkt, 16384 + 512, dtype=torch.uint8, device=dev)
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
        for gd in range(dt * dkt):
            jt, kt = gd // dkt, gd % dkt
            row0 = (H_ // n_cta) * s + 128 * jt
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


def _pretiled(gu8, gusf8, dn8, dnsf8, n_cta: int = SHARED_CTAS) -> torch.Tensor:
    key = (gu8.device, gu8.data_ptr(), gusf8.data_ptr(), dn8.data_ptr(), dnsf8.data_ptr(), n_cta)
    hit = _PRETILED.get(key)
    if hit is None:
        hit = _PRETILED[key] = (
            (gu8, gusf8, dn8, dnsf8),
            _pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta),
        )
    return hit[1]


def _fine_ranges(weights, total: int):
    """Contiguous FC1 item ranges [s_c, e_c) over the CTAs, sized as close to ``weights`` (ideal real-valued item
    counts summing to ``total``) as the split-unit protocol allows: every unit (S1 items) has at most one CTA boundary
    strictly inside it (its head and tail parts, summed by the tail's holder), so a range that starts inside a unit
    runs at least to that unit's end. Returns the item counts."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
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
    nb2: Optional[int] = None,
    units: int = U1,
    ioff: int = 0,
    boff: int = 0,
    zero_ok: bool = False,
):
    """The work split of ``m3_moe`` for each distinct-expert count G (4 .. 4 m_max): FC2 32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..) a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work, or (``fine``) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit (:func:`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and the largest block count."""
    gcap = TOP_K * m_max
    nb2 = nb2 or HIDDEN // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    blk_bytes = 32 * _ROW2_BYTES  # one block, one slot
    for g in range(TOP_K, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = (boff + b0s[c]) | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * units * _S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and not (
                    early0 and p == 0 and m_max >= 2
                ):  # early0: group 0's ranges the same for every G
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            items = [0] * num_ctas
            if total >= _S1 * num_ctas:

                def counts_for(limit):
                    # H9 bal (zero_ok): a CTA whose FC2 share already exceeds the level takes no FC1 stages this
                    # phase (an empty range, as small groups use) instead of the 6-stage floor
                    if zero_ok:
                        return [
                            v if v >= _S1 else 0 for v in ((limit - f) // _ITEM_BYTES for f in fc2)
                        ]
                    return [max(_S1, (limit - f) // _ITEM_BYTES) for f in fc2]

                lo, hi = 0, max(fc2) + total * _ITEM_BYTES
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
                    plan[g, c, p] = (ioff + i0) | ((ioff + i0 + items[c]) << 16)
                i0 += items[c]
            assert i0 == total
    return plan, max(nbs)


def make_phase_plan_dl(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    zero_ok: bool = False,
):
    """H4 DL: die d's routed roles [d N / 2, (d + 1) N / 2) get only uGPU shard d: FC2 output blocks [d H / 64, ..) and
    the FC1 units [d U1 / 2, ..) of every slot, at items [d 2 U1 S1, ..) (the kernel's shard-major unit decode)."""
    if num_ctas % 2:
        raise ValueError("m3_moe dl: even routed CTA count")
    h = num_ctas // 2
    parts = [
        make_phase_plan(
            h,
            m_max,
            lag,
            tail_w,
            early0,
            nb2=HIDDEN // 64,
            units=U1 // 2,
            ioff=d * 2 * U1 * _S1,
            boff=d * (HIDDEN // 64),
            zero_ok=zero_ok,
        )
        for d in range(2)
    ]
    return torch.cat([pl for pl, _ in parts], dim=1), max(n for _, n in parts)


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
        ar_g3: int = 0,
        sharded: bool = False,
        extra_cfg: Optional[dict] = None,
        sharded_scales_full: bool = False,
        n_shared: int = SHARED_CTAS,
        ar_v3: Optional[tuple] = None,
    ):
        """``inter``: the routed intermediate per rank (M3_G1_ITP). ``ar_g3``: world size of G3's unicast boundary to
        push into (0 = off; the PR's multicast ``arp`` path stays available)."""
        if inter != INTER:
            raise ValueError(
                f"m3_moe (TP8): routed intermediate {inter} per rank is not supported (only {INTER})"
            )
        self.inter = inter
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3MoeState allocates and compiles: build it before CUDA-graph capture"
            )
        sms_all = torch.cuda.get_device_properties(device).multi_processor_count
        self.n_shared = int(n_shared)
        if shared and (num_ctas or sms_all) + self.n_shared > sms_all:
            raise ValueError(
                f"m3_moe (shared): num_ctas + {self.n_shared} shared-expert CTAs must fit the {sms_all} SMs "
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
        self.m = num_tokens
        # One token: the trailing FC2s weigh nothing in the last phase's balance. Group 0's FC1 ranges are the same
        # for every distinct-expert count (early0).
        _dl = bool((extra_cfg or {}).get("dl"))
        if _dl and self.num_ctas + n_shared != sms:
            raise ValueError(
                f"m3_moe dl: roles by SM need {self.num_ctas} + {n_shared} == {sms} CTAs"
            )
        plan, nb_cap = (make_phase_plan_dl if _dl else make_phase_plan)(
            self.num_ctas,
            num_tokens,
            M3_MOE_LAG,
            0.0 if num_tokens == 1 else 1.0,
            True,
            zero_ok=bool((extra_cfg or {}).get("bal"))
            or bool(__import__("json").loads(knob("fused_state_cfg_overlay") or "{}").get("bal")),
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
        if shared:
            self.config["sh_i"] = SHARED_INTER
            self.config["shared"] = 1
            # G1: the shared expert's SiLU exactly as vLLM's silu_and_mul_with_clamp (bit-exact h)
            self.config["silu_vllm"] = 1
            if self.n_shared != SHARED_CTAS:
                self.config["nsh"] = self.n_shared
            if INTER > 768 and num_tokens >= 5:
                # G1 (TP2): with 24 shared CTAs the 5..8-token ring cannot also stage x (x read from global instead)
                self.config["xstage"] = 0
        self.shared = shared
        # 18 KB ring stages (A scales by cp.async into the instruction layout) per token count, unless configured
        self.config["sfa_cp"] = SFA_CP_BY_M[num_tokens - 1]
        # FC1 -> FC2 hand-off through per-group global B images + release counters, per token count
        self.config["b2img"] = B2IMG_BY_M[num_tokens - 1]
        # the finalize's all-reduce push with its call parities / descriptor preloaded, per token count
        self.config["ar_pre"] = AR_PRE_BY_M[num_tokens - 1]
        if STAGES_BY_M[num_tokens - 1] > 0:
            self.config["stages"] = STAGES_BY_M[num_tokens - 1]
        # G1: read the uGPU-localized split-N shards in place (layer(..., shard1=...)); no full buffers needed
        self.sharded = bool(sharded)
        if sharded:
            self.config.update({"sharded": 1, "sfa_cp": 1})
            if sharded_scales_full:
                # weights from the two shards, block scales from one full-layout copy (cat of the shards' scales):
                # the per-shard scale select costs ~6 % at 4 rows, the copy ~151 MB per layer at TP2
                self.config["sh_sf_off"] = 1
        self.sharded_scales_full = bool(sharded and sharded_scales_full)
        # G1: router logits bit-identical to vLLM's ops.fp32_router_gemm (its K partition and reduction order); the
        # PR's own GEMV order differs from it in the last bit on ~80 % of logits
        self.config["router_vllm"] = 1
        # G1: push into G3's unicast boundary slots (world 2 or 4); the arp must come from ar_push_desc_g3
        if ar_g3:
            self.config.update({"ar_g3": 1, "ar_w": int(ar_g3), "ar_pre": 1})
        # H7: push into G3 v3 prx slots; ar_v3 = (cluster, ndst) from the workspace's PushParams
        if ar_v3:
            self.config.update(
                {"ar_v3": 1, "ar_cl": int(ar_v3[0]), "ar_nd": int(ar_v3[1]), "ar_pre": 1}
            )
        # G1: debug builds and per-key trace-time overrides from the environment
        if M3_MOE_LAG != 2:
            self.config["lag"] = M3_MOE_LAG
        for _k, _env in (("hang_dbg", "M3_HANG_DBG"), ("timing", "M3_TIMING")):
            if _K.get(_env):
                self.config[_k] = 1
        if extra_cfg:
            self.config.update(extra_cfg)
        if knob("fused_state_cfg_overlay"):
            import json as _json

            self.config.update(_json.loads(knob("fused_state_cfg_overlay")))
        # W6 ext_route: routes from outside (1 = the post's tagged route lines, 2 = a static table)
        self.ext_route = int(self.config.get("ext_route", 0))
        if self.ext_route and (
            int(self.config.get("xstage", 1)) or not int(self.config.get("pad_x", 0))
        ):
            raise ValueError(
                "m3_moe ext_route: needs xstage=0 and pad_x=1 (the ring is free at entry; CTA 0's pad "
                "runs after the grid wait)"
            )
        # K-B1X4: B1 from the post's NVFP4 bytes (x4 / x4sf, lbh.x4_bufs), read after the grid wait
        self.b1_x4 = int(self.config.get("b1_x4", 0))
        if self.b1_x4 and not self.ext_route:
            raise ValueError(
                "m3_moe b1_x4: needs ext_route (the post that writes x4 is the hoist's lbv3r post)"
            )
        try:
            self.mod = mod = _kernel_module(self.config, _ENGINE_PATH)
        except AssertionError as err:
            raise ValueError(
                f"m3_moe: num_tokens={num_tokens}, num_ctas={self.num_ctas} does not fit the shared "
                f"memory / TMEM budget (fewer CTAs means more FC2 blocks per CTA): {err}"
            ) from err
        if (
            self.b1_x4
        ):  # (K-B1X4: ptxas otherwise picks 80 registers and spills at M = 4; M = 1 needs 114)
            self.config.setdefault("maxnreg", 120 if num_tokens == 1 else B1X4_MAXNREG)
        if mod.DL:
            _check_dl_roles(sms, self.num_ctas, getattr(mod, "NXTRA", mod.NSH))
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
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, mod.TD_STRIDE, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        # B2IMG: per-group FC2 B images and their unit counters (int32; every group gains CNT_PER_CALL per call)
        self.img32 = torch.zeros(max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.B2CNT_WORDS, dtype=torch.int32, **kw)  # H9 m19: B2_REPL copies
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
        shard1=None,
        scales_full=None,
        route_lines=None,
        rt_tab=None,
        rctr=None,
        x4=None,
        x4sf=None,
    ) -> "M3MoeLayer":
        """One MoE layer's operands (the TRTLLM-Gen NVFP4 expert buffers, read in place, never copied).
        ``shared_weights`` (a ``shared`` state): the shared expert's (gate_up e4m3 [768, 6144], its swizzled UE8M0
        scales, down e4m3 [6144, 384], its swizzled UE8M0 scales). W6 ext_route: ``route_lines`` (ext_route=1, int64
        [route_repl][route_maxt][16] or this layer's RouteBuf) / ``rt_tab`` (ext_route=2, int32 [M][4]), the
        layer's default for :meth:`run`. ``rctr`` (ext_route=1, required unless route_lines is
        a RouteBuf): the post's row counters, int32 [>= M][32] (RouteBuf.rctr), checked after the grid wait (R9).
        ``x4`` / ``x4sf`` (b1_x4=1): the post's NVFP4 B1 activation, uint8 [>= M][3072] / [>= M][384]
        (lbh.x4_bufs(blk).q / .sf; an X4Bufs as x4 gives both), the layer's default for :meth:`run`."""
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
            shard1=shard1 if self.sharded else None,
        )
        layer.route_lines, layer.rt_tab, layer.rctr = route_lines, rt_tab, rctr
        if route_lines is not None or rt_tab is not None or rctr is not None:
            self._ext_args(route_lines, rt_tab, rctr)
        if x4 is not None and x4sf is None and hasattr(x4, "sf"):  # an lbh.X4Bufs
            x4, x4sf = x4.q, x4.sf
        layer.x4, layer.x4sf = x4, x4sf
        if x4 is not None or x4sf is not None:
            self._x4_args(x4, x4sf)
        if self.sharded_scales_full:
            if scales_full is None:
                scales_full = (
                    torch.cat([w3_w1_weight_scale, shard1[1]], 1),
                    torch.cat([w2_weight_scale, shard1[3]], 1),
                )
            layer.w3_w1_weight_scale = scales_full[0].view(torch.uint8).contiguous()
            layer.w2_weight_scale = scales_full[1].view(torch.uint8).contiguous()
            layer.s1wb = layer.w3_w1_weight_scale.reshape(-1).view(torch.int32)
            layer.s2wb = layer.w2_weight_scale.reshape(-1).view(torch.int32)
            layer.scales_full = scales_full
            if self.config.get(
                "sfa_bulk"
            ):  # H4 DL: the spliced FC1 scale image of the full-layout scales
                r = layer.w3_w1_weight_scale.reshape(-1).view(torch.int32).view(-1, 4, 4, 32, 4)
                layer.sfa1_img = r.permute(0, 4, 2, 3, 1).contiguous().view(-1)
        if self.sharded and shard1 is None:
            raise ValueError(
                "m3_moe (sharded): pass shard1=(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale) "
                "of partition 1; the positional expert tensors are partition 0"
            )
        if self.shared:
            layer.shared_weights = _shared_views(shared_weights, "m3_moe")
            layer.shared_mxfp8 = tuple(shared_weights)
            if self.mod.SH_PRETILE:
                # (gate_up, pre-tiled stages, down, dummy): the kernel streams the pre-tiled copy
                gu8, _, dn8, _ = layer.shared_weights
                layer.shared_weights = (
                    gu8,
                    _pretiled(*layer.shared_weights, self.n_shared),
                    dn8,
                    self._dummy.view(torch.uint8),
                )
        return layer

    def _ext_args(self, route_lines, rt_tab, rctr=None):
        """(lgbuf slot, x32 slot) for ext_route: route_lines rides in lgbuf's slot (unused without the router GEMV),
        rt_tab (ext_route=2) or the post's row counters rctr (ext_route=1, K-SAFE R9) in x32's (unused with the fused
        router), so the kernel signature stays the m19 one. route_lines: int64 [route_repl][route_maxt][16] (e.g.
        RouteBuf.route.view(rb.repl, rb.maxt, 16)) or a RouteBuf (.route, .repl, .maxt, .rctr); rctr: int32 [>= M][32]
        (RouteBuf.rctr; taken from a RouteBuf route_lines when not given); rt_tab: int32 [>= M][4]. All 16-byte
        aligned (the kernel views assume it)."""
        mod = self.mod
        if not self.ext_route:
            if route_lines is not None or rt_tab is not None or rctr is not None:
                raise ValueError(
                    "m3_moe: route_lines / rt_tab / rctr need a state built with ext_route=1 / 2"
                )
            return self.lgbuf, self._dummy
        if self.ext_route == 1:
            want = (mod.ROUTE_REPL, mod.ROUTE_MAXT, 16)
            if route_lines is not None and hasattr(route_lines, "route"):  # a RouteBuf
                if (route_lines.repl, route_lines.maxt) != want[:2]:
                    raise ValueError(
                        f"m3_moe ext_route=1: RouteBuf (repl {route_lines.repl}, maxt {route_lines.maxt}) "
                        f"!= cfg (route_repl {want[0]}, route_maxt {want[1]})"
                    )
                if rctr is None:
                    rctr = route_lines.rctr
                route_lines = route_lines.route.view(*want)
            if (
                route_lines is None
                or route_lines.dtype != torch.int64
                or not route_lines.is_contiguous()
                or tuple(route_lines.shape) != want
                or route_lines.device != self.device
            ):
                got = None if route_lines is None else (route_lines.dtype, tuple(route_lines.shape))
                raise ValueError(
                    f"m3_moe ext_route=1: route_lines must be contiguous int64 {list(want)} "
                    f"(cfg route_repl / route_maxt) on {self.device}, got {got}"
                )
            if route_lines.data_ptr() % 16:
                raise ValueError("m3_moe ext_route=1: route_lines must be 16-byte aligned")
            if (
                rctr is None
                or rctr.dtype != torch.int32
                or not rctr.is_contiguous()
                or rctr.numel() < self.m * 32
                or rctr.device != self.device
                or rctr.data_ptr() % 16
            ):
                raise ValueError(
                    f"m3_moe ext_route=1: rctr (the post's row counters, RouteBuf.rctr) must be contiguous "
                    f"16-byte aligned int32 with >= {self.m * 32} words on {self.device}"
                )
            return route_lines.view(-1), rctr.view(-1)[
                : self.m * 32
            ]  # one shape per compiled config
        if rctr is not None:
            raise ValueError(
                "m3_moe: rctr is the ext_route=1 post check (no route lines at ext_route=2)"
            )
        if (
            rt_tab is None
            or rt_tab.dtype != torch.int32
            or not rt_tab.is_contiguous()
            or rt_tab.numel() < self.m * TOP_K
            or rt_tab.device != self.device
            or (rt_tab.dim() == 2 and rt_tab.shape[1] != TOP_K)
            or rt_tab.dim() > 2
        ):
            raise ValueError(
                f"m3_moe ext_route=2: rt_tab must be contiguous int32 [>= {self.m}][4] on {self.device}"
            )
        if rt_tab.data_ptr() % 16:
            raise ValueError("m3_moe ext_route=2: rt_tab must be 16-byte aligned")
        return self.lgbuf, rt_tab.view(-1)[: self.m * TOP_K]  # one shape per compiled config

    def _x4_args(self, x4, x4sf):
        """(logits slot, xsf32 slot): b1_x4 builds read the post's x4 values / scales there (both unused with the fused
        router); x4: uint8 [>= M][3072] (or an lbh.X4Bufs), x4sf: uint8 [>= M][384], contiguous, 16-byte aligned. The
        first M rows ride as int32 words (one shape per compiled config); rows >= M are never read."""
        if not getattr(self, "b1_x4", 0):
            if x4 is not None or x4sf is not None:
                raise ValueError("m3_moe: x4 / x4sf need a state built with b1_x4=1")
            return self._dummy.view(torch.float32), self._dummy
        if x4 is not None and x4sf is None and hasattr(x4, "sf"):
            x4, x4sf = x4.q, x4.sf
        for nm, t, w in (("x4", x4, HIDDEN // 2), ("x4sf", x4sf, HIDDEN // 16)):
            if (
                t is None
                or t.dtype != torch.uint8
                or t.dim() != 2
                or t.shape[1] != w
                or t.shape[0] < self.m
                or not t.is_contiguous()
                or t.device != self.device
                or t.data_ptr() % 16
            ):
                got = None if t is None else (t.dtype, tuple(t.shape), t.data_ptr() % 16)
                raise ValueError(
                    f"m3_moe b1_x4: {nm} must be contiguous 16-byte aligned uint8 [>= {self.m}][{w}] "
                    f"on {self.device} (lbh.x4_bufs(blk)), got {got}"
                )
        if self.mod.FC1_PER_TOKEN:  # s_t rides in row max(m, 16) of the scale buffer
            x4r = max(self.m, 16)
            if x4sf.shape[0] != x4r + 1:
                raise ValueError(
                    f"m3_moe fc1_per_token: x4sf must be [{x4r + 1}][{HIDDEN // 16}] (row {x4r} = fp32 s_t), "
                    f"got {tuple(x4sf.shape)}"
                )
            return x4.view(-1)[: self.m * (HIDDEN // 2)].view(torch.int32), x4sf.view(-1).view(
                torch.int32
            )
        return x4.view(-1)[: self.m * (HIDDEN // 2)].view(torch.int32), x4sf.view(-1)[
            : self.m * (HIDDEN // 16)
        ].view(torch.int32)

    def run(
        self,
        layer,
        hidden,
        out=None,
        arp=None,
        route_lines=None,
        rt_tab=None,
        rctr=None,
        x4=None,
        x4sf=None,
    ) -> torch.Tensor:
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
        lgb, xq = self.lgbuf, self._dummy
        if self.ext_route:
            lgb, xq = self._ext_args(
                layer.route_lines if route_lines is None else route_lines,
                layer.rt_tab if rt_tab is None else rt_tab,
                getattr(layer, "rctr", None) if rctr is None else rctr,
            )
        elif route_lines is not None or rt_tab is not None or rctr is not None:
            self._ext_args(route_lines, rt_tab, rctr)
        fz = (
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1),
            layer.fc31_input_scale.view(-1),
            lgb,
        )
        lgv, xs = self._x4_args(
            getattr(layer, "x4", None) if x4 is None else x4,
            getattr(layer, "x4sf", None) if x4sf is None else x4sf,
        )
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
            layer.sfa1_img
            if layer.sfa1_img is not None
            else layer.w3_w1_weight_scale.view(torch.uint8).reshape(-1).view(torch.int32),
            layer.w2_weight_scale.view(torch.uint8).reshape(-1).view(torch.int32),
            self.img32,
            self.cnt,
            layer.a1b,
            layer.a2b,
            layer.s1wb,
            layer.s2wb,
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
                16,
                16,
                16,
                16,
            ]
            sig = [_view(t, a) for t, a in zip(args, aligns)]
            key = (_ENGINE_PATH,) + tuple(sorted(self.config.items()))
            fn = _compiled.get(key)
            if fn is None:
                with _maxnreg(int(self.config.get("maxnreg", 0))):
                    fn = _compiled[key] = cute.compile(
                        self.mod.m3_moe,
                        *sig,
                        cute.runtime.make_fake_stream(),
                        options="--enable-tvm-ffi",
                    )
            self._fn = fn
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        return out


class M3MoeLayer:
    """One MoE layer on a :class:`M3MoeState`: ``layer(hidden)``. The expert buffers are read in
    place, never copied; the per-expert scalars are packed once (layer constants: rebuild the layer after reloading
    weights)."""

    shared_weights = None  # the kernel's view of the shared expert (a shared state)
    shared_mxfp8 = None  # the shared expert's (gate_up, its scales, down, its scales) as given
    route_lines = None  # W6 ext_route=1: this layer's route lines (lgbuf slot)
    rt_tab = None  # W6 ext_route=2: this layer's route table (x32 slot)
    rctr = None  # K-SAFE ext_route=1: the post's row counters (x32 slot)
    x4 = None  # K-B1X4 b1_x4=1: the post's x4 values (logits slot)
    x4sf = None  # K-B1X4 b1_x4=1: the post's x4 scales (xsf32 slot)

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
        shard1=None,
    ):
        e = NUM_EXPERTS
        r1 = INTER if shard1 is not None else 2 * INTER  # FC1 rows per expert per buffer
        r2 = HIDDEN // 2 if shard1 is not None else HIDDEN  # FC2 rows per expert per buffer
        expected = {
            "w3_w1_weight": (w3_w1_weight, (e, r1, HIDDEN // 2)),
            "w3_w1_weight_scale": (w3_w1_weight_scale, (e, r1, HIDDEN // 16)),
            "w2_weight": (w2_weight, (e, r2, INTER // 2)),
            "w2_weight_scale": (w2_weight_scale, (e, r2, INTER // 16)),
        }
        if shard1 is not None:
            for nm, t in zip(
                ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale"), shard1
            ):
                expected[nm + "[1]"] = (t, expected[nm][1])
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
        if shard1 is not None:
            b13, b13s, b2, b2s = (t.view(torch.uint8) for t in shard1)
        else:
            b13, b13s, b2, b2s = (
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w2_weight,
                self.w2_weight_scale,
            )
        self.shard1 = shard1
        self.a1b, self.a2b = b13, b2
        self.s1wb = b13s.reshape(-1).view(torch.int32)
        self.s2wb = b2s.reshape(-1).view(torch.int32)
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
        # H4 SFA_BULK: the FC1 scale words pre-spliced into the instruction-atom order, 2 KB per (16-atom raw group,
        # unit % 4), so a stage's scales are one bulk copy (not 16 4-byte cp.async per lane)
        self.sfa1_img = None
        if state.config.get("sfa_bulk") and shard1 is None:
            r = (
                self.w3_w1_weight_scale.reshape(-1).view(torch.int32).view(-1, 4, 4, 32, 4)
            )  # [g, q, kb, lane, cm]
            self.sfa1_img = r.permute(0, 4, 2, 3, 1).contiguous().view(-1)  # [g, cm, kb, lane, q]

    def __call__(
        self,
        hidden: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
        route_lines: Optional[torch.Tensor] = None,
        rt_tab: Optional[torch.Tensor] = None,
        rctr: Optional[torch.Tensor] = None,
        x4: Optional[torch.Tensor] = None,
        x4sf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``hidden``: contiguous bf16 [M, 6144]. ``arp`` (int64 [8], see :func:`ar_push_desc`): also push the output
        into an all-reduce exchange. ``route_lines`` / ``rt_tab`` / ``rctr`` (ext_route states), ``x4`` / ``x4sf``
        (b1_x4 states): override the layer's."""
        return self.state.run(
            self,
            hidden,
            out,
            arp=arp,
            route_lines=route_lines,
            rt_tab=rt_tab,
            rctr=rctr,
            x4=x4,
            x4sf=x4sf,
        )
