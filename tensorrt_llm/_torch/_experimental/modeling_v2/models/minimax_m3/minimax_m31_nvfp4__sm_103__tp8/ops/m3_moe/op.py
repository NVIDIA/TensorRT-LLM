# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 TP8 implementation."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

import torch

from ...m31_switches import m31_disabled

HIDDEN = 6144
INTER = 384
NUM_EXPERTS = 128
SHARED_INTER = 384
SHARED_CTAS = 24
TOP_K = 4
U1 = 2 * INTER // 32
MIN_CTAS = 64
_UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)
_ROW2_BYTES = INTER // 2 + INTER // 16
_S1 = 6
_ITEM_BYTES = _UNIT_BYTES // _S1
_modules: Dict[tuple, object] = {}
_compiled: Dict[tuple, object] = {}
SFA_CP_BY_M: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
B2IMG_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 1, 1)
AR_PRE_BY_M: Tuple[int, ...] = (1, 0, 1, 1, 1, 1, 1, 1)
STAGES_BY_M: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 6)
HB_REPL_BY_M: Tuple[int, ...] = (32, 32, 8, 8, 1, 1, 1, 1)
M8_CFG_BY_M: Dict[int, Dict[str, int]] = {8: {"b2img": 0}}
B1WIDE_TOKENS = frozenset({8})
LG_REPL = 32
LG_REPL_MAX_TOKENS = 4
LG_REPL_BY_M: Dict[int, int] = {8: 16}
FINE_PLAN = True
PLAN_FC2_W = 2.0
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
            + "_".join((f"{k}{v}" for k, v in key[1:]))
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
        return (False, "no CUDA")
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
        return (False, "needs the sm_100 family (tcgen05 block-scaled NVFP4 MMA)")
    try:
        import cutlass  # noqa: F401 - dependency availability check
    except ImportError:
        return (False, "CuTe DSL (nvidia-cutlass-dsl) is not installed")
    return (True, "")


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
        if t.dtype != torch.float8_e4m3fn or tuple(t.shape) != shape or (not t.is_contiguous()):
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
    H_, I_ = (HIDDEN, SHARED_INTER)
    P = I_ // n_cta
    split = 128 // (2 * P)
    gt = H_ // 128 // split
    dkt = I_ // 128
    gu = gu8.view(2 * I_, H_)
    dn = dn8.view(H_, I_)
    gkb, dkb = (H_ // 32, I_ // 32)

    def sf_bytes(sf, n, b, kbn):
        return sf[
            ((n >> 7) * (kbn // 4) + (b >> 2)) * 512 + (n & 31) * 16 + (n >> 5 & 3) * 4 + (b & 3)
        ]

    r = torch.arange(128, device=dev)
    q = r >> 5
    ri = r & 31
    octant = r // P
    chunk = torch.arange(8, device=dev)
    swz = chunk[None, :] ^ r[:, None] & 7
    out = torch.empty(n_cta, gt + 2 * dkt, 16384 + 512, dtype=torch.uint8, device=dev)
    for s in range(n_cta):
        nrow = (octant & 1) * I_ + P * s + r % P
        sl = octant >> 1
        for g in range(gt):
            kt = sl * gt + g
            tile = gu[nrow[:, None], kt[:, None] * 128 + torch.arange(128, device=dev)[None, :]]
            img = tile.view(128, 8, 16)[r[:, None], swz]
            out[s, g, :16384] = img.reshape(-1)
            kb = torch.arange(4, device=dev)
            codes = sf_bytes(gusf8, nrow[:, None], kt[:, None] * 4 + kb[None, :], gkb)
            atom = torch.empty(512, dtype=torch.uint8, device=dev)
            atom[(ri[:, None] * 16 + q[:, None] * 4 + kb[None, :]).reshape(-1)] = codes.reshape(-1)
            out[s, g, 16384:] = atom
        for gd in range(2 * dkt):
            jt, kt = (gd // dkt, gd % dkt)
            row0 = 256 * s + 128 * jt
            tile = dn[row0 : row0 + 128, kt * 128 : (kt + 1) * 128]
            img = tile.reshape(128, 8, 16)[r[:, None], swz]
            out[s, gt + gd, :16384] = img.reshape(-1)
            out[s, gt + gd, 16384:] = dnsf8[
                ((row0 >> 7) * (dkb // 4) + kt) * 512 : ((row0 >> 7) * (dkb // 4) + kt) * 512 + 512
            ]
    return out.view(-1)


_PRETILED: Dict[tuple, tuple] = {}


def _pretiled(gu8, gusf8, dn8, dnsf8) -> torch.Tensor:
    key = (gu8.device, gu8.data_ptr(), gusf8.data_ptr(), dn8.data_ptr(), dnsf8.data_ptr())
    hit = _PRETILED.get(key)
    if hit is None:
        hit = _PRETILED[key] = ((gu8, gusf8, dn8, dnsf8), _pretile_shared(gu8, gusf8, dn8, dnsf8))
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
        if s % _S1 and e < total:
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
    fc2_w: float = 1.0,
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
    blk_bytes = int(round(32 * _ROW2_BYTES * fc2_w))
    for g in range(TOP_K, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = b0s[c] | nbs[c] << 16
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * U1 * _S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and (not (early0 and p == 0 and (m_max >= 2))):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            items = [0] * num_ctas
            if total >= _S1 * num_ctas:

                def counts_for(limit):
                    return [max(_S1, (limit - f) // _ITEM_BYTES) for f in fc2]

                lo, hi = (0, max(fc2) + total * _ITEM_BYTES)
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
                lo_f, hi_f = (0.0, float(max(fc2) + total * _ITEM_BYTES))
                for _ in range(100):
                    mid = 0.5 * (lo_f + hi_f)
                    if sum((max(0.0, (mid - f) / _ITEM_BYTES) for f in fc2)) >= total:
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
                    plan[g, c, p] = i0 | i0 + items[c] << 16
                i0 += items[c]
            assert i0 == total
    return (plan, max(nbs))


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
        """``inter``: the routed intermediate per rank; this engine is built for INTER (384) only."""
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
        if shared and (num_ctas or sms_all) + SHARED_CTAS > sms_all:
            raise ValueError(
                (
                    "m3_moe (shared): num_ctas + "
                    f"{SHARED_CTAS}"
                    " shared-expert CTAs must fit the "
                    f"{sms_all}"
                    " SMs (all resident at once)"
                )
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
        plan, nb_cap = make_phase_plan(
            self.num_ctas,
            num_tokens,
            M3_MOE_LAG,
            0.0 if num_tokens == 1 else 1.0,
            True,
            fc2_w=1.0 if m31_disabled("PLANW") else PLAN_FC2_W,
        )
        if not m31_disabled("PLANW") and PLAN_FC2_W != 1.0:
            from tensorrt_llm.logger import logger as _planw_log

            _planw_log.info(
                (
                    "MiniMax-M3.1 MoE engine PLAN-W: FC2 bytes weighted "
                    f"{PLAN_FC2_W}"
                    " in the phase plan (tokens="
                    f"{num_tokens}"
                    ")"
                )
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
        elif num_tokens in LG_REPL_BY_M and (not m31_disabled("MOE_LGREPL16")):
            self.config["lg_repl"] = LG_REPL_BY_M[num_tokens]
            from tensorrt_llm.logger import logger as _lg_log

            _lg_log.info(
                (
                    "MiniMax-M3.1 MoE engine MOE_LGREPL16: "
                    f"{LG_REPL_BY_M[num_tokens]}"
                    " router-logit exchange replicas (tokens="
                    f"{num_tokens}"
                    ")"
                )
            )
        if shared:
            self.config["sh_i"] = SHARED_INTER
            self.config["shared"] = 1
        self.shared = shared
        self.config["sfa_cp"] = SFA_CP_BY_M[num_tokens - 1]
        self.config["b2img"] = B2IMG_BY_M[num_tokens - 1]
        self.config["ar_pre"] = AR_PRE_BY_M[num_tokens - 1]
        if STAGES_BY_M[num_tokens - 1] > 0:
            self.config["stages"] = STAGES_BY_M[num_tokens - 1]
        if num_tokens in M8_CFG_BY_M and (not m31_disabled("MOE_M8CFG")):
            self.config.update(M8_CFG_BY_M[num_tokens])
        if (
            num_tokens in B1WIDE_TOKENS
            and self.config.get("fused_router")
            and (not m31_disabled("MOE_B1WIDE"))
        ):
            self.config["b1wide"] = 1
            from tensorrt_llm.logger import logger as _b1w_log

            _b1w_log.info(
                f"MiniMax-M3.1 MoE engine B1WIDE: B1 quantize on every warp (tokens={num_tokens})"
            )
        if not m31_disabled("MOE_TAIL_TRIGGER"):
            self.config["tail_trigger"] = 1
        for key in ("tail_trigger",):
            if self.config.get(key):
                from tensorrt_llm.logger import logger as _rel_log

                _rel_log.info(
                    f"MiniMax-M3.1 MoE engine dependent release: {key} (tokens={num_tokens}, epoch stored early)"
                )
        try:
            self.mod = mod = _kernel_module(self.config, _ENGINE_PATH)
        except AssertionError as err:
            raise ValueError(
                (
                    "m3_moe: num_tokens="
                    f"{num_tokens}"
                    ", num_ctas="
                    f"{self.num_ctas}"
                    " does not fit the shared memory / TMEM budget (fewer CTAs means "
                    "more FC2 blocks per CTA): "
                    f"{err}"
                )
            ) from err
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        self.partials = torch.full(
            (mod.G_CAP * mod.U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw
        )
        hb_words = mod.G_CAP * num_tokens * mod.U1 * 4
        if mod.HB_REPL > 1:
            hb_words = mod.HB_REPL * mod.HB_STRIDE
        self.hbuf32 = torch.zeros(hb_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, 64, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        self.img32 = torch.zeros(max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.NG_CAP * 8 + 8, dtype=torch.int32, **kw)
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
        ``shared_weights`` (a ``shared`` state): the shared expert's (gate_up e4m3 [768, 6144], its swizzled UE8M0
        scales, down e4m3 [6144, 384], its swizzled UE8M0 scales)."""
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
            or (not hidden.is_contiguous())
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
        lgv, xq, xs = (self._dummy.view(torch.float32), self._dummy, self._dummy)
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
            key = (_ENGINE_PATH,) + tuple(sorted(self.config.items()))
            fn = _compiled.get(key)
            if fn is None:
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
    """One MoE layer on a:class:`M3MoeState`: ``layer(hidden)``. The expert buffers are read in
    place, never copied; the per-expert scalars are packed once (layer constants: rebuild the layer after reloading
    weights)."""

    shared_weights = None
    shared_mxfp8 = None

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
            if t.element_size() != 1 or tuple(t.shape) != shape or (not t.is_contiguous()):
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
            or (tuple(router_weight.shape) != (e, HIDDEN))
            or (not router_weight.is_contiguous())
            or (fc31_input_scale.dtype != torch.float32)
            or (fc31_input_scale.numel() != 1)
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
        """``hidden``: contiguous bf16 [M, 6144]. ``arp`` (int64 [8], see:func:`ar_push_desc`): also push the output
        into an all-reduce exchange."""
        return self.state.run(self, hidden, out, arp=arp)
