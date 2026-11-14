# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One TP4 layer's MoE-boundary block on the reference stages, built as the measured HYB arm built it (hyb_chain.py):
the reference variants router.lr4-prr4-q and moe.sht4-n48-q, their default Params (the m31q7 preset binds none), make()
on a views.Blk carrying this target's tensors, the engine's push re-pointed at this target's set A (seam X2), the o_proj
reading the LC's MXFP8 hand-off in place (seam X1).

What differs from the reference, and why (none changes a kernel, a config key or the launch order):
- weights: this target's, in the views.attn_weights / moe_weights form (``layer_views``). The reference post and router
  fold apply a Gemma norm as ``1 + w``; this target stores ``w + 1`` folded in BF16, so the view carries
  ``bf16(folded - 1)``, exact (checked): the post's fp32 ``1 + w`` is then this target's folded weight bit for bit.
- die shards: this target's routed weights cut into the two locality domains (LocalityMap, as the reference's
  domain_tensor), registered in expert_shards before make(); the full weights stay for the < 8-row engines.
- residual_mid / normed2 / residual_out / ffn_out: persistent buffers here, so the deferred front (QKV_FRONT) of this
  layer writes residual_mid in place and the next layer's front reads residual_out (no copies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from . import seams

ROWS = 8
H = 6144
KO = 2048  # o_proj K per TP4 rank: 16 q heads x 128 (weights.py q_width; the reference fused_o_proj_tp4_host.KO)
E = 128
# The engine module configuration the HYB arm runs (every rank).
HYB_ENGINE = {
    "AR_G3": 0,
    "AR_V3": 0,
    "AR_CL": 8,
    "AR_ND": 4,
    "AR_PRE": 1,
    "DL": 1,
    "SHARED": 1,
    "NSH": 24,
    "LAG": 6,
    "FC1D": 8,
    "XSTAGE": 0,
    "EXT_ROUTE": 1,
    "FUSED_ROUTER": 1,
}
# B2LAG off (seams.B2LAG_OFF, TRTLLM_M31_DISABLE_TP4WC_B2LAG=1): the reference shipped (no-B2LAG) engine
# module: every HYB_ENGINE field but
# LAG 6 -> 2 and FC1D 8 -> 4.
HYB_ENGINE_B2LAG_LAG = 6
HYB_ENGINE_B2LAG_OFF = {
    "AR_G3": 0,
    "AR_V3": 0,
    "AR_CL": 8,
    "AR_ND": 4,
    "AR_PRE": 1,
    "DL": 1,
    "SHARED": 1,
    "NSH": 24,
    "LAG": 2,
    "FC1D": 4,
    "XSTAGE": 0,
    "EXT_ROUTE": 1,
    "FUSED_ROUTER": 1,
}


def engine_matches(cfg: Dict[str, int]) -> bool:
    """Whether the built engine module is exactly the HYB engine under this process's B2LAG setting."""
    return cfg == (HYB_ENGINE_B2LAG_OFF if seams.B2LAG_OFF else HYB_ENGINE)


HYB_ENGINE_OVERLAY = {
    "ar_v3": 0,
    "ar_cl": 8,
}  # push into this target's set A with its cluster-8 call flags
_ENGINE_DEFAULTS = {
    "AR_G3": 0,
    "AR_V3": 0,
    "AR_CL": 8,
}  # m3_moe kernels without the attributes: no G3 / v3 push


class Geo:
    """views.Geo, verbatim semantics (hyb_chain / his_chain)."""

    def __init__(self, tp):
        self.tp = tp

    nh = property(lambda s: 4 // s.tp)
    hq = property(lambda s: 16 * s.nh)
    qkv_n = property(lambda s: (s.hq + 2 * s.nh + s.nh + 1) * 128)
    o_k = property(lambda s: s.hq * 128)
    inter = property(lambda s: 3072 // s.tp)


class Blk(SimpleNamespace):
    """views.Blk"""


def engine_module_config(mod) -> Dict[str, int]:
    """The engine module's push / schedule constants, the reference's engine_push_mode fields."""
    return {k: int(getattr(mod, k, _ENGINE_DEFAULTS.get(k, -1))) for k in HYB_ENGINE}


def unfold_gemma(folded: torch.Tensor) -> torch.Tensor:
    """bf16(folded - 1), and the proof that the reference fp32 ``1 + w`` gives ``folded`` back exactly."""
    raw = (folded.float() - 1.0).to(torch.bfloat16)
    if not torch.equal(raw.float() + 1.0, folded.float()):
        raise ValueError(
            "m31 tp4 hoist: the folded post norm is not 1 + an exact bf16 (cannot hand the reference post w)"
        )
    return raw.contiguous()


def domain_tensor(locmap, src: torch.Tensor, dom: int, keep: list) -> torch.Tensor:
    """A copy of src in locality domain dom (the family locality VMM mapping, granule-rounded), checked; src's copy
    in default memory when the device has no locality map."""
    if locmap is None:
        return src.clone()
    from ....kernels.m3ll import locality as L

    nbytes = src.numel() * src.element_size()
    g = locmap.granularity
    m = L._Mapping(locmap.device, [(-(-nbytes // g) * g, dom)])
    flat = torch.as_tensor(m, device=f"cuda:{locmap.device}")[:nbytes]
    keep.append(m)  # the mapping lives as long as the block
    for q in (flat.data_ptr(), flat.data_ptr() + nbytes - 1):
        if locmap.domain_of(q) != dom:
            raise RuntimeError(f"m31 tp4 hoist: shard byte {q:#x} is not in domain {dom}")
    t = flat.view(src.dtype).view(src.shape)
    t.copy_(src)
    return t


def die_shards(experts, inter: int, locmap, keep: list):
    """expert_shards._shards' cut (die p: FC1 rows [p*I, (p+1)*I), FC2 rows [p*H/2, (p+1)*H/2)), in die p's memory."""
    w13, w13s, w2, w2s = experts
    out = []
    for p in range(2):
        cut = lambda t, n: t[:, p * n : (p + 1) * n].contiguous()  # noqa: E731
        out.append(
            (
                domain_tensor(locmap, cut(w13, inter), p, keep),
                domain_tensor(locmap, cut(w13s, inter), p, keep),
                domain_tensor(locmap, cut(w2, H // 2), p, keep),
                domain_tensor(locmap, cut(w2s, H // 2), p, keep),
            )
        )
    return out


def layer_views(w: dict, moe, i: int, locmap, keep: list):
    """(aw, mw) of layer i: this target's tensors in the reference views form."""
    o_w, o_s = w[f"l{i}_o_w"].data, w[f"l{i}_o_s"].data
    # The reference o_proj host asserts (H, KO) (fused_o_proj_tp4_host.py:166) and unswizzles o_s as [H, KO / 32]: fail
    # here, with the reason, instead (a [6144, 6144] fixture reached that assert on GPU).
    if tuple(o_w.shape) != (H, KO) or not o_w.is_contiguous() or o_s.numel() != H * (KO // 32):
        raise ValueError(
            f"m31 tp4 hoist L{i}: o_proj weight {tuple(o_w.shape)} / scales {o_s.numel()} are not the "
            f"TP4 per-rank ({H}, {KO}) / {H * (KO // 32)} the reference o_proj is built for"
        )
    aw = SimpleNamespace(
        o=SimpleNamespace(w=o_w, s=o_s, n=o_w.shape[0], k=o_w.shape[1]),
        post_norm=unfold_gemma(w[f"l{i}_post_norm"].data),
    )
    experts = (
        w[f"l{i}_e_w31"].data.view(torch.uint8),
        w[f"l{i}_e_s31"].data.view(torch.float8_e4m3fn),
        w[f"l{i}_e_w2"].data.view(torch.uint8),
        w[f"l{i}_e_s2"].data.view(torch.float8_e4m3fn),
    )
    inter = experts[0].shape[1] // 2

    def nk(wt, s):
        return SimpleNamespace(w=wt.view(torch.float8_e4m3fn), s=s, n=wt.shape[0], k=wt.shape[1])

    mw = SimpleNamespace(
        experts=experts,
        full_weights_released=False,
        shards=die_shards(experts, inter, locmap, keep),
        g1_scale_c=moe.fc31_scale_c.float().contiguous(),
        g1_alphas=moe.fc31_alpha.float().contiguous(),
        g2_alphas=moe.fc2_alpha.float().contiguous(),
        gemm1_alpha=moe.act_alpha.float().contiguous(),
        gemm1_beta=moe.act_beta.float().contiguous(),
        gemm1_clamp_limit=moe.act_clamp.float().contiguous(),
        a1_gscale=moe.fc31_input_scale.float().reshape(-1)[:1].contiguous(),
        router=w[f"l{i}_router"].data.float().contiguous(),
        bias=w[f"l{i}_router_bias"].data.float().contiguous(),
        shared_gu=nk(w[f"l{i}_sh_gu_w"].data, w[f"l{i}_sh_gu_s"].data),
        shared_dn=nk(w[f"l{i}_sh_dn_w"].data, w[f"l{i}_sh_dn_s"].data),
        inter=inter,
    )
    return aw, mw


@dataclass
class HoistBlock:
    """Layer i's two reference stages and the buffers they read and write (fixed addresses: captured as is)."""

    layer: int
    blk: Blk
    router: object  # StageFn: o_proj (+ 16 router CTAs) then the prr4-q post
    moe: object  # StageFn: the ext-route engine, pushing into set A
    hold: dict  # the engine's push hold (router_hoist_host._hold): hold["arp"] is the set-A descriptor
    engine_config: Dict[str, int]
    keep: list
    placed: bool = (
        False  # die shards in their locality domains (DL = 1 as measured); False: default memory
    )
    armed: bool = False

    @property
    def residual_mid(self) -> torch.Tensor:
        return self.blk.residual_mid

    @property
    def residual_out(self) -> torch.Tensor:
        return self.blk.residual_out

    @property
    def ffn_out(self) -> torch.Tensor:
        return self.blk.ffn_out

    def arm(self, arp) -> None:
        """Point the engine's push at this target's set A (the MoE boundary's descriptor; set before capture)."""
        self.hold["arp"] = arp
        self.armed = True

    def __call__(self, oq: torch.Tensor, osf: torch.Tensor) -> None:
        """o_proj + post, then the engine. (oq, osf) must be the buffers the o_proj was built on (seam X1)."""
        if not self.armed:
            raise RuntimeError(f"m31 tp4 hoist L{self.layer}: the engine push was never armed")
        if (
            oq.data_ptr() != self.blk.h10_oq.data_ptr()
            or osf.data_ptr() != self.blk.h10_osf.data_ptr()
        ):
            raise RuntimeError(
                f"m31 tp4 hoist L{self.layer}: the LC's MXFP8 hand-off is not the o_proj's input"
            )
        self.router()
        self.moe()


def build(
    env, w: dict, moe, i: int, mx_buffers: Tuple[torch.Tensor, torch.Tensor], locmap=None
) -> HoistBlock:
    """Layer i's block (compiles on first use of each peer kernel config; MNNVL allocations are collective: call on
    every TP rank in the same layer order, outside graph capture). ``env`` is ``seams.install``'s state."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("m31 tp4 hoist: build during graph capture")
    from .tp4c1.families.moe import sht4_n48, sht4_n48_q
    from .tp4c1.families.router import lr4_prr4, lr4_prr4_q
    from .tp4c1.kernels.moe.engine import expert_shards
    from .tp4c1.kernels.moe.engine import router_hoist_host as hoist
    from .tp4c1.kernels.moe.router import fused_o_proj_tp4_host as ra4
    from .tp4c1.runtime.ctx import BuildContext

    if (ra4.H, ra4.KO) != (H, KO):
        raise RuntimeError(
            f"m31 tp4 hoist: the reference o_proj host is built for ({ra4.H}, {ra4.KO}), not ({H}, {KO})"
        )
    oq, osf = mx_buffers
    if oq.dim() != 2 or oq.shape[1] != KO or oq.shape[0] < ROWS or not oq.is_contiguous():
        raise ValueError(
            f"m31 tp4 hoist L{i}: the MXFP8 hand-off buffer {tuple(oq.shape)} is not [>= {ROWS}, {KO}]"
        )
    keep: list = []
    aw, mw = layer_views(w, moe, i, locmap, keep)
    dev = aw.o.w.device

    def e():
        return torch.zeros(ROWS, H, dtype=torch.bfloat16, device=dev)

    u = SimpleNamespace(
        ROT=64,
        PAGE=128,
        HQ=16,
        HKV=1,
        HI=1,
        D=128,
        LAYER=f"m31tp4hoist.L{i}.attn",
        INDEX_PREFIX=f"m31tp4hoist.L{i}.indexer",
    )
    blk = Blk(
        geo=Geo(4),
        t=ROWS,
        chain=SimpleNamespace(out=None),
        U=u,
        aw=aw,
        mw=mw,
        ar=None,
        aux=None,
        indexer=None,
        attn_layer=None,
        index_prefix=u.INDEX_PREFIX,
        layer_idx=i - 3,
        layer_id=i,
        ffn_partial=e(),
        residual_in=e(),
        normed=e(),
        residual_mid=e(),
        attn_partial=e(),
        normed2=e(),
        residual_out=e(),
        ffn_out=e(),
    )
    blk.h10_oq, blk.h10_osf = mx_buffers  # seam X1: the o_proj reads the LC's MXFP8 output in place
    expert_shards._st[id(mw)] = mw.shards
    stages = []
    with seams.no_jit(env.post, env.lamport):
        for variant, params in (
            (lr4_prr4_q.VARIANT, lr4_prr4.Lr4Prr4Params()),
            (sht4_n48_q.VARIANT, sht4_n48.Sht4N48Params()),
        ):
            blk.llk = BuildContext(
                env.rt, t=ROWS, layer=i, variant_id=variant.id, variant_debug=False
            )
            if variant.family == "moe":
                with env.knobs.scope({"M3_G1_CFG": json.dumps(HYB_ENGINE_OVERLAY)}):
                    st = variant.make(blk, params)
            else:
                st = variant.make(blk, params)
            if tuple(st.stages) != tuple(variant.stages):
                raise RuntimeError(
                    f"{variant.id} L{i}: stage covers {st.stages}, variant {variant.stages}"
                )
            stages.append(st)
    hold = hoist._hold(blk)
    if hold is None:
        raise RuntimeError(f"m31 tp4 hoist L{i}: no engine push hold registered for blk.ffn_out")
    cfg = engine_module_config(hoist._state_c1(blk).mod)
    if not engine_matches(cfg):
        raise RuntimeError(
            f"m31 tp4 hoist L{i}: engine {cfg} is not the measured HYB engine {HYB_ENGINE}"
            f" (B2LAG off: {seams.B2LAG_OFF})"
        )
    return HoistBlock(
        layer=i,
        blk=blk,
        router=stages[0],
        moe=stages[1],
        hold=hold,
        engine_config=cfg,
        keep=keep,
        placed=locmap is not None,
    )


_ENGAGED: Dict[Tuple[int, int], str] = {}


def log_engaged(rank: int, block: HoistBlock, hang_cap_ns: int) -> Optional[str]:
    """stderr, once per (rank, rows): the stable engagement line the tests grep."""
    import sys

    key = (rank, ROWS)
    if key in _ENGAGED:
        return None
    c = block.engine_config
    line = (
        f"[M31_TP4WC_HOIST] engaged: target sm_107 tp4 rank {rank} rows {ROWS} layer {block.layer}.. "
        f"o_proj lr4 ra 16 + post prr4-q + engine ext_route {c['EXT_ROUTE']} lag {c['LAG']} fc1d {c['FC1D']} "
        f"nsh {c['NSH']} dl {c['DL']} placed {int(block.placed)} | hang cap {hang_cap_ns} ns"
        f" | b2lag {'off' if seams.B2LAG_OFF else 'on'}"
    )
    _ENGAGED[key] = line
    print(line, file=sys.stderr, flush=True)
    return line
