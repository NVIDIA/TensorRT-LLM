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
"""M3.1 locality MoE engine variant (SM107 TP8, the 8-row decode call): ``m3_moe`` with die-local routed work.

A SM107 GPU has two locality domains (100 SMs + its half of HBM each, 12 remainder SMs). The stock engine (op.py,
the shared-expert unit loop SH20 at 8 tokens: 192 routed + 20 shared CTAs, one per SM) streams every expert's weights
from both domains on every CTA. This variant (kernel m31_local_moe_kernel.py, LOCAL = 1):

- weights: each layer's routed buffers are placed shard-major [2, E, rows / 2, K], half p in domain p's memory
  (locality VMM mappings; the FC2 scales padded to the 2 MiB granularity). Half p of an expert is the stock tensor's
  per-expert byte range [p n, (p + 1) n) (FC1 rows [p I, (p + 1) I), FC2 output rows [p H / 2, (p + 1) H / 2)), so the
  bytes are the stock ones;
- CTAs: a static %smid -> logical CTA table (LocalityMap.logical_map(96, 20), probed once per device) puts routed
  logical CTAs [96 p, 96 p + 96) on domain p's SMs and the shared CTAs on the rest;
- plan: local_plan.make_domain_phase_plan gives domain p's CTAs exactly half p's FC1 units and FC2 blocks; every
  unit's split stage (head + tail sums) is the stock plan's (local_plan.split_structure), so outputs are bit-identical.

Package: the TP8 target's ``local_moe`` (see ``aot_sources``).
Kill switch TRTLLM_M31_DISABLE_LOCAL_MOE (read at state build, i.e. load / capture time): :func:`make_state` builds
the stock :class:`M3MoeState` instead (its own CUDA graphs, stock weights only). Both states log an engagement line.
The variant compiles under its own AOT key ("m31_local_moe ...").
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.m3ll import (
    locality as _locality,
)

from ..kernels.m3_moe import op as _stock
from ..m31_switches import m31_disabled
from . import local_plan

# The stock engine's four 8-row tunings in the variant's kernel (k128, sfa_bulk, xdefer, b1split), all on. Kill switch
# TRTLLM_M31_DISABLE_TP8_MOETUNE=1 (none: the variant's original config and kernel). A tuning is set (as the kernel's
# local_* key) only where the stock engine has it on at these rows, so the stock kill switches
# TRTLLM_M31_DISABLE_MOE_{SFA_BULK,K128,XDEFER,B1SPLIT} still turn it off.
TUNES = ("k128", "sfa_bulk", "xdefer", "b1split")
TUNES_DEFAULT = TUNES


def tune_request() -> tuple:
    """The tunings asked for, in TUNES order."""
    if m31_disabled("TP8_MOETUNE"):
        return ()
    return TUNES_DEFAULT


def _apply_tunes(cfg: dict, stock_cfg: dict) -> dict:
    """Set the kernel's local_* key of each requested tuning the stock engine has on at these rows (stock_cfg = the
    inherited M3MoeState config); returns {tuning: "on" | "held (stock off)"}."""
    got = {}
    for t in tune_request():
        ok = bool(stock_cfg.get(t)) and (t != "sfa_bulk" or bool(stock_cfg.get("sfa_cp")))
        if ok:
            cfg["local_" + t] = 1
            if t == "b1split":
                cfg["local_b1s_tok"] = int(stock_cfg.get("b1s_tok", 4))
        got[t] = "on" if ok else "held (stock off)"
    return got


def _report_tunes(state, got: dict) -> None:
    """One line per state build: which tunings the kernel traced."""
    mod = state.mod
    flags = {"sfa_bulk": "SFA_BULK", "k128": "K128", "xdefer": "XDEFER", "b1split": "B1SPLIT"}
    kflags = ",".join(f"{v}:{int(bool(getattr(mod, v, False)))}" for v in flags.values())
    missing = [t for t, st in got.items() if st != "on" or not getattr(mod, flags[t], False)]
    _log(
        f"[M3 dispatch] LOCAL_MOE 8-row tunings device={state.device.index} tokens={state.m} "
        f"requested={','.join(got) or 'none'} kernel={kflags}"
        + (f" held={','.join(missing)}" if missing else "")
    )


def _fc1_image_half(h: torch.Tensor) -> torch.Tensor:
    """SFA_BULK: the bulk image of one domain half of a layer's FC1 scales (flat uint8): m31_moe_sfa_image's
    permutation [g, q, kb, lane, cm] -> [g, cm, kb, lane, q] of each 16-atom (8 KB) raw group, so a stage's 2 KB of
    atoms is one contiguous block. Built per half (a half is a whole number of groups); not cached by address
    (fc1_scale_image caches by data_ptr, which a freed temporary half could alias)."""
    if h.dtype != torch.uint8 or h.dim() != 1 or not h.is_contiguous() or h.numel() % 8192 != 0:
        raise ValueError(
            f"m31_local_moe SFA_BULK: an FC1 scale half must be contiguous uint8 of whole 8 KB groups, got "
            f"{h.dtype} {tuple(h.shape)}"
        )
    words = h.view(torch.int32).view(-1, 4, 4, 32, 4)  # [g, q, kb, lane, cm]
    return (
        words.permute(0, 4, 2, 3, 1).contiguous().view(-1).view(torch.uint8)
    )  # [g, cm, kb, lane, q]


LOCAL_PD = 96  # capacity: fixed: routed CTAs per domain (SH20: 192 routed = one FC2 block each)
LOCAL_TOKENS = frozenset(
    {8}
)  # capacity: fixed: the 8-row decode call (batch 1 at draft length 7) runs SH20
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_HERE, "m31_local_moe_kernel.py")
# The stock engines' AOT sources (cute_aot.py_files over the TP8 tree + the shared helpers) already cover this
# package; the variant's sources name its 3 files explicitly as well.
_OWN_SOURCES = tuple(
    os.path.join(_HERE, f)
    for f in ("local_plan.py", "m31_local_moe_kernel.py", "m31_local_moe_op.py")
)


def aot_sources():
    """The variant's AOT ``sources``: the stock m3_moe engine's + this package's 3 files."""
    return tuple(sorted(set(_stock._aot_sources()) | set(_OWN_SOURCES)))


_compiled: Dict[tuple, object] = {}
# per-device receipts (smid table, placement bytes) for the engagement line and the placement receipt
RECEIPTS: Dict[tuple, dict] = {}  # by (device index, placement)
# placed copies per source layer, keyed by the source buffers (an entry holds its sources: addresses not reused)
_PLACED: Dict[tuple, tuple] = {}
_LOGGED: set = set()


def local_moe_enabled() -> bool:
    return not m31_disabled("LOCAL_MOE")


def _log(msg: str):
    from tensorrt_llm.logger import logger

    logger.info(msg)


def make_state(device, num_tokens: int, shared: bool = True, **kw):
    """The m3_moe state for ``num_tokens`` rows: the locality variant at LOCAL_TOKENS on a two-domain device unless
    TRTLLM_M31_DISABLE_LOCAL_MOE, else the stock engine. Decided once here (load / capture time)."""
    why = None
    if not local_moe_enabled():
        why = "TRTLLM_M31_DISABLE_LOCAL_MOE"
    elif num_tokens not in LOCAL_TOKENS or not shared:
        why = f"tokens={num_tokens} shared={shared} (variant: {sorted(LOCAL_TOKENS)} rows, shared expert on)"
    elif not _stock.SH20:
        why = "SH20 off (the variant needs 192 routed + 20 shared CTAs)"
    if why is None:
        st = LocalMoeState(device, num_tokens, **kw)
        key = ("on", torch.device(device).index, num_tokens)
        if key not in _LOGGED:
            _LOGGED.add(key)
            r = st.receipt
            _log(
                f"[M3 dispatch] LOCAL_MOE taken (TP8 locality engine variant, tokens={num_tokens}, "
                f"{LOCAL_PD}+{LOCAL_PD} routed CTAs by domain + {st.mod.NSHC} shared, grid {r['grid']} = "
                f"{r['num_sms']} SMs, smap md5 {r['smap_md5']}, FC1_GEOMETRY={r['fc1_geometry']})"
            )
        return st
    key = ("off", torch.device(device).index, num_tokens)
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log(f"[M3 dispatch] LOCAL_MOE off ({why}): stock m3_moe engine, tokens={num_tokens}")
    return _stock.M3MoeState(device, num_tokens, shared=shared, **kw)


def _max_smem_per_sm(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    v = getattr(props, "shared_memory_per_multiprocessor", None)
    if v:
        return int(v)
    import cuda.bindings.driver as drv

    attr = drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    err, val = drv.cuDeviceGetAttribute(attr, device.index)
    if int(err) != 0:
        raise RuntimeError(f"m31_local_moe: cuDeviceGetAttribute failed ({err})")
    return int(val)


PLACEMENTS = (
    "domain",
    "swapped",
    "plain",
)  # the placement, then the negative control (halves swapped) and the allocation control (torch allocator)


def _place_halves(
    lm, halves: List[torch.Tensor], half_bytes: int, placement: str = "domain"
) -> torch.Tensor:
    """Halves (flat uint8, each <= half_bytes) at [k half_bytes, ..), zero padded: ``domain`` = half k in locality
    domain k's memory (asserted per byte range), ``swapped`` = in domain 1 - k, ``plain`` = the torch allocator."""
    if placement == "plain":
        out = torch.zeros(2, half_bytes, dtype=torch.uint8, device=f"cuda:{lm.device}")
        for k, h in enumerate(halves):
            out[k, : h.numel()].copy_(h)
        return out.view(-1)
    _Mapping = _locality._Mapping

    doms = (0, 1) if placement == "domain" else (1, 0)
    m = _Mapping(lm.device, [(half_bytes, doms[0]), (half_bytes, doms[1])])
    out = torch.as_tensor(m, device=f"cuda:{lm.device}").view(torch.uint8).view(2, half_bytes)
    out.zero_()
    for k, h in enumerate(halves):
        out[k, : h.numel()].copy_(h)
    out._locality_mapping = m
    for k in range(2):
        p = out[k].data_ptr()
        for q in (p, p + half_bytes - 1):
            if lm.domain_of(q) != doms[k]:
                raise RuntimeError(f"m31_local_moe: byte {q:#x} is not in domain {doms[k]}")
    return out.view(-1)


class LocalMoeState(_stock.M3MoeState):
    """:class:`op.M3MoeState` at 8 tokens with the LOCAL kernel, the domain plan and the static smid table."""

    def __init__(
        self,
        device: torch.device,
        num_tokens: int,
        num_ctas: Optional[int] = None,
        timing: bool = False,
        placement: str = "domain",
        **kw,
    ):
        """``timing`` (tests): the TIMING build, which records each CTA's %smid in tdbg[bx, 63]. ``placement`` (MB
        controls only; served = ``domain``): see :func:`_place_halves`."""
        if placement not in PLACEMENTS:
            raise ValueError(f"m31_local_moe: placement {placement!r} not in {PLACEMENTS}")
        self.placement = placement
        LocalityMap = _locality.LocalityMap

        device = torch.device(device)
        if num_tokens not in LOCAL_TOKENS:
            raise ValueError(
                f"m31_local_moe: built for {sorted(LOCAL_TOKENS)} tokens, not {num_tokens}"
            )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        sh = _stock.shared_ctas_for(num_tokens, True)
        if num_ctas not in (None, 2 * LOCAL_PD) or 2 * LOCAL_PD + sh != sms:
            raise ValueError(
                f"m31_local_moe: needs {2 * LOCAL_PD} routed + {sh} shared CTAs = the {sms} SMs (one CTA per SM)"
            )
        super().__init__(device, num_tokens, num_ctas=2 * LOCAL_PD, shared=True, **kw)
        lm = self.locality = LocalityMap.get(device)
        if lm.n_domains != 2 or lm.num_sms != sms:
            raise RuntimeError(
                f"m31_local_moe: {lm.n_domains} domains / {lm.num_sms} SMs (needs 2 / {sms})"
            )
        smap, ext = lm.logical_map(LOCAL_PD, sh)
        g = local_plan.geometry(_stock.INTER)
        cfg = dict(self.config)
        sf2_half = _stock.NUM_EXPERTS * _stock.HIDDEN * (_stock.INTER // 16) // 2
        sf2_half_pad = -(-sf2_half // lm.granularity) * lm.granularity
        cfg.update(
            local=1,
            local_pd=LOCAL_PD,
            sf2_half_atoms=sf2_half_pad // 512,
            fc1_geometry=0 if m31_disabled("MOE_FC1_GEOMETRY") else 1,
        )
        tunes = _apply_tunes(cfg, self.config)  # the 8-row tunings: the kernel's local_* keys
        if timing:
            cfg["timing"] = 1
        self.config = cfg
        self.sf2_half_bytes = sf2_half_pad
        stock_mod = self.mod
        self.mod = mod = _stock._kernel_module(cfg, _ENGINE_PATH)
        # the workspaces sized by the stock module are the variant's
        for name in (
            "G_CAP",
            "NG_CAP",
            "U1",
            "HB_REPL",
            "HB_STRIDE",
            "NSHC",
            "LG_REPL",
            "LG_STRIDE",
            "IMG_BYTES",
        ):
            if getattr(mod, name) != getattr(stock_mod, name):
                raise RuntimeError(f"m31_local_moe: {name} differs from the stock engine's")
        if mod.U1 != g["U1"] or mod.NCTA + mod.NSHC != sms:
            raise RuntimeError("m31_local_moe: geometry / grid mismatch")
        # one CTA per SM: the CTA's shared memory exceeds half an SM's, and the grid is the SM count (<= 212)
        smem_sm = _max_smem_per_sm(device)
        if not (2 * mod.BIG > smem_sm and mod.NCTA + mod.NSHC == sms <= 212):
            raise RuntimeError(
                f"m31_local_moe: 1 CTA/SM not guaranteed (BIG {mod.BIG} B vs {smem_sm} B/SM, grid "
                f"{mod.NCTA + mod.NSHC}, {sms} SMs)"
            )
        tw = 0.0 if num_tokens == 1 else 1.0
        plan, nb_cap = local_plan.make_domain_phase_plan(
            num_tokens, _stock.INTER, LOCAL_PD, _stock.M3_MOE_LAG, tw, True
        )
        errs = local_plan.validate_plan(plan, num_tokens, _stock.INTER, LOCAL_PD)
        stock_plan, _ = local_plan.stock_phase_plan(
            num_tokens, _stock.INTER, 2 * LOCAL_PD, _stock.M3_MOE_LAG, tw, True
        )
        a = local_plan.split_structure(plan, num_tokens, _stock.INTER, True)
        b = local_plan.split_structure(stock_plan, num_tokens, _stock.INTER, False)
        if errs or a.keys() != b.keys() or any(a[k][0] != b[k][0] for k in a):
            raise RuntimeError(
                f"m31_local_moe: domain plan invalid or split stages differ from stock: {errs[:3]}"
            )
        if nb_cap != cfg["nb2_cap"]:
            raise RuntimeError("m31_local_moe: FC2 block cap differs")
        self.plan = torch.tensor(plan, dtype=torch.int32).to(device)
        self.smap = smap.contiguous()
        lmap = [int(v) for v in self.smap.cpu()]
        self.receipt = RECEIPTS[(device.index, placement)] = {
            "device": device.index,
            "num_sms": sms,
            "grid": mod.NCTA + mod.NSHC,
            "per_domain": LOCAL_PD,
            "extra_domains": list(ext),
            "sm_domain": [int(v) for v in lm.sm_domain],
            "smap": lmap,
            "smap_md5": hashlib.md5(bytes(str(lmap), "ascii")).hexdigest(),
            "smem_per_cta": mod.BIG,
            "fc1_geometry": int(mod.FC1_GEOMETRY),
            "smem_per_sm": smem_sm,
            "sf2_half_bytes": sf2_half_pad,
            "placement": placement,
            "placed_bytes": 0,
        }
        self._layer_keys = set()
        _report_tunes(self, tunes)

    def layer(self, *a, shared_weights=None, **kw):
        base = super().layer(*a, shared_weights=shared_weights, **kw)
        key = (
            self.device,
            self.placement,
            base.w3_w1_weight.data_ptr(),
            base.w3_w1_weight_scale.data_ptr(),
            base.w2_weight.data_ptr(),
            base.w2_weight_scale.data_ptr(),
            bool(
                self.config.get("local_sfa_bulk")
            ),  # SFA_BULK: FC1 scale halves placed as the bulk image
        )
        hit = _PLACED.get(key)
        if hit is None:
            hit = _PLACED[key] = (
                (base.w3_w1_weight, base.w3_w1_weight_scale, base.w2_weight, base.w2_weight_scale),
                self._place(base),
            )
        if (
            key not in self._layer_keys
        ):  # this state's placed bytes (copies shared between states count in each)
            self._layer_keys.add(key)
            self.receipt["placed_bytes"] += sum(t.numel() for t in hit[1])
        base.local_weights = hit[1]
        return base

    def _place(self, base):
        """(w31, s31, w2, s2) shard-major [2, E, half]: half p of every expert (its byte range [p n, (p + 1) n) of the
        stock buffer: FC1 rows [p I, (p + 1) I), FC2 output rows [p H / 2, (p + 1) H / 2), their scales) placed per
        ``self.placement``; exact copies of the stock bytes. Only the FC2 scale halves need granularity padding."""
        lm = self.locality
        e = _stock.NUM_EXPERTS
        out = []
        for t, pad in (
            (base.w3_w1_weight, False),
            (base.w3_w1_weight_scale, False),
            (base.w2_weight, False),
            (base.w2_weight_scale, True),
        ):
            per = t.reshape(e, 2, -1)
            half = per[:, 0].numel()
            hb = -(-half // lm.granularity) * lm.granularity
            if hb != half and not pad:
                raise RuntimeError(
                    f"m31_local_moe: a {half} B half is not a {lm.granularity} B multiple"
                )
            if pad and hb != self.sf2_half_bytes:
                raise RuntimeError(
                    "m31_local_moe: FC2 scale half padding differs from the kernel's"
                )
            halves = [per[:, k].contiguous().view(-1) for k in range(2)]
            if t is base.w3_w1_weight_scale and self.config.get("local_sfa_bulk"):
                # SFA_BULK: half p of the FC1 scales placed as its bulk image (same bytes, permuted
                # per 8 KB group); the kernel reads it as both sfa1 (TMA descriptor, unused with SFA_CP) and s1w
                halves = [_fc1_image_half(h) for h in halves]
            out.append(_place_halves(lm, halves, hb, self.placement))
        torch.cuda.synchronize(self.device)
        return tuple(out)

    def run(self, layer, hidden, out=None, arp=None) -> torch.Tensor:
        m = self.m
        if arp is not None and self.config.get("out_skip"):
            _stock.OUT_SKIP_TAKEN[0] += 1
        if arp is not None and self.config.get("tail_trigger"):
            _stock.TAIL_TRIGGER_TAKEN[0] += 1
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != (m, _stock.HIDDEN)
            or not hidden.is_contiguous()
        ):
            raise ValueError(
                f"m31_local_moe: expects contiguous bf16 hidden states [{m}, {_stock.HIDDEN}]"
            )
        w31, s31, w2, s2 = layer.local_weights
        fz = (
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1),
            layer.fc31_input_scale.view(-1),
            self.lgbuf,
        )
        lgv, xq, xs = self._dummy.view(torch.float32), self._dummy, self._dummy
        if out is None:
            out = torch.empty(m, _stock.HIDDEN, dtype=torch.bfloat16, device=self.device)
        args = (
            w31,
            s31,
            w2,
            s2,
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
            *layer.shared_weights,
            self.hx,
            self.dnx,
            self._ar_off if arp is None else arp,
            s31.view(torch.int32),
            s2.view(torch.int32),
            self.img32,
            self.cnt,
            self.smap,
        )
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "m31_local_moe compiles on its first call: call it once before capture"
                )
            import cutlass.cute as cute

            aligns = [16] * len(args)
            aligns[12] = aligns[18] = 4  # epochs, fc31_input_scale (as the stock engine)
            sig = [_stock._view(t, a) for t, a in zip(args, aligns)]
            key = (_ENGINE_PATH,) + tuple(sorted(self.config.items()))
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute_aot.compile_or_load(
                    f"m31_local_moe {os.path.basename(key[0])} {key[1:]}",
                    lambda: cute.compile(
                        self.mod.m3_moe,
                        *sig,
                        cute.runtime.make_fake_stream(),
                        options="--enable-tvm-ffi",
                    ),
                    sources=aot_sources(),
                    tvm_ffi=True,
                )
            self._fn = fn
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        return out
