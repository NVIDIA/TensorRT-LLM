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
"""Host side of ``m3_moe16`` (m3_moe16_kernel.py): the MiniMax-M3 MoE layer (fused router + routed experts + shared
expert) for 9..16 decode tokens on one TP4 rank. The 1..8-token engine (op.M3MoeState) is untouched; select this
state only for num_tokens > 8.

Output: bf16 [M, 6144] = bf16(shared + routed), the routed part bit for bit TRTLLM-Gen's fused MoE on the fp32 router
GEMV logits (m3_router_gemv's math), the shared part as the in-model m3_mx_gemv shared expert.

    state = M3Moe16State(device, num_tokens=12)            # before CUDA-graph capture; one per token count
    layer = state.layer(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale, fc31_alpha, fc31_scale_c,
                        act_alpha, act_beta, act_clamp, fc2_alpha, e_score_correction_bias,
                        router_weight=router_weight, fc31_input_scale=fc31_input_scale,
                        shared_weights=(gate_up_w, gate_up_sf, down_w, down_sf))
    out = layer(hidden)                                       # first call compiles: call once before capture

Protocol state (per state object, shared by the layers that run on it in stream order): per-group FC2-B images and
unit counters, the B1 image + counter, per-CTA epochs. Every call adds exactly 192 to each group counter and NCTA to
the B1 counter, so nothing is reset between calls (graph replay safe).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .op import (
    HIDDEN,
    SHARED_CTAS,
    SHARED_INTER,
    U1,
    M3MoeLayer,
    _compiled,
    _kernel_module,
    _shared_views,
    _view,
    is_supported,
)


def _group_slots(g: int, p: int, perm: bool, min_ng: int = 5) -> int:
    """Slots of group p when the call has g distinct experts (the kernel's slot order)."""
    ng = (g + 3) // 4
    tail = g - 4 * (ng - 1)
    if p >= ng:
        return 0
    if perm and tail < 4 and ng >= max(3, min_ng):
        return tail if p == 1 else 4
    return min(4, g - 4 * p)


def make_phase_plan16(num_ctas: int, m_max: int, lag: int = 2, perm: bool = True, min_ng: int = 5):
    """op.make_phase_plan (early0, tail_w 1) with the kernel's group sizes (PERM: a partial last group is group 1)."""
    S1 = 6
    UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)
    ITEM = UNIT_BYTES // S1
    blk_bytes = 32 * (768 // 2 + 768 // 16)
    gcap = 4 * m_max
    nb2 = HIDDEN // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    for g in range(4, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = b0s[c] | (nbs[c] << 16)
        for p in range(ng):
            n_p = _group_slots(g, p, perm, min_ng)
            total = n_p * U1 * S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * _group_slots(g, p - lag, perm, min_ng) * blk_bytes
                if p == ng - 1 and not (p == 0 and m_max >= 2):
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += nbs[c] * _group_slots(g, q, perm, min_ng) * blk_bytes
            items = [0] * num_ctas
            if total >= S1 * num_ctas:

                def counts_for(limit):
                    return [max(S1, (limit - f) // ITEM) for f in fc2]

                lo, hi = 0, max(fc2) + total * ITEM
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
                        (c for c in range(num_ctas) if items[c] > S1),
                        key=lambda c: (-(fc2[c] + items[c] * ITEM), c),
                    )
                    items[c] -= 1
                    extra -= 1
            else:
                for c in sorted(sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // S1]):
                    items[c] = S1
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g, c, p] = i0 | ((i0 + items[c]) << 16)
                i0 += items[c]
            assert i0 == total
    return plan, max(nbs)


_ENGINE16_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3_moe16_kernel.py")
M16_MIN_TOKENS, M16_MAX_TOKENS = 9, 16
# Default routed CTAs (beside the SHARED_CTAS shared-expert CTAs).
M16_NUM_CTAS = 128
# Phase lag: phase p streams FC1 of group p and FC2 of group p - M16_LAG (the kernel's default lag).
M16_LAG = 3


class M3Moe16State:
    """Workspace + compiled kernel of ``m3_moe16`` for ``num_tokens`` (9..16) tokens per call."""

    fused_router = True
    shared = True

    def __init__(self, device: torch.device, num_tokens: int, num_ctas: Optional[int] = None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3Moe16State allocates and compiles: build it before CUDA-graph capture"
            )
        if not M16_MIN_TOKENS <= num_tokens <= M16_MAX_TOKENS:
            raise ValueError(
                f"m3_moe16 takes {M16_MIN_TOKENS}..{M16_MAX_TOKENS} tokens per call, not {num_tokens}"
            )
        self.device = torch.device(device)
        ok, why = is_supported(self.device)
        if not ok:
            raise RuntimeError(f"m3_moe16: {why}")
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.num_ctas = num_ctas or M16_NUM_CTAS
        if not 128 <= self.num_ctas <= sms - SHARED_CTAS:
            raise ValueError(
                f"m3_moe16 runs 128 .. {sms - SHARED_CTAS} routed CTAs + {SHARED_CTAS} shared (all resident)"
            )
        self.m = num_tokens
        plan, nb_cap = make_phase_plan16(self.num_ctas, num_tokens, M16_LAG)
        self.config = {
            "m_max": num_tokens,
            "num_ctas": self.num_ctas,
            "nb2_cap": nb_cap,
            "fused_router": 1,
            "timing": 0,
            "shared": 1,
        }
        try:
            self.mod = mod = _kernel_module(self.config, _ENGINE16_PATH)
        except AssertionError as err:
            raise ValueError(
                f"m3_moe16: num_tokens={num_tokens} does not fit the smem / TMEM budget: {err}"
            ) from err
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        # split units' head sums: every slot empty (0xFFFFFFFF) between calls
        self.partials = torch.full((mod.G_CAP * U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw)
        # FC2-B images per group + the B1 image; counters: group g at 8 g, B1 at 8 NG_CAP, CTA tickets at 8 NG_CAP + 8
        self.img = torch.zeros(mod.IMG_TOTAL // 4, dtype=torch.int32, **kw)
        self.cnt = torch.zeros(mod.NG_CAP * 8 + 16, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(self.num_ctas + mod.NSH, 64, dtype=torch.int64, **kw)
        self.lgbuf = torch.zeros(
            mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw
        )  # logits: tag << 32 | f32
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
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
    ) -> M3MoeLayer:
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
        layer.shared_weights = _shared_views(shared_weights, "m3_moe16")
        layer.shared_mxfp8 = tuple(shared_weights)
        return layer

    def run(self, layer, hidden, out=None, arp=None) -> torch.Tensor:
        m = self.m
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != (m, HIDDEN)
            or not hidden.is_contiguous()
        ):
            raise ValueError(f"m3_moe16: expects contiguous bf16 hidden states [{m}, {HIDDEN}]")
        if out is None:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        args = (
            layer.w3_w1_weight,
            layer.w3_w1_weight_scale.view(-1),
            layer.w2_weight,
            layer.w2_weight_scale.view(-1),
            layer.bias,
            layer.escal,
            self.plan.view(-1),
            self.partials,
            self.img,
            self.cnt,
            layer.w3_w1_weight_scale.reshape(-1).view(torch.int32),
            layer.w2_weight_scale.reshape(-1).view(torch.int32),
            layer.w3_w1_weight.reshape(-1).view(torch.int8),
            layer.w2_weight.reshape(-1).view(torch.int8),
            self.epochs,
            out.view(-1),
            self.route_dbg.view(-1),
            self.tdbg.view(-1),
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1),
            layer.fc31_input_scale.view(-1),
            self.lgbuf,
            *layer.shared_weights,
            self.hx,
            self.dnx,
            self._ar_off if arp is None else arp,
        )
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "m3_moe16 compiles on its first call: call it once before capture"
                )
            import cutlass.cute as cute

            sig = [_view(t, 16 if t.element_size() * t.numel() % 16 == 0 else 4) for t in args]
            key = (_ENGINE16_PATH,) + tuple(sorted(self.config.items()))
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
