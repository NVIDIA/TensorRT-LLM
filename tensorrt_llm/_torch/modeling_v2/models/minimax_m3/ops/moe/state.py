# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common MoE layer assembly and launches with target-owned compile caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import torch

from .runtime import HIDDEN, M3MoeLayer, _view, shared_views


@dataclass(frozen=True)
class MoeStateBindings:
    """Geometry, packing and existing compile-cache policy selected by a target."""

    inter: int
    shared_inter: int
    engine_path: str
    compiled: dict[tuple, object]
    pretiled: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    cache_module_config: bool


class M3MoeStateBase:
    """The targets allocate their own state and retain their own shape/tactic choices."""

    BINDINGS: ClassVar[MoeStateBindings]

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
        ``shared_weights`` (a ``shared`` state): gate_up e4m3 [2 * shared_inter, 6144],
        its swizzled UE8M0 scales, down e4m3 [6144, shared_inter], and its swizzled scales.
        """
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
            inter=self.BINDINGS.inter,
        )
        if self.shared:
            layer.shared_weights = shared_views(
                shared_weights, "m3_moe", shared_inter=self.BINDINGS.shared_inter
            )
            layer.shared_mxfp8 = tuple(shared_weights)
            if self.mod.SH_PRETILE:
                # (gate_up, pre-tiled stages, down, dummy): the kernel streams the pre-tiled copy
                gu8, _, dn8, _ = layer.shared_weights
                layer.shared_weights = (
                    gu8,
                    self.BINDINGS.pretiled(*layer.shared_weights),
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
            # Preserve each target's cache-key policy: the module config for TP2/4, self.config for TP8.
            # Each binding retains its target-owned cache dictionary.
            key = (self.BINDINGS.engine_path,) + tuple(
                sorted(
                    (
                        self.mod.M3_CONFIG if self.BINDINGS.cache_module_config else self.config
                    ).items()
                )
            )
            fn = self.BINDINGS.compiled.get(key)
            if fn is None:
                fn = self.BINDINGS.compiled[key] = cute.compile(
                    self.mod.m3_moe,
                    *sig,
                    cute.runtime.make_fake_stream(),
                    options="--enable-tvm-ffi",
                )
            self._fn = fn
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        return out
