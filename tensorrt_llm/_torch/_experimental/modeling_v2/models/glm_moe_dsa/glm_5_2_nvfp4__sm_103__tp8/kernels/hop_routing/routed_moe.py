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
"""routed_moe fork of package hop_routing: the routed MoE consuming the hop's routing.

The expert-cluster kernel of routed_moe (``routed_moe/kernel_xc.py``) with its routing
phase replaced: instead of loading the router logits and selecting the top 8
(one warp per token), warp ``t`` reads token ``t``'s eight expert ids and
weights that the hop_routing hop wrote, fills the same shared-memory tables
(``s_topk_id``, ``s_topk_w`` and the expert bitmap) and continues exactly as
routed_moe. Given the same routing, every later step (slot tables, schedule, FC1,
SwiGLU, quantization, FC2, the outputs of each mode) is routed_moe's code, so the
numerics are routed_moe's.

The routing operand is the hop's int32 ``[2, R, 8]`` tensor (ids, then the
FP32 bits of the weights; see ``hop.split_routing``). It is passed in routed_moe's
router-logits address slot; the weights start ``R * 8`` words after the ids.
It is produced on the stream by the hop_routing hop and read only after
``griddepcontrol.wait`` (routed_moe's own memory-order contract, see
``routed_moe/routed_moe.py``).

Before the wait nothing else changes, except under package cross_layer_expert_prefetch's speculative
prefetch, which the fork compiles in from the runner key exactly as routed_moe's
runner does (``spec_k``, ``spec_ctas``, ``spec_self`` of ``key[6]``). Then the
TMA warp of each issuing CTA also reads the layer's row of predicted expert
IDs (cross_layer_expert_prefetch's int32 row of ``SPEC_KMAX`` entries) with relaxed loads and issues
L2 bulk prefetches of those experts' weights, which are model parameters
that nothing writes while serving. With cross_layer_expert_prefetch's predictor ``self``
(``spec_experts.SERVED_CONFIG``) the row has one writer: CTA 0's MMA warp of the
same layer's fork, which stores the call's distinct experts ranked by row count
once the routing tables are
built, i.e. after that call's own dependency wait. A call's pre-wait read
therefore sees the row its layer's previous call stored (that call completed
before this launch, in stream order behind the intervening dependency
waits), or, on a CTA that becomes resident after this call's CTA 0 has
stored, this call's own experts. Either value only selects which immutable
bytes are prefetched (each ID is masked into [0, 256)), never what the
kernel reads or computes.

Entry points (``None`` means "use the routed_moe path with the logits"; the
decision is host-only and precedes any device work):

* ``routed_output_into_hop(module, bias, x, logits, routing, hop)``: routed_moe_into_hop,
  every routed-input mode of the post_moe_hop;
* ``routed_output(module, bias, x, logits, routing, do_finalize)``.

Only the expert-cluster schedule is forked; a call that routed_moe would run with
its grid-wide schedule (``v6``) declines.
"""

import threading
from typing import Dict, List, Optional

import torch

from ..routed_moe import routed_moe as routed_moe_mod

FEATURE = "hop_routing"
TOP_K = routed_moe_mod.TOP_K
MAX_ROWS = routed_moe_mod.MAX_ROWS

_LOCK = threading.Lock()
_COMPILED: Dict[tuple, object] = {}
_KERNEL_CLASS = None


def _kernel_class():
    """``HopRoutedExpertClusterKernel``, defined on first use (imports the CuTe DSL)."""
    global _KERNEL_CLASS
    if _KERNEL_CLASS is not None:
        return _KERNEL_CLASS
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

    from ..routed_moe import ptx
    from ..routed_moe.kernel_xc import RoutedMoeExpertClusterKernel

    @dsl_user_op
    def ld_nc_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u32 $0, [$1];",
                "=r,l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    class HopRoutedExpertClusterKernel(RoutedMoeExpertClusterKernel):
        """routed_moe's expert-cluster kernel reading the routing instead of computing it."""

        @cute.jit
        def _route_tokens(
            self, w, lane, num_tokens, logits_addr, bias_r, s_topk_id, s_topk_w, storage
        ):
            """Warp ``w`` takes token ``w``'s ids and weights from the hop's
            routing (``logits_addr``: ids ``[R, 8]`` int32, then weights
            ``[R, 8]`` FP32) and marks the experts in the bitmap."""
            if w < num_tokens:
                if lane < TOP_K:
                    idx = w * TOP_K + lane
                    e = Int32(ld_nc_u32(logits_addr + idx * 4))
                    wt = ptx.ld_nc_f32(logits_addr + (num_tokens * TOP_K + idx) * 4)
                    s_topk_id[idx] = e
                    s_topk_w[idx] = wt
                    ptx.atom_or_shared_u32(
                        Int32(storage.bitmap.data_ptr().toint()) + (e // 32) * 4,
                        Uint32(1) << Uint32(e % 32),
                    )

    del cutlass
    _KERNEL_CLASS = HopRoutedExpertClusterKernel
    return HopRoutedExpertClusterKernel


class HopRoutedMoe(routed_moe_mod.RoutedMoe):
    """routed_moe runner whose expert-cluster kernel consumes the hop's routing."""

    def ready_for_capture(
        self, finalize: bool, debug: int = 0, rows: int = MAX_ROWS, red: bool = False
    ) -> bool:
        key = self._key(finalize, debug, rows, red)
        dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
        ws = routed_moe_mod._WORKSPACES.get(dev)
        return key in _COMPILED and ws is not None and ws.num_ctas >= self.num_ctas

    def _compiled(self, finalize: bool, debug: int, rows: int = MAX_ROWS, red: bool = False):
        """routed_moe's compiled launchers with ``HopRoutedExpertClusterKernel`` (expert clusters only)."""
        key = self._key(finalize, debug, rows, red)
        if key[3] != "xc":
            raise ValueError("the hop_routing fork runs the expert-cluster schedule only")
        with _LOCK:
            fns = _COMPILED.get(key)
            if fns is not None:
                return fns
            import cutlass
            import cutlass.cute as cute
            from cuda.bindings import driver as cuda
            from cutlass.cute.runtime import make_ptr

            from ..routed_moe.kernel_xc import DELAY_FINALIZE, RoutedMoeFinalizeKernel

            ptrs = self._pointers(make_ptr, cute, cutlass)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            inkernel = finalize and self.inkernel_finalize()
            # Package cross_layer_expert_prefetch's speculative expert prefetch
            # (``RoutedMoe._key``).
            spec = key[6] or (0, 0, False)
            kern = _kernel_class()(
                self._xc_ctas(rows, finalize, red),
                debug=debug,
                finalize=inkernel,
                red_out=red,
                **routed_moe_mod.xc_kernel_args(),
                spec_k=spec[0],
                spec_ctas=spec[1],
                spec_self=spec[2],
            )
            ints = [cutlass.Int64(0)] * routed_moe_mod._NUM_XC_ADDRS
            routed = cute.compile(kern.run, *ptrs, *ints, cutlass.Int32(1), stream)
            if spec[0]:
                from ... import dispatch

                dispatch.log_capability(
                    FEATURE,
                    f"expert_prefetch[k={spec[0]},ctas={spec[1]},self={int(spec[2])}]",
                    True,
                )
            fin = None
            if finalize and not inkernel and not red:
                delay, delay_ns = routed_moe_mod.DELAY_INJECTION
                fin = cute.compile(
                    RoutedMoeFinalizeKernel(delay_ns=delay_ns if delay & DELAY_FINALIZE else 0).run,
                    cutlass.Int64(0),
                    cutlass.Int64(0),
                    cutlass.Int64(0),
                    cutlass.Int32(1),
                    stream,
                )
            fns = (routed, fin)
            _COMPILED[key] = fns
            return fns


def routing_reason(routing: Optional[torch.Tensor], x: torch.Tensor) -> str:
    """Why ``routing`` cannot feed the fork for activation ``x`` ("" if it can)."""
    if routing is None:
        return "no routing from the hop"
    rows = x.shape[0]
    if (
        routing.dtype != torch.int32
        or tuple(routing.shape) != (2, rows, TOP_K)
        or not routing.is_contiguous()
        or routing.device != x.device
    ):
        return f"routing {tuple(routing.shape)} {routing.dtype}"
    return ""


def _engaged_runner(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    routing: Optional[torch.Tensor],
    do_finalize: bool,
    red: bool = False,
) -> Optional[HopRoutedMoe]:
    """The fork's runner for this call, or ``None`` (host-only decision)."""
    from ... import dispatch

    reason = routing_reason(routing, x)
    ok = not reason
    if ok:
        # routed_moe's own eligibility (shapes, dtypes, device, routing method, expert owner).
        ok, reason = routed_moe_mod.eligibility(module, x, router_logits)
    owner = routed_moe_mod.resolve_expert_owner(module)[0] if ok else None
    if ok:
        rm = getattr(owner, "routing_method", None)
        native_bias = getattr(rm, "e_score_correction_bias", None)
        if (
            routing_bias is None
            or routing_bias.dtype != torch.bfloat16
            or tuple(routing_bias.shape) != (routed_moe_mod.NUM_EXPERTS,)
            or not routing_bias.is_contiguous()
            or routing_bias.device != x.device
        ):
            ok, reason = False, "router bias is not a contiguous BF16 [256] on the device"
        elif native_bias is not None and native_bias.data_ptr() != routing_bias.data_ptr():
            ok, reason = False, "router bias is not the routing method's bias"
    runner = getattr(owner, "_hop_routing_runner", None) if ok else None
    if ok and runner is not None and runner.bias.data_ptr() != routing_bias.data_ptr():
        runner = None
    if ok and runner is None:
        runner = HopRoutedMoe(owner, routing_bias)
    rows = x.shape[0]
    if ok and runner.schedule(rows, do_finalize, red) != "xc":
        ok, reason = False, "routed_moe would run its grid-wide schedule"
    if ok and torch.cuda.is_current_stream_capturing():
        if getattr(owner, "_hop_routing_runner", None) is None or not runner.ready_for_capture(
            do_finalize, rows=rows, red=red
        ):
            ok, reason = False, "not warmed up before CUDA graph capture"
    dispatch.log_capability(
        FEATURE, f"routed_moe_fork rows={rows} {reason or 'engaged'}", ok, reason
    )
    if not ok:
        return None
    owner._hop_routing_runner = runner
    return runner


def routed_output(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    routing: Optional[torch.Tensor],
    do_finalize: bool,
) -> Optional[List[torch.Tensor]]:
    """routed_moe's ``routed_output`` with the hop's routing; ``None``: use routed_moe."""
    runner = _engaged_runner(module, routing_bias, x, router_logits, routing, do_finalize)
    if runner is None:
        return None
    return runner(x, routing, do_finalize=do_finalize)


def routed_output_into_hop(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    routing: Optional[torch.Tensor],
    hop,
) -> Optional[List[torch.Tensor]]:
    """routed_moe's ``routed_output_into_hop`` (routed_moe_into_hop) with the hop's routing; same
    modes and returns. ``None``: use routed_moe."""
    if hop is None or not getattr(hop, "_routed_enabled", False):
        return None
    rows = x.shape[0]
    if not hop.can_run(rows):
        return None
    mode = getattr(hop, "routed_mode", "row")
    if mode == "row":
        runner = _engaged_runner(module, routing_bias, x, router_logits, routing, True)
        if runner is None:
            return None
        target = hop.routed_input_buffer(rows)
        return [runner(x, routing, do_finalize=True, out=target)]
    red = mode == "fp32"
    runner = _engaged_runner(module, routing_bias, x, router_logits, routing, False, red)
    if runner is None:
        return None
    if red:
        return runner(x, routing, red_out=hop.routed_acc_buffer(rows))
    rows_buf, scales_buf = hop.routed_raw_buffers(rows)
    runner(x, routing, do_finalize=False, raw_out=(rows_buf, scales_buf))
    return [rows_buf]
