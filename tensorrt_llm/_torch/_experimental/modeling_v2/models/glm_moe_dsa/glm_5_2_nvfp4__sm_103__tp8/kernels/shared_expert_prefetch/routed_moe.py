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
"""Routed side of package shared_expert_prefetch: hop_routing's routed_moe fork that stages the
shared expert in L2, on a chosen grid.

The MoE region ends with the later of two branches. On the
main stream, hop_routing's fork of routed_moe's expert-cluster kernel adds the routed experts
into routed_moe_into_hop's FP32 accumulator. On the auxiliary stream, the native BF16 shared
expert runs. Both stream their weights from DRAM at the same time, and the
routed grid is kept at 24 clusters so that the shared expert has SMs.

``SharedPrefetchRoutedMoe`` runs ``SharedPrefetchExpertClusterKernel`` (``kernel_xc.py``). That is
hop_routing's kernel plus an L2 prefetch of the layer's shared-expert weights, issued before its
dependency wait while the pre-MoE hop is still running. Its grid in the FP32
accumulator mode is ``min(co-resident clusters, DEFAULT_CLUSTERS)`` clusters, at
least ``ceil(R * 8 / 2)``. The schedule, its rules (routed_moe's
``xc_kernel_args``), the memory order after the wait and the arithmetic are those
of hop_routing's fork. The grid changes only the FP32 order in which top-k
contributions are added into the accumulator (routed_moe_into_hop's ``fp32``
class).

With package cross_layer_expert_prefetch the runner key carries cross_layer_expert_prefetch's
prefetch spec (``key[6]``, ``RoutedMoe._key``), which ``_compiled`` passes to
``SharedPrefetchExpertClusterKernel`` exactly as hop_routing's and routed_moe's runners do
(capability ``shared_expert_prefetch/expert_prefetch[k=..,ctas=.., self=..]``), and ``__call__``
passes cross_layer_expert_prefetch's row of predicted experts as the 18th routed_moe address, before
the two shared-expert addresses. The kernel then issues the shared-expert chunks and
cross_layer_expert_prefetch's chunks from one pre-wait prologue (``kernel_xc.py``).

``routed_output_into_hop(module, bias, x, logits, routing, hop, shared)``
returns ``None`` before any device work unless the hop takes the FP32
accumulator, hop_routing's routing is present, and ``shared`` is a BF16 shared expert
whose weights the kernel can prefetch. The caller then runs hop_routing's fork.
"""

import threading
from typing import Dict, List, Optional, Tuple

import torch

from ..hop_routing import routed_moe as hop_routed_mod
from ..routed_moe import routed_moe as routed_moe_mod

FEATURE = "shared_expert_prefetch"
DEFAULT_CLUSTERS = 24
SE_HIDDEN = 6144
SE_INTER = 256

_LOCK = threading.Lock()
_COMPILED: Dict[tuple, object] = {}


def shared_weights(shared) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], str]:
    """``(gate_up, down)`` weights of a served BF16 shared expert, or ``(None, reason)`` (host-only)."""
    if shared is None or type(shared).__name__ != "GatedMLP":
        return None, "no BF16 GatedMLP shared expert"
    w1 = getattr(getattr(shared, "gate_up_proj", None), "weight", None)
    w2 = getattr(getattr(shared, "down_proj", None), "weight", None)
    for name, w, shape in (
        ("gate_up", w1, (2 * SE_INTER, SE_HIDDEN)),
        ("down", w2, (SE_HIDDEN, SE_INTER)),
    ):
        if (
            not isinstance(w, torch.Tensor)
            or w.dtype != torch.bfloat16
            or tuple(w.shape) != shape
            or not w.is_contiguous()
            or not w.is_cuda
            or w.data_ptr() % 16
        ):
            return (
                None,
                f"shared {name} weight is not a contiguous 16-byte aligned BF16 {shape} device tensor",
            )
    return (w1, w2), ""


class SharedPrefetchRoutedMoe(hop_routed_mod.HopRoutedMoe):
    """hop_routing's fork running ``SharedPrefetchExpertClusterKernel`` on the
    shared_expert_prefetch grid (FP32 accumulator output only)."""

    def _xc_ctas(
        self, rows: int = routed_moe_mod.MAX_ROWS, finalize: bool = True, red: bool = False
    ) -> int:
        from ..routed_moe.kernel_xc import CLUSTER, MAX_ROUNDS

        if not red or finalize:
            return super()._xc_ctas(rows, finalize, red)
        co_resident = routed_moe_mod._max_active_clusters(self.device, CLUSTER)
        want = DEFAULT_CLUSTERS
        need = -(-rows * routed_moe_mod.TOP_K // MAX_ROUNDS)
        n = CLUSTER * min(co_resident, max(want, need))
        n = min(n, self.num_ctas)
        return n - n % CLUSTER

    def _key(
        self, finalize: bool, debug: int, rows: int = routed_moe_mod.MAX_ROWS, red: bool = False
    ):
        return super()._key(finalize, debug, rows, red) + (("shared_expert_prefetch", True),)

    def ready_for_capture(
        self,
        finalize: bool = False,
        debug: int = 0,
        rows: int = routed_moe_mod.MAX_ROWS,
        red: bool = True,
    ) -> bool:
        key = self._key(finalize, debug, rows, red)
        dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
        ws = routed_moe_mod._WORKSPACES.get(dev)
        return key in _COMPILED and ws is not None and ws.num_ctas >= self.num_ctas

    def _compiled(
        self, finalize: bool, debug: int, rows: int = routed_moe_mod.MAX_ROWS, red: bool = False
    ):
        """The compiled ``SharedPrefetchExpertClusterKernel`` launcher (expert clusters, FP32 accumulator only)."""
        key = self._key(finalize, debug, rows, red)
        if key[3] != "xc" or finalize or not red:
            raise ValueError(
                "the shared_expert_prefetch fork runs the expert-cluster schedule into the FP32 accumulator only"
            )
        with _LOCK:
            fn = _COMPILED.get(key)
            if fn is not None:
                return fn
            import cutlass
            import cutlass.cute as cute
            from cuda.bindings import driver as cuda
            from cutlass.cute.runtime import make_ptr

            from .kernel_xc import SharedPrefetchExpertClusterKernel

            ptrs = self._pointers(make_ptr, cute, cutlass)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            # Package cross_layer_expert_prefetch's speculative expert prefetch
            # (``RoutedMoe._key``).
            spec = key[6] or (0, 0, False)
            kern = SharedPrefetchExpertClusterKernel(
                self._xc_ctas(rows, finalize, red),
                debug=debug,
                finalize=False,
                red_out=True,
                **routed_moe_mod.xc_kernel_args(),
                spec_k=spec[0],
                spec_ctas=spec[1],
                spec_self=spec[2],
                se_pf=True,
            )
            ints = [cutlass.Int64(0)] * (routed_moe_mod._NUM_XC_ADDRS + 2)
            fn = cute.compile(kern.run, *ptrs, *ints, cutlass.Int32(1), stream)
            if spec[0]:
                from ... import dispatch

                dispatch.log_capability(
                    FEATURE,
                    f"expert_prefetch[k={spec[0]},ctas={spec[1]},self={int(spec[2])}]",
                    True,
                )
            _COMPILED[key] = fn
            return fn

    def __call__(
        self,
        x: torch.Tensor,
        routing: torch.Tensor,
        red_out: torch.Tensor,
        se_weights: Tuple[torch.Tensor, torch.Tensor],
        debug: int = 0,
        debug_buffer: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Add the routed experts of ``x`` (routing from the hop) into ``red_out`` (FP32 ``[R, H]``, zero on
        entry); ``se_weights`` = the shared expert's ``(gate_up, down)`` weights to prefetch."""
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda
        from cutlass.cute.runtime import make_ptr

        m = self.module
        rows = x.shape[0]
        x = routed_moe_mod._bulk_copy_ready(x)
        if self.schedule(rows, False, True) != "xc":
            raise ValueError("FP32 accumulator output needs the expert-cluster schedule")
        routed_fn = self._compiled(False, debug, rows, True)
        ws = routed_moe_mod._workspace(self.device, self.num_ctas)
        dbg = debug_buffer if debug_buffer is not None else ws.dummy
        current = torch.cuda.current_stream()
        ws.order_on(current)
        stream = cuda.CUstream(current.cuda_stream)
        ptrs = self._pointers(make_ptr, cute, cutlass)
        se_w1, se_w2 = se_weights
        addrs = [
            x.data_ptr(),
            routing.data_ptr(),
            self.bias.data_ptr(),
            m.fc31_alpha.data_ptr(),
            m.fc31_scale_c.data_ptr(),
            m.fc2_alpha.data_ptr(),
            m.fc31_input_scale.data_ptr(),
            ws.rows.data_ptr(),
            ws.scales.data_ptr(),
            ws.dummy.data_ptr(),
            red_out.data_ptr(),
            ws.tile_count.data_ptr(),
            dbg.data_ptr(),
            m.w3_w1_weight.data_ptr(),
            m.w3_w1_weight_scale.data_ptr(),
            m.w2_weight.data_ptr(),
            m.w2_weight_scale.data_ptr(),
        ]
        # cross_layer_expert_prefetch's row of predicted experts (the kernel ignores it unless
        # cross_layer_expert_prefetch is compiled in).
        spec = self._spec()
        addrs.append(spec[0].data_ptr() if spec is not None else ws.dummy.data_ptr())
        addrs += [se_w1.data_ptr(), se_w2.data_ptr()]
        assert len(addrs) == routed_moe_mod._NUM_XC_ADDRS + 2
        routed_fn(*ptrs, *[cutlass.Int64(a) for a in addrs], cutlass.Int32(rows), stream)
        return [red_out]


def _engaged_runner(module, routing_bias, x, router_logits, routing, shared):
    """``(runner, se_weights)`` for an FP32-accumulator call, or ``(None, None)`` (host-only decision)."""
    from ... import dispatch

    reason = hop_routed_mod.routing_reason(routing, x)
    ok = not reason
    se = None
    if ok:
        se, reason = shared_weights(shared)
        ok = se is not None
    if ok:
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
    runner = getattr(owner, "_shared_prefetch_runner", None) if ok else None
    if ok and runner is not None and runner.bias.data_ptr() != routing_bias.data_ptr():
        runner = None
    if ok and runner is None:
        runner = SharedPrefetchRoutedMoe(owner, routing_bias)
    rows = x.shape[0]
    if ok and runner.schedule(rows, False, True) != "xc":
        ok, reason = False, "routed_moe would run its grid-wide schedule"
    if ok and torch.cuda.is_current_stream_capturing():
        if getattr(owner, "_shared_prefetch_runner", None) is None or not runner.ready_for_capture(
            False, rows=rows, red=True
        ):
            ok, reason = False, "not warmed up before CUDA graph capture"
    grid = runner._xc_ctas(rows, False, True) if ok else 0
    dispatch.log_capability(
        FEATURE,
        f"routed rows={rows} grid={grid} sepf=1 {reason or 'engaged'}",
        ok,
        reason,
    )
    if not ok:
        return None, None
    owner._shared_prefetch_runner = runner
    return runner, se


def routed_output_into_hop(
    module,
    routing_bias: torch.Tensor,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    routing: Optional[torch.Tensor],
    hop,
    shared=None,
) -> Optional[List[torch.Tensor]]:
    """hop_routing's ``routed_output_into_hop`` (FP32 accumulator only) through
    ``SharedPrefetchExpertClusterKernel``, which prefetches ``shared``'s weights into L2; ``None``:
    use hop_routing's fork."""
    if (
        hop is None
        or not getattr(hop, "_routed_enabled", False)
        or getattr(hop, "routed_mode", "") != "fp32"
    ):
        return None
    rows = x.shape[0]
    if not hop.can_run(rows):
        return None
    runner, se = _engaged_runner(module, routing_bias, x, router_logits, routing, shared)
    if runner is None:
        return None
    return runner(x, routing, hop.routed_acc_buffer(rows), se)
