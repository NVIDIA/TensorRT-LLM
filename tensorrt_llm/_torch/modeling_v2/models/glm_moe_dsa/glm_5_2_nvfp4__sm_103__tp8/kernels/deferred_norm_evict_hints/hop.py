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
"""deferred_norm_evict_hints deferred-norm post-MoE hop: deferred_norm's hop with
deferred_norm_evict_hints's chain schedule.

``EvictHintsDeferredHop`` is :class:`DeferredNormHop` (same eligibility, scope,
contract checks, collective decisions, post_moe_hop workspace protocol, packet type and
materialization) launching ``EvictHintsHopKernel`` with ``EvictHintsHopConfig``'s L2 hints
and carveout. Its precompile covers the deferred_norm_evict_hints hop variants, deferred_norm's
materialization and the served ``EvictHintsProjection`` kernels, so that a packet
exists only where every rank can project it with deferred_norm_evict_hints's projection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ... import dispatch
from ..deferred_norm.hop import _ROUTED_INPUT_MODES, DeferredNormHop
from ..deferred_norm.packet import DeferredNormHidden, frag_elems
from ..post_moe_hop.kernels import STATE_WORDS
from ..post_moe_hop.workspace import PostMoeHopWorkspace, _all_ranks_agree
from .projection import EvictHintsProjConfig

FEATURE_ID = "deferred_norm_evict_hints"
# Poll-slice prefetch modes issued before the hop's dependency wait: they touch producer-owned
# Lamport addresses before the wait, so only tests build them (``projection.parse_cfg``).
PRE_WAIT_PREFETCH = ("pre3", "tpre3")


@dataclass(frozen=True)
class EvictHintsHopConfig:
    """Launch configuration of the deferred_norm_evict_hints hop (``hop_kernel.py``).

    ``ctas`` is the number of CTAs per token row (deferred_norm: 8); fewer CTAs leave
    more SMs free for the projection that the hop releases early. The
    square-sum partial order does not depend on it (``CTAS * WARPS = 24``
    partials of 256 columns each, in column order).

    The defaults launch deferred_norm's hop schedule unchanged. ``carveout=100`` with
    ``ffi=False`` (the only launcher whose kernels the carveout reaches) lets
    every projection CTA start next to a hop CTA, and its weight stream then
    delays the exchange. The other values are for experiments only.
    """

    carveout: int = -1
    poll_prefetch: str = "none"
    clear_hint: str = "none"
    ctas: int = 8
    # TVM-FFI launcher (True) or the DSL executor (False).
    ffi: bool = True
    # Unused dynamic shared memory of the hop launch (``hop_kernel.MAX_DYN_SMEM`` at most).
    dyn_smem: int = 0


class EvictHintsDeferredHop(DeferredNormHop):
    """deferred_norm's deferred hop launching the deferred_norm_evict_hints kernel (module docstring).

    Args:
        post_moe_hop, trigger_early, ctas, next_layer_full_indexer: as
            :class:`DeferredNormHop`.
        config: hop launch configuration (default: ``EvictHintsHopConfig()``).
        proj_config: the deferred_norm_evict_hints projection configuration the consumer will
            launch (default: ``EvictHintsProjConfig()``); compiled before
            the cross-rank agreement.
    """

    def __init__(
        self,
        post_moe_hop,
        trigger_early: bool = True,
        ctas: Optional[int] = None,
        next_layer_full_indexer: Optional[bool] = None,
        config: Optional[EvictHintsHopConfig] = None,
        proj_config: Optional[EvictHintsProjConfig] = None,
    ):
        self.hop_config = config or EvictHintsHopConfig()
        self.evict_hints_proj = proj_config or EvictHintsProjConfig()
        ctas = self.hop_config.ctas if ctas is None else ctas
        super().__init__(
            post_moe_hop,
            trigger_early=trigger_early,
            ctas=ctas,
            carveout=self.hop_config.carveout,
            next_layer_full_indexer=next_layer_full_indexer,
        )
        dispatch.log_capability(FEATURE_ID, "deferred_hop", self.engaged, self.reason)

    def _compiled(
        self,
        has_shared: bool,
        scale_fp32: bool,
        routed_input: Optional[str],
        debug_ts: bool = False,
    ):
        from .hop_kernel import compiled_evict_hints_hop

        return compiled_evict_hints_hop(
            self.hidden_size,
            self.top_k,
            self.mapping.tp_size,
            self.post_moe_hop.max_rows,
            self.ctas,
            has_shared,
            scale_fp32,
            self.use_pdl,
            self.trigger_early,
            STATE_WORDS,
            self.carveout,
            debug_ts,
            self.fault_delay_ns,
            routed_input,
            self.hop_config.poll_prefetch,
            self.hop_config.clear_hint,
            self.hop_config.ffi,
            self.hop_config.dyn_smem,
        )

    def _precompile(self) -> Optional[str]:
        from ..deferred_norm.hop_kernel import compiled_materialize
        from .projection import precompile_served

        for has_shared in (True, False):
            for scale_fp32 in (False, True):
                self._compiled(has_shared, scale_fp32, None)
        compiled_materialize(self.hidden_size, self.ctas, self.partials, self.use_pdl)
        return precompile_served(torch.cuda.current_device(), self.evict_hints_proj)

    def enable_routed_input(self) -> bool:
        """Collective; as :meth:`DeferredNormHop.enable_routed_input` with the deferred_norm_evict_hints kernels."""
        mode = self._post_moe_routed_mode()
        local_ok, reason = True, ""
        if not self.engaged:
            local_ok, reason = False, "deferred_norm_evict_hints hop not engaged"
        elif mode is None:
            local_ok, reason = False, "post_moe_hop hop's routed input is not enabled"
        else:
            try:
                for has_shared in (True, False):
                    self._compiled(has_shared, False, mode)
            except Exception as exc:  # noqa: BLE001 - converged across ranks below
                local_ok, reason = False, f"routed kernel compile failed: {exc!r}"
        agreed = False
        if self.engaged and mode is not None:
            comm = PostMoeHopWorkspace.comm_for(self.mapping)
            same_mode = [m for m in _ROUTED_INPUT_MODES if _all_ranks_agree(comm, mode == m)]
            if not same_mode:
                local_ok, reason = (
                    False,
                    "the post_moe_hop hops' routed-input modes differ across ranks",
                )
            agreed = _all_ranks_agree(comm, local_ok)
            if not agreed:
                reason = reason or "another rank is not capable"
        self.routed_engaged = agreed
        self.routed_mode = mode if agreed else None
        self.routed_reason = reason
        dispatch.log_capability(FEATURE_ID, f"deferred_hop_routed[{mode}]", agreed, reason)
        return agreed

    def _launch(
        self,
        fc2_ptr: int,
        fc2_row_bytes: int,
        fc2_rows: int,
        scales_ptr: int,
        perm_ptr: int,
        scale_fp32: bool,
        routed_input: Optional[str],
        shared_output,
        residual,
        norm_weight,
        eps,
        ts,
    ) -> Tuple[DeferredNormHidden, torch.Tensor]:
        num_rows = residual.shape[0]
        ws = self.post_moe_hop.workspace
        a_frag = torch.empty(
            (frag_elems(self.hidden_size),), dtype=residual.dtype, device=residual.device
        )
        residual_out = torch.empty_like(residual)
        sumsq = torch.empty((num_rows, self.partials), dtype=torch.float32, device=residual.device)
        has_shared = shared_output is not None
        assert ws.state.numel() == STATE_WORDS
        compiled = self._compiled(has_shared, scale_fp32, routed_input, ts is not None)
        args = (
            fc2_ptr,
            fc2_row_bytes,
            fc2_rows,
            scales_ptr,
            perm_ptr,
            shared_output.data_ptr() if has_shared else 0,
            residual.data_ptr(),
            norm_weight.data_ptr(),
            a_frag.data_ptr(),
            residual_out.data_ptr(),
            sumsq.data_ptr(),
            ws.mc.data_ptr(),
            ws.uc.data_ptr(),
            ws.partials.data_ptr(),
            0 if ts is None else ts.data_ptr(),
        )
        if self.hop_config.ffi:
            compiled(*args, ws.state, self.mapping.tp_rank, self.timeout_ns, num_rows)
        else:
            from cuda.bindings.driver import CUstream
            from cutlass import Int32, Int64

            ints = (Int64, Int64, Int32) + (Int64,) * 12
            compiled(
                *[t(v) for t, v in zip(ints, args)],
                Int64(ws.state.data_ptr()),
                Int32(self.mapping.tp_rank),
                Int64(self.timeout_ns),
                Int32(num_rows),
                CUstream(torch.cuda.current_stream().cuda_stream),
            )
        packet = DeferredNormHidden(
            a_frag=a_frag,
            sumsq=sumsq,
            residual=residual_out,
            gamma=norm_weight,
            eps=float(eps),
            materializer=self.materialize,
        )
        return packet, residual_out
