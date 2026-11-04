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
"""deferred-norm post-MoE hop (deferred-norm packet, numerical variant).

``DeferredNormHop`` wraps an engaged :class:`PostMoeHop` of one target MoE
layer whose next layer is a target attention layer. It runs the same
finalize / multicast all-reduce / residual protocol on the post_moe_hop workspace, but
returns a :class:`DeferredNormHidden` packet instead of the normalized hidden
state; the next layer's projection (``DeferredNormProjection``) applies the RMSNorm
scale after its GEMM. The residual output is bit-identical to post_moe_hop's.

The hop shares the post_moe_hop workspace protocol, including post_moe_hop's error handling:
the post_moe_hop's poll deadline (``timeout_ns``) bounds every deferred_norm poll; device
errors (a peer that never publishes, a permutation entry beyond the FC2
rows) set the sticky per-buffer error words that post_moe_hop's ``check_health``
reads, and every later post_moe_hop or deferred_norm call of the group publishes NaN until a
collective workspace reset. After the MoE runner the hop applies post_moe_hop's
raw-tuple contract: a dtype or shape that differs from the declared contract
raises ``PostMoeHopContractError`` (the runner already skipped its finalize, so no
native path is left), and a layout the kernel cannot read with 16-byte
vectors is copied into an aligned contiguous tensor on this rank only
(counted in the workspace's ``layout_repairs``).

With routed_moe_into_hop selected, routed_moe writes its routed result into the post_moe_hop's
routed-input buffer and the layer calls ``PostMoeHop.hop_routed``. After
:meth:`DeferredNormHop.enable_routed_input` (collective; only where this hop
is engaged and the post_moe_hop's routed input is enabled on every rank),
:meth:`DeferredNormHop.hop_routed` takes the same inputs and runs the same
protocol on the same routed buffer, and returns the deferred packet instead
(its materialization equals ``hop_routed``'s output bit for bit).

Eligibility is decided collectively once and per call from rank-identical
host state only, before the MoE runner runs, so every rank takes the same hop
kernel. The collective decision covers every kernel a packet can reach: the
hop, the materialization and both served projection kernels of the next
layer (shared- and full-indexer layers). A rank that cannot compile one of
them disengages the hop on every rank, so no rank produces a packet that
another rank's consumer could not project; the consumer's own compile is
then a cache hit and its remaining declines are rank-identical static checks.

The scope (``scope.py``; the served scope is ``shared``) is part of the same local
capability: with ``shared``, a hop whose next layer is a full-indexer layer (or
of unknown type) is incapable on this rank before any compile, and the single
agreement disengages it on every rank, so that layer keeps the post-MoE hop. The
number of collectives does not depend on the scope.
"""

from typing import List, Optional, Tuple

import torch

from ... import dispatch
from ..post_moe_hop import PostMoeHopContractError
from ..post_moe_hop.kernels import STATE_WORDS
from ..post_moe_hop.workspace import PostMoeHopWorkspace, _all_ranks_agree
from .packet import DeferredNormHidden, frag_elems
from .scope import UNRESTRICTED_SCOPE, decline_reason, next_layer_kind, selected_scope

FEATURE_ID = "deferred_norm"
MAX_ROWS = 8
DEFAULT_CTAS = 8
_SCALE_DTYPES = (torch.bfloat16, torch.float32)
# routed_moe_into_hop's routed-input modes (PostMoeHop.ROUTED_MODES), in agreement order.
_ROUTED_INPUT_MODES = ("row", "identity", "fp32")
_VECTOR_BYTES = 16


def _vector_ready(t: torch.Tensor, row_major_2d: bool) -> bool:
    """Whether the hop kernel may read ``t`` with 16-byte vectors at its strides."""
    if t.data_ptr() % _VECTOR_BYTES or t.stride(-1) != 1:
        return False
    if row_major_2d:
        return (t.stride(0) * t.element_size()) % _VECTOR_BYTES == 0
    return t.is_contiguous()


def _graph_mode_ok() -> Tuple[bool, str]:
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
    from tensorrt_llm._torch.utils import is_torch_compiling

    if is_torch_compiling():
        return False, "torch.compile graph keeps the native custom-op split"
    if is_in_breakable_cuda_graph():
        return False, "breakable CUDA graph keeps the native custom-op split"
    return True, ""


class DeferredNormHop:
    """Deferred-norm post-MoE hop on the post_moe_hop workspace.

    Args:
        post_moe_hop: the layer's engaged ``PostMoeHop`` (its workspace, mapping,
            dtype, PDL policy, poll deadline and test-only fault injection
            are reused).
        trigger_early: release the PDL dependents right after the hop's own
            dependency wait (so the next projection stages its weights while
            the hop communicates) instead of after the reduction.
        ctas: CTAs per token row (8 columns per thread).
        carveout: preferred shared-memory carveout of the hop kernel in percent
            (-1: driver default).
        next_layer_full_indexer: whether the layer after this hop's MoE layer
            owns an indexer; read only when
            the scope is not ``all``.
            ``None``: resolved from the constructing decoder layer
            (``scope.next_layer_kind``).
    """

    def __init__(
        self,
        post_moe_hop,
        trigger_early: bool = True,
        ctas: int = DEFAULT_CTAS,
        carveout: int = -1,
        next_layer_full_indexer: Optional[bool] = None,
    ):
        from .hop_kernel import hop_kernel_supported

        self.post_moe_hop = post_moe_hop
        self.mapping = post_moe_hop.mapping
        self.hidden_size = post_moe_hop.hidden_size
        self.top_k = post_moe_hop.top_k
        self.dtype = post_moe_hop.dtype
        self.use_pdl = post_moe_hop.use_pdl
        self.timeout_ns = post_moe_hop.timeout_ns
        self.fault_delay_ns = post_moe_hop.fault_delay_ns
        self.trigger_early = trigger_early
        self._repairs_logged = set()
        # routed_moe_into_hop routed input (enable_routed_input): engaged on every rank or none.
        self.routed_engaged = False
        self.routed_mode: Optional[str] = None
        self.routed_reason = "not enabled"
        self.carveout = carveout
        self.ctas = ctas
        self.threads = self.hidden_size // (8 * ctas)
        self.partials = ctas * (self.threads // 32)
        self.engaged = False
        reason = ""
        self.scope = selected_scope()
        # The next layer's indexer type matters only for a restricted scope.
        self.next_layer_full: Optional[bool] = None
        self.next_layer_idx: Optional[int] = None
        local_ok = bool(post_moe_hop.engaged)
        if not local_ok:
            reason = "post_moe_hop hop not engaged"
        else:
            why = hop_kernel_supported(self.hidden_size, self.top_k, self.mapping.tp_size, ctas)
            if why is not None:
                local_ok, reason = False, why
        if self.scope != UNRESTRICTED_SCOPE:
            full, self.next_layer_idx, how = next_layer_kind(post_moe_hop, next_layer_full_indexer)
            self.next_layer_full = full
            why = decline_reason(self.scope, full, self.next_layer_idx, how)
            if local_ok and why is not None:
                local_ok, reason = False, why
        if local_ok:
            try:
                why = self._precompile()
            except Exception as exc:  # noqa: BLE001 - converged across ranks below
                why = f"kernel compile failed: {exc!r}"
            if why is not None:
                local_ok, reason = False, why
        if post_moe_hop.engaged:
            # Same communicator and agreement helper as the post_moe_hop workspace.
            comm = PostMoeHopWorkspace.comm_for(self.mapping)
            if _all_ranks_agree(comm, local_ok):
                self.engaged = True
            else:
                reason = reason or "another rank is not capable"
        self.reason = reason
        dispatch.log_capability(FEATURE_ID, "deferred_hop", self.engaged, reason)
        if self.scope != UNRESTRICTED_SCOPE:
            into = "?" if self.next_layer_idx is None else self.next_layer_idx
            dispatch.log_capability(
                FEATURE_ID, f"scope[{self.scope}]_into_layer{into}", self.engaged, reason
            )

    def _precompile(self) -> Optional[str]:
        """Compile every kernel a packet of this hop can reach.

        Returns why a served projection shape is unsupported, or ``None``;
        compile errors propagate (converged across ranks by the caller).
        """
        from . import projection
        from .hop_kernel import compiled_hop, compiled_materialize

        for has_shared in (True, False):
            for scale_fp32 in (False, True):
                compiled_hop(
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
                    fault_delay_ns=self.fault_delay_ns,
                )
        compiled_materialize(self.hidden_size, self.ctas, self.partials, self.use_pdl)
        return projection.precompile_served(torch.cuda.current_device())

    def can_run(self, num_rows: int, attn_metadata) -> bool:
        """Rank-identical per-call decision, taken before the MoE runner runs."""
        if not self.engaged or not 0 < num_rows <= MAX_ROWS:
            return False
        if not self.post_moe_hop.can_run(num_rows):
            return False
        if (
            attn_metadata is None
            or attn_metadata.num_contexts != 0
            or (attn_metadata.num_generations == 0)
        ):
            return False
        ok, _ = _graph_mode_ok()
        return ok

    # ------------------------------------------------------------------
    # routed_moe_into_hop routed input (deferred_norm composed with routed_moe_into_hop): reachable
    # only after enable_routed_input(), i.e. with both "routed_moe_into_hop" and "deferred_norm"
    # selected. ------------------------------------------------------------------
    def _post_moe_routed_mode(self) -> Optional[str]:
        """The post_moe_hop's routed-input mode, or ``None`` when it is not enabled."""
        if not getattr(self.post_moe_hop, "_routed_enabled", False):
            return None
        return getattr(self.post_moe_hop, "routed_mode", None)

    def enable_routed_input(self) -> bool:
        """Collective: serve the post_moe_hop's routed input (routed_moe_into_hop) with the deferred hop.

        Requires this hop to be engaged and the post_moe_hop's routed input to be
        enabled (both rank-identical, agreed over the TP group). Compiles the
        routed variants of the post_moe_hop's current mode (shared output present
        and absent); every rank must succeed, otherwise the routed input
        stays off on all of them and the layer keeps ``PostMoeHop.hop_routed``.
        Call it again (collectively, outside capture) after the post_moe_hop
        switches modes.
        """
        from .hop_kernel import compiled_hop

        mode = self._post_moe_routed_mode()
        local_ok, reason = True, ""
        if not self.engaged:
            local_ok, reason = False, "deferred_norm hop not engaged"
        elif mode is None:
            local_ok, reason = False, "post_moe_hop hop's routed input is not enabled"
        else:
            try:
                for has_shared in (True, False):
                    compiled_hop(
                        self.hidden_size,
                        self.top_k,
                        self.mapping.tp_size,
                        self.post_moe_hop.max_rows,
                        self.ctas,
                        has_shared,
                        False,
                        self.use_pdl,
                        self.trigger_early,
                        STATE_WORDS,
                        self.carveout,
                        fault_delay_ns=self.fault_delay_ns,
                        routed_input=mode,
                    )
            except Exception as exc:  # noqa: BLE001 - converged across ranks below
                local_ok, reason = False, f"routed kernel compile failed: {exc!r}"
        agreed = False
        if self.engaged and mode is not None:
            # Same communicator and agreement helper as the post_moe_hop workspace;
            # both conditions above are rank-identical, so either every rank
            # reaches these collectives or none does. The post_moe_hop's mode comes
            # from each rank's environment: require the same mode everywhere
            # (one agreement per mode, in a fixed order).
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

    def can_run_routed(self, num_rows: int, attn_metadata) -> bool:
        """Rank-identical per-call decision for the routed input."""
        return (
            self.routed_engaged
            and self._post_moe_routed_mode() == self.routed_mode
            and self.can_run(num_rows, attn_metadata)
        )

    def _routed_view(self, num_rows: int):
        """``(mode, buffer view, scales)`` of the post_moe_hop's routed input for ``num_rows``."""
        base = self.post_moe_hop
        if self.routed_mode == "fp32":
            return "fp32", base.routed_acc_buffer(num_rows), None
        if self.routed_mode == "identity":
            rows, scales = base.routed_raw_buffers(num_rows)
            return "identity", rows, scales
        return "row", base.routed_input_buffer(num_rows), None

    def hop_routed(
        self,
        routed: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[DeferredNormHidden, torch.Tensor]:
        """Return ``(packet, residual_out)``; same inputs as ``PostMoeHop.hop_routed``.

        ``routed`` is the view of the post_moe_hop's routed-input buffer routed_moe wrote
        (``routed_output_into_hop``). Same arithmetic as ``hop_routed`` up to
        the residual; the RMSNorm is deferred into the packet.

        Raises:
            ValueError: the routed input is not engaged or ``can_run`` is
                false for this row count.
            PostMoeHopContractError: ``routed`` is not the hop's buffer for the row
                count, or a dtype / shape differs from the contract.
        """
        num_rows = residual.shape[0]
        if not (
            self.routed_engaged
            and self._post_moe_routed_mode() == self.routed_mode
            and 0 < num_rows <= MAX_ROWS
            and self.post_moe_hop.can_run(num_rows)
        ):
            raise ValueError(
                "deferred_norm routed hop called for an ineligible shape or before "
                "enable_routed_input(); check can_run_routed() first"
            )
        mode, expect, scales = self._routed_view(num_rows)
        rows_h = (num_rows, self.hidden_size)
        problems = []
        if (
            routed.data_ptr() != expect.data_ptr()
            or routed.dtype != expect.dtype
            or tuple(routed.shape) != tuple(expect.shape)
            or tuple(routed.stride()) != tuple(expect.stride())
        ):
            problems.append(
                f"routed {routed.dtype} {tuple(routed.shape)} is not the post_moe_hop hop's "
                f"{mode} buffer"
            )
        if residual.dtype != self.dtype or tuple(residual.shape) != rows_h:
            problems.append(f"residual {residual.dtype} {tuple(residual.shape)}")
        if shared_output is not None and (
            shared_output.dtype != self.dtype or tuple(shared_output.shape) != rows_h
        ):
            problems.append(f"shared_output {shared_output.dtype} {tuple(shared_output.shape)}")
        if norm_weight.dtype != self.dtype or tuple(norm_weight.shape) != (self.hidden_size,):
            problems.append(f"norm_weight {norm_weight.dtype} {tuple(norm_weight.shape)}")
        if problems:
            raise PostMoeHopContractError(
                "routed_moe_into_hop routed input violates the hop's contract (deferred_norm hop): "
                f"{'; '.join(problems)}"
            )
        if shared_output is not None:
            shared_output = self._vector_layout("shared_output", shared_output)
        residual = self._vector_layout("residual", residual)
        norm_weight = self._vector_layout("norm_weight", norm_weight)
        return self.launch_routed(routed, shared_output, residual, norm_weight, eps, scales=scales)

    def launch_routed(
        self,
        routed,
        shared_output,
        residual,
        norm_weight,
        eps,
        scales=None,
        ts: Optional[torch.Tensor] = None,
        row_bytes: Optional[int] = None,
    ) -> Tuple[DeferredNormHidden, torch.Tensor]:
        """Launch the routed variant on checked, vector-ready inputs (``hop_routed``).

        Tests only: ``ts`` as in :meth:`launch`; ``row_bytes`` overrides the
        routed buffer's row pitch (memcheck positive control).
        """
        if row_bytes is None:
            row_bytes = routed.stride(0) * routed.element_size()
        return self._launch(
            routed.data_ptr(),
            row_bytes,
            routed.shape[0],
            scales.data_ptr() if scales is not None else 0,
            0,
            False,
            self.routed_mode,
            shared_output,
            residual,
            norm_weight,
            eps,
            ts,
        )

    def _contract_violations(
        self,
        fc2_output,
        expert_scale_factor,
        expanded_idx_to_permuted_idx,
        shared_output,
        residual,
        norm_weight,
    ) -> List[str]:
        """Dtype/shape differences from the declared raw-tuple contract (rank-identical)."""
        num_rows = residual.shape[0]
        rows_h = (num_rows, self.hidden_size)
        problems = []
        if residual.dtype != self.dtype or tuple(residual.shape) != rows_h:
            problems.append(f"residual {residual.dtype} {tuple(residual.shape)}")
        if (
            fc2_output.dtype != self.dtype
            or fc2_output.dim() != 2
            or fc2_output.shape[1] < self.hidden_size
        ):
            problems.append(f"fc2_output {fc2_output.dtype} {tuple(fc2_output.shape)}")
        if expert_scale_factor.dtype not in _SCALE_DTYPES or tuple(expert_scale_factor.shape) != (
            num_rows,
            self.top_k,
        ):
            problems.append(
                f"expert_scale_factor {expert_scale_factor.dtype} "
                f"{tuple(expert_scale_factor.shape)}"
            )
        if expanded_idx_to_permuted_idx.dtype != torch.int32 or tuple(
            expanded_idx_to_permuted_idx.shape
        ) != (num_rows, self.top_k):
            problems.append(
                f"expanded_idx_to_permuted_idx {expanded_idx_to_permuted_idx.dtype} "
                f"{tuple(expanded_idx_to_permuted_idx.shape)}"
            )
        if shared_output is not None and (
            shared_output.dtype != self.dtype or tuple(shared_output.shape) != rows_h
        ):
            problems.append(f"shared_output {shared_output.dtype} {tuple(shared_output.shape)}")
        if norm_weight.dtype != self.dtype or tuple(norm_weight.shape) != (self.hidden_size,):
            problems.append(f"norm_weight {norm_weight.dtype} {tuple(norm_weight.shape)}")
        return problems

    def _vector_layout(self, name: str, t: torch.Tensor, row_major_2d: bool = False):
        """Return ``t`` or a 16-byte-aligned contiguous copy of it (rank-local)."""
        if _vector_ready(t, row_major_2d):
            return t
        misalignment, strides = t.data_ptr() % _VECTOR_BYTES, tuple(t.stride())
        if row_major_2d:
            t = t[:, : self.hidden_size]
        fixed = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        fixed.copy_(t)
        self.post_moe_hop.workspace.layout_repairs += 1
        if name not in self._repairs_logged:
            self._repairs_logged.add(name)
            from tensorrt_llm.logger import logger

            logger.warning(
                f"[glm_moe_dsa] deferred_norm/hop: {name} is not ready for 16-byte vector "
                f"access (data_ptr % 16 = {misalignment}, strides {strides}); "
                "copying it before every such call"
            )
        return fixed

    def __call__(
        self,
        fc2_output: torch.Tensor,
        expert_scale_factor: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[DeferredNormHidden, torch.Tensor]:
        """Return ``(packet, residual_out)``; same inputs as ``PostMoeHop``.

        Raises:
            ValueError: ``can_run`` is false for this row count.
            PostMoeHopContractError: a dtype or shape differs from the raw-tuple contract.
        """
        if not self.engaged or not 0 < residual.shape[0] <= MAX_ROWS:
            raise ValueError(
                "deferred_norm hop called for an ineligible shape; check can_run() first"
            )
        problems = self._contract_violations(
            fc2_output,
            expert_scale_factor,
            expanded_idx_to_permuted_idx,
            shared_output,
            residual,
            norm_weight,
        )
        if problems:
            raise PostMoeHopContractError(
                "raw MoE tuple violates the declared native raw-tuple contract after the runner "
                f"skipped its finalize (deferred_norm hop): {'; '.join(problems)}"
            )
        fc2_output = self._vector_layout("fc2_output", fc2_output, row_major_2d=True)
        expert_scale_factor = self._vector_layout("expert_scale_factor", expert_scale_factor)
        expanded_idx_to_permuted_idx = self._vector_layout(
            "expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx
        )
        if shared_output is not None:
            shared_output = self._vector_layout("shared_output", shared_output)
        residual = self._vector_layout("residual", residual)
        norm_weight = self._vector_layout("norm_weight", norm_weight)
        return self.launch(
            fc2_output,
            expert_scale_factor,
            expanded_idx_to_permuted_idx,
            shared_output,
            residual,
            norm_weight,
            eps,
        )

    def launch(
        self,
        fc2_output,
        expert_scale_factor,
        expanded_idx_to_permuted_idx,
        shared_output,
        residual,
        norm_weight,
        eps,
        ts: Optional[torch.Tensor] = None,
        fc2_rows: Optional[int] = None,
    ) -> Tuple[DeferredNormHidden, torch.Tensor]:
        """Launch on checked, vector-ready inputs (``__call__``).

        Tests only: ``ts``, int64 ``[R * ctas, 8]`` per-CTA timer stamps;
        ``fc2_rows`` overrides the permutation bound (memcheck positive
        control).
        """
        return self._launch(
            fc2_output.data_ptr(),
            fc2_output.stride(0) * 2,
            fc2_output.shape[0] if fc2_rows is None else fc2_rows,
            expert_scale_factor.data_ptr(),
            expanded_idx_to_permuted_idx.data_ptr(),
            expert_scale_factor.dtype.itemsize == 4,
            None,
            shared_output,
            residual,
            norm_weight,
            eps,
            ts,
        )

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
        from .hop_kernel import compiled_hop

        num_rows = residual.shape[0]
        ws = self.post_moe_hop.workspace
        a_frag = torch.empty(
            (frag_elems(self.hidden_size),), dtype=residual.dtype, device=residual.device
        )
        residual_out = torch.empty_like(residual)
        sumsq = torch.empty((num_rows, self.partials), dtype=torch.float32, device=residual.device)
        has_shared = shared_output is not None
        compiled = compiled_hop(
            self.hidden_size,
            self.top_k,
            self.mapping.tp_size,
            ws.max_rows,
            self.ctas,
            has_shared,
            scale_fp32,
            self.use_pdl,
            self.trigger_early,
            ws.state.numel(),
            self.carveout,
            ts is not None,
            self.fault_delay_ns,
            routed_input,
        )
        compiled(
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
            ws.state,
            self.mapping.tp_rank,
            self.timeout_ns,
            num_rows,
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

    def materialize(self, packet: DeferredNormHidden) -> torch.Tensor:
        """post_moe_hop's normalized output ``bf16((x * r) * gamma)`` from a packet."""
        return materialize(packet, self.ctas, self.use_pdl)


def materialize(
    packet: DeferredNormHidden, ctas: int = DEFAULT_CTAS, use_pdl: Optional[bool] = None
) -> torch.Tensor:
    from .hop_kernel import compiled_materialize

    if use_pdl is None:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        use_pdl = get_env_enable_pdl()
    x = packet.residual
    out = torch.empty_like(x)
    compiled = compiled_materialize(x.shape[1], ctas, packet.sumsq.shape[1], use_pdl)
    compiled(
        x.data_ptr(),
        packet.gamma.data_ptr(),
        packet.sumsq.data_ptr(),
        out.data_ptr(),
        packet.sumsq.view(torch.int32).view(-1),
        packet.eps,
        x.shape[0],
    )
    return out
