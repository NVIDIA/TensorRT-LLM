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
"""post_attention_hop: fused post-attention hop (the router contract).

One call replaces the native fused MNNVL one-shot ``AllReduce(RESIDUAL_RMS_NORM)``
of the post-attention boundary (``PRE_MOE_FUSION`` of MoE and MTP layers) and,
optionally, the router GEMM that consumes its output:

* ``residual_out = bf16(sum_rank partial) + residual`` and
  ``normed = RMSNorm(residual_out) * gamma`` are bit-identical to the native
  ``AllReduce(RESIDUAL_RMS_NORM)`` result on the MNNVL path, which normalizes
  the bf16-rounded ``residual_out``. They are not bit-identical to the unfused
  chain of a plain ``AllReduce`` followed by a separate add+RMSNorm (the native
  path of layers without ``PRE_MLP_FUSION``, e.g. the dense layers), which
  normalizes the FP32 sum: there only ``residual_out`` matches and ``normed``
  can differ by up to two bf16 ulps. Such call sites are not a supported
  destination.
* ``router_logits = normed @ router_weight.T`` in FP32 (``[R, 256]``), computed
  from the reduced, normalized rows, identical on every rank. They are not
  bit-identical to the native ``dsv3_router_gemm_op`` logits: both accumulate
  the same bf16 operands in FP32, but in a different order (tensor-core tiles
  summed in a fixed order), with an error against an FP64 reference of the
  same size as native's. Routing that consumes them can
  therefore select a different expert set than native on rows whose 8th and
  9th expert scores lie within that rounding error.

The exchange uses its own multicast workspace (three rotating Lamport buffers
and a device-side protocol state), separate from every native or other-package
workspace. One workspace serves all layers, CUDA graphs and eager calls of a
TP group, which issue their hops in order on the model's main stream; every rank
must issue the same sequence of hops. The native MNNVL all-reduce has the same
contract: one workspace and one device-side buffer state per TP group
(``MNNVLAllReduce.allreduce_mnnvl_workspaces``).

Bit-identity holds against the native MNNVL one-shot kernel, which the native
``AllReduce`` uses for these calls only when it selected MNNVL (strategy AUTO
across nodes, or MNNVL) and the GPU has at least ``8 * rows`` SMs (cluster of
8 CTAs per row). The hop therefore engages only for a call site whose native
``AllReduce`` is on MNNVL and for row counts whose grid fits the smallest SM
count of the group.

Eligibility is decided from rank-uniform facts only (feature selection, the
group's capability and SM count agreed at construction, the call site's native
all-reduce configuration, dtype and shape of the call), so all ranks take the
same path for every call and the collective sequence never diverges.
Unsupported calls stay on the native path before any device work. Host-side
counters (:func:`engagement_counts`) record engaged and native-fallback calls
per Python call, i.e. eager runs and CUDA graph captures, not graph replays.
"""

import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.mapping import Mapping

from ... import native_ops
from ...dispatch import log_capability

FEATURE = "post_attention_hop"
HIDDEN_SIZE = 6144
NUM_RANKS = 8
NUM_EXPERTS = 256
MAX_ROWS = 8
NUM_LAMPORT_BUFFERS = 3
LAMPORT_BUFFER_BYTES = MAX_ROWS * NUM_RANKS * HIDDEN_SIZE * 2
WORKSPACE_BYTES = LAMPORT_BUFFER_BYTES * NUM_LAMPORT_BUFFERS
# CTAs per cluster (one per 768-element slice of the hidden dimension) and the
# router layout's expert groups (kernel: kSlices, launchRouter).
CLUSTER_CTAS = 8
ROUTER_EXPERT_GROUPS = 4
ROUTER_MAX_ROW_GROUPS = 3

# Host-side call counters for dispatch-engagement reporting. They count Python
# calls (eager runs and CUDA graph captures), not graph replays.
_counts: Dict[str, int] = {}


def _count(key: str) -> None:
    _counts[key] = _counts.get(key, 0) + 1


def engagement_counts() -> Dict[str, int]:
    """Copy of the host-side call counters.

    Keys: ``engaged`` (hop calls), ``engaged_router`` (hop calls that also
    produced router logits), ``native_fallback`` (calls ``can_run`` sent to the
    native path) and ``native_fallback:<reason>`` per reason.
    """
    return dict(_counts)


def reset_engagement_counts() -> None:
    """Clear the host-side call counters."""
    _counts.clear()


def required_sms(rows: int, router: bool) -> int:
    """SMs one hop of ``rows`` rows needs.

    The native one-shot kernel uses one 8-CTA cluster per row only while
    ``8 * rows`` fits the SM count; the hop reproduces that layout. With router
    logits every CTA of the grid (``4 * min(rows, 3)`` clusters of 8 CTAs, one
    CTA per SM) must be co-resident, because CTAs spin on data that other CTAs
    of every rank publish.
    """
    need = CLUSTER_CTAS * rows
    if router:
        need = max(need, CLUSTER_CTAS * ROUTER_EXPERT_GROUPS * min(rows, ROUTER_MAX_ROW_GROUPS))
    return need


def native_allreduce_ineligibility(
    native_allreduce: Optional[Any], mapping: Mapping
) -> Optional[str]:
    """Reason the call site's native all-reduce is not the MNNVL path, or None.

    Args:
        native_allreduce: the call site's native ``AllReduce`` module (the one
            the hop replaces), or None when the call site does not pass it.
        mapping: the TP group's mapping.

    With ``native_allreduce`` the check is exact: the native fused
    ``RESIDUAL_RMS_NORM`` call runs the MNNVL one-shot kernel iff the module
    created its ``MNNVLAllReduce`` (strategy AUTO across nodes, or MNNVL).
    Without it, the hop requires that some native module of this TP group
    created the MNNVL workspace, which no strategy other than AUTO/MNNVL does.
    The answer depends only on configuration that is identical on all ranks.
    """
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    if native_allreduce is not None:
        if getattr(native_allreduce, "mnnvl_allreduce", None) is None:
            strategy = getattr(native_allreduce, "strategy", None)
            name = getattr(strategy, "name", str(strategy))
            return f"native all-reduce ({name}) does not use MNNVL"
        return None
    if mapping not in MNNVLAllReduce.allreduce_mnnvl_workspaces:
        return "no native MNNVL all-reduce on this TP group"
    return None


def static_ineligibility(mapping: Mapping, hidden_size: int, dtype: torch.dtype) -> Optional[str]:
    """Reason the configuration can never use post_attention_hop, or None.

    Every input is identical on all ranks of the TP group, so the answer is
    rank-uniform without communication.
    """
    from tensorrt_llm._utils import mpi_disabled

    if mapping.tp_size != NUM_RANKS:
        return f"tp_size {mapping.tp_size} != {NUM_RANKS}"
    if mapping.enable_attention_dp:
        return "attention DP has no post-attention all-reduce"
    if mapping.has_cp():
        return "context parallelism is not supported"
    if hidden_size != HIDDEN_SIZE:
        return f"hidden size {hidden_size} != {HIDDEN_SIZE}"
    if dtype != torch.bfloat16:
        return f"dtype {dtype} is not bfloat16"
    if mpi_disabled():
        return "workspace setup needs MPI"
    return None


class PostAttnHop:
    """Workspace and entry point of the fused post-attention hop for one TP group."""

    _instances: Dict[Mapping, Optional["PostAttnHop"]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        mapping: Mapping,
        hidden_size: int = HIDDEN_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        native_allreduce: Optional[Any] = None,
    ) -> Optional["PostAttnHop"]:
        """Return the group's hop, creating its workspace collectively on first use.

        Must be called at the same point of model construction on every rank of
        the TP group.

        Args:
            mapping: the TP group's mapping.
            hidden_size: model hidden size.
            dtype: activation dtype.
            native_allreduce: the call site's native ``AllReduce`` module whose
                fused ``RESIDUAL_RMS_NORM`` call the hop replaces; the hop
                serves this call site only if that module uses MNNVL.

        Returns:
            The hop, or None (native path) when the call site or the
            configuration is ineligible or any rank failed to set up the
            workspace.
        """
        with cls._lock:
            site_reason = native_allreduce_ineligibility(native_allreduce, mapping)
            if site_reason is not None:
                log_capability(FEATURE, f"post_attn_hop:{site_reason}", False, site_reason)
                return None
            if mapping in cls._instances:
                return cls._instances[mapping]
            reason = static_ineligibility(mapping, hidden_size, dtype)
            hop = None
            if reason is None:
                hop = cls(mapping)
                if not hop.enabled:
                    reason = hop.disabled_reason
                    hop = None
            log_capability(FEATURE, "post_attn_hop", hop is not None, reason or "")
            cls._instances[mapping] = hop
            return hop

    def __init__(self, mapping: Mapping):
        from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
        )
        from tensorrt_llm._utils import mpi_comm

        self.mapping = mapping
        self.rank = mapping.tp_rank
        self.enabled = False
        self.disabled_reason = ""
        self.handle = None
        self.uc_buffer = None
        self.mc_ptr = 0
        self.flags = None
        self.comm = None
        self.min_sm_count = 0

        # Local capability first, agreed over the world communicator so that
        # either every rank creates the TP communicator and workspace or none.
        local_reason = ""
        if not MnnvlMemory.supports_mnnvl():
            local_reason = "MNNVL is not supported on this device"
        world = mpi_comm()
        if world.allreduce(0 if local_reason else 1) != world.Get_size():
            self.disabled_reason = local_reason or "another rank is ineligible"
            return
        # The smallest SM count of all ranks bounds the row counts every rank
        # accepts (see required_sms), so per-call decisions stay rank-uniform.
        from mpi4py import MPI

        self.min_sm_count = world.allreduce(
            torch.cuda.get_device_properties(_mnnvl_device_index(mapping)).multi_processor_count,
            op=MPI.MIN,
        )

        comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[Exception] = None
        try:
            force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
            self.handle = _make_mnnvl_mcast_buffer(
                comm,
                WORKSPACE_BYTES,
                mapping,
                use_fabric_handle=force_mn or mapping.is_multi_node(),
            )
            self.uc_buffer = self.handle.get_uc_buffer(
                mapping.tp_rank, (WORKSPACE_BYTES // 4,), torch.float32, 0
            )
            self.mc_ptr = self.handle.get_mc_buffer((1,), torch.float32, 0).data_ptr()
            # Built like the native MNNVL flags: torch.tensor is not redirected
            # to the meta device under MetaInitMode during model construction.
            self.flags = torch.tensor(
                [0, 0, 0, 0],
                dtype=torch.int32,
                device=torch.device("cuda", _mnnvl_device_index(mapping)),
            )
            self.reset_protocol()
        except (RuntimeError, ValueError) as err:
            error = err
        ok = comm.allreduce(0 if error is not None else 1) == comm.Get_size()
        if not ok:
            self.disabled_reason = (
                f"workspace setup failed: {error}"
                if error is not None
                else "workspace setup failed on another rank"
            )
            self.handle = None
            self.uc_buffer = None
            self.flags = None
            return
        self.comm = comm
        self.enabled = True

    def reset_protocol(self) -> None:
        """Fill every Lamport buffer with the sentinel and zero the protocol state.

        Collective in effect: every rank must reset before any rank issues the
        next hop, which the caller guarantees with a barrier.
        """
        with torch.inference_mode():
            self.uc_buffer.fill_(-0.0)
            self.flags.zero_()
        torch.cuda.synchronize()

    def call_ineligibility(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        router_weight: Optional[torch.Tensor] = None,
    ) -> Optional[str]:
        """Reason this call must take the native path, or None (no device work)."""
        if not self.enabled:
            return "hop disabled"
        if residual is None:
            return "no residual"
        if (
            hidden_states.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or norm_weight.dtype != torch.bfloat16
        ):
            return "dtype"
        if (
            hidden_states.dim() < 2
            or hidden_states.shape[-1] != HIDDEN_SIZE
            or residual.shape != hidden_states.shape
            or norm_weight.shape != (HIDDEN_SIZE,)
        ):
            return "shape"
        rows = hidden_states.numel() // HIDDEN_SIZE
        if not 1 <= rows <= MAX_ROWS:
            return f"rows outside [1, {MAX_ROWS}]"
        if router_weight is not None and (
            router_weight.dtype != torch.bfloat16
            or router_weight.shape != (NUM_EXPERTS, HIDDEN_SIZE)
            or not router_weight.is_contiguous()
        ):
            return "router weight"
        if required_sms(rows, router_weight is not None) > self.min_sm_count:
            return "SM count"
        return None

    def can_run(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: torch.Tensor,
        router_weight: Optional[torch.Tensor] = None,
    ) -> bool:
        """Whether this call can use the fused hop (rank-uniform, no device work).

        A False answer is counted as a native fallback (engagement_counts).
        """
        reason = self.call_ineligibility(hidden_states, residual, norm_weight, router_weight)
        if reason is not None:
            _count("native_fallback")
            _count(f"native_fallback:{reason}")
            return False
        return True

    def __call__(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        router_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the hop; returns ``(normed, residual_out, router_logits or None)``.

        Callers check :meth:`can_run` first.
        """
        _count("engaged")
        if router_weight is not None:
            _count("engaged_router")
        normed, residual_out, logits = native_ops.post_attention_hop(
            hidden_states.contiguous(),
            residual.contiguous(),
            norm_weight.contiguous(),
            float(eps),
            router_weight,
            self.uc_buffer,
            self.mc_ptr,
            self.flags,
            self.rank,
            NUM_RANKS,
        )
        return normed, residual_out, (logits if router_weight is not None else None)
