# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel all-reduce of K3's row-parallel projections, as one fixed kernel call.

:class:`TpAllReduce` decides its transport once, at construction, from the mapping and the requested
strategy (``ModelConfig.allreduce_strategy``, ``AUTO`` in every K3 recipe):

* **MNNVL**: ``AUTO`` / ``MNNVL`` on a group MNNVL serves (BF16 / FP16 / FP32, no CP, aarch64 with
  every NVLink up, and the group spans nodes, an explicit ``MNNVL``, or ``TLLM_TEST_MNNVL=1``). Each
  call is ``trtllm::mnnvl_fusion_allreduce`` on the Lamport workspace of
  ``get_or_scale_allreduce_mnnvl_workspace`` (one per mapping, shared with every other MNNVL
  all-reduce and the pushed all-reduces of ``layers.py``; allocated here, a TP-wide collective,
  and grown on the first eager call that needs more). The kernel is one-shot up to
  ``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES`` of ``tokens * hidden * ranks * element size``, two-shot above.
  Every K3 all-reduce runs here on Rubin / GB300 NVL72: helix GEN groups span nodes, and the dense
  MLP's TP-4 subgroup is declared over the whole world.
* **The trtllm all-reduce op** (the catalog's ``allreduce`` entry: ``trtllm::allreduce`` on the IPC
  workspace of ``get_allreduce_workspace``) for any other group, with the strategy fixed: an
  explicit one as given, ``AUTO`` as the op's own size heuristic
  (``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1`` behaviour of main's ``AllReduce``; main otherwise
  autotunes the strategy there).
* **Identity** with one rank or under attention DP (no TP group to reduce over).

Fusions: ``NONE`` and the residual / plain RMSNorm epilogues the MNNVL kernel takes
(``AllReduceParams``), with main's output contract (a tensor, or a tuple for fusions with several
outputs).

For ``layers.py`` an MNNVL instance also answers ``mnnvl_allreduce`` (itself), ``mapping``,
``dtype``, ``symm_mem_allreduce`` (None) and ``get_required_workspace_size``.

Pushed MNNVL all-reduce, private to the Kimi K3 Rubin package.

A producer kernel may store its BF16 output straight into the Lamport buffers of the MNNVL
all-reduce that follows it (the K3 low-M FP8 GEMM's push epilogue, ``kernels/k3_lamport_ar.py``).
:func:`lamport_push_target` hands the buffers to the producer and :func:`finish_lamport_push`
completes the all-reduce, bitwise like the C++ kernel the call replaces.

This is the push API of the Rubin agentic branch (``189fe67b81``, ``distributed/ops.py``:
``LamportPushTarget``/``LamportPending`` and the ``AllReduce`` /
``MNNVLAllReduce`` methods ``lamport_push_target``, ``lamport_push_tensor`` and
``finish_lamport_push``), as free functions that take the all-reduce: ``layers.TpAllReduce``
(or main's ``AllReduce``), read through ``mnnvl_allreduce`` (the MNNVL transport: its ``mapping``,
``dtype`` and ``get_required_workspace_size``) and ``symm_mem_allreduce``. They work on the shared
MNNVL workspace of ``get_or_scale_allreduce_mnnvl_workspace`` (same dict keys and the same 9-word
``buffer_flags`` layout).

Differences from the branch, all forced by main:

* The C++ one-/two-shot switch is ``distributed.ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES``
  (``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES``): the pushed call must reproduce the kernel main's
  ``MNNVLAllReduce`` would have run.
* The device view of a workspace is built on the first eager call; during CUDA graph capture a
  workspace without a view, or one that would have to grow, gives no target (the caller falls
  back to the plain all-reduce). The eager warm-up forward that precedes every capture builds it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import allreduce
from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    MNNVLAllReduce,
    get_allreduce_workspace,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..common.sm107_launch import USE_PDL

MNNVL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
#: Fusions ``trtllm::mnnvl_fusion_allreduce`` runs (besides ``NONE``).
MNNVL_FUSION_OPS = frozenset(
    {
        AllReduceFusionOp.RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4,
    }
)


def mnnvl_serves(mapping: Mapping, dtype: torch.dtype, strategy: AllReduceStrategy) -> bool:
    """Whether the all-reduce of ``mapping`` runs on MNNVL for ``strategy`` (``AUTO`` / ``MNNVL``)."""
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        return False
    if dtype not in MNNVL_DTYPES or mapping.has_cp():
        return False
    # The engine's own decision (its single-node test bypass included) on the dtype and the
    # mapping checked above.
    return MNNVLAllReduce.is_mnnvl(
        mapping, dtype, explicitly_requested=strategy == AllReduceStrategy.MNNVL
    )


def mnnvl_workspace_bytes(num_tokens: int, hidden: int, world: int, dtype: torch.dtype) -> int:
    """Lamport buffer bytes one MNNVL call of ``[num_tokens, hidden]`` needs."""
    elem = torch.tensor([], dtype=dtype).element_size()
    if num_tokens * hidden * world * elem <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES:
        return num_tokens * hidden * world * elem
    return 2 * math.ceil(num_tokens / world) * world * hidden * elem


class TpAllReduce:
    """The all-reduce over ``mapping``'s TP group; ``self(x)`` returns the reduced tensor."""

    def __init__(
        self,
        mapping: Mapping,
        dtype: torch.dtype,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.strategy = strategy
        self.symm_mem_allreduce = None
        self.transport = "identity"
        self.workspace = None
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return
        if mnnvl_serves(mapping, dtype, strategy):
            self.transport = "mnnvl"
            get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)
            return
        if strategy in (
            AllReduceStrategy.MNNVL,
            AllReduceStrategy.SYMM_MEM,
            AllReduceStrategy.UB,
            AllReduceStrategy.LOWPRECISION,
        ):
            raise ValueError(f"K3 all-reduce: strategy {strategy.name} is not served here")
        self.transport = "op"
        if strategy not in (AllReduceStrategy.NCCL, AllReduceStrategy.NCCL_SYMMETRIC):
            self.workspace = get_allreduce_workspace(mapping)

    get_required_workspace_size = staticmethod(mnnvl_workspace_bytes)

    @property
    def mnnvl_allreduce(self) -> Optional["TpAllReduce"]:
        """Self on the MNNVL transport (what ``layers.py`` reads), else None."""
        return self if self.transport == "mnnvl" else None

    def __call__(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        params = all_reduce_params
        if self.transport == "identity" or (params is not None and not params.enable_allreduce):
            return x
        x = x.contiguous()
        params = params or AllReduceParams()
        if self.transport == "mnnvl":
            return self._mnnvl(x, params)
        return allreduce(
            x,
            params.residual,
            params.norm_weight,
            params.scale,
            params.bias,
            self.workspace,
            self.mapping.tp_group,
            self.strategy,
            params.fusion_op,
            params.eps,
            params.trigger_completion_at_end,
        )

    def _mnnvl(
        self, x: torch.Tensor, params: AllReduceParams
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        fusion_op = params.fusion_op
        is_fusion = fusion_op != AllReduceFusionOp.NONE
        if is_fusion and fusion_op not in MNNVL_FUSION_OPS:
            raise ValueError(f"the MNNVL all-reduce does not fuse {fusion_op!r}")
        hidden = x.shape[-1]
        num_tokens = x.numel() // hidden
        need = mnnvl_workspace_bytes(num_tokens, hidden, self.mapping.tp_size, self.dtype)
        if need >= 2**32 - 1:
            raise ValueError(
                f"MNNVL all-reduce of ({num_tokens}, {hidden}) over {self.mapping.tp_size} ranks "
                f"needs {need} workspace bytes, past the kernel's uint32 limit"
            )
        workspace = get_or_scale_allreduce_mnnvl_workspace(
            self.mapping, self.dtype, buffer_size_bytes=need
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            x,
            params.norm_weight,
            params.residual,
            params.eps,
            workspace["uc_buffer"].view(self.dtype).view(3, -1),
            workspace["buffer_flags"],
            is_fusion,
            params.scale,
            int(fusion_op),
            fp8_payload=False,
        )
        if is_fusion and len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]


# Pushed all-reduces of at most this many bytes (num_tokens * hidden_dim * group_size * 2) use the
# one-shot transport, larger ones the two-shot transport: the transport of the C++ kernel the
# call replaces. The result always reproduces the C++ kernel main selects.
_K3_AR_PUSH_ONE_SHOT_MAX_BYTES = int(_MNNVL_ONE_SHOT_THRESHOLD_BYTES)
# Pushed all-reduces whose consumer finishes them (``deferred``: the K3 attn_res kernel reduces
# the slots itself, after o_proj and after the row-parallel MoE tail) use the one-shot transport up
# to this many bytes, the two-shot transport above.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = _K3_AR_PUSH_ONE_SHOT_MAX_BYTES


@dataclasses.dataclass(frozen=True)
class LamportWorkspaceView:
    """Device addresses of one MNNVL all-reduce workspace for pushed calls, as the routed experts'
    output push takes them.

    ``buffer``: this rank's unicast mapping of the workspace; ``mc_base``: its multicast mapping;
    ``peer_ptrs``: int64 CUDA tensor with every rank's unicast mapping (this rank's own at index
    ``rank``); ``flags``: the uint32 rotation state shared with the C++ kernels;
    ``bytes_per_buffer``: size of one of the three Lamport buffers.
    """

    world: int
    rank: int
    buffer: torch.Tensor
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: torch.Tensor
    bytes_per_buffer: int

    @property
    def uc_local(self) -> int:
        """Device address of ``buffer``."""
        return self.buffer.data_ptr()


# Device views of the workspaces for pushed all-reduces, by the address of the workspace's buffer
# (a scaled workspace is a new buffer).
_LAMPORT_VIEWS: Dict[int, object] = {}
# The latest shared MNNVL workspace an eager call here got, by all-reduce mapping (workspaces only
# grow: a later one is at least as large).
_SEEN_WORKSPACES: Dict[Mapping, Dict[str, Any]] = {}


class LamportPushTarget:
    """One MNNVL all-reduce whose input the producer pushes.

    Made by :func:`lamport_push_target` for a single call. The producer stores its BF16
    ``[num_tokens, hidden]`` output into the Lamport slots of ``workspace`` (a
    ``k3_lamport_ar.LamportWorkspace``) with the ``one_shot`` or two-shot transport and sets
    ``pushed``; :func:`finish_lamport_push` then writes the all-reduce result into ``output``,
    reproducing the C++ one-shot kernel if ``cpp_one_shot``, else the two-shot kernel.
    """

    def __init__(self, workspace, one_shot: bool, cpp_one_shot: bool, output: torch.Tensor):
        self.workspace = workspace
        self.one_shot = one_shot
        self.cpp_one_shot = cpp_one_shot
        self.output = output
        self.pushed = False
        # Set by ``lamport_push_target(deferred=True)``: the one-shot transport, finished by the
        # consumer (``LamportPending``).
        self.deferred = False


class LamportPending:
    """A pushed all-reduce whose finish is left to its consumer
    (``finish_lamport_push(..., deferred=True)``).

    Every rank has stored its partial into the slots of the call (one-shot or two-shot
    transport), but no kernel has reduced them or advanced the workspace's rotation state: the
    consumer's kernel does both (``k3_attn_res_lamport``), or ``materialize`` runs
    the finish kernel into a tensor. Exactly one of the two must run, before the next all-reduce on
    the workspace.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.output.shape)

    @property
    def finish_args(self) -> Tuple[int, int, bool, int, int]:
        """``(finish_world, finish_chunk, finish_two_shot, mc_base, rank)`` of
        ``k3_attn_res_lamport``: the ranks per partial sum of the C++ kernel the result
        reproduces, the transport and the workspace addresses."""
        from ...kernels.k3r_comm import k3_lamport_ar

        ws = self.workspace
        return (
            ws.world,
            k3_lamport_ar.reduce_chunk(ws.world, self.target.cpp_one_shot),
            not self.target.one_shot,
            ws.mc_base,
            ws.rank,
        )

    def materialize(self) -> torch.Tensor:
        """Finish the all-reduce into a BF16 tensor (the finish kernel on the current stream)."""

        from .k3_lamport_finish import k3_lamport_finish

        return k3_lamport_finish(
            self.workspace,
            self.target.output,
            self.target.one_shot,
            use_pdl=USE_PDL,
            cpp_one_shot=self.target.cpp_one_shot,
        )


def _lamport_view(mnnvl: Any, workspace: dict):
    """The ``k3_lamport_ar.LamportWorkspace`` of an MNNVL workspace (built once per buffer)."""
    key = workspace["uc_buffer"].data_ptr()
    view = _LAMPORT_VIEWS.get(key)
    if view is None:
        from ...kernels.k3r_comm import k3_lamport_ar

        view = k3_lamport_ar.LamportWorkspace.from_mnnvl(
            workspace, mnnvl.mapping.tp_size, mnnvl.mapping.tp_rank
        )
        _LAMPORT_VIEWS[key] = view
        logger.info_once(
            f"[MNNVL AllReduce] pushed all-reduce: Lamport workspace view "
            f"for tp_size {mnnvl.mapping.tp_size} ({view.bytes_per_buffer} bytes per buffer)",
            key=("k3_ar_push_view", mnnvl.mapping.tp_size),
        )
    return view


def mnnvl_lamport_push_target(
    mnnvl: Any,
    num_tokens: int,
    hidden_dim: int,
    one_shot: Optional[bool] = None,
) -> Optional[LamportPushTarget]:
    """``MNNVLAllReduce.lamport_push_target`` of the branch: a target for a plain all-reduce of
    BF16 ``[num_tokens, hidden_dim]`` on the workspace of ``mnnvl`` (an all-reduce's MNNVL
    transport, ``mnnvl_allreduce``), or None. ``one_shot`` forces the transport instead of the
    size rule."""
    from ...kernels.k3r_comm import k3_lamport_ar

    world = mnnvl.mapping.tp_size
    nbytes = num_tokens * hidden_dim * world * 2
    cpp_one_shot = nbytes <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    if one_shot is None:
        one_shot = nbytes <= _K3_AR_PUSH_ONE_SHOT_MAX_BYTES
    need = max(
        mnnvl.get_required_workspace_size(num_tokens, hidden_dim, world, mnnvl.dtype),
        k3_lamport_ar.required_bytes(num_tokens, hidden_dim, world, one_shot),
    )
    if need >= 2**32 - 1:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        # Main raises when a workspace is created or grows during capture, and building the view
        # is a host-to-device copy: both must have happened in the eager warm-up of this shape.
        seen = _SEEN_WORKSPACES.get(mnnvl.mapping)
        if (
            seen is None
            or seen["buffer_size_bytes"] < need
            or seen["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS
        ):
            return None
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, mnnvl.dtype, buffer_size_bytes=need
    )
    if capturing:
        if workspace["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS:
            # Another all-reduce grew the workspace after the warm-up's push here.
            return None
    else:
        _SEEN_WORKSPACES[mnnvl.mapping] = workspace
    view = _lamport_view(mnnvl, workspace)
    output = torch.empty(
        num_tokens, hidden_dim, dtype=mnnvl.dtype, device=workspace["uc_buffer"].device
    )
    return LamportPushTarget(view, one_shot, cpp_one_shot, output)


def mnnvl_workspace_mpi_comm(workspace: Dict[str, object]):
    """The MPI communicator of a shared MNNVL all-reduce workspace
    (``get_or_scale_allreduce_mnnvl_workspace``), for the package's private Lamport workspaces:
    ``McastGPUBuffer`` takes its Fortran handle. The workspace keys it ``"comm"``; under a non-MPI
    orchestrator that entry is the TP ProcessGroup, which these workspaces do not support."""
    comm = workspace.get("comm")
    if comm is None or not hasattr(comm, "py2f"):
        raise RuntimeError(
            "Kimi K3 private MNNVL workspaces need the MPI communicator of the shared MNNVL "
            "workspace (MPI orchestrator, workspace attached)"
        )
    return comm


def _mnnvl_finish(
    target: LamportPushTarget, deferred: bool = False
) -> Union[torch.Tensor, LamportPending]:
    """``MNNVLAllReduce.finish_lamport_push`` of the branch."""

    from .k3_lamport_finish import k3_lamport_finish

    if deferred:
        return LamportPending(target)
    return k3_lamport_finish(
        target.workspace,
        target.output,
        target.one_shot,
        use_pdl=USE_PDL,
        cpp_one_shot=target.cpp_one_shot,
    )


def lamport_push_target(
    allreduce: Any,
    num_tokens: int,
    hidden_dim: int,
    all_reduce_params: Optional[AllReduceParams] = None,
    deferred: bool = False,
) -> Optional[LamportPushTarget]:
    """A ``LamportPushTarget`` if ``allreduce`` would run the MNNVL all-reduce on a BF16
    ``[num_tokens, hidden_dim]`` input with these parameters (plain sum, no fusion); else None.

    A producer that stores its output into the target (and sets ``pushed``) replaces the input of
    this all-reduce; :func:`finish_lamport_push` then returns the same result the all-reduce would.
    Targets must be finished in the order they were pushed, by every rank. ``deferred`` sets
    ``target.deferred`` for ``finish_lamport_push(deferred=True)``, whose consumer finishes the
    call; it takes the one-shot transport where the call is at most
    ``_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES`` (``num_tokens * hidden_dim * world * 2``), the
    two-shot one above.
    """
    mnnvl = getattr(allreduce, "mnnvl_allreduce", None)
    if (
        mnnvl is None
        or allreduce.mapping.tp_size == 1
        or getattr(allreduce, "symm_mem_allreduce", None) is not None
    ):
        return None
    if all_reduce_params is not None and (
        not all_reduce_params.enable_allreduce
        or all_reduce_params.fusion_op != AllReduceFusionOp.NONE
    ):
        return None
    if mnnvl.dtype != torch.bfloat16 or num_tokens <= 0 or hidden_dim % 8:
        return None
    if not deferred:
        return mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim)
    one_shot = (
        num_tokens * hidden_dim * allreduce.mapping.tp_size * 2
        <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
    )
    target = mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim, one_shot=one_shot)
    if target is not None:
        target.deferred = True
    return target


def lamport_push_tensor(allreduce: Any, target: LamportPushTarget, x: torch.Tensor) -> None:
    """Push the contiguous BF16 ``[num_tokens, hidden_dim]`` tensor ``x`` into ``target`` (a copy
    kernel), for a producer without the push epilogue."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape

    from .k3_lamport_push import k3_lamport_push

    k3_lamport_push(target.workspace, x.contiguous(), target.one_shot, use_pdl=USE_PDL)
    target.pushed = True


def finish_lamport_push(
    allreduce: Any,
    target: LamportPushTarget,
    deferred: bool = False,
) -> Union[torch.Tensor, LamportPending]:
    """The all-reduce result of a target the producer has pushed: a tensor, or with ``deferred``
    (where ``target.deferred``) a ``LamportPending`` that the next kernel finishes."""
    del allreduce  # the target carries its workspace; kept for the branch's call shape
    if not target.pushed:
        raise RuntimeError("the producer did not push this all-reduce")
    return _mnnvl_finish(target, deferred and target.deferred)


__all__ = [
    "LamportPending",
    "LamportPushTarget",
    "finish_lamport_push",
    "lamport_push_target",
    "lamport_push_tensor",
    "mnnvl_lamport_push_target",
]
