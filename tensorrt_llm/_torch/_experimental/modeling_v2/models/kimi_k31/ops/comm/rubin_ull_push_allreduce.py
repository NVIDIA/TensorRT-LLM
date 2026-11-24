# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 ULL target (``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``): the Lamport push all-
reduce workspaces and targets (the producers' pushed all-reduce sites, the fabric-handle choice).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from tensorrt_llm._torch.distributed.ops import (
    _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
    AllReduce,
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..common.rubin_ull_features import active as k31_features

# ar_push: producers may push into the all-reduce's Lamport buffers. Read at import.
_K3_AR_PUSH = bool(k31_features().ar_push)
# Pushed all-reduces of at most this many bytes (num_tokens * hidden_dim * group_size * 2) use the
# one-shot transport, larger ones the two-shot transport: the transport of the C++ kernel the call
# replaces. The result always reproduces the C++ kernel the shared threshold selects.
_K3_AR_PUSH_ONE_SHOT_MAX_BYTES = int(_MNNVL_ONE_SHOT_THRESHOLD_BYTES)
# Pushed all-reduces whose consumer finishes them (``deferred``: the K3 attn_res kernel reduces the
# slots itself, ar_finish_fused / rp_tail) use the one-shot transport up to this many bytes, the
# two-shot transport above.
_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES = _K3_AR_PUSH_ONE_SHOT_MAX_BYTES

# Device views of the workspaces for pushed all-reduces, by the address of the workspace's buffer
# (a scaled workspace is a new buffer).
_LAMPORT_VIEWS: Dict[int, object] = {}


def use_fabric_handle(mapping: Mapping) -> bool:
    """Whether the package's own multicast buffers use fabric handles: across nodes.

    The shared workspace also takes them on one node under main's ``TRTLLM_FORCE_MNNVL_AR=1``,
    which no production configuration set; POSIX FD handles are equally valid on one node.
    """
    return mapping.is_multi_node()


class LamportPushTarget:
    """One MNNVL all-reduce whose input the producer pushes (``ar_push``).

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
        # The call carries all-gather rows (``k3_lamport_ar`` "ag", one-shot transport): its
        # result is ``bf16(y + bf16(sum of the slots))``.
        self.ag = False


class LamportResult:
    """The result of a pushed two-shot all-reduce left in the MNNVL workspace
    (``finish_lamport_push(..., in_place=True)``).

    A consumer that runs next reads its rows with
    ``k3_lamport_ar.poll_result_v4(workspace.uc_local, workspace.flags, ...)`` after its own
    ``griddepcontrol.wait``; ``materialize`` copies them into a tensor instead. The rows live until
    the next all-reduce on the workspace re-arms their buffer: the next pushed call's finish kernel
    does that as soon as it is resident, which its producer allows only after its own
    ``griddepcontrol.wait`` (every earlier grid, the consumer included, has then completed); the
    C++ kernels re-arm after their wait. So the consumer must lie on every dependency path from this
    call to the next one on the workspace, and every kernel between them must wait before it
    triggers or completes.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.output.shape)

    def materialize(self) -> torch.Tensor:
        """The result as a BF16 tensor (a copy kernel on the current stream)."""
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

        return k3_lamport_ar.run_lamport_gather(
            self.workspace, self.target.output, use_pdl=get_env_enable_pdl()
        )


class LamportPending:
    """A pushed all-reduce whose finish is left to its consumer
    (``finish_lamport_push(..., deferred=True)``).

    Every rank has stored its partial into the slots of the call (one-shot or two-shot transport),
    but no kernel has reduced them or advanced the workspace's rotation state: the consumer's kernel
    does both (``k3_attn_res`` with ``lamport_finish``), or ``materialize`` runs the finish kernel
    into a tensor. Exactly one of the two must run, before the next all-reduce on the workspace.
    """

    def __init__(self, target: LamportPushTarget):
        self.target = target
        self.workspace = target.workspace
        self.shape = tuple(target.output.shape)

    @property
    def finish_args(self) -> Tuple[int, int, bool, int, int]:
        """``(world, chunk, two_shot, mc_base, rank)`` of ``k3_attn_res``'s ``lamport_finish``: the
        ranks per partial sum of the C++ kernel the result reproduces, the transport and the
        workspace addresses."""
        from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

        ws = self.workspace
        return (
            ws.world,
            k3_lamport_ar.reduce_chunk(ws.world, self.target.cpp_one_shot),
            not self.target.one_shot,
            ws.mc_base,
            ws.rank,
        )

    @property
    def finish_ag(self) -> bool:
        """Whether the call carries all-gather rows (``finish_ag`` of ``k3_attn_res``)."""
        return getattr(self.target, "ag", False)

    def materialize(self) -> torch.Tensor:
        """Finish the all-reduce into a BF16 tensor (the finish kernel on the current stream)."""
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

        from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

        return k3_lamport_ar.run_lamport_finish(
            self.workspace,
            self.target.output,
            self.target.one_shot,
            use_pdl=get_env_enable_pdl(),
            cpp_one_shot=self.target.cpp_one_shot,
            ag=self.finish_ag,
        )


def _lamport_view(mnnvl: MNNVLAllReduce, workspace: dict):
    """The ``k3_lamport_ar.LamportWorkspace`` of an MNNVL workspace, built once per buffer."""
    key = workspace["uc_buffer"].data_ptr()
    view = _LAMPORT_VIEWS.get(key)
    if view is None:
        from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

        view = k3_lamport_ar.LamportWorkspace.from_mnnvl(
            workspace, mnnvl.mapping.tp_size, mnnvl.mapping.tp_rank
        )
        _LAMPORT_VIEWS[key] = view
        logger.info_once(
            f"[MNNVL AllReduce] ar_push: Lamport workspace view "
            f"for tp_size {mnnvl.mapping.tp_size} "
            f"({view.bytes_per_buffer} bytes per buffer)",
            key=("k3_ar_push_view", mnnvl.mapping.tp_size),
        )
    return view


def prepare_lamport_view(all_reduce: AllReduce) -> None:
    """Build the device view of ``all_reduce``'s MNNVL workspace now, outside graph capture, so
    that a capture finds it. No-op without ``ar_push`` or an MNNVL BF16 path."""
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if not _K3_AR_PUSH or mnnvl is None or mnnvl.dtype != torch.bfloat16:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    _lamport_view(mnnvl, get_or_scale_allreduce_mnnvl_workspace(mnnvl.mapping, mnnvl.dtype))


def mnnvl_lamport_push_target(
    mnnvl: MNNVLAllReduce,
    num_tokens: int,
    hidden_dim: int,
    in_place: bool = False,
    one_shot: Optional[bool] = None,
) -> Optional[LamportPushTarget]:
    """A ``LamportPushTarget`` for a plain all-reduce of BF16 ``[num_tokens, hidden_dim]`` on the
    workspace of ``mnnvl``, or None. ``in_place``: the caller finishes it in place
    (``finish_lamport_push(in_place=True)``), which takes the two-shot transport at every size.
    ``one_shot`` (not with ``in_place``) forces the transport instead of the size rule. Not gated
    by ``ar_push``: the caller decides.

    The first call per workspace builds its device peer table, so during CUDA graph capture it
    returns None for a workspace without one (or one that would grow); the eager warm-up forwards
    make it.
    """
    from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

    world = mnnvl.mapping.tp_size
    nbytes = num_tokens * hidden_dim * world * 2
    cpp_one_shot = nbytes <= _MNNVL_ONE_SHOT_THRESHOLD_BYTES
    if one_shot is not None and in_place and one_shot:
        raise ValueError("the in-place finish needs the two-shot transport")
    if one_shot is None:
        one_shot = not in_place and nbytes <= _K3_AR_PUSH_ONE_SHOT_MAX_BYTES
    need = max(
        mnnvl.get_required_workspace_size(num_tokens, hidden_dim, world, mnnvl.dtype),
        k3_lamport_ar.required_bytes(num_tokens, hidden_dim, world, one_shot),
    )
    if need >= 2**32 - 1:
        return None
    if torch.cuda.is_current_stream_capturing():
        # Creating or growing the shared workspace and building the view (a host-to-device copy)
        # must both have happened in the eager warm-up of this shape.
        current = MNNVLAllReduce.allreduce_mnnvl_workspaces.get(mnnvl.mapping)
        if (
            current is None
            or current["buffer_size_bytes"] < need
            or current["uc_buffer"].data_ptr() not in _LAMPORT_VIEWS
        ):
            return None
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mnnvl.mapping, mnnvl.dtype, buffer_size_bytes=need
    )
    view = _lamport_view(mnnvl, workspace)
    output = torch.empty(
        num_tokens, hidden_dim, dtype=mnnvl.dtype, device=workspace["uc_buffer"].device
    )
    return LamportPushTarget(view, one_shot, cpp_one_shot, output)


def mnnvl_finish_lamport_push(
    target: LamportPushTarget,
    in_place: bool = False,
    deferred: bool = False,
) -> Union[torch.Tensor, LamportResult, LamportPending]:
    """Complete a pushed all-reduce; returns ``target.output``, or with ``in_place`` (two-shot
    transport only) a ``LamportResult``, or with ``deferred`` a ``LamportPending`` whose consumer
    finishes it (no kernel is launched here)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

    if deferred:
        return LamportPending(target)
    if in_place and not target.one_shot:
        k3_lamport_ar.run_lamport_finish_in_place(
            target.workspace,
            target.output.shape[0],
            target.output.shape[1],
            use_pdl=get_env_enable_pdl(),
            cpp_one_shot=target.cpp_one_shot,
        )
        return LamportResult(target)
    return k3_lamport_ar.run_lamport_finish(
        target.workspace,
        target.output,
        target.one_shot,
        use_pdl=get_env_enable_pdl(),
        cpp_one_shot=target.cpp_one_shot,
    )


def lamport_push_target(
    all_reduce: AllReduce,
    num_tokens: int,
    hidden_dim: int,
    in_place: bool = False,
    deferred: bool = False,
    force: bool = False,
) -> Optional[LamportPushTarget]:
    """With ``ar_push`` (or ``force``): a ``LamportPushTarget`` if ``all_reduce.forward`` would
    run the MNNVL all-reduce on a BF16 ``[num_tokens, hidden_dim]`` input; else None. For a
    plain-sum all-reduce only: callers never pass one with a fusion, and this does not check.

    A producer that stores its output into the target (and sets ``pushed``) replaces the input of
    this all-reduce; :func:`finish_lamport_push` then returns the same result ``forward`` would.
    Targets must be finished in the order they were pushed, by every rank. ``in_place`` requests
    the two-shot transport, which ``finish_lamport_push(in_place=True)`` needs. ``deferred`` sets
    ``target.deferred`` for ``finish_lamport_push(deferred=True)``, whose consumer finishes the
    call; it takes the one-shot transport where the call is at most
    ``_K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES`` (``num_tokens * hidden_dim * world * 2``), the
    two-shot one above.
    """
    mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
    if (
        not (_K3_AR_PUSH or force)
        or mnnvl is None
        or all_reduce.mapping.tp_size == 1
        or getattr(all_reduce, "symm_mem_allreduce", None) is not None
        or torch.compiler.is_compiling()
    ):
        return None
    if mnnvl.dtype != torch.bfloat16 or num_tokens <= 0 or hidden_dim % 8:
        return None
    if not deferred:
        return mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim, in_place)
    one_shot = (
        num_tokens * hidden_dim * all_reduce.mapping.tp_size * 2
        <= _K3_FUSED_FINISH_ONE_SHOT_MAX_BYTES
    )
    target = mnnvl_lamport_push_target(mnnvl, num_tokens, hidden_dim, one_shot=one_shot)
    if target is not None:
        target.deferred = True
    return target


def lamport_push_tensor(target: LamportPushTarget, x: torch.Tensor) -> None:
    """Push the contiguous BF16 ``[num_tokens, hidden_dim]`` tensor ``x`` into ``target`` (a copy
    kernel), for a producer without the push epilogue."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    from ...kernels.comm import rubin_ull_k3_lamport_ar as k3_lamport_ar

    k3_lamport_ar.run_lamport_push(
        target.workspace, x.contiguous(), target.one_shot, use_pdl=get_env_enable_pdl()
    )
    target.pushed = True


def finish_lamport_push(
    target: LamportPushTarget,
    in_place: bool = False,
    deferred: bool = False,
) -> Union[torch.Tensor, LamportResult, LamportPending]:
    """The all-reduce result of a target the producer has pushed: a tensor, or with ``in_place``
    (where the two-shot transport ran) a ``LamportResult`` that the next kernel reads from the
    workspace, or with ``deferred`` (where ``target.deferred``) a ``LamportPending`` that the next
    kernel finishes."""
    if not target.pushed:
        raise RuntimeError("the producer did not push this all-reduce")
    return mnnvl_finish_lamport_push(target, in_place, deferred and target.deferred)


__all__ = [
    "LamportPending",
    "LamportPushTarget",
    "LamportResult",
    "finish_lamport_push",
    "lamport_push_target",
    "lamport_push_tensor",
    "mnnvl_finish_lamport_push",
    "mnnvl_lamport_push_target",
    "prepare_lamport_view",
    "use_fabric_handle",
]
