# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL Lamport workspaces of the target's hops and exchanges.

The hops (``mnnvl_hop``, ``mnnvl_moe_finalize_hop``) run on the MNNVL all-reduce workspace of the
engine's ``AllReduce`` (``get_or_scale_allreduce_mnnvl_workspace``): three Lamport buffers whose
empty value is -0.0, and nine uint32 flags (current buffer, dirty buffer, bytes per buffer, dirty
stages, bytes to clear per stage, arrival counter). The hops read, clear and rotate them as the
built-in MNNVL all-reduce does, so both can follow each other on one workspace.

The workspace is replaced by a larger one when an all-reduce needs more than it holds, and a CUDA
graph keeps the workspace it launches on alive only when it was fetched during that graph's
capture. So the target fetches it for every hop it issues, eagerly and under capture, and never
keeps it across forwards.

The exchanges that are not all-reduces (``mnnvl_allgather``, the draft steps' argmax exchanges)
each own a ``LamportExchangeWorkspace``, built once and kept.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.distributed.ops import (
    _get_mnnvl_workspace_comm,
    _make_mnnvl_mcast_buffer,
    _mnnvl_device_index,
    _mnnvl_workspace_all_succeeded,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm.mapping import Mapping

# Lamport buffers of an MNNVL all-reduce workspace.
NUM_LAMPORT_BUFFERS = 3


def hop_buffer_bytes(num_tokens: int, hidden_size: int, tp_size: int) -> int:
    """Bytes of one Lamport buffer that a hop of ``[num_tokens, hidden_size]`` BF16 rows uses."""
    return num_tokens * hidden_size * tp_size * torch.bfloat16.itemsize


def hop_supported(num_tokens: int, hidden_size: int, tp_size: int) -> bool:
    """Whether the hops serve ``[num_tokens, hidden_size]`` BF16 rows over ``tp_size`` ranks.

    Asked of the current device: 2, 4, 8 or 16 ranks; one-shot (up to 1 MiB of Lamport data per
    rank, 1.5 MiB on 16 ranks); one row per cluster of 2 to 8 CTAs of at most 128 threads, eight
    elements per thread, covering the row exactly. Hidden 6144 on 16 ranks: 1 to 8 rows. Another
    shape runs on the built-in ``AllReduce``.
    """
    return bool(torch.ops.trtllm.glm_mnnvl_hop_supported(num_tokens, hidden_size, tp_size))


def _all_reduce_workspace(mapping: Mapping, num_tokens: int, hidden_size: int) -> dict:
    return get_or_scale_allreduce_mnnvl_workspace(
        mapping,
        torch.bfloat16,
        buffer_size_bytes=hop_buffer_bytes(num_tokens, hidden_size, mapping.tp_size),
    )


def hop_workspace(
    mapping: Mapping, num_tokens: int, hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The workspace operands of a hop of ``[num_tokens, hidden_size]`` BF16 rows.

    Returns ``(comm_buffer, buffer_flags)``: the ``[3, N]`` BF16 view of this rank's three Lamport
    buffers of the MNNVL all-reduce workspace of ``mapping``, and the workspace's nine uint32
    flags. The workspace grows first when its buffers hold less than the hop needs; it must not
    grow while a CUDA graph is captured, so every shape runs once eagerly before its capture.
    """
    workspace = _all_reduce_workspace(mapping, num_tokens, hidden_size)
    comm_buffer = workspace["uc_buffer"].view(torch.bfloat16).view(NUM_LAMPORT_BUFFERS, -1)
    return comm_buffer, workspace["buffer_flags"]


def publish_target(
    mapping: Mapping, num_tokens: int, hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Where the GEMM that produces a hop's input stores this rank's rows itself.

    Returns ``(lamport, flags)``: the multicast BF16 view (1-D) of the three Lamport buffers of the
    workspace ``hop_workspace`` returns for the same shape, and its flags viewed as int32. The GEMM
    (the o_proj Lamport publish) reads the current buffer index and the bytes per buffer from the
    flags after its grid dependency wait, releases its dependents right after that wait, and stores
    its BF16 output, -0.0 written as +0.0, in whole 16-byte vectors at ``[token][tp_rank][column]``
    of the current buffer. The hop that follows it on the stream then runs with
    ``shard_published=True`` on that workspace.
    """
    workspace = _all_reduce_workspace(mapping, num_tokens, hidden_size)
    elements = NUM_LAMPORT_BUFFERS * workspace["buffer_size_bytes"] // torch.bfloat16.itemsize
    lamport = workspace["handle"].get_mc_buffer((elements,), torch.bfloat16, 0)
    return lamport, workspace["buffer_flags"].view(torch.int32)


class LamportExchangeWorkspace:
    """A private MNNVL Lamport workspace of the TP group, for an exchange that is not an all-reduce.

    Three Lamport buffers whose unwritten 32-bit words hold 0xFFFFFFFF, so that every payload,
    -0.0 included, travels unchanged, and their nine flags in the layout of the all-reduce
    workspace's. Used by ``mnnvl_allgather`` (the verify logits) and by the draft-step kernels
    that exchange each rank's argmax (index, value) packet, with the token-table row and the
    hidden-half share where the next eh_proj comes from the token table (``draft_sample_glue``,
    ``draft_ehproj_table``). Each of those kernels empties the buffer of its previous call with
    its own threads, so every user owns its workspace, sized for its largest call.

    Construction is collective over the TP group: every rank constructs the same workspaces in the
    same order, outside CUDA graph capture (the target does it once, in ``post_load_weights``),
    and a failure on any rank raises on every rank.

    Args:
        mapping: The tensor-parallel mapping of the group.
        buffer_size_bytes: Bytes of one Lamport buffer, a positive multiple of 16.

    Attributes:
        comm_buffer: This rank's unicast view of the three buffers as 1-D int32, the kernels'
            ``comm_buffer`` (``exchange_buffer``) operand.
        buffer_flags: The nine uint32 Lamport flags, the kernels' ``buffer_flags``
            (``exchange_flags``) operand.
    """

    def __init__(self, mapping: Mapping, buffer_size_bytes: int):
        if buffer_size_bytes <= 0 or buffer_size_bytes % 16:
            raise ValueError(
                f"buffer_size_bytes must be a positive multiple of 16: {buffer_size_bytes}"
            )
        comm = _get_mnnvl_workspace_comm(mapping)
        workspace_bytes = NUM_LAMPORT_BUFFERS * buffer_size_bytes
        state = None
        error: Optional[Exception] = None
        try:
            handle = _make_mnnvl_mcast_buffer(
                comm, workspace_bytes, mapping, mapping.is_multi_node()
            )
            comm_buffer = handle.get_uc_buffer(
                mapping.tp_rank, (workspace_bytes // 4,), torch.int32, 0
            )
            # Filled under inference mode, like the all-reduce workspace: the engine may build
            # it while CUDA graph warm-up runs in inference mode.
            with torch.inference_mode():
                comm_buffer.fill_(-1)
                buffer_flags = torch.tensor(
                    [0, 2, buffer_size_bytes, 0, 0, 0, 0, 0, 0],
                    dtype=torch.uint32,
                    device=torch.device("cuda", _mnnvl_device_index(mapping)),
                )
            torch.cuda.synchronize()
            state = (handle, comm_buffer, buffer_flags)
        except Exception as e:
            error = e
        # Also fences the fill: no rank stores into a peer's buffer before the peer filled it.
        if not _mnnvl_workspace_all_succeeded(comm, error is None):
            raise RuntimeError(
                "MNNVL Lamport exchange workspace construction failed on at least one rank"
            ) from error
        assert state is not None
        # The handle keeps the multicast allocation mapped; the communicator stays with it.
        self._handle, self.comm_buffer, self.buffer_flags = state
        self._comm = comm
        self.mapping = mapping
        self.buffer_size_bytes = buffer_size_bytes
