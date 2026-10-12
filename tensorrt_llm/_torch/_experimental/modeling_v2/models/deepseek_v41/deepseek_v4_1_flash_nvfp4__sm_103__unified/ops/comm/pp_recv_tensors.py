# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receive tensors from the previous pipeline-parallel stage, in place.

The op is `torch.ops.trtllm.pp_recv_tensors`, registered in
`tensorrt_llm/_torch/distributed/communicator.py`.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

import tensorrt_llm._torch.distributed.communicator  # noqa: F401 — registers torch.ops.trtllm.pp_*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The DeepSeek-V4.1 target's stage hand-over, in order, as (name, trailing
#: dims, dtype) of a [T, ...] tensor: the hyper-connection stream h, the last
#: FFN mixes yF and their RMS statistic rF.
_HANDOVER = (
    ("h", (4, 5120), torch.bfloat16),
    ("yF", (24,), torch.float32),
    ("rF", (), torch.float32),
)
#: Eager row counts: every T up to max_num_tokens per rank.
_EAGER_ROWS = range(1, 16384 + 1)
#: The 46 CUDA-graph batch sizes (every G in 1..32 plus the padded default list
#: up to 256); a captured step hands over T = G (DSpark off) or 6 G (DSpark
#: verify), 82 values in 1..1536.
_GRAPH_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_GRAPH_ROWS = tuple(sorted(set(_GRAPH_BATCHES) | {6 * g for g in _GRAPH_BATCHES}))
#: (form, world size, TP size per stage, nodes). Stage 1 rank r receives from
#: rank r - TP. On two trays stage 0 is one 4-GPU node and stage 1 the other.
_FORMS = (("one node", 4, 2, 1), ("two trays", 8, 4, 2))


def _handover_cells() -> tuple[Cell, ...]:
    cells = []
    for form, world_size, tp, nodes in _FORMS:
        base = dict(world_size=world_size, tp_size=tp, pp_size=2, nodes=nodes, tensors=_HANDOVER)
        cells += [
            Cell(
                why=f"{form} pp2 x tp{tp} eager: every T 1..16384, all pairs at the same T",
                spec=dict(base, attention_dp=False, mode="eager", rows=_EAGER_ROWS),
            ),
            Cell(
                why=f"{form} pp2 x dep{tp} eager: every T 1..16384, the pairs at different T",
                spec=dict(base, attention_dp=True, mode="eager", rows=_EAGER_ROWS),
            ),
            Cell(
                why=f"{form} pp2 x tp{tp} graph: the 82 captured T, replayed in any order",
                spec=dict(base, attention_dp=False, mode="graph", rows=_GRAPH_ROWS),
            ),
        ]
    return tuple(cells)


class PpRecvTensors(OpWrapper):
    """Fill each tensor in `tensors`, in order, from `mapping.prev_pp_rank()`.

    The counterpart of `pp_send_tensors`: the sender's tensors have the same
    shapes and dtypes in the same order, and each receiving tensor is filled in
    place with the sender's bytes. Nothing is computed. Fusion boundary: the
    transfer only; allocating `tensors` is the caller's.

    Returns `tensors`, the list it filled (the same list object, the same
    tensors).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The same cells as `pp_send_tensors`: one hand-over certifies both halves.
    CELLS: tuple[Cell, ...] = _handover_cells()

    note = """
    Metadata consumed: none. The source is `mapping.prev_pp_rank()` of the
    mapping `init_pp_comm` was last called with; the engine's PP communicator
    (`PPCommNCCL`, the MPI path) must exist.

    The receive is enqueued on the current stream, eagerly and under CUDA-graph
    capture, so work after it on that stream reads the received bytes. A
    receive into a strided view writes the packed bytes into the first
    `numel x itemsize` bytes of the view's storage and leaves the view's own
    elements unwritten, silently.

    The rest of the hand-over contract -- what the eager and captured sends
    guarantee, the preconditions, the measured cost, what is not certified and
    the hand-run two-tray command -- is `pp_send_tensors`' note: neither half
    runs without the other, and one matrix certifies both.
    """

    def raw_call(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        torch.ops.trtllm.pp_recv_tensors(tensors)
        return tensors

    def reference(self, sent: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """What `tensors` must hold after the call: the sender's bytes.

        The documented exception for a collective: what a receive returns is
        not a function of the receiving buffers, so this takes the tensors the
        previous stage's `pp_send_tensors` sent, in order, and returns copies.
        """
        return [t.clone() for t in sent]

    def compare(self, out: Sequence[torch.Tensor], ref: Sequence[torch.Tensor]) -> None:
        """Bitwise per tensor.

        The op moves bytes and computes nothing, so any differing byte is a
        wrong delivery. The matrix's payloads differ in every 32-bit word from
        a missed receive (a 0xFF-filled buffer), the previous step's bytes and
        the other sender's bytes, and the gate fails each of them.
        """
        assert len(out) == len(ref), (len(out), len(ref))
        for k, (got, want) in enumerate(zip(out, ref)):
            assert got.dtype == want.dtype, (k, got.dtype, want.dtype)
            assert got.shape == want.shape, (k, got.shape, want.shape)
            bad = (got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).sum()
            assert bad.item() == 0, f"tensor {k}: {bad.item()} of {want.nbytes} bytes differ"

    def is_valid(self, tensors: List[torch.Tensor]) -> None:
        for t in tensors:
            assert t.is_contiguous(), (
                "every tensor must be contiguous (it is filled as packed bytes)"
            )
            assert t.is_cuda, "the PP communicator moves device tensors only"


pp_recv_tensors = PpRecvTensors()
