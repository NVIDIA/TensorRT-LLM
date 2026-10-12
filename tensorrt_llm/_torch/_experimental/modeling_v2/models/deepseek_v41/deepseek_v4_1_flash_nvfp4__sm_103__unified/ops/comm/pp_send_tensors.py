# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send tensors to the next pipeline-parallel stage (the matching rank of the next stage).

The op is `torch.ops.trtllm.pp_send_tensors`, registered in
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
#: (form, world size, TP size per stage, nodes). Stage 0 rank r sends to rank
#: r + TP. On two trays stage 0 is one 4-GPU node and stage 1 the other.
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


class PpSendTensors(OpWrapper):
    """Send each tensor in `tensors`, in order, to `mapping.next_pp_rank()`.

    The receiving rank posts `pp_recv_tensors` with tensors of the same shapes
    and dtypes in the same order, and each is filled with the sender's bytes.
    Nothing is computed. Fusion boundary: the transfer only; allocating the
    receive buffers and choosing what to send are the caller's.

    Returns None: the call writes nothing on this rank. Its result is the
    receiver's tensors, which `reference` and `compare` describe.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Form x layout x mode. The tensors are always the target's three. The
    # attention-DP layout shares the attention-TP layout's communicator through
    # `init_pp_comm`'s reuse of a live one of the same world size.
    CELLS: tuple[Cell, ...] = _handover_cells()

    note = """
    Metadata consumed: none. The destination is `mapping.next_pp_rank()` of the
    mapping `init_pp_comm` was last called with; the engine's PP communicator
    must exist (`init_pp_comm(mapping)`, which the model engine calls when the
    mapping has more than one stage). The certified communicator is
    `PPCommNCCL`, one world-size NCCL communicator (the MPI path).

    Eagerly each tensor is copied and the copy sent on the communicator's own
    stream after the current stream's pending work, so the caller may
    overwrite or free its tensors right after the call: with the receiver held
    back ~20 ms while the sender overwrote, dropped and reallocated its
    tensors, every delivery was the bytes at the time of the call. Under
    CUDA-graph capture the send is recorded on the capturing stream with no
    copy: a replay sends the static tensors' bytes as they are at replay time,
    and later writes to them on that stream do not reach the receiver. An
    eager send followed at once by a replay that sends arrives in call order.

    Preconditions: the receiver posts the same count, shapes and dtypes in the
    same order (a mismatch was not probed); every tensor contiguous and on a
    CUDA device -- a captured send of a strided view sends the first
    `numel x itemsize` bytes of its storage, silently; under CUDA graphs both
    stages capture and replay the same step.

    Measured on sm_103: about 0.25 ms per hand-over at decode sizes and 1.3 ms
    at T = 16384, with or without a tray boundary; T x 41,060 bytes per step.

    Not certified: T = 0 or T > 16384; any other tensor list; more than two
    stages, other TP widths per stage, context parallelism; the `PPCommTorch`
    path (`init_pp_comm` with MPI disabled); two nodes not in one NVLink
    domain; a sender and receiver that disagree; a captured send replayed
    without its captured receive; other collectives on the same GPUs during a
    hand-over; CPU tensors; a receiver held back longer than ~20 ms; more than
    8 eager sends in flight.

    The two-tray cells span two 4-GPU nodes of one NVL72 rack and cannot be
    collected on one node. Run them by hand from this target's comm test
    directory under tests/unittest/_torch/modeling_v2/models/deepseek_v41/:
      MODELING_V2_TWO_NODE_MPIRUN="<prefix>" python _pp_send_recv_op_matrix.py --two-node
    where <prefix> starts 8 ranks of the command after it, ranks 0-3 on one
    node and 4-7 on the other, with PYTHONPATH at the repo root (for example
    `mpirun -np 8 --host <node 0>:4,<node 1>:4 -x PYTHONPATH`). It must print
    PP_SEND_RECV_OK. The same run certifies `pp_recv_tensors`.
    """

    def raw_call(self, tensors: List[torch.Tensor]) -> None:
        torch.ops.trtllm.pp_send_tensors(tensors)

    def reference(self, tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """What the receiving rank's tensors must hold: the sender's bytes.

        The documented exception for a collective: a send's result is on
        another rank, so this states what the matching `pp_recv_tensors` must
        leave there, given the tensors this rank sends (copies, in order).
        """
        return [t.clone() for t in tensors]

    def compare(self, out: Sequence[torch.Tensor], ref: Sequence[torch.Tensor]) -> None:
        """Bitwise per tensor, on the receiver's tensors.

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
            # A strided view is sent as its packed storage bytes, not as the view.
            assert t.is_contiguous(), "every tensor must be contiguous"
            assert t.is_cuda, "the PP communicator moves device tensors only"


pp_send_tensors = PpSendTensors()
