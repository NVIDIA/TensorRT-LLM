# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot MNNVL all-reduce with Kimi K3's residual update (attention-residual selection + RMSNorm) as its
epilogue, over a caller-owned :class:`MnnvlWorkspace`."""

from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .mnnvl_workspace import MnnvlWorkspace

native_ops = load_family_ops(__name__)


class MnnvlAllreduceAttnRes(OpWrapper):
    """Return ``(normed, updated)``: ``updated = prefix_sum + allreduce(input)`` over ``workspace``'s TP group (the
    sum alone without ``prefix_sum``), ``normed`` = RMSNorm(attn_res(block_residual..., updated)). Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=(
                f"{snapshots} saved snapshots, "
                f"{'with' if prefix else 'without'} a prefix: one-shot residual selection"
            ),
            spec=dict(
                hidden=7168,
                tokens=(1, 2, 3, 4, 5, 6, 7, 8, 16),
                snapshots=snapshots,
                prefix=prefix,
                dtype=torch.bfloat16,
                rms_eps=1e-6,
                output_rms_eps=1e-6,
            ),
        )
        for snapshots in (0, 1, 5, 8, 11)
        for prefix in (True, False)
    )

    note = """
    The all-reduce, residual add, attention-residual selection and output
    RMSNorm are fused. Producing the row-parallel input, maintaining the
    snapshot bank and chaining prefix_sum remain the caller's work.
    The kernel rounds the all-reduce and updated prefix to bf16 and follows
    the unfused attn_res_add_rmsnorm_fwd rounding. The reference keeps the
    sum, selection and normalization in fp32 and rounds each output once.
    As the collective exception to mirroring __call__, reference takes
    every rank's input in rank order.
    Inputs, prefix and snapshots are read-only; outputs are fresh contiguous
    bf16 tensors. Fixed reduction order gives bitwise equality across ranks.

    MnnvlWorkspace is caller-owned state, one per TP group, not an entry:
    it launches nothing per call. It owns an McastGPUBuffer allocation,
    its rank-local unicast lamport view, the handle-exchange communicator,
    and uint32[9] buffer_flags: current buffer, dirty buffer, bytes per
    buffer, dirty stage count, four dirty byte counts and arrival count.
    Three buffers of buffer_bytes share one multicast mapping; each word
    is fp32 -0.0 when armed. A call writes T*H*W*2 bytes in one buffer.
    K3 uses 4 MiB: at H=7168 max_one_shot_tokens is 73 for W=4 and 18 for
    W=16. Capacity is fixed; oversized calls raise before touching state.

    The target creates the workspace eagerly in post_load_weights using
    MnnvlWorkspace.create(mapping, buffer_bytes, fabric_handle=None),
    before graph capture. Creation is collective only over that TP group;
    other PP groups need not participate. Before allocating, ranks agree
    they are outside capture, sizes are valid and three buffers fit free
    device memory. If any cannot allocate, all raise RuntimeError, none
    allocates, and MPI ranks free the communicator made for that attempt.
    Allocation failures that return are agreed and handled the same way.
    A failure inside handle exchange can instead leave peers waiting;
    it is not converted to an error on every rank. Creation refuses capture
    and arms every word and flag, waiting for every rank to arm before any
    returns. Fabric handles are required across nodes; otherwise POSIX file
    descriptors may be used. The default is mapping.is_multi_node(), with
    no environment-variable override.

    All MNNVL entries taking the same comm_buffer/buffer_flags share one
    rotation, including mnnvl_fusion_allreduce on either path and
    mnnvl_allgather_split. Every call takes the current buffer, advances
    current modulo three and leaves its buffer dirty with its written byte
    counts. The next call reads those flags after the previous grid and
    clears the previous call's stages by the previous sizes and layout,
    not its own size. Its current buffer must contain only empty words
    and this call's pushes. A smaller call must not leave older larger
    calls' words alive when the buffer comes round again.

    Every rank must make the same sequence of calls, with the same op,
    T, S and prefix presence at each position, across layers, decode steps,
    eager calls and graph replays. A call waits for the corresponding
    peers' rows. Different workspace objects have independent rotations,
    not independent ordering: all collectives on one stream must be
    issued in the same order across ranks. B-then-A on one rank against
    A-then-B on peers deadlocked at W=4. These distributed invariants
    cannot be checked by rank-local is_valid. Two swapped same-shaped
    calls on one workspace instead silently pair with peers' calls at
    those positions; every rank's results are wrong, and the next call is
    correct after the pair realigns. Extra or missing calls were not tested.

    The matrix retains 12 decode steps of 12 chained layers at
    T=8,8,8,2,7,8,1,1,8,16,3,8, with a random rank 5 ms late on each call.
    A re-arm sized by the current call passes single-call tests but fails
    this dip/regrow sequence, causing rank disagreement and serving hangs.
    It also checks 20 irregularly interleaved calls on two workspaces,
    a captured 12-layer T=8 step replayed eight times with fresh inputs and
    eager calls of other sizes between replays, wrong-order negatives,
    TP-group-only creation and a one-rank memory refusal with communicator
    cleanup and a correct subsequent call. All S=0..11 are exercised by
    chained layers; the standalone cells cover S=0,1,5,8,11 and both
    prefix-presence cases.

    Dtype, contiguous layout, shapes, one-buffer capacity and the kernel's
    H multiple of 1024 up to 8192, W in 2,4,8,16 and S<=11 restrictions
    already fail loudly. No process cache or compilation is involved and
    no metadata beyond workspace is consumed. The raw op looks up the
    multicast mapping by comm_buffer's data pointer; the owned handle
    keeps that registry entry alive. Unlike MNNVLAllReduce's mapping-keyed,
    demand-grown cache, this workspace's size is chosen at construction.

    Certification is SM100, H=7168, four ranks on one GB200 tray using
    POSIX handles. Multiples of 1/16 make updated exact for bitwise checks;
    normed is bounded against fp32 by 2e-2 of its largest magnitude, and
    all outputs must agree bitwise across ranks. The matrix accepts mpirun
    or srun with configurable world size. A separate W=16 fabric-handle
    run covered 110 cases at T=1..8,16,32,64, S=0,1,3,8,11 and both
    prefix cases against the unfused op; this matrix was not run at W=16.
    That kernel path adds ranks in two chunks of eight rather than one.
    """

    def __call__(
        self,
        input: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        workspace: MnnvlWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed, updated = native_ops.mnnvl_allreduce_attn_res(
            input,
            prefix_sum,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
            workspace.comm_buffer(input.dtype),
            workspace.buffer_flags,
        )
        return normed, updated

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        prefix_sum: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        updated = sum(x.float() for x in inputs_in_rank_order)
        if prefix_sum is not None:
            updated = updated + prefix_sum.float()
        v = torch.cat([block_residual.float(), updated.unsqueeze(0)], dim=0)
        rs = (v.square().mean(dim=-1) + rms_eps).rsqrt()
        logits = (v * rs[..., None] * (rms_weight.float() * res_weight.float())).sum(dim=-1)
        probs = torch.softmax(logits, dim=0)
        mixed = (probs[..., None] * v).sum(dim=0)
        out = mixed * (mixed.square().mean(dim=-1, keepdim=True) + output_rms_eps).rsqrt()
        dtype = inputs_in_rank_order[0].dtype
        return (out * output_rms_weight.float()).to(dtype), updated.to(dtype)


mnnvl_allreduce_attn_res = MnnvlAllreduceAttnRes()
