# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL all-reduce over a caller-owned :class:`MnnvlWorkspace`: the sum over the TP group, or with a residual the
sum + residual add + RMSNorm; one-shot up to ``one_shot_max_bytes``, two-shot above."""

from typing import Optional, Sequence, Tuple, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.functional import AllReduceFusionOp

from .mnnvl_workspace import MnnvlWorkspace

native_ops = load_family_ops(__name__)

__all__ = ["MnnvlWorkspace", "mnnvl_fusion_allreduce", "required_buffer_bytes"]


def required_buffer_bytes(
    num_tokens: int, hidden: int, world_size: int, dtype: torch.dtype, one_shot_max_bytes: int
) -> int:
    """Bytes of one Lamport buffer a call of ``num_tokens`` rows of ``hidden`` needs: ``num_tokens * hidden * world *
    element size`` when that is at most ``one_shot_max_bytes`` (one-shot), else two stages of ``num_tokens`` rounded up
    to a multiple of ``world`` (two-shot)."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    one_shot = num_tokens * hidden * world_size * itemsize
    if one_shot <= one_shot_max_bytes:
        return one_shot
    return 2 * -(-num_tokens // world_size) * world_size * hidden * itemsize


class MnnvlFusionAllreduce(OpWrapper):
    """The sum of ``input`` over ``workspace``'s TP group (a new tensor of ``input``'s shape), or with ``residual``,
    ``norm_weight`` and ``eps`` the pair ``(RMSNorm(sum + residual) * norm_weight, sum + residual)``. Sent one-shot
    when ``num_tokens * hidden * world * element size <= one_shot_max_bytes``, else two-shot; the workspace's buffers
    must hold the call (:func:`required_buffer_bytes`). Advances ``workspace`` by one call: every rank of the group
    makes the same MNNVL calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=(
                f"K3 hidden {hidden}, {'residual RMSNorm' if fused else 'plain sum'}: "
                "inclusive one-shot boundary and one byte below it"
            ),
            spec=dict(
                hidden=hidden,
                tokens=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64),
                fused=fused,
                paths=("one", "two"),
                dtype=torch.bfloat16,
                eps=1e-5,
            ),
        )
        for hidden in (3584, 7168)
        for fused in (False, True)
    ) + (
        Cell(
            why="65 tokens fit two-shot above two ranks after exceeding the one-shot buffer",
            spec=dict(
                hidden=7168,
                tokens=(65,),
                fused=False,
                paths=("two",),
                dtype=torch.bfloat16,
                eps=1e-5,
                min_world=4,
            ),
        ),
    )

    note = """
    Only plain sum and RESIDUAL_RMS_NORM are exposed; quantizing FP8/NVFP4
    fusions and scale are inert. Inputs, residual and norm_weight are
    read-only; outputs are fresh contiguous tensors on the input device.
    An input -0.0 travels as +0.0 because -0.0 is the Lamport empty word.
    The kernel rounds sums and updated to bf16, rounds each RMSNorm square
    to bf16, accumulates squares in fp32, and computes normalization products
    in fp32. The reference accumulates in fp32 and rounds each output once.
    As the collective exception to mirroring __call__, reference takes
    every rank's input in rank order.

    One-shot is selected inclusively at T*H*W*element_size <=
    one_shot_max_bytes. One kernel multicasts each rank's rows, waits for
    all ranks, sums and optionally adds the residual and normalizes.
    Two-shot scatters token t to rank t mod W, sums that rank's tokens and
    broadcasts bf16 sums. Plain output is copied by the same kernel;
    residual RMSNorm uses a second kernel. Up to eight ranks the paths
    sum in the same ascending fp32 order. At W=16 one-shot adds two chunks
    of eight while two-shot adds sequentially, so inexact sums can differ
    in their last bit. Norm products are x*rcp*g versus g*x*rcp, and may
    differ even below W=16. Every rank's output is bitwise identical.

    The caller owns the fixed-size MnnvlWorkspace; construction, ownership
    and failures are specified by mnnvl_allreduce_attn_res.note. Create
    eagerly in post_load_weights, collectively over the TP group, before
    capture, with all buffers armed before any rank returns. Returned
    allocation failures are agreed; failure inside handle exchange can
    strand peers. Tests retain mixed/all-rank capture refusals followed by
    a successful eager create without damaging live workspaces.

    Three Lamport buffers share one multicast mapping. One-shot needs
    T*H*W*element_size bytes per buffer. Two-shot needs
    2*ceil(T/W)*W*H*element_size: scatter writes the first stage and
    broadcast writes T*H*element_size bytes starting at buffer_bytes/2.
    The wrapper rejects an oversized call before mutation; the raw op
    does not guard capacity and can write past the allocation. A two-shot
    workspace must be a multiple of 32 bytes for 16-byte-vector alignment,
    although create accepts multiples of 16; the wrapper rejects this
    loudly. A 4 MiB buffer holds K3's calls up to 64 tokens and H=7168,
    whose two-shot need is at most 1.75 MiB. The matrix's largest one-shot
    call needs 3.5 MiB at W=4. At W>2 the 65-token rejection fits two-shot.

    All MNNVL entries on one workspace share one rotation. After each call
    uint32[9] flags record current+1 modulo three, this call's dirty buffer,
    unchanged capacity, one or two stages, dirty byte counts and zero
    arrivals. Byte counts are (T*H*W*element_size,0,0,0) one-shot, or
    (ceil(T/W)*W*H*element_size,T*H*element_size,0,0) two-shot.
    The next call waits for the preceding grid before reading flags, takes
    an armed buffer and clears the previous stages using their recorded
    sizes and layout, independent of its own path. Clearing only the
    current call's rows previously hid stale data until a dip/regrow
    sequence, causing rank disagreement and serving hangs.

    Every rank must issue the same sequence of ops, T, H, fusion and path,
    across layers, steps, eager calls and replays. The path matters because
    its layouts differ; the same ceiling on all ranks chooses the same
    path. Calls may be queued ahead in time, but all collectives on one
    stream must agree in order even across different workspace objects.
    Two objects have independent rotations, not independent ordering.
    Swapping same-shaped calls on one rank gives silently position-paired
    sums; the next call realigns. Different shapes or paths can hang.
    Extra/missing calls were not exercised. These distributed invariants
    are not rank-local is_valid checks. Residual/weight shapes and dtypes
    are unchecked raw pointers in the kernel and are checked by is_valid.
    Missing fusion operands, noncontiguous tensors, unsupported widths,
    negative ceilings, capacity and two-shot alignment already fail loudly.
    Certified bf16 2-D inputs have H divisible by eight; the raw op also
    accepts fp16, fp32 and flattened N-D input, not certified. Its bf16
    one-shot maximum H is 65536, W dispatch is 2,4,8,16,32,64.

    Tests retain 16 dip/regrow decode steps of eight chained layers,
    irregular calls on two workspaces, mixed K3-order steps on one
    workspace, three rounds of 52 mixed calls queued without host sync,
    random late ranks, and six-call captured mixed steps replayed eight
    times with fresh inputs and eager calls between replays. Rotation
    flags and every result are checked, with wrong-order negative controls
    on both paths. Exact multiples of 1/16 make plain sums and updated
    bitwise-checkable. Normed is bounded by 1e-2 of its largest magnitude:
    bf16 squares cost at most 2^-9 on rsqrt and output rounding 2^-8.

    No implicit metadata, cache or compilation is used. The workspace
    handle keeps the multicast-pointer registry entry alive. PDL is read
    once per process via TRTLLM_ENABLE_PDL, default on from SM90. With
    early_trigger=True the one-shot kernel releases dependents after its
    grid wait, two-shot after scatter; consumers still wait for the grid
    before reading outputs or flags. T, H and device SM count select launch
    shape and the squares' summation order. Certification used SM100,
    W=4 POSIX handles and W=16 fabric handles, PDL on; the matrix accepts mpirun or srun.

    K3 chooses 4 MiB for decode and 1 MiB for wide (9..64-token) steps:
    plain latent H=3584 and wide attention H=7168, fused drafter H=7168
    where k3_sandwich_plain does not apply. At W=16 the 4 MiB ceilings
    are T=36/18 for H=3584/7168, versus 9/4 at 1 MiB; at W=4 they are
    146/73 and 36/18. Raw calls default to 1 MiB. The K3 module
    keeps a mapping-keyed workspace grown in 8 MiB steps and a
    module ceiling with per-call overrides; this entry takes both explicitly.
    """

    def __call__(
        self,
        input: torch.Tensor,
        workspace: MnnvlWorkspace,
        one_shot_max_bytes: int,
        residual: Optional[torch.Tensor] = None,
        norm_weight: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fused = residual is not None
        if fused != (norm_weight is not None) or fused != (eps is not None):
            raise ValueError("mnnvl_fusion_allreduce: residual, norm_weight and eps go together")
        hidden = input.shape[-1]
        num_tokens = input.numel() // hidden
        need = required_buffer_bytes(
            num_tokens, hidden, workspace.world_size, input.dtype, one_shot_max_bytes
        )
        if need > workspace.buffer_bytes:
            raise ValueError(
                f"mnnvl_fusion_allreduce: the call needs {need} bytes per Lamport buffer, the workspace has "
                f"{workspace.buffer_bytes}"
            )
        two_shot = (
            num_tokens * hidden * workspace.world_size * input.element_size() > one_shot_max_bytes
        )
        if two_shot and workspace.buffer_bytes % 32:
            # The two-shot broadcast stage starts at buffer_bytes / 2 and is accessed in 16-byte vectors.
            raise ValueError(
                "mnnvl_fusion_allreduce: a two-shot call needs the workspace's buffer_bytes to be a multiple of 32, "
                f"not {workspace.buffer_bytes}"
            )
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM if fused else AllReduceFusionOp.NONE
        outputs = native_ops.k3_mnnvl_fusion_allreduce(
            input,
            norm_weight,
            residual,
            eps,
            workspace.comm_buffer(input.dtype),
            workspace.buffer_flags,
            fused,
            None,
            int(fusion_op),
            one_shot_max_bytes,
            True,
        )
        return (outputs[0], outputs[1]) if fused else outputs[0]

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        residual: Optional[torch.Tensor] = None,
        norm_weight: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        total = torch.stack([x.float() for x in inputs_in_rank_order]).sum(dim=0)
        total = torch.where(total == 0, 0.0, total)
        dtype = inputs_in_rank_order[0].dtype
        if residual is None:
            return total.to(dtype)
        updated = total + residual.float()
        normed = updated * torch.rsqrt(updated.square().mean(dim=-1, keepdim=True) + eps)
        return (normed * norm_weight.float()).to(dtype), updated.to(dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        workspace: MnnvlWorkspace,
        one_shot_max_bytes: int,
        residual: Optional[torch.Tensor] = None,
        norm_weight: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> None:
        if residual is not None:
            assert residual.shape == input.shape, (residual.shape, input.shape)
            assert residual.dtype == input.dtype, (residual.dtype, input.dtype)
        if norm_weight is not None:
            assert norm_weight.shape == (input.shape[-1],), (norm_weight.shape, input.shape)
            assert norm_weight.dtype == input.dtype, (norm_weight.dtype, input.dtype)


mnnvl_fusion_allreduce = MnnvlFusionAllreduce()
