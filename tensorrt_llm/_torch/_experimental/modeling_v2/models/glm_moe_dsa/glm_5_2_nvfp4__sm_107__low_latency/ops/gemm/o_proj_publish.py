# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention o_proj whose split-K epilogue stores the BF16 output straight into this rank's slot of
the post-attention MNNVL one-shot all-reduce's current Lamport buffer."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._low_m_splitk import (
    LamportPublish,
    device_limits,
    operands_overlap,
    publish_tactic,
    run_splitk_publish,
    splitk_default_tactic,
)

#: GLM-5.2 o_proj at tp16: 4 heads * v_head_dim 256 = 1024 -> hidden 6144, 16 ranks.
_GLM = dict(k=1024, n=6144, world_size=16)

#: Lamport buffers of an MNNVL one-shot workspace, and the int32 words of its buffer flags.
_LAMPORT_BUFFERS = 3
_FLAG_WORDS = 4


class GlmOProjPublish(OpWrapper):
    """`x @ weight.T` rounded to BF16, stored at `[token][rank][column]` of the all-reduce
    workspace's current Lamport buffer instead of into an output; returns None.

    The current buffer is `lamport` (bytes) + `flags[0] * flags[2]`: the flags' current index
    times their bytes per buffer, read once per CTA after the grid dependency wait (the previous
    all-reduce rotated them). Value (t, c) lands at element `(t * world_size + rank) * N + c` of
    that buffer, in BF16 with -0.0 written as +0.0 (-0.0 is the buffer's idle value): this
    rank's shard as the one-shot kernel would have stored it. The all-reduce of this output must
    then take the published shard (it neither loads nor stores its own) and poll every slot. The
    owner CTA of each 64 x 8 tile stores 16-byte vectors of 8 columns through `lamport`, which is
    the workspace's multicast address in production, so the slot fills on every rank.

    `out` gives the GEMM its output layout and is not written.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="verify step: 8 tokens, 1024 -> 6144, split-K 2 (192 CTAs), rank 5 of 16",
            spec=dict(_GLM, m=8, rank=5, current=1),
        ),
        Cell(
            why="MTP draft steps 1-6: 1 token, last rank, buffer index 2",
            spec=dict(_GLM, m=1, rank=15, current=2),
        ),
        Cell(
            why="verify step at 4 ranks: the slot arithmetic of another group size, buffer 0",
            spec=dict(_GLM, m=8, rank=2, current=0, world_size=4),
        ),
    )

    note = """
    The tactic is the shape's `splitk_default_tactic` with its A/B stages lowered to fit the FP32
    staging tile (`publish_tactic`; the c1 o_proj keeps all 12). PDL: the kernel is always
    expected to run with `TRTLLM_ENABLE_PDL` on (the default) in front of the all-reduce; its
    epilogue warps wait on the grid dependency, release the dependents (the all-reduce) right
    after the wait and only then read the flags. With PDL off the flags are read without a wait,
    which is ordered by the plain stream dependency.

    The Lamport workspace, its multicast mapping and its flags belong to the all-reduce; this op
    reads the flags and writes one slot per token, nothing else. Loud, so not guarded
    (ValueError before the launch): M outside [1, 32], K not a multiple of 128, N not a multiple
    of 8, non-BF16 operands, a `lamport` that is not a 1-D BF16 view or `flags` that is not a
    1-D int32 view. The first call of a (tactic, world size, rank) compiles the kernel; make it
    before CUDA-graph capture.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        lamport: torch.Tensor,
        flags: torch.Tensor,
        world_size: int,
        rank: int,
    ) -> None:
        """Publish `x @ weight.T` into the current Lamport buffer.

        Args:
            x: [M, K] BF16, contiguous: the attention output of this rank's heads.
            weight: [N, K] BF16, contiguous: o_proj's row-parallel shard.
            out: [M, N] BF16, contiguous, 32-byte aligned: layout only, not written.
            lamport: 1-D BF16 view of the workspace's Lamport buffers (multicast address).
            flags: 1-D int32 view of the workspace's buffer flags, 16-byte aligned: [0] the
                current buffer index, [2] the bytes per buffer.
            world_size: Ranks of the all-reduce group.
            rank: This rank in the group.
        """
        m, k = x.shape
        n = weight.shape[0]
        sm_count, smem_capacity = device_limits(x.device.index)
        tactic = publish_tactic(
            splitk_default_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity),
            smem_capacity=smem_capacity,
        )
        run_splitk_publish(
            x,
            weight.t(),
            out,
            lamport,
            flags,
            get_env_enable_pdl(),
            tactic,
            LamportPublish(world_size=world_size, rank=rank, trigger_after_wait=True),
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        lamport: torch.Tensor,
        flags: torch.Tensor,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        """The [M, N] BF16 values the op stores at slots `t * world_size + rank` of the current
        buffer: FP32-accumulated (TF32 off), rounded once, -0.0 as +0.0. Where they land is the
        test's to check; `out`, `lamport` and `flags` are not read."""
        with true_fp32_matmul():
            values = (x.float() @ weight.float().t()).to(torch.bfloat16)
        return torch.where(values == 0, torch.zeros_like(values), values)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        lamport: torch.Tensor,
        flags: torch.Tensor,
        world_size: int,
        rank: int,
    ) -> None:
        # A rank outside the group lands in another token's slots.
        assert 0 <= rank < world_size, f"rank {rank} is outside a group of {world_size}"
        # Every store goes through the flags' current buffer; slots past its end land in the
        # next buffer (or past the workspace) without complaint.
        assert flags.numel() >= _FLAG_WORDS and flags.data_ptr() % 16 == 0, (
            "flags must hold the four 16-byte aligned words the kernel reads"
        )
        current, _, bytes_per_buffer, _ = (int(v) for v in flags[:_FLAG_WORDS].tolist())
        m, n = x.shape[0], weight.shape[0]
        assert 0 <= current < _LAMPORT_BUFFERS, f"current buffer index {current} is out of range"
        assert m * world_size * n * lamport.element_size() <= bytes_per_buffer, (
            "the Lamport buffer is smaller than the [token][rank][column] slots it must hold"
        )
        assert (current + 1) * bytes_per_buffer <= lamport.numel() * lamport.element_size(), (
            "the current Lamport buffer runs past the end of `lamport`"
        )
        # The epilogue stores while other clusters still read x and the weight.
        for operand, name in ((x, "x"), (weight, "weight")):
            assert not operands_overlap(lamport, operand), (
                f"lamport must not share memory with {name}"
            )


o_proj_publish = GlmOProjPublish()
