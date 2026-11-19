# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill with a segment-parallel chain for one long sequence on few heads; :mod:`....ops.ssm.kda_prefill`
otherwise."""

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import kda_prefill

from ...kernels.ssm import kda_prefill_segments as _segments

#: Shortest sequence the segment-parallel chain takes: eager ctx steps only (prefill CUDA graphs stop at 4096
#: tokens), where the saving is >= 70 us per call at 12 heads.
MIN_TOKENS = 8192


def _spec(heads, seq_lens, fresh, state_dtype="float32") -> dict:
    n = len(seq_lens)
    return dict(
        heads=heads,
        seq_lens=tuple(seq_lens),
        slots=tuple(range(2, 2 + n)),
        fresh=tuple(fresh),
        state_dtype=state_dtype,
    )


class K31KdaPrefillSegments(OpWrapper):
    """:class:`.kda_prefill.KdaPrefill` (same arguments, result and state update) where one sequence of at least
    ``MIN_TOKENS`` tokens on few heads (``10 * heads <= SMs``, an fp32 pool, mode ``bt16``) runs the chunked chain
    in four chunk-aligned segments over one factor prep: segment 0 from the resident state; the middle segments
    once from a zero state and once as the identity map; the true segment starts composed in fp32; the remaining
    segments from those starts. Everything else calls ``kda_prefill``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP8 (12 heads): one 8192-token continuation, the shortest segment-parallel call",
            spec=_spec(12, (8192,), (False,)),
        ),
        Cell(
            why="TEP8 (12 heads): one fresh 10000-token prompt, segments cut at 16-token chunks",
            spec=_spec(12, (10000,), (True,)),
        ),
        Cell(
            why="TEP8 (12 heads): a 16384-token ctx chunk continuation, the M2 ctx step",
            spec=_spec(12, (16384,), (False,)),
        ),
        Cell(
            why="a 4096-token sequence (a prefill CUDA graph bucket): falls back to kda_prefill",
            spec=_spec(12, (4096,), (False,)),
        ),
        Cell(
            why="two sequences: falls back to kda_prefill",
            spec=_spec(12, (6000, 3000), (False, True)),
        ),
        Cell(
            why="96 heads (DEP16): falls back to kda_prefill",
            spec=_spec(96, (8192,), (False,)),
        ),
    )
    note = """
    Drop-in for kda_prefill in the attention-TP target's KDA prefill (same arguments). It takes only eager
    steps (sequences longer than the largest prefill CUDA graph bucket). The segment-parallel path
    runs the engine BT16 decomposition's factor prep once and its DV2 chain three times (the identity-map pass
    on a side stream joined before the composition), so a 16384-token chunk at 12 heads uses ~120 SMs instead
    of 24. Not bitwise to kda_prefill: the identity-map pass rounds its running product to bf16 like the chain
    rounds the state, so outputs and the final state move within the chunked kernel's own distance from an
    fp32 recurrence (2e-2 output / 1e-2 state, relative to the largest magnitude). Workspace, state scratch and
    small int metadata are allocated per call from the torch allocator; compiled once per gate bound, warm
    before capture. ARCHS lists only the architectures the cells were driven on.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state_pool: torch.Tensor,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        lower_bound: float,
        mode: str = "auto",
        max_seqlen: int = 0,
        seq_order: Optional[torch.Tensor] = None,
        cu_chunks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        heads, total = q.shape[2], q.shape[1]
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        if (
            state_indices.numel() == 1
            and total >= MIN_TOKENS
            and 10 * heads <= sms
            and mode in ("auto", "bt16")
        ):
            return _segments.kda_prefill_segments(
                q, k, v, g, beta, state_pool, state_indices, A_log, dt_bias, scale, lower_bound
            )
        return kda_prefill(
            q,
            k,
            v,
            g,
            beta,
            state_pool,
            state_indices,
            cu_seqlens,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            mode,
            max_seqlen,
            seq_order,
            cu_chunks,
        )

    def reference(self, *args, **kwargs) -> torch.Tensor:
        """kda_prefill's fp32 token-by-token recurrence."""
        return kda_prefill.reference(*args, **kwargs)

    def is_valid(self, *args, **kwargs) -> None:
        """kda_prefill's argument checks (the segment path reads the same tensors)."""
        return kda_prefill.is_valid(*args, **kwargs)


k31_kda_prefill_segments = K31KdaPrefillSegments()
