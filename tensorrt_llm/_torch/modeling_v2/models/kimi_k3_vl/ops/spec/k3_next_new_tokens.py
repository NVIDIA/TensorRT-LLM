# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The next step's new tokens of a speculative batch (the last accepted token, then the new
draft), optionally with the KV-length rewind after the draft forward, in one Triton launch."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import (
    fused_next_new_tokens,
    fused_next_new_tokens_rewind,
)


class K3NextNewTokens(OpWrapper):
    """``out[b] = cat([accepted_tokens[b, num_accepted_tokens[b] - 1], next_draft_tokens[b]])``,
    ``[batch, K + 1]`` in the promoted dtype of the two token tensors (new).

    ``accepted_tokens`` ``[>= batch, >= K + 1]`` and ``next_draft_tokens`` ``[batch, K]`` have
    unit inner strides. With ``kv_lens`` and ``rewind`` (``[num_gens]``, contiguous) also
    ``kv_lens[num_contexts + g] = max(kv_lens[num_contexts + g] - rewind[g], 0)``, in one program.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 rows of DSpark 5, 128 rows per program",
            spec=dict(batch=16, draft_len=5, rewind=False),
        ),
        Cell(
            why="helix8 c240: 20 rows of DSpark 5",
            spec=dict(batch=20, draft_len=5, rewind=False),
        ),
        Cell(
            why="helix8 c75/c174: 8 rows of DSpark 7 with the KV-length rewind in one program",
            spec=dict(batch=8, draft_len=7, rewind=True),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 rows of DSpark 7 with the KV-length rewind",
            spec=dict(batch=4, draft_len=7, rewind=True),
        ),
    )

    note = """
    ``SpecWorkerBase._prepare_next_new_tokens`` with identity batch indices. Without the rewind
    the kernel runs 128 rows per program; with it one program covers the batch and the rewind.
    """

    def __call__(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        num_contexts: int = 0,
    ) -> torch.Tensor:
        if kv_lens is None:
            return fused_next_new_tokens(accepted_tokens, next_draft_tokens, num_accepted_tokens)
        return fused_next_new_tokens_rewind(
            accepted_tokens, next_draft_tokens, num_accepted_tokens, kv_lens, rewind, num_contexts
        )

    def reference(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        num_contexts: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(out, kv_lens after the call or None)``; nothing is written."""
        batch, k = next_draft_tokens.shape
        dtype = torch.promote_types(accepted_tokens.dtype, next_draft_tokens.dtype)
        out = torch.empty((batch, k + 1), dtype=dtype, device=accepted_tokens.device)
        last = (num_accepted_tokens[:batch].to(torch.int64) - 1).unsqueeze(1)
        out[:, 0] = accepted_tokens[:batch].gather(1, last).squeeze(1).to(dtype)
        out[:, 1:] = next_draft_tokens.to(dtype)
        kv_out = None
        if kv_lens is not None:
            num_gens = rewind.shape[0]
            kv_out = kv_lens.clone()
            gen = kv_out[num_contexts : num_contexts + num_gens]
            kv_out[num_contexts : num_contexts + num_gens] = (gen - rewind.to(gen.dtype)).clamp(
                min=0
            )
        return out, kv_out

    def is_valid(
        self,
        accepted_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        num_contexts: int = 0,
    ) -> None:
        # The kernel reads accepted[b, num_accepted[b] - 1] unchecked.
        batch = next_draft_tokens.shape[0]
        n = num_accepted_tokens[:batch]
        assert bool(((n >= 1) & (n <= accepted_tokens.shape[1])).all()), "num_accepted range"
        assert (kv_lens is None) == (rewind is None), "kv_lens and rewind go together"


k3_next_new_tokens = K3NextNewTokens()
