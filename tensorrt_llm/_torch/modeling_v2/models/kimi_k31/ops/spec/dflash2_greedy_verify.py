# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 strict (greedy) verification of the greedy rows of a sampling batch, one CuTe DSL launch."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_greedy_verify import MAX_DRAFTS, dflash2_greedy_verify_cute


def _torch_dflash2_greedy_verify(*args, **kwargs):
    # Imported on use: dflash2_verify (the reference's home) imports this entry at its call site.
    from .dflash2_verify import dflash2_greedy_verify

    return dflash2_greedy_verify(*args, **kwargs)


#: Kimi K3.1: the target vocabulary (verify logits are gathered over TP).
_VOCAB = 163840


def _spec(batch: int, drafts: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        drafts=drafts,
        vocab=_VOCAB,
        dtype=torch.float32,
        draft_dtype=torch.int32,
        agree=0.7,
        ties=False,
        nan_rows=0,
    )
    spec.update(overrides)
    return spec


class K31DFlash2GreedyVerify(OpWrapper):
    """``(tau [n] int64, bonus [n] int64)``: strict acceptance against the target's argmax.

    ``target_logits [n, g + 1, V]``, ``draft_tokens [n, g]``: ``tau`` is the number of leading
    drafts equal to the first maximum of their target row, ``bonus`` the first maximum of row
    ``tau``. The same contract as the family's torch ``dflash2_greedy_verify``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1, w5, float32 logits", spec=_spec(1, 4)),
        Cell(why="c1, w8, float32 logits", spec=_spec(1, 7)),
        Cell(why="8 requests, w5, mostly agreeing drafts", spec=_spec(8, 4, agree=0.9)),
        Cell(why="64 requests, w5", spec=_spec(64, 4)),
        Cell(why="64 requests, w8", spec=_spec(64, 7)),
        Cell(
            why="bfloat16 logits with repeated maxima (ties go to the lower index)",
            spec=_spec(16, 4, dtype=torch.bfloat16, ties=True),
        ),
        Cell(why="float16 logits, w8", spec=_spec(8, 7, dtype=torch.float16)),
        Cell(
            why="NaN in target rows (argmax takes the first NaN), int64 drafts",
            spec=_spec(8, 4, nan_rows=3, draft_dtype=torch.int64),
        ),
        Cell(why="sm_107 DEP16 generation: 12 requests a rank, w8", spec=_spec(12, 7)),
        Cell(why="sm_107 Helix CP 8 generation: 16 requests, w6", spec=_spec(16, 5)),
    )

    note = """
    ``dflash2_sample_and_accept`` calls it for the greedy requests of a DFlash2 sampling batch
    (rows whose temperature is below the greedy threshold verify strictly, the others by block
    verification), with the gathered full-vocabulary target logits of the verify block, K = 4 or
    7. An all-greedy batch never gets here: it takes the engine's acceptance
    (``_sample_tokens_for_batch`` + the Triton ``strict_accept`` kernel). One launch in place of
    the reference's argmax / compare / cumprod / gather chain; integer outputs, bitwise the
    reference for every input. One CTA per (request, target row, column slice); the last CTA of
    a request combines the slices. Compiles once per (g, V, dtypes); the first call of a batch
    size allocates the workspace and must be eager, CUDA-graph capture then replays the launch.
    """

    def __call__(
        self, target_logits: torch.Tensor, draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return dflash2_greedy_verify_cute(target_logits, draft_tokens)

    def reference(
        self, target_logits: torch.Tensor, draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The family's torch strict acceptance (``ops/spec/dflash2_verify.py``)."""
        return _torch_dflash2_greedy_verify(target_logits, draft_tokens)

    def is_valid(self, target_logits: torch.Tensor, draft_tokens: torch.Tensor) -> None:
        # The kernel reads raw element bits at the given strides: another dtype is
        # reinterpreted, not converted.
        assert target_logits.dtype in (torch.float32, torch.bfloat16, torch.float16), (
            f"float32 / bfloat16 / float16 logits, got {target_logits.dtype}"
        )
        assert draft_tokens.dtype in (torch.int32, torch.int64), (
            f"int32/int64 draft tokens, got {draft_tokens.dtype}"
        )
        assert target_logits.dim() == 3 and draft_tokens.dim() == 2, (
            "target_logits [n, g + 1, V], draft_tokens [n, g]"
        )
        n, rows, vocab = target_logits.shape
        assert 1 <= rows - 1 <= MAX_DRAFTS, f"1..{MAX_DRAFTS} drafts, got {rows - 1}"
        assert tuple(draft_tokens.shape) == (n, rows - 1), "draft_tokens [n, g]"
        assert target_logits.stride(2) == 1, "target_logits needs a unit vocab stride"
        per_vec = 16 // target_logits.element_size()
        # Whole 16-byte vectors are read when V is a multiple of one: rows must be aligned.
        if vocab % per_vec == 0:
            assert target_logits.data_ptr() % 16 == 0, "target_logits must be 16-byte aligned"
            assert (
                target_logits.stride(0) % per_vec == 0 and target_logits.stride(1) % per_vec == 0
            ), "target_logits row strides must keep rows 16-byte aligned"


dflash2_greedy_verify = K31DFlash2GreedyVerify()
