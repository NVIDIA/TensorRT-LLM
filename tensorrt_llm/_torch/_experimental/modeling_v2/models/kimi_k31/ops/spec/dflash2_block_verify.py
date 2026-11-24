# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block verification (accept count and bonus token per request) as one CuTe DSL launch."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_block_verify import (
    MAX_CANDIDATES,
    MAX_DRAFTS,
    dflash2_block_verify_cute,
    dflash2_block_verify_seeded_cute,
)
from .dflash2_sampling import dflash2_uniforms


def _torch_dflash2_block_verify(*args, **kwargs):
    # Imported on use: dflash2_verify (the reference's home) imports this entry at its call site.
    from .dflash2_verify import dflash2_block_verify

    return dflash2_block_verify(*args, **kwargs)


#: Kimi K3.1 DFlash2: the target vocabulary (the verify logits are gathered over TP) and the
#: selector's candidates per slot.
_VOCAB = 163840
_TOP_K = 32


def _spec(batch: int, drafts: int, **overrides) -> dict:
    spec = dict(
        batch=batch,
        drafts=drafts,
        vocab=_VOCAB,
        top_k=_TOP_K,
        temperature=1.0,
        closeness=1.0,
        top_p_floor=0.0,
        stale=0.0,
        outside=0.0,
        padding_rows=0,
        degenerate_rows=0,
        draft_dtype=torch.int32,
        cand_dtype=torch.int64,
    )
    spec.update(overrides)
    return spec


class K31DFlash2BlockVerify(OpWrapper):
    """``(tau [n] int64, bonus [n] int64)``: block verification of each request's drafted block.

    Same arguments and contract as the family's torch ``dflash2_block_verify``: ``target_probs``
    ``[n, g + 1, V]`` float32 (row ``i`` scores the token after ``i`` accepted drafts),
    ``draft_tokens [n, g]``, the selector's ``candidate_ids`` / ``q`` ``[n, g, k]``, the
    acceptance uniforms ``coins [n, g]`` and the bonus uniforms ``final_coins [n]``; block
    verification is Sun et al. 2024, Algorithm 2. ``tau`` is the accepted draft count, ``bonus``
    the token drawn after them (from ``p_g`` when all are accepted, else from
    ``max(r_tau p_tau - q_tau, 0)`` by inverse CDF).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1, w5: a single request (the reference's own scan is CUB's here; see note)",
            spec=_spec(1, 4),
        ),
        Cell(why="c1, w8: a single request, seven drafts", spec=_spec(1, 7)),
        Cell(why="8 requests, w5, T = 1", spec=_spec(8, 4)),
        Cell(
            why="16 requests, w8, drafter close to the target (long accepted prefixes)",
            spec=_spec(16, 7, closeness=0.3),
        ),
        Cell(
            why="64 requests, w5, T = 0.6 and a close drafter: full acceptance, bonus from p_g",
            spec=_spec(64, 4, temperature=0.6, closeness=0.1),
        ),
        Cell(why="64 requests, w8, T = 1", spec=_spec(64, 7)),
        Cell(
            why="top-p style zeros in the target rows (residual mass on a truncated support)",
            spec=_spec(32, 4, top_p_floor=1e-6),
        ),
        Cell(
            why="unwalked requests (point-mass proposals) and drafts outside the candidates",
            spec=_spec(32, 7, stale=0.3, outside=0.1),
        ),
        Cell(
            why="CUDA-graph padding requests: zeroed logits give uniform target rows",
            spec=_spec(8, 4, padding_rows=3),
        ),
        Cell(
            why="rows without mass (all-zero and NaN target rows): the first-maximum fallback",
            spec=_spec(6, 4, degenerate_rows=4),
        ),
        Cell(
            why="int64 drafted tokens and int32 candidate ids",
            spec=_spec(4, 4, draft_dtype=torch.int64, cand_dtype=torch.int32),
        ),
        Cell(why="sm_107 DEP16 generation: 12 requests a rank, w8", spec=_spec(12, 7)),
        Cell(why="sm_107 helix8 generation: 16 requests, w6", spec=_spec(16, 5)),
    )

    note = """
    The DFlash2 worker's sampled verification (``dflash2_sample_and_accept``) calls it once per
    decode step with the generation requests' target probabilities (full vocabulary: the verify
    logits are gathered across TP, so the tensor-parallel width does not change the shapes),
    K = 4 or 7 drafts, the selector's top-32 candidates and the SplitMix64 uniforms
    (``uniforms[:, :K]``, ``uniforms[:, K]``). Batch 1..64 requests. One launch in place of the
    reference's ~90 (see the test's launch count); no host synchronization.

    Bitwise the reference wherever the reference is deterministic. Every float32 operation is the
    reference's, in its order: the candidate sums (torch's 32-wide reduction tree), the vocabulary
    sum of each interior target row (the CTAs are torch's reduction CTAs for the batch and the
    device's SM count), and the bonus row's cumulative sum (torch's chunked Sklansky scan); the
    test puts acceptance uniforms exactly at the reference's ``h_i`` and one ulp below it. The row
    sums follow torch's order when V is a multiple of 2048 and the batch's scan chunks are 1024
    columns (every K3.1 shape); other vocabularies use the kernel's own order. Two inputs fall
    outside: (1) a single request's bonus (n = 1): torch.cumsum of a one-row tensor runs CUB's
    decoupled-lookback scan, which is not reproducible run to run (about 1 % of bonus draws over
    a 163840 vocab differ between two calls of the reference itself); the kernel takes the scan
    torch runs for every n >= 2, and the test compares the bonus against the reference on the
    request duplicated to two rows (``tau`` against the single request); (2) a token repeated
    among one slot's candidates with nonzero mass on more than one lane makes the reference's
    scatter-add order-dependent; the selector's candidates are distinct.

    CTAs per (request, target row, row slice); the last CTA of a request finishes it,
    coordinated by a per-request counter in a workspace cached per (n, g, V). The first call of a shape compiles
    the kernel and allocates the workspace, so it must be eager; CUDA-graph capture then replays
    the launch.
    """

    def __call__(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return dflash2_block_verify_cute(
            target_probs, draft_tokens, candidate_ids, q, coins, final_coins
        )

    def reference(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The family's torch block verification (``ops/spec/dflash2_verify.py``)."""
        return _torch_dflash2_block_verify(
            target_probs, draft_tokens, candidate_ids, q, coins, final_coins
        )

    def is_valid(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        coins: torch.Tensor,
        final_coins: torch.Tensor,
    ) -> None:
        # The kernel reads raw float32 / int bits at the given strides: another dtype is
        # reinterpreted instead of converted.
        assert target_probs.dtype == torch.float32 and target_probs.dim() == 3, (
            f"float32 [n, g + 1, V] target_probs, got {target_probs.dtype} "
            f"{tuple(target_probs.shape)}"
        )
        assert q.dtype == torch.float32 and coins.dtype == torch.float32, "float32 q and coins"
        assert final_coins.dtype == torch.float32, "float32 final_coins"
        assert draft_tokens.dtype in (torch.int32, torch.int64), (
            f"int32/int64 draft tokens, got {draft_tokens.dtype}"
        )
        assert candidate_ids.dtype in (torch.int32, torch.int64), (
            f"int32/int64 candidate ids, got {candidate_ids.dtype}"
        )
        n, rows, vocab = target_probs.shape
        g = rows - 1
        k = candidate_ids.shape[-1]
        assert 1 <= g <= MAX_DRAFTS and k <= MAX_CANDIDATES, (
            f"1..{MAX_DRAFTS} drafts and <= {MAX_CANDIDATES} candidates, got g={g}, k={k}"
        )
        assert tuple(draft_tokens.shape) == (n, g), "draft_tokens [n, g]"
        assert tuple(candidate_ids.shape) == (n, g, k) and tuple(q.shape) == (n, g, k), (
            "candidate_ids and q [n, g, k]"
        )
        assert tuple(coins.shape) == (n, g) and tuple(final_coins.shape) == (n,), (
            "coins [n, g], final_coins [n]"
        )
        # Rows are read in 16-byte vectors when V % 4 == 0: the vocab stride must be 1 and every
        # row start 16-byte aligned, or the loads fault / read shifted columns.
        assert target_probs.stride(2) == 1, "target_probs needs a unit vocab stride"
        if vocab % 4 == 0:
            assert target_probs.data_ptr() % 16 == 0, "target_probs must be 16-byte aligned"
            assert target_probs.stride(0) % 4 == 0 and target_probs.stride(1) % 4 == 0, (
                "target_probs row strides must keep rows 16-byte aligned"
            )


dflash2_block_verify = K31DFlash2BlockVerify()


class K31DFlash2BlockVerifySeeded(K31DFlash2BlockVerify):
    """The same verification with its uniforms derived in the kernel: ``seed`` / ``offset``
    (int64 ``[n]``, the request's Philox pair for the accept slot) key the SplitMix64 stream of
    ``dflash2_uniforms(seed, offset, g + 1)``, whose first ``g`` draws are the acceptance coins
    and the last the bonus draw, exactly as ``dflash2_sample_and_accept`` builds them. ``block``
    is the sm_107 targets' argument and must be True: the entry verifies by blocks only.
    """

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1, w5, seeded uniforms", spec=_spec(1, 4)),
        Cell(why="c1, w8, seeded uniforms", spec=_spec(1, 7)),
        Cell(why="16 requests, w8, close drafter, seeded", spec=_spec(16, 7, closeness=0.3)),
        Cell(
            why="64 requests, w5, T = 0.6, seeded (full acceptance, bonus from p_g)",
            spec=_spec(64, 4, temperature=0.6, closeness=0.1),
        ),
        Cell(why="sm_107 DEP16 generation: 12 requests a rank, w8, seeded", spec=_spec(12, 7)),
        Cell(why="sm_107 helix8 generation: 16 requests, w6, seeded", spec=_spec(16, 5)),
    )

    note = """
    ``dflash2_sample_and_accept``'s ``dflash2_uniforms`` + ``dflash2_block_verify`` pair as one
    launch (the uniforms' ~40 int64 torch launches fold into the verification kernel). The
    uniforms are integer SplitMix64 arithmetic and an exact float conversion, so they are
    bitwise ``dflash2_uniforms``'; everything else is the unseeded entry's (see its note).
    """

    def __call__(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        block: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not block:
            raise ValueError("dflash2_block_verify_seeded verifies by blocks only (block=True)")
        return dflash2_block_verify_seeded_cute(
            target_probs, draft_tokens, candidate_ids, q, seed, offset
        )

    def reference(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        block: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``dflash2_sample_and_accept``'s uniforms and torch block verification."""
        k = draft_tokens.shape[1]
        uniforms = dflash2_uniforms(seed, offset, k + 1)
        return _torch_dflash2_block_verify(
            target_probs,
            draft_tokens,
            candidate_ids,
            q,
            uniforms[:, :k],
            uniforms[:, k],
        )

    def is_valid(
        self,
        target_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        candidate_ids: torch.Tensor,
        q: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        block: bool = True,
    ) -> None:
        assert block, "block verification only"
        n, rows = target_probs.shape[0], target_probs.shape[1]
        # The kernel reads 8-byte seeds and offsets: int32 ones would be read two at a time.
        assert seed.dtype == torch.int64 and offset.dtype == torch.int64, "int64 seed and offset"
        assert tuple(seed.shape) == (n,) and tuple(offset.shape) == (n,), "seed, offset [n]"
        super().is_valid(
            target_probs,
            draft_tokens,
            candidate_ids,
            q,
            torch.empty(n, rows - 1, device=q.device),
            torch.empty(n, device=q.device),
        )


dflash2_block_verify_seeded = K31DFlash2BlockVerifySeeded()
