# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw target scores for the accepted prefix of a linear speculative step."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ...executor.result import Logprob

HIDDEN_STATES_KEY = "spec_target_hidden_states"
GATHER_IDS_KEY = "spec_target_gather_ids"


@dataclass(kw_only=True)
class RawLogProbs:
    """Compact scores, ordered by request, speculative position, then top-k."""

    sampled: torch.Tensor
    ranks: torch.Tensor
    top_values: torch.Tensor
    top_indices: torch.Tensor

    def to_host(self, copy_to_host: Callable[[torch.Tensor], torch.Tensor]) -> "RawLogProbs":
        return RawLogProbs(
            sampled=copy_to_host(self.sampled),
            ranks=copy_to_host(self.ranks),
            top_values=copy_to_host(self.top_values),
            top_indices=copy_to_host(self.top_indices),
        )


def target_row_indices(
    output_rows: torch.Tensor,
    *,
    num_contexts: int,
    num_generations: int,
    width: int,
    verify_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map output-request rows to packed target rows without reading device counts.

    Contexts have one target row; generation rows have either ``width`` rows or
    their executed verification window. Padding positions use their request's
    first row, and are excluded from results by the accepted-token count.
    """
    if width < 1 or num_contexts < 0 or num_generations < 0:
        raise ValueError("target row counts and width must describe a linear step")
    if verify_lens is None:
        windows = torch.full(
            (num_contexts + num_generations,), width, device=output_rows.device, dtype=torch.long
        )
        windows[:num_contexts] = 1
    else:
        if verify_lens.ndim != 1 or verify_lens.shape[0] != num_contexts + num_generations:
            raise ValueError("verification windows must cover every output request")
        windows = verify_lens.to(dtype=torch.long)
    starts = torch.cumsum(windows, dim=0) - windows
    positions = torch.arange(width, device=output_rows.device)
    valid = positions.unsqueeze(0) < windows[output_rows].unsqueeze(1)
    rows = starts[output_rows].unsqueeze(1) + positions
    return torch.where(valid, rows, starts[output_rows].unsqueeze(1)), valid


def raw_logprobs(logits: torch.Tensor, tokens: torch.Tensor, top_k: int) -> RawLogProbs:
    """Score selected tokens against the original temperature-one target logits.

    ``logits`` has shape ``[requests, width, vocab]``. These are target scores,
    including for rejected-and-resampled bonus tokens; draft proposal scores
    and the rejection sampler's residual distribution are never returned.
    """
    if logits.ndim != 3 or tokens.shape != logits.shape[:2]:
        raise ValueError("target logits and accepted-token rows must have matching shapes")
    if not 0 <= top_k <= logits.shape[-1]:
        raise ValueError("top-k must fit the target vocabulary")
    logits = logits.float()
    normalizers = torch.logsumexp(logits, dim=-1)
    sampled_logits = logits.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)
    ranks = (logits > sampled_logits.unsqueeze(-1)).sum(-1)
    top_values, top_indices = torch.topk(logits, top_k, dim=-1)
    return RawLogProbs(
        sampled=sampled_logits - normalizers,
        ranks=ranks,
        top_values=top_values - normalizers.unsqueeze(-1),
        top_indices=top_indices,
    )


def token_logprobs(
    scores: RawLogProbs,
    row: int,
    tokens: list[int],
    *,
    top_k: int,
    simple_format: bool,
) -> list[float] | list[dict[int, Logprob]]:
    """Convert exactly the committed prefix, including a stop token, to results."""
    sampled = scores.sampled[row, : len(tokens)].tolist()
    if simple_format:
        return sampled
    ranks = scores.ranks[row, : len(tokens)].tolist()
    top_values = scores.top_values[row, : len(tokens), :top_k].tolist()
    top_indices = scores.top_indices[row, : len(tokens), :top_k].tolist()
    result = []
    for token, value, rank, values, indices in zip(tokens, sampled, ranks, top_values, top_indices):
        entry = {token: Logprob(logprob=value, rank=rank + 1)}
        entry.update(
            (index, Logprob(logprob=score, rank=position + 1))
            for position, (index, score) in enumerate(zip(indices, values))
        )
        result.append(entry)
    return result
