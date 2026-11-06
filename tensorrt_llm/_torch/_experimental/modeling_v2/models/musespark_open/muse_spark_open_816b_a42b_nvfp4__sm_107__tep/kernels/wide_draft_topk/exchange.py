# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's draft-head key exchange on draft steps of 33 or more rows (``wide_draft_topk``).

``draft_head`` exchanges, per draft row, every rank's per-chunk top
``selector_top_k`` keys (``draft_shard_topk``: 13 chunks x 16 keys at TP4)
through the MNNVL fp32 all-reduce, and ``draft_topk_merge`` rebuilds
``torch.topk``'s full-vocabulary result from them. That all-reduce leaves its
one-shot kernel above 1 MiB (rows x 4 ranks x 208 keys x 2 halves x 4 B: above
39 rows at TP4, six requests at K = 7). With this fast path a draft step of
``MIN_ROWS`` or more rows exchanges each rank's own top ``selector_top_k`` keys
only (``draft_rank_topk``). They hold every key of the full vocabulary's top
``selector_top_k`` that the rank owns: a key's position in its rank's order
(value desc, column asc) is at most its position in the global order (value
desc, vocab id asc), since vocab id = rank * shard + column orders a rank's
columns the same way. So the merge returns the same result bit for bit, from
1/13 of the bytes at TP4 (one-shot up to 512 rows). On these steps the
per-chunk stage runs ``CHUNK_WARPS`` warps per program and the merge over 16
keys per rank ``MERGE_LAUNCH``'s geometry instead of the one-request tuning;
neither changes an output bit (integer selections and exact gathers).

Draft steps of fewer rows (one request, and the two to four requests of the
9..32-row band) keep ``draft_head``'s exchange, kernels and launch geometry. The
install wraps the draft head's all-reduce (the worker's
``_draft_head_allreduce``) in a ``RankTopKExchange``, which
``draft_head.selector_draft_tokens`` asks per call (``takes``: a function of
the row count only, the same on every rank). Before wrapping it,
``tie_order_reason`` runs the rank top-k layout with every launch of this
fast path against the running ``torch.topk`` on the draft head's tie-heavy
probes (``draft_head.tie_order_probes``) and refuses the install if they
disagree. With the drafter's logit transform installed on the draft head
(``draft_head/logit_transform.py``) ``keys`` takes its canonical keys, and the
probes and their reference are the transformed ones.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
from torch import nn

from tensorrt_llm.logger import logger

from ...ops.spec.draft_rank_topk import MAX_KEYS, draft_rank_topk
from ...ops.spec.draft_shard_topk import CHUNK, FULL_BAND
from ..draft_head import head as _draft_head
from ..draft_head.logit_transform import DraftLogitTransform

#: Fewest draft rows (requests x K) the exchange takes: five requests at K = 7. Fewer
#: rows are the one-request path and the 9..32-row band.
MIN_ROWS = 33
#: Warps per program of the per-chunk stage on the steps the exchange takes.
CHUNK_WARPS = 4
#: The merge's launch by rows: ``(most rows, rank_block, num_warps)``; the first row
#: whose most rows the step stays within applies.
MERGE_LAUNCH = ((512, 128, 2), (1 << 20, 256, 2))
#: The worker attribute holding the draft head's key exchange (``draft_head.install``).
ATTRIBUTE = "_draft_head_allreduce"


class RankTopKExchange:
    """The draft head's MNNVL key all-reduce, carrying each rank's own top-k on wide steps."""

    def __init__(self, allreduce: Optional[nn.Module]):
        self.allreduce = allreduce

    def takes(self, rows: int) -> bool:
        """Whether a draft step of ``rows`` rows exchanges the rank top-k (``keys``)."""
        return rows >= MIN_ROWS

    def keys(
        self,
        local: torch.Tensor,
        top_k: int,
        rank: int,
        num_ranks: int,
        canon: Optional[torch.Tensor] = None,
        canon_band: tuple[int, int] = FULL_BAND,
    ) -> torch.Tensor:
        """This rank's own top ``top_k`` keys of ``local`` in the all-reduce layout.

        ``canon`` / ``canon_band``: a drafter logit transform's canonical keys
        (``draft_shard_topk``'s), passed by ``draft_head.selector_draft_tokens``.
        """
        return draft_rank_topk(
            local, top_k, CHUNK, CHUNK_WARPS, rank, num_ranks, canon=canon, canon_band=canon_band
        )

    def merge_launch(self, rows: int) -> dict:
        """``draft_topk_merge``'s launch geometry for a step of ``rows`` rows."""
        for most, rank_block, num_warps in MERGE_LAUNCH:
            if rows <= most:
                break
        return dict(rank_block=rank_block, num_warps=num_warps)

    def __call__(self, packed: torch.Tensor) -> torch.Tensor:
        return self.allreduce(packed)


def tie_order_reason(
    selector: nn.Module,
    num_ranks: int,
    topk=torch.topk,
    transform: Optional[DraftLogitTransform] = None,
) -> Optional[str]:
    """Why the rank top-k exchange would differ from this torch's ``torch.topk``, or None.

    ``draft_head.tie_order_reason``'s probes for the rank top-k layout: the
    probe logits split into ``num_ranks`` shards, each shard's ``keys``, their
    sum (what the all-reduce returns) merged with every ``MERGE_LAUNCH``
    geometry, ids and values compared bit for bit with ``topk`` as the built-in
    selector runs it. With ``transform`` (the draft head's) the keys are
    canonical, the merge reads its values table and the reference transforms
    the logits first. ``topk`` is a parameter so a test can stand in another order.
    """
    predecessor_codebook = selector.predecessor_codebook
    vocab, rank_dim = predecessor_codebook.shape
    shard = vocab // num_ranks
    top_k = selector.top_k
    device = predecessor_codebook.device
    exchange = RankTopKExchange(None)
    canon = _draft_head.canon_kwargs(transform)
    with torch.cuda.device(device):
        for probe, full in enumerate(_draft_head.tie_order_probes(vocab, shard, transform, device)):
            rows = full.shape[0]
            ref_values, ref_ids = _draft_head.probe_reference(full, top_k, transform, topk)
            keys = torch.stack(
                [
                    exchange.keys(
                        full[:, r * shard : (r + 1) * shard].contiguous(),
                        top_k,
                        r,
                        num_ranks,
                        **canon,
                    )
                    for r in range(num_ranks)
                ]
            ).sum(dim=0)
            gate = torch.zeros(1, rows, rank_dim, dtype=torch.bfloat16, device=device)
            anchor = torch.zeros(1, dtype=torch.int64, device=device)
            for _, rank_block, num_warps in MERGE_LAUNCH:
                ids, values, _, _ = _draft_head.draft_topk_merge(
                    keys,
                    shard,
                    1,
                    rows,
                    anchor,
                    gate,
                    predecessor_codebook,
                    selector.successor_codebook,
                    top_k,
                    num_ranks,
                    rank_block=rank_block,
                    num_warps=num_warps,
                    **_draft_head.values_kwargs(transform),
                )
                if not (torch.equal(ids, ref_ids) and _draft_head._same_bits(values, ref_values)):
                    differ = (ids != ref_ids).any(dim=-1).view(-1).nonzero().view(-1).tolist()
                    return (
                        f"torch {torch.__version__} orders top-k ties differently from the merge "
                        f"over rank top-k keys (rank_block {rank_block}, {num_warps} warps; probe "
                        f"{probe} rows {differ} differ)"
                    )
    return None


def unsupported_reason(llm: nn.Module) -> Optional[str]:
    """Why the rank top-k exchange cannot be installed on ``llm``'s draft head, or None."""
    if not _draft_head.installed(llm):
        return "the draft head (draft_head) is not installed"
    if llm.spec_worker.__dict__.get(ATTRIBUTE) is None:
        return "the draft head exchanges its keys by NCCL all-gather (no MNNVL all-reduce)"
    draft_model = getattr(llm, "draft_model", None)
    selector = getattr(draft_model, "candidate_selector", None)
    lm_head = getattr(draft_model, "lm_head", None)
    if selector is None or lm_head is None:
        return "the drafter has no candidate selector or LM head"
    shard = int(lm_head.out_features)
    if triton.cdiv(shard, CHUNK) * selector.top_k > MAX_KEYS:
        return f"a {shard}-column shard has more than {MAX_KEYS} per-chunk keys per row"
    return None


def install(llm: nn.Module) -> bool:
    """Wrap the draft head's key all-reduce in a ``RankTopKExchange``. Idempotent.

    Call after ``draft_head``'s install. Returns whether this call changed the worker.

    Raises:
        ValueError: when the draft head cannot take it (``unsupported_reason``) or
            the merge over rank top-k keys orders ties differently from this
            torch's ``torch.topk`` (``tie_order_reason``).
    """
    worker = getattr(llm, "spec_worker", None)
    if worker is not None and isinstance(worker.__dict__.get(ATTRIBUTE), RankTopKExchange):
        return False
    reason = unsupported_reason(llm)
    if reason is None:
        reason = tie_order_reason(
            llm.draft_model.candidate_selector,
            worker.mapping.tp_size,
            transform=worker.__dict__.get(_draft_head.TRANSFORM_ATTR),
        )
    if reason is not None:
        raise ValueError(f"[musespark_open] wide_draft_topk cannot run here: {reason}")
    worker.__dict__[ATTRIBUTE] = RankTopKExchange(worker.__dict__[ATTRIBUTE])
    logger.info(
        f"[musespark_open] wide_draft_topk: draft steps of at least {MIN_ROWS} rows exchange "
        f"each rank's own top-{llm.draft_model.candidate_selector.top_k} keys (per-chunk stage "
        f"{CHUNK_WARPS} warps, merge launch {MERGE_LAUNCH})"
    )
    return True
