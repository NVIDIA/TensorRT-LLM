# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ragged verification layout: per-request verify lengths in one packed batch.

Uniform-K scheduling picks one verify length for the whole batch. That leaves
real budget on the table, because requests differ a lot in how deep their draft
survives: giving a request with survival ``0.55, 0.22, 0.07`` the same window as
one sitting at ``0.98, 0.95, 0.91`` wastes verification on the first and starves
the second. Ragged verification gives each request its own length.

The packing follows the shape every variable-length attention path already uses:
a flat token axis plus an index pointer.

    verify_lens      [bs]      per-request number of verified positions
    qo_indptr        [bs + 1]  exclusive prefix sum; request r owns
                               [qo_indptr[r], qo_indptr[r + 1])
    extend_start_loc [bs]      == qo_indptr[:-1], the offset form some
                               attention backends want directly

Two invariants make the rest of the system safe:

* ``verify_lens >= 1`` for every request. Position 0 carries the bonus/anchor
  token; a request that verifies nothing makes no progress at all and would
  stall forever.
* ``sum(verify_lens) == graph_num_tokens`` after :meth:`fill_bucket`. The batch
  is padded up to a captured CUDA-graph token bucket, so the packed shape is one
  of a small fixed set even though the per-request split is arbitrary. That is
  what lets a graph capture a ragged batch: the *shape* is constant, the
  raggedness lives in the *contents* of ``verify_lens`` / ``qo_indptr``.

Why the token count has to land exactly on the bucket
-----------------------------------------------------
``seq_lens.sum()`` becomes ``attn_metadata.num_tokens``, which is what gets
all-gathered across attention-DP ranks and drives the MoE's chunk count. If the
packed token count under-reports the rows actually in the batch, attention and
the MoE disagree about how many tokens are in flight -- and nothing raises.

Reclaiming the bucket padding
-----------------------------
Rounding up to a bucket would normally waste the difference. :meth:`fill_bucket`
spends it in two stages: real requests first (up to ``max_verify_len``, so those
tokens verify real draft positions the step is already paying for), then pad rows
added to reach the captured batch size. Same idea as SGLang's
``align_verify_tokens_to_graph_tier``.

The real requests cannot always absorb everything -- a request verifies at most
``max_verify_len`` positions, so the batch's capacity is
``padded_bs * max_verify_len``. When the bucket exceeds what the real requests
can take, the remainder must go to pad rows; when it exceeds even the padded
capacity, :meth:`fill_bucket` raises rather than returning a short batch.
"""

import bisect
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

__all__ = [
    "RaggedVerifyLayout",
    "build_qo_indptr",
    "round_up_to_bucket",
    "row_ids_from_lens",
    "scatter_ragged_to_padded",
    "count_accepted_ragged",
    "fill_padded_rows_onehot",
    "ragged_gather_index_lists",
    "RaggedCaptureShape",
    "choose_ragged_capture_shape",
]


def round_up_to_bucket(total: int, buckets: Sequence[int]) -> int:
    """Smallest captured bucket that fits ``total``.

    Raises rather than clamping: a batch that exceeds the largest captured
    bucket has no graph, and silently running it eager costs far more than the
    trimming saves. Callers must reject such a batch *before* selecting a
    layout (see :func:`exceeds_captured_buckets`).
    """
    if not buckets:
        raise ValueError("round_up_to_bucket requires a non-empty bucket list")
    if total > buckets[-1]:
        raise ValueError(
            f"total {total} exceeds the largest captured bucket {buckets[-1]}; "
            f"the caller must reject this batch before building a layout"
        )
    return buckets[bisect.bisect_left(buckets, total)]


def exceeds_captured_buckets(total: int, buckets: Sequence[int]) -> bool:
    """Whether ``total`` has no captured bucket (check before building a layout)."""
    return bool(buckets) and total > buckets[-1]


def build_qo_indptr(verify_lens: torch.Tensor) -> torch.Tensor:
    """``[bs]`` lengths -> ``[bs + 1]`` exclusive prefix sum, on the same device.

    Kept as a tensor op with no host sync so it can run inside a captured graph.
    """
    if verify_lens.dim() != 1:
        raise ValueError(f"verify_lens must be 1-D, got {tuple(verify_lens.shape)}")
    lens = verify_lens.to(torch.int32)
    out = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=lens.device)
    torch.cumsum(lens, dim=0, out=out[1:])
    return out


@dataclass
class RaggedVerifyLayout:
    """One packed ragged verify batch.

    Attributes:
        verify_lens: ``[bs]`` int32, each >= ``min_verify_len``.
        qo_indptr: ``[bs + 1]`` int32 exclusive prefix sum of ``verify_lens``.
        extend_start_loc: ``[bs]`` int32, ``qo_indptr[:-1]``.
        graph_num_tokens: the captured token bucket this batch is padded to.
        total_verify_tokens: ``sum(verify_lens)``, host-side when known.
        num_real_requests: how many leading rows are real requests. Rows beyond
            it are padding added to reach the captured batch size; their outputs
            must be discarded. Equals ``bs`` until :meth:`fill_bucket` pads.
    """

    verify_lens: torch.Tensor
    qo_indptr: torch.Tensor
    extend_start_loc: torch.Tensor
    graph_num_tokens: int
    total_verify_tokens: Optional[int] = None
    num_real_requests: Optional[int] = None

    @property
    def bs(self) -> int:
        return int(self.verify_lens.shape[0])

    @property
    def num_pad_requests(self) -> int:
        return 0 if self.num_real_requests is None else self.bs - self.num_real_requests

    @classmethod
    def from_verify_lens(
        cls,
        verify_lens: torch.Tensor,
        *,
        graph_num_tokens: Optional[int] = None,
        buckets: Optional[Sequence[int]] = None,
        total_verify_tokens: Optional[int] = None,
    ) -> "RaggedVerifyLayout":
        """Pack ``[bs]`` per-request lengths, rounding up to a captured bucket.

        ``graph_num_tokens`` may be given directly (the usual path -- the caller
        already knows the bucket because it also had to pick the graph), or
        derived from ``buckets`` plus a host-side ``total_verify_tokens``.
        Deriving it from the device tensor would need a sync.
        """
        lens = verify_lens.to(torch.int32)
        if graph_num_tokens is None:
            if buckets is None or total_verify_tokens is None:
                raise ValueError(
                    "give graph_num_tokens, or buckets + total_verify_tokens; "
                    "deriving the total from the device tensor would sync"
                )
            graph_num_tokens = round_up_to_bucket(int(total_verify_tokens), buckets)
        indptr = build_qo_indptr(lens)
        return cls(
            verify_lens=lens,
            qo_indptr=indptr,
            extend_start_loc=indptr[:-1],
            graph_num_tokens=int(graph_num_tokens),
            total_verify_tokens=(
                int(total_verify_tokens) if total_verify_tokens is not None else None
            ),
        )

    def fill_bucket(
        self, *, max_verify_len: int, padded_bs: Optional[int] = None
    ) -> "RaggedVerifyLayout":
        """Pad the batch to the captured shape, spending the slack on real work.

        Two things have to line up before the batch can run:

        * the row count must equal the captured batch size (``padded_bs``), and
        * the token count must equal the captured bucket
          (``graph_num_tokens``) -- because ``seq_lens.sum()`` becomes
          ``attn_metadata.num_tokens``, which is what gets all-gathered and
          drives the MoE's chunk count. If it under-reports, attention and MoE
          disagree about how many tokens are in flight.

        So the slack is distributed in two stages:

        1. **Real requests first**, up to ``max_verify_len``. The step is
           already paying for the bucket, so these tokens verify real draft
           positions instead of nothing.
        2. **Pad rows** take whatever is left. They exist only to reach
           ``padded_bs``; every pad row needs at least one token because an
           empty range breaks ``qo_indptr``'s per-row slicing.

        Raises when the bucket cannot be hit exactly -- silently returning a
        short batch is what would desynchronize attention from the MoE.
        """
        if self.total_verify_tokens is None:
            raise ValueError("fill_bucket needs the host-side total_verify_tokens")
        real: List[int] = [int(v) for v in self.verify_lens.tolist()]
        n_real = len(real)
        padded_bs = n_real if padded_bs is None else int(padded_bs)
        if padded_bs < n_real:
            raise ValueError(
                f"padded_bs {padded_bs} < real request count {n_real}: the "
                f"captured batch size cannot hold this batch"
            )
        n_pad = padded_bs - n_real

        # Every pad row must carry at least one token.
        baseline = sum(real) + n_pad
        capacity = padded_bs * max_verify_len
        if baseline > self.graph_num_tokens:
            raise ValueError(
                f"bucket {self.graph_num_tokens} is too small: {sum(real)} real "
                f"tokens plus {n_pad} pad rows already needs {baseline}"
            )
        if self.graph_num_tokens > capacity:
            raise ValueError(
                f"bucket {self.graph_num_tokens} exceeds what {padded_bs} rows can "
                f"absorb at max_verify_len={max_verify_len} ({capacity}); pick a "
                f"denser bucket grid or a larger padded_bs"
            )

        lens = real + [1] * n_pad
        spare = self.graph_num_tokens - baseline
        # Round-robin over real rows first -- spreading keeps the extra
        # verification where survival is still plausible for several requests
        # instead of pushing one request deep into its low-survival tail. Only
        # once every real row is saturated does the remainder go to pad rows,
        # where it is genuinely wasted.
        for lo, hi in ((0, n_real), (n_real, padded_bs)):
            i = lo
            while spare > 0 and any(lens[j] < max_verify_len for j in range(lo, hi)):
                if lens[i] < max_verify_len:
                    lens[i] += 1
                    spare -= 1
                i = lo + (i - lo + 1) % max(hi - lo, 1)
        assert spare == 0, f"internal: {spare} tokens left after filling {lens}"

        filled = torch.tensor(lens, dtype=torch.int32, device=self.verify_lens.device)
        out = RaggedVerifyLayout.from_verify_lens(
            filled,
            graph_num_tokens=self.graph_num_tokens,
            total_verify_tokens=sum(lens),
        )
        out.num_real_requests = n_real
        return out

    def validate(
        self,
        *,
        min_verify_len: int = 1,
        max_verify_len: Optional[int] = None,
        exact_fill: bool = False,
    ) -> None:
        """Assert the invariants the rest of the pipeline relies on.

        ``exact_fill`` additionally requires the token count to hit the bucket
        exactly -- the post-:meth:`fill_bucket` contract. Check it on anything
        heading for a captured graph: a short batch does not raise anywhere
        downstream, it just makes ``seq_lens.sum()`` disagree with the row count
        the graph was captured for.

        Host-syncing, so this is for tests and debug builds -- not the hot path.
        """
        lens = [int(v) for v in self.verify_lens.tolist()]
        if not lens:
            raise ValueError("a ragged layout needs at least one request")
        if min(lens) < max(min_verify_len, 1):
            raise ValueError(
                f"every request must verify its anchor token "
                f"(verify_len >= {max(min_verify_len, 1)}), got {lens}"
            )
        if max_verify_len is not None and max(lens) > max_verify_len:
            raise ValueError(
                f"verify_len exceeds the drafted block: {max(lens)} > {max_verify_len}"
            )
        total = sum(lens)
        if self.total_verify_tokens is not None and total != self.total_verify_tokens:
            raise ValueError(f"total_verify_tokens {self.total_verify_tokens} != sum {total}")
        if total > self.graph_num_tokens:
            raise ValueError(
                f"packed tokens {total} exceed the captured bucket {self.graph_num_tokens}"
            )
        if exact_fill and total != self.graph_num_tokens:
            raise ValueError(
                f"packed tokens {total} do not fill the captured bucket "
                f"{self.graph_num_tokens}; seq_lens.sum() would disagree with "
                f"the row count the graph was captured for"
            )
        expected = build_qo_indptr(self.verify_lens)
        if not torch.equal(self.qo_indptr, expected):
            raise ValueError("qo_indptr is not the prefix sum of verify_lens")


@dataclass(frozen=True)
class RaggedCaptureShape:
    """The ``(rows, tokens)`` pair a ragged batch is padded up to.

    Attributes:
        padded_bs: captured batch size (number of rows, real + pad requests).
        bucket: captured token count, i.e. ``sum(verify_lens)`` after padding.
    """

    padded_bs: int
    bucket: int


def choose_ragged_capture_shape(
    *,
    num_real_requests: int,
    total_verify_tokens: int,
    bs_buckets: Sequence[int],
    token_buckets: Sequence[int],
    peer_stats: Optional[Sequence[Sequence[int]]] = None,
) -> RaggedCaptureShape:
    """Pick the captured ``(padded_bs, bucket)`` for a ragged batch.

    Uniform speculation needs one number -- the batch size -- because the token
    count follows from it (``bs * (K + 1)``). Ragged verification breaks that
    link: two batches with the same row count can hold very different token
    totals, so the captured shape is genuinely two-dimensional and both axes
    have to be agreed on before the graph is looked up.

    Under attention DP every rank must land on the *same* shape or the ranks
    replay different graphs and their collectives diverge. Ranks do not have
    equal batches, so agreement cannot come from each rank rounding its own
    numbers; it comes from rounding a reduction. Pass every rank's
    ``(num_real_requests, total_verify_tokens)`` as ``peer_stats`` (including
    this rank's own) and each rank computes an identical answer:

        padded_bs = round_up_bs(max_r  num_real_r)
        bucket    = round_up_bucket(padded_bs + max_r  slack_r)

    where ``slack_r = total_r - num_real_r`` is the drafted-position count on
    rank ``r``. The ``padded_bs +`` term is the floor every row contributes
    (each request verifies at least its bonus position), so the bucket is large
    enough for the widest rank's drafts on top of the widest rank's rows.

    Raises when no captured bucket fits, rather than clamping: a batch with no
    graph runs eager, which costs far more than the trimming saves, so the
    caller has to see it and shrink the batch instead.
    """
    if not bs_buckets:
        raise ValueError("choose_ragged_capture_shape requires bs_buckets")
    if num_real_requests < 0 or total_verify_tokens < num_real_requests:
        raise ValueError(
            f"total_verify_tokens {total_verify_tokens} cannot be below "
            f"num_real_requests {num_real_requests} (every request verifies "
            f"at least one position)"
        )

    stats = list(peer_stats) if peer_stats else [(num_real_requests, total_verify_tokens)]
    max_real = max(int(real) for real, _ in stats)
    max_slack = max(int(total) - int(real) for real, total in stats)

    sorted_bs = sorted({int(b) for b in bs_buckets})
    if max_real > sorted_bs[-1]:
        raise ValueError(
            f"{max_real} requests exceed the largest captured batch size "
            f"{sorted_bs[-1]}; the caller must shrink the batch"
        )
    padded_bs = sorted_bs[bisect.bisect_left(sorted_bs, max_real)]

    needed = padded_bs + max_slack
    bucket = round_up_to_bucket(needed, sorted({int(t) for t in token_buckets}))
    return RaggedCaptureShape(padded_bs=padded_bs, bucket=bucket)


def ragged_gather_index_lists(
    slots: Sequence[int], counts: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Row/column index lists for gathering a ragged block from a
    ``[num_slots, max_width]`` tensor.

    ``slots[i]`` contributes ``counts[i]`` entries taken from columns
    ``0..counts[i]-1``, concatenated in batch order. This is the ragged
    replacement for the ``tensor[slots, :width]`` strided gather the overlap
    scheduler uses when every request verifies the same number of positions;
    with a constant ``counts`` the two produce the identical flat sequence.
    """
    if len(slots) != len(counts):
        raise ValueError(
            f"slots and counts must be the same length, got {len(slots)} and {len(counts)}"
        )
    rows: List[int] = []
    cols: List[int] = []
    for slot, count in zip(slots, counts):
        if count < 0:
            raise ValueError(f"negative gather count {count} for slot {slot}")
        rows.extend([slot] * count)
        cols.extend(range(count))
    return rows, cols


def row_ids_from_lens(verify_lens: torch.Tensor) -> torch.Tensor:
    """``[bs]`` lengths -> ``[total]`` owning-request id for each packed token.

    ``repeat_interleave`` with a *tensor* of repeats is what makes the packing
    ragged; the uniform paths use a scalar repeat and get a fixed stride.
    """
    return torch.repeat_interleave(
        torch.arange(verify_lens.numel(), device=verify_lens.device), verify_lens.to(torch.long)
    )


def scatter_ragged_to_padded(
    flat: torch.Tensor,
    *,
    verify_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_len: int,
    pad_value: int = 0,
) -> torch.Tensor:
    """Unpack a flat ragged batch into ``[bs, max_len, ...]`` with padding.

    The verify path is packed (one flat token axis) because that is what the
    attention kernels want, but the acceptance logic is naturally rectangular
    (compare draft vs target position by position). This bridges the two
    without a host sync: token ``t`` belongs to request ``row_ids[t]`` at column
    ``t - qo_indptr[row_ids[t]]``.
    """
    bs = int(verify_lens.numel())
    rows = row_ids_from_lens(verify_lens)
    cols = torch.arange(flat.shape[0], device=flat.device) - qo_indptr.to(torch.long)[rows]
    out = flat.new_full((bs, max_len, *flat.shape[1:]), pad_value)
    out[rows, cols] = flat
    return out


def fill_padded_rows_onehot(
    probs: torch.Tensor,
    *,
    verify_lens: torch.Tensor,
) -> torch.Tensor:
    """Make ``[bs, max_len, vocab]`` padding rows a valid one-hot distribution.

    Scattering a ragged batch into a rectangle leaves all-zero probability rows
    past each request's window. Those rows are never read -- the accepted count
    is clamped to the window -- but a rejection-sampling kernel still walks them
    when every real position was accepted, and sampling from an all-zero
    distribution can index out of bounds. Putting all the mass on token 0 keeps
    the kernel well-defined without affecting any position that is read.

    Modifies ``probs`` in place and returns it.
    """
    positions = torch.arange(probs.shape[1], device=probs.device)
    invalid = positions.unsqueeze(0) >= verify_lens.to(positions.dtype).unsqueeze(1)
    probs[..., 0] = torch.where(invalid, torch.ones_like(probs[..., 0]), probs[..., 0])
    return probs


def count_accepted_ragged(
    *,
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    verify_lens: torch.Tensor,
) -> torch.Tensor:
    """Accepted draft-token count per request, honouring per-request windows.

    Args:
        draft_tokens: ``[bs, max_len]`` padded drafted tokens.
        target_tokens: ``[bs, max_len]`` padded target tokens at the same
            positions.
        verify_lens: ``[bs]`` how many positions each request actually verified.

    Positions at or beyond a request's ``verify_len`` were never sent to the
    target, so whatever sits there is stale padding. Masking them to "no match"
    before the cumulative product makes the run of matches stop at the window
    edge -- without the mask a stale slot could compare equal and credit an
    acceptance the target never made, which is a silent correctness bug rather
    than a throughput one.
    """
    if draft_tokens.shape != target_tokens.shape:
        raise ValueError(
            f"draft {tuple(draft_tokens.shape)} and target "
            f"{tuple(target_tokens.shape)} must have the same padded shape"
        )
    max_len = draft_tokens.shape[1]
    positions = torch.arange(max_len, device=draft_tokens.device)
    valid = positions.unsqueeze(0) < verify_lens.to(positions.dtype).unsqueeze(1)
    match = (draft_tokens == target_tokens) & valid
    return torch.cumprod(match.int(), dim=-1).sum(1)
