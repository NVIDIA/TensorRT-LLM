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
"""Ragged verification layout: per-request verify lengths in one packed batch,
as a flat token axis plus a ``qo_indptr`` exclusive prefix sum.

Two invariants keep the pipeline safe: ``verify_lens >= 1`` for every request
(position 0 carries the bonus/anchor token; a request that verifies nothing
never progresses), and ``sum(verify_lens) == graph_num_tokens`` after
:meth:`RaggedVerifyLayout.fill_bucket` -- ``seq_lens.sum()`` becomes
``attn_metadata.num_tokens``, which is all-gathered across attention-DP ranks
and drives the MoE's chunk count, and an under-filled bucket desynchronizes
them without raising.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

__all__ = [
    "RaggedVerifyLayout",
    "build_qo_indptr",
    "row_ids_from_lens",
    "scatter_ragged_to_padded",
    "count_accepted_ragged",
    "fill_padded_rows_onehot",
    "ragged_gather_index_lists",
    "RaggedPadSplit",
    "resolve_ragged_pad_split",
    "fill_bucket_device",
    "build_row_maps_device",
]


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
        graph_num_tokens: int,
        total_verify_tokens: Optional[int] = None,
    ) -> "RaggedVerifyLayout":
        """Pack ``[bs]`` lengths against an already selected graph token count."""
        lens = verify_lens.to(torch.int32)
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
        """Pad rows to ``padded_bs`` and tokens to exactly ``graph_num_tokens``
        (a short batch desynchronizes attention from the MoE; see the module
        docstring). Slack goes to real requests first (up to ``max_verify_len``),
        then to pad rows; every pad row needs at least one token or its empty
        ``qo_indptr`` range breaks per-row slicing. Raises when the bucket
        cannot be hit exactly.
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
        # Round-robin: spread extra tokens across real rows (not one request's
        # low-survival tail), then spill to pad rows.
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
        """Assert the layout invariants (host-syncing: tests/debug only, never
        the hot path).

        ``exact_fill`` additionally requires the token count to hit the bucket
        exactly -- the post-:meth:`fill_bucket` contract for anything heading
        into a captured graph; a short batch raises nowhere downstream.
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
class RaggedPadSplit:
    """A shared pad-row window and the token target left for real rows."""

    pad_len: int
    real_target: int


def resolve_ragged_pad_split(
    *,
    bucket: int,
    num_real_requests: int,
    total_real_tokens: int,
    padded_bs: int,
    max_verify_len: int,
    fixed_pad_len: Optional[int] = None,
) -> Optional[RaggedPadSplit]:
    """Resolve one feasible decomposition of a captured ragged bucket.

    All pad rows share one request object and therefore one verify length. The
    returned split keeps that length in ``[1, max_verify_len]`` and leaves the
    real rows a target between their current token floor and capacity.
    """
    num_pad_requests = int(padded_bs) - int(num_real_requests)
    if num_pad_requests < 0:
        raise ValueError("padded_bs cannot be smaller than num_real_requests")
    if max_verify_len < 1:
        raise ValueError("max_verify_len must be at least one")

    real_capacity = int(num_real_requests) * int(max_verify_len)
    real_floor = int(total_real_tokens)
    if num_pad_requests == 0:
        pad_len = 1
    elif fixed_pad_len is not None:
        pad_len = int(fixed_pad_len)
    else:
        lower = max(1, -(-(int(bucket) - real_capacity) // num_pad_requests))
        upper = min(
            int(max_verify_len),
            (int(bucket) - real_floor) // num_pad_requests,
        )
        if lower > upper:
            return None
        pad_len = lower

    if not 1 <= pad_len <= int(max_verify_len):
        return None
    real_target = int(bucket) - num_pad_requests * pad_len
    if not real_floor <= real_target <= real_capacity:
        return None
    return RaggedPadSplit(pad_len=pad_len, real_target=real_target)


def fill_bucket_device(
    verify_lens: torch.Tensor,
    *,
    num_real: torch.Tensor,
    graph_num_tokens: int,
    max_verify_len: int,
    pad_fill: Optional[int] = None,
) -> torch.Tensor:
    """Capture-safe :meth:`RaggedVerifyLayout.fill_bucket`: same allocation,
    computed entirely on device.

    The host round-robin decomposes into full cycles -- every row with
    headroom gains one token -- plus one final partial cycle where the first
    ``spare`` headroom rows in index order gain one. Each cycle is therefore
    ``grant = headroom & (cumsum(headroom) <= spare)``, and a row's headroom
    is at most ``max_verify_len - 1``, so a statically-unrolled loop of that
    many cycles per phase (real rows first, then pad rows) reproduces the
    host result token-for-token with no ``.item()`` or host sync.

    The device version cannot raise, so feasibility is the caller's contract,
    checkable from host-known quantities alone before launch:
    ``padded_bs <= graph_num_tokens <= padded_bs * max_verify_len`` and the
    scheduled budget small enough that real tokens plus one-per-pad-row fit
    the bucket. Under those bounds the result sums to exactly
    ``graph_num_tokens``.

    Args:
        verify_lens: ``[padded_bs]`` scheduled lengths; entries at or beyond
            ``num_real`` are ignored (pad rows restart from one token).
        num_real: 0-d integer tensor on the same device (host-known per step,
            but not a capture constant -- it changes across replays).
        graph_num_tokens: the captured token bucket (capture constant).
        max_verify_len: per-row ceiling (capture constant).
        pad_fill: when given, every pad row carries EXACTLY this many tokens
            and all slack goes to real rows. The host fit publishes this split
            (``ragged_pad_verify_len``) because its copy widths -- how many
            flat tokens belong to real rows -- must be known before launch;
            the device fill has to land on the same split or the host and
            device disagree about where real tokens end.
    Returns:
        ``[padded_bs]`` int32 filled lengths.
    """
    if verify_lens.dim() != 1:
        raise ValueError(f"verify_lens must be 1-D, got {tuple(verify_lens.shape)}")
    padded_bs = verify_lens.numel()
    device = verify_lens.device
    rows = torch.arange(padded_bs, device=device)
    is_real = rows < num_real
    lens = torch.where(
        is_real,
        verify_lens.to(torch.int32),
        torch.full(
            (padded_bs,), 1 if pad_fill is None else int(pad_fill), dtype=torch.int32, device=device
        ),
    )
    spare = graph_num_tokens - lens.sum()
    phases = (is_real,) if pad_fill is not None else (is_real, ~is_real)
    for phase_mask in phases:
        for _ in range(max(max_verify_len - 1, 0)):
            headroom = phase_mask & (lens < max_verify_len)
            in_cycle = torch.cumsum(headroom.to(torch.int64), dim=0) <= spare
            grant = (headroom & in_cycle).to(torch.int32)
            lens = lens + grant
            spare = spare - grant.sum()
    return lens


def build_row_maps_device(
    verify_lens: torch.Tensor,
    *,
    graph_num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``(req_idx, kv_correction)`` from filled lengths, on device.

    The token-major row maps prepare() stages from host lists decompose into
    a pure function of the filled ``verify_lens``: token ``t`` of a request
    with window ``v`` at within-request offset ``o`` attends KV extent
    ``kv_len - v + 1 + o``, i.e. correction ``o - v + 1``. Composing the
    returned correction with a ``kv_lens`` gather is exactly
    ``refresh_ragged_row_kv_lens``, so the caller reuses that op unchanged.

    ``verify_lens`` must already sum to ``graph_num_tokens``
    (post-:func:`fill_bucket_device`); the explicit total keeps
    ``repeat_interleave`` sync-free.

    Returns:
        ``req_idx``: ``[graph_num_tokens]`` int64 owning-row index.
        ``correction``: ``[graph_num_tokens]`` int32, in ``[1 - v, 0]``.
    """
    indptr = build_qo_indptr(verify_lens).to(torch.long)
    req_idx = row_ids_from_lens(verify_lens, total=graph_num_tokens)
    offset = torch.arange(graph_num_tokens, device=verify_lens.device) - indptr[req_idx]
    correction = (offset - verify_lens.to(torch.long)[req_idx] + 1).to(torch.int32)
    return req_idx, correction


def ragged_gather_index_lists(
    slots: Sequence[int], counts: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Row/column index lists for gathering a ragged block from a
    ``[num_slots, max_width]`` tensor.

    ``slots[i]`` contributes ``counts[i]`` entries from columns
    ``0..counts[i]-1``, concatenated in batch order; with a constant ``counts``
    this matches the uniform ``tensor[slots, :width]`` gather.
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


def row_ids_from_lens(verify_lens: torch.Tensor, *, total: int) -> torch.Tensor:
    """``[bs]`` lengths -> ``[total]`` owning-request id for each packed token.

    ``total`` (``sum(verify_lens)``) is mandatory: without ``output_size``,
    ``repeat_interleave`` with device-resident repeats syncs to size its
    output, which is illegal inside the captured graph.
    """
    return torch.repeat_interleave(
        torch.arange(verify_lens.numel(), device=verify_lens.device),
        verify_lens.to(torch.long),
        output_size=int(total),
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

    Sync-free (capturable): token ``t`` goes to request ``row_ids[t]`` at
    column ``t - qo_indptr[row_ids[t]]``, and ``flat.shape[0]`` supplies the
    ``output_size`` :func:`row_ids_from_lens` needs.
    """
    bs = int(verify_lens.numel())
    rows = row_ids_from_lens(verify_lens, total=flat.shape[0])
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

    Padding rows are never read, but a rejection-sampling kernel still walks
    them and sampling an all-zero row can index out of bounds; all mass goes
    to token 0, which affects no position that is read. Modifies ``probs`` in
    place and returns it.
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

    Positions at or beyond a request's ``verify_len`` hold stale padding and
    must be masked to "no match" before the cumulative product, or a stale
    slot can credit an acceptance the target never made.

    Args:
        draft_tokens: ``[bs, max_len]`` padded drafted tokens.
        target_tokens: ``[bs, max_len]`` padded target tokens.
        verify_lens: ``[bs]`` positions actually verified per request.
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
