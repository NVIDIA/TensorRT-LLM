# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-SM work partition of the indexer's paged MQA logits (DeepGEMM's paged metadata)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

from ._fp8_paged_mqa_logits import SPLIT_KV
from ._fp8_paged_mqa_logits import num_sms as _device_sms

#: The indexer K cache's page size, which the schedule call takes as its block size.
_PAGE = 64


def _spec(context_lens) -> dict:
    return dict(context_lens=tuple(context_lens), block_kv=_PAGE)


class GlmPagedMqaLogitsSchedule(OpWrapper):
    """``out`` [num_sms + 1, 2] int32: where each persistent CTA of the paged MQA logits starts.

    Request ``r`` with ``context_lens[r]`` keys is ``ceil(context_lens[r] / 256)`` units of work,
    one per 256-key split (two 128-key compute blocks, one per math warp group), numbered request by
    request. With ``total`` units, CTA ``s`` (0 <= s <= num_sms) starts at unit
    ``w = s * (total // num_sms) + min(s, total % num_sms)``: ``out[s] = (r, w - first unit of r)``
    for the request ``r`` holding unit ``w``, and ``(num_requests, 0)`` past the last unit. CTA
    ``s`` covers [out[s], out[s + 1]).

    Each request is one query atom (its whole verify window), so ``context_lens`` is passed as
    [num_requests, 1]. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="c1 key shard: one request, a full 33,152-key shard", spec=_spec([33_152])),
        Cell(why="an empty shard: one request of one key", spec=_spec([1])),
        Cell(why="one split per unit edge: 256 keys exactly", spec=_spec([256])),
        Cell(
            why="more splits than SMs: the whole 1M-key context on one rank",
            spec=_spec([1_048_596]),
        ),
        Cell(
            why="four requests, one shorter than a split",
            spec=_spec([3_000, 70_000, 129, 40_000]),
        ),
    )

    note = """
    Metadata consumed: none. The target passes the shard's context length (``hist_shard_meta``'s
    ``shard_ctx``) or the generation requests' indexer KV lengths, the page size (64) as
    ``block_kv``, the device's SM count (the logits kernel runs one CTA per SM) and a persistent
    [num_sms + 1, 2] int32 ``out`` (captured CUDA graphs read it). The schedule kernel waits on
    ``griddepcontrol.wait`` before it reads the lengths.

    ``block_kv`` must be 32, 64 or 128; it does not change the 256-key split on this
    architecture.
    """

    def __call__(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        out.copy_(
            get_paged_mqa_logits_metadata(context_lens.view(-1, 1), block_kv, num_sms),
            non_blocking=True,
        )
        return out

    def reference(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        """The partition, in Python integers; ``out`` only gives the result's device."""
        work = [-(-ctx // SPLIT_KV) for ctx in context_lens.tolist()]
        prefix = []
        total = 0
        for units in work:
            total += units
            prefix.append(total)
        quota, rem = divmod(total, num_sms)
        rows = []
        for sm in range(num_sms + 1):
            w = sm * quota + min(sm, rem)
            request = next((r for r, end in enumerate(prefix) if end > w), len(work))
            if request < len(work):
                rows.append((request, w - (prefix[request - 1] if request else 0)))
            else:
                rows.append((len(work), 0))
        return torch.tensor(rows, dtype=torch.int32, device=out.device)

    def is_valid(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> None:
        # Each element is read as one request's length.
        assert context_lens.dim() == 1, "context_lens must be [num_requests]"
        # The schedule is right for any SM count, but the logits kernel launches one CTA per SM
        # and reads entries [s] and [s + 1]: another count leaves keys unscored or reads past the
        # schedule.
        device = context_lens.device.index if context_lens.device.index is not None else 0
        assert num_sms == _device_sms(device), (
            f"num_sms={num_sms} differs from the device's SM count, the logits kernel's CTA count"
        )
        # A wider int type would be cast silently by the copy.
        assert out.dtype == torch.int32, f"out must be int32, got {out.dtype}"


paged_mqa_logits_schedule = GlmPagedMqaLogitsSchedule()
