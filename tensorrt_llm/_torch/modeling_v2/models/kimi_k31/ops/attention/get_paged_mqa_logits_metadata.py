# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer: the per-SM schedule of the paged MQA logits kernels (DeepGEMM's paged metadata).

Wraps the pybind ``tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata``, not a
``torch.ops.trtllm.*`` op: the DSA metadata calls the pybind directly.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata as _deep_gemm_paged_metadata

#: The block_kv the K3.1 DSA metadata passes, whatever the KV page size.
SCHEDULE_BLOCK_KV = 64

#: Keys per unit of schedule work on SM100-family GPUs (two 128-key compute blocks).
SPLIT_KV = 256


def _spec(rows, next_n=1, lens=(1, 1), **overrides) -> dict:
    """``rows`` schedule rows of ``next_n`` query tokens each, KV lengths drawn in ``lens``."""
    spec = dict(rows=rows, next_n=next_n, lens=tuple(lens), block_kv=SCHEDULE_BLOCK_KV, stagger=0)
    spec.update(overrides)
    return spec


class GetPagedMqaLogitsMetadata(OpWrapper):
    """Writes into ``out`` (int32 [num_sms + 1, 2]) where each persistent CTA of the paged MQA
    logits starts, and returns ``out``.

    ``context_lens`` is int32 [R, n]: R requests of n query tokens, request ``r``'s KV length in
    ``context_lens[r, n - 1]``. Request ``r`` is ``ceil(len_r / 256) * n`` units of work,
    numbered request by request. With ``total`` units, CTA ``s`` (0 <= s <= num_sms) starts at
    unit ``w = s * (total // num_sms) + min(s, total % num_sms)``: ``out[s] = (r * n, (w -
    first unit of r) // n)`` for the request ``r`` holding unit ``w`` (its first query token and
    256-key split), and ``(R * n, 0)`` past the last unit. CTA ``s`` covers ``[out[s],
    out[s + 1])``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one request at the top-k edge (2049 keys)", spec=_spec(1, lens=(2049, 2049))),
        Cell(why="one request of one key: a single unit of work", spec=_spec(1, lens=(1, 1))),
        Cell(why="256 keys exactly: one split", spec=_spec(1, lens=(256, 256))),
        Cell(
            why="decode, 8 requests of 3k-30k keys: fewer units than SMs",
            spec=_spec(8, lens=(3000, 30000)),
        ),
        Cell(
            why="decode, 256 requests of 1k-16k keys: many requests per CTA",
            spec=_spec(256, lens=(1000, 16000)),
        ),
        Cell(
            why="two requests of up to 256k keys: more splits than SMs",
            spec=_spec(2, lens=(200000, 262144)),
        ),
        Cell(
            why="the DSL in-kernel w5 split: 64 requests x 2 atom rows with staggered lengths",
            spec=_spec(128, lens=(12000, 16384), stagger=2),
        ),
        Cell(
            why="the DSL atom split of w8 at batch 256: 2048 schedule rows",
            spec=_spec(2048, lens=(2100, 9000)),
        ),
        Cell(
            why="DFlash2 w5 full-window schedule: [4, 5] lengths, 5 query tokens per request",
            spec=_spec(4, next_n=5, lens=(2045, 9000)),
        ),
        Cell(
            why="DFlash2 w8 full-window schedule at batch 64: [64, 8] lengths",
            spec=_spec(64, next_n=8, lens=(2040, 30000)),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this from the DSA metadata's on_update_kv_lens and the
    indexer's prepare_scheduler_metadata, and copies the result into a persistent buffer the
    captured logits kernels read: with the generation requests' indexer KV lengths as [B, 1]
    (scheduler_metadata_buffer, the CuTe DSL logits' schedule for every verify width), as
    [B, n] (scheduler_metadata_buffer_full_next_n, built whenever drafting is on, for DeepGEMM's
    paged logits), and as [B * atoms, 1] atom rows (scheduler_metadata_buffer_expanded, the
    DSL atom split). block_kv is always 64 and does not change the 256-key split; num_sms is
    the device's SM count.

    The DeepGEMM call allocates its result; ``out`` takes a copy so its address stays fixed.
    The schedule kernel waits on the preceding kernel (programmatic dependent launch) before it
    reads the lengths.
    """

    def __call__(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        out.copy_(_deep_gemm_paged_metadata(context_lens, block_kv, num_sms), non_blocking=True)
        return out

    def reference(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        """The partition in Python integers; ``out`` only gives the result's device. ``block_kv``
        is ignored."""
        num_requests, next_n = context_lens.shape
        lens = context_lens[:, next_n - 1].tolist()
        prefix, total = [], 0
        for length in lens:
            total += -(-length // SPLIT_KV) * next_n
            prefix.append(total)
        quota, rem = divmod(total, num_sms)
        rows = []
        for sm in range(num_sms + 1):
            w = sm * quota + min(sm, rem)
            lo, hi = 0, num_requests
            while lo < hi:
                mid = (lo + hi) // 2
                if prefix[mid] <= w:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < num_requests:
                first = prefix[lo - 1] if lo else 0
                rows.append((lo * next_n, (w - first) // next_n))
            else:
                rows.append((num_requests * next_n, 0))
        return torch.tensor(rows, dtype=torch.int32, device=out.device)

    def is_valid(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> None:
        # The schedule is right for any SM count, but the logits kernels launch one CTA per SM
        # and read entries [s] and [s + 1]: another count leaves keys unscored or reads past
        # the schedule.
        device = context_lens.device.index if context_lens.device.index is not None else 0
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        assert num_sms == sms, (
            f"num_sms={num_sms} differs from the device's SM count ({sms}), the logits "
            "kernels' CTA count"
        )
        # The copy would cast a wider integer type silently.
        assert out.dtype == torch.int32, f"out must be int32, got {out.dtype}"


get_paged_mqa_logits_metadata = GetPagedMqaLogitsMetadata()
