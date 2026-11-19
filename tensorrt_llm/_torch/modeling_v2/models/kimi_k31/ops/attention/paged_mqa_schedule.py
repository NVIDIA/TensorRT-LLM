# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer: the per-SM schedule of the paged MQA logits kernels, written in place (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.paged_mqa_schedule import (  # noqa: E501
    MAX_ROWS,
    paged_mqa_schedule_into,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.get_paged_mqa_logits_metadata import (  # noqa: E501
    get_paged_mqa_logits_metadata,
)


class PagedMqaSchedule(OpWrapper):
    """The schedule ``get_paged_mqa_logits_metadata`` computes (DeepGEMM's paged MQA logits
    metadata), written straight into ``out`` (int32 [num_sms + 1, 2]) by one CuTe DSL launch;
    returns ``out``. ``context_lens`` is int32 [R, n] with request ``r``'s KV length in
    ``context_lens[r, n - 1]``; ``block_kv`` does not move the schedule.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = get_paged_mqa_logits_metadata.CELLS + (
        Cell(
            why="ll B1 at 190k: one [1, 1] row (the DSL schedule of every verify width)",
            spec=dict(rows=1, next_n=1, lens=(190_000, 190_000), block_kv=64, stagger=0),
        ),
        Cell(
            why="DEP16 B32 at 120k-140k with the w7 in-kernel split: 64 atom rows",
            spec=dict(rows=64, next_n=1, lens=(120_000, 140_000), block_kv=64, stagger=2),
        ),
        Cell(
            why="the largest row count served",
            spec=dict(rows=MAX_ROWS, next_n=1, lens=(1, 4000), block_kv=64, stagger=0),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    A drop-in for get_paged_mqa_logits_metadata at its call sites (the DSA metadata's
    on_update_kv_lens / prepare_scheduler_metadata): same arguments, same integers, but one
    launch writing the persistent buffer in place instead of DeepGEMM's kernel plus a copy, and
    no C++. At most MAX_ROWS (16384) schedule rows; lengths must be non-negative. One CTA of 1024
    threads; programmatic dependent launch (it waits before reading the lengths).
    """

    def __call__(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        return paged_mqa_schedule_into(context_lens, num_sms, out)

    def reference(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> torch.Tensor:
        """The partition in Python integers (the DeepGEMM entry's reference)."""
        return get_paged_mqa_logits_metadata.reference(context_lens, block_kv, num_sms, out)

    def is_valid(
        self, context_lens: torch.Tensor, block_kv: int, num_sms: int, out: torch.Tensor
    ) -> None:
        device = context_lens.device.index if context_lens.device.index is not None else 0
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        assert num_sms == sms, (
            f"num_sms={num_sms} differs from the device's SM count ({sms}), the logits "
            "kernels' CTA count"
        )
        assert out.dtype == torch.int32 and out.is_contiguous(), "out must be contiguous int32"
        assert tuple(out.shape) == (num_sms + 1, 2)
        assert context_lens.dtype == torch.int32 and context_lens.dim() == 2
        assert context_lens.stride(1) == 1 or context_lens.shape[1] == 1
        assert context_lens.shape[0] <= MAX_ROWS, f"more than {MAX_ROWS} schedule rows"
        if context_lens.numel():
            assert int(context_lens[:, -1].min()) >= 0, "a negative KV length"


paged_mqa_schedule = PagedMqaSchedule()
