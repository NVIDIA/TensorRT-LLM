# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AGG_NVFP4_INDEXK decode reader: the pure-generation index scores of an NVFP4 index-K cache.

The vendored Q8KV4 decode indexer (``fmha_sm100_q8kv8``: E4M3 index-Q x NVFP4 index-K, the CuTe DSL kernel at two
index heads per rank) scores each request's history pages and writes them straight into the select layout
[index heads, pages, tokens] (``run_scores_hpr``), the layout of the FP8 path's ``max_score``; the served selector
then takes the top 16 exactly as on FP8 pages. The wrapper's ``plan()`` refuses CUDA graph capture, so its device
part (the per-request history-page prefix sum) runs here, inside the captured step, and the kernel is compiled before
warmup (``prebuild``). Scores are raw Q.K (no scale), as the FP8 scorers'.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

PAGE = 128


def _interface():
    from fmha_sm100_q8kv8 import q8_indexer_interface as qi

    return qi


class Nvfp4IndexDecodeScorer:
    """One per target; holds a wrapper per device and nothing step-dependent."""

    def __init__(self, num_heads: int):
        self.num_heads = int(num_heads)
        self._wrappers: Dict[torch.device, object] = {}

    def available(self) -> bool:
        try:
            qi = _interface()
        except ImportError:
            return False
        return hasattr(qi.BatchDecodeIndexerQ8KV4Wrapper, "run_scores_hpr") and hasattr(
            qi, "q8kv4_hpr_buffer"
        )

    def _wrapper(self, device: torch.device):
        w = self._wrappers.get(device)
        if w is None:
            w = _interface().BatchDecodeIndexerQ8KV4Wrapper(num_heads=self.num_heads)
            self._wrappers[device] = w
        return w

    def score(
        self,
        idx_q: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_len: int,
        workspace: torch.Tensor = None,
    ) -> torch.Tensor:
        """The scores of one decode step, [heads, C, B * query_len] fp32.

        ``idx_q`` [B * query_len, heads, 128] E4M3; ``cache`` [pages, 1, 128, 72] uint8; ``block_table`` [B, C] int32;
        ``seq_lens`` [B] int32. History pages are written; a row's local and later pages are left as allocated, which
        the selector never ranks.
        """
        qi = _interface()
        batch, cols = int(block_table.shape[0]), int(block_table.shape[1])
        device = idx_q.device
        _, view, kernel_out = qi.q8kv4_hpr_buffer(
            batch, cols, self.num_heads, int(query_len), device
        )
        if workspace is None:
            workspace = self.history_workspace(seq_lens, cols)
        w = self._wrapper(device)
        w._block_table = block_table
        w._seq_lens = seq_lens
        w._workspace = workspace
        w._query_len = int(query_len)
        w._scores = torch.empty(
            (batch, 0, 0), dtype=torch.float32, device=device
        )  # run_scores_hpr's row count
        w.run_scores_hpr(
            idx_q, cache.reshape(int(cache.shape[0]), PAGE, int(cache.shape[-1])), kernel_out
        )
        return view

    @staticmethod
    def history_workspace(seq_lens: torch.Tensor, cols: int) -> torch.Tensor:
        """The plan's device part: each request's history-page count (pages before its local one), prefix-summed, as
        the kernel's scheduler workspace (uint8 [(B + 1) * 4], int32 view). It depends on the step only (``seq_lens``
        and the table width), so a caller scoring every layer of one step may build it once and pass it to ``score``
        (NVFP4_STEP_CACHE); the values are the same as building it per call."""
        batch = int(seq_lens.shape[0])
        workspace = torch.empty(((batch + 1) * 4,), dtype=torch.uint8, device=seq_lens.device)
        scheduler = workspace.view(torch.int32)
        history = torch.div(seq_lens - 1, PAGE, rounding_mode="floor").clamp_(min=0, max=cols)
        scheduler[:1].zero_()
        torch.cumsum(history, dim=0, out=scheduler[1:])
        return workspace

    def prebuild(self, device: torch.device, query_lens) -> Tuple[int, ...]:
        """Compile the kernel for every decode query length before warmup and capture; returns them.

        The query length is the compile key (batch and pages are runtime); each runs once on a two-request step.
        """
        built = []
        for qlen in sorted({int(q) for q in query_lens}):
            batch, pages = 2, 4
            cache = torch.zeros((pages * batch, 1, PAGE, 72), dtype=torch.uint8, device=device)
            table = torch.arange(pages * batch, dtype=torch.int32, device=device).view(batch, pages)
            lens = torch.full((batch,), pages * PAGE, dtype=torch.int32, device=device)
            q = torch.zeros((batch * qlen, self.num_heads, 128), device=device).to(
                torch.float8_e4m3fn
            )
            self.score(q, cache, table, lens, qlen)
            built.append(qlen)
        torch.cuda.synchronize(device)
        return tuple(built)


__all__ = ["Nvfp4IndexDecodeScorer"]
