# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokens-only FULL sampling for gathered Rubin GLM verify logits.

Logits may be finite or -inf (masked), with a finite scaled maximum in each row. Temperature,
filters and RNG arrays are completed step inputs before the PDL logits producer.
The state is stream ordered and cannot serve concurrent CUDA streams. Native
scratch and counters are real mutated tensors; no indirect allocation is hidden.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

VOCAB_SIZE = 154880
MAX_ROWS = 24


def workspace_bytes(rows: int, vocab: int = VOCAB_SIZE) -> int:
    return ((4 + 32) * ((vocab + 511) // 512) + 4096) * rows * 4


def _supported(device, max_rows, logits, temperatures, top_ks, top_ps, min_ps, seed, offset):
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device != device
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or not 1 <= logits.shape[0] <= max_rows
        or logits.shape[1] != VOCAB_SIZE
        or not logits.is_contiguous()
        or logits.data_ptr() % 16
    ):
        return False
    rows = logits.shape[0]
    for value, dtype, count in (
        (temperatures, torch.float32, rows),
        (top_ks, torch.int32, rows),
        (top_ps, torch.float32, rows),
        (min_ps, torch.float32, rows),
        (seed, torch.int64, 1),
        (offset, torch.int64, 1),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.device != device
            or value.dtype != dtype
            or value.ndim != 1
            or not value.is_contiguous()
            or value.numel() < count
        ):
            return False
    return seed.numel() == offset.numel() and (seed.numel() == 1 or seed.numel() >= rows)


class GlmVerifyHeadSample(OpWrapper):
    """Sample FULL weights after min-p, tie-preserving top-k and inclusive top-p.

    The reference returns the categorical distribution, not an unrelated RNG's
    sampled token. Tests compare support/distribution and this op's own determinism.
    Different reduction order/random consumption can change IDs versus the shared
    sampler, even with identical per-row Philox initialization.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(
            why=f"GLM verify rows={rows}, FULL filters and per-token RNG",
            spec=dict(rows=rows, vocab=VOCAB_SIZE),
        )
        for rows in (1, 6, 12, 18, 24)
    )
    note = "Masked -inf logits allowed with finite row maxima; no cross-algorithm ID equality."

    def __call__(
        self, logits, temperatures, top_ks, top_ps, min_ps, seed, offset, row_counters, workspace
    ):
        return torch.ops.trtllm.glm_verify_head_sample_tokens(
            logits, temperatures, top_ks, top_ps, min_ps, seed, offset, row_counters, workspace
        )

    def reference(
        self, logits, temperatures, top_ks, top_ps, min_ps, seed, offset, row_counters, workspace
    ):
        """Independent categorical probabilities; RNG and scratch do not affect the law.

        Weight scaling is FP32; sorted float64 accumulation makes boundary tests
        use their specified mathematical mass, avoiding a duplicate CUDA reduction.
        """
        inv_temp = torch.where(
            temperatures[: len(logits)] > 0, 1.0 / temperatures[: len(logits)], 1.0 / 1.0e-6
        )
        scaled = logits * inv_temp[:, None]
        weights = (scaled - scaled.amax(dim=-1, keepdim=True)).exp()
        result = []
        for row, weight in enumerate(weights):
            floor = min_ps[row].clamp(0, 1)
            k = int(top_ks[row])
            if 0 < k < weight.numel():
                floor = torch.maximum(floor, weight.topk(k).values[-1])
            weight = torch.where(weight >= floor, weight, 0.0)
            if float(top_ps[row]) < 1:
                ordered = weight.sort(descending=True).values
                cumulative = ordered.double().cumsum(0)
                # First index of each tie gives the mass of STRICTLY greater weights.
                first = torch.searchsorted(-ordered, -weight, right=False)
                prefix = torch.cat((cumulative.new_zeros(1), cumulative))
                above = prefix[first]
                weight = torch.where(above <= top_ps[row].double() * cumulative[-1], weight, 0.0)
            result.append(weight / weight.sum())
        return torch.stack(result)

    def is_valid(
        self, logits, temperatures, top_ks, top_ps, min_ps, seed, offset, row_counters, workspace
    ):
        assert logits.is_cuda and _supported(
            logits.device, MAX_ROWS, logits, temperatures, top_ks, top_ps, min_ps, seed, offset
        )
        assert not torch.isnan(logits).any() and not torch.isposinf(logits).any()
        assert torch.isfinite(temperatures).all()
        inv_temp = torch.where(
            temperatures[: len(logits)] > 0, 1.0 / temperatures[: len(logits)], 1.0 / 1.0e-6
        )
        assert torch.isfinite((logits * inv_temp[:, None]).amax(dim=1)).all()
        assert torch.isfinite(top_ps).all() and ((top_ps >= 0) & (top_ps <= 1)).all()
        assert torch.isfinite(min_ps).all()
        assert row_counters.dtype == torch.int32 and row_counters.device == logits.device
        assert row_counters.ndim == 1 and row_counters.numel() >= len(logits)
        assert row_counters.is_contiguous() and not row_counters.any()
        assert workspace.dtype == torch.uint8 and workspace.device == logits.device
        assert workspace.ndim == 1 and workspace.is_contiguous()
        assert workspace.numel() >= workspace_bytes(len(logits)) and workspace.data_ptr() % 16 == 0


glm_verify_head_sample = GlmVerifyHeadSample()


class VerifyHeadSampleState:
    """Persistent scratch allocated before profiling/capture; one ordered execution stream."""

    def __init__(self, device, max_rows: int = MAX_ROWS, vocab: int = VOCAB_SIZE):
        device = torch.device(device)
        if device.type != "cuda" or not 1 <= max_rows <= MAX_ROWS or vocab != VOCAB_SIZE:
            raise ValueError("GLM verify sampling requires CUDA, rows1..24 and vocabulary154880")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            raise ValueError("GLM verify sampling requires SM107")
        self.device = device
        self.max_rows = max_rows
        self.row_counters = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.workspace = torch.empty(workspace_bytes(max_rows), dtype=torch.uint8, device=device)

    def supported(self, logits, temperatures, top_ks, top_ps, min_ps, seed, offset) -> bool:
        # Shape/layout-only eligibility is stable under CUDA graph parameter updates.
        if torch.compiler.is_compiling():
            return False
        return _supported(
            self.device, self.max_rows, logits, temperatures, top_ks, top_ps, min_ps, seed, offset
        )

    def sample(self, logits, temperatures, top_ks, top_ps, min_ps, seed, offset):
        if not self.supported(logits, temperatures, top_ks, top_ps, min_ps, seed, offset):
            raise ValueError("unsupported GLM verify sampling inputs; select shared sampler")
        return glm_verify_head_sample(
            logits,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            self.row_counters,
            self.workspace,
        )
