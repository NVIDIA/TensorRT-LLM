# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-shape sweep configs for the skip-softmax kernel microbench.

Mirrors the "Performance Benchmark" table in
docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md.

Model proxy: Qwen3-30B-A3B GQA (q_heads=64, kv_heads=4, head_dim=128).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FmhaConfig:
    """A single fmha.exe invocation shape."""

    name: str
    dtype: str  # "bf16" | "fp16" | "e4m3"
    batch: int
    num_heads_q: int
    num_heads_kv: int
    head_size: int
    seq_len_q: int  # q seq length (1 for decode)
    seq_len_kv: int  # kv seq length
    mask: str  # "causal" | "bidirectional"
    threshold_sweep: List[float] = field(default_factory=list)


# Threshold sweeps lifted directly from blog 16 §"Performance Benchmark"
# (Qwen3-30B columns), which were calibrated to span 0% -> 90% target sparsity
# on real workloads. Random data does not produce the same achieved sparsity,
# but the magnitude span is informative for kernel-level characterisation.
# Clean log-spaced integer thresholds - chosen for good achieved-sparsity
# coverage on random data (the kernel skip predicate hits its transition
# regime between ~10^4 and ~10^6 at head_dim=128 bf16).
PREFILL_THRESHOLDS = [
    0.0,
    1.0,
    10.0,
    100.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
    300_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
]

DECODE_THRESHOLDS = list(PREFILL_THRESHOLDS)


def llm_configs() -> List[FmhaConfig]:
    cfgs: List[FmhaConfig] = []
    # Prefill: bs=1, q==kv∈{16k,64k}, dtype∈{bf16,fp8}, causal mask, GQA(64/4)
    for seq_len in (16384, 65536):
        for dtype in ("bf16", "e4m3"):
            cfgs.append(
                FmhaConfig(
                    name=f"llm_prefill_{dtype}_s{seq_len}",
                    dtype=dtype,
                    batch=1,
                    num_heads_q=64,
                    num_heads_kv=4,
                    head_size=128,
                    seq_len_q=seq_len,
                    seq_len_kv=seq_len,
                    mask="causal",
                    threshold_sweep=list(PREFILL_THRESHOLDS),
                )
            )
    # Decode: bs=8 (lower than blog 16's 64 - fmha.exe test-bench scratch
    # allocates O(B*S*S) and OOMs on H200 at higher batches), q=1,
    # kv∈{16k,64k}, dtype∈{bf16,fp8}, causal mask, GQA(64/4)
    for kv_len in (16384, 65536):
        for dtype in ("bf16", "e4m3"):
            cfgs.append(
                FmhaConfig(
                    name=f"llm_decode_{dtype}_kv{kv_len}",
                    dtype=dtype,
                    batch=8,
                    num_heads_q=64,
                    num_heads_kv=4,
                    head_size=128,
                    seq_len_q=1,
                    seq_len_kv=kv_len,
                    mask="causal",
                    threshold_sweep=list(DECODE_THRESHOLDS),
                )
            )
    return cfgs
