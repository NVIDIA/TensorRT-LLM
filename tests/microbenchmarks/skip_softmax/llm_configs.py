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
# (Qwen3-30B columns), which were calibrated to span 0% → 90% target sparsity
# on real workloads. Random data does not produce the same achieved sparsity,
# but the magnitude span is informative for kernel-level characterisation.
PREFILL_THRESHOLDS = [
    0.0,
    18.76,
    44.37,
    104.97,
    248.40,
    587.18,
    1390.63,
    3293.04,
    7799.91,
    18471.56,
]

DECODE_THRESHOLDS = [
    0.0,
    0.32,
    0.86,
    2.30,
    6.17,
    16.52,
    44.26,
    118.62,
    317.99,
    852.20,
]


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
                ))
    # Decode: bs=64, q=1, kv∈{16k,64k}, dtype∈{bf16,fp8}, causal mask, GQA(64/4)
    for kv_len in (16384, 65536):
        for dtype in ("bf16", "e4m3"):
            cfgs.append(
                FmhaConfig(
                    name=f"llm_decode_{dtype}_kv{kv_len}",
                    dtype=dtype,
                    batch=64,
                    num_heads_q=64,
                    num_heads_kv=4,
                    head_size=128,
                    seq_len_q=1,
                    seq_len_kv=kv_len,
                    mask="causal",
                    threshold_sweep=list(DECODE_THRESHOLDS),
                ))
    return cfgs
