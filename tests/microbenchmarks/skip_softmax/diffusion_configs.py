"""Diffusion-shape sweep configs for the skip-softmax kernel microbench.

Mirrors the cubin family used by Wan2.2 T2V-A14B at 720x1280x81 frames per
the wan22-a14b-v1-vs-v2-calibration doc — bf16, head_dim=128, q_step=64,
kv_step=128, bidirectional mask.

The user explicitly said: ignore target_sparsity (which is the per-model
calibration knob). On random tensors threshold_scale_factor → achieved
sparsity is not the same curve, so we sweep threshold_scale_factor over a
log-spaced range wide enough to bracket 0% → ~99% achieved sparsity.
"""

from __future__ import annotations

from typing import List

from .llm_configs import FmhaConfig

DIFFUSION_THRESHOLDS = [
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


def diffusion_configs() -> List[FmhaConfig]:
    cfgs: List[FmhaConfig] = []
    for seq_len in (16384, 32768, 41472, 65536):
        cfgs.append(
            FmhaConfig(
                name=f"diff_bf16_s{seq_len}",
                dtype="bf16",
                batch=1,
                num_heads_q=24,
                num_heads_kv=24,  # Wan2.2 attn1 is MHA, not GQA
                head_size=128,
                seq_len_q=seq_len,
                seq_len_kv=seq_len,
                mask="bidirectional",
                threshold_sweep=list(DIFFUSION_THRESHOLDS),
            ))
    return cfgs
