# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose qualified post-load H3 experiments without changing the round-G control."""

import sys

import breuse_gateup
import round4_attention_runtime as attention
import round4_gateup_kernel as gateup_kernel
import torch
from long_round3_runtime import Runtime as RoundGRuntime
from round4_cache import Patch as ConditioningCache
from round4_reciprocal_runtime import Runtime as ReciprocalRuntime


def supports_down(a, weight, x, mod, indices):
    """Keep the full metadata guards and two observed full-request lengths."""
    m = a.shape[0]
    return (
        m in (37776, 37820, 38169)
        and a.shape == (m, 14336)
        and weight.shape == (5376, 14336)
        and a.dtype == weight.dtype == torch.float8_e4m3fn
        and x.dtype == torch.bfloat16
        and x.numel() == m * 5376
        and a.is_contiguous()
        and weight.is_contiguous()
        and x.is_contiguous()
        and mod.ndim == 2
        and mod.shape[1] == 6 * 5376
        and mod.dtype in (torch.bfloat16, torch.float32)
        and mod.is_contiguous()
        and mod.data_ptr() % 32 == 0
        and mod.stride(0) * mod.element_size() % 32 == 0
        and indices.numel() == m
        and indices.is_contiguous()
        and indices.dtype in (torch.int32, torch.int64)
    )


class Runtime:
    """One frozen U1 pipeline; configure outside requests, with graphs disabled."""

    MODES = ("native", "g2", "kf_cache", "kf_cache_recip", "kf_cache_recip_attention")

    def __init__(self, pipeline, base=None):
        self.g2 = base if base is not None else RoundGRuntime(pipeline)
        self.g2.configure(False)
        self.original_gateup = breuse_gateup.kernel
        self.down_class = type(self.g2.down)
        self.original_down_supports = self.down_class.supports
        blocks = pipeline.transformer.transformer_blocks
        self.cache = ConditioningCache(pipeline.transformer)
        mixed = sys.modules.get("h3_mixed_layout")
        if mixed is not None and mixed.active():
            blocks = mixed.require_layout(pipeline).fp8_blocks
        self.reciprocals = ReciprocalRuntime(blocks)
        attention.candidate._CFG = dict(
            tile_n=256,
            cluster_shape_mn=(4, 1),
            num_ab_stage=4,
            swizzle_size=7,
            fallback_cluster_shape_mn=(2, 1),
            mma_tile_k=128,
        )
        self.attention = attention.Patch(blocks)
        self.mode = "native"
        self.configure("native")

    def configure(self, mode):
        """Switch immutable-model experiments; all cache/key work stays in requests."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown round-four mode: {mode}")
        self.cache.configure(False)
        self.reciprocals.configure("off")
        self.attention.configure(False)
        attention.STATE.update(enabled=False, validate=False, capture=False)
        self.g2.configure(mode != "native")
        enhanced = mode not in ("native", "g2")
        breuse_gateup.kernel = gateup_kernel if enhanced else self.original_gateup
        self.down_class.supports = staticmethod(
            supports_down if enhanced else self.original_down_supports
        )
        if mode == "kf_cache_recip_attention":
            self.attention.configure(True)
            attention.STATE.update(enabled=True)
        if mode in ("kf_cache_recip", "kf_cache_recip_attention"):
            self.reciprocals.configure("all")
        if enhanced:
            self.cache.configure(True)
        self.mode = mode
