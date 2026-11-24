# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os

import torch

from tensorrt_llm._torch.visual_gen.runtime import QuantAttentionConfig, logger
from tensorrt_llm._torch.visual_gen.runtime.fmha import (
    enabled_fmha_libraries,
    is_context_prims_available,
)

from ....catalog.attention.prims_dense_attention import prims_dense_attention, vc_token_permutation
from ....catalog.attention.prims_routed_attention import (
    PrimsRoutes,
    PrimsSageQuantization,
    prims_routed_attention,
    supports_dense_sage_profile,
)

_VC_SMOOTH_FRACTION = 0.25
_VC_REFRESH_INTERVAL = 4


class PrimTSValueGrouping:
    """Token permutation and centroids held across denoising steps."""

    def __init__(self):
        self.permutation = None
        self.centroids = None

    def refresh(self, v: torch.Tensor) -> None:
        perm, centroids = vc_token_permutation(v, init_centroids=self.centroids)
        if self.permutation is None:
            self.permutation = perm.to(torch.int32)
            self.centroids = centroids
        else:
            # CUDA graph replay reads the same permutation and centroid buffers.
            self.permutation.copy_(perm.to(torch.int32))
            self.centroids.copy_(centroids)


class H3PrimTS:
    """Own the attention plans, compression schedule and provider selection for an H3 instance."""

    def __init__(self, recipe: QuantAttentionConfig | None, metadata_state: dict):
        self.recipe = recipe
        self.metadata_state = metadata_state
        enabled = enabled_fmha_libraries()
        self._enabled = enabled
        self._available: dict[str, bool] = {}
        self.fallback_enabled = "fallback" in enabled
        self.dense_explicitly_enabled = "prims_ts" in enabled and any(
            token.strip().removeprefix("+").strip() == "prims_ts"
            for token in os.environ.get("TLLM_FMHA_LIBS", "").split(",")
        )
        self.smooth_v_plan = "prims_ts_block_sparse" in enabled and (
            not self.fallback_enabled
            or enabled.index("prims_ts_block_sparse") < enabled.index("fallback")
        )
        if recipe is not None and recipe.vc_attention and not self._has_prims("prims_ts"):
            raise RuntimeError(
                "VC attention requires an available dense PrimTS provider (TLLM_FMHA_LIBS=+prims_ts)."
            )
        if (
            recipe is not None
            and recipe.q_block_size == 0
            and not (self._has_prims("prims_ts") or self._has_prims("prims_ts_block_sparse"))
        ):
            raise RuntimeError(
                "The per-tensor FP8 recipe requires an available PrimTS provider (TLLM_FMHA_LIBS=+prims_ts)."
            )
        self.dense_plans = {}
        self.vc_shape = None
        self.vc_grouping = None
        self.vc_step = None

    def _has_prims(self, name: str) -> bool:
        if name not in self._available:
            self._available[name] = name in self._enabled and is_context_prims_available(name)
        return self._available[name]

    @torch.compiler.disable
    def __call__(
        self,
        q,
        k,
        v,
        routes: PrimsRoutes | None,
        *,
        skip_softmax: bool,
        skip_threshold_active: bool,
    ):
        recipe = self.recipe
        if routes is not None:
            if not self._has_prims("prims_ts_block_sparse"):
                raise RuntimeError(
                    "H3 sparse attention requires an available PrimTS block-sparse FMHA library."
                )
            if recipe is not None and recipe.vc_attention:
                raise ValueError("VC-Attention requires a dense attention plan.")
            return self._block_sparse(q, k, v, routes)
        if recipe is None:
            return None
        if recipe.vc_attention or recipe.q_block_size == 0:
            if not self.dense_explicitly_enabled or not self._has_prims("prims_ts"):
                raise RuntimeError(
                    "The per-tensor FP8 recipe requires an available dense PrimTS provider (TLLM_FMHA_LIBS=+prims_ts)."
                )
            if skip_softmax:
                raise ValueError("Per-tensor and VC PrimTS attention require dense attention.")
            return self._dense_fp8(q, k, v)
        if recipe.smooth_v:
            if (
                self.smooth_v_plan
                and not skip_threshold_active
                and self._has_prims("prims_ts_block_sparse")
            ):
                out = self._block_sparse(q, k, v, None)
                if out is not None:
                    return out
            if self.fallback_enabled:
                logger.warning_once(
                    "The attention op quantizes V without the V smoothing this SageAttention recipe requests.",
                    key="fallback_fmha_sage_attn_smooth_v",
                )
        return None

    def _block_sparse(self, q, k, v, routes):
        recipe = self.recipe
        quantization = None
        if recipe is not None and recipe.q_block_size > 0:
            quantization = PrimsSageQuantization(
                qk_dtype=torch.int8 if recipe.qk_dtype == "int8" else torch.float8_e4m3fn,
                q_block_size=recipe.q_block_size,
                k_block_size=recipe.k_block_size,
                smooth_k=recipe.smooth_k,
                smooth_v=recipe.smooth_v,
            )
        if routes is None and not supports_dense_sage_profile(q, k, quantization):
            return None
        return prims_routed_attention(
            q,
            k,
            v,
            plans=self.metadata_state.setdefault("h3_block_sparse_plans", {}),
            quantization=quantization,
            **(routes or {}),
        )

    def _dense_fp8(self, q, k, v):
        recipe = self.recipe
        vc = {}
        if recipe.vc_attention:
            if recipe.vc_repair_budget:
                vc["vc_repair_budget"] = recipe.vc_repair_budget
            else:
                if self.vc_grouping is None or self.vc_shape != k.shape:
                    self.vc_grouping = PrimTSValueGrouping()
                    self.vc_shape = k.shape
                    self.vc_step = None
                step = self.metadata_state.get("denoise_step")
                smooth = step is None or step[0] < math.ceil(_VC_SMOOTH_FRACTION * step[1])
                refresh = self.vc_grouping.permutation is None or (
                    step is not None
                    and (
                        step[0] == 0
                        or (step[0] % _VC_REFRESH_INTERVAL == 0 and step[0] != self.vc_step)
                    )
                )
                if smooth and refresh:
                    self.vc_grouping.refresh(v)
                    self.vc_step = None if step is None else step[0]
                perm = self.vc_grouping.permutation
                if perm is None:
                    perm = torch.arange(k.size(1), dtype=torch.int32, device=k.device)
                    perm = perm.view(1, 1, -1).expand(k.size(0), k.size(2), -1).contiguous()
                vc.update(vc_permutation=perm, vc_demean=smooth)
        return prims_dense_attention(
            q, k, v, plans=self.dense_plans, quantize_qk=recipe.qk_dtype == "fp8", **vc
        )
