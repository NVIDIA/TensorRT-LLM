# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ....catalog.attention.prims_dense_attention import PrimTSValueGrouping, prims_dense_attention
from ....catalog.attention.prims_routed_attention import (
    PrimsSageQuantization,
    prims_routed_attention,
    supports_dense_sage_profile,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs


_FMHA_LIBS = (
    "fp4_mla",
    "triton_custom_mask",
    "cute_dsl_mla",
    "msa_decode",
    "msa_prefill",
    "flashinfer_sparse_mla",
    "prims_ts",
    "prims_ts_block_sparse",
    "prims_ts_sparse_mla",
    "prims_ts_dsa_sparse_mla",
    "gqa_kv_replica_fused",
    "gqa_kv_replica",
    "gqa_split_kv",
    "gqa_segmented",
    "gqa_decode",
    "flashinfer_trtllm_gen",
    "beam_shared_prefix",
    "fallback",
)
_DEFAULT_FMHA_LIBS = tuple(
    name for name in _FMHA_LIBS if name not in {"prims_ts", "beam_shared_prefix"}
)
_VC_SMOOTH_FRACTION = 0.25
_VC_REFRESH_INTERVAL = 4


def _enabled_fmha_libraries() -> tuple[str, ...]:
    value = os.environ.get("TLLM_FMHA_LIBS", "")
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        return _DEFAULT_FMHA_LIBS
    deltas = [t.startswith(("+", "-")) for t in tokens]
    if any(deltas) and not all(deltas):
        raise ValueError(
            "TLLM_FMHA_LIBS must use either an exact comma-separated list or only +/- delta entries."
        )
    names = [t[1:].strip() for t in tokens] if all(deltas) else tokens
    for name in names:
        if not name:
            raise ValueError(f"Invalid empty FMHA library entry in {value!r}.")
        if name not in _FMHA_LIBS:
            raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
    if not all(deltas):
        return tuple(dict.fromkeys(names))
    enabled = set(_DEFAULT_FMHA_LIBS)
    for token, name in zip(tokens, names, strict=True):
        if token[0] == "+":
            enabled.add(name)
        else:
            enabled.discard(name)
    return tuple(name for name in _FMHA_LIBS if name in enabled)


class H3PrimTS:
    """Own the attention plans, compression schedule and provider selection for an H3 instance."""

    def __init__(self, recipe: QuantAttentionConfig | None, metadata_state: dict):
        self.recipe = recipe
        self.metadata_state = metadata_state
        enabled = _enabled_fmha_libraries()
        self.fallback_enabled = "fallback" in enabled
        self.block_sparse_enabled = "prims_ts_block_sparse" in enabled
        self.dense_enabled = "prims_ts" in enabled
        self.smooth_v_plan = self.block_sparse_enabled and (
            not self.fallback_enabled
            or enabled.index("prims_ts_block_sparse") < enabled.index("fallback")
        )
        if recipe is not None and recipe.vc_attention and not self.dense_enabled:
            raise RuntimeError("The VC attention recipe requires TLLM_FMHA_LIBS=+prims_ts.")
        if (
            recipe is not None
            and recipe.q_block_size == 0
            and not (self.dense_enabled or self.block_sparse_enabled)
        ):
            raise RuntimeError("The per-tensor FP8 recipe requires TLLM_FMHA_LIBS=+prims_ts.")
        self.dense_plans = {}
        self.vc_shape = None
        self.vc_grouping = None
        self.vc_step = None

    @torch.compiler.disable
    def __call__(
        self,
        q,
        k,
        v,
        routes: BlockSparseForwardInputs | None,
        *,
        skip_softmax: bool,
        skip_threshold_active: bool,
    ):
        recipe = self.recipe
        if routes is not None:
            if not self.block_sparse_enabled:
                raise RuntimeError(
                    "H3 sparse attention requires the PrimTS block-sparse FMHA library."
                )
            if recipe is not None and recipe.vc_attention:
                raise ValueError("VC-Attention requires a dense attention plan.")
            return self._block_sparse(q, k, v, routes)
        if recipe is None:
            return None
        if recipe.vc_attention or recipe.q_block_size == 0:
            if not self.dense_enabled:
                raise RuntimeError("The per-tensor FP8 recipe requires TLLM_FMHA_LIBS=+prims_ts.")
            if skip_softmax:
                raise ValueError("Per-tensor and VC PrimTS attention require dense attention.")
            return self._dense_fp8(q, k, v)
        if recipe.smooth_v:
            if self.smooth_v_plan and not skip_threshold_active:
                out = self._block_sparse(q, k, v, None)
                if out is not None:
                    return out
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
        route_args = {}
        if routes is not None:
            route_args = dict(
                q_block_size=routes.q_block_size,
                kv_block_size=routes.kv_block_size,
                max_blocks_per_row=routes.max_blocks_per_row,
                block_indptr=routes.block_indptr,
                block_indices=routes.block_indices,
                exact_block_bits=routes.exact_block_bits,
                kv_valid_bits=routes.kv_valid_bits,
                k_summary=routes.k_summary,
                v_summary=routes.v_summary,
            )
        return prims_routed_attention(
            q,
            k,
            v,
            plans=self.metadata_state.setdefault("h3_block_sparse_plans", {}),
            quantization=quantization,
            **route_args,
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
