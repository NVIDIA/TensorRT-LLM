# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PrimTS contiguous plans for H3 attention."""

import math
from dataclasses import replace

import torch

from tensorrt_llm._torch.attention.backends.fmha.registry import (
    FMHA_LIBS,
    get_enabled_fmha_lib_classes,
)
from tensorrt_llm._torch.attention.backends.sparse.params import BlockSparseForwardInputs
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig


class H3PrimTS:
    """Plan caches and quantization state for contiguous attention on an H3 instance."""

    def __init__(self, recipe: QuantAttentionConfig | None, metadata_state: dict):
        self.recipe = recipe
        self.metadata_state = metadata_state
        enabled = get_enabled_fmha_lib_classes()
        block_sparse = FMHA_LIBS["prims_ts_block_sparse"]
        fallback = FMHA_LIBS["fallback"]
        self.fallback_enabled = fallback in enabled
        self.block_sparse_enabled = block_sparse in enabled
        self.dense_enabled = FMHA_LIBS["prims_ts"] in enabled
        self.smooth_v_plan = self.block_sparse_enabled and (
            not self.fallback_enabled or enabled.index(block_sparse) < enabled.index(fallback)
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
        self.vc_preprocessor = None

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
        from tensorrt_llm._torch.attention.backends.fmha.block_sparse_plan import BlockSparsePlan
        from tensorrt_llm._torch.attention.backends.prims_ts.sage import (
            SageAttentionConfig,
            SageAttentionParams,
        )
        from tensorrt_llm._torch.attention.kernels.sage_quant import sage_quantize

        batch, length, heads, dim = q.shape
        kv_heads = k.shape[2]
        dtype = q.dtype
        recipe = self.recipe
        qk_dtype, value_dtype, sage = dtype, dtype, None
        if recipe is not None and recipe.q_block_size > 0:
            sage = SageAttentionConfig(
                q_block_size=recipe.q_block_size,
                k_block_size=recipe.k_block_size,
                v_mean=recipe.smooth_v,
                k_summary_block_size=1,
            )
            qk_dtype = torch.int8 if recipe.qk_dtype == "int8" else torch.float8_e4m3fn
            value_dtype = torch.float8_e4m3fn
        q_block = routes.q_block_size if routes else 128
        kv_block = routes.kv_block_size if routes else 128
        capacity = routes.max_blocks_per_row if routes else None
        valid_bits = routes is not None and routes.kv_valid_bits is not None
        sparse_format = routes.sparse_format if routes else "bsr"
        proxy = routes.use_proxy_routes if routes else False
        if routes is not None and capacity is None:
            capacity = (length + kv_block - 1) // kv_block
        plans = self.metadata_state.setdefault("h3_block_sparse_plans", {})
        key = BlockSparsePlan(
            device=q.device,
            batch_size=batch,
            seq_len_q=length,
            kv_capacity=length,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            page_size=None,
            q_block_size=q_block,
            kv_block_size=kv_block,
            max_blocks_per_row=capacity,
            mask_type="dense",
            dtype=dtype,
            qk_dtype=qk_dtype,
            v_dtype=value_dtype,
            use_kv_valid_bits=valid_bits,
            sparse_format=sparse_format,
            use_proxy_routes=proxy,
            sage=sage,
            use_block_sparse=routes is not None,
        )
        if key not in plans:
            if routes is None and key.unsupported_reason() is not None:
                return None
            plans[key] = key.plan()

        sage_params = None
        if sage is not None:
            offsets = torch.arange(
                0, (batch + 1) * length, length, device=q.device, dtype=torch.int32
            )
            operands = sage_quantize(
                q.flatten(0, 1),
                k.flatten(0, 1),
                v.flatten(0, 1),
                offsets,
                offsets,
                q_block_size=sage.q_block_size,
                k_block_size=sage.k_block_size,
                qk_quant_dtype=qk_dtype,
                smooth_k=recipe.smooth_k,
                smooth_v=recipe.smooth_v,
            )
            q, k, v = operands.q.view_as(q), operands.k.view_as(k), operands.v.view_as(v)
            summary_scale = None
            if proxy:
                count = routes.k_summary.shape[1]
                summary_offsets = torch.arange(
                    0, (batch + 1) * count, count, device=q.device, dtype=torch.int32
                )
                summary = sage_quantize(
                    None,
                    routes.k_summary.flatten(0, 1),
                    routes.v_summary.flatten(0, 1),
                    None,
                    summary_offsets,
                    k_block_size=1,
                    qk_quant_dtype=qk_dtype,
                    stats=operands,
                )
                routes = replace(
                    routes,
                    k_summary=summary.k.view_as(routes.k_summary),
                    v_summary=summary.v.view_as(routes.v_summary),
                )
                summary_scale = summary.k_scale
            sage_params = SageAttentionParams(
                q_scale=operands.q_scale,
                k_scale=operands.k_scale,
                v_scale=operands.v_scale,
                k_summary_scale=summary_scale,
                v_mean=operands.v_mean,
            )
        route_args = {}
        if routes is not None:
            route_args = dict(
                block_indptr=routes.block_indptr,
                block_indices=routes.block_indices,
                exact_block_bits=routes.exact_block_bits,
                kv_valid_bits=routes.kv_valid_bits,
                k_summary=routes.k_summary,
                v_summary=routes.v_summary,
            )
        return plans[key].run(
            q,
            k,
            v,
            sm_scale=1.0 / math.sqrt(dim),
            out=torch.empty_like(q, dtype=dtype),
            sage=sage_params,
            **route_args,
        )

    def _dense_fp8(self, q, k, v):
        from tensorrt_llm._torch.attention.backends.prims_ts import (
            BatchPrefillTSWrapper,
            VCAttentionConfig,
        )
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
            VCAttentionPreprocessor,
            vc_quantize_repair,
        )

        batch, length, heads, dim = q.shape
        recipe = self.recipe
        dtype = q.dtype
        sm_scale = 1.0 / math.sqrt(dim)
        output_scale = 1.0
        vc_config = vc_params = None
        if recipe.vc_attention:
            vc_q = q if recipe.qk_dtype == "fp8" else None
            if recipe.vc_repair_budget:
                operands = vc_quantize_repair(k, v, budget=recipe.vc_repair_budget, q=vc_q)
            else:
                if self.vc_shape != k.shape:
                    self.vc_preprocessor = VCAttentionPreprocessor()
                    self.vc_shape = k.shape
                operands = self.vc_preprocessor.prepare(
                    k, v, q=vc_q, denoise_step=self.metadata_state.get("denoise_step")
                )
            k, v, vc_params = operands.k, operands.v, operands.params
            if operands.q is not None:
                q = operands.q
            vc_config = VCAttentionConfig(
                demean=operands.demean, repair_tiles=operands.repair_tiles
            )
        else:
            if recipe.qk_dtype == "fp8":
                q, q_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(q)
                k, k_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(k)
                sm_scale = (q_scale.float() * k_scale.float() * sm_scale).reshape(1)
            v, v_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(v)
            output_scale = v_scale.float().reshape(1)
        kv_heads = k.shape[2]
        plan_key = (
            q.device,
            batch,
            length,
            heads,
            kv_heads,
            dim,
            q.dtype,
            k.dtype,
            v.dtype,
            dtype,
            vc_config,
        )
        plans = self.dense_plans
        if plan_key not in plans:
            plan = BatchPrefillTSWrapper()
            plan.plan(
                device=q.device,
                batch_size=batch,
                max_q_len=length,
                max_kv_len=length,
                num_qo_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=dim,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                out_dtype=dtype,
                mask_type="dense",
                softmax_scale=sm_scale,
                output_scale=output_scale,
                vc_config=vc_config,
            )
            plans[plan_key] = plan
        out = torch.empty((batch, length, heads, dim), dtype=dtype, device=q.device)
        plans[plan_key].run(
            q,
            k,
            v,
            out=out,
            softmax_scale=sm_scale,
            output_scale=output_scale,
            vc=vc_params,
            validate=False,
        )
        return out
