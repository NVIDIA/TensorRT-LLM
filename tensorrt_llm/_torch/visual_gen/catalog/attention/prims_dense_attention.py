# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class PrimTSValueGrouping:
    """Caller-owned token permutation and k-means centroids for one value geometry."""

    def __init__(self):
        self.permutation = None
        self.centroids = None

    def refresh(self, v: torch.Tensor) -> None:
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
            vc_token_permutation,
        )

        perm, centroids = vc_token_permutation(v, init_centroids=self.centroids)
        if self.permutation is None:
            self.permutation = perm.to(torch.int32)
            self.centroids = centroids
        else:
            # CUDA graph replay reads the same permutation and centroid buffers.
            self.permutation.copy_(perm.to(torch.int32))
            self.centroids.copy_(centroids)


def _vc_operands(quantization, q, k, v, quantize_qk, permutation, demean, repair_budget):
    if repair_budget:
        return quantization.vc_quantize_repair(
            k, v, budget=repair_budget, q=q if quantize_qk else None
        )
    if quantize_qk:
        return quantization.vc_quantize_fp8(q, k, v, perm=permutation, demean=demean)
    return quantization.vc_quantize(k, v, perm=permutation, demean=demean)


class PrimsDenseAttention(OpWrapper):
    """Noncausal BSHD attention with per-tensor FP8 or per-channel value compression.

    A token permutation selects VC grouping; a positive repair budget selects VC repair tiles.
    The optional Q/K quantization uses per-tensor FP8 or VC's rotated, block-scaled FP8.
    Plans belong to the caller and may be reused across calls with matching geometry.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("per-tensor FP8 values", dict(quantize_qk=False)),
        Cell("per-tensor FP8 Q/K/V", dict(quantize_qk=True)),
        Cell("permuted VC values with tile means", dict(quantize_qk=False, grouping=True)),
        Cell("rotated Q/K and VC repair tiles", dict(quantize_qk=True, repair=True)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        quantize_qk: bool = False,
        vc_permutation: torch.Tensor | None = None,
        vc_demean: bool = True,
        vc_repair_budget: float = 0.0,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention import VC_K_BLOCK_SIZE

        if vc_permutation is not None and vc_repair_budget:
            raise ValueError("VC grouping and repair use separate value layouts")
        batch, q_len, heads, dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        ragged_repair = vc_repair_budget > 0 and kv_len % VC_K_BLOCK_SIZE != 0
        if ragged_repair:
            # The wheel masks the logical final tile of padded VC repair operands.
            from flashinfer.attention import prims_ts
            from flashinfer.attention.prims_ts.vc_attention import quantization
        else:
            from tensorrt_llm._torch.attention.backends import prims_ts
            from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention import quantization

        dtype = q.dtype
        scale = 1.0 / math.sqrt(dim)
        output_scale = 1.0
        vc_config = vc_params = None
        if vc_permutation is not None or vc_repair_budget:
            operands = _vc_operands(
                quantization, q, k, v, quantize_qk, vc_permutation, vc_demean, vc_repair_budget
            )
            k, v, vc_params = operands.k, operands.v, operands.params
            if operands.q is not None:
                q = operands.q
            vc_config = prims_ts.VCAttentionConfig(
                demean=operands.demean, repair_tiles=operands.repair_tiles
            )
        else:
            if quantize_qk:
                q, q_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(q)
                k, k_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(k)
                scale = (q_scale.float() * k_scale.float() * scale).reshape(1)
            v, v_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(v)
            output_scale = v_scale.float().reshape(1)
        key = (
            q.device,
            batch,
            q_len,
            kv_len,
            heads,
            kv_heads,
            dim,
            q.dtype,
            k.dtype,
            v.dtype,
            dtype,
            vc_config,
        )
        if key not in plans:
            plan = prims_ts.BatchPrefillTSWrapper()
            query_plan = (
                dict(max_seq_len_q=q_len, sm_scale=scale)
                if ragged_repair
                else dict(max_q_len=q_len, softmax_scale=scale)
            )
            plan.plan(
                device=q.device,
                batch_size=batch,
                max_kv_len=kv_len,
                num_qo_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=dim,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                out_dtype=dtype,
                mask_type="dense",
                output_scale=output_scale,
                vc_config=vc_config,
                **query_plan,
            )
            plans[key] = plan
        out = torch.empty((batch, q_len, heads, dim), dtype=dtype, device=q.device)
        # VC plans keep both scales. Per-tensor quantization refreshes them for every input.
        run_scales = (
            {} if vc_params is not None else dict(softmax_scale=scale, output_scale=output_scale)
        )
        plans[key].run(
            q,
            k,
            v,
            out=out,
            vc=vc_params,
            validate=False,
            **run_scales,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        quantize_qk: bool = False,
        vc_permutation: torch.Tensor | None = None,
        vc_demean: bool = True,
        vc_repair_budget: float = 0.0,
    ) -> torch.Tensor:
        """Dense attention before FP8 rounding. Grouping and repair change its approximation."""
        return (
            F.scaled_dot_product_attention(
                q.transpose(1, 2).double(),
                k.transpose(1, 2).double(),
                v.transpose(1, 2).double(),
                enable_gqa=q.size(2) != k.size(2),
            )
            .transpose(1, 2)
            .to(q.dtype)
        )

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        plans: dict,
        quantize_qk: bool = False,
        vc_permutation: torch.Tensor | None = None,
        vc_demean: bool = True,
        vc_repair_budget: float = 0.0,
    ) -> None:
        assert q.ndim == k.ndim == v.ndim == 4 and k.shape == v.shape
        assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1]
        assert min(*q.shape, *k.shape) > 0 and q.shape[2] % k.shape[2] == 0
        assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
        assert q.device == k.device == v.device
        for x in (q, k, v):
            assert x.stride() == tuple(math.prod(x.shape[i + 1 :]) for i in range(x.ndim))
            assert x.data_ptr() % 16 == 0
        if vc_permutation is not None or vc_repair_budget:
            assert q.shape[2:] == k.shape[2:] and q.size(-1) == 128
            assert 0 <= vc_repair_budget < 1 and (vc_permutation is None or vc_repair_budget == 0)
        if vc_permutation is not None:
            assert vc_permutation.shape == (k.size(0), k.size(2), k.size(1))
            assert vc_permutation.device == k.device and vc_permutation.dtype in (
                torch.int32,
                torch.int64,
            )
            assert torch.equal(
                vc_permutation.sort(dim=-1).values,
                torch.arange(k.size(1), device=k.device).expand_as(vc_permutation),
            )


prims_dense_attention = PrimsDenseAttention()
