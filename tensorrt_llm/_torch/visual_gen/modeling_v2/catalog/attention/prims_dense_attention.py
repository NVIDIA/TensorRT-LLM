# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..quantization.fp8_quantize import native_fp8_per_tensor_quantize


class VcTokenPermutation(OpWrapper):
    """Group BSHD values by three FP32 k-means updates with K=min(64, S).
    Matching [B*H,K,D] centroids warm-start the groups. Otherwise, each head samples K distinct
    tokens using the supplied generator or the device RNG. Return stable int64 [B,H,S] indices
    ordered by the last assignment and FP32 [B*H,K,D] centroids. Empty clusters retain their center.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Seeded token rows across batches and heads", dict(tokens=97, warm=False)),
        Cell("Short sequence with warm and empty clusters", dict(tokens=37, warm=True)),
    )

    def __call__(
        self,
        v: torch.Tensor,
        *,
        init_centroids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
            vc_token_permutation as group_values,
        )

        return group_values(v, init_centroids=init_centroids, generator=generator)

    def reference(self, v, *, init_centroids=None, generator=None):
        b, s, h, d = v.shape
        k = min(64, s)
        rows = v.transpose(1, 2).reshape(b * h, s, d).float().double()
        if init_centroids is None or init_centroids.shape != (b * h, k, d):
            seeds = torch.stack(
                [torch.randperm(s, device=v.device, generator=generator)[:k] for _ in range(b * h)]
            )
            centers = rows.gather(1, seeds.unsqueeze(-1).expand(-1, -1, d))
        else:
            centers = init_centroids.float().double()

        for _ in range(3):
            labels = torch.cdist(rows, centers, compute_mode="donot_use_mm_for_euclid_dist").argmin(
                -1
            )
            membership = F.one_hot(labels, k).transpose(1, 2).double()
            counts = membership.sum(-1, keepdim=True)
            means = ((membership @ rows) / counts.clamp_min(1)).float().double()
            centers = torch.where(counts > 0, means, centers)

        return labels.argsort(dim=-1, stable=True).reshape(b, h, s), centers.float()

    def is_valid(self, v, *, init_centroids=None, generator=None):
        assert v.ndim == 4 and min(v.shape) > 0 and v.is_floating_point()
        b, s, h, d = v.shape
        if init_centroids is not None and init_centroids.shape == (b * h, min(64, s), d):
            assert init_centroids.is_floating_point() and init_centroids.device == v.device
        elif generator is not None:
            assert generator.device == v.device


vc_token_permutation = VcTokenPermutation()


def _vc_operands(q, k, v, quantize_qk, permutation, demean, repair_budget):
    if repair_budget:
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
            vc_quantize_repair,
        )

        return vc_quantize_repair(k, v, budget=repair_budget, q=q if quantize_qk else None)
    if quantize_qk:
        from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
            vc_quantize_fp8,
        )

        return vc_quantize_fp8(q, k, v, perm=permutation, demean=demean)

    from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
        vc_quantize,
    )

    return vc_quantize(k, v, perm=permutation, demean=demean)


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
        Cell("BF16 Q/K with aligned VC repair", dict(quantize_qk=False, repair=True, kv_len=256)),
        Cell("rotated Q/K with aligned VC repair", dict(quantize_qk=True, repair=True, kv_len=256)),
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
        from tensorrt_llm._torch.attention.backends.prims_ts import (
            BatchPrefillTSWrapper,
            VCAttentionConfig,
        )

        if vc_permutation is not None and vc_repair_budget:
            raise ValueError("VC grouping and repair use separate value layouts")
        batch, q_len, heads, dim = q.shape
        kv_len, kv_heads = k.shape[1:3]
        dtype = q.dtype
        scale = 1.0 / math.sqrt(dim)
        output_scale = 1.0
        vc_config = vc_params = None
        if vc_permutation is not None or vc_repair_budget:
            operands = _vc_operands(
                q, k, v, quantize_qk, vc_permutation, vc_demean, vc_repair_budget
            )
            k, v, vc_params = operands.k, operands.v, operands.params
            if operands.q is not None:
                q = operands.q
            vc_config = VCAttentionConfig(
                demean=operands.demean, repair_tiles=operands.repair_tiles
            )
        else:
            if quantize_qk:
                q, q_scale = native_fp8_per_tensor_quantize(q)
                k, k_scale = native_fp8_per_tensor_quantize(k)
                scale = (q_scale.float() * k_scale.float() * scale).reshape(1)
            v, v_scale = native_fp8_per_tensor_quantize(v)
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
            plan = BatchPrefillTSWrapper()
            plan.plan(
                device=q.device,
                batch_size=batch,
                max_q_len=q_len,
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
                softmax_scale=scale,
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
