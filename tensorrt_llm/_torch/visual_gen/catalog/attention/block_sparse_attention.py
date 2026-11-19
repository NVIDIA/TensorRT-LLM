# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.video_sparse_attention.interface import (
    block_sparse_attn_from_indices_cute,
    is_cute_supported,
)

__all__ = ["block_sparse_attention", "is_cute_supported"]


class BlockSparseAttention(OpWrapper):
    """CuTe attention on contiguous FP16/BF16 BHSD tensors with head dimension 128.

    Each 64-token query tile supplies its own KV tile indices. Paired query
    tiles (2i, 2i+1) share a route count. Each KV tile supplies its valid prefix length.
    Returns attention output in the query dtype and per-query FP32 logsumexp.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "per-head top-K with a final unpaired query tile",
            spec=dict(counts=(2, 2, 2), sizes=(64, 64, 64), dtype=torch.bfloat16, scale=None),
        ),
        Cell(
            "different query and KV tile counts with partial KV tiles",
            spec=dict(counts=(3, 3, 1, 1), sizes=(8, 64, 37), dtype=torch.float16, scale=0.0625),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k_idx: torch.Tensor,
        q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return block_sparse_attn_from_indices_cute(
            q, k, v, q2k_idx, q2k_num, variable_block_sizes, sm_scale
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k_idx: torch.Tensor,
        q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = q.size(-1) ** -0.5 if sm_scale is None else sm_scale
        offsets = torch.arange(64, device=q.device)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        for b in range(q.size(0)):
            for h in range(q.size(1)):
                for i, indices in enumerate(q2k_idx[b, h]):
                    tiles = indices[: int(q2k_num[b, h, i])].long()
                    columns = (tiles[:, None] * 64 + offsets)[
                        offsets < variable_block_sizes[tiles, None]
                    ]
                    rows = slice(i * 64, (i + 1) * 64)
                    scores = (q[b, h, rows].float() @ k[b, h, columns].float().T) * scale
                    out[b, h, rows] = (scores.softmax(dim=-1) @ v[b, h, columns].float()).to(
                        q.dtype
                    )
                    lse[b, h, rows] = scores.logsumexp(dim=-1)
        return out, lse

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k_idx: torch.Tensor,
        q2k_num: torch.Tensor,
        variable_block_sizes: torch.Tensor,
        sm_scale: float | None = None,
    ) -> None:
        assert q.ndim == k.ndim == v.ndim == 4
        assert q.shape[:2] == k.shape[:2] and k.shape == v.shape
        assert q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
        assert q.shape[-1] == k.shape[-1] == 128
        assert q2k_idx.ndim == 4 and q2k_idx.shape[:-1] == q2k_num.shape
        assert q2k_num.shape[:2] == q.shape[:2] and q2k_num.size(2) * 64 == q.size(2)
        assert variable_block_sizes.ndim == 1 and variable_block_sizes.numel() * 64 == k.size(2)
        assert q2k_idx.dtype == q2k_num.dtype == variable_block_sizes.dtype == torch.int32
        assert all(t.device == q.device for t in (k, v, q2k_idx, q2k_num, variable_block_sizes))
        assert all(t.is_contiguous() and t.data_ptr() % 128 == 0 for t in (q, k, v))
        assert ((variable_block_sizes > 0) & (variable_block_sizes <= 64)).all()
        assert ((q2k_num > 0) & (q2k_num <= q2k_idx.size(-1))).all()
        assert torch.equal(q2k_num[..., :-1:2], q2k_num[..., 1::2]), (
            "paired query tiles need equal counts"
        )
        active = torch.arange(q2k_idx.size(-1), device=q.device) < q2k_num[..., None]
        indices = q2k_idx[active]
        assert ((indices >= 0) & (indices < variable_block_sizes.numel())).all()


block_sparse_attention = BlockSparseAttention()
