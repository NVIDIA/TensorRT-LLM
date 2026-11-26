# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from tensorrt_llm._torch.cute_dsl_utils import load_flash_attn_cute_forward
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.visual_gen.attention._fa4_compat import (
    install_flash_attn_tile_scheduler_compatibility,
)

# FA4 reads this setting when its CuTe interface is first imported.
os.environ.setdefault("FA_DISABLE_2CTA", "1")
_flash_attn_fwd_import_error = None
try:
    _flash_attn_fwd = load_flash_attn_cute_forward()
    install_flash_attn_tile_scheduler_compatibility()
except (AttributeError, ImportError, OSError) as e:
    _flash_attn_fwd = None
    _flash_attn_fwd_import_error = e


@torch.compiler.disable
def _run_fa4(q, k, v, softmax_scale, causal, seqused_k):
    if _flash_attn_fwd is None:
        raise ImportError(
            f"FlashAttention 4 is not available. Import error: {_flash_attn_fwd_import_error}"
        ) from _flash_attn_fwd_import_error
    output, lse, *_ = _flash_attn_fwd(
        q,
        k,
        v,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
        mask_mod=None,
        block_sparse_tensors=None,
        return_lse=True,
        num_splits=0,
    )
    return output, lse


class FlashAttention4(OpWrapper):
    """FA4 on NHD queries and KV with optional per-sample valid KV lengths.

    Returns NHD output in the query dtype and FP32 logsumexp in [B, Hq, Sq].
    Causal masking aligns the last query with the last valid key.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "BF16 self-attention with padded keys",
            dict(
                dtype=torch.bfloat16,
                sq=67,
                sk=67,
                hq=4,
                hk=4,
                dim=128,
                lengths=(49, 63),
                causal=False,
            ),
        ),
        Cell(
            "FP16 causal cross-attention with GQA",
            dict(dtype=torch.float16, sq=11, sk=27, hq=4, hk=2, dim=64, lengths=None, causal=True),
        ),
        Cell(
            "unmasked BF16 prefill",
            dict(
                dtype=torch.bfloat16, sq=17, sk=17, hq=2, hk=2, dim=64, lengths=None, causal=False
            ),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool = False,
        seqused_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or q.dtype != k.dtype
            or q.dtype != v.dtype
        ):
            raise ValueError("FA4 requires matching FP16 or BF16 Q/K/V tensors.")
        return _run_fa4(q, k, v, softmax_scale, causal, seqused_k)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool = False,
        seqused_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries_per_kv = q.size(2) // k.size(2)
        qh = q.transpose(1, 2).double()
        kh = k.transpose(1, 2).double().repeat_interleave(queries_per_kv, dim=1)
        vh = v.transpose(1, 2).double().repeat_interleave(queries_per_kv, dim=1)
        scores = (qh @ kh.transpose(-1, -2)) * softmax_scale
        positions = torch.arange(k.size(1), device=q.device)
        if seqused_k is not None:
            scores = scores.masked_fill(positions >= seqused_k[:, None, None, None], -torch.inf)
        if causal:
            lengths = k.size(1) if seqused_k is None else seqused_k[:, None, None, None]
            query_rows = torch.arange(q.size(1), device=q.device)[:, None]
            scores = scores.masked_fill(positions > query_rows + lengths - q.size(1), -torch.inf)
        output = (scores.softmax(-1).nan_to_num() @ vh).transpose(1, 2)
        return output.to(q.dtype), scores.logsumexp(-1).float()

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool = False,
        seqused_k: torch.Tensor | None = None,
    ) -> None:
        assert q.device == k.device == v.device
        if seqused_k is not None:
            assert seqused_k.device == q.device
            assert ((seqused_k >= 0) & (seqused_k <= k.size(1))).all(), (
                "valid key lengths must fit the KV sequence"
            )


flash_attention_4 = FlashAttention4()
