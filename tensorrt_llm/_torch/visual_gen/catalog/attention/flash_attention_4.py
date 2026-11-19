# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Keep the long-sequence kernel selection in effect at FA4 import.
os.environ.setdefault("FA_DISABLE_2CTA", "1")


def _install_cutlass_dsl_compatibility() -> None:
    """Restore CuTe aliases required by pinned third-party FA4 and QuACK."""
    import cutlass.cute as cute

    for name in ("ThrCopy", "ThrMma"):
        if not hasattr(cute.core, name) and hasattr(cute, name):
            setattr(cute.core, name, getattr(cute, name))
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor


def _install_flash_attn_tile_scheduler_compatibility() -> None:
    """Give FA4's four-axis CuTe tiles their own coordinate storage and MLIR reconstruction."""
    try:
        from flash_attn.cute import tile_scheduler
    except (ImportError, OSError):
        return
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Boolean, extract_mlir_values

    work_tile_info = tile_scheduler.WorkTileInfo
    if "__init__" in vars(work_tile_info):
        return

    def __init__(self, tile_idx: cute.Coord, is_valid_tile: Boolean) -> None:
        self._tile_idx = tile_idx
        self._is_valid_tile = Boolean(is_valid_tile)
        self._tile_idx_num_values = None

    def __extract_mlir_values__(self) -> list:
        tile_idx_values = extract_mlir_values(self._tile_idx)
        valid_values = extract_mlir_values(self._is_valid_tile)
        self._tile_idx_num_values = len(tile_idx_values)
        return tile_idx_values + valid_values

    @cute.jit
    def tile_idx(self) -> cute.Coord:
        return self._tile_idx

    @cute.jit
    def is_valid_tile(self) -> Boolean:
        return self._is_valid_tile

    work_tile_info.__init__ = __init__
    work_tile_info.__extract_mlir_values__ = __extract_mlir_values__
    work_tile_info.tile_idx = property(tile_idx)
    work_tile_info.is_valid_tile = property(is_valid_tile)


_flash_attn_fwd_import_error = None
try:
    _install_cutlass_dsl_compatibility()
    from flash_attn.cute.interface import _flash_attn_fwd

    _install_flash_attn_tile_scheduler_compatibility()
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
            "FP32 cross-attention via BF16",
            dict(dtype=torch.float32, sq=17, sk=19, hq=2, hk=2, dim=64, lengths=None, causal=False),
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
        dtype = q.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        output, lse = _run_fa4(q, k, v, softmax_scale, causal, seqused_k)
        return output.to(dtype), lse

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
        dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        queries_per_kv = q.size(2) // k.size(2)
        qh = q.to(dtype).transpose(1, 2).double()
        kh = k.to(dtype).transpose(1, 2).double().repeat_interleave(queries_per_kv, dim=1)
        vh = v.to(dtype).transpose(1, 2).double().repeat_interleave(queries_per_kv, dim=1)
        scores = (qh @ kh.transpose(-1, -2)) * softmax_scale
        positions = torch.arange(k.size(1), device=q.device)
        if seqused_k is not None:
            scores = scores.masked_fill(positions >= seqused_k[:, None, None, None], -torch.inf)
        if causal:
            lengths = k.size(1) if seqused_k is None else seqused_k[:, None, None, None]
            query_rows = torch.arange(q.size(1), device=q.device)[:, None]
            scores = scores.masked_fill(positions > query_rows + lengths - q.size(1), -torch.inf)
        output = (scores.softmax(-1).nan_to_num() @ vh).transpose(1, 2)
        return output.to(dtype).to(q.dtype), scores.logsumexp(-1).float()

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
        if seqused_k is not None:
            assert ((seqused_k >= 0) & (seqused_k <= k.size(1))).all(), (
                "valid key lengths must fit the KV sequence"
            )


flash_attention_4 = FlashAttention4()
