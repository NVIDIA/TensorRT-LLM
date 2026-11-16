# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's q side and this step's K-cache update in one CuTe DSL launch: the two
kernels of `cute_dsl_indexer_q_side_fused` and `cute_dsl_indexer_k_cache_update`, one grid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_indexer_q_side_fused as _q_kernel
from ...kernels.attention import self_attn_indexer_q_side_k_cache_fused as _kernel
from .cute_dsl_indexer_k_cache_update import cute_dsl_indexer_k_cache_update as _k_op
from .cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused as _q_op

_HEAD_DIM = 128
_HALF = 32


@dataclass(frozen=True)
class CompiledIndexerQSideKCache:
    """The compiled launcher and the two compile-time flags it was built for."""

    fn: Callable
    is_neox: bool
    cache_contiguous: bool


def build(device_id: int, *, is_neox: bool, cache_contiguous: bool) -> CompiledIndexerQSideKCache:
    """Compile the launcher (ue8m0 scales, PDL launch, 8-byte key loads). `cache_contiguous`
    must describe every cache view it will write (dense addressing on a strided per-layer
    view writes into other layers). One compile, cached per process; allocates nothing."""
    with torch.cuda.device(device_id):
        fn = _kernel.compile_indexer_q_side_k_cache(
            device_id, bool(is_neox), bool(cache_contiguous)
        )
    return CompiledIndexerQSideKCache(fn, bool(is_neox), bool(cache_contiguous))


class CuteDslIndexerQSideKCache(OpWrapper):
    """`cute_dsl_indexer_q_side_fused` and `cute_dsl_indexer_k_cache_update` on the same
    `T` tokens, in one launch.

    `q` `[T, 32, 128]` bf16, `position_ids` `[T]` int32, `cos_sin` the `[max_pos, 64]` fp32
    rope table (`[cos(32) | sin(32)]`; the K update reads it as `[max_pos, 2, 32]`), `wq`
    `[T, 32]` fp32: the q side writes `q_fp8_out` and `weights_out` as that op does.
    `indexer_k` `[T, 128]` bf16 (8-byte aligned), `norm_weight` / `norm_bias` fp32 `[128]`,
    `k_cache` the `[blocks, page, 1, 132]` uint8 cache view, `slot_mapping_fp8` /
    `slot_mapping_scale` int64 (the first `T` read): the K update writes the cache as that op
    does (rows with a negative slot skipped). Returns `(q_fp8_out, weights_out, k_cache)`.
    Every byte equals the two ops' own launches.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The MTP3 / MTP5 wide decode steps: 32 sequences x 4 / 6 tokens, fewer sequences
    # (17 x 4, 25 x 6), and odd counts on each side of the 128-row boundary. The pool is
    # every layer's block-major allocation (a strided per-layer view); padded rows carry
    # slot -1. One dense single-layer pool takes the contiguous launcher.
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP5 full decode batch: 32 x 6 tokens", spec=dict(tokens=192, pad_rows=0, pool_layers=78)),
        Cell(why="MTP3 full decode batch: 32 x 4 tokens", spec=dict(tokens=128, pad_rows=0, pool_layers=78)),
        Cell(why="MTP5 at 25 sequences (150 tokens)", spec=dict(tokens=150, pad_rows=0, pool_layers=78)),
        Cell(why="MTP3 at 17 sequences (68 tokens), padding rows", spec=dict(tokens=68, pad_rows=5, pool_layers=78)),
        Cell(why="65 tokens: one past the narrow decode rows", spec=dict(tokens=65, pad_rows=0, pool_layers=78)),
        Cell(why="97 tokens: odd, inside the 128 cell", spec=dict(tokens=97, pad_rows=0, pool_layers=78)),
        Cell(why="129 tokens: one past 128", spec=dict(tokens=129, pad_rows=3, pool_layers=78)),
        Cell(why="191 tokens: one short of 192", spec=dict(tokens=191, pad_rows=0, pool_layers=78)),
        Cell(why="a dense single-layer pool (contiguous launcher)", spec=dict(tokens=192, pad_rows=4, pool_layers=1)),
    )  # fmt: skip

    note = """
    `build(device_id, is_neox=..., cache_contiguous=...)` compiles one launcher; the target
    builds both cache addressing modes at its first forward (as the K update's) and passes the
    one matching the layer's view. Rows are a runtime value. The keys must be 8-byte aligned
    (the launcher only has the vectorised loads; `__call__` raises otherwise) and the slot
    and position ranges are not checked on the device: the K update's `validate=True`
    host check belongs on its own first eager call. Only interleaved rope is certified.
    """

    def __call__(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSideKCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = _q_kernel.validate_abi(q, position_ids, cos_sin, wq, q_fp8_out, weights_out)
        if (
            tuple(indexer_k.shape) != (tokens, _HEAD_DIM)
            or indexer_k.dtype != torch.bfloat16
            or not indexer_k.is_contiguous()
            or indexer_k.data_ptr() % 8
            or k_cache.dim() != 4
            or k_cache.stride(-1) != 1
            or slot_mapping_fp8.shape[0] < tokens
            or slot_mapping_scale.shape[0] < tokens
        ):
            raise ValueError(
                f"q side | K update: indexer_k {tuple(indexer_k.shape)} {indexer_k.dtype} "
                f"(8-byte aligned contiguous [{tokens}, {_HEAD_DIM}] bf16 needed), k_cache "
                f"{tuple(k_cache.shape)} {tuple(k_cache.stride())}, slots "
                f"{slot_mapping_fp8.shape[0]} / {slot_mapping_scale.shape[0]}"
            )
        compiled.fn(
            q,
            position_ids,
            cos_sin,
            wq,
            q_fp8_out,
            weights_out,
            indexer_k,
            norm_weight,
            norm_bias,
            cos_sin.view(-1, 2, _HALF),
            k_cache,
            slot_mapping_fp8[:tokens],
            slot_mapping_scale[:tokens],
            tokens,
            float(weight_scale),
            int(k_cache.stride(0)),
            int(k_cache.stride(1)),
            int(k_cache.stride(2)),
            int(k_cache.stride(3)),
            cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        )
        return q_fp8_out, weights_out, k_cache

    def reference(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSideKCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The two ops' references: fresh `(q_fp8, weights)` (bit-exact) and the cache image
        after the write (a fresh contiguous tensor; codes within one e4m3 step, see the K
        update's). The outputs, `k_cache` and `compiled` are not modified."""
        q_fp8, weights = _q_op.reference(
            q, position_ids, cos_sin, wq, weight_scale, q_fp8_out, weights_out,
            is_neox=is_neox, compiled=None,
        )  # fmt: skip
        cache = _k_op.reference(
            indexer_k, norm_weight, norm_bias, position_ids, cos_sin.view(-1, 2, _HALF), k_cache,
            slot_mapping_fp8, slot_mapping_scale, int(q.shape[0]), is_neox=is_neox, compiled=None,
        )  # fmt: skip
        return q_fp8, weights, cache

    def is_valid(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSideKCache,
    ) -> None:
        # The two ops' own guards: the rope mode and cache addressing are compile-time
        # constants the launcher does not check; positions and slots are read unchecked.
        assert compiled.is_neox == bool(is_neox), (
            f"compiled for is_neox={compiled.is_neox}, called with is_neox={is_neox}"
        )
        _q_op.is_valid(
            q, position_ids, cos_sin, wq, weight_scale, q_fp8_out, weights_out,
            is_neox=is_neox, compiled=compiled,
        )  # fmt: skip
        _k_op.is_valid(
            indexer_k, norm_weight, norm_bias, position_ids, cos_sin.view(-1, 2, _HALF), k_cache,
            slot_mapping_fp8, slot_mapping_scale, int(q.shape[0]), is_neox=is_neox,
            compiled=compiled,
        )  # fmt: skip


cute_dsl_indexer_q_side_k_cache = CuteDslIndexerQSideKCache()
