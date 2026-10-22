# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Norm, rope, fp8-pack and append the DSA indexer keys to the paged K cache, in one launch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_indexer_k_norm_rope_fp8_cache_update as _kernel

_HEAD_DIM = 128
_ROPE_DIM = 64
_HALF = _ROPE_DIM // 2
_SCALE_BYTES = 4
#: The kernel's LayerNorm epsilon, a compile-time constant (not a parameter).
NORM_EPS = 1.0e-6
_E4M3_MAX = 448.0
_MIN_AMAX = 1.0e-12

#: The two launch shapes the kernel's own wrapper picks between: 2 rows per CTA
#: up to `_SMALL_TOKEN_THRESHOLD` tokens, 8 above.
_ROWS_PER_BLOCK = (_kernel._SMALL_TOKEN_ROWS_PER_BLOCK, _kernel._FUSED_CAT_SCATTER_ROWS_PER_BLOCK)


@dataclass(frozen=True)
class CompiledIndexerKCacheUpdate:
    """One launcher per (rows per CTA, vectorised input loads), and the flags they share."""

    fns: Mapping[tuple[int, bool], Callable]
    is_neox: bool
    cache_contiguous: bool


def build(device_id: int, *, is_neox: bool, cache_contiguous: bool) -> CompiledIndexerKCacheUpdate:
    """Compile the four launchers a step can need: 2 or 8 rows per CTA, scalar or 8-byte loads.

    `cache_contiguous` must describe every cache view the launchers will write:
    the contiguous variant addresses a strided per-layer view as if it were
    dense. Four compiles of a few seconds each, cached per process by the kernel
    module; allocates nothing. ue8m0 scales and the PDL launch are fixed.
    """
    fns = {}
    for rows_per_block in _ROWS_PER_BLOCK:
        for vectorized in (False, True):
            fns[(rows_per_block, vectorized)] = (
                _kernel.compile_indexer_k_norm_rope_fp8_cache_update(
                    device_id=device_id,
                    use_ue8m0=True,
                    is_neox=bool(is_neox),
                    rows_per_block=rows_per_block,
                    vectorized_loads=vectorized,
                    cache_contiguous=bool(cache_contiguous),
                )
            )
    return CompiledIndexerKCacheUpdate(
        fns=fns, is_neox=bool(is_neox), cache_contiguous=bool(cache_contiguous)
    )


class CuteDslIndexerKCacheUpdate(OpWrapper):
    """Turn this step's pre-norm indexer keys into indexer K cache entries, in place.

    For each of the first `num_tokens` rows of `indexer_k` (`[T, 128]` bf16):

    1. LayerNorm over the row (`eps = 1e-6`, fp32 `norm_weight` / `norm_bias`),
       rounded to bf16;
    2. rope on the first 64 dims at `position_ids[t]`, from
       `rotary_cos_sin[pos] = [[cos(32)], [sin(32)]]` (NeoX halves, or
       interleaved pairs), rounded to bf16;
    3. `scale = max(amax, 1e-12) / 448` rounded up to a power of two, and the
       e4m3 codes `row / scale`;
    4. the 128 code bytes go to flat byte `slot_mapping_fp8[t]` of `k_cache`
       and the fp32 scale to flat byte `slot_mapping_scale[t]`, where "flat"
       is the logical row-major index of `k_cache`'s `[blocks, page, 1, D]`
       shape, mapped through its strides. A row with either slot negative is
       skipped.

    `k_cache` is written in place and returned. Nothing else is written.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target writes every layer's keys, every step: decode rows (32
    # sequences x MTP 2 at most, CUDA-graph padded rows carry slot -1) and
    # prefill steps up to max_num_tokens. Its K pool is one block-major
    # allocation of every layer, so a layer's cache is a strided view whose
    # blocks are dense; 32 tokens per page, 132 bytes per token.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, MTP 2 on one sequence: the 2-row-per-CTA launcher",
            spec=dict(tokens=2, pool_layers=78, blocks=64, pad_rows=0, misalign=False),
        ),
        Cell(
            why="the full decode batch with CUDA-graph padding rows at slot -1",
            spec=dict(tokens=64, pool_layers=78, blocks=64, pad_rows=9, misalign=False),
        ),
        Cell(
            why="a prefill past 1024 rows: the 8-row-per-CTA launcher",
            spec=dict(tokens=1500, pool_layers=78, blocks=96, pad_rows=0, misalign=False),
        ),
        Cell(
            why="max_num_tokens prefill: 8192 rows, positions 0..8191",
            spec=dict(tokens=8192, pool_layers=78, blocks=300, pad_rows=0, misalign=False),
        ),
        Cell(
            why="keys not 8-byte aligned take the scalar-load launcher",
            spec=dict(tokens=33, pool_layers=78, blocks=64, pad_rows=0, misalign=True),
        ),
        Cell(
            why="a single-layer pool: a dense view and the contiguous launcher",
            spec=dict(tokens=64, pool_layers=1, blocks=64, pad_rows=4, misalign=False),
        ),
    )

    note = """
    `build(device_id, is_neox=..., cache_contiguous=...)` compiles the four
    launchers once; the target builds them on its first forward (the pool
    layout is an engine value) and passes the result as `compiled`. `__call__`
    picks the launcher on the host: 2 rows per CTA up to 1024 tokens, 8 above,
    and 8-byte loads when `indexer_k` is 8-byte aligned. There is no stock-op
    fallback; every row count runs the kernel.

    `validate=True` makes the kernel's wrapper check slot and position ranges
    on the device, which is a host sync and breaks a CUDA-graph capture: pass
    it on an eager call only. With it off those ranges are unchecked and
    `is_valid` states them.

    Only interleaved rope is certified (the target's mode). The codes are compared
    against the reference within one e4m3 step (see the test): the kernel's
    LayerNorm sums in a different order and uses an approximate rsqrt.
    """

    def __call__(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> torch.Tensor:
        rows_per_block = _kernel._select_rows_per_block(int(num_tokens))
        vectorized = _kernel._can_vectorize_loads(indexer_k)
        _kernel.indexer_k_norm_rope_fp8_cache_update(
            indexer_k,
            norm_weight,
            norm_bias,
            position_ids,
            rotary_cos_sin,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_tokens,
            use_ue8m0=True,
            is_neox=is_neox,
            compiled_fn=compiled.fns[(rows_per_block, vectorized)],
            validate=validate,
        )
        return k_cache

    def reference(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> torch.Tensor:
        """The cache image after the write, as a fresh contiguous tensor shaped like `k_cache`.

        `k_cache` is read (the bytes not written keep their value), not modified.
        LayerNorm and rope run in fp64 and are rounded to bf16 at the two points
        the kernel materialises bf16; the scale and codes are then exact.
        `compiled` and `validate` are not read.
        """
        n = int(num_tokens)
        dev = indexer_k.device
        x = indexer_k[:n].double()
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(
            var + NORM_EPS
        ) * norm_weight.double() + norm_bias.double()
        normed = normed.float().to(torch.bfloat16).double()

        pe, nope = normed[:, :_ROPE_DIM], normed[:, _ROPE_DIM:]
        table = rotary_cos_sin[position_ids[:n].long()].double()  # [n, 2, 32]
        cos, sin = table[:, 0], table[:, 1]
        if is_neox:
            partner = torch.cat([pe[:, _HALF:], pe[:, :_HALF]], dim=-1)
            cos_e, sin_e = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
            sign = torch.cat([-torch.ones(_HALF), torch.ones(_HALF)])
        else:
            partner = pe.reshape(n, _HALF, 2).flip(-1).reshape(n, _ROPE_DIM)
            cos_e, sin_e = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
            sign = torch.tensor([-1.0, 1.0]).repeat(_HALF)
        sign = sign.to(device=dev, dtype=torch.float64)
        pe = (pe * cos_e + sign * partner * sin_e).float().to(torch.bfloat16)
        row = torch.cat([pe.float(), nope.float()], dim=-1)  # [n, 128] fp32, bf16 values

        amax = row.abs().amax(dim=-1)
        floor = torch.tensor(_MIN_AMAX, dtype=torch.float32, device=dev)
        inv_max = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32, device=dev)
        bits = (torch.maximum(amax, floor) * inv_max).contiguous().view(torch.int32)
        exp_bits = bits & 0x7F800000
        exp_bits = torch.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
        scale = exp_bits.view(torch.float32)
        inv_scale = (0x7F000000 - exp_bits).view(torch.float32)
        codes = (row * inv_scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        scale_bytes = scale.contiguous().view(torch.uint8).reshape(n, _SCALE_BYTES)

        out = k_cache.clone(memory_format=torch.contiguous_format)
        flat = out.view(-1)
        fp8_slot = slot_mapping_fp8[:n].long()
        scale_slot = slot_mapping_scale[:n].long()
        live = (fp8_slot >= 0) & (scale_slot >= 0)
        cols = torch.arange(_HEAD_DIM, device=dev)
        flat[(fp8_slot[live].unsqueeze(1) + cols).reshape(-1)] = codes[live].reshape(-1)
        cols = torch.arange(_SCALE_BYTES, device=dev)
        flat[(scale_slot[live].unsqueeze(1) + cols).reshape(-1)] = scale_bytes[live].reshape(-1)
        return out

    def is_valid(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> None:
        # Both are compile-time constants the wrapper does not check against the
        # call: the other rope mode writes plausible wrong keys, and contiguous
        # addressing on a strided per-layer view writes into other layers.
        assert compiled.is_neox == bool(is_neox), (
            f"compiled for is_neox={compiled.is_neox}, called with is_neox={is_neox}"
        )
        assert not compiled.cache_contiguous or k_cache.is_contiguous(), (
            "compiled with contiguous cache addressing but k_cache is a strided view "
            f"(stride {tuple(k_cache.stride())}); the kernel writes outside it"
        )
        # Everything below is what `validate=True` checks on the device; with it
        # off (every call inside a CUDA graph) nothing does.
        n = int(num_tokens)
        assert 0 <= n <= indexer_k.shape[0], (
            f"num_tokens {n} exceeds the {indexer_k.shape[0]} rows of indexer_k"
        )
        if n == 0:
            return
        pos = position_ids[:n]
        assert int(pos.min()) >= 0 and int(pos.max()) < rotary_cos_sin.shape[0], (
            f"a position is outside the {rotary_cos_sin.shape[0]}-row rope table; "
            "the kernel reads past it"
        )
        fp8_slot, scale_slot = slot_mapping_fp8[:n], slot_mapping_scale[:n]
        assert bool(((fp8_slot >= 0) == (scale_slot >= 0)).all()), (
            "slot_mapping_fp8 and slot_mapping_scale disagree on which rows are written"
        )
        size = k_cache.numel()
        live = fp8_slot >= 0
        if bool(live.any()):
            f, s = fp8_slot[live], scale_slot[live]
            assert bool(((f % 4 == 0) & (s % 4 == 0)).all()), (
                "slots must be 4-byte aligned: the kernel stores 32-bit words"
            )
            assert int(f.max()) + _HEAD_DIM <= size and int(s.max()) + _SCALE_BYTES <= size, (
                f"a slot writes past the {size}-byte cache view"
            )


cute_dsl_indexer_k_cache_update = CuteDslIndexerKCacheUpdate()
