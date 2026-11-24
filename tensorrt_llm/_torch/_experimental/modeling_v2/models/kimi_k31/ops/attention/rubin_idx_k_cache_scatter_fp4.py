# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer: write MXFP4 index keys and their block exponents into the paged FP4 index-key
cache (CUDA).

The FP4 counterpart of ``indexer_k_cache_scatter_op``, over the same compiled op: a page of the FP4
cache holds P rows of 64 key bytes, then the P int32 exponent words (``rubin_idx_fp4_quantize``'s
format). The compiled op is looked up on the first call, so the entry imports on a host without
the built extension.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Bytes of one key: 64 packed E2M1 bytes, and its four UE8M0 exponents as one int32.
DATA_BYTES = 64
SCALE_BYTES = 4
TOKEN_BYTES = DATA_BYTES + SCALE_BYTES


@functools.cache
def _trtllm_ops():
    """``torch.ops.trtllm`` with the compiled ops registered, on the first call."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    return torch.ops.trtllm


def unravel_bytes(flat: torch.Tensor, shape) -> tuple[torch.Tensor, ...]:
    """Flat byte offsets of a [pages, P, 1, 68] cache to its four indices (logical shape)."""
    i3 = flat % shape[3]
    flat = flat // shape[3]
    i2 = flat % shape[2]
    flat = flat // shape[2]
    i1 = flat % shape[1]
    return flat // shape[1], i1, i2, i3


def _spec(num_tokens, **overrides) -> dict:
    spec = dict(
        num_tokens=num_tokens,
        tokens_per_block=64,
        num_pages=64,
        page_stride_pad=0,
        slot_rows=None,
        skipped_every=None,
    )
    spec.update(overrides)
    return spec


class RubinIdxKCacheScatterFp4(OpWrapper):
    """For ``t < num_tokens``: the 64 bytes of ``k_fp4[t]`` go to flat byte offsets
    ``slot_mapping_fp4[t] + [0, 64)`` and the 4 bytes of ``k_scale[t]`` to
    ``slot_mapping_scale[t] + [0, 4)`` of ``k_cache`` (uint8 ``[pages, P, 1, 68]``).

    A token at slot ``s`` of page ``p`` has its key bytes at ``p * P * 68 + s * 64`` and its
    exponent word at ``p * P * 68 + P * 64 + s * 4``. A flat offset is unravelled over the cache's
    logical shape and stored through its strides, so a page view with a padded page stride is
    addressed correctly. A token whose key or exponent offset is negative is skipped: helix8
    decode steps pass -1 for the rows another rank owns. ``k_fp4`` is int8 or uint8 ``[T, 64]``
    (packed E2M1 pairs), ``k_scale`` int32 ``[T, 1]`` (four UE8M0 exponents), the slot mappings
    int64 ``[>= num_tokens]``. Writes in place; returns None.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token", spec=_spec(1)),
        Cell(why="a DFlash2 k7 verify step of 4 requests: 32 tokens", spec=_spec(32)),
        Cell(why="a DEP16 k7 verify step at batch 12: 96 tokens", spec=_spec(96)),
        Cell(why="a 3000-token prefill chunk over random pages", spec=_spec(3000)),
        Cell(
            why="a padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(700, page_stride_pad=256),
        ),
        Cell(
            why="num_tokens below the slot-mapping length: rows past it are left alone",
            spec=_spec(100, slot_rows=160),
        ),
        Cell(
            why="a helix8 decode step: the rows another rank owns have slot -1 and are skipped",
            spec=_spec(48, skipped_every=2),
        ),
    )
    note = """
    The Rubin default target calls this once per DSA layer per step when its index-key cache is
    FP4, with the step's FP4 index keys (``rubin_idx_fp4_quantize`` of the normalized wk
    projection), their exponent words, the layer's FP4 index-key cache view, the metadata's slot
    mappings for the 68-byte layout and the step's token count. Byte copies only: exact and
    deterministic; distinct tokens must map to distinct slots. Captures in a CUDA graph.
    """

    def __call__(
        self,
        k_fp4: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp4: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        _trtllm_ops().indexer_k_cache_scatter_op(
            k_fp4, k_scale, k_cache, slot_mapping_fp4, slot_mapping_scale, num_tokens
        )

    def reference(
        self,
        k_fp4: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp4: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Byte scatter through the logical shape, in place; a token with a negative slot is
        skipped."""
        shape = k_cache.shape
        keys = slot_mapping_fp4[:num_tokens].long()
        scales = slot_mapping_scale[:num_tokens].long()
        kept = (keys >= 0) & (scales >= 0)
        data = k_fp4[:num_tokens].contiguous().view(torch.uint8)[kept]
        flat = keys[kept, None] + torch.arange(DATA_BYTES, device=k_cache.device)
        k_cache[unravel_bytes(flat, shape)] = data
        scale = k_scale[:num_tokens].contiguous().view(torch.uint8).reshape(num_tokens, SCALE_BYTES)
        flat = scales[kept, None] + torch.arange(SCALE_BYTES, device=k_cache.device)
        k_cache[unravel_bytes(flat, shape)] = scale[kept]

    def is_valid(
        self,
        k_fp4: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp4: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        # The op copies any 1-byte keys and any 4-byte scales: FP8 keys or their fp32 scales go in
        # as bytes and are read back as E2M1 codes and exponents.
        assert k_fp4.dtype in (torch.int8, torch.uint8) and k_fp4.shape[1] == DATA_BYTES, (
            f"k_fp4 must be packed E2M1 pairs [T, 64], got {k_fp4.dtype} {tuple(k_fp4.shape)}"
        )
        assert k_scale.dtype == torch.int32, (
            f"k_scale must be int32 exponent words, got {k_scale.dtype}"
        )
        assert k_cache.shape[2:] == (1, TOKEN_BYTES), (
            f"k_cache must be [pages, P, 1, 68], got {tuple(k_cache.shape)}"
        )
        assert slot_mapping_fp4.shape[0] >= num_tokens and k_fp4.shape[0] >= num_tokens, (
            "num_tokens exceeds the slot mappings or the keys"
        )
        # A token with a negative slot is skipped; the others must address the 68-byte layout.
        keys = slot_mapping_fp4[:num_tokens].long()
        scales = slot_mapping_scale[:num_tokens].long()
        kept = (keys >= 0) & (scales >= 0)
        keys, scales = keys[kept], scales[kept]
        if keys.numel() == 0:
            return
        # Offsets past the cache's logical size write into whatever follows it.
        page_bytes = k_cache.shape[1] * TOKEN_BYTES
        logical = k_cache.shape[0] * page_bytes
        assert int(keys.max()) + DATA_BYTES <= logical, "a key slot past the cache"
        assert int(scales.max()) + SCALE_BYTES <= logical, "a scale slot past the cache"
        # Slot mappings of another page layout (the FP8 cache's 132-byte tokens) stay inside the
        # cache but put the bytes across key rows and exponent words.
        key_area = k_cache.shape[1] * DATA_BYTES
        key_at = keys % page_bytes
        scale_at = scales % page_bytes - key_area
        assert bool(((key_at % DATA_BYTES == 0) & (key_at < key_area)).all()), (
            "a key slot off the 64-byte key rows of its page"
        )
        assert bool(((scale_at >= 0) & (scale_at % SCALE_BYTES == 0)).all()), (
            "a scale slot off the exponent words of its page"
        )


rubin_idx_k_cache_scatter_fp4 = RubinIdxKCacheScatterFp4()
