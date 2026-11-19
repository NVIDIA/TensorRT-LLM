# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 DSA indexer: gather a prefill chunk's FP8 index keys and scales from the paged cache."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.gather_k_cache_paged import (
    k31_gather_k_cache_paged as _k31_gather_k_cache_paged,
)


def _chunk(tokens: int, why: str, **rest) -> Cell:
    """An indexer K cache of 64-token pages, 132 bytes per token (128 FP8 key bytes + one FP32
    scale), whose pages sit at a page stride wider than a page (one layer of a multi-layer pool),
    and a chunk of ``tokens`` cached keys."""
    spec = dict(tokens=tokens, tokens_per_block=64, head_dim=128, pages=0, start=0)
    spec.update(rest)
    return Cell(why=why, spec=spec)


class K31GatherKCachePaged(OpWrapper):
    """``(k, scale)`` of tokens ``[k_token_start, k_token_start + num_k_tokens)``: ``k`` FP8 E4M3
    ``[num_k_tokens, head_dim]`` and ``scale`` FP32 ``[num_k_tokens, 1]``, read from the uint8 cache
    view ``k_cache_4d [pages, tokens_per_block, 1, bytes_per_token]`` at the flat byte offsets
    ``slot_mapping_fp8`` / ``slot_mapping_scale`` (int64, one per token; offsets index the cache
    as if its pages were packed, so a page stride wider than a page is followed). A token whose
    slot is negative is skipped: its output rows are not written.

    The same bytes as ``trtllm::indexer_k_cache_gather_op``. One CuTe DSL kernel.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _chunk(1, "one cached key (one partial program)"),
        _chunk(2954, "a 2954-token prefill chunk"),
        _chunk(8192, "an 8k prefill chunk, starting mid-mapping", start=4096),
        _chunk(300, "skipped tokens (negative slots) inside the chunk", holes=True),
        _chunk(16384, "a 16k prefill chunk of cached keys, one wave of CTAs"),
        _chunk(65536, "64k cached keys: more than one wave of CTAs"),
        _chunk(200, "slot offsets off the 16-byte grid (the byte-copy path)", misaligned=True),
    )

    note = """
    Prefill only: the indexer runs it for every prefill chunk with cached keys. Layouts other
    than the one above (more than one head, a token stride other than bytes_per_token, a payload
    width outside 64 / 128 / 256, slot mappings that are not contiguous int64) make the op
    raise. Rows of skipped tokens hold whatever the fresh allocation held, as with the engine
    op. Keys move as 16-byte vectors from 16-byte aligned sources (every slot of the indexer's
    own layout) and byte by byte otherwise; scales as one word from 4-byte aligned sources.
    Programmatic dependent launch follows TRTLLM_ENABLE_PDL (default on): the kernel waits for
    its producer before it reads the slot mappings and, at up to one wave of CTAs, releases its
    dependent once its loads are issued.
    """

    def __call__(
        self,
        k_cache_4d: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_k_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k31_gather_k_cache_paged(
            k_cache_4d, slot_mapping_fp8, slot_mapping_scale, k_token_start, num_k_tokens, head_dim
        )

    def reference(
        self,
        k_cache_4d: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_k_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte gathers in torch; skipped tokens' rows are zero here (the op leaves them
        unwritten, so the test compares only the gathered rows)."""
        pages, tpb, _, per_token = k_cache_4d.shape
        page_bytes = tpb * per_token
        flat = k_cache_4d.as_strided((pages, page_bytes), (k_cache_4d.stride(0), 1))

        def gather(slots: torch.Tensor, width: int) -> torch.Tensor:
            s = slots[k_token_start : k_token_start + num_k_tokens].long()
            ok = s >= 0
            s = s.clamp_min(0)
            rows = flat[s // page_bytes]
            idx = (s % page_bytes).unsqueeze(1) + torch.arange(width, device=s.device)
            return torch.gather(rows, 1, idx) * ok.unsqueeze(1).to(torch.uint8)

        k = gather(slot_mapping_fp8, head_dim)
        scale = gather(slot_mapping_scale, 4)
        return k.view(torch.float8_e4m3fn), scale.view(torch.float32).view(num_k_tokens, 1)

    def is_valid(
        self,
        k_cache_4d: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_k_tokens: int,
        head_dim: int,
    ) -> None:
        # The kernel reads the slot mappings by flat index without bounds: a range past their end
        # reads whatever follows them.
        assert k_token_start + num_k_tokens <= min(
            slot_mapping_fp8.numel(), slot_mapping_scale.numel()
        ), "the token range must lie inside both slot mappings"


k31_gather_k_cache_paged = K31GatherKCachePaged()
