# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cache an index-K producer writes, and the NVFP4 index-K pack after it (shared by the context targets)."""

from typing import Optional, Tuple

import torch

from . import nvfp4_index_k


def index_k_producer_cache(
    kv_cache_manager, index_k_cache: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The E4M3 cache an index-K producer writes, and the NVFP4 index-K cache to pack from it afterwards (or None).

    An NVFP4 index-K layer (indexer_kv_dtype='nvfp4') is written through the cache manager's FP8 sink, which holds
    each token's E4M3 index-K at its main-pool slot for every layer; the layer's NVFP4 pages are packed from the sink
    right after its producer (pack_nvfp4_index_k).
    """
    if not nvfp4_index_k.is_nvfp4_index_cache(index_k_cache):
        return index_k_cache, None
    sink = getattr(kv_cache_manager, "get_nvfp4_index_k_sink", lambda: None)()
    if sink is None:
        raise RuntimeError("MiniMax-M3 NVFP4 index-K needs the cache manager's FP8 index-K sink.")
    return sink, index_k_cache


def pack_nvfp4_index_k(
    nvfp4_index_cache: Optional[torch.Tensor],
    sink: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_tokens: int,
) -> None:
    """Pack the first ``num_tokens`` slots of ``out_cache_loc`` from the sink into the NVFP4 index-K pages (negative
    slots skipped; capture-safe). No-op for an E4M3 index-K cache."""
    if nvfp4_index_cache is not None:
        nvfp4_index_k.write_from_sink(sink, out_cache_loc, num_tokens, nvfp4_index_cache)
