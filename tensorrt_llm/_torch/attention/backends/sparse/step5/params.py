# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-5 sparse-indexer cache geometry, resolved from the checkpoint config.

Step-5 interleaves sliding-window layers with full-attention layers. The
full-attention layers run sparse GQA: a DSA-style indexer scores per-region
summaries of a one-head proxy key and selects the regions each query attends
to. The indexer's cached state per full-attention layer is

* the per-token bf16 proxy key ``k`` and its per-dimension softmax logits
  ``z`` (one K head, ``proxy_dim`` wide each), and
* one fp8 e4m3 summary per completed region of ``region_block_size`` tokens:
  the per-dimension softmax(z)-weighted mean of the region's ``k``.

Everything here is read from ``text_config`` (``layer_types``,
``num_hidden_layers`` and ``sparse_config``) so the cache and the indexer
cannot disagree on a width. The block counts a sliding-window pool needs at the
engine's limits are here too, as plain functions of those limits.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch

# The cache stores the indexer state at these dtypes whatever the main K/V
# dtype is: they are kernel contracts, not checkpoint weight dtypes.
STEP5_INDEX_KZ_DTYPE: Final = torch.bfloat16
STEP5_INDEX_SUMMARY_DTYPE: Final = torch.float8_e4m3fn
# The per-token row is [k | z]: index 0 is the proxy key, 1 its softmax logits.
STEP5_INDEX_KZ_PLANES: Final = 2
STEP5_INDEX_KEY_PLANE: Final = 0
STEP5_INDEX_Z_PLANE: Final = 1


def step5_kv_head_dim(head_dim: int) -> int:
    """Width of a K/V head in the cache: ``head_dim`` zero-padded to a power of two.

    Step-5's heads are 192 wide. The paged-context FMHA on sm_103 has no head-192
    kernel (trtllm-gen takes 32, 64, 80, 128 or a multiple of 128) and the sparse
    GQA kernel takes power-of-two widths, so both attention kinds run on heads
    zero-padded to 256 and their pools hold 256-wide heads. Zero pad columns add
    nothing to a dot product or to the attention output.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    return 1 << (head_dim - 1).bit_length()


def step5_window_blocks(window: int, tokens_per_block: int, headroom: int) -> int:
    """Blocks one request holds in a ``window``-token life cycle while it decodes.

    The live interval is ``window + headroom - 1`` tokens (the window plus the
    tokens a step appends past the last committed one); at the worst page
    offset it touches one block more than it fills.
    """
    if window <= 0 or tokens_per_block <= 0:
        raise ValueError(
            f"window and tokens_per_block must be positive, got {window} and {tokens_per_block}"
        )
    return -(-(window + headroom - 2) // tokens_per_block) + 1


def step5_sliding_pool_blocks(
    *,
    max_batch_size: int,
    window: int,
    chunk_tokens: int,
    tokens_per_block: int,
    headroom: int,
    beam_width: int = 1,
) -> int:
    """Blocks a ``window``-token pool needs to serve the engine's limits.

    ``max_batch_size`` resident requests each hold their window
    (:func:`step5_window_blocks`, per beam), and a step's context tokens -- at
    most ``chunk_tokens`` (``max_num_tokens``) over all context requests -- are
    all written before the window slides past them, so they add their own
    blocks on top: a request prefilling a ``chunk_tokens`` chunk behind a full
    window fits beside ``max_batch_size - 1`` decoding requests.
    """
    if max_batch_size <= 0 or chunk_tokens < 0:
        raise ValueError(
            f"max_batch_size must be positive and chunk_tokens non-negative, got "
            f"{max_batch_size} and {chunk_tokens}"
        )
    per_request = max(1, beam_width) * step5_window_blocks(window, tokens_per_block, headroom)
    return max_batch_size * per_request + -(-chunk_tokens // tokens_per_block)


def step5_text_config(config: Any) -> Any:
    """The text config of a Step-5 checkpoint, flat or composite.

    The vision-language checkpoint nests the decoder's fields under
    ``text_config``; a flattened config carries them at the top level.
    """
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def _field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


@dataclass(frozen=True)
class Step5IndexerCacheGeometry:
    """Which layers carry indexer state, and how wide that state is."""

    # Global ids of the decoder layers that run the sparse indexer.
    sparse_layer_ids: tuple[int, ...]
    # Width of the indexer's proxy key, z logits and region summary.
    proxy_dim: int
    # Tokens compressed into one summary.
    region_block_size: int
    # Number of decoder layers (``num_hidden_layers``; MTP layers excluded).
    num_layers: int

    def __post_init__(self) -> None:
        if self.proxy_dim <= 0 or self.region_block_size <= 0:
            raise ValueError(
                "Step-5 sparse_config needs a positive proxy_dim and region_block_size, "
                f"got {self.proxy_dim} and {self.region_block_size}"
            )
        summary_bytes = self.proxy_dim * STEP5_INDEX_SUMMARY_DTYPE.itemsize
        if summary_bytes % self.region_block_size:
            raise ValueError(
                f"A {self.proxy_dim}-wide fp8 summary per {self.region_block_size} tokens "
                "is not a whole number of bytes per token"
            )

    @classmethod
    def from_config(cls, config: Any) -> "Step5IndexerCacheGeometry":
        """Resolve the geometry from a Step-5 config (flat or composite)."""
        text = step5_text_config(config)
        sparse_config = getattr(text, "sparse_config", None)
        if sparse_config is None:
            raise ValueError("A Step-5 config needs text_config.sparse_config")
        num_layers = getattr(text, "num_hidden_layers", None)
        layer_types = getattr(text, "layer_types", None)
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"Step-5 num_hidden_layers must be positive, got {num_layers!r}")
        if not layer_types or len(layer_types) < num_layers:
            raise ValueError(
                f"Step-5 layer_types must name each of the {num_layers} decoder layers, "
                f"got {0 if not layer_types else len(layer_types)} entries"
            )
        num_k_heads = int(_field(sparse_config, "sparse_indexer_num_k_heads") or 1)
        if num_k_heads != 1:
            raise NotImplementedError(
                f"The Step-5 indexer cache holds one proxy-key head, got {num_k_heads}"
            )
        sparse_layer_ids: tuple[int, ...] = ()
        if _field(sparse_config, "enabled") is not False:
            sparse_types = set(_field(sparse_config, "apply_to_layer_types") or ())
            sparse_layer_ids = tuple(
                layer_idx
                for layer_idx in range(num_layers)
                if getattr(layer_types[layer_idx], "name", str(layer_types[layer_idx]))
                in sparse_types
            )
        return cls(
            sparse_layer_ids=sparse_layer_ids,
            proxy_dim=int(_field(sparse_config, "proxy_dim")),
            region_block_size=int(_field(sparse_config, "region_block_size")),
            num_layers=num_layers,
        )

    @property
    def kz_bytes_per_token(self) -> int:
        """Bytes of one token's ``[k | z]`` row."""
        return STEP5_INDEX_KZ_PLANES * self.proxy_dim * STEP5_INDEX_KZ_DTYPE.itemsize

    @property
    def summary_bytes_per_token(self) -> int:
        """Bytes of region summary one token accounts for."""
        return self.proxy_dim * STEP5_INDEX_SUMMARY_DTYPE.itemsize // self.region_block_size

    @property
    def bytes_per_token(self) -> int:
        """All indexer state one token of one sparse layer costs."""
        return self.kz_bytes_per_token + self.summary_bytes_per_token

    def regions_per_block(self, tokens_per_block: int) -> int:
        """Regions in one KV block; a region never straddles two blocks."""
        if tokens_per_block % self.region_block_size:
            raise ValueError(
                f"tokens_per_block {tokens_per_block} must be a multiple of the Step-5 "
                f"region_block_size {self.region_block_size}, so that a region never "
                "straddles two KV blocks"
            )
        return tokens_per_block // self.region_block_size

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.sparse_layer_ids


__all__ = [
    "STEP5_INDEX_KEY_PLANE",
    "STEP5_INDEX_KZ_DTYPE",
    "STEP5_INDEX_KZ_PLANES",
    "STEP5_INDEX_SUMMARY_DTYPE",
    "STEP5_INDEX_Z_PLANE",
    "Step5IndexerCacheGeometry",
    "step5_kv_head_dim",
    "step5_sliding_pool_blocks",
    "step5_text_config",
    "step5_window_blocks",
]
