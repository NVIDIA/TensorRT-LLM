# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""specDSA configuration: the compile-time shape of one DSA model.

A ``SpecDsaConfig`` holds what a specDSA kernel is specialized for: the indexer (heads, head dim, K-cache dtype and
page size, compression ratio), the selection (top-K, longest row), the attention (heads, KV row layout, sliding window,
attention sink) and the limits of the two implementations (batch sizes). It is frozen and hashable, validated when it
is built, and independent of torch and CuTe DSL, so it can be imported anywhere.

The kernels of this package are specialized for one spec, ``DS_V4_FLASH``. ``SUPPORTED_SPECS`` lists the specs they were
built and validated for; their structural constants are defined from it and their constructors reject a config that
differs in ``kernel_key``. A model with another top-K, KV layout or indexer is a new spec plus the kernel variants it
needs, not a fork of the kernels (see ``README.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = [
    "DS_V4_FLASH",
    "DS_V4_FLASH_FP4",
    "INDEX_TILE",
    "KNOWN_SHAPES",
    "NUM_SMS",
    "SUPPORTED_SPECS",
    "SpecDsaConfig",
    "check_device",
    "require_built_for",
    "resolve_spec",
    "supported_spec",
]

# SMs of the device the kernels are built for (B200): one persistent CTA per SM.
NUM_SMS = 148
# Compressed positions per indexer MMA tile (a property of the kernels, not of a model).
INDEX_TILE = 128

INDEX_K_DTYPES = ("fp8", "fp4")
KV_FORMATS = ("v4_e4m3_rows",)


@dataclass(frozen=True)
class SpecDsaConfig:
    """The shape of one DSA model, as far as the specDSA kernels depend on it.

    Attributes:
        name: label of the spec (the key of the model tables, e.g. ``"DS-V4-Flash"``); not part of ``kernel_key``.
        index_heads: heads of the indexer.
        index_head_dim: head dim of the indexer.
        index_k_dtype: dtype of the indexer K cache, ``"fp8"`` (one E4M3 byte per element and a float32 scale per
            128 elements) or ``"fp4"`` (packed E2M1 and a UE8M0 exponent per 32 elements), as in the cache manager of
            the DeepSeek-V4 attention backend.
        index_page_tokens: compressed positions per indexer K page.
        compress_ratio: positions of the sequence per compressed (indexer) position.
        topk: positions selected per row.
        max_index_len: longest row, in compressed positions.
        attn_heads: heads of the attention.
        kv_dim: elements (E4M3 bytes) of one KV row.
        kv_format: layout of the KV cache rows, one of ``KV_FORMATS``.
        kv_page_rows: compressed KV rows per page.
        window: rows of the sliding window.
        swa_page_rows: sliding-window rows per page.
        has_sink: whether the attention has a per-head sink.
        megakernel_max_batch: largest batch size of the megakernel (0: no megakernel).
        large_batch: whether the large-batch kernels serve the batch sizes above the megakernel's.
        max_batch: largest batch size.
    """

    name: str
    index_heads: int
    index_head_dim: int
    index_k_dtype: str
    index_page_tokens: int
    compress_ratio: int
    topk: int
    max_index_len: int
    attn_heads: int
    kv_dim: int
    kv_format: str
    kv_page_rows: int
    window: int
    swa_page_rows: int
    has_sink: bool
    megakernel_max_batch: int
    large_batch: bool
    max_batch: int

    def __post_init__(self):
        def bad(msg):
            raise ValueError(f"SpecDsaConfig {self.name!r}: {msg}")

        if not self.name:
            raise ValueError("SpecDsaConfig: name must not be empty")
        for field in (
            "index_heads",
            "index_head_dim",
            "index_page_tokens",
            "compress_ratio",
            "topk",
            "max_index_len",
            "attn_heads",
            "kv_dim",
            "kv_page_rows",
            "window",
            "swa_page_rows",
            "max_batch",
        ):
            if not isinstance(getattr(self, field), int) or getattr(self, field) < 1:
                bad(f"{field} must be a positive int, got {getattr(self, field)!r}")
        if not isinstance(self.megakernel_max_batch, int) or self.megakernel_max_batch < 0:
            bad(f"megakernel_max_batch must be an int >= 0, got {self.megakernel_max_batch!r}")
        if self.index_k_dtype not in INDEX_K_DTYPES:
            bad(f"index_k_dtype must be one of {INDEX_K_DTYPES}, got {self.index_k_dtype!r}")
        if self.index_k_dtype == "fp8" and self.index_head_dim % 128:
            bad(
                "fp8 indexer K (a scale per 128 elements) needs index_head_dim to be a multiple of 128"
            )
        if self.index_k_dtype == "fp4" and self.index_head_dim % 32:
            bad("fp4 indexer K needs index_head_dim to be a multiple of 32")
        if self.kv_format not in KV_FORMATS:
            bad(f"kv_format must be one of {KV_FORMATS}, got {self.kv_format!r}")
        if self.topk > self.max_index_len:
            bad(f"topk {self.topk} exceeds max_index_len {self.max_index_len}")
        if self.max_index_len % self.index_page_tokens:
            bad("max_index_len must be a multiple of index_page_tokens")
        if INDEX_TILE % self.index_page_tokens:
            bad(f"index_page_tokens must divide the indexer tile ({INDEX_TILE})")
        if self.megakernel_max_batch > self.max_batch:
            bad("megakernel_max_batch exceeds max_batch")
        if not self.large_batch and self.max_batch > self.megakernel_max_batch:
            bad("batch sizes above megakernel_max_batch need the large-batch kernels")

    @property
    def shape_key(self) -> tuple[int, int, int, int]:
        """(index heads, index head dim, top-k, compression ratio): the key of the repository's model tables."""
        return (self.index_heads, self.index_head_dim, self.topk, self.compress_ratio)

    @property
    def kernel_key(self) -> tuple:
        """Everything the kernels are specialized for (the spec without its label and batch-size limits)."""
        return (
            self.index_heads,
            self.index_head_dim,
            self.index_k_dtype,
            self.index_page_tokens,
            self.compress_ratio,
            self.topk,
            self.max_index_len,
            self.attn_heads,
            self.kv_dim,
            self.kv_format,
            self.kv_page_rows,
            self.window,
            self.swa_page_rows,
            self.has_sink,
        )

    @property
    def min_index_len(self) -> int:
        """Shortest row: a row selects ``topk`` of its positions."""
        return self.topk

    @property
    def selected_rows(self) -> int:
        """KV rows an attention row reads: the sliding window, then the selection."""
        return self.window + self.topk

    @property
    def pages_per_tile(self) -> int:
        """Indexer K pages of one indexer tile."""
        return INDEX_TILE // self.index_page_tokens

    @property
    def index_page_bytes(self) -> int:
        """Bytes of one indexer K page: the token bytes of the cache manager times the tokens of a page."""
        if self.index_k_dtype == "fp8":
            return self.index_page_tokens * (self.index_head_dim + self.index_head_dim // 128 * 4)
        return self.index_page_tokens * (self.index_head_dim // 2 + self.index_head_dim // 32)

    @property
    def default_softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.kv_dim)


# DeepSeek-V4-Flash, CSA layers: 64 x 128 FP8 indexer over 32-position pages of the 4x-compressed sequence, top-512,
# 64 attention heads over 512-byte E4M3 KV rows, a 128-row sliding window and a per-head sink.
DS_V4_FLASH = SpecDsaConfig(
    name="DS-V4-Flash",
    index_heads=64,
    index_head_dim=128,
    index_k_dtype="fp8",
    index_page_tokens=32,
    compress_ratio=4,
    topk=512,
    max_index_len=1 << 18,
    attn_heads=64,
    kv_dim=512,
    kv_format="v4_e4m3_rows",
    kv_page_rows=32,
    window=128,
    swa_page_rows=128,
    has_sink=True,
    # the megakernel is built for up to 48 rows; at 48 the large-batch kernels are the faster ones
    megakernel_max_batch=47,
    large_batch=True,
    max_batch=1024,
)

# The same model with the MXFP4 indexer K cache of the cache manager (E2M1 data and a UE8M0 scale per 32 elements,
# 68 bytes per position) and an FP4 indexer query with a scale per 32 elements.
# The FP4 K stream is twice as fast per byte, but the megakernel row cost (the epilogue warps' hand-over of every
# row, the teams' selection and attention) does not shrink with it: from 21 rows the large-batch kernels are faster.
DS_V4_FLASH_FP4 = replace(
    DS_V4_FLASH, name="DS-V4-Flash-FP4", index_k_dtype="fp4", megakernel_max_batch=20
)

# The specs the kernels were built and validated for.
SUPPORTED_SPECS = {DS_V4_FLASH.name: DS_V4_FLASH, DS_V4_FLASH_FP4.name: DS_V4_FLASH_FP4}

# Indexer shapes of the DeepSeek models (index heads, index head dim, top-k, compression ratio), as in the model tables
# of the top-K adapter; only the ones in SUPPORTED_SPECS have kernels.
KNOWN_SHAPES = {
    "DS-V3.2": (64, 128, 2048, 1),
    "DS-V4-Flash": (64, 128, 512, 4),
    "DS-V4-Pro": (64, 128, 1024, 4),
}


def _not_supported(what: str) -> NotImplementedError:
    supported = ", ".join(f"{s.name} {s.shape_key}" for s in SUPPORTED_SPECS.values())
    return NotImplementedError(f"specDSA has no kernels for {what}; supported: {supported}")


def resolve_spec(
    index_heads: int,
    index_head_dim: int,
    topk: int,
    compress_ratio: int,
    *,
    index_k_dtype: str = "fp8",
    kv_format: str = "v4_e4m3_rows",
) -> SpecDsaConfig:
    """The supported spec of an indexer shape ``(heads, head dim, top-k, compression ratio)``.

    The shape alone does not identify a model (two models can share it with different KV caches), so the K-cache dtype
    and the KV row format take part in the match.
    """
    shape = (index_heads, index_head_dim, topk, compress_ratio)
    for spec in SUPPORTED_SPECS.values():
        if (
            spec.shape_key == shape
            and spec.index_k_dtype == index_k_dtype
            and spec.kv_format == kv_format
        ):
            return spec
    known = next((n for n, s in KNOWN_SHAPES.items() if s == shape), None)
    raise _not_supported(
        f"the indexer shape {shape} with a {index_k_dtype} K cache and {kv_format!r} KV rows"
        + (f" ({known})" if known else "")
    )


def supported_spec(config: SpecDsaConfig | None = None) -> SpecDsaConfig:
    """``config`` (default: ``DS_V4_FLASH``) if the kernels serve it, else ``NotImplementedError``.

    The kernels serve a config whose ``kernel_key`` is one of ``SUPPORTED_SPECS`` and whose batch-size limits are not
    above that spec's; a config with lower limits is accepted (it routes more batch sizes to the large-batch kernels or
    rejects them).
    """
    if config is None:
        return DS_V4_FLASH
    for spec in SUPPORTED_SPECS.values():
        if spec.kernel_key == config.kernel_key:
            if (
                config.megakernel_max_batch > spec.megakernel_max_batch
                or config.max_batch > spec.max_batch
                or (config.large_batch and not spec.large_batch)
            ):
                raise _not_supported(
                    f"{config.name!r} with batch-size limits above those of {spec.name}"
                )
            return config
    raise _not_supported(f"{config.name!r} {config.shape_key}")


def require_built_for(config: SpecDsaConfig | None, built_for: SpecDsaConfig) -> SpecDsaConfig:
    """The config a kernel object is built with.

    ``built_for`` if ``config`` is None, else ``config`` after checking that it has the same ``kernel_key`` (the kernels
    are specialized for ``built_for``).
    """
    if config is None:
        return built_for
    if config.kernel_key != built_for.kernel_key:
        raise NotImplementedError(
            f"this specDSA kernel is specialized for {built_for.name!r} {built_for.shape_key}, "
            f"got {config.name!r} {config.shape_key}"
        )
    return config


def check_device(device=None) -> tuple[bool, str]:
    """Whether ``device`` can run the kernels: CuTe DSL installed, SM100 with all ``NUM_SMS`` SMs (one CTA per SM)."""
    import torch

    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return False, "nvidia-cutlass-dsl is not available"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    props = torch.cuda.get_device_properties(
        device if device is not None else torch.cuda.current_device()
    )
    if (props.major, props.minor) != (10, 0):
        return False, f"requires SM100, got SM{props.major}{props.minor}"
    if props.multi_processor_count != NUM_SMS:
        return False, f"requires {NUM_SMS} SMs, got {props.multi_processor_count}"
    return True, ""
