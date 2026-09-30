# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 layer ownership; ratio one denotes an uncompressed global cache."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import torch

from ..params import SparseBackendForwardArgs, SparseMetadataParams, SparseParams

if TYPE_CHECKING:
    from .metadata import CSA2TrtllmMetadata


class CSA2Mode(Enum):
    SWA = "swa"
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass(frozen=True)
class CSA2Layer:
    layer_idx: int
    compress_ratio: int
    kv_source: int | None
    index_source: int | None
    candidate_source: int | None

    @property
    def mode(self) -> CSA2Mode:
        if self.kv_source is None:
            return CSA2Mode.SWA
        if self.kv_source == self.layer_idx:
            return CSA2Mode.FULL
        if self.index_source == self.layer_idx:
            return CSA2Mode.REINDEX
        return CSA2Mode.REUSE


@dataclass(frozen=True)
class CSA2Layout:
    """Model-owned static layout, including optional SWA-only draft layers.

    Source IDs use global model-layer numbering. Consumers must reside with
    their sources, or the executor must explicitly transfer shared state.
    """

    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int | None = None
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_topk: int = 512
    window_size: int = 128

    @classmethod
    def from_hf_config(cls, config: object) -> CSA2Layout:
        """Read CSA2 geometry from either the wrapper or text checkpoint config."""
        if config is None:
            raise ValueError("CSA2 requires a checkpoint text configuration")
        text = (
            config.get("text_config", config)
            if isinstance(config, dict)
            else getattr(config, "text_config", config)
        )

        def value(name: str):
            return text[name] if isinstance(text, dict) else getattr(text, name)

        return cls(
            tuple(value("compress_ratios")),
            tuple(value("kv_source_layer_ids")),
            tuple(value("index_source_layer_ids")),
            value("candidate_source_layer_id"),
            value("candidate_topk_blocks"),
            value("candidate_block_size"),
            value("index_topk"),
            value("sliding_window"),
        )

    def __post_init__(self) -> None:
        if not self.compress_ratios or any(r not in (0, 1, 2) for r in self.compress_ratios):
            raise ValueError("CSA2 compression ratios must be 0, 1 or 2")
        for ids in (self.kv_source_layer_ids, self.index_source_layer_ids):
            if tuple(sorted(set(ids))) != ids:
                raise ValueError("CSA2 source IDs must be unique and increasing")
            if any(i < 0 or i >= len(self.compress_ratios) for i in ids):
                raise ValueError("CSA2 source ID is outside the layer range")
            if any(self.compress_ratios[i] == 0 for i in ids):
                raise ValueError("SWA-only layers cannot own global KV or indices")
        if not set(self.kv_source_layer_ids).issubset(self.index_source_layer_ids):
            raise ValueError("Every CSA2 KV source must also be an index source")
        if (
            min(
                self.candidate_topk_blocks,
                self.candidate_block_size,
                self.index_topk,
                self.window_size,
            )
            <= 0
        ):
            raise ValueError("CSA2 window and selection sizes must be positive")
        candidate = self.candidate_source_layer_id
        if candidate is not None:
            if candidate not in self.kv_source_layer_ids or self.compress_ratios[candidate] != 1:
                raise ValueError("The candidate source must own an uncompressed global cache")
            if candidate != self.kv_source_layer_ids[-1]:
                raise ValueError("Candidate consumers must share the candidate source's KV")
        kv_source = None
        index_source = None
        for i, ratio in enumerate(self.compress_ratios):
            if ratio == 0:
                kv_source = index_source = None
                continue
            if i in self.kv_source_layer_ids:
                kv_source = i
            if i in self.index_source_layer_ids:
                index_source = i
            if kv_source is None or index_source is None:
                raise ValueError(f"CSA2 layer {i} has no preceding KV/index source")
            if ratio != self.compress_ratios[kv_source]:
                raise ValueError(f"CSA2 layer {i} changes its source's compression ratio")

    def layer(self, layer_idx: int) -> CSA2Layer:
        if not 0 <= layer_idx < len(self.compress_ratios):
            raise ValueError("CSA2 layer is outside the configured range")
        ratio = self.compress_ratios[layer_idx]
        if ratio == 0:
            return CSA2Layer(layer_idx, ratio, None, None, None)
        kv_source = max(i for i in self.kv_source_layer_ids if i <= layer_idx)
        index_source = max(i for i in self.index_source_layer_ids if i <= layer_idx)
        candidate = self.candidate_source_layer_id
        return CSA2Layer(
            layer_idx,
            ratio,
            kv_source,
            index_source,
            candidate if candidate is not None and layer_idx >= candidate else None,
        )


@dataclass(frozen=True)
class CSA2Params(SparseParams):
    """Lowered parameters for the prepared CSA2 TRTLLM attention backend."""

    algorithm: Literal["csa2"] = field(init=False, default="csa2")
    indices_block_size: int = field(init=False, default=1)
    layout: CSA2Layout | None = None
    compute_backend: str = "auto"
    skip_indexer_for_short_seqs: bool = True
    use_cute_dsl_topk: bool = True
    enable_heuristic_topk: bool = False
    use_self_sampling_topk: bool = True
    use_cute_dsl_paged_mqa_logits: bool = False
    use_gvr_emission: bool = False
    fuse_index_q: bool = False
    use_packed_sparse_attention: bool = False
    fuse_packed_output_rope: bool = False
    # Native E4M3 compute staging on trtllm-gen. ``None`` stages wherever the
    # native path supports it: trtllm-gen without packed attention, and only
    # when an FP8 staging dtype is requested. An explicit ``True`` still demands
    # that path and is rejected elsewhere rather than silently downgraded, and
    # ``False`` pins BF16 staging.
    use_fp8_staging: bool | None = None
    # Score only the published candidate blocks in candidate-consuming (Reindex)
    # indexer layers with DeepGEMM's sparse MQA-logits kernels (SM100, MXFP4
    # index rows). Falls back to dense logits plus candidate masking when the
    # kernels or the geometry are unavailable, and is moot while the candidate
    # pool covers every admitted position.
    use_sparse_candidate_logits: bool = True

    def __post_init__(self) -> None:
        if self.fuse_packed_output_rope and not self.use_packed_sparse_attention:
            raise ValueError("Packed output RoPE fusion requires packed sparse attention")
        if self.use_gvr_emission and (
            not self.enable_heuristic_topk
            or self.use_self_sampling_topk
            or not self.use_cute_dsl_paged_mqa_logits
        ):
            raise ValueError("CSA2 emission requires temporal GVR and CuTe DSL paged MQA")


def select_csa2_backend(sm_version: int) -> str:
    """Match the DSV4 hardware families without confusing SM120 with SM100."""
    if sm_version == 90:
        return "flash_mla"
    if 100 <= sm_version < 110:
        return "trtllm"
    if sm_version in (120, 121):
        return "flashinfer"
    raise ValueError(f"CSA2 attention is unsupported on SM{sm_version}")


@dataclass(kw_only=True, slots=True)
class CSA2BackendForwardArgs(SparseBackendForwardArgs):
    """Selected packed-pool inputs for the standard sparse prediction hook.

    SWA uses CSA2 FP8 rows, main uses CSA2 FP4 rows, both uint8 [pool_rows, bytes].
    swa_indices and inherited topk_indices are [queries, selected] pool-relative
    row indices; -1 denotes padding. topk_indices addresses main_pool.
    """

    swa_pool: torch.Tensor | None = None
    swa_indices: torch.Tensor | None = None
    main_pool: torch.Tensor | None = None
    # Logical MAIN group indices for shared-domain staging, before page mapping.
    main_logical_indices: torch.Tensor | None = None
    # (page_table, page_size, max_positions, requests, visible): when present,
    # ``topk_indices`` hold logical positions that the native staging kernels
    # map to physical rows themselves.
    main_mapping: tuple | None = None
    # (kv_source, index_source, query_start): layers with equal groups share
    # one staging compaction per forward.
    stage_group: tuple | None = None
    state: CSA2ForwardState | None = None
    query_start: int = 0
    output_position_ids: torch.Tensor | None = None
    output_rotary_cos_sin: torch.Tensor | None = None


@dataclass(frozen=True)
class RowTransform:
    """RMSNorm and/or interleaved RoPE applied to BF16 rows while they are written to a cache.

    ``norm`` is ``(weight, eps)``; ``rope`` is ``(positions, cos_sin, rope_dim)`` with int32
    positions per row and a float32 ``[positions, rope_dim]`` table (cos half, sin half)
    rotating the trailing ``rope_dim`` channels in GPT-J pairs.
    """

    norm: tuple[torch.Tensor, float] | None = None
    rope: tuple[torch.Tensor, torch.Tensor, int] | None = None


@dataclass(kw_only=True, slots=True)
class CSA2ForwardState:
    """Projected module inputs and per-forward state consumed by sparse prediction.

    ``swa_kv`` / ``main_kv`` / ``index_k`` hold the rows before the transforms
    named next to them; the cache writes apply those in the quantize launch.
    """

    metadata: CSA2TrtllmMetadata
    swa_kv: torch.Tensor
    index_q: torch.Tensor | None = None
    index_q_scale: torch.Tensor | None = None
    index_weights: torch.Tensor | None = None
    main_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    output_position_ids: torch.Tensor | None = None
    output_rotary_cos_sin: torch.Tensor | None = None
    swa_transform: RowTransform | None = None
    main_transform: RowTransform | None = None
    index_transform: RowTransform | None = None


@dataclass(frozen=True)
class CSA2MetadataParams(SparseMetadataParams):
    """Checkpoint layout shared by runtime metadata and cache ownership."""

    layout: CSA2Layout
