# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What one forward step looks like, derived once and shared by every layer of the step.

The engine prepares the attention metadata (``TrtllmAttentionMetadata``) and, inside it,
the hybrid-cache metadata of this model (``mamba_metadata``, the class the shell declares
as ``mamba_metadata_cls``): device lengths, the GLM block tables and the host schedules of
the context requests. Everything here reads those prepared fields; nothing reconstructs
cache tables or lengths, so a step captured in a CUDA graph replays on the refreshed
buffers.

A step's rows are packed context first: ``num_ctx_tokens`` rows of ``num_contexts``
context requests, then the generation rows, ``gen_tokens_per_request`` per generation
request (1 for plain decode, 1 + the draft length while MTP verifies drafts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._utils import async_tensor_h2d

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"
DENSE_MLP = "dense"
SPARSE_MLP = "sparse"


@dataclass(frozen=True)
class LayerSchedule:
    """The checkpoint's per-layer attention and MLP kinds (``layer_types``,
    ``mlp_layer_types``), cross-checked against the redundant config fields."""

    attention: tuple[str, ...]
    mlp: tuple[str, ...]

    @property
    def num_layers(self) -> int:
        return len(self.attention)

    @classmethod
    def from_text_config(cls, text: PretrainedConfig) -> "LayerSchedule":
        num_layers = int(text.num_hidden_layers)
        attention = tuple(text.layer_types)
        mlp = tuple(text.mlp_layer_types)
        for name, values, allowed in (
            ("layer_types", attention, {LINEAR_ATTENTION, SPARSE_ATTENTION}),
            ("mlp_layer_types", mlp, {DENSE_MLP, SPARSE_MLP}),
        ):
            if len(values) != num_layers or set(values) - allowed:
                raise ValueError(f"{name} {values} does not list {num_layers} known layer kinds")
        schedule = cls(attention, mlp)
        linear = dict(getattr(text, "linear_attn_config", None) or {})
        kda = tuple(i for i, t in enumerate(attention) if t == LINEAR_ATTENTION)
        if linear.get("kda_layers") is not None and tuple(linear["kda_layers"]) != kda:
            raise ValueError("linear_attn_config.kda_layers disagrees with layer_types")
        first_dense = getattr(text, "first_k_dense_replace", None)
        dense = tuple(i for i, t in enumerate(mlp) if t == DENSE_MLP)
        if first_dense is not None and dense != tuple(range(int(first_dense))):
            raise ValueError("mlp_layer_types disagrees with first_k_dense_replace")
        return schedule


@dataclass(frozen=True)
class ContextRows:
    """Device row schedule of the packed context tokens of one step.

    ``positions[i]`` is packed context token ``i``'s cache position and ``request_ids[i]``
    its request (the batch row of its block table). ``final_positions`` /
    ``final_request_ids`` list the pool-final positions this step completes, one per pool,
    so pool-key refreshes never race.
    """

    positions: torch.Tensor  # [num_ctx_tokens] int64
    request_ids: torch.Tensor  # [num_ctx_tokens] int32
    final_positions: torch.Tensor  # [F] int64
    final_request_ids: torch.Tensor  # [F] int32

    @classmethod
    def build(
        cls, cu_seqlens: Sequence[int], cached_lens: Sequence[int], kpool: int, device: torch.device
    ) -> "ContextRows":
        positions: list[int] = []
        request_ids: list[int] = []
        finals: list[int] = []
        final_ids: list[int] = []
        for i in range(len(cu_seqlens) - 1):
            length = int(cu_seqlens[i + 1]) - int(cu_seqlens[i])
            cached = int(cached_lens[i])
            positions.extend(range(cached, cached + length))
            request_ids.extend([i] * length)
            first_final = cached + (-(cached + 1)) % kpool
            for pos in range(first_final, cached + length, kpool):
                finals.append(pos)
                final_ids.append(i)
        # Pinned non-blocking copies do not wait for the GPU work already queued.
        return cls(
            positions=async_tensor_h2d(positions, torch.long, device),
            request_ids=async_tensor_h2d(request_ids, torch.int32, device),
            final_positions=async_tensor_h2d(finals, torch.long, device),
            final_request_ids=async_tensor_h2d(final_ids, torch.int32, device),
        )


@dataclass(frozen=True)
class GenerationRows:
    """Device row schedule of the generation requests of one step, built from the live
    ``kv_lens`` only (so CUDA graphs replay it on refreshed metadata).

    Row ``b * T + t`` is generation request ``b``'s token ``t``: ``positions`` is ``[B, T]``,
    ``visible`` (the row's visible prefix length) is ``[B * T]``, and ``tables`` is the
    request's block table repeated per row, as the pool scoring and expansion kernels address
    it.
    """

    positions: torch.Tensor  # [B, T] int64
    visible: torch.Tensor  # [B * T] int64
    tables: torch.Tensor  # [B * T, max pages] int64

    @classmethod
    def build(
        cls, kv_lens: torch.Tensor, block_tables: torch.Tensor, tokens: int
    ) -> "GenerationRows":
        steps = torch.arange(tokens, device=kv_lens.device)
        positions = (kv_lens - tokens).unsqueeze(1) + steps
        visible = (positions + 1).reshape(-1)
        return cls(positions, visible, block_tables.repeat_interleave(tokens, dim=0))


@dataclass
class StepSchedule:
    """Per-step facts and row schedules shared by every layer of one forward."""

    num_contexts: int
    num_ctx_tokens: int
    num_generations: int
    #: Tokens per generation request: 1 for plain decode, 1 + draft length while verifying.
    gen_tokens_per_request: int
    #: Host packed offsets and cached lengths of the context requests (prefill only).
    ctx_cu_seqlens: list[int]
    cached_lens: list[int]
    #: Live per-request visible lengths (cached + this step's tokens) on the device.
    kv_lens: torch.Tensor
    #: The GLM block tables of the batch, ``[batch, max pages]`` int64 slot ids.
    block_tables: torch.Tensor
    metadata: AttentionMetadata
    _context_rows: dict[int, ContextRows] = field(default_factory=dict)
    _generation_rows: Optional[GenerationRows] = None

    @property
    def num_generation_tokens(self) -> int:
        return self.num_generations * self.gen_tokens_per_request

    def context_rows(self, kpool: int, device: torch.device) -> ContextRows:
        rows = self._context_rows.get(kpool)
        if rows is None:
            rows = ContextRows.build(
                self.ctx_cu_seqlens, self.cached_lens[: self.num_contexts], kpool, device
            )
            self._context_rows[kpool] = rows
        return rows

    def generation_rows(self) -> GenerationRows:
        if self._generation_rows is None:
            self._generation_rows = GenerationRows.build(
                self.kv_lens[self.num_contexts :],
                self.block_tables[self.num_contexts : self.num_contexts + self.num_generations],
                self.gen_tokens_per_request,
            )
        return self._generation_rows

    @classmethod
    def build(cls, attn_metadata: AttentionMetadata) -> "StepSchedule":
        """The step's schedule from the engine-prepared metadata.

        Prefill reads the host context schedules the metadata prepared before the forward;
        decode and verify read ``kv_lens_cuda``, which carries the overlap scheduler's
        corrections and MTP's rewinds on the device.
        """
        if attn_metadata.kv_cache_manager is None:
            raise ValueError("GLM-5.3-Flash needs a KV cache manager")
        hybrid = attn_metadata.mamba_metadata
        if hybrid is None or getattr(hybrid, "glm_block_tables", None) is None:
            raise RuntimeError(
                "GLM-5.3-Flash needs the prepared GLM hybrid-cache metadata (glm_block_tables); "
                "prepare the attention metadata before eager execution or CUDA graph capture"
            )
        live = attn_metadata.kv_lens_cuda
        if live is None:
            raise ValueError("GLM-5.3-Flash needs the prepared kv_lens_cuda")
        batch = int(attn_metadata.seq_lens.shape[0])
        num_contexts = int(attn_metadata.num_contexts)
        num_generations = batch - num_contexts
        num_ctx_tokens = int(attn_metadata.num_ctx_tokens)
        tokens = 1
        if num_generations > 0:
            gen_tokens = int(attn_metadata.num_tokens) - num_ctx_tokens
            if gen_tokens <= 0 or gen_tokens % num_generations:
                raise ValueError(
                    f"{gen_tokens} generation tokens do not split evenly over "
                    f"{num_generations} generation requests"
                )
            tokens = gen_tokens // num_generations
        return cls(
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            num_generations=num_generations,
            gen_tokens_per_request=tokens,
            ctx_cu_seqlens=hybrid.glm_ctx_cu_seqlens,
            cached_lens=hybrid.glm_cached_lens_host,
            kv_lens=live[:batch],
            block_tables=hybrid.glm_block_tables[:batch],
            metadata=attn_metadata,
        )
