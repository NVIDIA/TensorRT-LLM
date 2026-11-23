# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The static-FP8 H3 pipeline with packed QKV exchange on four GPUs."""

import torch

from tensorrt_llm._torch.visual_gen.attention.attention import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent

from ..pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from .output_exchange import exchange_fp8_output, static_fp8_output_scale
from .qkv_exchange import pack_qkv


class Fp8UlyssesAttention(MiniMaxH3Attention):
    _fp8_output_exchange_enabled: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fp8_out_scale_cache_key = None
        self._fp8_out_scale_valid = False
        self._fp8_output_exchange_calls = 0
        self._bf16_output_exchange_calls = 0

    def _validated_fp8_output_scale(self) -> torch.Tensor | None:
        if not self._fp8_output_exchange_enabled:
            return None
        if getattr(self.attn, "world_size", 1) != 4:
            return None
        inner = getattr(self.attn, "inner_backend", None)
        if getattr(inner, "preferred_layout", None) != AttentionTensorLayout.NHD:
            return None
        out_linear = self.to_out[0]
        scale = static_fp8_output_scale(out_linear)
        if scale is None:
            return None
        key = (id(out_linear), scale.data_ptr(), scale._version)
        if key != self._fp8_out_scale_cache_key:
            self._fp8_out_scale_valid = bool(
                torch.isfinite(scale).all() and (scale > 0).all()
            )
            self._fp8_out_scale_cache_key = key
        return scale if self._fp8_out_scale_valid else None

    def _attn_impl(self, q, k, v, **kwargs):
        if (
            q.ndim != 3
            or q.shape[0] != 1
            or q.shape[1] == 0
            or q.shape[2] != 56 * 128
            or not all(
                t.shape == q.shape and t.dtype == torch.bfloat16 and t.stride(-1) == 1
                for t in (q, k, v)
            )
            or (
                kwargs.get("key_padding_mask") is not None
                and not kwargs.get("discard_padded_queries")
            )
        ):
            return super()._attn_impl(q, k, v, **kwargs)
        return self._packed_forward(q, k, v, **kwargs)

    @torch.compiler.disable
    def _packed_forward(self, q, k, v, **kwargs):
        rows = q.shape[1]
        rank = self.attn._ulysses_rank
        mask = kwargs.get("key_padding_mask")
        total = (
            4 * rows if mask is None else self.attn.inner_backend._padding_prefix_lengths(mask)[0]
        )
        valid_rows = max(0, min(rows, total - rank * rows))
        q, k, v = (t.view(1, rows, 56, 128) for t in (q, k, v))
        send, descales = pack_qkv(q, k, v, valid_rows, self.attn.process_group)
        out_scale = self._validated_fp8_output_scale()
        if out_scale is not None:
            raw_out = self.attn.forward_packed_fp8(
                send, fp8_descales=descales, return_pre_a2a=True, **kwargs
            )
            if (
                raw_out.ndim == 4
                and raw_out.shape == (1, 4 * rows, 14, 128)
                and raw_out.dtype == torch.bfloat16
                and raw_out.is_cuda
            ):
                self._fp8_output_exchange_calls += 1
                return exchange_fp8_output(raw_out, out_scale, self.attn.process_group)
            self._bf16_output_exchange_calls += 1
            return self.attn._output_a2a(raw_out, 1, 4 * rows).flatten(2)
        self._bf16_output_exchange_calls += 1
        return self.attn.forward_packed_fp8(send, fp8_descales=descales, **kwargs).flatten(2)


class Fp8UlyssesBlock(MiniMaxH3TransformerBlock):
    attention_cls = Fp8UlyssesAttention


class Fp8UlyssesTransformer(MiniMaxH3Transformer3DModel):
    block_cls = Fp8UlyssesBlock
    replicate_token_refiner = True


class MiniMaxH3Fp8U4Pipeline(MiniMaxH3Pipeline):
    def _init_transformer(self) -> None:
        self.transformer = Fp8UlyssesTransformer(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )
