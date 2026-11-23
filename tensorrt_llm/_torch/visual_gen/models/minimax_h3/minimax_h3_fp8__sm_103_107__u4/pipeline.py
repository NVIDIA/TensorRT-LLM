# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The static-FP8 H3 pipeline with packed QKV exchange on four GPUs."""

import torch

from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent

from ..pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from .qkv_exchange import pack_qkv


class Fp8UlyssesAttention(MiniMaxH3Attention):
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
