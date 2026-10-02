# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen text encoder used by MiniMax-H3 visual generation."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3ForTextEmbedding
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from .packing import MINIMAX_H3_TEXT_ENCODER_LAYER


class MiniMaxH3TensorParallelTextEncoder(torch.nn.Module):
    """TRTLLM Qwen3 backbone that returns MiniMax-H3's layer-50 state.

    MiniMax-H3 consumes the unnormalized hidden state from the Qwen3-VL text
    backbone rather than final logits. The regular TRTLLM Qwen3 model already
    shards attention/MLP weights when ``mapping.tp_size > 1``; this wrapper only
    adapts that backbone to the visual-gen prompt-encoding contract.
    """

    _LANGUAGE_MODEL_PREFIX = "model.language_model."
    _BARE_LANGUAGE_MODEL_PREFIX = "language_model."
    _LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

    def __init__(
        self,
        text_config: Any,
        mapping: Mapping,
        device: torch.device,
        *,
        target_layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if text_config.num_hidden_layers <= target_layer:
            raise ValueError(
                "MiniMax-H3 requires the unnormalized hidden state after Qwen3-VL "
                f"layer {target_layer}, but the encoder has "
                f"{text_config.num_hidden_layers} layers."
            )

        self.target_layer = target_layer
        self.mapping = mapping
        self.device = device
        self.dtype = torch.bfloat16

        config = copy.deepcopy(text_config)
        config.architectures = ["Qwen3ForTextEmbedding"]
        config.torch_dtype = self.dtype
        config.num_hidden_layers = target_layer
        config.disable_fuse_rope = True
        if getattr(config, "rope_scaling", None) is None:
            config.rope_scaling = {}
        config.rope_scaling["type"] = "mrope"

        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            attn_backend="VANILLA",
            # MiniMax-H3 can load its Qwen text encoder on a subset of ranks.
            # NCCL keeps all-reduce scoped to that TP group and avoids the
            # custom workspace setup that synchronizes the full world.
            allreduce_strategy=AllReduceStrategy.NCCL,
            max_num_tokens=max_num_tokens or getattr(config, "max_position_embeddings", 8192),
        )
        model_config._frozen = True
        self.model = Qwen3ForTextEmbedding(model_config).to(device).eval()

    @classmethod
    def from_pretrained_config(
        cls,
        checkpoint_dir: str | Path,
        mapping: Mapping,
        device: torch.device,
        *,
        target_layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
    ) -> "MiniMaxH3TensorParallelTextEncoder":
        config = AutoConfig.from_pretrained(
            checkpoint_dir,
            subfolder="text_encoder",
        )
        if not hasattr(config, "text_config"):
            raise ValueError("MiniMax-H3 text_encoder config must contain text_config.")
        return cls(
            config.text_config,
            mapping,
            device,
            target_layer=target_layer,
        )

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        mapped_weights = {}
        for key, value in weights.items():
            mapped_key = self._map_weight_key(key)
            if mapped_key is None:
                continue
            mapped_weights[mapped_key] = value
        self.model.load_weights(mapped_weights)

    def _map_weight_key(self, key: str) -> str | None:
        if key.startswith("model.visual."):
            return None
        if key.startswith(self._LANGUAGE_MODEL_PREFIX):
            key = f"model.{key[len(self._LANGUAGE_MODEL_PREFIX) :]}"
        elif key.startswith(self._BARE_LANGUAGE_MODEL_PREFIX):
            key = f"model.{key[len(self._BARE_LANGUAGE_MODEL_PREFIX) :]}"

        if key.startswith(("lm_head.", "model.lm_head.")):
            return None

        lookup_key = key if key.startswith("model.") else f"model.{key}"
        layer_match = self._LAYER_RE.match(lookup_key)
        if layer_match is not None and int(layer_match.group(1)) >= self.target_layer:
            return None
        return key

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        seq_len = input_ids.shape[0]
        attn_metadata = VanillaAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=seq_len,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            num_contexts=1,
            request_ids=[0],
            mapping=self.mapping,
        )
        attn_metadata.prepare_encoder_only()

        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long)
        position_ids = position_ids.view(1, 1, seq_len).expand(3, 1, seq_len).contiguous()

        hidden_states = self.model.model.embed_tokens(input_ids)
        residual = None
        for decoder_layer in self.model.model.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states.unsqueeze(0)
