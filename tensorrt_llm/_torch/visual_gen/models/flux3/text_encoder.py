# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 text encoder: stacked hidden states of selected Qwen3-VL layers, run eagerly."""

import math
from typing import Any, Sequence, Tuple

import torch
from torch import Tensor, nn

# Hidden states stacked along channels; 8 x 2560 is the transformer's joint_attention_dim.
HIDDEN_STATE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
# Prompts are right-padded to a multiple of TEXT_PAD_MULTIPLE tokens, at most TEXT_MAX_TOKENS.
# The DiT attends to the padding, so the padded length is part of the conditioning.
TEXT_PAD_MULTIPLE = 80
TEXT_MAX_TOKENS = 8192
# Padding token position id; matches the transformers version FLUX.3 was trained with.
_PADDING_POSITION_ID = 1


def padded_text_length(
    num_tokens: int, pad_multiple: int = TEXT_PAD_MULTIPLE, max_tokens: int = TEXT_MAX_TOKENS
) -> int:
    """Window of a prompt with ``num_tokens`` tokens: rounded up to ``pad_multiple``, capped."""
    return min(math.ceil(num_tokens / pad_multiple) * pad_multiple, max_tokens)


def trim_for_text(model: nn.Module, last_layer: int) -> None:
    """Keep a Qwen3-VL model's first ``last_layer`` decoder layers and drop its vision tower.

    The final norm becomes Identity when layers are cut, so hidden_states[last_layer] stays the
    un-normed layer output that the untrimmed model returns.
    """
    language_model = model.model.language_model
    num_layers = len(language_model.layers)
    if not 1 <= last_layer <= num_layers:
        raise ValueError(f"FLUX.3 text encoder layer {last_layer} is not in 1..{num_layers}")
    if last_layer < num_layers:
        language_model.layers = language_model.layers[:last_layer]
        language_model.norm = nn.Identity()
    # FLUX.3 encodes text only; the vision tower is never called.
    model.model.visual = None


class Flux3TextEncoder(nn.Module):
    """Chat-templated prompt to stacked Qwen3-VL hidden states [1, L, layers * hidden]."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        hidden_state_layers: Sequence[int] = HIDDEN_STATE_LAYERS,
        max_tokens: int = TEXT_MAX_TOKENS,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.hidden_state_layers = tuple(hidden_state_layers)
        self.max_tokens = max_tokens
        self.pad_multiple = pad_multiple

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        processor_path: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Flux3TextEncoder":
        """Load Qwen3-VL in ``dtype``, trim it on the host, then move it to ``device``."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype)
        trim_for_text(model, max(HIDDEN_STATE_LAYERS))
        processor = AutoProcessor.from_pretrained(processor_path)
        return cls(model.to(device).eval(), processor)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def tokenize(self, prompt: str) -> Tuple[Tensor, Tensor]:
        """Token ids and attention mask, each [1, L], of the chat-templated, right-padded prompt."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        num_tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        )["input_ids"].shape[1]
        length = padded_text_length(num_tokens, self.pad_multiple, self.max_tokens)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=length,
        )
        return inputs["input_ids"], inputs["attention_mask"]

    @torch.inference_mode()
    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Stacked hidden states [1, L, len(hidden_state_layers) * hidden] in the encoder dtype."""
        device = self.device
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        position_ids = torch.arange(input_ids.shape[1], device=device)[None].masked_fill(
            attention_mask == 0, _PADDING_POSITION_ID
        )
        output = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[k] for k in self.hidden_state_layers], dim=1)
        stacked = stacked.to(self.dtype)
        batch, num_layers, length, hidden = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch, length, num_layers * hidden)

    def forward(self, prompt: str) -> Tensor:
        """Encode one prompt; see :meth:`tokenize` and :meth:`encode`."""
        return self.encode(*self.tokenize(prompt))
