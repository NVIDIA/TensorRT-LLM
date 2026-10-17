# SPDX-FileCopyrightText: Copyright (c) 2026 Black Forest Labs, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FLUX.3 text encoder: Qwen3-VL stacking hidden states across selected layers.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


@torch.library.custom_op("trtllm::flux3_text_rms_norm", mutates_args=())
def _text_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Preserve Qwen's eager RMSNorm reduction and BF16 rounding under compile."""
    normalized = x.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)


@_text_rms_norm.register_fake
def _text_rms_norm_fake(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return torch.empty_like(x)


class _CompileRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        return _text_rms_norm(x, self.weight, self.variance_epsilon)


@dataclass
class Qwen3VLEmbedderParams:
    # Layer indices (1-indexed into hidden_states) to stack along the channel dim.
    # 8 layers × 2560 (4B hidden) = 20480 = context_in_dim for the small model.
    output_layer: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32])
    torch_dtype: str = "bfloat16"
    max_length: int = 8192
    pad_multiple: int = 80
    use_compile: bool = False


class Qwen3VLEmbedder(torch.nn.Module):
    """FLUX.3 text encoder: stacks hidden states from selected Qwen3-VL layers.

    Small model (Qwen3-VL-4B, hidden=2560): 8 layers × 2560 → context_in_dim 20480.
    """

    def __init__(
        self,
        model_spec: str,
        params: Qwen3VLEmbedderParams | None = None,
        tokenizer_spec: Optional[str] = None,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if params is None:
            params = Qwen3VLEmbedderParams()
        self.params = params
        self.dtype = getattr(torch, params.torch_dtype)
        self.output_layer = list(params.output_layer)
        self.max_length = params.max_length

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_spec, torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(tokenizer_spec or model_spec)
        self.processor.tokenizer.padding_side = "right"

        lm = self.model.model.language_model
        if not self.output_layer or any(k < 0 or k > len(lm.layers) for k in self.output_layer):
            raise ValueError("output_layer must select existing Qwen hidden states")
        last_layer = max(self.output_layer)
        if last_layer < len(lm.layers):
            # Truncated outputs must retain the intermediate, pre-final-norm states.
            lm.layers = lm.layers[:last_layer]
            lm.norm = torch.nn.Identity()

        if params.use_compile:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

            for name, module in list(lm.named_modules()):
                if isinstance(module, Qwen3VLTextRMSNorm):
                    parent, _, child = name.rpartition(".")
                    lm.get_submodule(parent).add_module(
                        child, _CompileRMSNorm(module.weight, module.variance_epsilon)
                    )
            self._forward_tensors = torch.compile(
                self._forward_tensors,
                dynamic=False,
                options={"emulate_precision_casts": True, "triton.cudagraphs": False},
            )

    @torch.no_grad()
    def _forward_hidden(self, prompts: list[str]) -> list[Tensor]:
        contexts = []
        for prompt in prompts:
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            tokenizer = self.processor.tokenizer
            real = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self.max_length
            )["input_ids"].shape[1]
            length = min(
                math.ceil(real / self.params.pad_multiple) * self.params.pad_multiple,
                self.max_length,
            )
            tokens = tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=length
            )
            ctx = self._forward_tensors(
                tokens["input_ids"].to(self.model.device),
                tokens["attention_mask"].to(self.model.device),
            )
            contexts.append(ctx.clone() if self.params.use_compile else ctx)
        return contexts

    def _forward_tensors(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return torch.cat([out.hidden_states[k] for k in self.output_layer], dim=-1).to(self.dtype)

    @torch.inference_mode()
    def encode(self, prompts: List[str]) -> list[Tensor]:
        return self._forward_hidden(prompts)
