# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve request shapes for numerically sensitive Qwen-Image operations."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenImageBatchLayout:
    """Original text shapes and mask presence in a merged transformer call.

    SDPA and small GEMMs can select different BF16 arithmetic for a different
    shape or mask. Keep these calls at their original request shapes while the
    large image projections and image MLPs operate on the combined batch.
    """

    sample_counts: tuple[int, ...]
    text_lengths: tuple[int, ...]
    mask_is_none: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.sample_counts or not (
            len(self.sample_counts) == len(self.text_lengths) == len(self.mask_is_none)
        ):
            raise ValueError("Qwen batch layout must describe every original request")
        if any(n < 1 for n in self.sample_counts + self.text_lengths):
            raise ValueError("Qwen batch sample counts and text lengths must be positive")

    def apply_to_text(
        self, function: Callable[[torch.Tensor], torch.Tensor], value: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        for chunk, length in zip(value.split(self.sample_counts), self.text_lengths, strict=True):
            output = function(chunk[:, :length].contiguous())
            outputs.append(F.pad(output, (0, 0, 0, value.shape[1] - length)))
        return torch.cat(outputs)

    def apply_to_images(
        self, function: Callable[[torch.Tensor], torch.Tensor], value: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [function(chunk.contiguous()) for chunk in value.split(self.sample_counts)]
        )

    def attention(
        self,
        function: Callable[..., torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        timestep: torch.Tensor | None,
    ) -> torch.Tensor:
        text_length = max(self.text_lengths)
        packed_qkv = q.stride(1) == q.shape[-1] + k.shape[-1] + v.shape[-1]
        outputs = []
        offset = 0
        for count, length, mask_is_none in zip(
            self.sample_counts, self.text_lengths, self.mask_is_none, strict=True
        ):
            end = offset + count
            parts = [
                torch.cat([x[offset:end, :length], x[offset:end, text_length:]], dim=1)
                for x in (q, k, v)
            ]
            if packed_qkv:
                # Strides also influence SDPA kernel selection. Preserve the
                # packed QKV layout produced by the fused QK-norm/RoPE path.
                parts = torch.cat(parts, dim=-1).split(
                    [q.shape[-1], k.shape[-1], v.shape[-1]], dim=-1
                )
            kwargs = {}
            if not mask_is_none:
                if attention_mask is None:
                    raise ValueError("Qwen batch layout requires the original attention mask")
                mask = attention_mask[offset:end]
                kwargs["key_padding_mask"] = torch.cat(
                    [mask[:, :length], mask[:, text_length:]], dim=1
                )
            output = function(
                *parts,
                timestep=timestep[offset:end] if timestep is not None else None,
                **kwargs,
            )
            outputs.append(
                torch.cat(
                    [
                        output[:, :length],
                        output.new_zeros(count, text_length - length, output.shape[-1]),
                        output[:, length:],
                    ],
                    dim=1,
                )
            )
            offset = end
        return torch.cat(outputs)


def collate_prompt_encodings(
    encodings: list[tuple[torch.Tensor, torch.Tensor | None]],
    sample_counts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor | None, QwenImageBatchLayout]:
    layout = QwenImageBatchLayout(
        sample_counts,
        tuple(embeds.shape[1] for embeds, _ in encodings),
        tuple(mask is None for _, mask in encodings),
    )
    length = max(layout.text_lengths)
    embeds = torch.cat([F.pad(x, (0, 0, 0, length - x.shape[1])) for x, _ in encodings])
    mask = None
    if not all(layout.mask_is_none) or len(set(layout.text_lengths)) != 1:
        mask = torch.cat(
            [
                F.pad(
                    torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if m is None
                    else m.bool(),
                    (0, length - x.shape[1]),
                )
                for x, m in encodings
            ]
        )
    return embeds, mask, layout
