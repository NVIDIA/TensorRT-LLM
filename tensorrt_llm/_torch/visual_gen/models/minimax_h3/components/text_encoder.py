# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

import torch

from ..packing import MINIMAX_H3_TEXT_ENCODER_LAYER


def encode_text_tokens(
    encoder,
    processor,
    token_ids: list[int],
    vision_inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    offloader,
) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    num_layers = encoder.config.text_config.num_hidden_layers
    if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            "MiniMax-H3 requires the unnormalized hidden state after Qwen3-VL "
            f"layer {MINIMAX_H3_TEXT_ENCODER_LAYER}, but the encoder has "
            f"{num_layers} layers."
        )

    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long,
        device=device,
    )
    vision_kwargs = {
        name: value.to(device, encoder.dtype) if name.startswith("pixel_") else value.to(device)
        for name, value in vision_inputs.items()
    }
    with offloader.context_if_requested("text_encoder"):
        outputs = encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision_kwargs,
        )
    return outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=device, dtype=dtype)
