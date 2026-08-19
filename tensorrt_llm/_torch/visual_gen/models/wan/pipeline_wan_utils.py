# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata


def wan_attn_metadata_kwargs(
    model: Any,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, AttentionMetadata]:
    """Name WAN's attention metadata sites for ``forward``.

    Maps the site names from ``WanTransformer3DModel.create_attn_metadata()``
    onto the ``attn_metadata_*`` keyword arguments of its ``forward``.
    """
    sites = model.create_attn_metadata(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=encoder_hidden_states_image,
        batch_size=batch_size,
    )
    kwargs = {
        "attn_metadata_self": sites["self"],
        "attn_metadata_cross_text": sites["cross_text"],
    }
    if "cross_image" in sites:
        kwargs["attn_metadata_cross_image"] = sites["cross_image"]
    return kwargs


def retrieve_latents(
    encoder_output: Any,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "argmax",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")
