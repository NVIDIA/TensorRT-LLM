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

from dataclasses import dataclass
from inspect import getattr_static

from torch import nn


@dataclass(frozen=True)
class DecoderLoraOwner:
    """Decoder module path and checkpoint prefixes before the layer index.

    HF prefixes exclude the optional ``base_model.model`` transport prefix.
    An empty module path selects the text model itself.
    """

    module_path: str = ""
    hf_prefixes: tuple[str, ...] = (
        "layers",
        "model.layers",
        "backbone.layers",
        "language_model.layers",
        "language_model.model.layers",
        "language_model.backbone.layers",
        "model.language_model.layers",
        "transformer.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
        "model.decoder.layers",
    )
    nemo_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.encoder.layers",
        "model.decoder.layers",
        "language_model.encoder.layers",
        "decoder.layers",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.module_path, str) or (
            self.module_path and any(not part for part in self.module_path.split("."))
        ):
            raise ValueError(f"Invalid decoder LoRA owner path {self.module_path!r}")


def resolve(model: nn.Module, *, require_submodule: bool) -> tuple[DecoderLoraOwner, nn.Module]:
    """Resolve the decoder owner, defaulting undeclared text models to their root."""
    model_description = (
        f"{'Multimodal model' if require_submodule else 'Model'} {type(model).__name__}"
    )
    try:
        owner = model.lora_owner
    except (AttributeError, NotImplementedError) as exc:
        undeclared = object()
        # A descriptor that raises AttributeError is still an explicit declaration.
        if (
            not require_submodule
            and isinstance(exc, AttributeError)
            and getattr_static(model, "lora_owner", undeclared) is undeclared
        ):
            return DecoderLoraOwner(), model
        raise ValueError(
            f"{model_description} must declare a supported decoder LoRA owner: {exc}"
        ) from exc
    if not isinstance(owner, DecoderLoraOwner):
        raise ValueError(
            f"{model_description} has invalid decoder LoRA owner {owner!r}; "
            "expected DecoderLoraOwner"
        )
    if require_submodule and not owner.module_path:
        raise ValueError(
            f"{model_description} has invalid decoder LoRA owner path {owner.module_path!r}"
        )

    try:
        decoder = model.get_submodule(owner.module_path)
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(
            f"{model_description} cannot resolve decoder LoRA owner {owner.module_path!r}: {exc}"
        ) from exc
    if not isinstance(decoder, nn.Module) or (require_submodule and decoder is model):
        raise ValueError(
            f"{model_description} decoder LoRA owner {owner.module_path!r} "
            "must select a decoder submodule"
        )
    return owner, decoder
