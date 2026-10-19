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
