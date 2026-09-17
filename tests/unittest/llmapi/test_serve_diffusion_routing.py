# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for trtllm-serve's diffusion-vs-regular-model routing.

`get_is_diffusion_only_model` decides whether a checkpoint is served via the
VisualGen (diffusion) path or the regular PyTorch model path. A unified
checkpoint such as Cosmos3-Nano ships both a diffusers layout
(``model_index.json``) and a registered VLM architecture (``config.json``); it
must route to the regular model path rather than VisualGen.
"""

import json

from tensorrt_llm.commands.utils import get_is_diffusion_only_model

# `is_diffusers_model_path` only checks for `_diffusers_version`, and
# `has_registered_llm_architecture` only reads `architectures`, so the minimal
# fixtures below carry just those single fields.
_DIFFUSERS_MODEL_INDEX = {"_diffusers_version": "0.30.0"}
_REGISTERED_ARCH_CONFIG = {"architectures": ["Cosmos3ForConditionalGeneration"]}
_UNREGISTERED_ARCH_CONFIG = {"architectures": ["SomeUnknownPipeline"]}


def test_diffusion_only_for_pure_diffusion_checkpoint(tmp_path):
    # A diffusers layout with no config.json is a pure generative checkpoint and
    # must route to the VisualGen path.
    (tmp_path / "model_index.json").write_text(json.dumps(_DIFFUSERS_MODEL_INDEX))

    assert get_is_diffusion_only_model(str(tmp_path)) is True


def test_diffusion_only_for_unregistered_arch(tmp_path):
    # Diffusers layout plus a config.json whose architecture is not registered
    # with the PyTorch backend still routes to the VisualGen path.
    (tmp_path / "model_index.json").write_text(json.dumps(_DIFFUSERS_MODEL_INDEX))
    (tmp_path / "config.json").write_text(json.dumps(_UNREGISTERED_ARCH_CONFIG))

    assert get_is_diffusion_only_model(str(tmp_path)) is True


def test_dual_checkpoint_prefers_registered_model(tmp_path):
    # A unified checkpoint (e.g. Cosmos3-Nano) shipping both a diffusers layout
    # and a registered VLM architecture must route to the regular model path.
    (tmp_path / "model_index.json").write_text(json.dumps(_DIFFUSERS_MODEL_INDEX))
    (tmp_path / "config.json").write_text(json.dumps(_REGISTERED_ARCH_CONFIG))

    assert get_is_diffusion_only_model(str(tmp_path)) is False
