# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# used by tests/unittest/llmapi/apps/_test_openai_chat_multimodal.py

import tempfile
from pathlib import Path
from typing import Optional

import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm.inputs import ExtraProcessedInputs, TextPrompt
from tensorrt_llm.sampling_params import SamplingParams

_attach_multimodal_embeddings_orig = Qwen2VLInputProcessorBase.attach_multimodal_embeddings


# signature taken from tensorrt_llm/inputs/registry.py
def _attach_multimodal_embeddings(
    self,
    inputs: TextPrompt,
    multimodal_embedding: dict[str, list[torch.Tensor]],
    sampling_params: SamplingParams,
) -> tuple[list[int], Optional[ExtraProcessedInputs]]:
    try:
        _attach_multimodal_embeddings_orig(self, inputs, multimodal_embedding, sampling_params)
    except NotImplementedError:
        pass
    else:
        raise ValueError(
            "Remove this custom module, Qwen2VLInputProcessorBase implements attach_multimodal_embeddings"
        )

    tempdir = tempfile.gettempdir()
    file_path = Path(tempdir) / "multimodal_embedding.pickle"
    with open(file_path, "wb") as f:
        torch.save(multimodal_embedding, f)
    raise ValueError(file_path)


Qwen2VLInputProcessorBase.attach_multimodal_embeddings = _attach_multimodal_embeddings
