# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest import mock

import transformers

import tensorrt_llm.models.mamba.model as mamba_model
from tensorrt_llm.models.mamba.model import MambaForCausalLM


def test_from_hugging_face_preloaded_model_does_not_unbound_error() -> None:
    # Passing a preloaded transformers model previously raised
    # UnboundLocalError because hf_model_dir was only assigned on the
    # string-path branch but referenced unconditionally (issue #15501).
    preloaded = mock.MagicMock(spec=transformers.PreTrainedModel)

    with (
        mock.patch.object(
            mamba_model.MambaConfig, "from_hugging_face", return_value=mock.MagicMock()
        ),
        mock.patch.object(mamba_model, "convert_hf_mamba", return_value={}) as convert_hf,
        mock.patch.object(mamba_model, "convert_from_hf_checkpoint") as convert_ckpt,
        mock.patch.object(MambaForCausalLM, "__init__", return_value=None),
        mock.patch.object(MambaForCausalLM, "load", return_value=None),
    ):
        MambaForCausalLM.from_hugging_face(preloaded)

    # The preloaded model is converted directly; the checkpoint path is not used.
    convert_hf.assert_called_once()
    assert convert_hf.call_args.args[0] is preloaded
    convert_ckpt.assert_not_called()
