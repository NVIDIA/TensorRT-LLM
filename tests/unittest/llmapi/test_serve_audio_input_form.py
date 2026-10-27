# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""trtllm-serve refuses a spectrogram audio form before it loads a model.

Its audio inputs are decoded to waveforms, so the setting could never take
effect; failing early saves the model load and warmup.
"""

import click
import pytest

from tensorrt_llm.commands.serve import launch_server
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "multimodal_config",
    [
        # As a YAML override leaves it, and as a built config object.
        {"audio_input_form": "spectrogram"},
        MultimodalConfig(audio_input_form="spectrogram"),
    ],
    ids=["mapping", "object"],
)
def test_launch_server_rejects_a_spectrogram_audio_form_before_loading(multimodal_config) -> None:
    llm_args = {
        "backend": "pytorch",
        "model": "dummy",
        "multimodal_config": multimodal_config,
    }
    # Raised before the socket is bound or any model is loaded: a port that
    # nothing may bind, and a model path that does not exist, are never reached.
    with pytest.raises(click.BadParameter, match="spectrogram"):
        launch_server("localhost", 1, llm_args)


@pytest.mark.parametrize(
    "multimodal_config",
    [None, {}, {"audio_input_form": "waveform"}, MultimodalConfig()],
    ids=["absent", "empty", "mapping", "object"],
)
def test_launch_server_accepts_the_waveform_audio_form(multimodal_config) -> None:
    llm_args = {"backend": "pytorch", "model": "dummy", "num_serve_frontends": 2}
    if multimodal_config is not None:
        llm_args["multimodal_config"] = multimodal_config
    # The next guard in launch_server fires instead, which shows the audio
    # form check let the waveform default through.
    with pytest.raises(click.BadParameter, match="single serving frontend"):
        launch_server("localhost", 0, llm_args, report_addr="/tmp/a.addr")
