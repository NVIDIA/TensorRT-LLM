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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

mamba2_mixer_mod = pytest.importorskip("tensorrt_llm._torch.modules.mamba.mamba2_mixer")
Mamba2Mixer = mamba2_mixer_mod.Mamba2Mixer


def test_mamba2_mixer_post_load_weights_caches_derived_state():
    mixer = Mamba2Mixer.__new__(Mamba2Mixer)
    nn.Module.__init__(mixer)
    mixer.norm = SimpleNamespace(is_nvfp4=False)
    mixer.A = torch.tensor([1.0, 2.0])
    mixer.dt_bias = torch.tensor([3.0, 4.0])
    mixer.D = torch.tensor([5.0, 6.0])
    mixer.head_dim = 2
    mixer.d_state = 3

    mixer.post_load_weights()

    assert mixer._A_expanded.shape == (2, 2, 3)
    assert mixer._dt_bias_expanded.shape == (2, 2)
    assert mixer._D_expanded.shape == (2, 2)
    assert not hasattr(mixer, "_weights_transformed")
