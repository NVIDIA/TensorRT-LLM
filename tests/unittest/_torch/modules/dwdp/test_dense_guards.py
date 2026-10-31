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
"""CPU tests of the ``ModelLoader`` guards that refuse weight updates under dense DWDP.

Dense DWDP setup marks the model and its decoder with ``_dense_dwdp_active`` once their dense
weights live in staging slots shared across layers.  ``ModelLoader.reload`` and
``ModelLoader.begin_update_weights`` must then raise before they change anything.
"""

from unittest.mock import MagicMock

import pytest
from torch import nn

from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

pytestmark = pytest.mark.cpu_only


class _CausalLM(nn.Module):
    """Holds its decoder as ``model``, like ``DecoderModelForCausalLM``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()


class _Wrapper(nn.Module):
    """Reaches the decoder through a property, like the Kimi-K3 VL wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = _CausalLM()

    @property
    def model(self) -> nn.Module:
        return self.llm.model


@pytest.mark.parametrize("action", ["reload", "update"])
@pytest.mark.parametrize("owner", ["model", "decoder"])
@pytest.mark.parametrize("model_class", [_CausalLM, _Wrapper])
def test_weight_updates_refuse_a_dense_dwdp_model_before_any_change(
    model_class: type[nn.Module], owner: str, action: str
) -> None:
    model = model_class()
    (model if owner == "model" else model.model)._dense_dwdp_active = True
    loader = ModelLoader.__new__(ModelLoader)
    loader.weight_mapper = MagicMock(name="weight_mapper", model=model)

    with pytest.raises(RuntimeError, match=f"Cannot {action} weights: dense DWDP"):
        if action == "reload":
            loader.reload(model, {"weight": None})
        else:
            loader.begin_update_weights()

    assert loader.weight_mapper.mock_calls == []
