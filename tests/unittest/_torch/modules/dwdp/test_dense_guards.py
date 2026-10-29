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
weights live in staging slots shared across layers.  ``ModelLoader.reload`` (R1)
and ``ModelLoader.begin_update_weights`` (R2) must then raise before they change anything: reload
before it resets ``_weights_transformed`` or loads a byte, the update session before the mapper
opens it.  Without the mark both behave as before.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

pytestmark = pytest.mark.cpu_only


class _Transformed(nn.Module):
    """A module whose weights went through a one-shot transform."""

    def __init__(self) -> None:
        super().__init__()
        self._weights_transformed = True


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Transformed()])


class _CausalLM(nn.Module):
    """Holds its decoder as ``model``, like ``DecoderModelForCausalLM``."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Decoder()
        self.loaded: list[dict] = []

    def load_weights(self, weights: dict, weight_mapper: object) -> None:
        self.loaded.append(weights)


class _Wrapper(nn.Module):
    """Reaches the decoder through a property, like the Kimi-K3 VL wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.llm = _CausalLM()

    @property
    def model(self) -> nn.Module:
        return self.llm.model

    def load_weights(self, weights: dict, weight_mapper: object) -> None:
        self.llm.load_weights(weights, weight_mapper)


def _loader(model: nn.Module) -> ModelLoader:
    """A loader that has loaded ``model``: its persistent weight mapper is bound to it."""
    loader = ModelLoader.__new__(ModelLoader)
    loader.weight_mapper = MagicMock(name="weight_mapper")
    loader.weight_mapper.model = model
    loader._call_load_weights = MagicMock(
        side_effect=lambda load, weights, mapper, **_kwargs: load(weights, mapper)
    )
    return loader


def _transformed_flags(model: nn.Module) -> list[bool]:
    return [
        module._weights_transformed
        for module in model.modules()
        if hasattr(module, "_weights_transformed")
    ]


def _mark(model: nn.Module, owner: str) -> None:
    """Mark the model itself or its decoder the way dense DWDP setup does."""
    target = model if owner == "model" else model.model
    target._dense_dwdp_active = True


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """``reload`` synchronizes the current CUDA stream; stay on the host."""
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock(name="current_stream"))


@pytest.mark.parametrize("owner", ["model", "decoder"])
@pytest.mark.parametrize("model_class", [_CausalLM, _Wrapper])
@pytest.mark.parametrize("allow_partial_loading", [False, True])
def test_reload_refuses_a_dense_dwdp_model_before_any_change(
    model_class, owner, allow_partial_loading
):
    model = model_class()
    _mark(model, owner)
    loader = _loader(model)

    with pytest.raises(RuntimeError, match="dense DWDP"):
        loader.reload(model, {"w": torch.ones(1)}, allow_partial_loading=allow_partial_loading)

    assert _transformed_flags(model) == [True]
    loader._call_load_weights.assert_not_called()


@pytest.mark.parametrize("owner", ["model", "decoder"])
@pytest.mark.parametrize("model_class", [_CausalLM, _Wrapper])
def test_begin_update_weights_refuses_a_dense_dwdp_model_before_any_change(model_class, owner):
    model = model_class()
    _mark(model, owner)
    loader = _loader(model)

    with pytest.raises(RuntimeError, match="dense DWDP"):
        loader.begin_update_weights()

    loader.weight_mapper.begin_update_weights.assert_not_called()


@pytest.mark.parametrize("mark", [None, False])
@pytest.mark.parametrize("model_class", [_CausalLM, _Wrapper])
def test_reload_without_dense_dwdp_resets_and_loads(model_class, mark):
    model = model_class()
    if mark is not None:
        model._dense_dwdp_active = mark
        model.model._dense_dwdp_active = mark
    loader = _loader(model)
    weights = {"w": torch.ones(1)}

    loader.reload(model, weights)

    assert _transformed_flags(model) == [False]
    loader._call_load_weights.assert_called_once()
    causal_lm = model if isinstance(model, _CausalLM) else model.llm
    assert causal_lm.loaded == [weights]


@pytest.mark.parametrize("mark", [None, False])
def test_begin_update_weights_without_dense_dwdp_opens_the_session(mark):
    model = _CausalLM()
    if mark is not None:
        model._dense_dwdp_active = mark
        model.model._dense_dwdp_active = mark
    loader = _loader(model)

    loader.begin_update_weights()

    loader.weight_mapper.begin_update_weights.assert_called_once_with()


def test_begin_update_weights_still_requires_a_weight_mapper():
    loader = _loader(_CausalLM())
    loader.weight_mapper = None

    with pytest.raises(RuntimeError, match="weight_mapper was not initialized"):
        loader.begin_update_weights()
