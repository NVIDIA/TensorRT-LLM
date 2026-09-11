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
"""Op-provider re-resolution when layerwise quantization leaves a layer BF16.

A checkpoint may declare NVFP4 at model level while leaving individual MoE
layers unquantized. ``ConfigurableMoE`` installs the final per-layer quant
config on the backend only just before ``create_weights()``, so the provider
picked in ``TRTLLMGenFusedMoE.__init__`` was resolved against the model-level
config. Only FlashInfer implements the BF16 MoE kernels, so such a layer must
move providers in ``create_weights()``; otherwise it keeps the native provider
and fails in ``run_bf16_moe`` at warmup.

These tests are CPU-only: they drive the provider-selection logic on bare
instances with the heavy pieces (quant methods, op backend construction,
communication strategies) stubbed out.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.moe.fused_moe import fused_moe_trtllm_gen
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import (
    ConfigurableMoE,
    NVLinkTwoSided,
    NVLinkTwoSidedFlashinfer,
)
from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# The l0_cpu stage runs pytest with ``-m cpu_only`` and its conftest drops any
# test file that lacks this marker, so without it these tests never run in CI.
pytestmark = pytest.mark.cpu_only


class _RecordingOpBackendFactory:
    """Stand-in for ``get_op_backend``.

    The real FlashInfer op backend imports ``flashinfer`` in its constructor,
    which the provider decision itself does not need.
    """

    def __init__(self):
        self.requested = []

    def __call__(self, name):
        self.requested.append(name)
        return mock.Mock(name=f"{name}_op_backend")


class _StubQuantMethod:
    def create_weights(self, module, *args):
        pass


def _make_backend(quant_config):
    """A ``TRTLLMGenFusedMoE`` carrying only what provider selection reads."""
    backend = TRTLLMGenFusedMoE.__new__(TRTLLMGenFusedMoE)
    backend.quant_config = quant_config
    # ``is_situ_activation`` is a read-only property derived from
    # ``activation.kind``, so feed it the carrier rather than the answer.
    backend.activation = SimpleNamespace(kind=ActivationType.Swiglu)
    backend.activation_type = ActivationType.Swiglu
    backend.bias = False
    backend._weights_created = False
    backend.num_fused_shared_expert = 0
    return backend


@pytest.fixture
def op_backend_factory():
    factory = _RecordingOpBackendFactory()
    with mock.patch.object(fused_moe_trtllm_gen, "get_op_backend", factory):
        yield factory


def _init_provider(backend, factory):
    """Run the ``__init__`` half of provider selection."""
    backend._select_op_provider()
    return factory.requested[-1]


def _run_create_weights(backend):
    with (
        mock.patch.object(TRTLLMGenFusedMoE, "_get_quant_method", return_value=_StubQuantMethod()),
        mock.patch.object(TRTLLMGenFusedMoE, "_check_configs"),
    ):
        TRTLLMGenFusedMoE.create_weights(backend)


def test_bf16_layer_moves_to_flashinfer_in_create_weights(op_backend_factory):
    """Model-level NVFP4 at init, BF16 layer config at create_weights()."""
    backend = _make_backend(QuantConfig(quant_algo=QuantAlgo.NVFP4))

    # FlashInfer is not selected for NVFP4 unless explicitly requested, so the
    # layer starts on the native provider.
    assert _init_provider(backend, op_backend_factory) == "trtllm"
    assert backend.use_flashinfer is False

    # Layerwise quantization leaves this layer unquantized; ConfigurableMoE
    # installs that config on the backend just before create_weights().
    backend.quant_config = QuantConfig()

    with mock.patch.object(
        TRTLLMGenFusedMoE, "_is_flashinfer_fused_moe_available", return_value=True
    ):
        _run_create_weights(backend)

    assert backend.use_flashinfer is True
    assert op_backend_factory.requested == ["trtllm", "flashinfer"]


def test_unchanged_layer_keeps_its_provider(op_backend_factory):
    """A layer whose quant config still says NVFP4 stays on the native op."""
    backend = _make_backend(QuantConfig(quant_algo=QuantAlgo.NVFP4))

    assert _init_provider(backend, op_backend_factory) == "trtllm"
    _run_create_weights(backend)

    assert backend.use_flashinfer is False
    assert op_backend_factory.requested == ["trtllm", "trtllm"]


def test_bf16_layer_stays_native_without_flashinfer_bf16_kernels(op_backend_factory):
    """Without the FlashInfer BF16 kernels there is nowhere else to go."""
    backend = _make_backend(QuantConfig(quant_algo=QuantAlgo.NVFP4))
    _init_provider(backend, op_backend_factory)
    backend.quant_config = QuantConfig()

    with mock.patch.object(
        TRTLLMGenFusedMoE, "_is_flashinfer_fused_moe_available", return_value=False
    ):
        _run_create_weights(backend)

    assert backend.use_flashinfer is False


class _ProviderSwitchingBackend:
    """Minimal backend whose ``create_weights`` flips the op provider."""

    def __init__(self, initial, final):
        self.use_flashinfer = initial
        self._final = final
        self.quant_config = None
        self._weights_created = True  # skip the activation-param re-install

    def create_weights(self):
        self.use_flashinfer = self._final
        return "created"


def _make_wrapper(backend, comm):
    moe = ConfigurableMoE.__new__(ConfigurableMoE)
    moe.backend = backend
    moe.comm = comm
    moe.layer_idx = 3
    moe.quant_config = QuantConfig()
    moe._override_quant_config = None
    moe.use_flashinfer = backend.use_flashinfer
    return moe


def test_wrapper_mirror_follows_the_backend():
    backend = _ProviderSwitchingBackend(initial=False, final=True)
    moe = _make_wrapper(backend, comm=None)

    assert ConfigurableMoE.create_weights(moe) == "created"
    assert moe.use_flashinfer is True
    # The final per-layer config is what the backend resolves against.
    assert backend.quant_config is moe.quant_config


def test_wrapper_mirror_unchanged_when_provider_is_unchanged():
    backend = _ProviderSwitchingBackend(initial=False, final=False)
    comm = object()
    moe = _make_wrapper(backend, comm=comm)

    ConfigurableMoE.create_weights(moe)

    assert moe.use_flashinfer is False
    assert moe.comm is comm


@pytest.mark.parametrize("comm_cls", [NVLinkTwoSided, NVLinkTwoSidedFlashinfer])
def test_provider_switch_under_nvlink_two_sided_is_rejected(comm_cls):
    """The two-sided NVLink strategies are provider-specific.

    Re-selecting one here would run a collective in the middle of per-layer
    weight creation, so the mismatch is reported instead of repaired.
    """
    comm = comm_cls.__new__(comm_cls)

    backend = _ProviderSwitchingBackend(initial=False, final=True)
    moe = _make_wrapper(backend, comm=comm)

    with pytest.raises(NotImplementedError, match="layer 3"):
        ConfigurableMoE.create_weights(moe)
