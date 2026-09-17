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
"""``Linear`` turns away an FP4 checkpoint its GPU has no kernel for at
construction, instead of failing from the first forward pass -- but only where
no backend could have served it."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    Linear,
    MarlinNVFP4LinearMethod,
    W4A8NVFP4FP8LinearMethod,
    W4A16NVFP4LinearMethod,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


@contextmanager
def running_on(sm_version):
    """Pretend the process sees one GPU architecture, with Marlin kernels built.

    Marlin decides which architectures are in range for NVFP4, so the ops have
    to look present for the SM to be the only variable under test.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=sm_version)
        )
        stack.enter_context(patch("torch.ops.trtllm.marlin_nvfp4_gemm", create=True))
        stack.enter_context(patch("torch.ops.trtllm.gptq_marlin_repack", create=True))
        yield


def deferred_linear(quant_algo, **kwargs):
    """A Linear whose weights are not created yet, so create_weights() -- the
    step the architecture check guards -- can be driven on its own."""
    return Linear(
        128,
        128,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=quant_algo),
        reduce_output=False,
        skip_create_weights_in_init=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "quant_algo",
    [
        QuantAlgo.NVFP4,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_unsupported_fp4_is_rejected_before_any_weight_is_created(quant_algo):
    """The whole point of the check: no quant method is chosen and no parameter
    is allocated once the architecture is known to be out of range."""
    with running_on(90):
        linear = deferred_linear(quant_algo)

        with (
            patch("tensorrt_llm._torch.modules.linear.get_quant_method") as get_quant_method,
            pytest.raises(ValueError, match="SM90"),
        ):
            linear.create_weights()

    get_quant_method.assert_not_called()
    assert not linear._weights_created
    assert not hasattr(linear, "quant_method")
    assert not hasattr(linear, "weight")


def test_w4a16_nvfp4_is_not_rejected_on_hopper():
    """W4A16_NVFP4 dequantizes its weights before the GEMM, so it never needed
    the FP4 tensor cores the W4A4 modes are gated on."""
    with running_on(90):
        linear = deferred_linear(QuantAlgo.W4A16_NVFP4)
        linear.create_weights()

    assert linear._weights_created
    # The exact method, not the Marlin subclass: nothing opted into Marlin.
    assert type(linear.quant_method) is W4A16NVFP4LinearMethod


def test_nvfp4_marlin_opt_in_is_not_rejected_on_hopper():
    """Marlin serves an NVFP4 checkpoint weight-only on Ada/Hopper, so the
    architecture check has to read the same backend list ``get_quant_method``
    does rather than turn SM90 away outright."""
    with running_on(90):
        linear = deferred_linear(QuantAlgo.NVFP4, nvfp4_allowed_backends=["marlin"])
        linear.create_weights()

    assert linear._weights_created
    assert isinstance(linear.quant_method, MarlinNVFP4LinearMethod)


def test_nvfp4_rejection_on_hopper_points_at_the_marlin_opt_in():
    with running_on(90):
        linear = deferred_linear(QuantAlgo.NVFP4)
        with pytest.raises(ValueError, match="marlin"):
            linear.create_weights()


def test_unknown_architecture_still_builds_the_layer():
    """``get_sm_version`` reports -1 with no visible device, which is not
    evidence of a rejection -- the layer still has to build."""
    with running_on(-1):
        linear = deferred_linear(QuantAlgo.NVFP4)
        linear.create_weights()

    assert linear._weights_created


@pytest.mark.parametrize(
    ("present_key", "missing_key"),
    [
        ("weight_scale_2", "input_scale"),
        ("input_scale", "weight_scale_2"),
    ],
)
def test_w4a8_nvfp4_names_the_scale_the_checkpoint_omitted(present_key, missing_key):
    """Without both of these ``alpha`` cannot be computed. Say which one the
    checkpoint is missing, rather than raise on ``None`` a few lines later."""
    module = SimpleNamespace(tp_mode=None)
    weights = [{present_key: torch.ones(1, dtype=torch.float32)}]

    with pytest.raises(ValueError, match=missing_key) as excinfo:
        W4A8NVFP4FP8LinearMethod().load_weight_scales(
            module, weights, quant_algo=QuantAlgo.W4A8_NVFP4_FP8
        )

    assert QuantAlgo.W4A8_NVFP4_FP8.name in str(excinfo.value)
