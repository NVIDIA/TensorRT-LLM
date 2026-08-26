# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-denoising-step activation precision for static-FP8 Cosmos3.

The outer denoising steps run the resident FP8 weights through a 16-bit GEMM
instead of the checkpoint's quantized activation path. Two properties have to
hold or the feature is silently absent: the step policy must select the same
path for every call within a step, and the callers that pre-quantize a shared
activation must stand down while it is selected -- otherwise a "high precision"
step still receives FP8 activations and nothing changed.
"""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.step_precision import (
    StepPrecisionController,
    StepPrecisionFp8LinearMethod,
    apply_fp8_w8a16_linear,
)

# Pure-CPU test (stub modules, no device): runs in the CPU lane (l0_cpu.yml),
# which selects with `-m cpu_only`; GPU stages deselect it via `not cpu_only`.
pytestmark = pytest.mark.cpu_only


class _StubBaseMethod:
    """Stands in for FP8QDQLinearMethod; records whether the FP8 path ran."""

    def __init__(self):
        self.calls = 0

    def apply(self, module, input, bias=None):
        self.calls += 1
        return torch.zeros(input.shape[0], module.weight.shape[0], dtype=input.dtype)


class _StubLinear(torch.nn.Module):
    def __init__(self, out_features=4, in_features=8, scale=2.0):
        super().__init__()
        self.weight = torch.ones(out_features, in_features, dtype=torch.bfloat16)
        self.weight_scale = torch.tensor(scale, dtype=torch.float32)


class TestStepPolicy:
    @pytest.mark.parametrize(
        "step_index,expected",
        [
            (0, True),
            (1, True),
            (2, True),
            (3, False),
            (25, False),
            (46, False),
            (47, True),
            (49, True),
        ],
    )
    def test_first_and_last_steps_are_high_precision(self, step_index, expected):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        controller.set_step(step_index, num_steps=50)
        assert controller.high_precision is expected

    def test_selection_is_a_pure_function_of_the_step(self):
        """CFG branches call separately; they must not disagree within a step."""
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        controller.set_step(1, num_steps=50)
        first = controller.high_precision
        controller.set_step(1, num_steps=50)
        assert controller.high_precision is first is True

    def test_single_step_schedule_stays_on_the_quantized_path(self):
        """A one-step schedule is the warmup probe, not an all-edge request."""
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        controller.set_step(0, num_steps=1)
        assert controller.high_precision is False

    def test_zero_windows_disable_the_feature(self):
        controller = StepPrecisionController(first_steps=0, last_steps=0)
        for step in range(4):
            controller.set_step(step, num_steps=4)
            assert controller.high_precision is False

    def test_overlapping_windows_cover_every_step(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        for step in range(4):
            controller.set_step(step, num_steps=4)
            assert controller.high_precision is True

    def test_reset_clears_state(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        controller.set_step(0, num_steps=50)
        controller.reset()
        assert controller.high_precision is False

    @pytest.mark.parametrize("first,last", [(-1, 3), (3, -1)])
    def test_negative_windows_rejected(self, first, last):
        with pytest.raises(ValueError, match="non-negative"):
            StepPrecisionController(first_steps=first, last_steps=last)

    def test_out_of_range_step_rejected(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        with pytest.raises(IndexError):
            controller.set_step(50, num_steps=50)

    def test_non_positive_num_steps_rejected(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        with pytest.raises(ValueError, match="num_steps must be positive"):
            controller.set_step(0, num_steps=0)


class TestDispatch:
    def test_middle_step_uses_the_checkpoint_path(self):
        base = _StubBaseMethod()
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        method = StepPrecisionFp8LinearMethod(base, controller)
        controller.set_step(10, num_steps=50)
        module = _StubLinear()
        method.apply(module, torch.ones(2, 8, dtype=torch.bfloat16))
        assert base.calls == 1

    def test_edge_step_bypasses_the_checkpoint_path(self):
        base = _StubBaseMethod()
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        method = StepPrecisionFp8LinearMethod(base, controller)
        controller.set_step(0, num_steps=50)
        module = _StubLinear()
        out = method.apply(module, torch.ones(2, 8, dtype=torch.bfloat16))
        assert base.calls == 0
        # weight 1.0 * scale 2.0, summed over in_features=8 -> 16 per output.
        assert torch.allclose(out, torch.full_like(out, 16.0))

    def test_high_precision_is_published_for_sharing_callers(self):
        """GatedMLP/Attention read this attribute to stand down. Contract test."""
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        method = StepPrecisionFp8LinearMethod(_StubBaseMethod(), controller)
        controller.set_step(10, num_steps=50)
        assert method.high_precision is False
        controller.set_step(0, num_steps=50)
        assert method.high_precision is True

    def test_wrapper_forwards_unknown_attributes(self):
        base = _StubBaseMethod()
        base.quantizes_nvfp4_activations = False
        method = StepPrecisionFp8LinearMethod(base, StepPrecisionController(3, 3))
        assert method.quantizes_nvfp4_activations is False


class TestW8A16Apply:
    def test_dequantized_weight_matches_scaled_reference(self):
        module = _StubLinear(out_features=3, in_features=4, scale=0.5)
        module.weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        out = apply_fp8_w8a16_linear(module, x, bias=None)
        expected = torch.nn.functional.linear(x, module.weight.to(x.dtype) * 0.5)
        assert torch.allclose(out, expected)

    def test_bias_is_applied(self):
        module = _StubLinear(out_features=2, in_features=3, scale=1.0)
        module.weight = torch.zeros(2, 3, dtype=torch.bfloat16)
        bias = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        out = apply_fp8_w8a16_linear(module, torch.ones(1, 3, dtype=torch.bfloat16), bias)
        assert torch.allclose(out, bias.unsqueeze(0))

    def test_prequantized_activation_is_rejected(self):
        """The failure this feature can have silently: an FP8 activation on a
        16-bit step means a sharing caller did not stand down, and the step is
        not actually running in higher precision."""
        module = _StubLinear()
        x = torch.ones(2, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        with pytest.raises(RuntimeError, match="must stand down"):
            apply_fp8_w8a16_linear(module, x, bias=None)
