# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component tests for step precision: <component> x <StepPrecisionConfig>.

Level-1 policy tests live in test_cosmos3_step_precision.py and use stubs.
These build the real quantized components the feature acts on -- a split
GatedMLP and a shared-quant Attention -- drive them through the config, and
compare feature-on against feature-off in the same job. There is no stored
reference: the comparisons are either exact arithmetic or an A/B against the
same module's other path.

The property that actually matters is not "a flag flipped". These components
quantize the shared activation *above* the Linear, so if they do not stand
down while a high-precision step is selected, the step still runs on FP8
activations and the feature is silently absent. That is what is pinned here.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.visual_gen.models.cosmos3.step_precision import (
    StepPrecisionController,
    install_step_precision,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import StepPrecisionConfig

HIDDEN, INTERMEDIATE, TOKENS = 512, 1024, 128
# Chosen so the whole chain stays inside FP8's representable range. The
# structural tests next door use much smaller scales, which is fine when only
# topology is asserted, but here the intermediate activation would underflow
# FP8 to exactly zero and every numerical comparison would be vacuous.
INPUT_SCALE, WEIGHT_SCALE = 1e-2, 1.0
NUM_STEPS = 50

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_mlp(split=True):
    config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8))
    mlp = GatedMLP(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        bias=False,
        activation=F.silu,
        dtype=torch.bfloat16,
        config=config,
        split_gate_up=split,
    ).cuda()
    for linear in _linears(mlp):
        linear.weight.data.copy_(
            (torch.randn(*linear.weight.shape, device="cuda", dtype=torch.bfloat16) * 0.05).to(
                torch.float8_e4m3fn
            )
        )
        linear.weight_scale.data.fill_(WEIGHT_SCALE)
        linear.input_scale.data.fill_(INPUT_SCALE)
        linear.inv_input_scale.data.fill_(1.0 / INPUT_SCALE)
    return mlp


def _linears(mlp):
    names = ("gate_proj", "up_proj") if mlp.split_gate_up else ("gate_up_proj",)
    return [getattr(mlp, n) for n in names] + [mlp.down_proj]


def _install(mlp, config: StepPrecisionConfig):
    """Install exactly as the transformer does, straight from the config object."""
    if not config.enable:
        return None
    controller = StepPrecisionController(
        first_steps=config.first_steps, last_steps=config.last_steps
    )
    assert install_step_precision([mlp], controller) == len(_linears(mlp))
    return controller


@requires_cuda
def test_config_defaults_are_on_with_three_step_windows():
    """The shipped default: enabled, three steps at each end."""
    config = StepPrecisionConfig()
    assert config.enable is True
    assert (config.first_steps, config.last_steps) == (3, 3)


@requires_cuda
def test_disabled_config_installs_nothing():
    mlp = _make_mlp()
    assert _install(mlp, StepPrecisionConfig(enable=False)) is None
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    # Without a wrapper nothing publishes high_precision, so the shared-input
    # optimization stays engaged on every step.
    assert mlp._can_share_gate_up_quantization(x) is True


@requires_cuda
def test_shared_quantization_stands_down_only_on_edge_steps():
    """The integration property the whole feature rests on.

    gate/up consume one activation, which the split path quantizes once above
    the Linear. On a 16-bit step that must not happen, or the step receives FP8
    activations and is not actually running in higher precision.
    """
    mlp = _make_mlp()
    controller = _install(mlp, StepPrecisionConfig())
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    controller.set_step(0, NUM_STEPS)
    assert mlp._can_share_gate_up_quantization(x) is False, "edge step still pre-quantizes"

    controller.set_step(NUM_STEPS // 2, NUM_STEPS)
    assert mlp._can_share_gate_up_quantization(x) is True, "middle step lost the optimization"

    controller.set_step(NUM_STEPS - 1, NUM_STEPS)
    assert mlp._can_share_gate_up_quantization(x) is False, "final step still pre-quantizes"


@requires_cuda
def test_edge_step_linear_matches_exact_dequantized_reference():
    """On a 16-bit step a projection is plain bf16 arithmetic, so it is exact.

    The reference is the same resident FP8 weight dequantized by its own
    weight_scale -- nothing is read that the checkpoint did not already supply.
    """
    mlp = _make_mlp()
    controller = _install(mlp, StepPrecisionConfig())
    controller.set_step(0, NUM_STEPS)

    linear = mlp.gate_proj
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    got = linear.quant_method.apply(linear, x, None)
    expected = F.linear(x, linear.weight.to(x.dtype) * linear.weight_scale.to(x.dtype))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@requires_cuda
def test_edge_and_middle_steps_produce_different_output():
    """Feature-on vs feature-off, same module, same input, same job.

    If these matched, the 16-bit step would be doing nothing -- which is the
    failure mode a flag-only test cannot see.
    """
    mlp = _make_mlp()
    controller = _install(mlp, StepPrecisionConfig())
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    controller.set_step(NUM_STEPS // 2, NUM_STEPS)
    quantized = mlp(x).float()
    controller.set_step(0, NUM_STEPS)
    high_precision = mlp(x).float()

    assert torch.isfinite(high_precision).all()
    assert high_precision.abs().mean().item() > 0, "output collapsed to zero"
    # Scale-relative: an absolute tolerance here would pass trivially whenever
    # the configured scales push the output near zero.
    relative = ((high_precision - quantized).abs().mean() / high_precision.abs().mean()).item()
    assert relative > 1e-3, (
        f"the 16-bit step produced the same output as the quantized step "
        f"(relative difference {relative:.3g}), so it did not engage"
    )


@requires_cuda
def test_edge_step_is_closer_to_the_unquantized_reference():
    """The feature's actual claim: less activation-quantization error.

    Reference runs the same weights and activations with no activation
    quantization anywhere, which is what a 16-bit step approximates.
    """
    mlp = _make_mlp()
    controller = _install(mlp, StepPrecisionConfig())
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def dequant(linear):
        return linear.weight.to(torch.float32) * linear.weight_scale.to(torch.float32)

    xf = x.float()
    gate = F.linear(xf, dequant(mlp.gate_proj))
    up = F.linear(xf, dequant(mlp.up_proj))
    reference = F.linear(F.silu(gate) * up, dequant(mlp.down_proj))

    controller.set_step(NUM_STEPS // 2, NUM_STEPS)
    quantized_err = (mlp(x).float() - reference).abs().mean().item()
    controller.set_step(0, NUM_STEPS)
    high_precision_err = (mlp(x).float() - reference).abs().mean().item()

    assert high_precision_err < quantized_err, (
        f"16-bit step was not closer to the unquantized reference: "
        f"{high_precision_err:.6g} vs {quantized_err:.6g}"
    )


@requires_cuda
def test_zero_windows_keep_every_step_quantized():
    """An operator can turn the behaviour off without turning the feature off."""
    mlp = _make_mlp()
    controller = _install(mlp, StepPrecisionConfig(first_steps=0, last_steps=0))
    x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    for step in (0, NUM_STEPS // 2, NUM_STEPS - 1):
        controller.set_step(step, NUM_STEPS)
        assert mlp._can_share_gate_up_quantization(x) is True
