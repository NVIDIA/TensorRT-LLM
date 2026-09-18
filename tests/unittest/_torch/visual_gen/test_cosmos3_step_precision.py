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

from tensorrt_llm._torch.modules.linear import W8A16FP8LinearMethod
from tensorrt_llm._torch.visual_gen.models.cosmos3.step_precision import (
    StepPrecisionController,
    install_step_precision,
    parse_diffusion_step_policy,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _policy(**overrides):
    policy = {
        "schema_version": 1,
        "type": "first_last_n",
        "index_space": "denoising_loop_iteration",
        "scope": ["transformer"],
        "default_mode": "native",
        "first_steps": {"count": 3, "mode": "a16"},
        "last_steps": {"count": 3, "mode": "a16"},
        "overlap": "a16",
        "reasoner": "a16",
    }
    policy.update(overrides)
    return {"runtime": {"diffusion_step_policy": policy}}


# Pure-CPU test (stub modules, no device): runs in the CPU lane (l0_cpu.yml),
# which selects with `-m cpu_only`; GPU stages deselect it via `not cpu_only`.
pytestmark = pytest.mark.cpu_only


class _StubBaseMethod:
    """Stands in for FP8QDQLinearMethod; records whether the FP8 path ran."""

    # Mirrors LinearMethodBase: every method declares this, so callers read it
    # directly and a rename fails loudly instead of silently defaulting.
    requires_unquantized_activation = False

    def __init__(self):
        self.calls = 0

    def apply(self, module, input, bias=None):
        self.calls += 1
        return torch.zeros(input.shape[0], module.weight.shape[0], dtype=input.dtype)


class _StubLinear(torch.nn.Module):
    def __init__(self, out_features=4, in_features=8, scale=2.0):
        super().__init__()
        # float8: W8A16FP8LinearMethod rejects anything else, since it can
        # only reinterpret an FP8 checkpoint, never quantize one.
        self.weight = torch.ones(out_features, in_features).to(torch.float8_e4m3fn)
        self.weight_scale = torch.tensor(scale, dtype=torch.float32)
        self.quant_method = _StubBaseMethod()


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


def _fp8_linear():
    """A real Linear with a real FP8QDQLinearMethod, which is what the
    installer looks for. Weights are not created; only the method matters."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.linear import Linear

    return Linear(
        8,
        8,
        dtype=torch.bfloat16,
        quant_config=ModelConfig(
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)
        ).get_quant_config(),
    )


def _holder(*modules):
    """An nn.Module parent, since install_step_precision walks .modules()."""
    parent = torch.nn.Module()
    for i, m in enumerate(modules):
        parent.add_module(f"child{i}", m)
    return parent


class TestDispatch:
    """The controller binds one of two methods; nothing wraps or proxies."""

    def _governed(self, first=3, last=3):
        controller = StepPrecisionController(first_steps=first, last_steps=last)
        module = _StubLinear()
        base = module.quant_method
        controller.govern(module, base)
        return controller, module, base

    def test_middle_step_binds_the_checkpoint_path(self):
        controller, module, base = self._governed()
        controller.set_step(10, num_steps=50)
        assert module.quant_method is base

    def test_edge_step_binds_the_16bit_path(self):
        controller, module, base = self._governed()
        controller.set_step(0, num_steps=50)
        assert isinstance(module.quant_method, W8A16FP8LinearMethod)

    def test_reset_restores_the_checkpoint_path(self):
        controller, module, base = self._governed()
        controller.set_step(0, num_steps=50)
        controller.reset()
        assert module.quant_method is base

    def test_the_16bit_method_is_shared_across_modules(self):
        """Stateless, so one instance serves every Linear: no per-module copy."""
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        a, b = _StubLinear(), _StubLinear()
        controller.govern(a, a.quant_method)
        controller.govern(b, b.quant_method)
        controller.set_step(0, num_steps=50)
        assert a.quant_method is b.quant_method

    def test_stand_down_contract_is_published_by_the_bound_method(self):
        """GatedMLP/Attention read this to stand down. Contract test."""
        controller, module, _ = self._governed()
        controller.set_step(10, num_steps=50)
        assert not module.quant_method.requires_unquantized_activation
        controller.set_step(0, num_steps=50)
        assert module.quant_method.requires_unquantized_activation


class TestW8A16Apply:
    def test_dequantized_weight_matches_scaled_reference(self):
        module = _StubLinear(out_features=3, in_features=4, scale=0.5)
        module.weight = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        out = W8A16FP8LinearMethod().apply(module, x, bias=None)
        expected = torch.nn.functional.linear(x, module.weight.to(x.dtype) * 0.5)
        assert torch.allclose(out, expected)

    def test_bias_is_applied(self):
        module = _StubLinear(out_features=2, in_features=3, scale=1.0)
        module.weight = torch.zeros(2, 3).to(torch.float8_e4m3fn)
        bias = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        out = W8A16FP8LinearMethod().apply(module, torch.ones(1, 3, dtype=torch.bfloat16), bias)
        assert torch.allclose(out, bias.unsqueeze(0))

    def test_prequantized_activation_is_rejected(self):
        """The failure this feature can have silently: an FP8 activation on a
        16-bit step means a sharing caller did not stand down, and the step is
        not actually running in higher precision."""
        module = _StubLinear()
        x = torch.ones(2, 8, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        with pytest.raises(RuntimeError, match="16-bit activation"):
            W8A16FP8LinearMethod().apply(module, x, bias=None)

    def test_fp32_activation_is_rejected(self):
        """A16 is enforced, not merely documented: fp32 would silently be A32."""
        module = _StubLinear()
        x = torch.ones(2, 8, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="16-bit activation"):
            W8A16FP8LinearMethod().apply(module, x, bias=None)

    def test_non_fp8_weights_are_rejected(self):
        """The method reinterprets an FP8 checkpoint; it cannot quantize one."""
        module = _StubLinear()
        module.weight = torch.ones(4, 8, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="float8_e4m3fn weights"):
            W8A16FP8LinearMethod().apply(module, torch.ones(2, 8, dtype=torch.bfloat16), bias=None)


class TestPolicyParsing:
    """The checkpoint states the recipe; a shape we do not implement must fail.

    Half-honouring a policy is the dangerous outcome: a checkpoint that asked
    for something we silently ignored looks exactly like the feature working.
    """

    def test_published_policy_is_accepted(self):
        policy = parse_diffusion_step_policy(_policy())
        assert (policy.first_steps, policy.last_steps) == (3, 3)
        assert policy.reasoner_high_precision is True

    @pytest.mark.parametrize(
        "config",
        [None, {}, {"runtime": {}}, {"runtime": {"other": 1}}, "not-a-mapping"],
        ids=["none", "empty", "no-policy", "other-key", "not-mapping"],
    )
    def test_absent_policy_returns_none(self, config):
        assert parse_diffusion_step_policy(config) is None

    def test_scope_without_transformer_is_inert(self):
        assert parse_diffusion_step_policy(_policy(scope=["vae"])) is None

    def test_reasoner_native_is_honoured(self):
        policy = parse_diffusion_step_policy(_policy(reasoner="native"))
        assert policy.reasoner_high_precision is False

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"schema_version": 2}, "schema_version"),
            ({"schema_version": True}, "schema_version"),
            ({"type": "every_n"}, "type"),
            ({"index_space": "sampler_step"}, "index_space"),
            ({"default_mode": "a16"}, "default_mode"),
            ({"overlap": "native"}, "overlap"),
            ({"reasoner": "fp8"}, "reasoner"),
        ],
    )
    def test_unimplemented_policy_shapes_are_refused(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            parse_diffusion_step_policy(_policy(**overrides))

    def test_unknown_field_is_refused(self):
        with pytest.raises(ValueError, match="Unknown diffusion_step_policy fields"):
            parse_diffusion_step_policy(_policy(future_knob="x"))

    def test_missing_field_is_refused(self):
        config = _policy()
        del config["runtime"]["diffusion_step_policy"]["overlap"]
        with pytest.raises(ValueError, match="Missing diffusion_step_policy fields"):
            parse_diffusion_step_policy(config)

    @pytest.mark.parametrize(
        "value", [{"count": -1, "mode": "a16"}, {"count": 3, "mode": "native"}, {"count": 3}]
    )
    def test_bad_step_range_is_refused(self, value):
        with pytest.raises((ValueError, TypeError)):
            parse_diffusion_step_policy(_policy(first_steps=value))

    def test_non_mapping_policy_is_refused(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            parse_diffusion_step_policy({"runtime": {"diffusion_step_policy": []}})


class TestReasonerPath:
    """The reasoner runs once per request, on the first transformer call, so
    its precision is stated by the policy rather than derived from a step
    index -- which would match only while that call lands inside a window."""

    def test_always_high_ignores_the_step(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        module = _fp8_linear()
        assert install_step_precision([_holder(module)], controller, always_high=True) == 1
        for step in (0, 10, 25, 49):
            controller.set_step(step, num_steps=50)
            assert isinstance(module.quant_method, W8A16FP8LinearMethod)

    def test_generation_path_still_follows_the_step(self):
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        module = _StubLinear()
        base = module.quant_method
        controller.govern(module, base)
        controller.set_step(25, num_steps=50)
        assert module.quant_method is base

    def test_reasoner_is_never_rebound_by_a_later_step(self):
        """Bound once at install: nothing in the step loop governs it."""
        controller = StepPrecisionController(first_steps=3, last_steps=3)
        module = _fp8_linear()
        install_step_precision([_holder(module)], controller, always_high=True)
        bound = module.quant_method
        controller.set_step(25, num_steps=50)
        controller.reset()
        assert module.quant_method is bound


class TestTransformerWiring:
    """Pins the wiring between checkpoint and towers.

    The component tests call install_step_precision themselves, so they cannot
    see a mistake in how the transformer decides: reading the wrong config key,
    or giving the unconditional path to the generation tower instead of the
    reasoner. Both would leave every other test green.
    """

    @staticmethod
    def _fp8_linear():
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.linear import Linear

        return Linear(
            8,
            8,
            dtype=torch.bfloat16,
            quant_config=ModelConfig(
                quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)
            ).get_quant_config(),
        )

    def _stub_transformer(self, quantization_config):
        """A transformer-shaped object: the two towers and the model config."""
        from types import SimpleNamespace

        from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

        model_config = DiffusionModelConfig(
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
            pretrained_config=SimpleNamespace(quantization_config=quantization_config),
        )
        return SimpleNamespace(
            model_config=model_config,
            gen_layers=torch.nn.ModuleList([self._fp8_linear()]),
            language_model=SimpleNamespace(layers=torch.nn.ModuleList([self._fp8_linear()])),
            step_precision_controller=None,
        )

    @staticmethod
    def _install(stub):
        from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
            Cosmos3VFMTransformer,
        )

        Cosmos3VFMTransformer._maybe_install_step_precision(stub)

    def test_policy_wraps_generation_step_gated_and_reasoner_always(self):
        stub = self._stub_transformer(_policy())
        self._install(stub)

        assert stub.step_precision_controller is not None
        gen = stub.gen_layers[0].quant_method
        reasoner = stub.language_model.layers[0].quant_method
        # The distinguishing property: mid-schedule the towers disagree.
        stub.step_precision_controller.set_step(25, num_steps=50)
        gen = stub.gen_layers[0].quant_method
        reasoner = stub.language_model.layers[0].quant_method
        assert not isinstance(gen, W8A16FP8LinearMethod), "generation tower is not step-gated"
        assert isinstance(reasoner, W8A16FP8LinearMethod), "reasoner tower is not unconditional"

    def test_reasoner_native_leaves_the_reasoner_alone(self):
        stub = self._stub_transformer(_policy(reasoner="native"))
        self._install(stub)
        stub.step_precision_controller.set_step(0, num_steps=50)
        assert isinstance(stub.gen_layers[0].quant_method, W8A16FP8LinearMethod)
        assert not isinstance(stub.language_model.layers[0].quant_method, W8A16FP8LinearMethod)

    def test_checkpoint_without_a_policy_governs_nothing(self):
        stub = self._stub_transformer({})
        self._install(stub)
        assert stub.step_precision_controller is None
        for tower in (stub.gen_layers[0], stub.language_model.layers[0]):
            assert not isinstance(tower.quant_method, W8A16FP8LinearMethod)

    def test_policy_is_read_from_the_documented_config_key(self):
        """A policy under any other key must not be picked up."""
        stub = self._stub_transformer({"diffusion_step_policy": _policy()["runtime"]})
        self._install(stub)
        assert stub.step_precision_controller is None

    def test_installing_twice_keeps_the_layers_on_the_live_controller(self) -> None:
        """post_load_weights running twice must not orphan the layers.

        The second install must leave every governed layer steered by the
        controller the transformer now holds; otherwise set_denoising_step
        stops reaching them, silently, with the edge steps landing on the
        quantized path.
        """
        stub = self._stub_transformer(_policy())
        self._install(stub)
        first_controller = stub.step_precision_controller

        self._install(stub)
        second_controller = stub.step_precision_controller
        assert second_controller is not None, "controller was cleared by the second install"
        assert first_controller is not None

        # The live controller actually steers the generation tower.
        second_controller.set_step(0, num_steps=50)
        assert isinstance(stub.gen_layers[0].quant_method, W8A16FP8LinearMethod)
        second_controller.set_step(25, num_steps=50)
        assert not isinstance(stub.gen_layers[0].quant_method, W8A16FP8LinearMethod)
        # And the reasoner stays unconditional across both.
        assert isinstance(stub.language_model.layers[0].quant_method, W8A16FP8LinearMethod)
