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
"""CPU tests for Cosmos3 per-request sigma grid selection (``extra_params["sigma_grid"]``)."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import COSMOS3_EXTRA_SPECS
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import (
    Cosmos3SamplingPolicy,
    native_flow_base_sigmas,
    resolve_sampling_schedule,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.transfer import resolve_transfer_config
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import QWEN3_RECIPE
from tensorrt_llm.visual_gen.params import VisualGenParams, validate_visual_gen_params

# The CPU-only CI stage selects tests with `-m cpu_only`.
pytestmark = pytest.mark.cpu_only

# Scheduler fields shared by the Cosmos3 Nano, Super and Edge base checkpoints.
BASE_UNIPC_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "num_train_timesteps": 1000,
    "flow_shift": 1.0,
    "prediction_type": "flow_prediction",
    "predict_x0": True,
    "use_flow_sigmas": True,
    "use_karras_sigmas": True,
    "sigma_max": 200.0,
    "sigma_min": 0.147,
    "solver_order": 2,
    "solver_type": "bh2",
    "final_sigmas_type": "zero",
    "timestep_spacing": "linspace",
    "lower_order_final": True,
}

DISTILLED_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": True,
    "fixed_step_sampler_config": {"t_list": [1.0, 0.9, 0.6, 0.3], "sample_type": "sde"},
}

# Recorded from cosmos-framework 9a70e64d (`fm_solvers_unipc.py`
# FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0,
# use_dynamic_shifting=False).set_timesteps(steps, shift=shift), the call its
# UniPC sampler makes). Each trajectory runs the synthetic velocity
# v = 0.05*x + 0.3*sin(t/1000) - 0.1 in float64 from
# x0 = linspace(-1, 1, 8).reshape(1, 2, 2, 2) through every step().
# (10.0, 35) is the framework's video default, (3.0, 50) its text-to-image default.
NATIVE_FRAMEWORK_FIXTURES = {
    (10.0, 35): {
        "timesteps": [
            999,
            996,
            993,
            990,
            987,
            983,
            979,
            975,
            971,
            966,
            961,
            956,
            950,
            944,
            937,
            930,
            922,
            913,
            904,
            893,
            882,
            869,
            855,
            838,
            820,
            799,
            775,
            747,
            714,
            673,
            624,
            563,
            483,
            377,
            227,
        ],
        "sigmas": [
            0.9998999238014221,
            0.9969649910926819,
            0.9938710331916809,
            0.9906046390533447,
            0.987151026725769,
            0.9834935665130615,
            0.9796137809753418,
            0.9754906892776489,
            0.9711006283760071,
            0.9664168953895569,
            0.9614089131355286,
            0.9560418725013733,
            0.9502756595611572,
            0.9440640807151794,
            0.9373533725738525,
            0.9300810098648071,
            0.922173261642456,
            0.9135431051254272,
            0.9040868282318115,
            0.8936798572540283,
            0.8821711540222168,
            0.8693760633468628,
            0.8550660610198975,
            0.838955283164978,
            0.8206809759140015,
            0.7997758388519287,
            0.7756277918815613,
            0.7474188208580017,
            0.7140304446220398,
            0.6738920211791992,
            0.624726414680481,
            0.5631024241447449,
            0.48359769582748413,
            0.37710919976234436,
            0.22709190845489502,
            0.0,
        ],
        "final": [
            -0.99361417654613,
            -0.7218476409017736,
            -0.45008110525741596,
            -0.17831456961305864,
            0.09345196603129906,
            0.3652185016756565,
            0.636985037320014,
            0.9087515729643721,
        ],
    },
    (3.0, 50): {
        "timesteps": [
            999,
            992,
            985,
            978,
            971,
            963,
            956,
            948,
            939,
            931,
            922,
            913,
            904,
            894,
            884,
            874,
            864,
            853,
            841,
            829,
            817,
            805,
            792,
            778,
            764,
            749,
            734,
            718,
            701,
            684,
            666,
            647,
            627,
            606,
            585,
            562,
            538,
            512,
            486,
            458,
            428,
            396,
            363,
            327,
            290,
            249,
            206,
            160,
            111,
            57,
        ],
        "sigmas": [
            0.9996664524078369,
            0.9929074645042419,
            0.9859634041786194,
            0.9788264632225037,
            0.9714885950088501,
            0.9639410972595215,
            0.9561749696731567,
            0.9481804966926575,
            0.9399474263191223,
            0.9314650297164917,
            0.9227216839790344,
            0.9137052297592163,
            0.9044026732444763,
            0.8948001265525818,
            0.8848828673362732,
            0.8746352195739746,
            0.8640403151512146,
            0.8530802130699158,
            0.8417357206344604,
            0.8299862742424011,
            0.8178097009658813,
            0.8051823973655701,
            0.7920788526535034,
            0.7784715294837952,
            0.7643308639526367,
            0.7496247887611389,
            0.7343188524246216,
            0.7183755040168762,
            0.7017540335655212,
            0.6844102740287781,
            0.666296124458313,
            0.6473590731620789,
            0.6275417804718018,
            0.6067813038825989,
            0.5850088000297546,
            0.5621483325958252,
            0.5381162762641907,
            0.5128201842308044,
            0.48615768551826477,
            0.45801496505737305,
            0.42826521396636963,
            0.3967667818069458,
            0.36336082220077515,
            0.32786861062049866,
            0.290088415145874,
            0.24979163706302643,
            0.20671817660331726,
            0.16057077050209045,
            0.11100822687149048,
            0.05763683468103409,
            0.0,
        ],
        "final": [
            -0.9869313906225763,
            -0.7151467269877677,
            -0.44336206335296,
            -0.1715773997181523,
            0.1002072639166558,
            0.371991927551464,
            0.6437765911862716,
            0.9155612548210794,
        ],
    },
    (1.0, 35): {
        "timesteps": [
            999,
            970,
            941,
            913,
            884,
            856,
            827,
            799,
            770,
            742,
            713,
            685,
            656,
            627,
            599,
            570,
            542,
            513,
            485,
            456,
            428,
            399,
            371,
            342,
            313,
            285,
            256,
            228,
            199,
            171,
            142,
            114,
            85,
            57,
            28,
        ],
        "sigmas": [
            0.9990000128746033,
            0.970457136631012,
            0.9419143199920654,
            0.9133714437484741,
            0.8848285675048828,
            0.8562857508659363,
            0.827742874622345,
            0.7991999983787537,
            0.7706571817398071,
            0.7421143054962158,
            0.7135714292526245,
            0.6850285530090332,
            0.6564857363700867,
            0.6279428601264954,
            0.599399983882904,
            0.5708571672439575,
            0.5423142910003662,
            0.5137714147567749,
            0.485228568315506,
            0.45668572187423706,
            0.42814287543296814,
            0.39959999918937683,
            0.3710571527481079,
            0.3425142765045166,
            0.3139714300632477,
            0.28542858362197876,
            0.25688570737838745,
            0.22834286093711853,
            0.19979999959468842,
            0.1712571382522583,
            0.14271429181098938,
            0.11417143046855927,
            0.08562856912612915,
            0.05708571523427963,
            0.028542857617139816,
            0.0,
        ],
        "final": [
            -0.9868242505403478,
            -0.7150305797828035,
            -0.4432369090252578,
            -0.17144323826771254,
            0.10035043248983266,
            0.37214410324737796,
            0.6439377740049234,
            0.9157314447624686,
        ],
    },
}


def _policy(*, native_flow_schedule: bool = False) -> tuple[Cosmos3SamplingPolicy, object]:
    base = UniPCMultistepScheduler.from_config(BASE_UNIPC_CONFIG)
    return Cosmos3SamplingPolicy.from_scheduler(
        base, native_flow_schedule=native_flow_schedule
    ), base


def _distilled_policy() -> Cosmos3SamplingPolicy:
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_CONFIG)
    return Cosmos3SamplingPolicy.from_scheduler(scheduler)


def _resolve(policy, *, sigma_grid=None, flow_shift=None, mode_flow_shift=1.0, mode_karras=None):
    return resolve_sampling_schedule(
        policy,
        sigma_grid=sigma_grid,
        flow_shift=flow_shift,
        mode_flow_shift=mode_flow_shift,
        mode_use_karras_sigmas=mode_karras,
    )


def _programmed(policy, base, schedule, steps):
    """Build the request scheduler for ``schedule`` and program its timesteps."""
    scheduler = policy.set_flow_shift(
        base, schedule.flow_shift, use_karras_sigmas=schedule.use_karras_sigmas
    )
    policy.set_timesteps(scheduler, steps, device="cpu", grid=schedule.grid)
    return scheduler


def _trajectory(scheduler) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64).reshape(1, 2, 2, 2)
    for t in scheduler.timesteps:
        v = 0.05 * x + 0.3 * torch.sin(t.double() / 1000.0) - 0.1
        x = scheduler.step(v, t, x, return_dict=False)[0]
    return x


class TestNativeGrid:
    @pytest.mark.parametrize("shift,steps", sorted(NATIVE_FRAMEWORK_FIXTURES))
    def test_matches_cosmos_framework_exactly(self, shift, steps):
        fixture = NATIVE_FRAMEWORK_FIXTURES[(shift, steps)]
        policy, base = _policy()
        schedule = _resolve(policy, sigma_grid="native", flow_shift=shift)
        scheduler = _programmed(policy, base, schedule, steps)

        assert scheduler.timesteps.dtype == torch.int64
        assert scheduler.timesteps.tolist() == fixture["timesteps"]
        assert scheduler.sigmas.dtype == torch.float32
        assert len(scheduler.sigmas) == steps + 1
        assert scheduler.sigmas.tolist() == fixture["sigmas"]
        assert scheduler.sigmas[-1].item() == 0.0
        # The grid above must match exactly. The trajectory checks the solver
        # update on that grid; UniPC derives its step coefficients from float32
        # sigmas, so results recorded on another host agree to float32 precision.
        np.testing.assert_allclose(
            _trajectory(scheduler).flatten().tolist(), fixture["final"], rtol=0, atol=1e-7
        )

    def test_upper_level_is_the_framework_float32_endpoint(self):
        levels = native_flow_base_sigmas(35, 1000)
        assert len(levels) == 35
        assert levels[0] == float(np.float32(0.999))
        assert levels[0] != 0.999
        assert levels[-1] == pytest.approx(0.999 / 35)

    def test_native_is_honored_on_a_native_flow_checkpoint(self):
        """An explicit native grid on Edge-style checkpoints uses the float32
        endpoint, not the checkpoint's legacy float64 one."""
        policy, base = _policy(native_flow_schedule=True)
        fixture = NATIVE_FRAMEWORK_FIXTURES[(10.0, 35)]
        scheduler = _programmed(
            policy, base, _resolve(policy, sigma_grid="native", flow_shift=10.0), 35
        )
        assert scheduler.sigmas.tolist() == fixture["sigmas"]

    def test_native_differs_from_the_legacy_native_flow_grid(self):
        """The float32 upper level changes integer timesteps at some step counts
        (here 9 steps at shift 1) and stored sigmas at the video default."""
        policy, base = _policy(native_flow_schedule=True)
        native = _programmed(policy, base, _resolve(policy, sigma_grid="native"), 9)
        legacy = _programmed(policy, base, _resolve(policy), 9)
        assert legacy.timesteps.tolist() != native.timesteps.tolist()
        assert native.timesteps.tolist() == [999, 888, 777, 666, 555, 444, 333, 222, 111]
        native_video = _programmed(
            policy, base, _resolve(policy, sigma_grid="native", flow_shift=10.0), 35
        )
        legacy_video = _programmed(policy, base, _resolve(policy, mode_flow_shift=10.0), 35)
        assert torch.equal(native_video.timesteps, legacy_video.timesteps)
        assert not torch.equal(native_video.sigmas, legacy_video.sigmas)


class TestDiffusersGrid:
    def test_matches_the_pinned_scheduler_flow_branch(self):
        policy, base = _policy()
        scheduler = _programmed(
            policy, base, _resolve(policy, sigma_grid="diffusers", flow_shift=10.0), 35
        )
        reference = UniPCMultistepScheduler.from_config(
            BASE_UNIPC_CONFIG, flow_shift=10.0, use_karras_sigmas=False
        )
        reference.set_timesteps(35, device="cpu")
        assert torch.equal(scheduler.timesteps, reference.timesteps)
        assert torch.equal(scheduler.sigmas, reference.sigmas)
        # The scheduler's near-one safeguard lowers the first level from 1.0.
        assert scheduler.sigmas[0].item() == pytest.approx(1.0 - 1e-6, abs=1e-7)

    def test_differs_from_native_at_the_video_default(self):
        policy, base = _policy()
        native = _programmed(
            policy, base, _resolve(policy, sigma_grid="native", flow_shift=10.0), 35
        )
        diffusers = _programmed(
            policy, base, _resolve(policy, sigma_grid="diffusers", flow_shift=10.0), 35
        )
        differing = int((native.timesteps != diffusers.timesteps).sum())
        assert differing == 14


class TestKarrasGrid:
    def test_matches_the_checkpoint_karras_grid(self):
        policy, base = _policy()
        scheduler = _programmed(policy, base, _resolve(policy, sigma_grid="karras"), 35)
        reference = UniPCMultistepScheduler.from_config(BASE_UNIPC_CONFIG)
        reference.set_timesteps(35, device="cpu")
        assert scheduler.config.use_karras_sigmas is True
        assert torch.equal(scheduler.timesteps, reference.timesteps)
        assert torch.equal(scheduler.sigmas, reference.sigmas)

    @pytest.mark.parametrize("native_flow_schedule", [False, True])
    @pytest.mark.parametrize("mode_karras", [None, False])
    def test_explicit_karras_reaches_every_path(self, native_flow_schedule, mode_karras):
        """Edge (native flow), V2V and transfer (forced Karras off) honor an
        explicit Karras request instead of their own routing."""
        policy, base = _policy(native_flow_schedule=native_flow_schedule)
        schedule = _resolve(
            policy, sigma_grid="karras", mode_flow_shift=3.0, mode_karras=mode_karras
        )
        assert schedule.grid == "karras"
        assert schedule.use_karras_sigmas is True
        scheduler = _programmed(policy, base, schedule, 35)
        reference = UniPCMultistepScheduler.from_config(BASE_UNIPC_CONFIG)
        reference.set_timesteps(35, device="cpu")
        assert torch.equal(scheduler.sigmas, reference.sigmas)


class TestOmittedSelector:
    """Without sigma_grid every mode keeps its existing grid and shift."""

    def test_base_video_keeps_karras_and_mode_shift(self):
        policy, base = _policy()
        schedule = _resolve(policy, mode_flow_shift=1.0)
        assert (schedule.grid, schedule.flow_shift, schedule.use_karras_sigmas) == (
            None,
            1.0,
            None,
        )
        assert not schedule.inactive_flow_shift
        scheduler = _programmed(policy, base, schedule, 35)
        reference = UniPCMultistepScheduler.from_config(BASE_UNIPC_CONFIG)
        reference.set_timesteps(35, device="cpu")
        assert scheduler is base
        assert torch.equal(scheduler.sigmas, reference.sigmas)

    def test_v2v_keeps_the_diffusers_flow_grid(self):
        policy, base = _policy()
        schedule = _resolve(policy, mode_flow_shift=10.0, mode_karras=False)
        assert (schedule.grid, schedule.flow_shift, schedule.use_karras_sigmas) == (
            None,
            10.0,
            False,
        )
        scheduler = _programmed(policy, base, schedule, 35)
        reference = UniPCMultistepScheduler.from_config(
            BASE_UNIPC_CONFIG, flow_shift=10.0, use_karras_sigmas=False
        )
        reference.set_timesteps(35, device="cpu")
        assert torch.equal(scheduler.sigmas, reference.sigmas)

    @pytest.mark.parametrize("mode_karras", [None, False])
    def test_native_flow_checkpoint_keeps_its_legacy_grid(self, mode_karras):
        policy, base = _policy(native_flow_schedule=True)
        schedule = _resolve(policy, mode_flow_shift=3.0, mode_karras=mode_karras)
        assert schedule.grid is None
        scheduler = _programmed(policy, base, schedule, 35)
        legacy = UniPCMultistepScheduler.from_config(
            BASE_UNIPC_CONFIG, flow_shift=3.0, use_karras_sigmas=False
        )
        legacy.set_timesteps(35, device="cpu", sigmas=np.linspace(1.0 - 1.0 / 1000, 0.0, 36)[:-1])
        assert torch.equal(scheduler.timesteps, legacy.timesteps)
        assert torch.equal(scheduler.sigmas, legacy.sigmas)

    def test_explicit_flow_shift_alone_keeps_the_grid_and_is_marked_inactive(self):
        policy, _ = _policy()
        schedule = _resolve(policy, flow_shift=10.0, mode_flow_shift=1.0)
        assert schedule.grid is None
        assert schedule.flow_shift == 10.0
        assert schedule.inactive_flow_shift

    def test_distilled_keeps_its_fixed_schedule(self):
        schedule = _resolve(_distilled_policy(), flow_shift=5.0, mode_flow_shift=1.0)
        assert schedule.grid is None


class TestRequestContract:
    def test_explicit_karras_with_explicit_flow_shift_is_rejected(self):
        policy, _ = _policy()
        with pytest.raises(ValueError, match="no effect with sigma_grid='karras'"):
            _resolve(policy, sigma_grid="karras", flow_shift=10.0)

    def test_explicit_karras_ignores_an_inherited_shift(self):
        policy, _ = _policy()
        schedule = _resolve(policy, sigma_grid="karras", mode_flow_shift=10.0)
        assert schedule.grid == "karras"
        assert not schedule.inactive_flow_shift

    @pytest.mark.parametrize("grid", ["native", "diffusers"])
    def test_linear_grid_without_flow_shift_uses_the_mode_shift(self, grid):
        policy, _ = _policy()
        schedule = _resolve(policy, sigma_grid=grid, mode_flow_shift=1.0)
        assert schedule.flow_shift == 1.0
        assert schedule.use_karras_sigmas is False

    @pytest.mark.parametrize("grid", ["native", "diffusers"])
    @pytest.mark.parametrize("shift", [0.0, -1.0, float("inf"), float("nan")])
    def test_linear_grid_rejects_invalid_shifts(self, grid, shift):
        policy, _ = _policy()
        with pytest.raises(ValueError, match="positive finite"):
            _resolve(policy, sigma_grid=grid, flow_shift=shift)

    def test_native_rejects_a_shift_the_solver_would_alter(self):
        policy, _ = _policy()
        with pytest.raises(ValueError, match="native grid cannot be reproduced"):
            _resolve(policy, sigma_grid="native", flow_shift=1e4)

    @pytest.mark.parametrize(
        "override",
        [
            {"use_dynamic_shifting": True},
            {"shift_terminal": 0.1},
            {"final_sigmas_type": "sigma_min"},
        ],
    )
    def test_native_requires_default_flow_settings(self, override):
        base = UniPCMultistepScheduler.from_config({**BASE_UNIPC_CONFIG, **override})
        policy = Cosmos3SamplingPolicy.from_scheduler(base)
        with pytest.raises(ValueError, match="default flow settings"):
            _resolve(policy, sigma_grid="native", flow_shift=10.0)
        # The Diffusers grid follows whatever the scheduler config declares.
        assert _resolve(policy, sigma_grid="diffusers", flow_shift=10.0).grid == "diffusers"

    @pytest.mark.parametrize("grid", ["karras", "native", "diffusers"])
    def test_distilled_checkpoints_reject_an_explicit_grid(self, grid):
        with pytest.raises(ValueError, match="distilled"):
            _resolve(_distilled_policy(), sigma_grid=grid)

    def test_non_flow_scheduler_rejects_an_explicit_grid(self):
        base = UniPCMultistepScheduler.from_config({**BASE_UNIPC_CONFIG, "use_flow_sigmas": False})
        policy = Cosmos3SamplingPolicy.from_scheduler(base)
        with pytest.raises(ValueError, match="use_flow_sigmas"):
            _resolve(policy, sigma_grid="native", flow_shift=10.0)

    @pytest.mark.parametrize("value", ["Karras", "linear", "framework", 1])
    def test_unknown_values_are_rejected_at_validation(self, value):
        params = VisualGenParams(extra_params={"sigma_grid": value})
        with pytest.raises(ValueError, match="sigma_grid"):
            validate_visual_gen_params(
                params, declared_defaults=None, extra_param_specs=COSMOS3_EXTRA_SPECS
            )

    @pytest.mark.parametrize("value", ["karras", "native", "diffusers"])
    def test_known_values_pass_validation(self, value):
        params = VisualGenParams(extra_params={"sigma_grid": value, "flow_shift": 10.0})
        validate_visual_gen_params(
            params, declared_defaults=None, extra_param_specs=COSMOS3_EXTRA_SPECS
        )

    def test_selector_defaults_to_unset(self):
        assert COSMOS3_EXTRA_SPECS["sigma_grid"].default is None


class TestIsolation:
    def test_grid_mismatch_with_the_scheduler_config_is_an_error(self):
        policy, base = _policy()
        with pytest.raises(RuntimeError, match="does not match"):
            policy.set_timesteps(base, 35, device="cpu", grid="native")

    def test_requests_never_mutate_the_checkpoint_scheduler(self):
        policy, base = _policy()
        before = copy.deepcopy(dict(base.config))
        policy_before = copy.deepcopy(dict(policy.unipc_base_config))
        for grid, shift in [("native", 10.0), ("diffusers", 3.0), ("karras", None)]:
            _programmed(policy, base, _resolve(policy, sigma_grid=grid, flow_shift=shift), 35)
        assert dict(base.config) == before
        assert dict(policy.unipc_base_config) == policy_before

    def test_alternating_grids_do_not_leak_solver_state(self):
        """karras -> native -> karras on the same schedulers reproduces fresh runs."""
        policy, base = _policy()
        karras = _resolve(policy, sigma_grid="karras")
        native = _resolve(policy, sigma_grid="native", flow_shift=10.0)

        first_karras = _trajectory(_programmed(policy, base, karras, 12))
        native_run = _trajectory(_programmed(policy, base, native, 12))
        second_karras = _trajectory(_programmed(policy, base, karras, 12))

        fresh_policy, fresh_base = _policy()
        fresh_native = _trajectory(_programmed(fresh_policy, fresh_base, native, 12))
        assert torch.equal(first_karras, second_karras)
        assert torch.equal(native_run, fresh_native)


class TestRepeatedLeadingTimesteps:
    """Strongly shifted grids repeat their leading integer timesteps; the solver
    must still run every step of the schedule, in order, from step 0."""

    @pytest.mark.parametrize(
        ("native_flow_schedule", "sigma_grid", "flow_shift", "steps"),
        [
            (False, "native", 100.0, 35),
            (False, "native", 10.0, 113),
            (False, "diffusers", 100.0, 35),
            (True, None, 100.0, 35),
        ],
    )
    def test_full_trajectory_runs_from_the_first_step(
        self, native_flow_schedule, sigma_grid, flow_shift, steps
    ):
        policy, base = _policy(native_flow_schedule=native_flow_schedule)
        schedule = _resolve(policy, sigma_grid=sigma_grid, mode_flow_shift=flow_shift)
        scheduler = _programmed(policy, base, schedule, steps)
        timesteps = scheduler.timesteps.tolist()
        assert len(set(timesteps)) < len(timesteps)

        first = _trajectory(scheduler)
        assert scheduler.step_index == steps
        assert torch.isfinite(first).all()

        policy.set_timesteps(scheduler, steps, device="cpu", grid=schedule.grid)
        assert torch.equal(_trajectory(scheduler), first)


def _bare_pipeline(policy, base) -> Cosmos3OmniMoTPipeline:
    pipeline = Cosmos3OmniMoTPipeline.__new__(Cosmos3OmniMoTPipeline)
    pipeline.sampling = policy
    pipeline.scheduler = base
    pipeline._base_scheduler = base
    pipeline._scheduler_cache = {}
    pipeline.family = QWEN3_RECIPE.name
    pipeline._device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(device=torch.device("cpu"))
    return pipeline


class TestPipelineSchedulerCache:
    def test_linear_grids_get_separate_cache_entries(self):
        policy, base = _policy()
        pipeline = _bare_pipeline(policy, base)
        native = pipeline._scheduler_for_schedule(
            _resolve(policy, sigma_grid="native", flow_shift=10.0)
        )
        diffusers = pipeline._scheduler_for_schedule(
            _resolve(policy, sigma_grid="diffusers", flow_shift=10.0)
        )
        assert native is not diffusers
        assert {key[0] for key in pipeline._scheduler_cache} == {"native", "diffusers"}
        # A recipe is built once and reused.
        again = pipeline._scheduler_for_schedule(
            _resolve(policy, sigma_grid="native", flow_shift=10.0)
        )
        assert again is native

    def test_caller_supplied_shifts_stay_out_of_the_cache(self):
        policy, base = _policy()
        pipeline = _bare_pipeline(policy, base)
        pipeline._scheduler_for_schedule(_resolve(policy, sigma_grid="native", flow_shift=7.25))
        assert pipeline._scheduler_cache == {}


class _StopAfterTokenize(Exception):
    pass


def _request_pipeline(policy, base) -> Cosmos3OmniMoTPipeline:
    """A pipeline that runs forward() through request resolution and stops at tokenization."""
    pipeline = _bare_pipeline(policy, base)
    pipeline.audio_gen = False
    pipeline.action_gen = False
    pipeline.default_use_system_prompt = False

    def stop(*args, **kwargs):
        raise _StopAfterTokenize

    pipeline._tokenize_prompt = stop
    return pipeline


def _forward(pipeline, **overrides):
    kwargs = dict(
        prompt="a robot arm cleaning a plate",
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=1,
        max_sequence_length=8,
        frame_rate=8.0,
        use_duration_template=False,
        use_resolution_template=False,
        use_guardrails=False,
    )
    kwargs.update(overrides)
    pipeline.forward(**kwargs)


class TestRequestPath:
    def test_native_selection_reaches_the_request_scheduler(self):
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, sigma_grid="native", flow_shift=10.0)
        assert float(pipeline.scheduler.config.flow_shift) == 10.0
        assert pipeline.scheduler.config.use_karras_sigmas is False
        assert ("native", 10.0, False, "video") in pipeline._scheduler_cache
        # The checkpoint scheduler is untouched; the next omitted request uses it again.
        assert base.config.use_karras_sigmas is True
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline)
        assert pipeline.scheduler is base

    def test_conflicting_explicit_values_fail_the_request(self):
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(ValueError, match="no effect with sigma_grid='karras'"):
            _forward(pipeline, sigma_grid="karras", flow_shift=10.0)

    def test_inactive_flow_shift_warns_once_per_request(self, monkeypatch):
        import tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 as pipe_mod

        warnings = []
        monkeypatch.setattr(pipe_mod.logger, "warning", warnings.append)
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, flow_shift=10.0)
        assert len([w for w in warnings if "has no effect" in w]) == 1
        assert pipeline.scheduler.config.use_karras_sigmas is True

    def test_no_warning_when_the_shift_applies(self, monkeypatch):
        import tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 as pipe_mod

        warnings = []
        monkeypatch.setattr(pipe_mod.logger, "warning", warnings.append)
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, sigma_grid="native", flow_shift=10.0)
        edge_policy, edge_base = _policy(native_flow_schedule=True)
        edge = _request_pipeline(edge_policy, edge_base)
        with pytest.raises(_StopAfterTokenize):
            _forward(edge, flow_shift=10.0)
        assert not [w for w in warnings if "has no effect" in w]

    def test_inactive_flow_shift_warns_on_rank_zero_only(self, monkeypatch):
        import tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 as pipe_mod

        warnings = []
        monkeypatch.setattr(pipe_mod.logger, "warning", warnings.append)
        monkeypatch.setattr(Cosmos3OmniMoTPipeline, "rank", property(lambda self: 1))
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, flow_shift=10.0)
        assert not [w for w in warnings if "has no effect" in w]

    def test_every_stream_runs_the_selected_grid(self):
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        for stream in ("audio", "action"):
            copy_ = UniPCMultistepScheduler.from_config(base.config)
            setattr(pipeline, f"{stream}_scheduler", copy_)
            setattr(pipeline, f"_base_{stream}_scheduler", copy_)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, sigma_grid="native", flow_shift=10.0)
        for scheduler in (pipeline.scheduler, pipeline.audio_scheduler, pipeline.action_scheduler):
            assert float(scheduler.config.flow_shift) == 10.0
            assert scheduler.config.use_karras_sigmas is False
        assert {key[3] for key in pipeline._scheduler_cache} == {"video", "audio", "action"}
        assert {key[0] for key in pipeline._scheduler_cache} == {"native"}

    def test_v2v_honors_an_explicit_grid(self):
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, video=_V2V_FIXTURE_MP4.read_bytes(), sigma_grid="karras")
        assert pipeline.scheduler.config.use_karras_sigmas is True
        with pytest.raises(_StopAfterTokenize):
            _forward(pipeline, video=_V2V_FIXTURE_MP4.read_bytes(), sigma_grid="native")
        assert pipeline.scheduler.config.use_karras_sigmas is False
        assert float(pipeline.scheduler.config.flow_shift) == 10.0
        assert ("native", 10.0, False, "video") in pipeline._scheduler_cache


_V2V_FIXTURE_MP4 = Path(__file__).parent / "test_data" / "cosmos3_v2v_ref_9f_bframes.mp4"


def _transfer_config():
    return resolve_transfer_config(
        {"edge": True}, SimpleNamespace(num_frames=93, guidance_scale=None), None
    )


class TestTransferRequestPath:
    def _pipeline(self, policy, base, captured):
        pipeline = _request_pipeline(policy, base)

        def fake_forward_transfer(**kwargs):
            captured.update(kwargs)

        pipeline._forward_transfer = fake_forward_transfer
        return pipeline

    def test_selection_reaches_the_transfer_path(self):
        captured = {}
        policy, base = _policy()
        pipeline = self._pipeline(policy, base, captured)
        _forward(
            pipeline,
            video=_V2V_FIXTURE_MP4.read_bytes(),
            transfer_config=_transfer_config(),
            sigma_grid="native",
            flow_shift=10.0,
        )
        assert captured["sigma_grid"] == "native"
        assert captured["flow_shift"] == 10.0

    @pytest.mark.parametrize(
        "policy_factory,overrides,match",
        [
            (lambda: _policy()[0], {"sigma_grid": "karras", "flow_shift": 10.0}, "no effect"),
            (_distilled_policy, {"sigma_grid": "native"}, "distilled"),
        ],
        ids=["karras-with-shift", "distilled"],
    )
    def test_invalid_selection_fails_before_transfer_work(self, policy_factory, overrides, match):
        captured = {}
        policy = policy_factory()
        base = UniPCMultistepScheduler.from_config(BASE_UNIPC_CONFIG)
        pipeline = self._pipeline(policy, base, captured)
        with pytest.raises(ValueError, match=match):
            _forward(
                pipeline,
                video=_V2V_FIXTURE_MP4.read_bytes(),
                transfer_config=_transfer_config(),
                **overrides,
            )
        assert captured == {}


class TestInferPlumbing:
    def test_infer_passes_the_selection_to_forward(self, monkeypatch):
        policy, base = _policy()
        pipeline = _request_pipeline(policy, base)
        captured = {}
        monkeypatch.setattr(pipeline, "forward", lambda **kwargs: captured.update(kwargs))
        params = VisualGenParams(
            height=16,
            width=16,
            num_frames=5,
            num_inference_steps=4,
            guidance_scale=1.0,
            frame_rate=8.0,
            extra_params={"sigma_grid": "native", "flow_shift": 10.0, "use_guardrails": False},
        )
        pipeline.infer(SimpleNamespace(prompt="a robot arm", params=params))
        assert captured["sigma_grid"] == "native"
        assert captured["flow_shift"] == 10.0
