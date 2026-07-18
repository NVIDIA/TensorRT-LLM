# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Cosmos3 sampling policy (base vs distilled checkpoints)
and its pipeline wiring: scheduler loading, recipe validation, generation
defaults, mode resolution, and the guidance-1.0 denoise-loop contract."""

import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_T2I_PARAMS,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import (
    DISTILLED_GUIDANCE_SCALE,
    Cosmos3SamplingPolicy,
    load_scheduler,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

pytestmark = [pytest.mark.cosmos3, pytest.mark.usefixtures("disable_cosmos3_guardrails")]

# The relevant subset of the 4-Step checkpoint's scheduler config
# (values verbatim; keys unrelated to distilled detection omitted).
DISTILLED_SIGMAS = (1.0, 0.9375, 0.8333333333333334, 0.625)
DISTILLED_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": True,
    "use_karras_sigmas": False,
    "fixed_step_requires_explicit_sigmas": True,
    "fixed_step_sampler_config": {
        "sample_type": "sde",
        "t_list": list(DISTILLED_SIGMAS),
    },
}

UNIPC_SCHEDULER_CONFIG = {
    "_class_name": "UniPCMultistepScheduler",
    "num_train_timesteps": 1000,
    "flow_shift": 1.0,
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "solver_order": 2,
}

SKIP_NON_SCHEDULER = ["text_tokenizer", "tokenizer", "vae", "sound_tokenizer"]


def _write_scheduler_config(checkpoint_dir, config: dict) -> None:
    scheduler_dir = checkpoint_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    with open(scheduler_dir / "scheduler_config.json", "w") as f:
        json.dump(config, f)


def _distilled_policy() -> Cosmos3SamplingPolicy:
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_SCHEDULER_CONFIG)
    return Cosmos3SamplingPolicy.from_scheduler(scheduler)


def _base_scheduler() -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler.from_config(UNIPC_SCHEDULER_CONFIG)


def _base_policy() -> Cosmos3SamplingPolicy:
    return Cosmos3SamplingPolicy.from_scheduler(_base_scheduler())


def _bare_pipeline(**attrs) -> Cosmos3OmniMoTPipeline:
    """A pipeline instance without heavyweight __init__; ``rank``/``dtype``/
    ``device`` are BasePipeline properties and must not be set here."""
    pipeline = object.__new__(Cosmos3OmniMoTPipeline)
    defaults = dict(
        audio_gen=False,
        action_gen=False,
        sampling=Cosmos3SamplingPolicy(),
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(pipeline, key, value)
    return pipeline


def _fake_request(output_type="video", **param_overrides):
    """A DiffusionRequest look-alike with executor-merged (None = unset) params."""
    params = SimpleNamespace(
        height=None,
        width=None,
        num_inference_steps=None,
        guidance_scale=None,
        num_frames=COSMOS3_720P_PARAMS["num_frames"],
        max_sequence_length=COSMOS3_720P_PARAMS["max_sequence_length"],
        frame_rate=COSMOS3_720P_PARAMS["frame_rate"],
        seed=0,
        negative_prompt=None,
        image=None,
        extra_params={"output_type": output_type},
    )
    for key, value in param_overrides.items():
        setattr(params, key, value)
    return SimpleNamespace(prompt="x", params=params)


class TestSchedulerLoading:
    def test_flow_match_declared(self, tmp_path):
        _write_scheduler_config(tmp_path, DISTILLED_SCHEDULER_CONFIG)
        assert isinstance(load_scheduler(str(tmp_path)), FlowMatchEulerDiscreteScheduler)

    def test_unipc_declared(self, tmp_path):
        _write_scheduler_config(tmp_path, UNIPC_SCHEDULER_CONFIG)
        assert isinstance(load_scheduler(str(tmp_path)), UniPCMultistepScheduler)

    def test_missing_class_name_defaults_to_unipc(self, tmp_path):
        config = {k: v for k, v in UNIPC_SCHEDULER_CONFIG.items() if k != "_class_name"}
        _write_scheduler_config(tmp_path, config)
        assert isinstance(load_scheduler(str(tmp_path)), UniPCMultistepScheduler)

    def test_unknown_class_name_raises(self, tmp_path):
        """Silently substituting UniPC for an unknown declared scheduler would
        sample the checkpoint with the wrong integrator."""
        _write_scheduler_config(tmp_path, {**UNIPC_SCHEDULER_CONFIG, "_class_name": "DDIM"})
        with pytest.raises(ValueError, match="DDIM"):
            load_scheduler(str(tmp_path))


class TestPolicyFacts:
    def test_is_distilled(self):
        assert _distilled_policy().is_distilled
        assert not _base_policy().is_distilled
        assert not Cosmos3SamplingPolicy().is_distilled

    def test_diffusers_retains_unexpected_config_keys(self):
        """Canary: diffusers must keep the unexpected fixed_step_sampler_config
        key in scheduler.config; if an upgrade drops it, distilled detection
        silently breaks."""
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_SCHEDULER_CONFIG)
        policy = Cosmos3SamplingPolicy.from_scheduler(scheduler)
        assert policy.fixed_sigmas == DISTILLED_SIGMAS
        assert scheduler.config.stochastic_sampling is True

    def test_sigma_values_coerced_to_floats(self):
        config = {
            **DISTILLED_SCHEDULER_CONFIG,
            "fixed_step_sampler_config": {"t_list": [1, "0.5"]},
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
        assert Cosmos3SamplingPolicy.from_scheduler(scheduler).fixed_sigmas == (1.0, 0.5)

    def test_generation_default_overrides(self):
        assert _distilled_policy().generation_default_overrides() == {
            "num_inference_steps": 4,
            "guidance_scale": DISTILLED_GUIDANCE_SCALE,
        }
        assert _base_policy().generation_default_overrides() == {}

    def test_num_steps(self):
        assert _distilled_policy().num_steps(2) == 4
        assert _base_policy().num_steps(2) == 2

    def test_checkpoint_flow_shift(self):
        assert _base_policy().checkpoint_flow_shift == 1.0
        assert _distilled_policy().checkpoint_flow_shift == 1.0  # no UniPC config

    def test_scheduler_step_kwargs(self):
        generator = torch.Generator().manual_seed(7)
        assert _distilled_policy().scheduler_step_kwargs(generator) == {"generator": generator}
        assert _base_policy().scheduler_step_kwargs(generator) == {}


class TestMalformedRecipeValidation:
    """Only two recipes are valid; everything else must fail at load."""

    @pytest.mark.parametrize("broken_fixed_step", [None, {}, {"t_list": []}])
    def test_required_sigmas_missing_raises(self, broken_fixed_step):
        config = {k: v for k, v in DISTILLED_SCHEDULER_CONFIG.items()}
        config.pop("fixed_step_sampler_config")
        if broken_fixed_step is not None:
            config["fixed_step_sampler_config"] = broken_fixed_step
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)

        with pytest.raises(ValueError, match="fixed_step_requires_explicit_sigmas"):
            Cosmos3SamplingPolicy.from_scheduler(scheduler)

    def test_t_list_on_unipc_raises(self):
        """UniPC cannot honor the distilled policy (no seeded step noise,
        no baked-in guidance) even though its set_timesteps accepts sigmas."""
        config = {
            **UNIPC_SCHEDULER_CONFIG,
            "fixed_step_sampler_config": DISTILLED_SCHEDULER_CONFIG["fixed_step_sampler_config"],
        }
        scheduler = UniPCMultistepScheduler.from_config(config)

        with pytest.raises(ValueError, match="Unsupported Cosmos3 sampling recipe"):
            Cosmos3SamplingPolicy.from_scheduler(scheduler)

    def test_unipc_with_declared_requirement_but_no_sigmas_raises(self):
        config = {**UNIPC_SCHEDULER_CONFIG, "fixed_step_requires_explicit_sigmas": True}
        scheduler = UniPCMultistepScheduler.from_config(config)

        with pytest.raises(ValueError, match="fixed_step_requires_explicit_sigmas"):
            Cosmos3SamplingPolicy.from_scheduler(scheduler)

    def test_unipc_with_flag_and_sigmas_gets_unsupported_error(self):
        config = {
            **UNIPC_SCHEDULER_CONFIG,
            "fixed_step_requires_explicit_sigmas": True,
            "fixed_step_sampler_config": DISTILLED_SCHEDULER_CONFIG["fixed_step_sampler_config"],
        }
        scheduler = UniPCMultistepScheduler.from_config(config)

        with pytest.raises(ValueError, match="Unsupported Cosmos3 sampling recipe"):
            Cosmos3SamplingPolicy.from_scheduler(scheduler)

    def test_flow_match_without_sigmas_raises(self):
        config = {k: v for k, v in DISTILLED_SCHEDULER_CONFIG.items()}
        config.pop("fixed_step_sampler_config")
        config.pop("fixed_step_requires_explicit_sigmas")
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)

        with pytest.raises(ValueError, match="Unsupported Cosmos3 sampling recipe"):
            Cosmos3SamplingPolicy.from_scheduler(scheduler)


class TestValidateRequest:
    @pytest.mark.parametrize("steps", [None, 4])
    @pytest.mark.parametrize("guidance", [None, 1, 1.0])
    def test_valid_values_pass(self, steps, guidance):
        _distilled_policy().validate_request(steps, guidance)

    @pytest.mark.parametrize("bad_steps", [1, 10, 35, 50, 100])
    def test_explicit_steps_mismatch_raises(self, bad_steps):
        with pytest.raises(ValueError, match="distilled"):
            _distilled_policy().validate_request(bad_steps, None)

    @pytest.mark.parametrize("bad_guidance", [0.5, 3.5, 6.0, 7.0])
    def test_explicit_guidance_mismatch_raises(self, bad_guidance):
        with pytest.raises(ValueError, match="distilled"):
            _distilled_policy().validate_request(None, bad_guidance)

    def test_base_policy_accepts_anything(self):
        _base_policy().validate_request(17, 5.5)
        _base_policy().validate_request(None, None)


class TestFlowShift:
    def test_unipc_rebuilds_on_change(self):
        policy = _base_policy()
        scheduler = _base_scheduler()

        rebuilt = policy.set_flow_shift(scheduler, 3.0)
        assert rebuilt is not scheduler
        assert isinstance(rebuilt, UniPCMultistepScheduler)
        assert float(rebuilt.config.flow_shift) == 3.0

    def test_current_shift_read_from_scheduler_config(self):
        """No separate shift-tracking state: a second call with the same target
        on the rebuilt instance is a no-op; restoring rebuilds again."""
        policy = _base_policy()
        rebuilt = policy.set_flow_shift(_base_scheduler(), 3.0)
        assert policy.set_flow_shift(rebuilt, 3.0) is rebuilt

        restored = policy.set_flow_shift(rebuilt, 1.0)
        assert restored is not rebuilt
        assert float(restored.config.flow_shift) == 1.0

    def test_distilled_is_structural_noop(self):
        policy = _distilled_policy()
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_SCHEDULER_CONFIG)
        assert policy.set_flow_shift(scheduler, 3.0) is scheduler


class TestSetTimesteps:
    def test_distilled_programs_fixed_sigmas(self):
        policy = _distilled_policy()
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_SCHEDULER_CONFIG)
        policy.set_timesteps(scheduler, num_inference_steps=4, device="cpu")
        expected = [s * 1000.0 for s in DISTILLED_SIGMAS]
        assert torch.allclose(scheduler.timesteps.float(), torch.tensor(expected), atol=1e-3)

    def test_base_programs_step_count(self):
        policy = _base_policy()
        scheduler = _base_scheduler()
        policy.set_timesteps(scheduler, num_inference_steps=7, device="cpu")
        assert len(scheduler.timesteps) == 7


class TestStochasticStepDeterminism:
    """The seeded generator must fully determine the SDE noise trajectory."""

    def _run_steps(self, seed):
        policy = _distilled_policy()
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(DISTILLED_SCHEDULER_CONFIG)
        policy.set_timesteps(scheduler, num_inference_steps=4, device="cpu")
        generator = torch.Generator().manual_seed(seed)
        kwargs = policy.scheduler_step_kwargs(generator)

        latents = torch.zeros(1, 4, 1, 2, 2)
        velocity = torch.full_like(latents, 0.5)
        for t in scheduler.timesteps:
            latents = scheduler.step(velocity, t, latents, return_dict=False, **kwargs)[0]
        return latents

    def test_same_seed_reproduces_sde_trajectory(self):
        assert torch.equal(self._run_steps(seed=123), self._run_steps(seed=123))

    def test_different_seeds_diverge(self):
        assert not torch.equal(self._run_steps(seed=123), self._run_steps(seed=456))


class TestGenerationDefaults:
    def test_distilled_defaults_report_checkpoint_truth(self):
        params = _bare_pipeline(sampling=_distilled_policy()).default_generation_params
        assert params["num_inference_steps"] == 4
        assert params["guidance_scale"] == DISTILLED_GUIDANCE_SCALE
        assert params["height"] is None  # mode-dependent, resolved in infer()
        assert params["num_frames"] == COSMOS3_720P_PARAMS["num_frames"]

    def test_base_defaults_leave_mode_dependent_fields_unset(self):
        params = _bare_pipeline().default_generation_params
        for field in ("height", "width", "num_inference_steps", "guidance_scale"):
            assert params[field] is None
        assert params["num_frames"] == COSMOS3_720P_PARAMS["num_frames"]
        assert params["max_sequence_length"] == COSMOS3_720P_PARAMS["max_sequence_length"]


class TestInferModeResolution:
    def _captured_forward_kwargs(self, pipeline, req):
        captured = {}
        pipeline.forward = lambda **kwargs: captured.update(kwargs)
        pipeline.infer(req)
        return captured

    def test_video_unset_resolves_to_video_table(self):
        got = self._captured_forward_kwargs(_bare_pipeline(), _fake_request("video"))
        assert got["height"] == COSMOS3_720P_PARAMS["height"]
        assert got["width"] == COSMOS3_720P_PARAMS["width"]
        assert got["num_inference_steps"] == COSMOS3_720P_PARAMS["num_inference_steps"]
        assert got["guidance_scale"] == COSMOS3_720P_PARAMS["guidance_scale"]

    def test_t2i_unset_resolves_to_t2i_table(self):
        got = self._captured_forward_kwargs(_bare_pipeline(), _fake_request("image"))
        assert got["height"] == COSMOS3_T2I_PARAMS["height"]
        assert got["width"] == COSMOS3_T2I_PARAMS["width"]
        assert got["num_inference_steps"] == COSMOS3_T2I_PARAMS["num_inference_steps"]
        assert got["guidance_scale"] == COSMOS3_T2I_PARAMS["guidance_scale"]

    def test_explicit_values_pass_through(self):
        req = _fake_request("image", height=512, num_inference_steps=20)
        got = self._captured_forward_kwargs(_bare_pipeline(), req)
        assert got["height"] == 512
        assert got["num_inference_steps"] == 20
        assert got["width"] == COSMOS3_T2I_PARAMS["width"]

    def test_distilled_merged_defaults_pass_through(self):
        req = _fake_request("image", num_inference_steps=4, guidance_scale=1.0)
        got = self._captured_forward_kwargs(_bare_pipeline(sampling=_distilled_policy()), req)
        assert got["num_inference_steps"] == 4
        assert got["guidance_scale"] == DISTILLED_GUIDANCE_SCALE
        assert got["height"] == COSMOS3_T2I_PARAMS["height"]


class TestPipelineSchedulerLoading:
    def test_distilled_checkpoint_loads_flow_match(self, tmp_path):
        _write_scheduler_config(tmp_path, DISTILLED_SCHEDULER_CONFIG)
        pipeline = _bare_pipeline()

        pipeline.load_standard_components(
            str(tmp_path), torch.device("cpu"), skip_components=SKIP_NON_SCHEDULER
        )

        assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
        assert pipeline.sampling.is_distilled
        assert pipeline.sampling.fixed_sigmas == DISTILLED_SIGMAS

    def test_base_checkpoint_loads_unipc(self, tmp_path):
        _write_scheduler_config(tmp_path, UNIPC_SCHEDULER_CONFIG)
        pipeline = _bare_pipeline()

        pipeline.load_standard_components(
            str(tmp_path), torch.device("cpu"), skip_components=SKIP_NON_SCHEDULER
        )

        assert isinstance(pipeline.scheduler, UniPCMultistepScheduler)
        assert not pipeline.sampling.is_distilled
        assert pipeline.sampling.checkpoint_flow_shift == 1.0

    def test_audio_scheduler_is_separate_same_class_instance(self, tmp_path):
        _write_scheduler_config(tmp_path, DISTILLED_SCHEDULER_CONFIG)
        pipeline = _bare_pipeline(audio_gen=True)

        pipeline.load_standard_components(
            str(tmp_path), torch.device("cpu"), skip_components=SKIP_NON_SCHEDULER
        )

        assert isinstance(pipeline.audio_scheduler, FlowMatchEulerDiscreteScheduler)
        assert pipeline.audio_scheduler is not pipeline.scheduler


class TestWarmupAndForwardValidation:
    def test_warmup_steps_follow_distilled_schedule(self):
        assert _bare_pipeline(sampling=_distilled_policy()).default_warmup_steps == 4

    def test_warmup_steps_base_default(self):
        assert _bare_pipeline().default_warmup_steps == 2  # BasePipeline default

    @pytest.mark.parametrize(
        "policy_factory, expected_guidance",
        [(_distilled_policy, DISTILLED_GUIDANCE_SCALE), (_base_policy, 6.0)],
    )
    def test_warmup_guidance_uses_pipeline_defaults(self, policy_factory, expected_guidance):
        pipeline = _bare_pipeline(sampling=policy_factory())
        captured = {}
        pipeline.forward = lambda **kwargs: captured.update(kwargs)

        pipeline._run_warmup(height=720, width=1280, num_frames=9, steps=4)

        assert captured["guidance_scale"] == expected_guidance

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"num_inference_steps": 10, "guidance_scale": 1.0},
            {"num_inference_steps": 4, "guidance_scale": 3.5},
        ],
    )
    def test_forward_rejects_explicit_mismatch(self, bad_kwargs):
        pipeline = _bare_pipeline(sampling=_distilled_policy())
        with pytest.raises(ValueError, match="distilled"):
            pipeline.forward(prompt="x", seed=0, use_guardrails=False, **bad_kwargs)

    @pytest.mark.parametrize("bad_output_type", ["imgae", "png", "", "both"])
    def test_invalid_output_type_raises(self, bad_output_type):
        pipeline = _bare_pipeline()
        with pytest.raises(ValueError, match="output_type"):
            pipeline.forward(prompt="x", seed=0, use_guardrails=False, output_type=bad_output_type)


class _AdditiveScheduler:
    """step(v, t, x) = x + v, so final latents are the exact sum of all
    predictions. The strict signature also pins that the loop passes no silent
    extra step kwargs unless the caller supplies them."""

    def __init__(self, timesteps):
        self.timesteps = timesteps

    def step(self, model_output, timestep, sample, return_dict=False):
        assert return_dict is False
        return (sample + model_output,)


class _GeneratorRecordingScheduler(_AdditiveScheduler):
    def __init__(self, timesteps):
        super().__init__(timesteps)
        self.generators = []

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        self.generators.append(generator)
        return super().step(model_output, timestep, sample, return_dict)


def _denoise_ready_pipeline() -> Cosmos3OmniMoTPipeline:
    return _bare_pipeline(
        pipeline_config=SimpleNamespace(visual_gen_mapping=None),
        cache_accelerator=None,
        _predenoise_pending=False,
        _postdenoise_pending=False,
        _is_warmup=False,
        _profile_range=None,
    )


class TestDistilledDenoiseLoop:
    POS_IDS = torch.arange(8).unsqueeze(0)
    NEG_IDS = torch.arange(8).unsqueeze(0) + 100
    POS_MASK = torch.ones(1, 8, dtype=torch.long)
    NEG_MASK = torch.zeros(1, 8, dtype=torch.long)

    def _run(self, scheduler=None, scheduler_step_kwargs=None):
        pipeline = _denoise_ready_pipeline()
        timesteps = torch.tensor([s * 1000.0 for s in DISTILLED_SIGMAS])
        scheduler = scheduler if scheduler is not None else _AdditiveScheduler(timesteps)
        calls = []

        def forward_fn(latent_input, extra_streams, step_index, timestep, embeds, extras):
            calls.append(
                {
                    "batch": latent_input.shape[0],
                    "timestep": float(timestep[0]),
                    "text_ids": extras["text_ids"],
                }
            )
            return torch.full_like(latent_input, 0.5)

        latents = torch.zeros(1, 4, 3, 2, 2)
        result = pipeline.denoise(
            latents=latents,
            scheduler=scheduler,
            prompt_embeds=self.POS_IDS,
            neg_prompt_embeds=self.NEG_IDS,
            guidance_scale=DISTILLED_GUIDANCE_SCALE,
            forward_fn=forward_fn,
            extra_cfg_tensors={
                "text_ids": (self.POS_IDS, self.NEG_IDS),
                "text_mask": (self.POS_MASK, self.NEG_MASK),
            },
            scheduler_step_kwargs=scheduler_step_kwargs,
        )
        return result, calls

    def test_guidance_one_single_forward_per_step(self):
        result, calls = self._run()

        assert len(calls) == 4, "one forward per distilled step, no CFG branch"
        assert all(c["batch"] == 1 for c in calls), "no CFG batch duplication"
        assert all(c["text_ids"] is self.POS_IDS for c in calls), "positive prompt only"
        assert [c["timestep"] for c in calls] == pytest.approx(
            [s * 1000.0 for s in DISTILLED_SIGMAS], abs=1e-3
        )
        assert torch.all(result == 2.0)  # 4 additive steps of +0.5 from 0

    def test_scheduler_step_kwargs_reach_every_step(self):
        generator = torch.Generator().manual_seed(7)
        timesteps = torch.tensor([s * 1000.0 for s in DISTILLED_SIGMAS])
        scheduler = _GeneratorRecordingScheduler(timesteps)

        result, calls = self._run(
            scheduler=scheduler,
            scheduler_step_kwargs=_distilled_policy().scheduler_step_kwargs(generator),
        )

        assert len(calls) == 4
        assert scheduler.generators == [generator] * 4
        assert torch.all(result == 2.0)


class TestRegistryDispatch:
    def test_model_index_class_name_dispatches(self, tmp_path):
        with open(tmp_path / "model_index.json", "w") as f:
            json.dump({"_class_name": "Cosmos3OmniPipeline"}, f)
        assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "Cosmos3OmniMoTPipeline"

    def test_hf_id_registered(self):
        entry = PIPELINE_REGISTRY["Cosmos3OmniMoTPipeline"]
        assert "nvidia/Cosmos3-Super-Text2Image-4Step" in entry.hf_ids
