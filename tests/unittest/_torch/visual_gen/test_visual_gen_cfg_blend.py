# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2: CFG branch combination when a guidance interval drops the scale to 1.0.

``u + 1.0 * (c - u)`` is the identity in exact arithmetic but not in BF16, so a step
outside the guidance interval must hand the scheduler the conditional prediction
itself. Both branches still run: this is a rounding contract, not CFG skipping.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline

# The released Cosmos3 Policy recipe: guidance only on the first, highest-noise step.
GUIDANCE_SCALE = 3.0
GUIDANCE_INTERVAL = (960.0, 1001.0)
TIMESTEPS = torch.tensor([999.0, 937.0, 833.0, 624.0])


def _pipeline() -> Cosmos3OmniMoTPipeline:
    """A pipeline instance carrying only what ``BasePipeline.denoise`` touches."""
    pipeline = object.__new__(Cosmos3OmniMoTPipeline)
    for key, value in {
        "_device": torch.device("cpu"),
        "pipeline_config": SimpleNamespace(visual_gen_mapping=None),
        "cache_accelerator": None,
        "_is_warmup": True,
    }.items():
        setattr(pipeline, key, value)
    return pipeline


class _RecordingScheduler:
    """Records every velocity handed to it and leaves the latents untouched, so
    each step's forward sees the same input and the recorded outputs are comparable."""

    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.seen = []

    def step(self, model_output, timestep, sample, return_dict=False, **kwargs):
        self.seen.append(model_output.clone())
        return (sample,)


def _branches(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    uncond = torch.randn(shape, generator=generator).bfloat16()
    cond = torch.randn(shape, generator=generator).bfloat16()
    return uncond, cond


def _run_denoise():
    """Run the real denoise loop over the Policy schedule with a video and an action
    stream, returning what each scheduler received plus the forward batch sizes."""
    pipeline = _pipeline()
    video_uncond, video_cond = _branches((1, 4, 3, 2, 2), seed=17)
    action_uncond, action_cond = _branches((1, 33, 64), seed=23)

    video_scheduler = _RecordingScheduler(TIMESTEPS)
    action_scheduler = _RecordingScheduler(TIMESTEPS)
    batches = []

    def forward_fn(latent_input, extra_streams, step_index, timestep, embeds, extras):
        batches.append(latent_input.shape[0])
        return (
            torch.cat([video_uncond, video_cond]),
            {"action": torch.cat([action_uncond, action_cond])},
        )

    pipeline.denoise(
        latents=torch.zeros(1, 4, 3, 2, 2, dtype=torch.bfloat16),
        scheduler=video_scheduler,
        prompt_embeds=torch.arange(8).unsqueeze(0),
        neg_prompt_embeds=torch.arange(8).unsqueeze(0) + 100,
        guidance_scale=GUIDANCE_SCALE,
        forward_fn=forward_fn,
        extra_streams={"action": (torch.zeros(1, 33, 64, dtype=torch.bfloat16), action_scheduler)},
        guidance_interval=GUIDANCE_INTERVAL,
    )
    return {
        "video": (video_uncond, video_cond, video_scheduler.seen),
        "action": (action_uncond, action_cond, action_scheduler.seen),
        "batches": batches,
    }


@pytest.mark.parametrize("stream", ["video", "action"])
def test_steps_outside_the_interval_pass_the_conditional_through(stream):
    result = _run_denoise()
    uncond, cond, seen = result[stream]

    assert len(seen) == len(TIMESTEPS)
    # Step 0 (t=999) is inside [960, 1001] and keeps full guidance.
    torch.testing.assert_close(seen[0], uncond + GUIDANCE_SCALE * (cond - uncond), atol=0, rtol=0)
    # Steps 1-3 fall outside it, so the scale is 1.0 and the conditional is returned as is.
    for step, velocity in enumerate(seen[1:], start=1):
        assert torch.equal(velocity, cond), f"step {step} re-derived the conditional velocity"


@pytest.mark.parametrize("stream", ["video", "action"])
def test_the_blend_at_scale_one_really_is_lossy(stream):
    """Negative control: without the short-circuit these operands do not round-trip,
    so the assertions above are sensitive rather than vacuously true."""
    result = _run_denoise()
    uncond, cond, _ = result[stream]

    naive = uncond + 1.0 * (cond - uncond)
    assert not torch.equal(naive, cond)


def test_both_branches_still_run_on_every_step():
    """The scale-1.0 path is a rounding contract, not CFG skipping: the batch stays
    doubled on all four steps, so kernel selection is unchanged."""
    assert _run_denoise()["batches"] == [2] * len(TIMESTEPS)


class TestCombineCfgBranches:
    """Direct checks on the shared helper, including the rescale interaction."""

    def test_scale_above_one_matches_the_reference_blend(self):
        pipeline = _pipeline()
        uncond, cond = _branches((4, 8), seed=5)
        torch.testing.assert_close(
            pipeline._combine_cfg_branches(uncond, cond, 3.0, 0.0),
            uncond + 3.0 * (cond - uncond),
            atol=0,
            rtol=0,
        )

    def test_rescale_is_skipped_when_no_guidance_was_applied(self):
        """Rescaling exists to tame over-exposure from guidance; at scale 1.0 there is
        none to tame, and applying it would reintroduce the rounding this avoids."""
        pipeline = _pipeline()
        uncond, cond = _branches((4, 8), seed=11)
        assert torch.equal(pipeline._combine_cfg_branches(uncond, cond, 1.0, 0.7), cond)

    def test_rescale_still_applies_when_guidance_is_active(self):
        pipeline = _pipeline()
        uncond, cond = _branches((4, 8), seed=13)
        blended = uncond + 3.0 * (cond - uncond)
        torch.testing.assert_close(
            pipeline._combine_cfg_branches(uncond, cond, 3.0, 0.7),
            pipeline._rescale_noise_cfg(blended, cond, 0.7),
            atol=0,
            rtol=0,
        )


class TestSkipUncondAtScaleOne:
    """``skip_uncond_at_scale_one`` drops the discarded branch instead of computing it.

    Opt-in, because halving the batch changes GEMM shapes and therefore kernel
    selection. The velocity it produces must still be the conditional one.
    """

    def _run(self, skip: bool):
        pipeline = _pipeline()
        video_uncond, video_cond = _branches((1, 4, 3, 2, 2), seed=31)
        action_uncond, action_cond = _branches((1, 33, 64), seed=37)
        video_scheduler = _RecordingScheduler(TIMESTEPS)
        action_scheduler = _RecordingScheduler(TIMESTEPS)
        seen = []

        def forward_fn(latent_input, extra_streams, step_index, timestep, embeds, extras):
            batch = latent_input.shape[0]
            seen.append(
                {
                    "batch": batch,
                    "embeds": embeds.shape[0],
                    "text_ids": extras["text_ids"].shape[0],
                }
            )
            # Mirror the CFG layout: first half unconditional, second half conditional.
            video = torch.cat([video_uncond, video_cond]) if batch == 2 else video_cond
            action = torch.cat([action_uncond, action_cond]) if batch == 2 else action_cond
            return video, {"action": action}

        pipeline.denoise(
            latents=torch.zeros(1, 4, 3, 2, 2, dtype=torch.bfloat16),
            scheduler=video_scheduler,
            prompt_embeds=torch.arange(8).unsqueeze(0),
            neg_prompt_embeds=torch.arange(8).unsqueeze(0) + 100,
            guidance_scale=GUIDANCE_SCALE,
            forward_fn=forward_fn,
            extra_cfg_tensors={
                "text_ids": (torch.arange(8).unsqueeze(0), torch.arange(8).unsqueeze(0) + 100)
            },
            extra_streams={
                "action": (torch.zeros(1, 33, 64, dtype=torch.bfloat16), action_scheduler)
            },
            guidance_interval=GUIDANCE_INTERVAL,
            skip_uncond_at_scale_one=skip,
        )
        return seen, video_scheduler.seen, action_scheduler.seen, video_cond, action_cond

    def test_batch_halves_only_outside_the_interval(self):
        seen, _, _, _, _ = self._run(skip=True)
        # t=999 is inside [960, 1001] and keeps both branches; the rest drop one.
        assert [s["batch"] for s in seen] == [2, 1, 1, 1]
        # The embeddings and extras must be sliced to match, or the model sees
        # the negative prompt while being asked for the conditional prediction.
        assert [s["embeds"] for s in seen] == [2, 1, 1, 1]
        assert [s["text_ids"] for s in seen] == [2, 1, 1, 1]

    def test_off_by_default_keeps_both_branches(self):
        seen, _, _, _, _ = self._run(skip=False)
        assert [s["batch"] for s in seen] == [2] * len(TIMESTEPS)

    def test_velocity_is_the_conditional_one_either_way(self):
        """Skipping must not change which velocity reaches the scheduler."""
        _, vid_skip, act_skip, video_cond, action_cond = self._run(skip=True)
        _, vid_keep, act_keep, _, _ = self._run(skip=False)
        for step in range(1, len(TIMESTEPS)):
            assert torch.equal(vid_skip[step], video_cond)
            assert torch.equal(act_skip[step], action_cond)
            assert torch.equal(vid_skip[step], vid_keep[step])
            assert torch.equal(act_skip[step], act_keep[step])
