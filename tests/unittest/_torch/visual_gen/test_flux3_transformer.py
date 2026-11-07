# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests of the packed FLUX.3 transformer forward against an independent Diffusers-math one."""

import pytest
import torch
from flux3_diffusers_reference import Flux3Reference
from flux3_test_utils import assert_parity, diffusers_weights, model_config, tiny_config

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.flux3 import Flux3Transformer2DModel, bounds, pack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX.3 forward needs CUDA")

DEVICE = "cuda"
MODALITIES = ("image", "image_cond", "text")
# Per-sample (image, reference, text) token counts.
SHAPES = {
    "t2i": ((16, 20), (0, 0), (7, 12)),
    "i2i": ((16, 20), (12, 12), (7, 12)),
    "mixed_refs": ((16, 20), (0, 12), (7, 12)),
    "equal_segments": ((16, 16), (12, 12), (8, 8)),
}
TIMESTEPS, TARGETS = (0.7, 0.3), (0.5, 0.1)
FLOW_MAP = [pytest.param(True, id="klein"), pytest.param(False, id="base")]


def gpu_config(flow_map: bool, head_dim: int = 64) -> dict:
    return tiny_config(
        flow_map,
        attention_head_dim=head_dim,
        axes_dims_rope=[head_dim // 4] * 4,
        in_channels=16,
        cond_in_channels=24,
        joint_attention_dim=48,
        num_layers=1,
        num_single_layers=2,
    )


def build(
    flow_map: bool, backend: str = "VANILLA", head_dim: int = 64
) -> tuple[Flux3Transformer2DModel, Flux3Reference, dict]:
    config = gpu_config(flow_map, head_dim)
    weights = {
        name: value.to(DEVICE) for name, value in diffusers_weights(config, unit_scale=True).items()
    }
    model = Flux3Transformer2DModel(model_config(config, backend=backend)).to(DEVICE)
    model.load_weights(weights)
    model.post_load_weights()
    return model.eval(), Flux3Reference(weights, config), config


def positions(name: str, length: int) -> torch.Tensor:
    """(t, h, w, l) ids: text on l, images on a scaled h, w grid, references at l = -1."""
    index = torch.arange(length, device=DEVICE, dtype=torch.float32)
    ids = torch.zeros(length, 4, device=DEVICE)
    if name == "text":
        ids[:, 3] = index
    else:
        ids[:, 1], ids[:, 2] = (index // 4) * 0.5, (index % 4) * 0.5
        if name == "image_cond":
            ids[:, 3] = -1
    return ids


def make_samples(config: dict, *lengths: tuple[int, ...], seed: int = 0) -> list[dict]:
    """Random samples with ``lengths[modality][sample]`` tokens, in MODALITIES order."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    widths = {
        "image": config["in_channels"],
        "image_cond": config["cond_in_channels"],
        "text": config["joint_attention_dim"],
    }
    samples = []
    for counts in zip(*lengths):
        sample = {}
        for name, count in zip(MODALITIES, counts):
            if count:
                tokens = torch.randn(count, widths[name], generator=generator, device=DEVICE)
                sample[name] = (tokens.to(torch.bfloat16), positions(name, count))
        samples.append(sample)
    return samples


def packed_inputs(
    model: Flux3Transformer2DModel, samples: list[dict], timesteps: tuple[float, ...]
) -> dict:
    packed = pack(samples, model.packed_channels)
    packed.timesteps["image"].copy_(torch.tensor(timesteps))
    return packed.kwargs()


def target_kwargs(flow_map: bool, targets: tuple[float, ...]) -> dict:
    return {"target_timesteps": torch.tensor(targets, device=DEVICE)} if flow_map else {}


def reference_outputs(
    reference: Flux3Reference,
    samples: list[dict],
    timesteps: tuple[float, ...],
    targets: tuple[float, ...] | None,
) -> list[torch.Tensor]:
    outputs = []
    for index, sample in enumerate(samples):
        cond = sample.get("image_cond")
        outputs.append(
            reference(
                *sample["image"],
                *sample["text"],
                timestep=torch.tensor([timesteps[index]], device=DEVICE),
                image_cond=cond[0] if cond else None,
                image_cond_ids=cond[1] if cond else None,
                target_timestep=None
                if targets is None
                else torch.tensor([targets[index]], device=DEVICE),
            )
        )
    return outputs


@pytest.mark.parametrize(
    "backend, head_dim",
    [
        pytest.param("VANILLA", 32, id="vanilla-unfused_qk_norm"),
        pytest.param("VANILLA", 64, id="vanilla-fused_qk_norm"),
        pytest.param("TRTLLM", 64, id="trtllm"),
    ],
)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_matches_diffusers_reference(flow_map: bool, shape: str, backend: str, head_dim: int):
    """Packed per-sample timesteps, references and text lengths match one-sample Diffusers math."""
    model, reference, config = build(flow_map, backend, head_dim)
    samples = make_samples(config, *SHAPES[shape])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    actual = model(**kwargs, **target_kwargs(flow_map, TARGETS))["x_image"]
    expected = reference_outputs(reference, samples, TIMESTEPS, TARGETS if flow_map else None)
    for (start, end), sample_expected in zip(bounds(SHAPES[shape][0]), expected):
        assert_parity(actual[start:end], sample_expected, relative_l2=3e-2, cosine=0.9995)


@torch.inference_mode()
def test_packed_samples_are_isolated():
    model, _, config = build(flow_map=False)
    samples = make_samples(config, *SHAPES["mixed_refs"])
    kwargs = packed_inputs(model, samples, TIMESTEPS)
    packed = model(**kwargs)["x_image"]
    for index, (start, end) in enumerate(bounds(SHAPES["mixed_refs"][0])):
        single = packed_inputs(model, samples[index : index + 1], TIMESTEPS[index : index + 1])
        assert_parity(
            packed[start:end], model(**single)["x_image"], relative_l2=1e-2, cosine=0.9999
        )
    text_start = SHAPES["mixed_refs"][2][0]
    kwargs["x_text"][text_start:] += 1
    changed = model(**kwargs)["x_image"]
    image_end = SHAPES["mixed_refs"][0][0]
    torch.testing.assert_close(changed[:image_end], packed[:image_end])
    assert not torch.equal(changed[image_end:], packed[image_end:])


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_prepared_static_inputs_match_uncached(flow_map: bool):
    model, _, config = build(flow_map)
    kwargs = packed_inputs(model, make_samples(config, *SHAPES["mixed_refs"]), TIMESTEPS)
    prepared = model.prepare_static_inputs(**kwargs)
    assert "_flux3_joint_cos" in prepared
    assert ("_flux3_text_tokens" in prepared) != flow_map
    assert not any(key.endswith("_vec") for key in prepared)
    targets = target_kwargs(flow_map, TARGETS)
    torch.testing.assert_close(
        model(**kwargs, **prepared, **targets)["x_image"], model(**kwargs, **targets)["x_image"]
    )


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_conditioning_runs_once_per_sample(flow_map: bool):
    model, _, config = build(flow_map)
    kwargs = packed_inputs(model, make_samples(config, *SHAPES["i2i"]), TIMESTEPS)
    modules = [
        model.timestep_embedder.linear_1,
        model.norm_out.linear,
        *(modulation.linear for modulation in model.early_stream_modulations.values()),
        *(modulation.linear for modulation in model.single_stream_modulations.values()),
    ]
    if flow_map:
        modules.append(model.target_timestep_embedder.linear_1)
    rows = []
    handles = [
        module.register_forward_pre_hook(lambda module, args: rows.append(args[0].shape[0]))
        for module in modules
    ]
    try:
        model(**kwargs, **target_kwargs(flow_map, TARGETS))
    finally:
        for handle in handles:
            handle.remove()
    assert rows and max(rows) <= len(TIMESTEPS), rows


@pytest.mark.parametrize("flow_map", FLOW_MAP)
@torch.inference_mode()
def test_cuda_graph_replay_follows_layout_and_timesteps(flow_map: bool):
    """Swapped text lengths keep tensor shapes, so the layout key alone separates their graphs."""
    model, _, config = build(flow_map)
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    forward = runner.wrap(model.forward)
    try:
        for text_lengths, timesteps in (
            ((7, 12), (0.7, 0.3)),
            ((12, 7), (0.7, 0.3)),
            ((7, 12), (0.4, 0.2)),
        ):
            samples = make_samples(config, (16, 16), (0, 0), text_lengths, seed=len(runner.graphs))
            kwargs = packed_inputs(model, samples, timesteps)
            prepared = model.prepare_static_inputs(**kwargs)
            targets = target_kwargs(flow_map, tuple(t / 2 for t in timesteps))
            expected = model(**kwargs, **prepared, **targets)["x_image"]
            actual = forward(**kwargs, **prepared, **targets)["x_image"].clone()
            torch.testing.assert_close(actual, expected)
        assert len(runner.graphs) == 2
    finally:
        runner.clear()
