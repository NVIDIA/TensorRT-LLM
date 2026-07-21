# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FLUX.2 reference-image conditioning."""

import io
from collections.abc import Iterator
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from tensorrt_llm._torch.visual_gen.models.flux.pipeline_flux2 import (
    MAX_REFERENCE_IMAGES,
    Flux2Pipeline,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_reference_images_accepts_pil_path_and_bytes(tmp_path) -> None:
    pil_image = PIL.Image.new("L", (64, 64), color=128)
    image_path = tmp_path / "reference.png"
    pil_image.save(image_path)

    images = Flux2Pipeline._load_reference_images([pil_image, str(image_path), _png_bytes()])

    assert len(images) == 3
    assert all(image.mode == "RGB" for image in images)
    assert [image.size for image in images] == [(64, 64), (64, 64), (64, 64)]


@pytest.mark.parametrize("image", [[], [object()]])
def test_load_reference_images_rejects_invalid_inputs(image: list[object]) -> None:
    with pytest.raises(ValueError):
        Flux2Pipeline._load_reference_images(image)


def test_load_reference_images_rejects_excessive_references() -> None:
    images = [PIL.Image.new("RGB", (64, 64)) for _ in range(MAX_REFERENCE_IMAGES + 1)]

    with pytest.raises(ValueError, match="at most"):
        Flux2Pipeline._load_reference_images(images)


def test_preprocess_reference_images_caps_area_and_aligns_to_16() -> None:
    pipeline = Flux2Pipeline.__new__(Flux2Pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.image_processor = Flux2ImageProcessor(vae_scale_factor=16)
    images = [PIL.Image.new("RGB", (81, 65)), PIL.Image.new("RGB", (2048, 1024))]

    processed = pipeline._preprocess_reference_images(images)

    assert processed[0].shape == (1, 3, 64, 80)
    for tensor in processed:
        height, width = tensor.shape[-2:]
        assert height % 16 == 0
        assert width % 16 == 0
        assert height * width <= 1024 * 1024


def test_target_dimensions_default_to_first_processed_reference() -> None:
    condition_images = [torch.zeros(1, 3, 64, 80), torch.zeros(1, 3, 96, 112)]

    assert Flux2Pipeline._resolve_target_dimensions(None, None, condition_images) == (64, 80)
    assert Flux2Pipeline._resolve_target_dimensions(128, None, condition_images) == (128, 80)
    assert Flux2Pipeline._resolve_target_dimensions(None, None, None) == (1024, 1024)


@pytest.mark.parametrize("cache_backend", ["teacache", "cache_dit"])
@pytest.mark.parametrize("reference_count", [1, 3])
def test_reference_images_run_with_cache_acceleration(
    cache_backend: str,
    reference_count: int,
) -> None:
    pipeline = Flux2Pipeline.__new__(Flux2Pipeline)
    pipeline.pipeline_config = SimpleNamespace(cache_backend=cache_backend)
    pipeline._encode_prompt = lambda _prompt, _max_length: (
        torch.zeros(1, 2, 8),
        torch.zeros(2, 4),
    )
    pipeline._preprocess_reference_images = lambda images: [
        torch.zeros(1, 3, 16, 16) for _ in images
    ]
    pipeline._prepare_latents = lambda _batch_size, _height, _width, _generator: (
        torch.zeros(1, 4, 8),
        torch.zeros(4, 4),
    )
    pipeline._prepare_image_latents = lambda images, batch_size: (
        torch.zeros(batch_size, 4 * len(images), 8),
        torch.zeros(4 * len(images), 4),
    )

    class Transformer:
        guidance_embeds = False
        sharder = SimpleNamespace(is_active=False)

        def __init__(self) -> None:
            self.sequence_lengths: list[int] = []

        def parameters(self) -> Iterator[torch.Tensor]:
            return iter([torch.empty(0)])

        def __call__(self, hidden_states: torch.Tensor, **_kwargs) -> tuple[torch.Tensor]:
            self.sequence_lengths.append(hidden_states.shape[1])
            return (hidden_states + 1,)

    class Scheduler:
        config = SimpleNamespace(use_flow_sigmas=False)

        def set_timesteps(self, *_args, **_kwargs) -> None:
            self.timesteps = torch.tensor([1000.0])

        def set_begin_index(self, _index: int) -> None:
            return None

    transformer = Transformer()
    pipeline.transformer = transformer
    pipeline.scheduler = Scheduler()

    denoised_latents: list[torch.Tensor] = []

    def denoise(**kwargs) -> torch.Tensor:
        result = kwargs["forward_fn"](
            kwargs["latents"],
            {},
            0,
            kwargs["timesteps"][0],
            kwargs["prompt_embeds"],
            {},
        )
        denoised_latents.append(result)
        return result

    pipeline.denoise = denoise
    pipeline.decode_latents = lambda _latents, _decode_fn: torch.zeros(1, 16, 16, 3)

    references = [_png_bytes() for _ in range(reference_count)]
    result = pipeline.forward(
        prompt="edit this image",
        seed=0,
        image=references[0] if reference_count == 1 else references,
        num_inference_steps=1,
    )

    assert transformer.sequence_lengths == [4 + 4 * reference_count]
    assert denoised_latents[0].shape == (1, 4, 8)
    assert result.image.shape == (1, 16, 16, 3)


def test_prepare_image_ids_assigns_distinct_time_offsets() -> None:
    first = torch.zeros(1, 128, 2, 3)
    second = torch.zeros(1, 128, 1, 2)

    image_ids = Flux2Pipeline._prepare_image_ids([first, second])

    assert image_ids.shape == (8, 4)
    torch.testing.assert_close(image_ids[:6, 0], torch.full((6,), 10.0))
    torch.testing.assert_close(image_ids[6:, 0], torch.full((2,), 20.0))
    assert torch.count_nonzero(image_ids[:, 3]) == 0


def test_reference_sequence_length_rejects_nondivisible_parallel_shape() -> None:
    sharder = SimpleNamespace(is_active=True, size=4)

    with pytest.raises(ValueError, match="not divisible by the configured sequence-parallel"):
        Flux2Pipeline._validate_reference_sequence_length(
            target_seq_len=16,
            reference_seq_len=6,
            sharder=sharder,
        )


def test_reference_sequence_length_accepts_divisible_parallel_shape() -> None:
    sharder = SimpleNamespace(is_active=True, size=4)

    Flux2Pipeline._validate_reference_sequence_length(
        target_seq_len=16,
        reference_seq_len=8,
        sharder=sharder,
    )


def test_patchify_and_pack_reference_latents() -> None:
    latents = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6)
    pipeline = Flux2Pipeline.__new__(Flux2Pipeline)

    patchified = Flux2Pipeline._patchify_latents(latents)
    packed = pipeline._pack_latents(patchified)

    assert patchified.shape == (1, 8, 2, 3)
    assert packed.shape == (1, 6, 8)
    torch.testing.assert_close(packed[0, 0], latents[0, :, :2, :2].reshape(-1))


def test_encode_vae_image_uses_mode_patchify_and_batch_norm() -> None:
    source_latents = torch.arange(1 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 4, 6)

    class LatentDistribution:
        mode_called = False

        def mode(self) -> torch.Tensor:
            self.mode_called = True
            return source_latents

    latent_distribution = LatentDistribution()
    vae = SimpleNamespace(
        encode=lambda _image: SimpleNamespace(latent_dist=latent_distribution),
        bn=SimpleNamespace(running_mean=torch.zeros(8), running_var=torch.ones(8)),
        config=SimpleNamespace(batch_norm_eps=0.0),
    )
    pipeline = Flux2Pipeline.__new__(Flux2Pipeline)
    pipeline.vae = vae

    encoded = pipeline._encode_vae_image(torch.zeros(1, 3, 32, 48))

    assert latent_distribution.mode_called
    torch.testing.assert_close(encoded, Flux2Pipeline._patchify_latents(source_latents))
