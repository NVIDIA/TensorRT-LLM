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
"""CPU tests for MiniMax-H3's `fl2va` keyframe conditioning.

These cover the three places where an image-to-video request differs from a
text-to-video one, each of which is silent when wrong: the keyframe reaches
the video VAE as a single-frame clip, the vision rows carry the video AdaLN
modality, and a follower keyframe is cover-cropped rather than stretched.
"""

import os
from io import BytesIO

os.environ["TLLM_DISABLE_MPI"] = "1"

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3Pipeline,
    fit_keyframe_to_canvas,
)

LATENT_CHANNELS = 24


def _gradient_image(width: int, height: int) -> Image.Image:
    """An image whose pixels encode their own coordinates, so crops are checkable."""
    x = torch.arange(width, dtype=torch.float32).view(1, -1).expand(height, width)
    y = torch.arange(height, dtype=torch.float32).view(-1, 1).expand(height, width)
    pixels = torch.stack([x % 256, y % 256, torch.zeros_like(x)], dim=-1)
    return Image.fromarray(pixels.to(torch.uint8).numpy(), mode="RGB")


class _RecordingVae:
    """A stand-in video VAE that records the pixel tensor it is handed."""

    def __init__(self):
        self.seen_shapes = []
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            latents_mean=[0.0] * LATENT_CHANNELS,
            latents_std=[1.0] * LATENT_CHANNELS,
        )

    def encode(self, pixels, return_dict=False):
        assert return_dict is False
        self.seen_shapes.append(tuple(pixels.shape))
        batch, _, num_frames, height, width = pixels.shape
        latents = torch.zeros(batch, LATENT_CHANNELS, num_frames, height // 16, width // 16)
        return (SimpleNamespace(sample=lambda generator=None: latents.clone()),)


class TestKeyframeVaeEncode:
    """`_encode_keyframes` has to present a still as a one-frame video."""

    def test_keyframe_is_encoded_as_single_frame_clip(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        vae = _RecordingVae()
        pipeline.vae = vae

        conditions = pipeline._encode_keyframes([_gradient_image(64, 32)])

        # (1, 3, 1, H, W): batch, RGB, a single frame, then the canvas. Repeating
        # the still across three frames would also decode, but conditions the
        # transformer on a different clip than the released model does.
        assert vae.seen_shapes == [(1, 3, 1, 32, 64)]
        assert len(conditions) == 1
        assert conditions[0].shape == (1, LATENT_CHANNELS, 1, 2, 4)

    def test_normalization_uses_imagenet_statistics(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        captured = {}

        class _CapturingVae(_RecordingVae):
            def encode(self, pixels, return_dict=False):
                captured["pixels"] = pixels.clone()
                return super().encode(pixels, return_dict=return_dict)

        pipeline.vae = _CapturingVae()
        white = Image.new("RGB", (64, 32), (255, 255, 255))

        pipeline._encode_keyframes([white])

        # A white pixel is (1 - mean) / std per channel.
        expected = torch.tensor(
            [
                (1.0 - 0.485) / 0.229,
                (1.0 - 0.456) / 0.224,
                (1.0 - 0.406) / 0.225,
            ]
        )
        torch.testing.assert_close(captured["pixels"][0, :, 0, 0, 0], expected)

    def test_keyframe_latents_are_normalized_by_vae_statistics(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        vae = _RecordingVae()
        vae.config = SimpleNamespace(
            latents_mean=[2.0] * LATENT_CHANNELS,
            latents_std=[4.0] * LATENT_CHANNELS,
        )
        pipeline.vae = vae

        conditions = pipeline._encode_keyframes([_gradient_image(64, 32)])

        # The stub samples all-zero latents, so (0 - 2) / 4 = -0.5.
        torch.testing.assert_close(conditions[0], torch.full_like(conditions[0], -0.5))


class TestKeyframeInputContract:
    def test_load_keyframe_accepts_bytes_and_single_item_list(self):
        image = Image.new("RGB", (8, 4), (12, 34, 56))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()

        decoded = MiniMaxH3Pipeline._load_keyframe([payload], "last")

        assert decoded.mode == "RGB"
        assert decoded.size == image.size
        assert decoded.getpixel((0, 0)) == image.getpixel((0, 0))

    def test_load_keyframe_rejects_multiple_keyframes(self):
        with pytest.raises(ValueError, match="single last keyframe"):
            MiniMaxH3Pipeline._load_keyframe(["a", "b"], "last")


class TestFrameRateContract:
    def test_fixed_frame_rate_accepts_model_rate(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.fps = 24

        pipeline._validate_frame_rate(24.0)

    def test_fixed_frame_rate_rejects_other_rates(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.fps = 24

        with pytest.raises(ValueError, match="fixed 24 fps"):
            pipeline._validate_frame_rate(30.0)


class TestPromptInputContract:
    def _pipeline(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.tokenizer = _StubTokenizer()
        pipeline.processor = SimpleNamespace(
            image_processor=_StubImageProcessor(),
            create_mm_token_type_ids=lambda batch: [[0] * len(batch[0])],
        )
        pipeline.text_encoder = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64)),
            dtype=torch.float32,
            model=lambda input_ids, **kwargs: SimpleNamespace(
                hidden_states=[torch.zeros(1, input_ids.shape[1], 8)] * 51
            ),
        )
        pipeline._conditioner_device = None
        pipeline._conditioner_offloaded = False
        pipeline.transformer = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
        return pipeline

    def test_single_prompt_list_is_unwrapped(self):
        text_embeds, _ = self._pipeline()._encode_prompt(["a cat"])

        assert text_embeds.shape[-1] == 8

    def test_multiple_prompts_are_rejected(self):
        with pytest.raises(ValueError, match="got 2 prompts"):
            self._pipeline()._encode_prompt(["a cat", "a dog"])


class _StubTokenizer:
    """Maps every word to a distinct id and names the vision specials."""

    _SPECIALS = {"<|vision_start|>": 1001, "<|image_pad|>": 1002, "<|vision_end|>": 1003}

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [hash(token) % 997 for token in text.split()]}

    def convert_tokens_to_ids(self, token):
        return self._SPECIALS[token]


class _StubImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors="pt"):
        assert return_tensors == "pt"
        # One 1x8x8 grid per image: 64 patches, 16 tokens after the 2x2 merge.
        grid = torch.tensor([[1, 8, 8]] * len(images))
        return {"pixel_values": torch.zeros(len(images) * 64, 3), "image_grid_thw": grid}


class TestConditionerSwap:
    """The single-card path: one encode call between two host<->device moves.

    Only one of the transformer and the ~62 GB conditioner fits on a <128 GB
    card, so the swap order is load-bearing -- the transformer has to leave
    before the conditioner arrives, and it has to come back even if the encode
    raises. Recorded against stubs so the ordering is checked without 124 GB.
    """

    @staticmethod
    def _pipeline(offloaded):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        moves = []
        pipeline._conditioner_offloaded = offloaded
        pipeline._conditioner_device = None
        pipeline.text_encoder = SimpleNamespace(
            to=lambda target, *_a, **_k: moves.append(("conditioner", str(target)))
        )
        pipeline.transformer = SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1, device="cuda:0")]),
            to=lambda target, *_a, **_k: moves.append(("transformer", str(target))),
        )
        return pipeline, moves

    def test_resident_conditioner_moves_nothing(self):
        pipeline, moves = self._pipeline(offloaded=False)

        with pipeline._conditioner_on_device(torch.device("cuda:0")):
            pass

        assert moves == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a GPU")
    def test_transformer_leaves_before_the_conditioner_arrives(self):
        pipeline, moves = self._pipeline(offloaded=True)

        with pipeline._conditioner_on_device(torch.device("cuda:0")):
            # Mid-swap: the transformer is on the host, the conditioner on card.
            assert moves == [("transformer", "cpu"), ("conditioner", "cuda:0")]

        # And back, so the next request finds the transformer where it was.
        assert moves == [
            ("transformer", "cpu"),
            ("conditioner", "cuda:0"),
            ("conditioner", "cpu"),
            ("transformer", "cuda:0"),
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a GPU")
    def test_the_transformer_returns_even_when_the_encode_raises(self):
        """A failed encode must not leave the transformer stranded on the host."""
        pipeline, moves = self._pipeline(offloaded=True)

        with pytest.raises(RuntimeError, match="encode blew up"):
            with pipeline._conditioner_on_device(torch.device("cuda:0")):
                raise RuntimeError("encode blew up")

        assert moves[-2:] == [("conditioner", "cpu"), ("transformer", "cuda:0")]


class TestCrossCardEncode:
    """With the conditioner on its own card, `_encode_prompt` has to bridge them."""

    @staticmethod
    def _pipeline(transformer_device, conditioner_device):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline.tokenizer = _StubTokenizer()
        pipeline.processor = SimpleNamespace(
            image_processor=_StubImageProcessor(),
            create_mm_token_type_ids=lambda batch: [[0] * len(batch[0])],
        )
        seen = {}

        def _model(input_ids, **kwargs):
            # Everything the conditioner is handed must already be on its card.
            seen["input_ids"] = input_ids.device
            seen["mm_token_type_ids"] = kwargs["mm_token_type_ids"].device
            seen["attention_mask"] = kwargs["attention_mask"].device
            for name in ("pixel_values", "image_grid_thw"):
                if name in kwargs:
                    seen[name] = kwargs[name].device
            hidden = torch.zeros(1, input_ids.shape[1], 8, device=input_ids.device)
            return SimpleNamespace(hidden_states=[hidden] * 51)

        pipeline.text_encoder = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64)),
            dtype=torch.float32,
            model=_model,
            to=lambda *_a, **_k: None,
        )
        pipeline._conditioner_offloaded = False
        pipeline._conditioner_device = conditioner_device
        pipeline.transformer = SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1, device=transformer_device)])
        )
        return pipeline, seen

    def test_single_card_keeps_everything_on_one_device(self):
        pipeline, seen = self._pipeline(torch.device("cpu"), None)

        embeds, _ = pipeline._encode_prompt("a cat")

        assert seen["input_ids"] == torch.device("cpu")
        assert embeds.device == torch.device("cpu")

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs a second GPU")
    def test_inputs_land_on_the_conditioners_card(self):
        pipeline, seen = self._pipeline(torch.device("cuda:0"), torch.device("cuda:1"))

        embeds, tags = pipeline._encode_prompt("a cat", [_gradient_image(64, 64)])

        # Inputs on card 1, embeddings handed back on card 0. A tensor left on
        # the wrong card raises inside the conditioner rather than degrading.
        for name, device in seen.items():
            assert device == torch.device("cuda:1"), f"{name} was on {device}"
        assert embeds.device == torch.device("cuda:0")
        assert tags.shape[0] == embeds.shape[1]

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs a second GPU")
    def test_text_only_request_also_crosses_back(self):
        pipeline, seen = self._pipeline(torch.device("cuda:0"), torch.device("cuda:1"))

        embeds, _ = pipeline._encode_prompt("a cat")

        assert seen["input_ids"] == torch.device("cuda:1")
        assert embeds.device == torch.device("cuda:0")


class TestConditionerDevice:
    """`conditioner_device` parks the conditioner on a second card."""

    def test_defaults_to_no_explicit_placement(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline._conditioner_device = None
        pipeline.transformer = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))

        # With no placement the conditioner follows the transformer.
        assert pipeline.conditioner_device == torch.device("cpu")

    def test_explicit_placement_overrides_the_transformer_card(self):
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        pipeline._conditioner_device = torch.device("cuda:1")
        pipeline.transformer = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))

        assert pipeline.conditioner_device == torch.device("cuda:1")

    @pytest.mark.parametrize("spec", ["cpu", "cuda"])
    def test_rejects_specs_that_are_not_a_specific_card(self, spec):
        with pytest.raises(ValueError):
            MiniMaxH3Pipeline._parse_conditioner_device(spec)

    def test_rejects_a_card_that_is_not_visible(self):
        visible = torch.cuda.device_count()
        with pytest.raises(ValueError, match="out of range"):
            MiniMaxH3Pipeline._parse_conditioner_device(f"cuda:{visible + 8}")


class TestKeyframePromptTags:
    """Vision rows have to carry the video tag, not the text tag."""

    @staticmethod
    def _pipeline():
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        tokenizer = _StubTokenizer()
        image_processor = _StubImageProcessor()
        pipeline.tokenizer = tokenizer
        pipeline.processor = SimpleNamespace(
            image_processor=image_processor,
            create_mm_token_type_ids=lambda batch: [[0] * len(batch[0])],
        )

        captured = {}

        def _model(input_ids, **kwargs):
            captured["input_ids"] = input_ids
            captured["kwargs"] = kwargs
            hidden = torch.zeros(1, input_ids.shape[1], 8)
            return SimpleNamespace(hidden_states=[hidden] * 51)

        pipeline.text_encoder = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64)),
            dtype=torch.float32,
            model=_model,
            to=lambda *_args, **_kwargs: None,
        )
        pipeline._conditioner_offloaded = False
        pipeline._conditioner_device = None
        # `self.device` reads the transformer's first parameter. A real module
        # cannot be attached to a `__new__`'d `nn.Module`, so stand in for it.
        pipeline.transformer = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
        return pipeline, captured

    def test_vision_rows_are_tagged_as_video(self):
        pipeline, _ = self._pipeline()

        _, tags = pipeline._encode_prompt("a cat", [_gradient_image(64, 64)])

        # The label "<Picture 1>: " is text, the 16 image tokens plus the two
        # vision delimiters are video: the transformer's per-row AdaLN picks a
        # modality off this tag, so a text-tagged vision row is conditioned by
        # the wrong embedding.
        video_rows = int((tags == MINIMAX_H3_VIDEO_TAG).sum())
        assert video_rows == 16 + 2
        assert tags[-2:].tolist() == [MINIMAX_H3_TEXT_TAG] * 2
        assert set(tags.tolist()) == {MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG}

    def test_prompt_without_keyframes_is_all_text(self):
        pipeline, captured = self._pipeline()

        embeds, tags = pipeline._encode_prompt("a cat")

        assert set(tags.tolist()) == {MINIMAX_H3_TEXT_TAG}
        assert "pixel_values" not in captured["kwargs"]
        assert embeds.shape[1] == tags.shape[0]

    def test_keyframe_pixels_reach_the_conditioner(self):
        pipeline, captured = self._pipeline()

        embeds, tags = pipeline._encode_prompt("a cat", [_gradient_image(64, 64)])

        assert captured["kwargs"]["image_grid_thw"].tolist() == [[1, 8, 8]]
        assert embeds.shape[1] == tags.shape[0] == captured["input_ids"].shape[1]

    def test_two_keyframes_are_labelled_in_order(self):
        pipeline, captured = self._pipeline()

        _, tags = pipeline._encode_prompt(
            "a cat", [_gradient_image(64, 64), _gradient_image(64, 64)]
        )

        assert int((tags == MINIMAX_H3_VIDEO_TAG).sum()) == 2 * (16 + 2)
        assert captured["kwargs"]["image_grid_thw"].tolist() == [[1, 8, 8]] * 2


class TestKeyframeCanvasFit:
    """The opening anchor is stretched, a follower is cover-cropped."""

    def test_exact_size_is_passed_through_untouched(self):
        keyframe = _gradient_image(128, 64)
        assert fit_keyframe_to_canvas(keyframe, 128, 64, stretch=True) is keyframe
        assert fit_keyframe_to_canvas(keyframe, 128, 64, stretch=False) is keyframe

    @pytest.mark.parametrize("stretch", [True, False])
    def test_output_always_matches_the_canvas(self, stretch):
        fitted = fit_keyframe_to_canvas(_gradient_image(100, 200), 128, 64, stretch=stretch)
        assert fitted.size == (128, 64)

    def test_follower_is_cover_cropped_about_the_center(self):
        # A 64x64 keyframe onto a 32x16 canvas: scale = max(32/64, 16/64) = 0.5,
        # so it resizes to 32x32 and crops 8 rows off the top and bottom.
        fitted = fit_keyframe_to_canvas(_gradient_image(64, 64), 32, 16, stretch=False)
        assert fitted.size == (32, 16)

        # The gradient's green channel carries the source y coordinate. Row 0 of
        # the canvas comes from source y = 16 (8 cropped rows at the 0.5 scale),
        # so the top row is mid-gradient rather than the source's y = 0.
        assert fitted.getpixel((0, 0))[1] == pytest.approx(16, abs=2)
        assert fitted.getpixel((0, 15))[1] == pytest.approx(46, abs=2)

    def test_stretched_anchor_keeps_the_whole_source(self):
        # Stretching maps the source origin onto the canvas origin, so the top
        # row stays near y = 0; the cover-crop of the same image starts at
        # y = 16 instead. (LANCZOS averages over its support, so the corner is
        # near-zero rather than exactly zero.)
        keyframe = _gradient_image(64, 64)
        stretched = fit_keyframe_to_canvas(keyframe, 32, 16, stretch=True)
        cropped = fit_keyframe_to_canvas(keyframe, 32, 16, stretch=False)

        assert stretched.getpixel((0, 0))[1] == pytest.approx(0, abs=4)
        assert cropped.getpixel((0, 0))[1] > stretched.getpixel((0, 0))[1] + 8
        # The stretch reaches the source's last row, the crop stops short of it.
        assert stretched.getpixel((0, 15))[1] == pytest.approx(60, abs=4)

    def test_upscaling_a_small_follower_still_covers_the_canvas(self):
        fitted = fit_keyframe_to_canvas(_gradient_image(10, 10), 64, 32, stretch=False)
        assert fitted.size == (64, 32)
