# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import transformers
from test_modeling_multimodal import llm_models_root
from test_modeling_nemotron_h import extract_decode_logprobs

from tensorrt_llm import LLM
from tensorrt_llm._torch.models.modeling_multimodal_utils import get_multimodal_embeddings
from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    NanoV2VLInputProcessor,
    NanoV2VLVisionEncoder,
    NemotronH_Nano_VL_V2,
)
from tensorrt_llm._torch.models.modeling_parakeet import ProjectedParakeet
from tensorrt_llm.inputs import (
    AudioData,
    create_input_processor,
    create_input_processor_with_hash,
    default_multimodal_input_loader,
    prompt_inputs,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig
from tensorrt_llm.sampling_params import SamplingParams

MODEL_PATH = str(os.path.join(llm_models_root(), "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"))


@pytest.fixture(scope="function")
def data_dict_fixture():
    test_data_root = Path(os.path.join(llm_models_root(), "multimodals", "test_data"))
    data_dict = {
        "image": {
            "single": {
                "prompts": ["Describe the natural environment in the image."],
                "media": [str(test_data_root / "seashore.png")],
            },
            "multiple": {
                "prompts": ["Describe the difference between the two images."],
                "media": [
                    str(test_data_root / "seashore.png"),
                    str(test_data_root / "seashore.png"),
                ],
            },
        },
        "video": {
            "single": {
                "prompts": ["Describe the natural environment in the video."],
                "media": [str(test_data_root / "world.mp4")],
            },
            "multiple": {
                "prompts": ["Describe the difference between the two videos."],
                "media": [str(test_data_root / "world.mp4"), str(test_data_root / "world.mp4")],
            },
        },
    }
    return data_dict


@pytest.fixture(scope="function")
def nano_llm_model():
    """Fixture to create and cleanup the Nemotron nano VL model."""
    # Since nemotron-h series models are with both attention and mamba cache,
    # we use the top-level LLM to create the engine to make things simpler.
    nano_llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_batch_size=24,
        cuda_graph_config=CudaGraphConfig(),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, mamba_ssm_cache_dtype="float32"),
    )
    yield nano_llm

    # Cleanup.
    nano_llm.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.parametrize("condition", ["single", "multiple"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_nemotron_nano_v2_vl_input_processor(data_dict_fixture, condition, modality):
    # Create input processor for NemotronH_Nano_VL_V2.
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    input_processor = create_input_processor(MODEL_PATH, tokenizer=tokenizer)
    input_processor_with_hash = create_input_processor_with_hash(input_processor)

    # Reference results.
    reference_results = {
        "image": {
            "single": {
                "prompt_pattern": "<image>",
                "prompt_token_ids_length": 282,
                "pixel_values_shape": (1, 3, 512, 512),
                "num_patches": torch.tensor([1]),
            },
            "multiple": {
                "prompt_pattern": "<image 1><image> <image 2><image>",
                "prompt_token_ids_length": 550,
                "pixel_values_shape": (2, 3, 512, 512),
                "num_patches": torch.tensor([2]),
            },
        },
        "video": {
            "single": {
                "prompt_pattern": "<video>",
                "prompt_token_ids_length": 2202,
                "pixel_values_shape": (8, 3, 512, 512),
                "num_patches": torch.tensor([8]),
            },
            "multiple": {
                "prompt_pattern": "<video>\n<video>",
                "prompt_token_ids_length": 4381,
                "pixel_values_shape": (16, 3, 512, 512),
                "num_patches": torch.tensor([16]),
            },
        },
    }

    prompts = data_dict_fixture[modality][condition]["prompts"]
    media = data_dict_fixture[modality][condition]["media"]
    inputs = default_multimodal_input_loader(
        tokenizer=input_processor.tokenizer,
        model_dir=MODEL_PATH,
        model_type="NemotronH_Nano_VL_V2",
        modality=modality,
        prompts=prompts,
        media=media,
        image_data_format="pt",
        num_frames=8,
        device="cpu",
    )
    inputs = [prompt_inputs(i) for i in inputs]

    # Check special tokens in the prompt.
    final_prompt = inputs[0]["prompt"]
    prompt_pattern = reference_results[modality][condition]["prompt_pattern"]
    assert prompt_pattern in final_prompt, f"{final_prompt=} is not expected."

    prompt_token_ids, extra_processed_inputs = input_processor_with_hash(
        inputs[0], sampling_params=None
    )

    # Check the output of the input processor.
    prompt_token_ids_length = len(prompt_token_ids)
    pixel_values_shape = extra_processed_inputs["multimodal_data"][modality]["pixel_values"].shape
    num_patches = extra_processed_inputs["multimodal_data"][modality]["num_patches"]
    ref_prompt_token_ids_length = reference_results[modality][condition]["prompt_token_ids_length"]
    ref_pixel_values_shape = reference_results[modality][condition]["pixel_values_shape"]
    ref_num_patches = reference_results[modality][condition]["num_patches"]
    assert prompt_token_ids_length == ref_prompt_token_ids_length, (
        f"{prompt_token_ids_length=} is not expected."
    )
    assert pixel_values_shape == ref_pixel_values_shape, f"{pixel_values_shape=} is not expected."
    assert num_patches == ref_num_patches, f"{num_patches=} is not expected."


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("condition", ["single", "multiple"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_nemotron_nano_v2_vl_model_sanity_check(
    data_dict_fixture, nano_llm_model, condition, modality
):
    nano_llm = nano_llm_model
    sampling_params = SamplingParams(
        max_tokens=5,
        temperature=0.0,
        add_special_tokens=False,
        return_generation_logits=True,
    )

    # The reference data is generated by running the model with the same prompts and media.
    reference_data_dict = {
        "image": {
            "single": torch.tensor(
                [-8.9814e-01, -1.5258e-01, -7.6061e-04, -6.3735e-01, -3.1303e-02]
            ),
            "multiple": torch.tensor([-0.5807, -0.7470, -0.0100, -0.1203, -0.0551]),
        },
        "video": {
            "single": torch.tensor([-0.6011, -0.0327, -0.8864, -0.3832, -0.5950]),
            "multiple": torch.tensor([-0.4956, -0.8749, -0.0095, -1.2541, -0.9490]),
        },
    }
    prompts = data_dict_fixture[modality][condition]["prompts"]
    media = data_dict_fixture[modality][condition]["media"]
    inputs = default_multimodal_input_loader(
        tokenizer=nano_llm.tokenizer,
        model_dir=MODEL_PATH,
        model_type="NemotronH_Nano_VL_V2",
        modality=modality,
        prompts=prompts,
        media=media,
        image_data_format="pt",
        num_frames=8,
        device="cpu",
    )
    outputs = nano_llm.generate(
        inputs,
        sampling_params,
    )
    decode_logprobs = extract_decode_logprobs(outputs[0])
    ref_decode_logprobs = reference_data_dict[modality][condition]

    diff = torch.abs(decode_logprobs - ref_decode_logprobs)
    if diff.max() > 0.3:
        raise ValueError(
            f"Max difference is too large: {decode_logprobs=} | {ref_decode_logprobs=}"
        )
    else:
        print("Passed! Max difference is within tolerance")


class TestEncodeMultimodalDispatch:
    def _make_mock_model(self):
        """Create a minimal mock with the attributes `_encode_multimodal` needs."""
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
        model.sound_encoder = mock.MagicMock(spec=ProjectedParakeet)
        return model

    @staticmethod
    def _assert_compatible_with_chunked_prefill(multimodal_embeddings):
        # NOTE: `multimodal_embeddings` is expected to be the output of `_encode_multimodal`.
        # The below checks help verify that we can make use of `get_multimodal_embeddings` and its
        # caching feature. Otherwise, we would be re-encoding the items each chunk during chunked
        # prefill.
        assert len(multimodal_embeddings) == 1
        assert isinstance(multimodal_embeddings[0], torch.Tensor)

    def test_encode_multimodal_dispatches_audio(self):
        model = self._make_mock_model()
        fake_audio_embeds = torch.randn(10, 128)
        model._encode_audio = mock.MagicMock(return_value=fake_audio_embeds)

        mm_param = mock.MagicMock()
        mm_param.multimodal_data = {"modality_type": "audio", "audio": {}}

        # Call the real method on our mock
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [mm_param])

        model._encode_audio.assert_called_once_with(mm_param)
        model.vision_encoder.assert_not_called()
        self._assert_compatible_with_chunked_prefill(result)
        assert torch.equal(result[0], fake_audio_embeds)

    def test_encode_multimodal_dispatches_image(self):
        model = self._make_mock_model()
        fake_image_embeds = torch.randn(10, 128)
        model.vision_encoder.return_value = ([fake_image_embeds], [None])

        mm_param = mock.MagicMock()
        mm_param.multimodal_data = {
            "modality_type": "image",
            "image": {"pixel_values": torch.randn(1, 100, 768)},
        }

        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [mm_param])

        model.vision_encoder.assert_called_once_with([mm_param])
        self._assert_compatible_with_chunked_prefill(result)

    def test_encode_multimodal_unknown_modality_raises(self):
        """Unknown modality raises ValueError."""
        model = self._make_mock_model()
        mm_param = mock.MagicMock()
        mm_param.multimodal_data = {"modality_type": "smell"}

        with pytest.raises(ValueError, match="Unknown modality"):
            NemotronH_Nano_VL_V2._encode_multimodal(model, [mm_param])


class TestSoundPlaceholderInjection:
    """Test the sound placeholder token's injection points.

    They should follow the appropriate video, and be able to handle situations where videos with
    audio and without audio both exist.
    """

    VIDEO_TOKEN = "<video>"
    SOUND_TOKEN = "<so_embedding>"

    def _call_extract_audio_from_video(self, text_prompt, video_audios):
        """Call the real _extract_audio_from_video with a minimal mock model.

        _prepare_audio_features is stubbed to pass through the text unchanged
        so we can test only the placeholder injection logic.
        """
        model = MagicMock()
        model.video_context_token = self.VIDEO_TOKEN
        model._sound_context_token = self.SOUND_TOKEN
        model._audio_extractor = MagicMock()  # not None → passes early return
        model._prepare_audio_features = MagicMock(side_effect=lambda text, _: (text, {}))
        return NanoV2VLInputProcessor._extract_audio_from_video(model, text_prompt, video_audios)

    def _make_audio(self) -> AudioData:
        return AudioData(samples=np.zeros(16000), sample_rate=16000)

    def test_two_videos_only_second_has_audio(self):
        """When video1 is silent and video2 has audio, the sound placeholder
        should be injected after the *second* <video>, not the first."""
        text_prompt = f"Watch {self.VIDEO_TOKEN} and {self.VIDEO_TOKEN} carefully."

        video_audios = [
            None,  # video 1: no audio
            self._make_audio(),
        ]

        result, _ = self._call_extract_audio_from_video(text_prompt, video_audios)

        expected = f"Watch {self.VIDEO_TOKEN} and {self.VIDEO_TOKEN}{self.SOUND_TOKEN} carefully."
        assert result == expected, (
            f"Sound placeholder injected after the wrong <video> token.\n"
            f"  Expected: {expected!r}\n"
            f"  Got:      {result!r}"
        )

    def test_three_videos_first_and_third_have_audio(self):
        """Sound placeholders should follow the first and third <video> tokens."""
        text_prompt = f"A {self.VIDEO_TOKEN} B {self.VIDEO_TOKEN} C {self.VIDEO_TOKEN} D"

        video_audios = [
            self._make_audio(),
            None,  # video 2: no audio
            self._make_audio(),
        ]

        result, _ = self._call_extract_audio_from_video(text_prompt, video_audios)

        expected = (
            f"A {self.VIDEO_TOKEN}{self.SOUND_TOKEN} B "
            f"{self.VIDEO_TOKEN} C "
            f"{self.VIDEO_TOKEN}{self.SOUND_TOKEN} D"
        )
        assert result == expected, (
            f"Sound placeholders attached to wrong <video> positions.\n"
            f"  Expected: {expected!r}\n"
            f"  Got:      {result!r}"
        )


class TestEncodeMultimodalAudioOrder:
    """Test video / audio embedding order in multi-item scenarios."""

    def _make_video_param(self, *, has_audio: bool) -> MultimodalParams:
        """Create a MultimodalParams for a video, optionally with audio."""
        audio_data = None
        if has_audio:
            audio_data = {
                "input_audio_features": torch.randn(1, 80, 100),
                "feature_attention_mask": torch.ones(1, 100),
                "has_audio": [True],
                "audio_num_clips": torch.tensor([1]),
            }
        video_payload = {"audio": audio_data} if audio_data else {}
        param = MagicMock(spec=MultimodalParams)
        param.multimodal_data = {
            "modality_type": "video",
            "video": video_payload,
        }
        return param

    def test_multi_video_audio_interleaving(self):
        # Two videos with audio: each returned embedding should be
        # [vision_i, audio_i], not [vision_1 + vision_2, audio_1 + audio_2]."""
        hidden = 16
        v1_len, a1_len = 10, 3
        v2_len, a2_len = 8, 4

        v1_emb = torch.randn(v1_len, hidden)
        a1_emb = torch.randn(a1_len, hidden)
        v2_emb = torch.randn(v2_len, hidden)
        a2_emb = torch.randn(a2_len, hidden)

        # vision_encoder is called once per param; return the right tensor.
        vision_returns = [
            ([v1_emb], [list(range(v1_len))]),
            ([v2_emb], [list(range(v2_len))]),
        ]

        audio_returns = [(a1_emb, [a1_len]), (a2_emb, [a2_len])]

        params = [
            self._make_video_param(has_audio=True),
            self._make_video_param(has_audio=True),
        ]

        # Build a minimal mock of NemotronH_Nano_VL_V2 with only the attributes `_encode_multimodal`
        # touches.
        model = MagicMock()
        model.vision_encoder = MagicMock(side_effect=vision_returns)
        model.sound_encoder = MagicMock()  # not None -> audio path taken
        model._encode_audio_data = MagicMock(side_effect=audio_returns)
        # Mock the interleaver to simply concatenate vision + audio, since this test only verifies
        # per-param dispatch, not interleaving math.
        model._interleave_video_audio_embeddings = MagicMock(
            side_effect=lambda v, a, *args, **kwargs: torch.cat([v, a], dim=0)
        )

        # Call the real method, bound to our mock.
        # `_encode_multimodal` now returns a single-element list whose tensor
        # concatenates per-request embeddings in order.
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, params)

        assert len(result) == 1, "Should return a single concatenated tensor"
        combined = result[0]

        # First video: [v1, a1]
        expected_0 = torch.cat([v1_emb, a1_emb], dim=0)
        assert torch.equal(combined[: v1_len + a1_len], expected_0), (
            "Video 1 slice should be [vision_1, audio_1]"
        )

        # Second video: [v2, a2]
        expected_1 = torch.cat([v2_emb, a2_emb], dim=0)
        assert torch.equal(combined[v1_len + a1_len :], expected_1), (
            "Video 2 slice should be [vision_2, audio_2]"
        )

    def test_video_without_audio_skips_audio_concat(self):
        """A video param with no audio should return vision-only embeddings."""
        hidden = 16
        v_len = 6
        v_emb = torch.randn(v_len, hidden)

        param = self._make_video_param(has_audio=False)

        model = MagicMock()
        model.vision_encoder = MagicMock(return_value=([v_emb], [list(range(v_len))]))
        model.sound_encoder = MagicMock()
        model._encode_audio_data = MagicMock()

        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [param])

        assert len(result) == 1
        assert torch.equal(result[0], v_emb), "Vision-only video should not have audio appended"
        model._encode_audio_data.assert_not_called()

    def test_no_audio_concat_when_sound_encoder_is_none(self):
        hidden = 16
        v_len = 5
        v_emb = torch.randn(v_len, hidden)

        param = self._make_video_param(has_audio=True)

        model = MagicMock()
        model.vision_encoder = MagicMock(return_value=([v_emb], [list(range(v_len))]))
        model.sound_encoder = None  # no audio support
        model._encode_audio_data = MagicMock()

        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [param])

        assert len(result) == 1
        assert torch.equal(result[0], v_emb)
        model._encode_audio_data.assert_not_called()


class TestInterleaveVideoAudioEmbeddings:
    """Directly test `_interleave_video_audio_embeddings` with synthetic data."""

    @staticmethod
    def _make_model(patch_size=14, downsample_ratio=0.5, temporal_patch_size=2):
        """Build a minimal mock whose vision_encoder has real geometry."""
        vision_enc = MagicMock()
        vision_enc.video_temporal_patch_size = temporal_patch_size
        vision_enc.patch_size = patch_size
        vision_enc.downsample_ratio = downsample_ratio
        vision_enc._video_tubelet_geometry = (
            lambda t, T, ih, iw: NanoV2VLVisionEncoder._video_tubelet_geometry(
                vision_enc, t, T, ih, iw
            )
        )
        model = MagicMock()
        model.vision_encoder = vision_enc
        return model

    def test_two_videos_both_with_audio(self):
        """Two videos of different sizes, each with audio -> interleaved [v1, a1, v2, a2]."""
        hidden = 8
        # Use geometry that gives deterministic token counts.
        # patch_size=14, downsample_ratio=0.5 -> wh = (ih // 14 * 0.5) * (iw // 14 * 0.5)
        # Video 1: ih=iw=28 -> wh = 1*1 = 1, T=2,t=2 -> 1 tubelet -> vision_count = 1
        # Video 2: ih=56, iw=84 -> wh = 2*3 = 6, T=2,t=2 -> 1 tubelet -> vision_count = 6
        model = self._make_model(patch_size=14, downsample_ratio=0.5, temporal_patch_size=2)
        video_sizes = [[2, 1, 28, 28], [2, 1, 56, 84]]  # [t, tiles, ih, iw]

        v1 = torch.randn(1, hidden)
        v2 = torch.randn(6, hidden)
        vision_emb = torch.cat([v1, v2], dim=0)

        a1 = torch.randn(3, hidden)
        a2 = torch.randn(5, hidden)
        audio_emb = torch.cat([a1, a2], dim=0)

        result = NemotronH_Nano_VL_V2._interleave_video_audio_embeddings(
            model,
            vision_emb=vision_emb,
            audio_emb=audio_emb,
            per_clip_audio_counts=[3, 5],
            has_audio=[True, True],
            audio_num_clips=torch.tensor([1, 1]),
            video_sizes=video_sizes,
            evs_num_tokens=None,
        )

        expected = torch.cat([v1, a1, v2, a2], dim=0)
        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_mixed_audio_presence(self):
        """Three videos of different sizes: first has audio, second has none, third has audio."""
        hidden = 4
        # patch_size=14, downsample_ratio=0.5, t=2, T=2 -> 1 tubelet, num_tiles=1
        # Video 1: ih=28, iw=84  -> wh = 1*3 = 3 vision tokens
        # Video 2: ih=28, iw=140 -> wh = 1*5 = 5 vision tokens
        # Video 3: ih=28, iw=196 -> wh = 1*7 = 7 vision tokens
        model = self._make_model(patch_size=14, downsample_ratio=0.5, temporal_patch_size=2)
        video_sizes = [[2, 1, 28, 84], [2, 1, 28, 140], [2, 1, 28, 196]]

        v1 = torch.randn(3, hidden)
        v2 = torch.randn(5, hidden)
        v3 = torch.randn(7, hidden)
        vision_emb = torch.cat([v1, v2, v3], dim=0)

        a1 = torch.randn(2, hidden)
        a3 = torch.randn(4, hidden)
        audio_emb = torch.cat([a1, a3], dim=0)

        result = NemotronH_Nano_VL_V2._interleave_video_audio_embeddings(
            model,
            vision_emb=vision_emb,
            audio_emb=audio_emb,
            per_clip_audio_counts=[2, 4],
            has_audio=[True, False, True],
            audio_num_clips=torch.tensor([1, 1]),
            video_sizes=video_sizes,
            evs_num_tokens=None,
        )

        # Expected: [v1, a1, v2, v3, a3]
        expected = torch.cat([v1, a1, v2, v3, a3], dim=0)
        assert result.shape == expected.shape
        assert torch.equal(result, expected)

    def test_multi_clip_audio(self):
        """Audio for one video is split across multiple clips."""
        hidden = 4
        model = self._make_model(patch_size=14, downsample_ratio=0.5, temporal_patch_size=2)
        video_sizes = [[2, 1, 28, 28]]

        v1 = torch.randn(1, hidden)
        # Two clips: 3 tokens + 2 tokens = 5 audio tokens total
        audio_emb = torch.randn(5, hidden)

        result = NemotronH_Nano_VL_V2._interleave_video_audio_embeddings(
            model,
            vision_emb=v1,
            audio_emb=audio_emb,
            per_clip_audio_counts=[3, 2],
            has_audio=[True],
            audio_num_clips=torch.tensor([2]),  # 2 clips for 1 video
            video_sizes=video_sizes,
            evs_num_tokens=None,
        )

        expected = torch.cat([v1, audio_emb], dim=0)
        assert result.shape == expected.shape
        assert torch.equal(result, expected)


class TestEncodeMultimodalContract:
    """Verify `_encode_multimodal` conforms to the contract expected by `get_multimodal_embeddings`.

    The key assumption is that the `encoder_forward_fn` passed to it returns something whose length
    is 1, and can be indexed by `[0]` to return a single `torch.Tensor`.
    """

    HIDDEN = 128

    def _make_mock_model(self):
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
        model.sound_encoder = mock.MagicMock(spec=ProjectedParakeet)
        return model

    def _make_mm_param(self, modality_type, **extra):
        # Mirror the shape produced by the input processor: a nested dict keyed
        # by `modality_type` holds per-modality side-channel data (e.g. audio
        # payload for video). Extra kwargs override / extend that nested dict.
        param = mock.MagicMock()
        nested = extra.pop(modality_type, {})
        param.multimodal_data = {
            "modality_type": modality_type,
            modality_type: nested,
            **extra,
        }
        return param

    def test_returns_list_with_a_single_element(self):
        model = self._make_mock_model()
        model.vision_encoder.return_value = ([torch.randn(5, self.HIDDEN)], [None])

        param = self._make_mm_param("image")
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [param])

        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_concatenated_tensor_for_multiple_multimodal_items(self):
        """Multiple multimodal items must be concatenated into a single tensor.

        `get_multimodal_embeddings` requires `len(embeddings) == 1` and splits by per-request token
        counts in order to cache the embeddings.
        """
        model = self._make_mock_model()
        emb_a = torch.randn(5, self.HIDDEN)
        emb_b = torch.randn(3, self.HIDDEN)
        model.vision_encoder.side_effect = [
            ([emb_a], [None]),
            ([emb_b], [None]),
        ]

        params = [self._make_mm_param("image"), self._make_mm_param("image")]
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, params)

        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)
        # Verify concatenation order is preserved.
        assert torch.equal(result[0][:5], emb_a)
        assert torch.equal(result[0][5:], emb_b)

    def test_mixed_modalities_still_single_tensor(self):
        """Image + audio requests produce one concatenated tensor."""
        model = self._make_mock_model()
        img_emb = torch.randn(5, self.HIDDEN)
        audio_emb = torch.randn(3, self.HIDDEN)
        model.vision_encoder.return_value = ([img_emb], [None])
        model._encode_audio = mock.MagicMock(return_value=audio_emb)

        params = [
            self._make_mm_param("image"),
            self._make_mm_param("audio", audio={}),
        ]
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, params)

        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)

    def test_empty_params_returns_empty_list(self):
        model = self._make_mock_model()
        result = NemotronH_Nano_VL_V2._encode_multimodal(model, [])
        assert result == []


class TestChunkedPrefillCaching:
    """Verify that `_encode_multimodal` output is compatible with `get_multimodal_embeddings`.

    Specifically, we want to test that the caching functionality is exercised and not skipped due
    to the return type not being compatible.

    The test structure for each modality:

    1. Build `MultimodalParams` with a real `MultimodalRuntimeData` (past_seen_token_num=0,
       simulating the first chunk).
    2. Call `get_multimodal_embeddings` with the real _encode_multimodal wired to mock sub-encoders
       -> encoder MUST be invoked.
    3. Verify embeddings were cached in `multimodal_data`.
    4. Call `get_multimodal_embeddings` again with the SAME params (simulating a second chunk) ->
       encoder must NOT be invoked.
    """

    HIDDEN = 128
    NUM_TOKENS = 10

    def _make_mock_model(self):
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
        model.sound_encoder = mock.MagicMock(spec=ProjectedParakeet)
        return model

    def _make_param_with_runtime(self, modality_type, num_tokens, **extra):
        """Build a real MultimodalParams with runtime data for caching."""
        # Single contiguous mm block of `num_tokens` tokens, none cached yet:
        # mask = all-True over the chunk, cumsum = [1, 2, ..., num_tokens].
        embed_mask_cumsum = torch.arange(1, num_tokens + 1, dtype=torch.int64)
        runtime = MultimodalRuntimeData(
            past_seen_token_num=0,
            chunk_end_pos=num_tokens,
            embed_mask_cumsum=embed_mask_cumsum,
        )
        # Mirror the nested shape produced by the input processor: a dict
        # keyed by `modality_type` holds per-modality side-channel data.
        nested = extra.pop(modality_type, {})
        return MultimodalParams(
            multimodal_data={
                "modality_type": modality_type,
                modality_type: nested,
                **extra,
            },
            multimodal_runtime=runtime,
        )

    def _make_encoder_fn(self, model):
        """Wrap `_encode_multimodal` as a callable for get_multimodal_embeddings."""

        def encoder_fn(params):
            return NemotronH_Nano_VL_V2._encode_multimodal(model, params)

        return encoder_fn

    @pytest.mark.parametrize("modality", ["image", "video"])
    def test_vision_encoder_not_called_on_second_chunk(self, modality):
        model = self._make_mock_model()
        fake_emb = torch.randn(self.NUM_TOKENS, self.HIDDEN)
        model.vision_encoder.return_value = ([fake_emb], [None])

        param = self._make_param_with_runtime(modality, self.NUM_TOKENS)
        encoder_fn = self._make_encoder_fn(model)

        # First call: encoder must run and cache the result.
        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert len(result) == 1
        assert result[0].shape == (self.NUM_TOKENS, self.HIDDEN)
        assert model.vision_encoder.call_count == 1

        # Embedding is now cached in multimodal_data.
        assert "multimodal_embedding" in param.multimodal_data

        # Second call: encoder must NOT run - embeddings come from cache.
        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert model.vision_encoder.call_count == 1, (
            "`vision_encoder` was called again on the second chunk. "
            "Caching is broken - `_encode_multimodal` likely violates the "
            "`get_multimodal_embeddings` return type contract."
        )
        assert len(result2) == 1
        assert torch.equal(result2[0], result[0])

    def test_audio_encoder_not_called_on_second_chunk(self):
        model = self._make_mock_model()
        fake_emb = torch.randn(self.NUM_TOKENS, self.HIDDEN)
        model._encode_audio = mock.MagicMock(return_value=fake_emb)

        param = self._make_param_with_runtime("audio", self.NUM_TOKENS, audio={})
        encoder_fn = self._make_encoder_fn(model)

        # First call.
        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert len(result) == 1
        assert model._encode_audio.call_count == 1

        # Second call - should use cache.
        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert model._encode_audio.call_count == 1, (
            "`_encode_audio` was called again on the second chunk. "
            "Caching is broken - `_encode_multimodal` likely violates the "
            "`get_multimodal_embeddings` return type contract."
        )
        assert torch.equal(result2[0], result[0])

    def test_multi_request_batch_caching(self):
        """Two image requests in one batch: both cached after one call."""
        model = self._make_mock_model()
        emb_a = torch.randn(5, self.HIDDEN)
        emb_b = torch.randn(3, self.HIDDEN)
        model.vision_encoder.side_effect = [
            ([emb_a], [None]),
            ([emb_b], [None]),
        ]

        param_a = self._make_param_with_runtime("image", 5)
        param_b = self._make_param_with_runtime("image", 3)
        encoder_fn = self._make_encoder_fn(model)

        # First call: encoder runs for both.
        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)
        assert model.vision_encoder.call_count == 2  # once per param

        # Both should be cached.
        assert "multimodal_embedding" in param_a.multimodal_data
        assert "multimodal_embedding" in param_b.multimodal_data

        # Second call: encoder must not run again.
        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert model.vision_encoder.call_count == 2, (
            "`vision_encoder` was called again on the second chunk. "
            "Caching is broken - `_encode_multimodal` likely violates the "
            "`get_multimodal_embeddings` return type contract."
        )
        assert torch.equal(result2[0], result[0])
