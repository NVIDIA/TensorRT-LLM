# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import MethodType
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
        trust_remote_code=True,
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


@pytest.mark.threadleak(enabled=False)
def test_nemotron_nano_v2_vl_image_batch_equivalence(nano_llm_model):
    """End-to-end equivalence check for cross-request image batching.

    Two distinct image+prompt requests are sent (a) together in one
    `generate` call so the engine batches them in a single forward step
    (and thus a single `_encode_multimodal` invocation with two
    multimodal_params), and (b) separately in two `generate` calls. With
    greedy decoding, the resulting token IDs must be identical and the
    logprobs must match within bf16 tolerance. This is intended to detect
    cross-request leakage or ordering bugs introduced by a future change
    that batches per-modality across requests inside the vision encoder.
    """
    nano_llm = nano_llm_model
    test_data_root = Path(os.path.join(llm_models_root(), "multimodals", "test_data"))
    prompts = [
        "Describe the natural environment in the image.",
        "Describe the object and the weather condition in the image.",
    ]
    media = [str(test_data_root / "seashore.png"), str(test_data_root / "inpaint.png")]

    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0.0,
        add_special_tokens=False,
        return_generation_logits=True,
    )

    def _build_inputs(prompts_subset, media_subset):
        return default_multimodal_input_loader(
            tokenizer=nano_llm.tokenizer,
            model_dir=MODEL_PATH,
            model_type="NemotronH_Nano_VL_V2",
            modality="image",
            prompts=prompts_subset,
            media=media_subset,
            image_data_format="pt",
            num_frames=8,
            device="cpu",
        )

    # Path A: both requests in one generate call -> engine batches them.
    batched_inputs = _build_inputs(prompts, media)
    batched_outputs = nano_llm.generate(batched_inputs, sampling_params)
    assert len(batched_outputs) == 2

    # Path B: each request in its own generate call.
    sep_outputs = []
    for p, m in zip(prompts, media):
        sep_inputs = _build_inputs([p], [m])
        sep_outputs.append(nano_llm.generate(sep_inputs, sampling_params)[0])

    for i, (b_out, s_out) in enumerate(zip(batched_outputs, sep_outputs)):
        b_token_ids = list(b_out.outputs[0].token_ids)
        s_token_ids = list(s_out.outputs[0].token_ids)
        assert b_token_ids == s_token_ids, (
            f"Request {i}: token_ids differ between batched and separate runs.\n"
            f"  batched : {b_token_ids}\n"
            f"  separate: {s_token_ids}"
        )

        b_logp = extract_decode_logprobs(b_out).cpu()
        s_logp = extract_decode_logprobs(s_out).cpu()
        max_diff = (b_logp - s_logp).abs().max().item()
        # bf16 reductions in attention / layernorm produce small but
        # nonzero diffs between batched-forward and per-request-forward
        # even for the same input. Token IDs (greedy) are the stronger
        # equivalence signal; logprobs use a looser tolerance, well
        # below the 0.3 threshold used by the sanity test.
        assert max_diff < 0.15, (
            f"Request {i}: logprob diff too large ({max_diff:.4f}).\n"
            f"  batched : {b_logp}\n"
            f"  separate: {s_logp}"
        )


@pytest.mark.threadleak(enabled=False)
def test_nemotron_nano_v2_vl_video_batch_equivalence(nano_llm_model):
    """End-to-end equivalence check for cross-request video batching.

    Mirror of `test_nemotron_nano_v2_vl_image_batch_equivalence` for
    video: two distinct video+prompt requests sent (a) together in one
    `generate` call (engine batches them, vision_encoder sees both
    multimodal_params at once) and (b) separately in two `generate`
    calls. With greedy decoding, token IDs must match and logprobs stay
    within bf16 tolerance.

    Intended to detect cross-video tubelet leakage if a future change
    batches the temporal-video path across requests inside the vision
    encoder.
    """
    nano_llm = nano_llm_model
    test_data_root = Path(os.path.join(llm_models_root(), "multimodals", "test_data"))
    prompts = [
        "Describe the natural environment in the video.",
        "Describe the scene in the video briefly.",
    ]
    media = [str(test_data_root / "world.mp4"), str(test_data_root / "world.mp4")]

    sampling_params = SamplingParams(
        max_tokens=16,
        temperature=0.0,
        add_special_tokens=False,
        return_generation_logits=True,
    )

    def _build_inputs(prompts_subset, media_subset):
        return default_multimodal_input_loader(
            tokenizer=nano_llm.tokenizer,
            model_dir=MODEL_PATH,
            model_type="NemotronH_Nano_VL_V2",
            modality="video",
            prompts=prompts_subset,
            media=media_subset,
            image_data_format="pt",
            num_frames=8,
            device="cpu",
        )

    batched_inputs = _build_inputs(prompts, media)
    batched_outputs = nano_llm.generate(batched_inputs, sampling_params)
    assert len(batched_outputs) == 2

    sep_outputs = []
    for p, m in zip(prompts, media):
        sep_inputs = _build_inputs([p], [m])
        sep_outputs.append(nano_llm.generate(sep_inputs, sampling_params)[0])

    for i, (b_out, s_out) in enumerate(zip(batched_outputs, sep_outputs)):
        b_token_ids = list(b_out.outputs[0].token_ids)
        s_token_ids = list(s_out.outputs[0].token_ids)
        assert b_token_ids == s_token_ids, (
            f"Request {i}: token_ids differ between batched and separate runs.\n"
            f"  batched : {b_token_ids}\n"
            f"  separate: {s_token_ids}"
        )

        b_logp = extract_decode_logprobs(b_out).cpu()
        s_logp = extract_decode_logprobs(s_out).cpu()
        max_diff = (b_logp - s_logp).abs().max().item()
        assert max_diff < 0.15, (
            f"Request {i}: logprob diff too large ({max_diff:.4f}).\n"
            f"  batched : {b_logp}\n"
            f"  separate: {s_logp}"
        )


def _wire_plan_based_encode(model, hidden_dim: int = 128) -> None:
    """Bind the real flat-item `_encode_multimodal` machinery onto a mock model.

    A `mock.MagicMock(spec=NemotronH_Nano_VL_V2)` auto-mocks the adapter /
    resolver methods the encode dispatch routes through. This rebinds the real
    methods (and fixes device/dtype/hidden-dim resolvers) so the production
    ghost-free per-item scatter path runs end-to-end against test-supplied
    `vision_encoder` / `_encode_audio` stubs.

    `_encode_multimodal` walks the flat-item assembly (`MixedModalityAssembly`
    + `_nano_extract_items`, both module-level) and dispatches each modality
    bucket through `_vision_encoder_adapter` / `_audio_encoder_adapter`. There is
    no post-encode interleave: video-embedded audio is hoisted to a first-class
    `(audio, k)` item, so the scatter produces the canonical layout directly.
    The single-item payload slicer (`_nano_slice_payload`) stays auto-mocked —
    these `_encode_multimodal` contract tests stub the encoders and assert only
    layout / call shape, not the per-modality slice tensors.
    """
    for name in (
        "_encode_multimodal",
        "_vision_encoder_adapter",
        "_audio_encoder_adapter",
    ):
        setattr(model, name, MethodType(getattr(NemotronH_Nano_VL_V2, name), model))
    model._build_single_modality_param = NemotronH_Nano_VL_V2._build_single_modality_param
    model._encode_multimodal_device = lambda: torch.device("cpu")
    model._encode_multimodal_dtype = lambda: torch.float32
    model._encode_multimodal_hidden_dim = lambda: hidden_dim


class TestEncodeMultimodalDispatch:
    def _make_mock_model(self):
        """Create a minimal mock with the attributes `_encode_multimodal` needs."""
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
        model.sound_encoder = mock.MagicMock(spec=ProjectedParakeet)
        _wire_plan_based_encode(model)
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
        # All audio is encoded via the batched `_encode_audio` helper.
        model._encode_audio = mock.MagicMock(return_value=[(fake_audio_embeds, [10])])

        mm_param = mock.MagicMock()
        audio_data = {"foo": "bar", "num_tokens": 10}
        mm_param.multimodal_data = {"modality_type": "audio", "audio": audio_data}
        # No runtime metadata: skip the assembly's `total_embeds_in_request`
        # cross-check (a bare MagicMock would auto-mock it to an int and trip the
        # check). This unit test pins audio dispatch routing, not the row count
        # reconciliation (covered by test_multimodal_encoding.py).
        mm_param.multimodal_runtime = None

        # Call the real method on our mock
        result = model._encode_multimodal([mm_param])

        model._encode_audio.assert_called_once_with([audio_data])
        model.vision_encoder.assert_not_called()
        self._assert_compatible_with_chunked_prefill(result)
        assert torch.equal(result[0], fake_audio_embeds)

    def test_encode_multimodal_dispatches_image(self):
        model = self._make_mock_model()
        fake_image_embeds = torch.randn(10, 128)
        model.vision_encoder.return_value = ([fake_image_embeds], [None])

        mm_param = mock.MagicMock()
        image_payload = {"pixel_values": torch.randn(1, 100, 768), "num_tokens": 10}
        mm_param.multimodal_data = {
            "modality_type": "image",
            "image": image_payload,
        }
        mm_param.multimodal_input = None
        mm_param.multimodal_runtime = None

        result = model._encode_multimodal([mm_param])

        # Vision encoder is invoked exactly once with a single-element list
        # (the per-item single-modality param view built by the adapter).
        model.vision_encoder.assert_called_once()
        (vision_call_args,) = model.vision_encoder.call_args.args
        assert len(vision_call_args) == 1
        assert vision_call_args[0].multimodal_data["modality_type"] == "image"
        assert vision_call_args[0].multimodal_data["image"] is image_payload
        self._assert_compatible_with_chunked_prefill(result)

    def test_encode_multimodal_unknown_modality_raises(self):
        """Unknown modality raises ValueError."""
        model = self._make_mock_model()
        mm_param = mock.MagicMock()
        mm_param.multimodal_data = {"modality_type": "smell", "smell": {"num_tokens": 1}}

        with pytest.raises(ValueError, match="Unknown modality"):
            model._encode_multimodal([mm_param])


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
    """`_encode_multimodal` row order for multi-video-with-audio requests.

    With the audio hoist, a video-with-audio request is no longer a nested
    `video_payload["audio"]` special case: the input processor promotes each
    video's audio to a first-class top-level `(audio, k)` item at its own prompt
    rank. The interleave / post-encode re-placement is deleted; the assembly's
    plain per-item scatter produces the canonical `[v0, a0, v1, a1]` layout
    directly. These tests exercise that ordering through the real
    `_encode_multimodal` against stubbed bucket encoders.
    """

    def _make_multi_video_audio_param(
        self, video_lengths: list, audio_lengths: list
    ) -> MultimodalParams:
        """Build one multi-modality param holding N hoisted video+audio items.

        Models the post-hoist shape: a list `modality_type`, a flat
        `multimodal_item_order` interleaving `(video,k),(audio,k)`, matching
        per-slot `multimodal_embedding_lengths`, and aggregate top-level
        `video` / `audio` payloads (sliced per item by the auto-mocked slicer).
        """
        order = []
        lengths = []
        for k, (v_len, a_len) in enumerate(zip(video_lengths, audio_lengths, strict=True)):
            order.append({"modality": "video", "index": k})
            lengths.append(v_len)
            order.append({"modality": "audio", "index": k})
            lengths.append(a_len)
        param = MagicMock(spec=MultimodalParams)
        param.multimodal_data = {
            "modality_type": ["video", "audio"],
            "video": {"pixel_values": "vids"},
            "audio": {"input_audio_features": "auds"},
            "multimodal_item_order": order,
            "multimodal_embedding_lengths": lengths,
        }
        param.multimodal_input = None
        param.multimodal_runtime = None
        return param

    def test_multi_video_audio_interleaving(self):
        # PRIMARY multi-video-with-audio case at the `_encode_multimodal` level:
        # two hoisted videos with audio in one request must scatter to
        # [v0, a0, v1, a1], NOT [v0, v1, a0, a1]. The video bucket is encoded in
        # one batched vision_encoder call (per-item embeddings in bucket order);
        # the audio bucket in one `_encode_audio` call; the per-item scatter
        # weaves them back by prompt rank.
        hidden = 16
        v0_len, a0_len = 10, 3
        v1_len, a1_len = 8, 4

        v0_emb = torch.randn(v0_len, hidden)
        a0_emb = torch.randn(a0_len, hidden)
        v1_emb = torch.randn(v1_len, hidden)
        a1_emb = torch.randn(a1_len, hidden)

        # vision_encoder is called once for the whole video bucket (both videos),
        # returning one embedding per item in bucket (= input) order.
        # Per-video EVS token counts are TENSORS (what the real vision_encoder /
        # apply_evs_per_video returns), so the multi-video concat in
        # `_vision_encoder_adapter` can `torch.cat` them.
        vision_return = ([v0_emb, v1_emb], [torch.arange(v0_len), torch.arange(v1_len)])
        # `_encode_audio` is called once for the whole audio bucket, returning one
        # (emb, per_clip_counts) per item in order.
        audio_returns = [(a0_emb, [a0_len]), (a1_emb, [a1_len])]

        param = self._make_multi_video_audio_param(
            video_lengths=[v0_len, v1_len], audio_lengths=[a0_len, a1_len]
        )

        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock(return_value=vision_return)
        model.sound_encoder = MagicMock()  # not None -> audio bucket encoded
        model._encode_audio = MagicMock(return_value=audio_returns)
        _wire_plan_based_encode(model, hidden_dim=hidden)

        result = model._encode_multimodal([param])

        assert len(result) == 1, "Should return a single concatenated tensor"
        combined = result[0]
        assert combined.shape == (v0_len + a0_len + v1_len + a1_len, hidden)

        # Canonical scatter layout is [v0, a0, v1, a1] (prompt-rank order), not
        # the bucket order [v0, v1, a0, a1].
        expected = torch.cat([v0_emb, a0_emb, v1_emb, a1_emb], dim=0)
        assert torch.equal(combined, expected), (
            "Hoisted video+audio items must scatter to [v0, a0, v1, a1]"
        )
        # One batched call per bucket (no per-item launches).
        model.vision_encoder.assert_called_once()
        model._encode_audio.assert_called_once()

    def test_pure_video_no_audio_is_vision_only(self):
        """A pure single-modality video request yields vision-only embeddings.

        Without an audio item there is nothing to hoist or scatter: the
        single-modality AGGREGATE path runs only the vision bucket and the
        audio encoder is never touched.
        """
        hidden = 16
        v_len = 6
        v_emb = torch.randn(v_len, hidden)

        param = MagicMock(spec=MultimodalParams)
        param.multimodal_data = {
            "modality_type": "video",
            "video": {"num_tokens": v_len},
        }
        param.multimodal_input = None
        param.multimodal_runtime = None

        model = MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = MagicMock(return_value=([v_emb], [list(range(v_len))]))
        model.sound_encoder = MagicMock()  # present, but no audio item to encode
        model._encode_audio = MagicMock()
        _wire_plan_based_encode(model, hidden_dim=hidden)

        result = model._encode_multimodal([param])

        assert len(result) == 1
        assert torch.equal(result[0], v_emb), "Vision-only video should not have audio appended"
        model._encode_audio.assert_not_called()


class TestEncodeAudio:
    """Numerical equivalence: batched audio vs per-input encoding.

    Uses a deterministic stub for `sound_encoder` so the test does not
    depend on a checkpoint that ships sound weights (the test fixture's
    12B-v2-VL has none). The stub mirrors what the real encoder
    contracts: maps ``[N, T_in, mel]`` to ``[N, T_out, hidden]`` with
    a fixed temporal subsampling factor and exposes
    ``encoder._get_subsampling_output_length``.
    """

    MEL_BINS = 4
    HIDDEN = 8
    SUBSAMPLE = 2  # 2 input timesteps -> 1 output timestep

    def _make_stub_sound_encoder(self):
        # The model invokes:
        #   sound_embeds = sound_encoder(features, mask)
        #   valid_output_lens = sound_encoder.encoder._get_subsampling_output_length(valid_input_lens)
        # so the stub is a small Linear over the time dim plus a method.
        proj = torch.nn.Linear(self.MEL_BINS, self.HIDDEN, bias=False)

        class _StubEncoder(torch.nn.Module):
            def __init__(self_inner):
                super().__init__()

            def _get_subsampling_output_length(self_inner, valid_input_lens):
                return torch.div(valid_input_lens, TestEncodeAudio.SUBSAMPLE, rounding_mode="floor")

        class _Stub(torch.nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.proj = proj
                self_inner.encoder = _StubEncoder()

            def forward(self_inner, features, mask):
                # features: [N, T, mel] -> mel-projected, then mean-pool every
                # SUBSAMPLE timesteps to mimic temporal subsampling.
                x = self_inner.proj(features)  # [N, T, hidden]
                T = x.shape[1]
                T_trim = (T // TestEncodeAudio.SUBSAMPLE) * TestEncodeAudio.SUBSAMPLE
                x = x[:, :T_trim].reshape(
                    x.shape[0],
                    T_trim // TestEncodeAudio.SUBSAMPLE,
                    TestEncodeAudio.SUBSAMPLE,
                    -1,
                )
                return x.mean(dim=2)  # [N, T_out, hidden]

        return _Stub()

    def _make_audio_data(self, num_clips, time_len, valid_lens):
        features = torch.randn(num_clips, time_len, self.MEL_BINS)
        mask = torch.zeros(num_clips, time_len, dtype=torch.long)
        for i, vl in enumerate(valid_lens):
            mask[i, :vl] = 1
        return {"input_audio_features": features, "feature_attention_mask": mask}

    def test_batched_matches_per_input(self):
        """Bucket output equals N singleton `_encode_audio` calls.

        Compares ``encode([a1, a2])`` against ``[encode([a1])[0], encode([a2])[0]]``
        — the contract is that the i-th batched result is identical to a
        per-input call for input i.
        """
        torch.manual_seed(0)
        stub = self._make_stub_sound_encoder()
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.sound_encoder = stub
        model.model_dtype = torch.float32

        # Two inputs with different time / clip counts.
        a1 = self._make_audio_data(num_clips=2, time_len=10, valid_lens=[10, 6])
        a2 = self._make_audio_data(num_clips=1, time_len=14, valid_lens=[12])

        per_input_results = [
            NemotronH_Nano_VL_V2._encode_audio(model, [a1])[0],
            NemotronH_Nano_VL_V2._encode_audio(model, [a2])[0],
        ]
        batched_results = NemotronH_Nano_VL_V2._encode_audio(model, [a1, a2])

        assert len(batched_results) == 2
        for (b_emb, b_counts), (s_emb, s_counts) in zip(batched_results, per_input_results):
            assert b_counts == s_counts
            assert torch.allclose(b_emb, s_emb, atol=1e-6, rtol=1e-6)

    def test_empty_input(self):
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        assert NemotronH_Nano_VL_V2._encode_audio(model, []) == []


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
        _wire_plan_based_encode(model, hidden_dim=self.HIDDEN)
        return model

    def _make_mm_param(self, modality_type, *, num_tokens: int = 0, **extra):
        # Mirror the shape produced by the input processor: a nested dict keyed
        # by `modality_type` holds per-modality side-channel data (e.g. audio
        # payload for video). Extra kwargs override / extend that nested dict.
        param = mock.MagicMock()
        nested = extra.pop(modality_type, {})
        nested.setdefault("num_tokens", num_tokens)
        param.multimodal_data = {
            "modality_type": modality_type,
            modality_type: nested,
            **extra,
        }
        param.multimodal_input = None
        param.multimodal_runtime = None
        return param

    def test_returns_list_with_a_single_element(self):
        model = self._make_mock_model()
        model.vision_encoder.return_value = ([torch.randn(5, self.HIDDEN)], [None])

        param = self._make_mm_param("image", num_tokens=5)
        result = model._encode_multimodal([param])

        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_concatenated_tensor_for_multiple_multimodal_items(self):
        """Multiple multimodal items must be concatenated into a single tensor.

        `get_multimodal_embeddings` requires `len(embeddings) == 1` and splits by per-request token
        counts in order to cache the embeddings.

        Image params are batched into a single `vision_encoder` call that
        returns a per-param embedding list.
        """
        model = self._make_mock_model()
        emb_a = torch.randn(5, self.HIDDEN)
        emb_b = torch.randn(3, self.HIDDEN)
        model.vision_encoder.return_value = ([emb_a, emb_b], [None, None])

        params = [
            self._make_mm_param("image", num_tokens=5),
            self._make_mm_param("image", num_tokens=3),
        ]
        result = model._encode_multimodal(params)

        # All image params go through a single batched vision_encoder call.
        assert model.vision_encoder.call_count == 1
        # The bucket call receives reconstructed single-modality param views,
        # not the original mock params — verify the call shape rather than identity.
        (vision_call_args,) = model.vision_encoder.call_args.args
        assert len(vision_call_args) == 2
        for view, original in zip(vision_call_args, params, strict=True):
            assert view.multimodal_data["modality_type"] == "image"
            assert view.multimodal_data["image"] is original.multimodal_data["image"]
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
        model._encode_audio = mock.MagicMock(return_value=[(audio_emb, [3])])

        params = [
            self._make_mm_param("image", num_tokens=5),
            self._make_mm_param("audio", num_tokens=3, audio={"num_tokens": 3}),
        ]
        result = model._encode_multimodal(params)

        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)

    def test_single_param_mixed_modalities_still_single_tensor(self):
        """One mixed param (image + video + audio) scatters in prompt-order rank.

        With explicit prompt order (audio, video, image) the per-item scatter
        must lay out the final tensor in MultimodalPromptOrder rank order, even
        though the encoder buckets (image / video) are each visited in their own
        bucket call. The default `modality_type`-order layout is already covered
        by `test_single_concatenated_tensor_for_multiple_multimodal_items` and
        `test_mixed_modalities_still_single_tensor`.
        """
        expected_order = ["audio", "video", "image"]
        model = self._make_mock_model()
        # Distinct token counts per modality so the output ordering is a real
        # discriminator: image=2, video=3, audio=4.
        embs = {
            "image": torch.randn(2, self.HIDDEN),
            "video": torch.randn(3, self.HIDDEN),
            "audio": torch.randn(4, self.HIDDEN),
        }

        # The vision encoder is called once per vision bucket (image and video).
        # The bucket visitation order follows the assembly's active-modality
        # order, not a fixed image/video order, so dispatch on the single-modality
        # view's `modality_type` rather than a positional `side_effect` list.
        def _vision_stub(bucket_params):
            modality = bucket_params[0].multimodal_data["modality_type"]
            return ([embs[modality]], [None])

        model.vision_encoder.side_effect = _vision_stub
        model._encode_audio = mock.MagicMock(return_value=[(embs["audio"], [4])])

        param = mock.MagicMock()
        param.multimodal_data = {
            "modality_type": ["image", "video", "audio"],
            "image": {"num_tokens": 2},
            "video": {"num_tokens": 3},
            "audio": {"num_tokens": 4},
            "multimodal_item_order": [
                {"modality": "audio", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
            "multimodal_embedding_lengths": [4, 3, 2],
        }
        param.multimodal_input = None
        param.multimodal_runtime = None

        result = model._encode_multimodal([param])

        assert len(result) == 1
        expected = torch.cat([embs[m] for m in expected_order], dim=0)
        assert torch.equal(result[0], expected)
        assert model.vision_encoder.call_count == 2
        model._encode_audio.assert_called_once()

    def test_empty_params_returns_empty_list(self):
        model = self._make_mock_model()
        result = model._encode_multimodal([])
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
        _wire_plan_based_encode(model, hidden_dim=self.HIDDEN)
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
        # Stash `num_tokens` on the payload so the plan extractor can
        # resolve the per-item token count without needing the production
        # fallback chain.
        nested = extra.pop(modality_type, {})
        nested.setdefault("num_tokens", num_tokens)
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
            return model._encode_multimodal(params)

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
        # Audio is encoded via the batched `_encode_audio` helper.
        model._encode_audio = mock.MagicMock(return_value=[(fake_emb, [self.NUM_TOKENS])])

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
        """Two image requests in one batch: both cached after a single batched call."""
        model = self._make_mock_model()
        emb_a = torch.randn(5, self.HIDDEN)
        emb_b = torch.randn(3, self.HIDDEN)
        # All image params are encoded in a single batched call.
        model.vision_encoder.return_value = ([emb_a, emb_b], [None, None])

        param_a = self._make_param_with_runtime("image", 5)
        param_b = self._make_param_with_runtime("image", 3)
        encoder_fn = self._make_encoder_fn(model)

        # First call: encoder runs once for the whole image batch.
        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)
        assert model.vision_encoder.call_count == 1, (
            "image params should be encoded in a single batched vision_encoder call"
        )

        # Both should be cached.
        assert "multimodal_embedding" in param_a.multimodal_data
        assert "multimodal_embedding" in param_b.multimodal_data

        # Second call: encoder must not run again.
        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert model.vision_encoder.call_count == 1, (
            "`vision_encoder` was called again on the second chunk. "
            "Caching is broken - `_encode_multimodal` likely violates the "
            "`get_multimodal_embeddings` return type contract."
        )
        assert torch.equal(result2[0], result[0])
