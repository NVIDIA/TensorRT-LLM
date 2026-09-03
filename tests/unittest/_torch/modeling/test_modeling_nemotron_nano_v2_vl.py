# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import transformers
from test_modeling_multimodal import llm_models_root
from test_modeling_nemotron_h import extract_decode_logprobs

from tensorrt_llm import LLM
from tensorrt_llm._torch.models import modeling_nemotron_nano as nemotron_nano
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.modeling_multimodal_utils import get_multimodal_embeddings
from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    NanoV2VLInputProcessor,
    NanoV2VLMultimodalEncoder,
    NanoV2VLVisionEncoder,
    NemotronH_Nano_VL_V2,
    _get_vision_encoder_cuda_graph_config,
    _normalize_vision_weights,
)
from tensorrt_llm._torch.models.modeling_parakeet import ProjectedParakeet
from tensorrt_llm._torch.models.modeling_radio import split_fused_qkv
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_VISION_ENCODER_MAPPING
from tensorrt_llm.inputs import (
    AudioData,
    VideoData,
    create_input_processor,
    create_input_processor_with_hash,
    default_multimodal_input_loader,
    prompt_inputs,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    MultimodalConfig,
    MultimodalEncoderCudaGraphConfig,
)
from tensorrt_llm.sampling_params import SamplingParams

MODEL_PATH = str(os.path.join(llm_models_root(), "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"))


def _make_minimal_nano_model_config():
    llm_config = SimpleNamespace(vocab_size=128)
    pretrained_config = SimpleNamespace(
        llm_config=llm_config,
        torch_dtype=torch.bfloat16,
        img_context_token_id=20,
        video_context_token_id=21,
        sound_context_token_id=None,
        sound_config=None,
    )
    return SimpleNamespace(
        pretrained_config=pretrained_config,
        quant_config=SimpleNamespace(exclude_modules=None),
        quant_config_dict=None,
        video_pruning_rate=None,
    )


@pytest.mark.cpu_only
def test_nemotron_nano_registers_native_multimodal_epd_components():
    """Every arch served by the native Nano VL class advertises MM EPD support."""
    for arch in (
        "NemotronH_Nano_VL_V2",
        "NemotronH_Nano_Omni_Reasoning_V3",
        "NemotronH_Omni_Reasoning_V3",
    ):
        vision_encoder_cls, vlm_base_model = MODEL_CLASS_VISION_ENCODER_MAPPING[arch]
        assert vision_encoder_cls is NanoV2VLMultimodalEncoder
        assert vlm_base_model is None
    assert NanoV2VLInputProcessor.support_mm_disagg is True
    assert NemotronH_Nano_VL_V2.support_mm_disagg is True


def _assert_nano_video_handoff(handoff):
    """Shared assertions for the EPD video handoff: split runs stay grouped under one MM item."""
    assert handoff.prompt_token_ids == [101, 30, 20, 20, 31, 55, 30, 20, 20, 31, 102]
    assert handoff.multimodal_lengths == [8]
    assert handoff.multimodal_positions == [1]
    assert handoff.multimodal_embedding_lengths == [4]
    assert handoff.multimodal_item_run_cu_offsets == [0, 2]
    assert handoff.multimodal_run_positions == [1, 6]
    assert handoff.multimodal_run_lengths == [4, 4]
    assert handoff.special_token_offsets == [0, 3, 4, 7]


@pytest.mark.parametrize(
    "input_field, input_value, asserts_encode_not_called",
    [
        # Detokenized prompt text path: the tokenizer may encode the prompt.
        ("prompt", "Question <video> answer", False),
        # Tokenized handoff path: prompt text is absent, so encode must not be called.
        ("prompt_token_ids", [101, 98, 102], True),
    ],
    ids=["prompt", "prompt_token_ids"],
)
@pytest.mark.cpu_only
def test_nemotron_nano_epd_handoff_preserves_non_contiguous_video_runs(
    input_field, input_value, asserts_encode_not_called
):
    """Split video prompt runs stay grouped under one MM item, with or without prompt text."""
    processor = object.__new__(NanoV2VLInputProcessor)
    processor._config = SimpleNamespace(
        llm_config=SimpleNamespace(vocab_size=1000, hidden_size=16),
    )
    if asserts_encode_not_called:
        # In the tokenized path the tokenizer must never be invoked; a side effect
        # turns any accidental call into a hard failure.
        encode_mock = MagicMock(side_effect=AssertionError("tokenizer should not be called"))
    else:
        encode_mock = MagicMock(return_value=[101, 98, 102])
    processor._tokenizer = SimpleNamespace(encode=encode_mock)
    processor.img_context_token_id = 20
    processor._img_start_token_ids = [30]
    processor._img_end_token_ids = [31]
    processor._sound_context_token_id = None
    processor._sound_start_token_id = None
    processor._sound_end_token_id = None

    processor.get_num_tokens_per_video = MagicMock(return_value=8)
    processor.expand_prompt_token_ids_for_mm = MagicMock(
        return_value=([101, 30, 20, 20, 31, 55, 30, 20, 20, 31, 102], None)
    )

    video = VideoData(frames=[object()], metadata={}, audio=None)
    handoff = processor.build_disagg_prefill_multimodal_inputs(
        {
            input_field: input_value,
            "multi_modal_data": {"video": [video]},
        },
        [{"tensor_size": (4, 16)}],
    )

    if asserts_encode_not_called:
        processor._tokenizer.encode.assert_not_called()
        processor.expand_prompt_token_ids_for_mm.assert_called_once()
        assert processor.expand_prompt_token_ids_for_mm.call_args.args[0] == [101, 98, 102]
    _assert_nano_video_handoff(handoff)


@pytest.mark.parametrize(
    "env_value, expects_encoder",
    [
        # Normal worker: the vision encoder must be built and loaded for raw MM prefill.
        ("0", True),
        # MM E/P/D full-model worker: consumes attached embeddings, so the encoder is deferred.
        ("1", False),
    ],
    ids=["normal_worker", "mm_epd_worker"],
)
@pytest.mark.cpu_only
def test_nemotron_nano_multimodal_encoder_load_by_worker_role(env_value, expects_encoder):
    """Encoder load depends on whether the worker runs raw MM prefill or consumes embeddings."""
    fake_encoder = MagicMock()
    fake_encoder.eval.return_value = fake_encoder
    fake_encoder.to.return_value = fake_encoder
    vision_encoder_cls = MagicMock(return_value=fake_encoder)

    fake_mapper = MagicMock()
    mapper_cls = MagicMock(return_value=fake_mapper)

    model = SimpleNamespace(
        _mm_model_config=_make_minimal_nano_model_config(),
        vision_encoder=None,
        sound_encoder=None,
        llm=MagicMock(),
        model_config=SimpleNamespace(),
    )
    weights = {
        "vision_model.weight": torch.empty(0),
        "mlp1.weight": torch.empty(0),
        "language_model.weight": torch.empty(0),
    }

    with (
        mock.patch.dict(os.environ, {"TLLM_MULTIMODAL_DISAGGREGATED": env_value}),
        mock.patch.object(nemotron_nano, "NanoV2VLVisionEncoder", vision_encoder_cls),
        mock.patch.object(nemotron_nano, "NemotronHHfWeightMapper", mapper_cls),
    ):
        NemotronH_Nano_VL_V2.load_weights(model, weights)

    if expects_encoder:
        vision_encoder_cls.assert_called_once_with(model._mm_model_config)
        fake_encoder.load_weights.assert_called_once_with(weights)
    else:
        vision_encoder_cls.assert_not_called()


@pytest.mark.cpu_only
def test_nemotron_nano_rejects_evs_attached_video_embeddings():
    """EVS needs retained-token metadata that E/P attached embeddings do not carry."""
    model = SimpleNamespace(
        video_pruning_rate=0.5,
        _validate_evs_context_batch=MagicMock(),
    )
    attn_metadata = SimpleNamespace(num_contexts=1, num_generations=0)
    param = MultimodalParams(
        multimodal_data={
            "modality_type": "video",
            "multimodal_embedding": torch.zeros(1, 4),
        }
    )

    with pytest.raises(ValueError, match="EVS video pruning is not supported"):
        NemotronH_Nano_VL_V2.forward(
            model,
            attn_metadata,
            input_ids=torch.tensor([[20]], dtype=torch.long),
            multimodal_params=[param],
        )


def test_get_vision_encoder_cuda_graph_config():
    config = MultimodalEncoderCudaGraphConfig(buckets=[(1280, 1)])
    mm_config = MultimodalConfig(encoder_cuda_graph={"vision": config})

    assert _get_vision_encoder_cuda_graph_config(mm_config) is config


def test_get_vision_encoder_cuda_graph_config_rejects_unknown_modalities():
    mm_config = MultimodalConfig(
        encoder_cuda_graph={"audio": MultimodalEncoderCudaGraphConfig(buckets=[(1024, 1)])}
    )

    with pytest.raises(ValueError, match="Unsupported multimodal encoder CUDA graph modalities"):
        _get_vision_encoder_cuda_graph_config(mm_config)


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
@pytest.mark.cpu_only
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
    (and thus a single `_encode_image_group` invocation with two
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
class TestChunkedPrefillCaching:
    """Verify chunked-prefill caching still works through the group encoder path.

    On the first chunk `get_multimodal_embeddings` runs the encoder and
    caches the returned tensor into `multimodal_data["multimodal_embedding"]`.
    A second call with the same params must skip the encoder. The
    encoder_fn here is `encode_multimodal_by_groups` bound to the model's
    three per-modality groups — the shape all Nemotron production forwards
    take.
    """

    HIDDEN = 128
    NUM_TOKENS = 10

    def _make_mock_model(self):
        model = mock.MagicMock(spec=NemotronH_Nano_VL_V2)
        model.vision_encoder = mock.MagicMock(spec=NanoV2VLVisionEncoder)
        model.sound_encoder = mock.MagicMock(spec=ProjectedParakeet)
        return model

    def _make_param_with_runtime(self, modality, num_tokens, **extra):
        """`MultimodalParams` with per-modality bucket + runtime metadata.

        `multimodal_embedding_lengths` is required for the group encoder path:
        `_lengths_by_modality` uses it to slice the encoder output back into
        per-modality tensors. Single-item request → single-entry list.
        """
        embed_mask_cumsum = torch.arange(1, num_tokens + 1, dtype=torch.int64)
        runtime = MultimodalRuntimeData(
            past_seen_token_num=0,
            chunk_end_pos=num_tokens,
            embed_mask_cumsum=embed_mask_cumsum,
        )
        nested = extra.pop(modality, {})
        return MultimodalParams(
            multimodal_data={
                "modality_type": modality,
                modality: nested,
                "multimodal_embedding_lengths": [num_tokens],
                **extra,
            },
            multimodal_runtime=runtime,
        )

    def _make_encoder_fn(self, model):
        """Route through `encode_multimodal_by_groups` with real group methods
        bound to the mock's stubbed sub-encoders — matches the production
        `NemotronH_Nano_VL_V2.forward` path.

        The group encoder_fns are invoked via `**build_batched_input(...)`,
        which spreads the pack dict as kwargs — so the wrappers must accept
        `multimodal_params` by keyword.
        """
        from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
            EncoderGroup,
            encode_multimodal_by_groups,
        )

        def _pack(params):
            return {"multimodal_params": params}

        def _run_group(method):
            return lambda multimodal_params: method(model, multimodal_params)

        groups = (
            EncoderGroup(("image",), _run_group(NemotronH_Nano_VL_V2._encode_image_group), _pack),
            EncoderGroup(("video",), _run_group(NemotronH_Nano_VL_V2._encode_video_group), _pack),
            EncoderGroup(("audio",), _run_group(NemotronH_Nano_VL_V2._encode_audio_group), _pack),
        )
        return lambda params: encode_multimodal_by_groups(groups, params)

    @pytest.mark.parametrize("modality", ["image", "video"])
    def test_vision_encoder_not_called_on_second_chunk(self, modality):
        model = self._make_mock_model()
        fake_emb = torch.randn(self.NUM_TOKENS, self.HIDDEN)
        model.vision_encoder.return_value = ([fake_emb], [None])

        param = self._make_param_with_runtime(modality, self.NUM_TOKENS)
        encoder_fn = self._make_encoder_fn(model)

        # First call: encoder runs and caches.
        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert len(result) == 1
        assert result[0].shape == (self.NUM_TOKENS, self.HIDDEN)
        assert model.vision_encoder.call_count == 1
        assert "multimodal_embedding" in param.multimodal_data

        # Second call: comes from cache.
        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert model.vision_encoder.call_count == 1, (
            "`vision_encoder` was called again on the second chunk. Caching is broken."
        )
        assert torch.equal(result2[0], result[0])

    def test_audio_encoder_not_called_on_second_chunk(self):
        model = self._make_mock_model()
        fake_emb = torch.randn(self.NUM_TOKENS, self.HIDDEN)
        model._encode_audio = mock.MagicMock(return_value=[(fake_emb, [self.NUM_TOKENS])])

        param = self._make_param_with_runtime("audio", self.NUM_TOKENS, audio={})
        encoder_fn = self._make_encoder_fn(model)

        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert len(result) == 1
        assert model._encode_audio.call_count == 1

        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param],
        )
        assert model._encode_audio.call_count == 1, (
            "`_encode_audio` was called again on the second chunk. Caching is broken."
        )
        assert torch.equal(result2[0], result[0])

    def test_multi_request_batch_caching(self):
        """Two image requests in one batch: both cached after a single batched call."""
        model = self._make_mock_model()
        emb_a = torch.randn(5, self.HIDDEN)
        emb_b = torch.randn(3, self.HIDDEN)
        model.vision_encoder.return_value = ([emb_a, emb_b], [None, None])

        param_a = self._make_param_with_runtime("image", 5)
        param_b = self._make_param_with_runtime("image", 3)
        encoder_fn = self._make_encoder_fn(model)

        result = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert len(result) == 1
        assert result[0].shape == (8, self.HIDDEN)
        assert model.vision_encoder.call_count == 1, (
            "image params should be encoded in a single batched vision_encoder call"
        )
        assert "multimodal_embedding" in param_a.multimodal_data
        assert "multimodal_embedding" in param_b.multimodal_data

        result2 = get_multimodal_embeddings(
            encoder_forward_fn=encoder_fn,
            multimodal_params=[param_a, param_b],
        )
        assert model.vision_encoder.call_count == 1, (
            "`vision_encoder` was called again on the second chunk. Caching is broken."
        )
        assert torch.equal(result2[0], result[0])


@pytest.mark.cpu_only
class TestModelOptVisionWeightNormalization:
    """`_normalize_vision_weights` maps ModelOpt vision keys onto the release layout."""

    HIDDEN = 4
    BLOCKS = "vision_model.radio_model.model.blocks."
    LAYER = "vision_model.encoder.layer.0."

    def _part(self, offset: int, suffix: str) -> torch.Tensor:
        """A tensor whose every element identifies which q/k/v part it came from."""
        shape = (self.HIDDEN, self.HIDDEN) if suffix == "weight" else (self.HIDDEN,)
        n = int(np.prod(shape))
        return (torch.arange(n, dtype=torch.float32) + offset * 1000).reshape(shape)

    def _qkv_parts(self, suffixes=("weight", "bias")) -> dict:
        return {
            part: {suffix: self._part(offset, suffix) for suffix in suffixes}
            for part, offset in (("query", 1), ("key", 2), ("value", 3))
        }

    def _modelopt_checkpoint(self, parts: dict) -> dict:
        h = self.HIDDEN
        weights = {
            # Not a vision weight: must be left behind for the LLM loader.
            "model.embed_tokens.weight": torch.zeros(2, h),
            # Derived by this port rather than loaded, so dropped, not rejected.
            "vision_model.summary_idxs": torch.arange(2),
            "vision_projector.vision_final_layernorm.weight": torch.zeros(h),
            "vision_projector.mlp1.norm.weight": torch.zeros(h),
            "vision_projector.mlp1.linear1.weight": torch.zeros(h, h),
            "vision_projector.mlp1.linear2.weight": torch.zeros(h, h),
            "vision_model.embeddings.patch_projection.weight": torch.zeros(h, h),
            "vision_model.embeddings.position_embedding": torch.zeros(1, h),
            "vision_model.embeddings.cls_register_token": torch.zeros(1, 1, h),
            self.LAYER + "attention.output.dense.weight": torch.zeros(h, h),
            self.LAYER + "mlp.fc1.weight": torch.zeros(h, h),
            self.LAYER + "norm1.weight": torch.zeros(h),
            self.LAYER + "layer_scale1.lambda1": torch.ones(h),
            self.LAYER + "layer_scale2.lambda1": torch.ones(h),
        }
        for part, tensors in parts.items():
            for suffix, tensor in tensors.items():
                weights[f"{self.LAYER}attention.attention.{part}.{suffix}"] = tensor
        return weights

    def test_fused_qkv_survives_the_split_radio_performs(self):
        """Fuse, then split with the real RADIO splitter: does q come back as q?

        The expected value is the caller's own tensor, so this fails only if the
        two halves of the contract move apart -- which nothing else would catch,
        since q/k/v share a shape and load_state_dict stays happy either way.
        """
        parts = self._qkv_parts()
        remapped = _normalize_vision_weights(self._modelopt_checkpoint(parts))

        for suffix in ("weight", "bias"):
            fused_key = f"{self.BLOCKS}0.attn.qkv.{suffix}"
            split = split_fused_qkv(fused_key, remapped[fused_key])
            for part, proj in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
                assert torch.equal(
                    split[fused_key.replace("attn.qkv.", f"attn.{proj}.")],
                    parts[part][suffix],
                ), f"{part} did not land in {proj}"

    def test_modelopt_keys_all_land_in_the_release_namespace(self):
        """Nothing is invented and nothing is left over -- the whole key set, spelled out."""
        remapped = _normalize_vision_weights(self._modelopt_checkpoint(self._qkv_parts()))
        assert set(remapped) == {
            "mlp1.0.weight",
            "mlp1.1.weight",
            "mlp1.3.weight",
            "vision_model.radio_model.model.patch_generator.embedder.weight",
            "vision_model.radio_model.model.patch_generator.pos_embed",
            "vision_model.radio_model.model.patch_generator.cls_token.token",
            self.BLOCKS + "0.attn.proj.weight",
            self.BLOCKS + "0.attn.qkv.weight",
            self.BLOCKS + "0.attn.qkv.bias",
            self.BLOCKS + "0.mlp.fc1.weight",
            self.BLOCKS + "0.norm1.weight",
        }

    def test_release_layout_is_left_alone(self):
        """A release checkpoint keeps its own spelling, and re-running changes nothing."""
        release = {
            "model.embed_tokens.weight": torch.zeros(2, self.HIDDEN),
            "mlp1.0.weight": torch.zeros(self.HIDDEN),
            self.BLOCKS + "0.attn.qkv.weight": torch.zeros(3 * self.HIDDEN, self.HIDDEN),
        }
        once = _normalize_vision_weights(release)
        assert set(once) == {"mlp1.0.weight", self.BLOCKS + "0.attn.qkv.weight"}
        assert set(_normalize_vision_weights(once)) == set(once)

    @pytest.mark.parametrize(
        "bad_key",
        [
            "vision_model.encoder.layer.0.attention.attention.rotary_emb.inv_freq",
            "vision_model.some_new_submodule.weight",
        ],
    )
    def test_unrecognized_vision_key_is_rejected(self, bad_key):
        """A key nobody mapped must stop the load, not be dropped on the floor."""
        weights = self._modelopt_checkpoint(self._qkv_parts())
        weights[bad_key] = torch.zeros(self.HIDDEN)
        with pytest.raises(ValueError, match="Unrecognized vision"):
            _normalize_vision_weights(weights)

    def test_unfusable_qkv_suffix_is_rejected(self):
        """A quantized vision export would attach scales, which are not rows to concatenate.

        The unrecognized-key guards do not reach here: anything under
        ``attention.attention.{query,key,value}`` is claimed by the fusion
        branch before those guards run.
        """
        parts = self._qkv_parts(suffixes=("weight", "bias", "weight_scale"))
        with pytest.raises(ValueError, match="weight_scale"):
            _normalize_vision_weights(self._modelopt_checkpoint(parts))

    def test_mismatched_qkv_suffixes_are_rejected(self):
        """Q with a bias and K without means one of them was misread."""
        weights = self._modelopt_checkpoint(self._qkv_parts())
        del weights[self.LAYER + "attention.attention.key.bias"]
        with pytest.raises(ValueError, match="key carries"):
            _normalize_vision_weights(weights)

    def test_missing_qkv_part_is_rejected(self):
        """Fusing two of three projections would produce a wrongly shaped tensor."""
        weights = self._modelopt_checkpoint(self._qkv_parts())
        for suffix in ("weight", "bias"):
            del weights[f"{self.LAYER}attention.attention.key.{suffix}"]
        with pytest.raises(ValueError, match=r"missing \['key'\]"):
            _normalize_vision_weights(weights)

    def test_non_identity_layer_scale_is_rejected(self):
        """The RADIO port has no LayerScale, so dropping a real one would be silent."""
        weights = self._modelopt_checkpoint(self._qkv_parts())
        weights[self.LAYER + "layer_scale1.lambda1"] = torch.full((self.HIDDEN,), 0.1)
        with pytest.raises(ValueError, match="not all-ones"):
            _normalize_vision_weights(weights)


@pytest.mark.cpu_only
def test_nemotron_nano_frees_the_modelopt_projector_shard():
    """The ModelOpt projector tensors stop being referenced once the encoder has loaded.

    `mark_consumed` is what lets the loader release mmap pages module by module;
    a prefix that is never marked keeps its shard resident for the rest of load.
    """
    fake_encoder = MagicMock()
    fake_encoder.eval.return_value = fake_encoder
    fake_encoder.to.return_value = fake_encoder

    model = SimpleNamespace(
        _mm_model_config=_make_minimal_nano_model_config(),
        vision_encoder=fake_encoder,
        sound_encoder=None,
        llm=MagicMock(),
        model_config=SimpleNamespace(),
    )
    weights = ConsumableWeightsDict(
        {
            "vision_projector.mlp1.linear1.weight": torch.empty(0),
            "vision_model.radio_model.weight": torch.empty(0),
            "language_model.weight": torch.empty(0),
        }
    )

    with (
        mock.patch.dict(os.environ, {"TLLM_MULTIMODAL_DISAGGREGATED": "0"}),
        mock.patch.object(nemotron_nano, "NemotronHHfWeightMapper", MagicMock()),
    ):
        NemotronH_Nano_VL_V2.load_weights(model, weights)

    assert "vision_projector.mlp1.linear1.weight" not in weights
    assert "vision_model.radio_model.weight" not in weights
