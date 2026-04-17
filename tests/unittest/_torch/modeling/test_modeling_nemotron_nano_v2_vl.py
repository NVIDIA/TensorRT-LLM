import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import transformers
from test_modeling_multimodal import llm_models_root
from test_modeling_nemotron_h import extract_decode_logprobs

from tensorrt_llm import LLM
from tensorrt_llm._torch.models.modeling_multimodal_utils import get_multimodal_embeddings
from tensorrt_llm._torch.models.modeling_nemotron_nano import (
    NanoV2VLVisionEncoder,
    NemotronH_Nano_VL_V2,
)
from tensorrt_llm._torch.models.modeling_parakeet import ProjectedParakeet
from tensorrt_llm.inputs import (
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
        max_batch_size=2,
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
            "multiple": torch.tensor([-0.4717, -0.7776, -0.0251, -1.2290, -1.0705]),
        },
        "video": {
            "single": torch.tensor([-1.4745, -0.0674, -1.4121, -0.2152, -1.6297]),
            "multiple": torch.tensor([-0.9425, -0.2328, -0.0083, -1.6257, -0.6572]),
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
        param = mock.MagicMock()
        param.multimodal_data = {"modality_type": modality_type, **extra}
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
        runtime = MultimodalRuntimeData(
            past_seen_token_num=0,
            mm_token_lengths=[num_tokens],
            mm_token_positions=[0],
            chunk_end_pos=num_tokens,
            special_token_offsets=[],
        )
        return MultimodalParams(
            multimodal_data={"modality_type": modality_type, **extra},
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
