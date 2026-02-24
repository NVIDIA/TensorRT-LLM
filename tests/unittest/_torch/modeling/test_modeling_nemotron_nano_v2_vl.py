import os
from pathlib import Path

import pytest
import torch
import transformers
from test_modeling_multimodal import llm_models_root
from test_modeling_nemotron_h import extract_decode_logprobs

from tensorrt_llm import LLM
from tensorrt_llm.inputs import (
    create_input_processor,
    create_input_processor_with_hash,
    default_multimodal_input_loader,
    prompt_inputs,
)
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
                "prompt_pattern": "<video>\n<video>\n",
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
                [-8.5795e-01, -1.5373e-01, -7.2846e-04, -6.3667e-01, -3.1307e-02]
            ),
            "multiple": torch.tensor([-0.5846, -0.6330, -0.0124, -0.1146, -0.0172]),
        },
        "video": {
            "single": torch.tensor([-0.5612, -0.0334, -0.8856, -0.4056, -0.6041]),
            "multiple": torch.tensor([-0.4943, -0.9333, -0.0096, -1.2496, -0.9441]),
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
