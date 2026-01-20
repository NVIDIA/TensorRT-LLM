import os
from pathlib import Path

import defs.ci_profiler
import pytest
from defs.common import (convert_weights, test_llm_torch_multi_lora_support,
                         venv_check_call, venv_mpi_check_call)
from defs.conftest import get_device_memory, get_sm_version
from defs.trt_test_alternative import check_call

from tensorrt_llm import LLM
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)

ROUGE1_ACCURACY_THRESHOLD = 20


@pytest.mark.parametrize("nemotron_nas_model_root", [
    "DeciLM-7B",
],
                         indirect=True)
def test_nemotron_nas_summary_1gpu(nemotron_nas_example_root, llm_venv,
                                   nemotron_nas_model_root, llm_datasets_root,
                                   llm_rouge_root, engine_dir, cmodel_dir):
    model_name = Path(nemotron_nas_model_root).name
    if "51B" in model_name and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    print(f"Model name: {model_name}")
    dtype = 'float16'
    ckpt_type = "hf"

    print("Converting checkpoint...")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=nemotron_nas_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=nemotron_nas_model_root,
                                data_type=dtype,
                                ckpt_type=ckpt_type,
                                gpus=1,
                                tp_size=1,
                                trust_remote_code=True)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={4}",
        f"--max_input_len={2048}", "--kv_cache_type=paged",
        "--remove_input_padding=enable", "--gemm_plugin=auto",
        "--gpt_attention_plugin=auto"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    summary_cmd = [
        f"{nemotron_nas_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}", "--test_hf", "--hf_device_map_auto",
        "--batch_size=1", "--test_trt_llm",
        f"--hf_model_dir={nemotron_nas_model_root}", "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={ROUGE1_ACCURACY_THRESHOLD}",
        "--no_add_special_tokens", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]

    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("nemotron_nas_model_root", [
    "DeciLM-7B",
    "Llama-3_1-Nemotron-51B-Instruct",
],
                         indirect=True)
def test_nemotron_nas_summary_2gpu(nemotron_nas_example_root, llm_venv,
                                   nemotron_nas_model_root, llm_datasets_root,
                                   llm_rouge_root, engine_dir, cmodel_dir):
    model_name = Path(nemotron_nas_model_root).name
    if "51B" in model_name and get_device_memory() < 80000:
        pytest.skip("device memory is insufficient.")

    print(f"Model name: {model_name}")
    dtype = 'float16'
    ckpt_type = "hf"

    print("Converting checkpoint...")
    model_dir = convert_weights(llm_venv=llm_venv,
                                example_root=nemotron_nas_example_root,
                                cmodel_dir=cmodel_dir,
                                model=model_name,
                                model_path=nemotron_nas_model_root,
                                data_type=dtype,
                                ckpt_type=ckpt_type,
                                gpus=2,
                                tp_size=2,
                                trust_remote_code=True)

    print("Building engines...")
    build_cmd = [
        "trtllm-build", f"--checkpoint_dir={model_dir}",
        f"--output_dir={engine_dir}", f"--max_batch_size={4}",
        f"--max_input_len={2048}", "--kv_cache_type=paged",
        "--remove_input_padding=enable", "--gemm_plugin=auto",
        "--gpt_attention_plugin=auto"
    ]

    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Running inference...")

    mpi_cmd = ["mpirun", "-n", "2", "--allow-run-as-root"]

    summary_cmd = [
        f"{nemotron_nas_example_root}/../../../summarize.py",
        f"--engine_dir={engine_dir}", "--test_hf", "--hf_device_map_auto",
        "--batch_size=1", "--test_trt_llm",
        f"--hf_model_dir={nemotron_nas_model_root}", "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={ROUGE1_ACCURACY_THRESHOLD}",
        "--no_add_special_tokens", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, mpi_cmd, summary_cmd)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("nemotron_nas_model_root", [
    "Llama-3.1-Nemotron-Nano-8B-v1",
],
                         indirect=True)
def test_nemotron_nano_8b_lora_torch(nemotron_nas_example_root, llm_venv,
                                     nemotron_nas_model_root, llm_datasets_root,
                                     llm_rouge_root, engine_dir, cmodel_dir):
    """Run Nemotron Nano 8B with multiple dummy LoRAs using LLM-API Torch backend."""

    expected_outputs = {
        'llama-3.1-nemotron-nano-8b-v1': [
            " I am having a bit of a problem with my computer. The screen is black, but my monitor is still giving me the same signals. The brightness",
            " How is the climate like? What are some of the typical foods and drinks of the region? What is the economy like? How does the city compare",
            " I have heard that it's possible but can be dangerous. What are the potential risks? Are there any safety guidelines? I should probably check some references",
            " I can't do that right now. But I can suggest that if you're interested in music trends, you can check Spotify's \"Discover Weekly\"",
            " The capital of France is Paris. But wait, I think there's another city called Paris. No, no, that's the same city. Maybe"
        ],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")

    model_name = os.path.basename(nemotron_nas_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=nemotron_nas_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=1,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("nemotron_nas_model_root", [
    "Llama-3_3-Nemotron-Super-49B-v1",
],
                         indirect=True)
@pytest.mark.parametrize(
    "llm_lora_model_root",
    ['Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_NIM_r32'],
    indirect=True)
def test_nemotron_super_49b_real_lora_torch(nemotron_nas_example_root, llm_venv,
                                            nemotron_nas_model_root,
                                            llm_lora_model_root,
                                            llm_datasets_root, llm_rouge_root,
                                            engine_dir, cmodel_dir):
    """Run Nemotron Super 49B with real LoRA adapters using LLM-API Torch backend."""

    print("Testing Nemotron Super 49B with real LoRA adapters...")

    print(f"Using real LoRA from: {llm_lora_model_root}")

    defs.ci_profiler.start("test_nemotron_real_lora_torch")

    lora_config = LoraConfig(
        lora_dir=[llm_lora_model_root],
        max_lora_rank=32,  # From adapter_config.json: "r": 32
        max_loras=1,
        max_cpu_loras=1,
    )

    with LLM(model=nemotron_nas_model_root,
             lora_config=lora_config,
             tensor_parallel_size=4,
             dtype="bfloat16",
             max_batch_size=2,
             max_input_len=512,
             max_seq_len=1024,
             max_beam_width=1,
             load_format="dummy") as llm:

        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms."
        ]

        sampling_params = SamplingParams(max_tokens=50,
                                         temperature=0.7,
                                         top_p=0.9)

        lora_request = [
            LoRARequest("nemotron-lora", 0, llm_lora_model_root),
            LoRARequest("nemotron-lora", 1, llm_lora_model_root)
        ]

        print("Running inference with real LoRA adapter...")
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_request)

        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}: {prompts[i]}")
            print(f"Response {i+1}: {output.outputs[0].text}")
            print("-" * 50)

        assert len(outputs) == 2
        assert len(outputs[0].outputs) > 0
        assert len(outputs[1].outputs) > 0
        assert len(outputs[0].outputs[0].text) > 0
        assert len(outputs[1].outputs[0].text) > 0

    defs.ci_profiler.stop("test_nemotron_real_lora_torch")
    print(
        f"test_nemotron_real_lora_torch: {defs.ci_profiler.elapsed_time_in_sec('test_nemotron_real_lora_torch')} sec"
    )


@pytest.mark.skip(reason="TODO: Test OOMs on 8 GPUs - to fix")
@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("nemotron_nas_model_root", [
    "Llama-3_1-Nemotron-Ultra-253B-v1",
],
                         indirect=True)
def test_nemotron_ultra_253b_lora_torch(nemotron_nas_example_root, llm_venv,
                                        nemotron_nas_model_root,
                                        llm_datasets_root, llm_rouge_root,
                                        engine_dir, cmodel_dir):
    """Run Nemotron Ultra 253B with multiple dummy LoRAs using LLM-API Torch backend."""

    expected_outputs = {
        'Llama-3_1-Nemotron-Ultra-253B-v1': ["...", "...", "...", "...", "..."],
    }

    print("Testing with LLM-API Torch backend...")

    defs.ci_profiler.start("test_llm_torch_multi_lora_support")
    model_name = os.path.basename(nemotron_nas_model_root).lower()
    test_llm_torch_multi_lora_support(
        hf_model_dir=nemotron_nas_model_root,
        llm_venv=llm_venv,
        num_loras=2,
        lora_rank=8,
        target_hf_modules=["q_proj", "k_proj", "v_proj"],
        zero_lora_weights=True,
        tensor_parallel_size=8,
        expected_outputs=expected_outputs[model_name])
    defs.ci_profiler.stop("test_llm_torch_multi_lora_support")
    print(
        f"test_llm_torch_multi_lora_support: {defs.ci_profiler.elapsed_time_in_sec('test_llm_torch_multi_lora_support')} sec"
    )
