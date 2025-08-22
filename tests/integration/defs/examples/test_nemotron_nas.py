from pathlib import Path

import defs.ci_profiler
import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
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


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("nemotron_nas_model_root", [
    "Llama-3_3-Nemotron-Super-49B-v1",
],
                         indirect=True)
def test_nemotron_super_49b_real_lora_torch(nemotron_nas_example_root, llm_venv,
                                            nemotron_nas_model_root,
                                            llm_datasets_root, llm_rouge_root,
                                            engine_dir, cmodel_dir):
    """Run Nemotron Super 49B with real LoRA adapters using LLM-API Torch backend."""

    print("Testing Nemotron Super 49B with real LoRA adapters...")

    lora_adapter_path = f"/code/tensorrt_llm/llama-3.3-nemotron-super-49b-v1/llama-3.3-nemotron-super-49b-v1_vlora-1a2cb80-v2"
    print(f"Using real LoRA from: {lora_adapter_path}")

    defs.ci_profiler.start("test_nemotron_real_lora_torch")

    lora_config = LoraConfig(
        lora_dir=[lora_adapter_path],
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
             max_beam_width=1) as llm:

        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms."
        ]

        sampling_params = SamplingParams(max_tokens=50,
                                         temperature=0.7,
                                         top_p=0.9)

        lora_request = [LoRARequest("nemotron-lora", 0, lora_adapter_path)]

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
