import os

import pytest
from defs.common import venv_check_call
from defs.conftest import llm_models_root, skip_pre_blackwell


@pytest.mark.parametrize("enable_fp4",
                         [pytest.param(True, marks=skip_pre_blackwell), False],
                         ids=["enable_fp4", "disable_fp4"])
@pytest.mark.parametrize("model_name", ["llama-3.1-8b"])
def test_llm_llama_1gpu(
    mmlu_dataset_root,
    enable_fp4,
    llama_example_root,
    model_name,
    llm_venv,
):
    models_root = llm_models_root()
    if enable_fp4:
        model_dir = os.path.join(models_root, "nvfp4-quantized",
                                 "Meta-Llama-3.1-8B")
    else:
        model_dir = os.path.join(models_root, "llama-3.1-model",
                                 "Meta-Llama-3.1-8B")

    print("Run MMLU test")
    accuracy_map = {
        'llama-3.1-8b': 0.61,
    }
    acc_thres = accuracy_map[model_name]
    mmlu_cmd = [
        f"{llama_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=pytorch",
        "--check_accuracy",
        "--enable_chunked_prefill",
        f"--accuracy_threshold={acc_thres}",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@pytest.mark.parametrize("enable_fp4",
                         [pytest.param(True, marks=skip_pre_blackwell), False],
                         ids=["enable_fp4", "disable_fp4"])
@pytest.mark.parametrize("enable_fp8", [
    pytest.param(True, marks=pytest.mark.skip_device_not_contain(["H100"])),
    False
],
                         ids=["enable_fp8", "disable_fp8"])
@pytest.mark.parametrize("model_name", ["deepseek-v3-lite"])
def test_llm_deepseek_1gpu(
    mmlu_dataset_root,
    enable_fp4,
    enable_fp8,
    llama_example_root,
    model_name,
    llm_venv,
):
    models_root = llm_models_root()
    if enable_fp4:
        model_dir = os.path.join(models_root, "DeepSeek-V3-Lite",
                                 "nvfp4_moe_only")
    elif enable_fp8:
        model_dir = os.path.join(models_root, "DeepSeek-V3-Lite", "fp8")
    else:
        model_dir = os.path.join(models_root, "DeepSeek-V3-Lite", "bf16")

    print("Run MMLU test")
    accuracy_map = {
        'deepseek-v3-lite': 0.68,
    }
    acc_thres = accuracy_map[model_name]
    mmlu_cmd = [
        f"{llama_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=pytorch",
        "--check_accuracy",
        "--enable_overlap_scheduler",
        f"--accuracy_threshold={acc_thres}",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param('Llama-3.3-70B-Instruct-fp8',
                 'modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp8',
                 marks=pytest.mark.skip_device_not_contain(["B200", "H100"])),
    pytest.param('Llama-3.3-70B-Instruct-fp4',
                 'modelopt-hf-model-hub/Llama-3.3-70B-Instruct-fp4',
                 marks=pytest.mark.skip_device_not_contain(["B200"])),
])
def test_mmlu_llmapi_4gpus(llm_venv, llama_example_root, mmlu_dataset_root,
                           model_name, model_path):
    models_root = llm_models_root()
    model_dir = os.path.join(models_root, model_path)

    print(f"Run MMLU test on {model_name}.")
    accuracy_map = {
        'Llama-3.3-70B-Instruct-fp8': 0.804,
        'Llama-3.3-70B-Instruct-fp4': 0.785,
    }
    acc_thres = accuracy_map[model_name]
    mmlu_cmd = [
        f"{llama_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=pytorch",
        "--check_accuracy",
        "--enable_chunked_prefill",
        f"--accuracy_threshold={acc_thres}",
        f"--tp_size=4",
    ]

    venv_check_call(llm_venv, mmlu_cmd)


@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("model_name,model_path", [
    pytest.param('Mixtral-8x7B-Instruct-v0.1-fp8',
                 'modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp8',
                 marks=pytest.mark.skip_device_not_contain(["B200", "H100"])),
    pytest.param('Mixtral-8x7B-Instruct-v0.1-fp4',
                 'modelopt-hf-model-hub/Mixtral-8x7B-Instruct-v0.1-fp4',
                 marks=pytest.mark.skip_device_not_contain(["B200"])),
])
def test_mmlu_llmapi_2gpus(llm_venv, llama_example_root, mmlu_dataset_root,
                           model_name, model_path):
    models_root = llm_models_root()
    model_dir = os.path.join(models_root, model_path)

    print(f"Run MMLU test on {model_name}.")
    accuracy_map = {
        'Mixtral-8x7B-Instruct-v0.1-fp8': 0.679,
        'Mixtral-8x7B-Instruct-v0.1-fp4': 0.669,
    }
    acc_thres = accuracy_map[model_name]
    mmlu_cmd = [
        f"{llama_example_root}/../mmlu_llmapi.py",
        "--data_dir",
        f"{mmlu_dataset_root}",
        "--hf_model_dir",
        f"{model_dir}",
        "--backend=pytorch",
        "--check_accuracy",
        "--enable_chunked_prefill",
        f"--accuracy_threshold={acc_thres}",
        f"--tp_size=2",
    ]

    venv_check_call(llm_venv, mmlu_cmd)
