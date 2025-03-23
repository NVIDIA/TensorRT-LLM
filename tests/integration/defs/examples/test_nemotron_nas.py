from pathlib import Path

import pytest
from defs.common import convert_weights, venv_check_call, venv_mpi_check_call
from defs.conftest import get_device_memory
from defs.trt_test_alternative import check_call

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
        f"{nemotron_nas_example_root}/../summarize.py",
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
        f"{nemotron_nas_example_root}/../summarize.py",
        f"--engine_dir={engine_dir}", "--test_hf", "--hf_device_map_auto",
        "--batch_size=1", "--test_trt_llm",
        f"--hf_model_dir={nemotron_nas_model_root}", "--check_accuracy",
        f"--tensorrt_llm_rouge1_threshold={ROUGE1_ACCURACY_THRESHOLD}",
        "--no_add_special_tokens", f"--dataset_dir={llm_datasets_root}",
        f"--rouge_dir={llm_rouge_root}"
    ]

    venv_mpi_check_call(llm_venv, mpi_cmd, summary_cmd)
