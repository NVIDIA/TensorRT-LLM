import os
import sys
import time

import pytest
import torch
import yaml

from .common import (check_server_ready, prepare_llmapi_model_repo,
                     set_llmapi_decoupled_mode)
from .conftest import find_repo_root, venv_check_call, venv_check_output
from .trt_test_alternative import call, check_call, print_info

LLM_ROOT = os.environ.get("LLM_ROOT", find_repo_root())
sys.path.append(os.path.join(LLM_ROOT, "triton_backend"))


@pytest.fixture(autouse=True)
def stop_triton_server():
    # Make sure Triton server are killed before each test.
    call("pkill -9 -f tritonserver", shell=True)
    call("pkill -9 -f trtllmExecutorWorker", shell=True)
    time.sleep(2)
    yield
    # Gracefully terminate Triton Server after each test.
    call("pkill -f tritonserver", shell=True)
    call("pkill -f trtllmExecutorWorker", shell=True)
    time.sleep(8)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["tensorrt_llm"])
@pytest.mark.parametrize("DECOUPLED_MODE", [False, True],
                         ids=["disableDecoupleMode", "enableDecoupleMode"])
# TODO: [JIRA-4496] Add batch support in llmapi backend and add tests here.
@pytest.mark.parametrize("TRITON_MAX_BATCH_SIZE", ["0"])
@pytest.mark.parametrize("TENSOR_PARALLEL_SIZE", ["1", "4"])
def test_llmapi_backend(E2E_MODEL_NAME, DECOUPLED_MODE, TRITON_MAX_BATCH_SIZE,
                        TENSOR_PARALLEL_SIZE,
                        llm_backend_inflight_batcher_llm_root, llm_backend_venv,
                        llm_backend_dataset_root, tiny_llama_model_root):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    if torch.cuda.device_count() < int(TENSOR_PARALLEL_SIZE):
        pytest.skip("Skipping. Not enough GPUs.")

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo)
    set_llmapi_decoupled_mode(new_model_repo, DECOUPLED_MODE)
    model_config_path = os.path.join(new_model_repo, "tensorrt_llm", "1",
                                     "model.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_config["triton_config"]["decoupled"] = DECOUPLED_MODE
    model_config["triton_config"]["max_batch_size"] = int(TRITON_MAX_BATCH_SIZE)
    model_config["tensor_parallel_size"] = int(TENSOR_PARALLEL_SIZE)
    model_config["kv_cache_config"] = {"free_gpu_memory_fraction": 0.8}
    model_config["model"] = tiny_llama_model_root
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    print_info(f"DEBUG:: model_config: {model_config}")
    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    cmd = f"python3 {launch_server_py} --world_size={TENSOR_PARALLEL_SIZE} --model_repo={new_model_repo}"
    if TENSOR_PARALLEL_SIZE == "4":
        cmd += " --trtllm_llmapi_launch"
        cmd += " --oversubscribe"
    else:
        cmd += " --no-mpi"
    print_info(f"DEBUG:: launch_server with args: {cmd}")
    check_call(cmd, shell=True)
    check_server_ready()

    # Speed up the test by running multiple tests with different configurations sharing the same triton server.
    protocols = ["http", "grpc"]
    STREAMS = [False, True]
    if DECOUPLED_MODE:
        protocols = ['grpc']  # Triton only support grpc in decoupled mode
        STREAMS = [True]  # Triton only support non-streaming in decoupled mode
    else:
        STREAMS = [False
                   ]  # Triton only support non-streaming in non-decoupled mode

    for protocol in protocols:
        for STREAM in STREAMS:
            print_info(
                f"DEBUG:: protocol: {protocol}, STREAM: {STREAM}, DECOUPLED_MODE: {DECOUPLED_MODE}"
            )
            run_cmd = [
                f"{llm_backend_inflight_batcher_llm_root}/end_to_end_test.py",
                f"--protocol={protocol}",
                f"--test-llmapi",
                f"--model-name={E2E_MODEL_NAME}",
                f"--max-input-len=192",
                f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
            ]
            if STREAM:
                run_cmd += [
                    "--streaming",
                ]

            print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
            venv_check_call(llm_backend_venv, run_cmd)

            run_cmd = [
                f"{llm_backend_inflight_batcher_llm_root}/benchmark_core_model.py",
                f"--max-input-len=300",
                f"--tensorrt-llm-model-name={E2E_MODEL_NAME}",
                f"--protocol={protocol}",
                f"--test-llmapi",
            ]
            if DECOUPLED_MODE:
                # Triton rejects ModelInfer RPC on decoupled models; the
                # benchmark must use async_stream_infer in that mode.
                run_cmd.append("--decoupled")
            run_cmd += [
                'dataset',
                f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
                f"--tokenizer-dir={tiny_llama_model_root}",
            ]

            print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
            venv_check_call(llm_backend_venv, run_cmd)

            # Test request cancellation with stop request
            run_cmd = [
                f"{llm_backend_repo_root}/tools/llmapi_client.py",
                "--request-output-len=200", '--stop-after-ms=25'
            ]
            if DECOUPLED_MODE:
                # On a decoupled server, ModelInfer RPC is rejected;
                # llmapi_client.py must use the bidirectional stream
                # RPC, which it routes through when --streaming is set.
                run_cmd.append('--streaming')

            output = venv_check_output(llm_backend_venv, run_cmd)
            assert 'Request is cancelled' in output

            # Test request cancellation with  request cancel
            run_cmd += ['--stop-via-request-cancel']
            output = venv_check_output(llm_backend_venv, run_cmd)
            assert 'Request is cancelled' in output

            # Test request cancellation for non-existing request and
            # completed request. The helper script uses ModelInfer RPC
            # (async_infer), which Triton rejects on a decoupled model;
            # only exercise it in the non-decoupled config.
            if not DECOUPLED_MODE:
                run_cmd = [
                    f"{llm_backend_repo_root}/tools/tests/test_llmapi_cancel.py"
                ]
                output = venv_check_output(llm_backend_venv, run_cmd)


@pytest.mark.parametrize("E2E_MODEL_NAME", ["tensorrt_llm"])
@pytest.mark.parametrize("TENSOR_PARALLEL_SIZE", ["1"])
def test_llmapi_lora(E2E_MODEL_NAME, TENSOR_PARALLEL_SIZE,
                     llm_backend_inflight_batcher_llm_root, llm_backend_venv,
                     tiny_llama_model_root, tiny_llama_lora_model_root):
    """E2E LoRA test for the new llmapi triton backend.

    Templates `model.yaml` with `lora_config:` pointing at a TinyLlama
    HF LoRA adapter, launches Triton with the llmapi backend, and sends
    one request via `llmapi_client.py --lora-id/--lora-name/--lora-path`.
    Asserts that the response carries generated text — proving the new
    lora_id/lora_name/lora_path inputs reach `LLM.generate_async(
    lora_request=...)` and adapter-applied inference completes.
    """
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    if torch.cuda.device_count() < int(TENSOR_PARALLEL_SIZE):
        pytest.skip("Skipping. Not enough GPUs.")

    # Prepare model repo with lora_config
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo)
    set_llmapi_decoupled_mode(new_model_repo, False)
    model_config_path = os.path.join(new_model_repo, "tensorrt_llm", "1",
                                     "model.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_config["triton_config"]["decoupled"] = False
    model_config["triton_config"]["max_batch_size"] = 0
    model_config["tensor_parallel_size"] = int(TENSOR_PARALLEL_SIZE)
    model_config["kv_cache_config"] = {"free_gpu_memory_fraction": 0.8}
    model_config["model"] = tiny_llama_model_root
    model_config["lora_config"] = {
        "lora_dir": [tiny_llama_lora_model_root],
        "max_lora_rank": 64,
        "max_loras": 1,
        "max_cpu_loras": 1,
    }
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    # Launch Triton Server
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    cmd = (f"python3 {launch_server_py} "
           f"--world_size={TENSOR_PARALLEL_SIZE} "
           f"--model_repo={new_model_repo} --no-mpi")
    print_info(f"DEBUG:: launch_server with args: {cmd}")
    check_call(cmd, shell=True)
    check_server_ready()

    # Send a LoRA request via llmapi_client.py
    run_cmd = [
        f"{llm_backend_repo_root}/tools/llmapi_client.py",
        "--text=I've noticed you seem a bit down lately. "
        "Is there anything you'd like to talk about?",
        "--request-output-len=32",
        "--lora-id=0",
        "--lora-name=mental-health",
        f"--lora-path={tiny_llama_lora_model_root}",
        f"--model-name={E2E_MODEL_NAME}",
    ]
    print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
    output = venv_check_output(llm_backend_venv, run_cmd)
    assert "Output text:" in output, (
        f"Expected 'Output text:' in client output, got: {output[:500]}")


def test_llmapi_backend_multi_instance(llm_backend_inflight_batcher_llm_root,
                                       llm_backend_venv,
                                       llm_backend_dataset_root,
                                       tiny_llama_model_root):
    llm_backend_repo_root = os.path.join(LLM_ROOT, "triton_backend")

    # Prepare model repo
    new_model_repo = os.path.join(llm_backend_repo_root, "triton_repo")
    prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo)
    set_llmapi_decoupled_mode(new_model_repo, True)

    # Modify model.yaml
    model_config_path = os.path.join(new_model_repo, "tensorrt_llm", "1",
                                     "model.yaml")
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    model_config["triton_config"]["decoupled"] = True
    model_config["triton_config"]["max_batch_size"] = 0
    model_config["tensor_parallel_size"] = 1
    # Low KV cache to ensure both instances fit on GPU 0
    model_config["kv_cache_config"] = {"free_gpu_memory_fraction": 0.3}
    model_config["model"] = tiny_llama_model_root
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)

    # Modify config.pbtxt for 2 instances on GPU 0
    config_pbtxt_path = os.path.join(new_model_repo, "tensorrt_llm",
                                     "config.pbtxt")
    with open(config_pbtxt_path, "r") as f:
        config_content = f.read()
    # Replace instance_group to have 2 instances
    original_instance_group = "instance_group [\n  {\n    count: 1\n    kind : KIND_CPU\n  }\n]"
    assert original_instance_group in config_content, (
        f"Expected instance_group block not found in config.pbtxt. "
        f"The config.pbtxt format may have changed. Content:\n{config_content[:500]}"
    )
    config_content = config_content.replace(
        original_instance_group,
        "instance_group [\n  {\n    count: 2\n    kind : KIND_CPU\n  }\n]\n\nparameters {\n  key: \"gpu_device_ids\"\n  value: { string_value: \"0;0\" }\n}"
    )
    with open(config_pbtxt_path, "w") as f:
        f.write(config_content)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    print_info(f"DEBUG:: model_config: {model_config}")
    with open(config_pbtxt_path, "r") as f:
        print_info(f"DEBUG:: config.pbtxt:\n{f.read()}")

    # Launch Triton Server with --no-mpi (required for multi-instance)
    launch_server_py = os.path.join(llm_backend_repo_root, "scripts",
                                    "launch_triton_server.py")
    cmd = f"python3 {launch_server_py} --world_size=1 --model_repo={new_model_repo} --no-mpi"
    print_info(f"DEBUG:: launch_server with args: {cmd}")
    check_call(cmd, shell=True)
    check_server_ready()

    # Test with grpc protocol and streaming (decoupled mode)
    protocol = "grpc"

    # Run end_to_end_test
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/end_to_end_test.py",
        f"--protocol={protocol}",
        "--test-llmapi",
        "--model-name=tensorrt_llm",
        "--max-input-len=192",
        f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
        "--streaming",
    ]
    print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
    venv_check_call(llm_backend_venv, run_cmd)

    # Run benchmark_core_model
    run_cmd = [
        f"{llm_backend_inflight_batcher_llm_root}/benchmark_core_model.py",
        "--max-input-len=300",
        "--tensorrt-llm-model-name=tensorrt_llm",
        f"--protocol={protocol}",
        "--test-llmapi",
        "--decoupled",
        "dataset",
        f"--dataset={os.path.join(llm_backend_dataset_root, 'mini_cnn_eval.json')}",
        f"--tokenizer-dir={tiny_llama_model_root}",
    ]
    print_info("DEBUG:: run_cmd: python3 " + " ".join(run_cmd))
    venv_check_call(llm_backend_venv, run_cmd)
