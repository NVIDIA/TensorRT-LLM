import os
import queue
import subprocess
import threading
import time

import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci
from defs.conftest import llm_models_root, skip_no_hopper
from defs.trt_test_alternative import popen, print_error, print_info
from openai import OpenAI
from requests.exceptions import RequestException


def check_server_ready(http_port, timeout_timer=600, sleep_interval=5):
    timer = 0
    while timer <= timeout_timer:
        try:
            url = f"http://0.0.0.0:{http_port}/health"
            r = requests.get(url, timeout=sleep_interval)
            if r.status_code == 200:
                # trtllm-serve health endpoint just returns status 200, no JSON body required
                break
        except RequestException:
            pass

        time.sleep(sleep_interval)
        timer += sleep_interval

    if timer > timeout_timer:
        raise TimeoutError(
            f"Error: Launch trtllm-serve timed out, timer is {timeout_timer} seconds."
        )

    print_info(
        f"trtllm-serve launched successfully! Cost {timer} seconds to launch server."
    )


def wait_for_log(log_queue, expected_log, timeout=10):
    """Waits for a specific log message to appear in the queue."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_logs = "".join(list(log_queue.queue))
        if expected_log in current_logs:
            return True
        time.sleep(0.1)
    return False


def check_openai_chat_completion(http_port,
                                 model_name="TinyLlama-1.1B-Chat-v1.0"):
    """
    Test the launched trtllm-serve server using OpenAI client.

    Args:
        http_port: The port where the server is running
        model_name: The model name to use for the chat completion
    """
    print_info("Testing trtllm-serve server with OpenAI chat completion...")

    try:
        # Create OpenAI client pointing to the local server
        client = OpenAI(
            base_url=f"http://localhost:{http_port}/v1",
            api_key="tensorrt_llm",
        )

        # Make a chat completion request
        response = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "system",
                "content": "you are a helpful assistant"
            }, {
                "role": "user",
                "content": "What is the capital of China?"
            }],
            max_tokens=50,
        )

        # Validate the response
        assert response is not None, "Response should not be None"
        assert hasattr(response,
                       'choices'), "Response should have choices attribute"
        assert len(
            response.choices) > 0, "Response should have at least one choice"
        assert hasattr(response.choices[0],
                       'message'), "Choice should have message attribute"
        assert hasattr(response.choices[0].message,
                       'content'), "Message should have content attribute"

        content = response.choices[0].message.content
        assert content is not None, "Message content should not be None"
        assert len(content.strip()) > 0, "Message content should not be empty"

        print_info(f"Chat completion test passed!")
        print_info(
            f"Model: {response.model if hasattr(response, 'model') else 'Unknown'}"
        )
        print_info(f"Response: {content}")

        return response

    except Exception as e:
        print_error(f"Chat completion test failed: {str(e)}")
        raise


@pytest.mark.parametrize("config_flag", ["--extra_llm_api_options", "--config"])
@skip_no_hopper
def test_config_file_loading(serve_test_root, config_flag):
    """Test config file loading via both --extra_llm_api_options and --config flags."""
    test_configs_root = f"{serve_test_root}/test_configs"

    # moe backend = CUTLASS which only supports fp8 blockscale on Hopper
    config_file = f"{test_configs_root}/Qwen3-30B-A3B-FP8.yml"
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B-FP8"

    # Assert that required files and directories exist
    assert os.path.exists(
        test_configs_root
    ), f"test_configs_root directory does not exist: {test_configs_root}"
    assert os.path.exists(
        config_file), f"config_file does not exist: {config_file}"
    assert os.path.exists(
        model_path), f"model_path does not exist: {model_path}"

    port = get_free_port_in_ci()
    cmd = [
        "trtllm-serve",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--backend",
        "pytorch",
        config_flag,
        config_file,
    ]

    print_info(f"Launching trtllm-serve with {config_flag}...")
    with popen(cmd):
        check_server_ready(http_port=port)
        # Extract model name from the model path for consistency
        model_name = model_path.split('/')[-1]  # "Qwen3-30B-A3B-FP8"
        # Test the server with OpenAI chat completion
        check_openai_chat_completion(http_port=port, model_name=model_name)


@skip_no_hopper
def test_env_overrides_pdl(tmp_path):
    """
    This test ensures that the `env_overrides` configuration option effectively propagates
    environment variables to the server workers. Specifically, it sets `TRTLLM_ENABLE_PDL=1`
    (Programmatic Dependent Launch) via config and verifies it overrides the env var initially set to 0.

    1. This model (TinyLlama-1.1B-Chat-v1.0) architecture uses RMSNorm, which triggers 'flashinfer' kernels that use PDL when `TRTLLM_ENABLE_PDL=1`.
    2. When `TRTLLM_ENABLE_PDL=1` is actually propagated into worker env, flashinfer custom ops log "PDL enabled" to stdout/stderr.
    """
    pdl_enabled = "1"
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        yaml.dump({
            "backend": "pytorch",
            "env_overrides": {
                "TRTLLM_ENABLE_PDL": pdl_enabled
            }
        }))

    port = get_free_port_in_ci()
    env = os.environ.copy()
    # Pre-load with 0 to verify override works
    env.update({
        "TLLM_LOG_LEVEL": "INFO",
        "LLM_MODELS_ROOT": llm_models_root(),
        "TRTLLM_ENABLE_PDL": "0"
    })

    cmd = [
        "trtllm-serve", "serve",
        f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "--host", "0.0.0.0", "--port",
        str(port), "--backend", "pytorch", "--config",
        str(config_file)
    ]

    with popen(cmd,
               env=env,
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               universal_newlines=True) as proc:
        output_queue = queue.Queue()
        threading.Thread(
            target=lambda:
            [output_queue.put(line) for line in iter(proc.stdout.readline, '')],
            daemon=True).start()

        check_server_ready(http_port=port, timeout_timer=300)
        response = OpenAI(base_url=f"http://localhost:{port}/v1",
                          api_key="tensorrt_llm").chat.completions.create(
                              model="TinyLlama-1.1B-Chat-v1.0",
                              messages=[{
                                  "role": "user",
                                  "content": "Test"
                              }],
                              max_tokens=10)
        assert response and response.choices[0].message.content

        # Directly wait for the PDL log we expect
        if not wait_for_log(output_queue, "PDL enabled", timeout=10):
            logs = "".join(list(output_queue.queue))
            print_error(
                f"Timeout waiting for 'PDL enabled'. Captured logs:\n{logs}")
            assert False

    logs = ''.join(output_queue.queue)
    assert "Overriding TRTLLM_ENABLE_PDL: '0' -> '1'" in logs
