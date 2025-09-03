import os
import time

import requests
from defs.conftest import llm_models_root, skip_pre_hopper
from defs.trt_test_alternative import popen, print_error, print_info
from openai import OpenAI
from requests.exceptions import RequestException


def check_server_ready(http_port="8000", timeout_timer=600, sleep_interval=5):
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


def check_openai_chat_completion(http_port="8000",
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


@skip_pre_hopper
def test_extra_llm_api_options(serve_test_root):
    test_configs_root = f"{serve_test_root}/test_configs"
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

    cmd = [
        "trtllm-serve",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--backend",
        "pytorch",
        "--extra_llm_api_options",
        config_file,
    ]

    print_info("Launching trtllm-serve...")
    with popen(cmd):
        check_server_ready()
        # Extract model name from the model path for consistency
        model_name = model_path.split('/')[-1]  # "Qwen3-30B-A3B-FP8"
        # Test the server with OpenAI chat completion
        check_openai_chat_completion(model_name=model_name)
