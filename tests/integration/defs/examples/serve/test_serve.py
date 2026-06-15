import os
import queue
import subprocess
import threading
import time

import pytest
import requests
import yaml
from defs.common import get_free_port_in_ci
from defs.conftest import llm_models_root, skip_no_hopper, skip_pre_blackwell
from defs.trt_test_alternative import popen, print_error, print_info
from openai import OpenAI
from requests.exceptions import RequestException


def _wait_for_server_ready(proc, http_port, timeout=7200, sleep_interval=0.5):
    """Wait for server /health to return 200.

    Fails immediately if the server process exits unexpectedly, rather than
    waiting out the full timeout. Mirrors the pattern in openai_server.py.
    """
    url = f"http://0.0.0.0:{http_port}/health"
    start = time.time()
    while True:
        try:
            if requests.get(url, timeout=sleep_interval).status_code == 200:
                break
        except RequestException:
            pass

        result = proc.poll()
        if result is not None and result != 0:
            raise RuntimeError(
                f"trtllm-serve exited unexpectedly with code {result}.")

        if time.time() - start > timeout:
            raise TimeoutError(
                f"trtllm-serve did not become ready within {timeout}s.")

        time.sleep(sleep_interval)


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


@skip_pre_blackwell
def test_nemotron3_super_120b_nvfp4(serve_test_root):
    """Test Nemotron 3 Super 120B NVFP4 with chunked prefill + MTP=3.

    Sends a mix of short and long prompts concurrently to verify that the server
    starts successfully and all requests complete without errors.
    """
    model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    config_file = f"{serve_test_root}/test_configs/Nemotron3_Super_120B_NVFP4.yml"

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(config_file), f"Config not found: {config_file}"

    port = get_free_port_in_ci()
    env = os.environ.copy()
    env["TLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    cmd = [
        "trtllm-serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--max_batch_size",
        "8",
        "--tp_size",
        "1",
        "--ep_size",
        "1",
        "--max_num_tokens",
        "8192",
        "--trust_remote_code",
        "--reasoning_parser",
        "nano-v3",
        "--tool_parser",
        "qwen3_coder",
        "--extra_llm_api_options",
        config_file,
        "--max_seq_len",
        "1048576",
    ]

    model_name = os.path.basename(model_path)
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="tensorrt_llm",
    )

    # Short prompt: exercises MTP decode path.
    short_prompt = "What is the capital of France?"

    # Long prompt: 3000+ words to exceed max_num_tokens (8192 tokens) and
    # force chunked prefill. The content is repeated to guarantee length.
    long_prompt = (
        "Please summarize the following passage in one sentence.\n\n" +
        ("The quick brown fox jumps over the lazy dog. " * 600))

    with popen(cmd, env=env) as proc:
        _wait_for_server_ready(proc, http_port=port, timeout=7200)
        print_info("Server ready — sending mixed short+long prompt batch...")

        # Send all requests in parallel threads so the server batches them.
        results = {}
        errors = {}

        def _complete(label, prompt):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_completion_tokens=1024,
                    stream=False,
                )
                assert len(resp.choices) == 1, \
                    f"[{label}] expected 1 choice, got {len(resp.choices)}"
                results[label] = resp.choices[0].message
            except Exception as exc:
                errors[label] = str(exc)

        threads = []
        # 1 long-prompt request + 3 short-prompt requests sent simultaneously.
        threads.append(
            threading.Thread(name="long_0",
                             target=_complete,
                             args=("long_0", long_prompt),
                             daemon=True))
        for i in range(3):
            threads.append(
                threading.Thread(name=f"short_{i}",
                                 target=_complete,
                                 args=(f"short_{i}", short_prompt),
                                 daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)

        hung = [t.name for t in threads if t.is_alive()]
        assert not hung, f"Timed out waiting for request threads: {hung}"
        assert not errors, f"Requests failed: {errors}"
        assert "long_0" in results, "Long prompt (chunked prefill) got no response"
        assert all(
            f"short_{i}" in results
            for i in range(3)), "One or more short prompts got no response"

        for label, msg in results.items():
            # A valid reasoning-model response must have reasoning_content.
            # content may be empty if the token budget was consumed by thinking,
            # which is expected model behaviour, not a server error.
            assert len(msg.reasoning_content) > 0, \
                f"[{label}] empty reasoning_content — request did not complete"
            print_info(f"[{label}] reasoning: {msg.reasoning_content!r}")
            print_info(f"[{label}] content:   {msg.content!r}")

    print_info("test_nemotron3_super_120b_nvfp4 PASSED")


_NEMOTRON3_NANO_OMNI_MODEL_DIR = "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
# Public stable media URLs for multimodal tests.
_OMNI_IMAGE_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
_OMNI_VIDEO_URL = "https://blogs.nvidia.com/wp-content/uploads/2023/04/nvidia-studio-itns-wk53-scene-in-omniverse-1280w.mp4"
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.fixture(scope="module")
def _nemotron3_nano_omni_server():
    """Start the Nemotron-3-Nano-Omni-30B NVFP4 server once for all omni tests.

    Module-scoped so the server starts once and stays alive across all four
    test cases (text_reasoning_on, text_reasoning_off, image, video),
    avoiding the cost of four separate server startups.
    """
    from defs.conftest import get_llm_root
    llm_root = get_llm_root()
    model_path = f"{llm_models_root()}/{_NEMOTRON3_NANO_OMNI_MODEL_DIR}"
    config_file = (f"{llm_root}/tests/integration/defs/examples/serve/"
                   f"test_configs/Nemotron3_Nano_Omni_30B_NVFP4.yml")

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(config_file), f"Config not found: {config_file}"

    port = get_free_port_in_ci()
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "trtllm-serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust_remote_code",
        "--reasoning_parser",
        "nano-v3",
        "--tool_parser",
        "qwen3_coder",
        "--extra_llm_api_options",
        config_file,
    ]

    with popen(cmd, env=env) as proc:
        _wait_for_server_ready(proc, http_port=port, timeout=7200)
        print_info("Nemotron-3-Nano-Omni server ready.")
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="tensorrt_llm",
        )
        yield client, os.path.basename(model_path)


# See the step-by-step recipe for Nemotron-3-Nano-Omni inference in the official usage guide:
# https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano-Omni/trtllm_cookbook.ipynb
@skip_pre_blackwell
@pytest.mark.parametrize("modality", [
    "text_reasoning_on",
    "text_reasoning_off",
    "text_streaming",
    "tool_calling",
    "image",
    "video",
])
def test_nemotron3_nano_omni_nvfp4(modality, _nemotron3_nano_omni_server):
    """Nemotron-3-Nano-Omni-30B NVFP4 multimodal functional tests.

    Parametrized over six modality cases sharing a single server instance:
      text_reasoning_on  — haiku prompt, thinking + answer both present.
      text_reasoning_off — short prompt, no thinking step.
      text_streaming     — streaming chat completion, token-by-token delivery.
      tool_calling       — function calling via OpenAI tools schema.
      image              — image URL, non-empty description.
      video              — video URL, non-empty description (also exercises
                           audio extraction from the video stream).
    """
    client, model_name = _nemotron3_nano_omni_server

    if modality == "text_reasoning_on":
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Write a haiku about GPUs"
                },
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            stream=False,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert len(msg.reasoning_content) > 0, \
            "empty reasoning_content — thinking step did not run"
        assert len(msg.content) > 0, "empty content — haiku response missing"
        print_info(
            f"[text_reasoning_on] thinking: {msg.reasoning_content[:500]!r}...")
        print_info(f"[text_reasoning_on] content:  {msg.content[:500]!r}...")

    elif modality == "text_reasoning_off":
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Give me 3 bullet points about TensorRT-LLM."
                },
            ],
            temperature=0.2,
            max_completion_tokens=256,
            stream=False,
            extra_body=_NO_THINK,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert len(msg.content) > 0, "empty content — direct answer missing"
        print_info(f"[text_reasoning_off] content: {msg.content[:500]!r}...")

    elif modality == "text_streaming":
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What are the first 5 prime numbers?"
                },
            ],
            temperature=0.6,
            max_completion_tokens=1024,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        full_content = "".join(chunks)
        assert len(chunks) > 1, "expected multiple streamed chunks"
        assert len(full_content) > 0, "empty content — streaming failed"
        print_info(f"[text_streaming] chunks={len(chunks)}, "
                   f"content: {full_content[:500]!r}...")

    elif modality == "tool_calling":
        tools = [{
            "type": "function",
            "function": {
                "name": "calculate_tip",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_total": {
                            "type": "integer",
                            "description": "The total amount of the bill"
                        },
                        "tip_percentage": {
                            "type": "integer",
                            "description": "The percentage of tip to be applied"
                        },
                    },
                    "required": ["bill_total", "tip_percentage"],
                },
            },
        }]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role":
                    "user",
                    "content":
                    "My bill is $50. What will be the amount for 15% tip?"
                },
            ],
            tools=tools,
            temperature=0.6,
            top_p=0.95,
            max_completion_tokens=512,
            stream=False,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert msg.tool_calls is not None and len(msg.tool_calls) > 0, \
            "no tool_calls — function calling failed"
        tool_call = msg.tool_calls[0]
        assert tool_call.function.name == "calculate_tip", \
            f"expected calculate_tip, got {tool_call.function.name}"
        print_info(f"[tool_calling] reasoning: "
                   f"{msg.reasoning_content[:500]!r}...")
        print_info(f"[tool_calling] function: {tool_call.function.name}, "
                   f"args: {tool_call.function.arguments}")

    elif modality == "image":
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what you see in this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _OMNI_IMAGE_URL
                        }
                    },
                ],
            }],
            temperature=0.2,
            max_completion_tokens=512,
            stream=False,
            extra_body=_NO_THINK,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert len(
            msg.content) > 0, "empty content — image understanding failed"
        print_info(f"[image] content: {msg.content[:500]!r}...")

    elif modality == "video":
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what happens in this video."
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": _OMNI_VIDEO_URL
                        }
                    },
                ],
            }],
            temperature=0.2,
            max_completion_tokens=2048,
            stream=False,
            extra_body=_NO_THINK,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert len(
            msg.content) > 0, "empty content — video understanding failed"
        print_info(f"[video] content: {msg.content[:500]!r}...")
