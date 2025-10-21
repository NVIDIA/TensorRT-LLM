import multiprocessing as mp
import time

import pytest
import yaml
from _model_test_utils import get_small_model_config  # type: ignore
from click.testing import CliRunner
from openai import OpenAI

from tensorrt_llm._utils import get_free_port
from tensorrt_llm.commands.serve import main as serve_main


def _run_serve_with_click(args):
    runner = CliRunner()
    # Blocks while server runs
    result = runner.invoke(serve_main, args, catch_exceptions=False)
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


@pytest.mark.timeout(360)
def test_trtllm_serve_openai_chat_completion(tmp_path):
    # Prepare small model config and extra options yaml
    config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    extra_args = config["args"]

    extra_options_path = tmp_path / "extra_llm_api_options.yaml"
    with open(extra_options_path, "w") as f:
        yaml.safe_dump(extra_args, f)

    host = "127.0.0.1"
    port = get_free_port()

    # Use the same `model` string for server and client requests
    model_id = extra_args["model"]

    args = [
        "serve",
        f"{model_id}",
        "--backend",
        "_autodeploy",
        "--host",
        host,
        "--port",
        str(port),
        "--extra_llm_api_options",
        str(extra_options_path),
    ]

    ctx = mp.get_context("spawn")
    server = ctx.Process(target=_run_serve_with_click, args=(args,))
    server.start()

    try:
        # Wait for server to be ready by polling /v1/models via OpenAI client
        client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="tensorrt_llm")

        start_time = time.time()
        last_err = None
        while time.time() - start_time < 90:
            if not server.is_alive():
                raise RuntimeError("Server process exited prematurely")
            try:
                # Lightweight readiness probe
                _ = client.models.list()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1)
        else:
            raise TimeoutError(f"Server did not become ready in time: {last_err}")

        # Send a small chat completion request
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "Say 'ok'"},
            ],
            max_tokens=8,
        )

        # print response
        print(f"{resp=}")

        assert hasattr(resp, "choices") and len(resp.choices) > 0
        first = resp.choices[0]
        # new OpenAI client returns .message for chat completions
        assert getattr(first, "message", None) is not None
        # Content may be a string or a structured list depending on client version
        _ = getattr(first.message, "content", None)

    finally:
        # Terminate server and clean up
        if server.is_alive():
            server.terminate()
            server.join(timeout=20)
        if server.is_alive():
            server.kill()
            server.join(timeout=20)
