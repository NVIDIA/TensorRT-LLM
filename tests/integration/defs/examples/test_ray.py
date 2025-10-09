import os
import subprocess

import pytest
from defs.common import venv_check_call, wait_for_server
from defs.conftest import get_device_count, llm_models_root


@pytest.fixture(scope="module")
def ray_example_root(llm_root):
    example_root = os.path.join(llm_root, "examples", "ray_orchestrator")
    return example_root


def test_llm_inference_async_ray(ray_example_root, llm_venv):
    script_path = os.path.join(ray_example_root, "llm_inference_async_ray.py")
    venv_check_call(llm_venv, [script_path])


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size,pp_size,ep_size", [
    (2, 1, -1),
    (1, 2, -1),
    (2, 2, -1),
    (2, 1, 2),
],
                         ids=["tp2", "pp2", "tp2pp2", "tep2"])
def test_llm_inference_distributed_ray(ray_example_root, llm_venv, tp_size,
                                       pp_size, ep_size):
    world_size = tp_size * pp_size

    if get_device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs.")

    script_path = os.path.join(ray_example_root,
                               "llm_inference_distributed_ray.py")

    cmd = [
        script_path, "--tp_size",
        str(tp_size), "--pp_size",
        str(pp_size), "--moe_ep_size",
        str(ep_size)
    ]

    if ep_size != -1:
        model_dir = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
        cmd.extend(["--model_dir", model_dir])

    venv_check_call(llm_venv, cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2"])
def test_ray_disaggregated_serving(ray_example_root, llm_venv, tp_size):
    if get_device_count() < tp_size * 2:
        pytest.skip(f"Need {tp_size * 2} GPUs.")

    disagg_dir = os.path.join(ray_example_root, "disaggregated")
    script_path = os.path.join(disagg_dir, "disagg_serving_local.sh")

    subprocess.run("ray stop --force", shell=True, check=False)

    proc = subprocess.Popen(
        ["bash", script_path, "--executor", "ray", "--tp_size",
         str(tp_size)],
        cwd=disagg_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert wait_for_server("localhost", 8000, timeout_seconds=180), \
            "Disaggregated server failed to start within 3 minutes"

        result = subprocess.run([
            "curl", "-sS", "-w", "\n%{http_code}",
            "http://localhost:8000/v1/completions", "-H",
            "Content-Type: application/json", "-d",
            '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","prompt":"NVIDIA is a great company because","max_tokens":16,"temperature":0}'
        ],
                                capture_output=True,
                                text=True,
                                timeout=30)

        *body_lines, status_line = result.stdout.strip().splitlines()
        body = "\n".join(body_lines)
        status = int(status_line)

        print("HTTP status:", status)
        print("Response body:", body)

        assert result.returncode == 0, f"curl exit {result.returncode}"
        assert status == 200, f"Expected 200, got {status}"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

        subprocess.run("ray stop --force", shell=True, check=False)
        subprocess.run("pkill -9 -f trtllm-serve", shell=True, check=False)
