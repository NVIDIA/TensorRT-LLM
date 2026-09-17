import os
import time

from .trt_test_alternative import check_call, check_output, print_info


def check_server_ready(http_port="8000", timeout_timer=None, sleep_interval=5):
    env_timeout = int(os.getenv('TRITON_SERVER_LAUNCH_TIMEOUT', '300'))
    if timeout_timer is None:
        timeout = env_timeout
    else:
        timeout = max(timeout_timer, env_timeout)
    timer = 0
    while True:
        if http_port == "8000":
            status = check_output(
                r"curl -s -w %{http_code} 0.0.0.0:8000/v2/health/ready || true",
                shell=True).strip()
        elif http_port == "8003":
            status = check_output(
                r"curl -s -w %{http_code} 0.0.0.0:8003/v2/health/ready || true",
                shell=True).strip()
        if status == "200":
            break
        elif timer <= timeout:
            time.sleep(sleep_interval)
            timer += sleep_interval
        elif timer > timeout:
            raise TimeoutError(
                f"Error: Launch Triton server timed out, timer is {timeout} seconds."
            )

    print_info(
        f"Triton server launched successfully! Cost {timer} seconds to launch server."
    )


def prepare_llmapi_model_repo(llm_backend_repo_root, new_model_repo):
    origin_model_repo = os.path.join(llm_backend_repo_root, "all_models",
                                     "llmapi")
    check_call(f"rm -rf {new_model_repo}", shell=True)
    check_call(f"cp -R {origin_model_repo} {new_model_repo}", shell=True)


def set_llmapi_decoupled_mode(new_model_repo, decoupled):
    """Set Triton transaction policy via config.pbtxt for the llmapi model.

    Append `model_transaction_policy { decoupled: ... }` so Triton sees the
    model as decoupled (or not). Required because launch_triton_server.py
    uses `--disable-auto-complete-config`, which prevents the llmapi
    model.py `auto_complete_config` callback (where decoupled is taken from
    model.yaml) from running. Without this, Triton treats the model as
    non-decoupled and rejects multi-response sends from the streaming path.
    """
    config_pbtxt = os.path.join(new_model_repo, "tensorrt_llm", "config.pbtxt")
    with open(config_pbtxt, "a") as f:
        f.write("\nmodel_transaction_policy {\n"
                f"  decoupled: {str(bool(decoupled)).lower()}\n"
                "}\n")
