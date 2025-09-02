import json
import os
import signal
import subprocess
import sys
import tempfile
from typing import Callable
import threading

import os
import signal
import subprocess
import time

import pytest
import requests
import yaml
from defs.conftest import get_sm_version, llm_models_root, skip_arm, skip_no_hopper
from defs.trt_test_alternative import check_call, check_output, popen

from tensorrt_llm.logger import logger

import time
import socket
import requests
from requests.exceptions import RequestException, JSONDecodeError


def check_server_ready(http_port="8000", timeout_timer=300, sleep_interval=5):

    timeout = timeout_timer
    timer = 0
    while timer <= timeout:
        try:
            url = f"http://0.0.0.0:{http_port}/health"
            r = requests.get(url, timeout=sleep_interval)
            if r.status_code == 200:
                # trtllm-serve health endpoint just returns status 200, no JSON body required
                break
        except (RequestException, JSONDecodeError):
            pass

        # 无论成功还是失败，都要更新timer并检查超时
        time.sleep(sleep_interval)
        timer += sleep_interval

    # 如果循环结束但没有break，说明超时了
    if timer > timeout:
        raise TimeoutError(
            f"Error: Launch trtllm-serve timed out, timer is {timeout} seconds."
        )

    logger.info(
        f"trtllm-serve launched successfully! Cost {timer} seconds to launch server."
    )


def test_torch_compile_enabled(serve_test_root):
    test_configs_root = f"{serve_test_root}/test_configs"
    config_file = f"{test_configs_root}/rcca_5340941.yml"
    model_path = f"{llm_models_root()}/Qwen3/Qwen3-30B-A3B-FP8"

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

    logger.info("Launching trtllm-serve...")
    # 保存进程对象以便后续管理，所有输出直接显示到控制台
    process = subprocess.Popen(cmd)

    try:
        check_server_ready()
        logger.info(f"trtllm-serve is ready")
        # 这里可以添加你的测试代码

    finally:
        # 确保进程被正确清理
        logger.info("Cleaning up server process...")
        try:
            process.terminate()
            process.wait(timeout=10)  # 等待最多10秒
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate gracefully, killing it...")
            process.kill()
            process.wait()
