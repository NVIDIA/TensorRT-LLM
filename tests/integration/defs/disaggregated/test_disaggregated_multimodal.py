# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path
from typing import Any

import openai
import pytest
import yaml
from defs.conftest import llm_models_root
from disagg_test_utils import terminate
from test_disaggregated import get_ucx_tls, setup_disagg_cluster

_SPLIT_VIDEO_PATH_ENV = "TRTLLM_QWEN3VL_EPD_VIDEO_SMOKE_PATH"
_MODEL_PATH_ENV = "TRTLLM_QWEN3VL_EPD_MODEL_PATH"
_SPLIT_VIDEO_RELATIVE_PATHS = (
    "multimodals/test_data/lsfEHOtYGyk.mp4",
    "videomme-smoke/hf/hub/datasets--lmms-lab--Video-MME/snapshots/"
    "ead1408f75b618502df9a1d8e0950166bf0a2a0b/video/lsfEHOtYGyk.mp4",
)
_SPLIT_VIDEO_PROMPT = (
    "Which player does the video call for to finally put the ball in the basket?\n"
    "A. Player number 2.\n"
    "B. Player number 4.\n"
    "C. Player number 1.\n"
    "D. Player number 3.\n"
    "Answer with only the option letter."
)


def _find_split_video_path(models_root: str) -> Path:
    override = os.environ.get(_SPLIT_VIDEO_PATH_ENV)
    if override:
        path = Path(override)
        if not path.exists():
            pytest.skip(f"{_SPLIT_VIDEO_PATH_ENV}={path} does not exist")
        return path

    for relative_path in _SPLIT_VIDEO_RELATIVE_PATHS:
        path = Path(models_root) / relative_path
        if path.exists():
            return path

    pytest.skip(
        "Qwen3-VL EPD split-video smoke requires lsfEHOtYGyk.mp4. "
        f"Stage it under LLM_MODELS_ROOT or set {_SPLIT_VIDEO_PATH_ENV}."
    )


def _find_model_path(models_root: str, model_relative_path: str) -> Path:
    override = os.environ.get(_MODEL_PATH_ENV)
    if override:
        path = Path(override)
        if not path.exists():
            pytest.skip(f"{_MODEL_PATH_ENV}={path} does not exist")
        return path

    path = Path(models_root) / model_relative_path
    if not path.exists():
        pytest.skip(f"Qwen3-VL model is not staged at {path}")
    return path


def _make_qwen3vl_epd_config(model_path: str) -> dict[str, Any]:
    common_worker_config: dict[str, Any] = {
        "attn_backend": "FLASHINFER",
        "backend": "pytorch",
        "cache_transceiver_config": {
            "backend": "UCX",
            "max_tokens_in_buffer": 2048,
        },
        "cuda_graph_config": None,
        "disable_overlap_scheduler": True,
        "enable_attention_dp": False,
        "kv_cache_config": {
            "dtype": "auto",
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "free_gpu_memory_fraction": 0.6,
        },
        "max_batch_size": 1,
        "max_num_tokens": 4096,
        "max_seq_len": 65536,
        "pipeline_parallel_size": 1,
        "print_iter_log": True,
        "tensor_parallel_size": 1,
    }
    context_config = common_worker_config | {"max_num_tokens": 32768}
    generation_config = common_worker_config | {"max_num_tokens": 4096}

    return {
        "backend": "pytorch",
        "hostname": "localhost",
        "model": model_path,
        "context_servers": {
            "num_instances": 1,
            "router": {
                "type": "round_robin",
            },
            **context_config,
        },
        "generation_servers": {
            "num_instances": 1,
            "router": {
                "type": "round_robin",
            },
            **generation_config,
        },
    }


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(70000)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "model_relative_path", ["Qwen3/Qwen3-VL-8B-Instruct"], ids=["Qwen3-VL-8B-Instruct"]
)
def test_qwen3vl_epd_video_split_item_runs(model_relative_path: str, llm_venv, tmp_path):
    models_root = llm_models_root()
    model_path = _find_model_path(models_root, model_relative_path)
    video_path = _find_split_video_path(models_root)
    config_path = tmp_path / "qwen3vl_epd_video_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(_make_qwen3vl_epd_config(str(model_path)), sort_keys=False),
        encoding="utf-8",
    )

    env = llm_venv._new_env.copy()
    env["TLLM_MULTIMODAL_DISAGGREGATED"] = "1"
    env["UCX_MM_ERROR_HANDLING"] = "y"
    env["UCX_TLS"] = get_ucx_tls()

    ctx_workers, gen_workers, disagg_server, work_dir = [], [], None, None
    try:
        _, ctx_workers, gen_workers, disagg_server, server_port, work_dir = setup_disagg_cluster(
            str(config_path),
            model_name=str(model_path),
            env=env,
            cwd=llm_venv.get_working_directory(),
            server_start_timeout=900,
        )
        client = openai.OpenAI(
            api_key="tensorrt_llm", base_url=f"http://localhost:{server_port}/v1"
        )
        response = client.chat.completions.create(
            model=str(model_path),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _SPLIT_VIDEO_PROMPT,
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": str(video_path),
                            },
                        },
                    ],
                }
            ],
            max_tokens=8,
            temperature=0.0,
        )
        generated_text = response.choices[0].message.content or ""
        assert generated_text.strip()
    finally:
        terminate(*ctx_workers, *gen_workers, disagg_server)
        if work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
