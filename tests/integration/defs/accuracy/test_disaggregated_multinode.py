# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import re
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Iterator, Optional

import openai
import pytest
import requests

from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, LlmArgs
from tensorrt_llm.llmapi.tokenizer import DeepseekV4Tokenizer
from tests.unittest.llmapi.apps.openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

from ..conftest import llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .test_disaggregated_serving import DuckLLM, MyThreadPoolExecutor, Result, run_accuracy_test

# Run from the repository root without installing the repository into the image:
#
#   TRTLLM_TEST_ROOT=/lustre/fsw/coreai_comparch_trtllm/lizhiz/tllm_dsv4
#   TRTLLM_IMAGE_ROOT=/lustre/share/coreai_comparch_trtllm/lizhiz/rubin_dsv4
#   TRTLLM_TEST_IMAGE="${TRTLLM_IMAGE_ROOT}/b8bc42c3ae/trtllm.sqsh"
#   TRTLLM_TEST_LOG="${TRTLLM_TEST_ROOT}/build_images/b8bc42c3ae/test_logs"
#   mkdir -p "${TRTLLM_TEST_LOG}"
#   srun \
#     --job-name=coreai_comparch_aarwlt-dsv4.disagg-acc \
#     --partition=batch-xdr \
#     --account=coreai_comparch_aarwlt \
#     --nodes=2 \
#     --ntasks=2 \
#     --ntasks-per-node=1 \
#     --time=04:00:00 \
#     --output="${TRTLLM_TEST_LOG}/dsv4_disagg_%j.log" \
#     --error="${TRTLLM_TEST_LOG}/dsv4_disagg_%j.log" \
#     --container-image="${TRTLLM_TEST_IMAGE}" \
#     --container-mounts=/lustre:/lustre \
#     --container-workdir="${TRTLLM_TEST_ROOT}" \
#     bash -lc '\
#   export LLM_MODELS_ROOT=/lustre/fsw/coreai_comparch_trtllm/common/llm-models; \
#   export HF_HOME=/lustre/fsw/coreai_comparch_trtllm/lizhiz/hf_cache; \
#   export TLLM_LOG_LEVEL=INFO; export PYTHONPATH=; \
#   unset TRTLLM_MOE_A2A_DISABLE_CFT_COUNTED_WRITES; \
#   TEST_FILE=tests/integration/defs/accuracy/test_disaggregated_multinode.py; \
#   python3 -m pytest -q -s \
#   "${TEST_FILE}::TestDeepSeekV4ProDSparkMultinode::test_gsm8k_1p1d_dep4"'
#
# Slurm node rank 0 hosts the context worker and disaggregated frontend. Node
# rank 1 hosts the generation worker. The test waits for both workers before
# starting the frontend because DSpark makes generation startup substantially
# slower than context startup.


def _expand_slurm_nodelist(nodelist: str) -> list[str]:
    if not nodelist:
        return []

    groups = []
    group_chars = []
    bracket_depth = 0
    for char in nodelist:
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1

        if char == "," and bracket_depth == 0:
            groups.append("".join(group_chars))
            group_chars = []
        else:
            group_chars.append(char)
    groups.append("".join(group_chars))

    nodes = []
    for group in groups:
        match = re.fullmatch(r"(.+?)\[(.+)]", group)
        if match is None:
            nodes.append(group)
            continue

        prefix, suffixes = match.groups()
        for suffix in suffixes.split(","):
            range_match = re.fullmatch(r"(\d+)-(\d+)", suffix)
            if range_match is None:
                nodes.append(f"{prefix}{suffix}")
                continue

            start_text, end_text = range_match.groups()
            width = len(start_text)
            nodes.extend(
                f"{prefix}{value:0{width}d}" for value in range(int(start_text), int(end_text) + 1)
            )
    return nodes


def _wait_for_endpoint_ready(
    url: str,
    timeout: int,
    interval: int = 3,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Endpoint {url} was not ready within {timeout} seconds")


def _wait_for_endpoint_down(
    url: str,
    timeout: int,
    interval: int = 1,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(url, timeout=10)
        except requests.RequestException:
            return
        time.sleep(interval)
    raise TimeoutError(f"Endpoint {url} remained up after {timeout} seconds")


NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = _expand_slurm_nodelist(os.environ.get("SLURM_NODELIST", ""))
SLURM_NTASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

CTX_SERVER_PORT = 8001
GEN_SERVER_PORT = 8002
DISAGG_SERVER_PORT = 8000
SERVER_START_TIMEOUT = 7200

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"


MODEL_PATH = str(Path(llm_models_root()) / "DeepSeek-V4-Pro-DSpark")

EXTRA_EVALUATOR_KWARGS = {
    "apply_chat_template": True,
    "system_prompt": (
        "Solve the problem carefully. End your response with a final line "
        "exactly in the form #### <answer>, using the simplest numeric form "
        "without units or trailing zeros."
    ),
}


def _require_two_node_allocation() -> None:
    if len(NODE_LIST) != 2:
        pytest.skip("This test requires exactly two Slurm nodes")
    if SLURM_NTASKS_PER_NODE != 1:
        pytest.skip("This test requires one pytest task per Slurm node")


def _is_context_node() -> bool:
    return NODE_RANK == 0


def _is_generation_node() -> bool:
    return NODE_RANK == 1


def _worker_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_"))
        and key
        not in {
            "MASTER_ADDR",
            "MASTER_PORT",
            "UCX_TLS",
            "UCX_NET_DEVICES",
        }
    }
    # Pyxis starts these worker processes as root; preserve the explicit
    # Open MPI container override for mpi4py dynamic worker spawn.
    env.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET": "1073741824",
            # Avoid auto-selecting the NVL72 RDMA VF, whose container-visible
            # IPv6 address cannot be bound by UCX on this cluster.
            "UCX_NET_DEVICES": os.environ.get("UCX_NET_DEVICES", "eth0"),
            "UCX_TLS": "tcp,self,sm,cuda_copy,cuda_ipc",
        }
    )
    return env


def _worker_config(enable_dspark: bool) -> dict[str, object]:
    config: dict[str, object] = {
        "attn_backend": "TRTLLM",
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "enable_attention_dp": True,
        "moe_config": {
            # DeepSeek-V4 routed experts use NVFP4 in this test. Use the
            # MegaMoE CuTe DSL path for the NVFP4 checkpoint.
            "backend": "MEGAMOE_CUTEDSL",
        },
        # Keep the DEP4 worker envelope aligned with the aggregate guard. A
        # batch-128 DSpark graph leaves too little headroom for the post-capture
        # 4096-token warmup on rank 0.
        "max_batch_size": 64,
        "cuda_graph_config": {"max_batch_size": 64},
        "max_seq_len": 4096,
        "max_num_tokens": 4096,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.5,
        },
        "enable_chunked_prefill": False,
        "disable_overlap_scheduler": True,
        "enable_iter_perf_stats": True,
        "print_iter_log": True,
        "custom_tokenizer": "deepseek_v4",
        "cache_transceiver_config": {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": 4096,
        },
    }
    if enable_dspark:
        config["speculative_config"] = {
            "decoding_type": "DSpark",
            "max_draft_len": 5,
            "speculative_model": MODEL_PATH,
        }
    return config


@pytest.fixture(scope="module")
def worker() -> Iterator[Optional[RemoteOpenAIServer]]:
    _require_two_node_allocation()
    if _is_context_node():
        port = CTX_SERVER_PORT
    elif _is_generation_node():
        port = GEN_SERVER_PORT
    else:
        yield None
        return

    with RemoteOpenAIServer(
        MODEL_PATH,
        port=port,
        cli_args=["--tp_size", "4", "--pp_size", "1"],
        host="0.0.0.0",
        env=_worker_env(),
        llmapi_launch=False,
        rank=0,
        extra_config=_worker_config(enable_dspark=_is_generation_node()),
    ) as server:
        yield server


@pytest.fixture(scope="module")
def disagg_server(
    worker: Optional[RemoteOpenAIServer],
) -> Iterator[Optional[RemoteDisaggOpenAIServer]]:
    del worker
    if _is_context_node():
        _wait_for_endpoint_ready(
            f"http://{NODE_LIST[1]}:{GEN_SERVER_PORT}/health",
            timeout=SERVER_START_TIMEOUT,
        )
        with RemoteDisaggOpenAIServer(
            ctx_servers=[f"{NODE_LIST[0]}:{CTX_SERVER_PORT}"],
            gen_servers=[f"{NODE_LIST[1]}:{GEN_SERVER_PORT}"],
            port=DISAGG_SERVER_PORT,
            llmapi_launch=False,
            env=_worker_env(),
        ) as server:
            yield server
    else:
        yield None


@contextlib.contextmanager
def _accuracy_llm(
    server: RemoteDisaggOpenAIServer,
) -> Iterator[DuckLLM]:
    client = openai.OpenAI(
        api_key=RemoteOpenAIServer.DUMMY_API_KEY,
        base_url=server.url_for("v1"),
        timeout=1_800_000,
    )
    args = LlmArgs(model=MODEL_PATH)
    args.quant_config.quant_algo = "FP8_BLOCK_SCALES"
    args.speculative_config = DSparkDecodingConfig(
        max_draft_len=5,
        speculative_model=MODEL_PATH,
    )
    tokenizer = DeepseekV4Tokenizer.from_pretrained(MODEL_PATH)

    with MyThreadPoolExecutor(max_workers=128) as thread_pool:

        def send_request(
            prompt: str,
            sampling_params: Optional[SamplingParams],
            streaming: bool,
        ) -> RequestOutput:
            if sampling_params is None:
                sampling_params = SamplingParams()
            response = client.completions.create(
                model=MODEL_PATH,
                prompt=prompt,
                stream=streaming,
                max_tokens=sampling_params.max_tokens,
                n=sampling_params.n,
                temperature=(
                    sampling_params.temperature if sampling_params.top_p is not None else 0
                ),
                top_p=sampling_params.top_p,
                stop=sampling_params.stop,
                seed=sampling_params.seed,
            )
            result = Result(
                id=0,
                sampling_params=sampling_params,
                outputs=[
                    CompletionOutput(text=choice.text, index=index)
                    for index, choice in enumerate(response.choices)
                ],
            )
            output = RequestOutput._from_generation_result(result, prompt=prompt)
            setattr(output, "result", result.result)
            return output

        def generate_async(
            prompt: str,
            sampling_params: Optional[SamplingParams] = None,
            streaming: bool = False,
        ) -> Future[RequestOutput]:
            future = thread_pool.submit(send_request, prompt, sampling_params, streaming)
            thread_pool.futures.append(future)
            return future

        yield DuckLLM(args, tokenizer, generate_async)


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device_memory(140000)
@skip_pre_blackwell
class TestDeepSeekV4ProDSparkMultinode(LlmapiAccuracyTestHarness):
    def test_gsm8k_1p1d_dep4(
        self,
        disagg_server: Optional[RemoteDisaggOpenAIServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        health_url = f"http://{NODE_LIST[0]}:{DISAGG_SERVER_PORT}/health/"
        if _is_context_node():
            assert disagg_server is not None
            monkeypatch.setenv("INTEGRATION_TEST", "0")
            with _accuracy_llm(disagg_server) as llm:
                run_accuracy_test(
                    llm,
                    MODEL_NAME,
                    ["GSM8K"],
                    extra_evaluator_kwargs={GSM8K: EXTRA_EVALUATOR_KWARGS},
                )
            disagg_server.terminate()
        elif _is_generation_node():
            _wait_for_endpoint_ready(
                health_url,
                timeout=SERVER_START_TIMEOUT,
            )
            _wait_for_endpoint_down(
                health_url,
                timeout=14400,
            )
        else:
            raise AssertionError(f"Unexpected Slurm node rank {NODE_RANK}")
