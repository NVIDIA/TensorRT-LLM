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
"""Two-node disaggregated GSM8K accuracy for DeepSeek-V4-Pro with DSpark.

Needs a Slurm allocation of exactly two nodes with one pytest task per node
(--nodes=2 --ntasks-per-node=1), four GPUs per node and LLM_MODELS_ROOT
holding DeepSeek-V4-Pro-nvfp4-DSpark. Run the same pytest command on both
nodes, for example::

    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 <container options> \
        pytest accuracy/test_disaggregated_multinode.py::\
TestDeepSeekV4ProDSparkMultinode::test_gsm8k_1p1d_dep4

Slurm node rank 0 hosts the context worker and the disaggregated frontend.
Node rank 1 hosts the generation worker. The frontend starts only after both
workers are healthy, because DSpark makes generation startup substantially
slower than context startup.
"""

import contextlib
import json
import os
import re
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Iterator, Optional

import openai
import pytest
import requests
from test_common.cluster_env import apply_ucx_env

from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, LlmArgs
from tensorrt_llm.llmapi.tokenizer import DeepseekV4Tokenizer

from ..conftest import get_device_memory, llm_models_root, skip_pre_blackwell
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from .test_disaggregated_serving import DuckLLM, MyThreadPoolExecutor, Result, run_accuracy_test

# The OpenAI server launchers live in the unit-test tree.
__extra_import_path__ = ["~/tests/unittest/llmapi/apps"]
from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer  # noqa: E402


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
    required_failures: int = 10,
) -> None:
    """Return once `url` has failed `required_failures` probes in a row.

    One missed probe is not a shutdown: the frontend can be slow to answer while
    the accuracy run keeps it busy.
    """
    deadline = time.monotonic() + timeout
    consecutive_failures = 0
    while time.monotonic() < deadline:
        try:
            requests.get(url, timeout=10)
            consecutive_failures = 0
        except requests.RequestException:
            consecutive_failures += 1
            if consecutive_failures >= required_failures:
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
# NVFP4 export (nvidia/deepseek-v4-pro-nvfp4-dspark) with the embedded DSpark
# drafter; MegaMoE CuTe DSL is the backend that serves it.
MODEL_PATH = str(Path(llm_models_root()) / "DeepSeek-V4-Pro-nvfp4-DSpark")

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
        }
    }
    # Pyxis runs as root; dynamic MPI spawn needs these explicit overrides.
    env.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET": "1073741824",
        }
    )
    apply_ucx_env(
        env,
        cluster_name=os.environ.get("SLURM_CLUSTER_NAME", ""),
    )
    return env


def _worker_config(enable_dspark: bool) -> dict[str, object]:
    moe_tactic = "cutedsl.mega_moe"
    config: dict[str, object] = {
        "attn_backend": "TRTLLM",
        "tensor_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "enable_attention_dp": True,
        "moe_config": {
            # DeepSeek-V4 routed experts use NVFP4 in this test. Use the
            # MegaMoE CuTe DSL path for the NVFP4 checkpoint.
            "tactic": moe_tactic,
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
            # A target tactic does not reach the drafter; pin it to the same
            # implementation.
            "moe_tactic": moe_tactic,
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
        # Seconds per request attempt; with the client's default two retries a
        # stalled request still ends well inside the test timeout.
        timeout=3600,
    )
    args = LlmArgs(model=MODEL_PATH)
    # Selects the GSM8K reference row; the NVFP4 export declares
    # MIXED_PRECISION (NVFP4 routed experts, everything else unquantized).
    args.quant_config.quant_algo = "MIXED_PRECISION"
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
@pytest.mark.skip_less_device(4)
@pytest.mark.skipif(
    get_device_memory() < 250000,
    reason="This test requires at least 250000 MiB of memory per GPU",
)
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


@pytest.mark.timeout(5400)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["nvfp4"])
def test_nemotron_shortcut_tp4_handoff(precision, record_property) -> None:
    """The GDP state and sparse index pages must follow the main KV to generation."""
    _check_shortcut_handoff(record_property, precision, use_mtp=False)


@pytest.mark.timeout(5400)
@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
def test_nemotron_shortcut_tp4_prefix_mtp_handoff(precision, record_property) -> None:
    """A warm P-side prefix resumes on an overlapping D-side MTP worker."""
    _check_shortcut_handoff(record_property, precision, use_mtp=True)


def _check_shortcut_handoff(record_property, precision: str, *, use_mtp: bool) -> None:
    from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
    from tensorrt_llm._utils import get_sm_version
    from tensorrt_llm.llmapi.disagg_utils import ServerRole
    from tensorrt_llm.serve.perf_metrics import build_metrics_record_from_headers

    _require_two_node_allocation()
    if get_sm_version() != 103:
        pytest.skip("This handoff test requires four GB300 GPUs per worker")
    model = str(Path(llm_models_root()) / "nemotron_shortcut" / precision)
    index_topk = json.loads((Path(model) / "config.json").read_text())["text_config"]["index_topk"]
    env = {**_worker_env(), MODELING_V2_VALIDATE_ENV: "1"}
    options = {
        "modeling_v2": "require",
        "disable_mm_encoder": True,
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "moe_tensor_parallel_size": 2,
        "moe_expert_parallel_size": 2,
        "enable_attention_dp": False,
        "disable_overlap_scheduler": False,
        "max_num_tokens": 4096,
        "max_seq_len": 4096,
        "max_batch_size": 1,
        "enable_chunked_prefill": False,
        "cuda_graph_config": None,
        "kv_cache_config": {
            "use_kv_cache_manager_v2": True,
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.5,
        },
    }
    if use_mtp:
        options.update(
            speculative_config={"decoding_type": "MTP", "max_draft_len": 2},
            per_request_spec_decode_stats=True,
        )
        options["kv_cache_config"].update(
            enable_block_reuse=True,
            mamba_state_config={"additional_snapshot_offsets_from_end": [1]},
        )
    worker_args = {"host": "0.0.0.0", "env": env, "llmapi_launch": False, "rank": 0, "wait": False}
    prompts = (
        "The capital of France is",
        "Read this record and complete the last line.\n"
        + "Entry: cobalt.\n" * 560
        + "The repeated entry is",
    )
    if use_mtp:
        prompts += (prompts[-1],)

    def complete(server, prompt):
        response = requests.post(
            server.url_for("v1/completions"),
            timeout=180,
            headers={"X-TRTLLM-return-metrics": "1"},
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "top_k": 1,
                "seed": 0,
                "max_tokens": 16,
                "detokenize": False,
            },
        )
        response.raise_for_status()
        ids = response.json()["choices"][0]["token_ids"]
        assert len(ids) > 1, response.text
        return ids, response

    def check_cached_tokens(n, role, response):
        cached = response.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
        record_property(f"prompt_{n}_{role}_cached", cached)
        if n == 1:
            assert cached == 0, f"Expected a cold prefix on {role}"
        elif n == 2:
            assert cached > index_topk, f"Expected a restored sparse prefix on {role}"
        return cached

    expected = []
    if _is_context_node():
        with RemoteOpenAIServer(
            model, port=CTX_SERVER_PORT, extra_config=options, **worker_args
        ) as agg:
            agg.wait_for_server(timeout=SERVER_START_TIMEOUT)
            for n, prompt in enumerate(prompts):
                ids, response = complete(agg, prompt)
                expected.append(ids)
                if use_mtp:
                    check_cached_tokens(n, "aggregate", response)
    options.update(
        return_perf_metrics=True,
        cache_transceiver_config={"backend": "NIXL", "transceiver_runtime": "PYTHON"},
    )
    port = CTX_SERVER_PORT if _is_context_node() else GEN_SERVER_PORT
    role = ServerRole.CONTEXT if _is_context_node() else ServerRole.GENERATION
    if use_mtp and role == ServerRole.GENERATION:
        options["kv_cache_config"]["enable_block_reuse"] = False
        options["kv_cache_config"].pop("mamba_state_config")
    with RemoteOpenAIServer(
        model, port=port, extra_config=options, role=role, **worker_args
    ) as worker:
        worker.wait_for_server(timeout=SERVER_START_TIMEOUT)
        if _is_generation_node():
            url = f"http://{NODE_LIST[0]}:{DISAGG_SERVER_PORT}/health"
            _wait_for_endpoint_ready(url, timeout=SERVER_START_TIMEOUT, interval=1)
            _wait_for_endpoint_down(url, timeout=SERVER_START_TIMEOUT)
            return
        _wait_for_endpoint_ready(
            f"http://{NODE_LIST[1]}:{GEN_SERVER_PORT}/health", timeout=SERVER_START_TIMEOUT
        )
        with RemoteDisaggOpenAIServer(
            [f"{NODE_LIST[0]}:{CTX_SERVER_PORT}"],
            [f"{NODE_LIST[1]}:{GEN_SERVER_PORT}"],
            port=DISAGG_SERVER_PORT,
            env=env,
            disagg_config={"return_perf_metrics": True},
        ) as pd:
            actual = []
            for n, (prompt, reference) in enumerate(zip(prompts, expected, strict=True)):
                ids, response = complete(pd, prompt)
                actual.append(ids)
                record = build_metrics_record_from_headers(response.headers, "gen")
                assert record, response.headers
                timing = record["phases"]["gen"]["timing_metrics"]
                start, end = timing["kv_cache_transfer_start"], timing["kv_cache_transfer_end"]
                assert 0 < start < end, timing
                length = response.json()["usage"]["prompt_tokens"]
                record_property(f"prompt_{n}_tokens", length)
                record_property(f"prompt_{n}_aggregate", reference)
                record_property(f"prompt_{n}_disaggregated", ids)
                record_property(f"prompt_{n}_kv_transfer", (start, end))
                if n == 1:
                    assert length > index_topk, (
                        "The second prompt must exercise sparse index selection"
                    )
                if use_mtp:
                    stats = response.json()["choices"][0]["speculative_decoding"]
                    record_property(f"prompt_{n}_draft", stats)
                    assert 0 < stats["total_accepted_draft_tokens"] <= stats["total_draft_tokens"]
                    hit = check_cached_tokens(n, "context", response)
                    if n == 2:
                        record_property("warm_prefix_tokens", hit)
            assert actual == expected
            if use_mtp:
                assert actual[1] == actual[2], "Restoring the prefix changed the continuation"
