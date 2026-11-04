# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-node Mistral Large 4 TEP4 serving with the existing GSM8K accuracy gate."""

import json
import os
import socket
import time
from pathlib import Path

import openai
import pytest
import requests
import test_common.magic_import  # noqa: F401
from mpi4py import MPI
from test_common import cluster_env
from test_common.http_utils import wait_for_endpoint_down, wait_for_endpoint_ready

__extra_import_path__ = ["~/tests/unittest/llmapi/apps"]
from openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

from defs.accuracy.accuracy_core import GSM8K
from defs.accuracy.test_disaggregated_serving import (
    DuckLLM,
    MyThreadPoolExecutor,
    Result,
    run_accuracy_test,
)
from defs.conftest import llm_models_root
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_free_port, get_sm_version
from tensorrt_llm.executor.utils import LlmLauncherEnvs, create_mpi_comm_session
from tensorrt_llm.llmapi import CompletionOutput, RequestOutput, SamplingParams
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size, split_mpi_env
from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer


def _check_serving(url: str, model: str, output: Path) -> None:
    """Exercise sparse context, prefix reuse, then the unchanged full GSM8K gate."""
    for name in ("INTEGRATION_TEST", "TRTLLM_ACCURACY_NO_REFERENCE"):
        assert os.environ.get(name, "0") == "0", f"Disable {name} for baseline validation"
    with (
        openai.OpenAI(
            api_key="1234567890", base_url=f"{url}/v1", timeout=600, max_retries=0
        ) as client,
        MyThreadPoolExecutor(max_workers=16) as pool,
    ):

        def send_request(prompt: str, sampling_params: SamplingParams, streaming: bool):
            assert not streaming and not sampling_params.use_beam_search
            assert sampling_params.guided_decoding is None
            response = client.completions.create(
                model=model,
                prompt=prompt,
                stream=False,
                max_tokens=sampling_params.max_tokens,
                n=sampling_params.n,
                best_of=sampling_params.best_of,
                temperature=sampling_params.temperature if sampling_params.top_p is not None else 0,
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
            request_output = RequestOutput._from_generation_result(result, prompt=prompt)
            setattr(request_output, "result", result.result)
            return request_output

        def generate_async(prompt: str, sampling_params: SamplingParams, streaming: bool = False):
            future = pool.submit(send_request, prompt, sampling_params, streaming)
            pool.futures.append(future)
            return future

        llm = DuckLLM(
            LlmArgs(model=model, checkpoint_format="mistral_large_4"),
            load_hf_tokenizer(model),
            generate_async,
            url,
        )
        llm.args.quant_config.quant_algo = "MIXED_PRECISION"
        llm.args.quant_config.kv_cache_quant_algo = "FP8"
        filler = "The grass is green. The sky is blue. There and back again. "
        prompt = (
            filler * 600
            + "\nRemember this pass key: 71432.\n"
            + filler * 600
            + "\nWhat is the pass key? Answer with just the number:"
        )
        tokens = llm.tokenizer.encode(prompt)
        assert 2048 < len(tokens) < 32700
        # One cold/warm pair also catches loss of the shared latent or indexer history.
        with (output / "requests.jsonl").open("w") as evidence:
            for warm in (False, True):
                response = requests.post(
                    f"{url}/v1/completions",
                    json=dict(model=model, prompt=tokens, temperature=0, max_tokens=16),
                    timeout=600,
                )
                response.raise_for_status()
                result = response.json()
                evidence.write(json.dumps(result) + "\n")
                evidence.flush()
                assert "71432" in result["choices"][0]["text"], result
                usage = result["usage"]
                assert usage["prompt_tokens"] == len(tokens), usage
                assert usage["completion_tokens"] > 1, usage
                cached = usage["prompt_tokens_details"]["cached_tokens"]
                assert cached > 0 if warm else cached == 0, usage
        run_accuracy_test(
            llm,
            "mistralai/Mistral-Large-4-1T-A52B",
            [GSM8K],
            extra_evaluator_kwargs={
                GSM8K: dict(num_fewshot=5, log_samples=True, output_path=str(output / "gsm8k"))
            },
        )


def _worker_info() -> tuple[int, str, int]:
    MPI.COMM_WORLD.Barrier()
    return MPI.COMM_WORLD.rank, socket.gethostname(), get_free_port()


def _serve_worker(
    workers: list[tuple[int, str, int]], model: str, config: dict, output: Path, url: str
) -> dict:
    rank = MPI.COMM_WORLD.rank
    result = {"rank": rank}
    if rank not in (0, 4):
        return result
    role = ServerRole.CONTEXT if rank == 0 else ServerRole.GENERATION
    metrics_dir = output / f"rank{rank}" / "metrics"
    # Keep node-local MPI instances separate from the outer CI worker pool.
    env, _ = split_mpi_env(
        [
            *LlmLauncherEnvs,
            "tllm_mpi_size",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
        ]
    )
    cluster_env.apply_ucx_env(env, cluster_name=os.environ.get("SLURM_CLUSTER_NAME"))
    with RemoteOpenAIServer(
        model,
        host=workers[rank][1],
        port=workers[rank][2],
        rank=0,
        role=role,
        extra_config={**config, "perf_metrics_output_dir": str(metrics_dir)},
        env=env,
        wait=False,
    ) as server:
        wait_for_endpoint_ready(f"{url}/health", timeout=1800, server_proc=server.proc)
        wait_for_endpoint_down(f"{url}/health", timeout=7200)
    if role is ServerRole.GENERATION:
        records = [
            json.loads(line)
            for path in metrics_dir.glob("perf_metrics-generation-*.jsonl")
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == GSM8K.NUM_SAMPLES + 2
        assert all(record["status"] == "complete" for record in records)
        for record in records:
            timing = record["perf_metrics"]["timing_metrics"]
            assert timing["kv_cache_size"] > 0
            assert 0 < timing["kv_cache_transfer_start"] <= timing["kv_cache_transfer_end"]
        result["transfers"] = len(records)
    return result


@pytest.mark.timeout(7200)
def test_mistral_large4_disagg(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    assert get_mpi_world_size() == 8 and MPI.COMM_WORLD.size == 1, (
        "Run under the CI trtllm-llmapi-launch with two nodes and four workers per node"
    )
    assert get_sm_version() == 103
    session = create_mpi_comm_session(8)

    workers = sorted(session.submit_sync(_worker_info))
    assert len(workers) == 8 and [rank for rank, _, _ in workers] == list(range(8)), workers
    hosts = [host for _, host, _ in workers]
    assert len(set(hosts)) == 2 and len(set(hosts[:4])) == len(set(hosts[4:])) == 1, hosts
    model = f"{llm_models_root()}/Mistral-Large-4-1T-A52B-NVFP4"
    output = Path(os.environ.get("TRTLLM_ACCURACY_OUTPUT_DIR", tmp_path))
    output.mkdir(parents=True, exist_ok=True)
    proxy_port = get_free_port()
    url = f"http://{hosts[0]}:{proxy_port}"

    worker_config = dict(
        checkpoint_format="mistral_large_4",
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        enable_attention_dp=False,
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        max_batch_size=16,
        max_num_tokens=32768,
        max_seq_len=32768,
        enable_chunked_prefill=False,
        cuda_graph_config=dict(max_batch_size=16, enable_padding=True),
        kv_cache_config=dict(enable_block_reuse=True, free_gpu_memory_fraction=0.85),
        cache_transceiver_config=dict(backend="NIXL", transceiver_runtime="PYTHON"),
        return_perf_metrics=True,
    )
    with (
        capfd.disabled(),
        RemoteDisaggOpenAIServer(
            ctx_servers=[f"{hosts[0]}:{workers[0][2]}"],
            gen_servers=[f"{hosts[4]}:{workers[4][2]}"],
            port=proxy_port,
            disagg_config={"hostname": hosts[0]},
            wait_ready=False,
            server_start_timeout=1800,
        ) as proxy,
    ):
        session.submit(_serve_worker, workers, model, worker_config, output, url, sync=True)
        try:
            wait_for_endpoint_ready(f"{url}/health", timeout=1800, server_proc=proxy.proc)
            _check_serving(url, model, output)
        finally:
            proxy.terminate()
            deadline = time.monotonic() + 180
            while not (results := session.poll()):
                assert time.monotonic() < deadline, "CI MPI workers did not stop"
                time.sleep(1)
            if isinstance(results, BaseException):
                raise results
            assert len(results) == 8, results
            (output / "handoff.json").write_text(json.dumps(results) + "\n")
    assert len(session.submit_sync(_worker_info)) == 8
