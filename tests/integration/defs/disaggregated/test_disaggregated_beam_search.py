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
"""Beam-search parity across independent NIXL context and generation workers."""

from __future__ import annotations

import multiprocessing
import os
from contextlib import ExitStack, contextmanager
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest

if TYPE_CHECKING:
    from tensorrt_llm import DisaggregatedParams

_NEW_TOKENS = 16
_TOKENS_PER_BLOCK = 32
_WORKER_TIMEOUT = 300


def _run_worker(
    connection: Connection,
    gpu: str,
    model: str,
    disaggregated: bool,
    beam_width: int,
    enable_block_reuse: bool,
) -> None:
    # Each child owns one standalone MPI world. Pipes carry the test's control
    # messages; MPI publish/lookup is not supported by the Open MPI 5 CI image.
    for name in list(os.environ):
        if name.startswith(("OMPI_", "PMIX_", "PMI_")):
            del os.environ[name]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    os.environ["TLLM_KV_CACHE_MANAGER_V2_BACKEND"] = "cpp"
    os.environ.setdefault("UCX_TLS", "^ib,gdr_copy")

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi import CacheTransceiverConfig, KvCacheConfig

    transceiver = (
        CacheTransceiverConfig(transceiver_runtime="PYTHON", backend="NIXL")
        if disaggregated
        else None
    )
    with (
        connection,
        LLM(
            model=model,
            backend="pytorch",
            max_batch_size=1,
            max_beam_width=beam_width,
            max_seq_len=256,
            max_num_tokens=256,
            enable_chunked_prefill=False,
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
            kv_cache_config=KvCacheConfig(
                use_kv_cache_manager_v2=True,
                enable_block_reuse=enable_block_reuse,
                tokens_per_block=_TOKENS_PER_BLOCK,
                max_tokens=4096,
            ),
            cache_transceiver_config=transceiver,
        ) as llm,
    ):
        manager = llm._executor.engine.kv_cache_manager
        assert isinstance(manager, KVCacheManagerV2), type(manager).__name__
        connection.send("ready")
        sampling = SamplingParams(
            max_tokens=_NEW_TOKENS,
            n=beam_width,
            use_beam_search=True,
            ignore_eos=True,
        )
        while (request := connection.recv()) is not None:
            prompt, params = request
            result = llm.generate(prompt, sampling, use_tqdm=False, disaggregated_params=params)
            connection.send(
                (
                    [list(output.token_ids) for output in result.outputs],
                    result.cached_tokens,
                    result.outputs[0].disaggregated_params,
                )
            )


def _receive(
    connection: Connection, worker: BaseProcess
) -> str | tuple[list[list[int]], int, DisaggregatedParams | None]:
    assert connection.poll(_WORKER_TIMEOUT), (
        f"Worker {worker.name} timed out; exitcode={worker.exitcode}"
    )
    try:
        return connection.recv()
    except EOFError as error:
        worker.join(timeout=5)
        raise AssertionError(
            f"Worker {worker.name} exited with code {worker.exitcode}; see its traceback"
        ) from error


@contextmanager
def _worker(
    gpu: str,
    model: str,
    beam_width: int,
    enable_block_reuse: bool,
    *,
    disaggregated: bool,
) -> Iterator[tuple[Connection, BaseProcess]]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_run_worker,
        args=(child, gpu, model, disaggregated, beam_width, enable_block_reuse),
        name=f"beam-{'disagg' if disaggregated else 'aggregate'}-gpu-{gpu}",
    )
    process.start()
    child.close()
    try:
        assert _receive(parent, process) == "ready"
        yield parent, process
    finally:
        if process.is_alive():
            try:
                parent.send(None)
            except (BrokenPipeError, ConnectionResetError):
                pass
            process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        parent.close()
        process.close()


def _generate(
    worker: tuple[Connection, BaseProcess],
    prompt: list[int],
    params: DisaggregatedParams | None = None,
) -> tuple[list[list[int]], int, DisaggregatedParams | None]:
    connection, process = worker
    connection.send((prompt, params))
    result = _receive(connection, process)
    assert isinstance(result, tuple)
    return result


@pytest.mark.parametrize(
    ("beam_width", "enable_block_reuse"),
    [(2, True), (4, False)],
    ids=["beam2-reuse", "beam4-no-reuse"],
)
def test_disaggregated_beam_search_v2(beam_width: int, enable_block_reuse: bool) -> None:
    """Preserve every beam across aligned, partial-block, and repeated transfers."""
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import DisaggregatedParams

    assert torch.cuda.device_count() >= 2, "This regression requires two GPUs"
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible_gpus.split(",") if visible_gpus else ["0", "1"]
    model = str(Path(os.environ["LLM_MODELS_ROOT"]) / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [
        tokenizer.encode("Explain how plants use sunlight to produce energy. " * 12)[:64],
        tokenizer.encode("Describe how a telescope observes distant stars and galaxies. " * 12)[
            :67
        ],
    ]
    assert [len(prompt) for prompt in prompts] == [64, 67]

    # Keep the reference separate so both transfer peers begin with cold caches.
    with _worker(gpus[0], model, beam_width, enable_block_reuse, disaggregated=False) as aggregate:
        expected = [_generate(aggregate, prompt)[0] for prompt in prompts]
    assert all(len(beams) == beam_width for beams in expected)
    assert all(len(beam) == _NEW_TOKENS for beams in expected for beam in beams)

    with ExitStack() as stack:
        context = stack.enter_context(
            _worker(gpus[0], model, beam_width, enable_block_reuse, disaggregated=True)
        )
        generation = stack.enter_context(
            _worker(gpus[1], model, beam_width, enable_block_reuse, disaggregated=True)
        )
        for repeat in range(2):
            for prompt, reference in zip(prompts, expected):
                _, cached_tokens, params = _generate(
                    context, prompt, DisaggregatedParams(request_type="context_only")
                )
                assert params is not None
                assert params.request_type == "context_only"
                if repeat and enable_block_reuse:
                    assert cached_tokens >= _TOKENS_PER_BLOCK, "Repeated prefill did not reuse KV"
                params.request_type = "generation_only"
                actual, generation_cached_tokens, _ = _generate(generation, prompt, params)
                # Generation cached_tokens includes the transferred prompt even
                # on a cold cache, so only prefill reports local reuse here.
                if not enable_block_reuse:
                    assert cached_tokens == 0
                assert actual == reference, (
                    f"prompt_length={len(prompt)}, repeat={repeat}: "
                    f"disaggregated beams {actual} != aggregated beams {reference}"
                )
                print(
                    f"beam_width={beam_width}, prompt_length={len(prompt)}, repeat={repeat}: "
                    f"all beams match; cached_tokens context={cached_tokens}, "
                    f"generation={generation_cached_tokens}",
                    flush=True,
                )
