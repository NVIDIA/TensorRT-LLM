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

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import SpecMetadata
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.postproc_worker import PostprocWorkerConfig
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig, RequestOutput
from tensorrt_llm.llmapi.mpi_session import MpiSession
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from ..conftest import llm_models_root

MODEL = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
SPEC_MODEL = f"{llm_models_root()}/Qwen3/Qwen3-8B"
EAGLE3_MODEL = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"
PROMPT_TOKEN_IDS = [1] + [42] * 63 + [43]
CHANGED_FINAL_PROMPT_TOKEN_IDS = PROMPT_TOKEN_IDS[:-1] + [44]
SPEC_PROMPT_TOKEN_IDS = (
    [1] + [42] * 63 + [43],
    [1] + [44] * 63 + [45],
)
_TP_GRAPH_PROBE_DIR_ENV = "TLLM_FINAL_CONTEXT_CUDA_GRAPH_PROBE_DIR"


class _PromotedContextGraphExecutionReport(TypedDict):
    promoted_context_request_count: int
    graph_batch_size: int | None
    enable_spec_decode: bool
    replayed: bool


class _RankGraphExecutionReport(TypedDict):
    rank: int
    executions: list[_PromotedContextGraphExecutionReport]


@dataclass
class _PromotedContextGraphExecution:
    promoted_context_request_ids: frozenset[int]
    key: KeyType | None
    enable_spec_decode: bool
    replayed: bool = False


class _CudaGraphExecutionProbe:
    """Observe promoted-context graph selection without replacing execution."""

    def __init__(self, runner: CUDAGraphRunner) -> None:
        self._maybe_get_cuda_graph = runner.maybe_get_cuda_graph
        self._replay = runner.replay
        self._executions: list[_PromotedContextGraphExecution] = []
        self._pending_execution: _PromotedContextGraphExecution | None = None

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        enable_spec_decode: bool,
        attn_metadata: Any,
        spec_metadata: SpecMetadata | None = None,
        draft_tokens_cuda: torch.Tensor | None = None,
        new_tensors_device: SampleStateTensors | None = None,
        spec_resource_manager: BaseResourceManager | None = None,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        peft_cache_data_type: torch.dtype | None = None,
    ) -> tuple[Any | None, Any | None, KeyType | None]:
        # A new decision means the preceding one reached eager execution if it
        # did not call replay. Keep that earlier observation unchanged.
        self._pending_execution = None
        result = self._maybe_get_cuda_graph(
            batch,
            enable_spec_decode,
            attn_metadata,
            spec_metadata,
            draft_tokens_cuda,
            new_tensors_device,
            spec_resource_manager,
            promoted_context_request_ids=promoted_context_request_ids,
            peft_cache_data_type=peft_cache_data_type,
        )
        if promoted_context_request_ids:
            execution = _PromotedContextGraphExecution(
                promoted_context_request_ids=promoted_context_request_ids,
                key=result[2],
                enable_spec_decode=enable_spec_decode,
            )
            self._executions.append(execution)
            self._pending_execution = execution
        return result

    def replay(
        self,
        key: KeyType,
        current_inputs: dict[str, Any],
    ) -> torch.Tensor | None:
        output = self._replay(key, current_inputs)
        if self._pending_execution is not None and self._pending_execution.key == key:
            self._pending_execution.replayed = True
            self._pending_execution = None
        return output

    @property
    def executions(self) -> tuple[_PromotedContextGraphExecution, ...]:
        return tuple(self._executions)


def _get_worker_cuda_graph_runner(
    worker: GenerationExecutorWorker,
) -> CUDAGraphRunner:
    assert isinstance(worker.engine, PyExecutor)
    model_engine = worker.engine.model_engine
    assert isinstance(model_engine, PyTorchModelEngine)
    return model_engine.cuda_graph_runner


class _CudaGraphProbeWorker(GenerationExecutorWorker):
    """Install the graph probe independently in every TP worker process."""

    def setup_engine(self) -> None:
        super().setup_engine()
        runner = _get_worker_cuda_graph_runner(self)
        self._cuda_graph_execution_probe = _CudaGraphExecutionProbe(runner)
        runner.maybe_get_cuda_graph = self._cuda_graph_execution_probe.maybe_get_cuda_graph
        runner.replay = self._cuda_graph_execution_probe.replay

    def shutdown(self) -> None:
        # Each TP rank owns a separate Python process and CUDA graph runner.
        # Persist one report per rank before the engine is released so the
        # parent test can prove that every rank selected and replayed a graph.
        if not self.doing_shutdown:
            probe_dir = os.getenv(_TP_GRAPH_PROBE_DIR_ENV)
            assert probe_dir is not None
            report: _RankGraphExecutionReport = {
                "rank": mpi_rank(),
                "executions": [
                    {
                        "promoted_context_request_count": len(
                            execution.promoted_context_request_ids
                        ),
                        "graph_batch_size": (
                            execution.key[0] if execution.key is not None else None
                        ),
                        "enable_spec_decode": execution.enable_spec_decode,
                        "replayed": execution.replayed,
                    }
                    for execution in self._cuda_graph_execution_probe.executions
                ],
            }
            report_path = Path(probe_dir) / f"rank-{report['rank']}.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
        super().shutdown()


def _create_cuda_graph_probe_ipc_executor(
    worker_kwargs: dict[str, object],
    model_world_size: int,
    mpi_session: MpiSession | None,
    postproc_worker_config: PostprocWorkerConfig,
    is_llm_executor: bool | None,
    use_worker: bool = False,
) -> GenerationExecutorProxy:
    assert not use_worker, "The TP2 probe requires separate worker processes"
    return GenerationExecutorProxy(
        worker_kwargs,
        model_world_size=model_world_size,
        mpi_session=mpi_session,
        worker_cls=_CudaGraphProbeWorker,
        postproc_worker_config=postproc_worker_config,
        is_llm_executor=is_llm_executor,
    )


def _get_cuda_graph_runner(llm: LLM) -> CUDAGraphRunner:
    assert isinstance(llm._executor, GenerationExecutorWorker)
    return _get_worker_cuda_graph_runner(llm._executor)


def _assert_reused_context_used_cuda_graph(
    executions: tuple[_PromotedContextGraphExecution, ...],
) -> None:
    assert len(executions) == 1, (
        "Expected exactly one promoted final-context graph decision for the "
        f"reused request, got {len(executions)}"
    )
    execution = executions[0]
    assert len(execution.promoted_context_request_ids) == 1
    assert execution.key is not None, "The promoted final-context row fell back to eager prefill"
    assert execution.key[0] == 1
    assert execution.replayed, (
        "The selected CUDA graph was not replayed for the promoted context row"
    )


def _read_rank_graph_execution_reports(
    report_dir: Path,
    world_size: int,
) -> tuple[_RankGraphExecutionReport, ...]:
    reports: list[_RankGraphExecutionReport] = []
    for rank in range(world_size):
        report_path = report_dir / f"rank-{rank}.json"
        assert report_path.is_file(), f"Missing CUDA graph report for TP rank {rank}"
        report = cast(
            _RankGraphExecutionReport,
            json.loads(report_path.read_text(encoding="utf-8")),
        )
        assert report["rank"] == rank
        reports.append(report)
    return tuple(reports)


def _assert_rank_reused_context_used_cuda_graph(
    report: _RankGraphExecutionReport,
) -> None:
    executions = report["executions"]
    assert len(executions) == 1, (
        f"TP rank {report['rank']} observed {len(executions)} promoted "
        "final-context graph decisions instead of one"
    )
    execution = executions[0]
    assert execution["promoted_context_request_count"] == 1
    assert execution["graph_batch_size"] == 1, (
        f"TP rank {report['rank']} fell back to eager prefill"
    )
    assert not execution["enable_spec_decode"]
    assert execution["replayed"], (
        f"TP rank {report['rank']} selected but did not replay the CUDA graph"
    )


def _generate_cold_and_reused(
    use_kv_cache_manager_v2: bool,
    sampling_params: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
    guided_decoding_backend: Literal["xgrammar"] | None = None,
) -> tuple[RequestOutput, RequestOutput, tuple[_PromotedContextGraphExecution, ...]]:
    # Two complete 32-token blocks can be reused, leaving exactly the final
    # prompt token for the second request. The first and last IDs are distinct
    # so an accidental cursor shift is visible in output/logit parity.
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1], enable_padding=False)
    guided_decoding_args: dict[str, str] = {}
    if guided_decoding_backend is not None:
        guided_decoding_args["guided_decoding_backend"] = guided_decoding_backend

    # Class/instance-level probes do not cross the default TP1 worker process.
    # Keep the real PyExecutor in-process so the wrappers below can observe the
    # actual graph decision and replay while still calling the original code.
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    with LLM(
        model=MODEL,
        max_batch_size=1,
        max_num_tokens=128,
        kv_cache_config=kv_cache_config,
        cuda_graph_config=cuda_graph_config,
        **guided_decoding_args,
    ) as llm:
        cold = llm.generate([PROMPT_TOKEN_IDS], sampling_params)[0]
        runner = _get_cuda_graph_runner(llm)
        probe = _CudaGraphExecutionProbe(runner)
        monkeypatch.setattr(runner, "maybe_get_cuda_graph", probe.maybe_get_cuda_graph)
        monkeypatch.setattr(runner, "replay", probe.replay)
        reused = llm.generate([PROMPT_TOKEN_IDS], sampling_params)[0]

    return cold, reused, probe.executions


def _generate_changed_final_token_cold_and_reused(
    use_kv_cache_manager_v2: bool,
    sampling_params: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    RequestOutput,
    RequestOutput,
    tuple[_PromotedContextGraphExecution, ...],
]:
    """Compare a cold changed-tail request with prefix reuse from another tail."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    # Use a separate engine for the cold reference so it cannot populate the
    # two shared cache blocks exercised by the promoted request below.
    with LLM(
        model=MODEL,
        max_batch_size=1,
        max_num_tokens=128,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=False),
    ) as reference_llm:
        cold = reference_llm.generate([CHANGED_FINAL_PROMPT_TOKEN_IDS], sampling_params)[0]

    with LLM(
        model=MODEL,
        max_batch_size=1,
        max_num_tokens=128,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=False),
    ) as reuse_llm:
        reuse_llm.generate([PROMPT_TOKEN_IDS], sampling_params)
        runner = _get_cuda_graph_runner(reuse_llm)
        probe = _CudaGraphExecutionProbe(runner)
        monkeypatch.setattr(runner, "maybe_get_cuda_graph", probe.maybe_get_cuda_graph)
        monkeypatch.setattr(runner, "replay", probe.replay)
        reused = reuse_llm.generate([CHANGED_FINAL_PROMPT_TOKEN_IDS], sampling_params)[0]

    return cold, reused, probe.executions


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_final_token_reuse_cuda_graph(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the minimal final-token reuse case without optional features."""
    cold, reused, graph_executions = _generate_cold_and_reused(
        use_kv_cache_manager_v2,
        SamplingParams(max_tokens=4, end_id=-1),
        monkeypatch,
    )

    assert cold.outputs[0].token_ids == reused.outputs[0].token_ids
    _assert_reused_context_used_cuda_graph(graph_executions)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_changed_final_token_reuse_cuda_graph(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the promoted row reads a changed final prompt token."""
    cold, reused, graph_executions = _generate_changed_final_token_cold_and_reused(
        use_kv_cache_manager_v2,
        SamplingParams(max_tokens=4, end_id=-1, temperature=0),
        monkeypatch,
    )

    assert PROMPT_TOKEN_IDS[:-1] == CHANGED_FINAL_PROMPT_TOKEN_IDS[:-1]
    assert PROMPT_TOKEN_IDS[-1] != CHANGED_FINAL_PROMPT_TOKEN_IDS[-1]
    assert cold.outputs[0].token_ids == reused.outputs[0].token_ids
    _assert_reused_context_used_cuda_graph(graph_executions)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skip_less_device(2)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_final_token_reuse_cuda_graph_tp2(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify every TP2 rank replays the graph for final-token reuse."""
    report_dir = tmp_path / "tp2-cuda-graph-reports"
    report_dir.mkdir()
    monkeypatch.setenv(_TP_GRAPH_PROBE_DIR_ENV, str(report_dir))
    monkeypatch.setattr(
        GenerationExecutor,
        "_create_ipc_executor",
        staticmethod(_create_cuda_graph_probe_ipc_executor),
    )

    with LLM(
        model=MODEL,
        tensor_parallel_size=2,
        max_batch_size=1,
        max_num_tokens=128,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=False),
    ) as llm:
        sampling_params = SamplingParams(max_tokens=4, end_id=-1)
        cold = llm.generate([PROMPT_TOKEN_IDS], sampling_params)[0]
        reused = llm.generate([PROMPT_TOKEN_IDS], sampling_params)[0]

    assert cold.outputs[0].token_ids == reused.outputs[0].token_ids
    reports = _read_rank_graph_execution_reports(report_dir, world_size=2)
    for report in reports:
        _assert_rank_reused_context_used_cuda_graph(report)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_context_logits_after_final_token_reuse(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify context logits when reuse leaves one prompt token to compute."""
    cold, reused, graph_executions = _generate_cold_and_reused(
        use_kv_cache_manager_v2,
        SamplingParams(
            max_tokens=4,
            end_id=-1,
            return_context_logits=True,
        ),
        monkeypatch,
    )

    assert cold.outputs[0].token_ids == reused.outputs[0].token_ids
    assert cold.context_logits is not None
    assert reused.context_logits is not None
    assert cold.context_logits.shape[0] == len(PROMPT_TOKEN_IDS)
    assert reused.context_logits.shape[0] == 1
    _assert_reused_context_used_cuda_graph(graph_executions)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_guided_decoding_after_final_token_reuse(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify guided decoding when reuse leaves one prompt token to compute."""
    cold, reused, graph_executions = _generate_cold_and_reused(
        use_kv_cache_manager_v2,
        SamplingParams(
            max_tokens=4,
            end_id=-1,
            # Keep the grammar permissive so output equality tests execution-
            # path parity rather than narrow-format generation behavior.
            guided_decoding=GuidedDecodingParams(regex=r".*"),
        ),
        monkeypatch,
        guided_decoding_backend="xgrammar",
    )

    assert cold.outputs[0].token_ids == reused.outputs[0].token_ids
    _assert_reused_context_used_cuda_graph(graph_executions)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_zero_runtime_draft_speculation_after_final_token_reuse(
    use_kv_cache_manager_v2: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a zero-draft speculative iteration replays the decode graph."""
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        free_gpu_memory_fraction=0.6,
    )
    speculative_config = Eagle3DecodingConfig(
        max_draft_len=1,
        speculative_model=EAGLE3_MODEL,
        eagle3_one_model=True,
        # Batch size one drafts one token. Larger batches use the implicit
        # zero-draft schedule entry and therefore exercise this stage's gate.
        draft_len_schedule={1: 1},
    )
    sampling_params = SamplingParams(max_tokens=4, end_id=-1, temperature=0)

    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    with LLM(
        model=SPEC_MODEL,
        max_batch_size=2,
        max_num_tokens=256,
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2], enable_padding=True),
        speculative_config=speculative_config,
    ) as llm:
        cold = [llm.generate([prompt], sampling_params)[0] for prompt in SPEC_PROMPT_TOKEN_IDS]
        runner = _get_cuda_graph_runner(llm)
        probe = _CudaGraphExecutionProbe(runner)
        monkeypatch.setattr(runner, "maybe_get_cuda_graph", probe.maybe_get_cuda_graph)
        monkeypatch.setattr(runner, "replay", probe.replay)
        reused = llm.generate(list(SPEC_PROMPT_TOKEN_IDS), sampling_params)

    assert [output.outputs[0].token_ids for output in cold] == [
        output.outputs[0].token_ids for output in reused
    ]
    assert len(probe.executions) == 1
    execution = probe.executions[0]
    # Scheduler timing may produce either one promoted final-context row plus
    # one generation sibling, or two promoted final-context rows. Both valid
    # batch-two shapes must use the zero-runtime-draft graph.
    assert 1 <= len(execution.promoted_context_request_ids) <= len(SPEC_PROMPT_TOKEN_IDS)
    assert execution.enable_spec_decode
    assert execution.key is not None, (
        "The zero-runtime-draft promoted rows fell back to eager prefill"
    )
    assert execution.key[:2] == (2, 0)
    assert execution.replayed
