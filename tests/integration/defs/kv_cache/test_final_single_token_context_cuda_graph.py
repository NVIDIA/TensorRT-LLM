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

from dataclasses import dataclass
from typing import Any, Literal

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
from tensorrt_llm.executor.worker import GenerationExecutorWorker
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, RequestOutput
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from ..conftest import llm_models_root

MODEL = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
PROMPT_TOKEN_IDS = [1] + [42] * 63 + [43]


@dataclass
class _PromotedContextGraphExecution:
    promoted_context_request_ids: frozenset[int]
    key: KeyType | None
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
            promoted_context_request_ids,
        )
        if promoted_context_request_ids:
            execution = _PromotedContextGraphExecution(
                promoted_context_request_ids=promoted_context_request_ids,
                key=result[2],
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


def _get_cuda_graph_runner(llm: LLM) -> CUDAGraphRunner:
    assert isinstance(llm._executor, GenerationExecutorWorker)
    assert isinstance(llm._executor.engine, PyExecutor)
    model_engine = llm._executor.engine.model_engine
    assert isinstance(model_engine, PyTorchModelEngine)
    return model_engine.cuda_graph_runner


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
