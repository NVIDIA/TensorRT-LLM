# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Acceptance tests for import paths consumed by NVIDIA Dynamo.

These tests guarantee that every ``from tensorrt_llm.X import Y`` used by
the Dynamo codebase (https://github.com/ai-dynamo/dynamo) resolves
successfully.  A failing test means a rename/removal will break Dynamo on
the next TRT-LLM upgrade.

To regenerate this list, scan the Dynamo repo for ``tensorrt_llm`` imports:

    grep -rn -e "from tensorrt_llm" -e "import tensorrt_llm" --include="*.py" <dynamo-repo> | sort -u

Changes to this file should be reviewed carefully — removing an entry means
accepting that the corresponding import may break downstream.  If a breaking
change has to be made, inform people from ``trt-llm-dynamo-devs`` by posting a
message in the Slack channel ``swdl-dynamo-trtllm-dev`` so that Dynamo people
are aware of the change.
"""

import importlib

import pytest

# Each entry is (module_path, symbol_name).  The test ID shows both.
DYNAMO_IMPORTS = [
    # -- top-level --
    ("tensorrt_llm", "LLM"),
    ("tensorrt_llm", "MultimodalEncoder"),
    ("tensorrt_llm", "logger"),
    # -- llmapi --
    ("tensorrt_llm.llmapi", "BuildConfig"),
    ("tensorrt_llm.llmapi", "CapacitySchedulerPolicy"),
    ("tensorrt_llm.llmapi", "DynamicBatchConfig"),
    ("tensorrt_llm.llmapi", "KvCacheConfig"),
    ("tensorrt_llm.llmapi", "SchedulerConfig"),
    ("tensorrt_llm.llmapi", "DisaggregatedParams"),
    ("tensorrt_llm.llmapi.llm", "BaseLLM"),
    ("tensorrt_llm.llmapi.llm", "SamplingParams"),
    ("tensorrt_llm.tokenizer", "TOKENIZER_ALIASES"),
    ("tensorrt_llm.llmapi.llm_args", "KvCacheConnectorConfig"),
    ("tensorrt_llm.llmapi.llm_args", "TorchLlmArgs"),
    ("tensorrt_llm.llmapi.llm_utils", "update_llm_args_with_extra_options"),
    ("tensorrt_llm.llmapi.tokenizer", "tokenizer_factory"),
    ("tensorrt_llm.llmapi.disagg_utils", "get_global_disagg_request_id"),
    # -- sampling / scheduling --
    ("tensorrt_llm.sampling_params", "LogitsProcessor"),
    ("tensorrt_llm.sampling_params", "GuidedDecodingParams"),
    ("tensorrt_llm.scheduling_params", "SchedulingParams"),
    # -- executor --
    ("tensorrt_llm.executor.result", "GenerationResult"),
    ("tensorrt_llm.executor.result", "Logprob"),
    ("tensorrt_llm.executor.utils", "RequestError"),
    # -- metrics --
    ("tensorrt_llm.metrics", "MetricsCollector"),
    ("tensorrt_llm.metrics.collector", "MetricsCollector"),
    # -- inputs --
    ("tensorrt_llm.inputs.multimodal", "apply_mm_hashes"),
    ("tensorrt_llm.inputs.utils", "load_image"),
    # -- bindings --
    ("tensorrt_llm.bindings.internal.batch_manager", "LlmRequest"),
    # -- _torch / visual_gen --
    ("tensorrt_llm._torch.visual_gen", "VisualGenArgs"),
    ("tensorrt_llm._torch.visual_gen", "PipelineLoader"),
    ("tensorrt_llm._torch.visual_gen", "CudaGraphConfig"),
    ("tensorrt_llm._torch.visual_gen", "ParallelConfig"),
    ("tensorrt_llm._torch.visual_gen", "PipelineConfig"),
    ("tensorrt_llm._torch.visual_gen", "TeaCacheConfig"),
    ("tensorrt_llm._torch.visual_gen", "TorchCompileConfig"),
    ("tensorrt_llm._torch.visual_gen.config", "AttentionConfig"),
    ("tensorrt_llm._torch.visual_gen.executor", "DiffusionRequest"),
    ("tensorrt_llm._torch.visual_gen.output", "PipelineOutput"),
    ("tensorrt_llm._torch.visual_gen.pipeline", "BasePipeline"),
    # -- _torch / auto_deploy --
    ("tensorrt_llm._torch.auto_deploy", "LLM"),
    ("tensorrt_llm._torch.auto_deploy", "LlmArgs"),
    # -- _torch / shared_tensor --
    ("tensorrt_llm._torch.shared_tensor", "SharedTensorContainer"),
    ("tensorrt_llm._torch.shared_tensor.shared_tensor", "SharedTensorContainer"),
    ("tensorrt_llm._torch.shared_tensor.shared_tensor", "_SharedTensorRebuildMethodRegistry"),
    # -- _torch / pyexecutor --
    ("tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector", "KvCacheConnectorScheduler"),
    ("tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector", "KvCacheConnectorWorker"),
    ("tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector", "SchedulerOutput"),
    # -- bare module imports (import tensorrt_llm.X.Y) --
    ("tensorrt_llm._torch.visual_gen", None),
]


@pytest.mark.parametrize(
    "module_path, symbol",
    DYNAMO_IMPORTS,
    ids=[f"{m}-{s}" if s else f"{m}" for m, s in DYNAMO_IMPORTS],
)
def test_dynamo_import(module_path: str, symbol: str) -> None:
    """Verify that *module_path* exposes *symbol* (the way Dynamo imports it).

    When *symbol* is ``None`` the entry represents a bare
    ``import tensorrt_llm.X.Y`` and we only check that the module is
    importable.
    """
    mod = importlib.import_module(module_path)
    if symbol is not None:
        assert hasattr(mod, symbol), (
            f"'{symbol}' is no longer importable from '{module_path}'. "
            f"This import is used by the Dynamo codebase — renaming or removing "
            f"it will break the next Dynamo upgrade of TRT-LLM."
        )
