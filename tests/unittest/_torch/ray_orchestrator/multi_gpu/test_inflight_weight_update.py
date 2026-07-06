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
"""End-to-end test for in-flight (non-draining) weight update.

The draining variant (``drain=True``) waits for ``active_requests`` and
``waiting_queue`` to empty before firing the control action, so the engine has
exclusive access during the weight reload. That path is covered by
``test_llm_update_weights*``.

This test exercises the ``drain=False`` path instead: weights are updated while
generation requests are still in flight. It verifies the race called out in
``PyExecutor._prepare_and_schedule_batch`` -- a refit that pauses
``GENERATION_IN_PROGRESS`` requests must not leave ``_forward_step`` consuming
entries whose KV cache has just been freed -- by asserting that:

1. The in-flight requests submitted before the update finish without error.
2. A fresh generation after the update matches the HuggingFace reference,
   confirming the new weights were actually applied.
"""

import asyncio
from typing import List, Optional, Tuple

import pytest
import torch
from _torch.ray_orchestrator.single_gpu.test_llm_update_weights import (
    RefHFModelWithIPCHandles,
    compare_logits,
)
from transformers import AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel
from utils.util import skip_pre_hopper

from tensorrt_llm import AsyncLLM
from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

# Module path used by the Ray worker to import the extension below. With
# ``TLLM_RAY_FORCE_LOCAL_CLUSTER=1`` (set by the ray_orchestrator conftest) the
# workers are local processes that inherit this driver's sys.path, so the test
# module is importable by qualname.
_EXTENSION_CLS = (
    "_torch.ray_orchestrator.multi_gpu.test_inflight_weight_update.InflightUpdateWorkerExtension"
)


class InflightUpdateWorkerExtension(WorkerExtension):
    """Default RLHF worker extension, but with a non-draining ``update_weights``.

    Only the decorator changes: ``drain=False`` makes the control action fire as
    soon as a fetched batch contains the control sentinel, instead of waiting for
    the engine to drain. The body is inherited unchanged from
    ``WorkerExtension.update_weights``.
    """

    @control_action_decorator(drain=False)
    def update_weights(self, ipc_handles: Optional[dict] = None):
        # Call the raw, undecorated body of the base implementation. Invoking
        # ``super().update_weights(...)`` directly would re-enter the drain=True
        # control_action context manager and nest control actions.
        return WorkerExtension.update_weights.__wrapped__(self, ipc_handles)


async def _run_generate_async(
    llm: AsyncLLM,
    hf_model: RefHFModel,
    prompts: List[List[int]],
    sampling_params: SamplingParams,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Async counterpart of single_gpu ``run_generate``: generate, then compare
    against the HuggingFace reference run over the produced token ids."""
    outputs = await asyncio.gather(*[llm.generate_async(p, sampling_params) for p in prompts])
    llm_logits = [o.outputs[0].generation_logits for o in outputs]
    llm_responses = [o.outputs[0].token_ids for o in outputs]

    input_ids, attention_mask, position_ids = RefHFModel.pad_data(prompts, llm_responses)
    ref_logits = hf_model.generate_batch_with_padding(
        input_ids, attention_mask, position_ids, llm_responses, return_logits=True
    )
    return llm_logits, ref_logits


@pytest.mark.ray
@pytest.mark.asyncio
@skip_pre_hopper
async def test_inflight_weight_update():
    model_dir = str(llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")
    num_hidden_layers = 1

    # Reference HF model providing the "new" weights via CUDA IPC handles.
    hf_model = RefHFModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    prompts_texts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = [tokenizer.encode(p) for p in prompts_texts]
    del tokenizer

    # Block reuse MUST stay disabled here, otherwise Phase D reuses KV blocks
    # that were computed under the old (dummy) weights and the logits check
    # fails. The reason is *when* blocks are registered into the reuse pool --
    # there are two registration points, and one of them happens AFTER the
    # post-update reset_prefix_cache():
    #
    #   1. When prefill finishes, store_context_blocks() registers the prompt's
    #      KV blocks (computed under the old weights at t~0). The finalize step
    #      update_weights(None) -> reset_prefix_cache() clears these.
    #   2. When a request COMPLETES, free_resources() -> removeSequence() calls
    #      releaseBlocks(seq, llmRequest) (because reuse is on) and re-registers
    #      the whole sequence -- including those old-weight prompt blocks -- back
    #      into the reuse pool.
    #
    # The in-flight requests only complete in Phase C, i.e. AFTER the Phase B
    # reset, so registration point (2) re-poisons the pool with old-weight
    # blocks that reset_prefix_cache() can no longer catch. Phase D reuses the
    # same prompts, hits those stale blocks, and reads dummy-weight KV.
    # With reuse disabled, removeSequence() calls releaseBlocks(seq, nullopt)
    # -- blocks are freed but never registered -- so Phase D recomputes cleanly.
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.1)

    async with AsyncLLM(
        model=model_dir,
        ray_worker_extension_cls=_EXTENSION_CLS,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        load_format="dummy",
        kv_cache_config=kv_cache_config,
        model_kwargs={"num_hidden_layers": num_hidden_layers},
    ) as llm:
        # Phase A: launch long-running generations and keep them in flight
        # (large max_tokens so they are still GENERATION_IN_PROGRESS when the
        # weight update fires). Do NOT await yet.
        inflight_params = SamplingParams(temperature=0, max_tokens=512)
        inflight_tasks = [
            asyncio.ensure_future(llm.generate_async(p, inflight_params)) for p in prompts
        ]
        # Give prefill a moment to start so the control sentinel lands behind
        # in-flight requests rather than an empty engine.
        await asyncio.sleep(1.0)

        # Phase B: update weights while requests are in flight. drain=False, so
        # the action fires at the next step boundary without draining.
        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0, 1])
        await llm.collective_rpc("update_weights", args=(ipc_handles,))
        # Finalize (process_weights_after_loading + prefix cache reset).
        await llm.collective_rpc("update_weights", args=(None,))

        # Phase C: the in-flight requests must finish without error despite the
        # mid-flight pause/refit. This is the KV-already-freed race guard.
        inflight_outputs = await asyncio.gather(*inflight_tasks)
        for out in inflight_outputs:
            assert out.outputs[0].token_ids, (
                "In-flight request returned no tokens after a non-draining weight update"
            )

        # Phase D: a fresh generation after the update must match the HF
        # reference, confirming the new weights took effect.
        verify_params = SamplingParams(temperature=0, return_generation_logits=True, max_tokens=32)
        llm_logits, ref_logits = await _run_generate_async(llm, hf_model, prompts, verify_params)
        compare_logits(llm_logits, ref_logits)
