# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SimModelEngine: dummy model engine for simulation mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    from .resource_manager import ResourceManager
    from .scheduler import ScheduledRequests


class SimModelEngine:
    """Model engine that returns dummy logits without running GPU inference.

    Used in simulation mode to exercise the scheduler and request state
    machine without loading model weights or executing forward passes.

    Implements the same interface as ModelEngine but avoids importing
    ModelEngine (and its heavy dependency chain) at module level.
    """

    def __init__(self, llm_args: TorchLlmArgs, vocab_size: int,
                 max_num_sequences: int, time_predictor=None, clock=None):
        self.llm_args = llm_args
        self.vocab_size = vocab_size
        self._max_num_sequences = max_num_sequences
        self.time_predictor = time_predictor
        self.clock = clock

        # Attributes read by PyExecutor.__init__ and _executor_loop
        self.spec_config = None
        self.enable_attention_dp = False
        self.iter_states = {}
        self.is_warmup = False
        self.enable_spec_decode = False
        self.runtime_draft_len = 0
        self.max_draft_len = 0
        self.kv_cache_dtype_byte_size = None
        self.attn_metadata = None
        self.use_mrope = False
        self.without_logits = False
        self._max_cuda_graph_batch_size = 0
        self._cuda_graph_batch_sizes = []
        self.spec_metadata = None
        self.is_draft_model = False
        self.dtype = torch.float16
        self.input_processor = None
        self.input_processor_with_hash = None

        logger.info("[SimModelEngine] Initialized (vocab_size=%d, "
                    "max_num_sequences=%d)", vocab_size, max_num_sequences)

    def get_max_num_sequences(self) -> int:
        return self._max_num_sequences

    def forward(self, scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager, new_tensors_device=None,
                gather_context_logits: bool = False,
                cache_indirection_buffer=None,
                num_accepted_tokens_device=None):
        num_ctx_requests = scheduled_requests.num_context_requests
        num_ctx_tokens = sum(r.context_chunk_size
                            for r in scheduled_requests.context_requests)
        num_gen_tokens = len(scheduled_requests.generation_requests)

        self.iter_states = {
            'num_ctx_requests': num_ctx_requests,
            'num_ctx_tokens': num_ctx_tokens,
            'num_generation_tokens': num_gen_tokens,
        }

        total_tokens = num_ctx_tokens + num_gen_tokens

        if self.time_predictor is not None:
            from .sim_predictor import SimBatch, SimBatchRequest

            requests = []
            for r in scheduled_requests.context_requests:
                requests.append(SimBatchRequest(
                    input_length=r.context_chunk_size,
                    past_kv_length=r.get_num_tokens(0) - r.context_chunk_size))
            for r in scheduled_requests.generation_requests:
                requests.append(SimBatchRequest(
                    input_length=1,
                    past_kv_length=r.get_num_tokens(0) - 1))

            batch = SimBatch(
                num_context_requests=num_ctx_requests,
                num_context_tokens=num_ctx_tokens,
                num_generation_requests=num_gen_tokens,
                num_generation_tokens=num_gen_tokens,
                requests=requests)
            predicted_time = self.time_predictor.predict(batch)
            if self.clock is not None:
                self.clock.step(predicted_time)
                logger.debug(
                    "[SimModelEngine] iter=%d predicted=%.3fms total=%.3fms",
                    self.clock.num_iterations,
                    predicted_time * 1000,
                    self.clock.total_time_s * 1000)

        logits = torch.zeros(total_tokens, self.vocab_size)
        return {'logits': logits}

    def warmup(self, resource_manager) -> None:
        pass
