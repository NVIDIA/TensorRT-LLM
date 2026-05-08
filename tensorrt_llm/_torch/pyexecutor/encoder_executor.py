# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

import torch

from tensorrt_llm.logger import logger


class EncoderExecutor:
    """Executor for models using the encode-only path.

    Primary path: batch_forward(inputs) — synchronous batch execution.
    Delegates to model_engine.encoder_forward() for all heavy lifting
    (pre-allocated buffers, attention metadata, torch.compile).

    This executor has no background thread, no scheduler, no sampler,
    and no request queue. It runs entirely on the calling thread.
    """

    def __init__(self, model_engine, dist):
        self.model_engine = model_engine
        self.dist = dist

        logger.info(
            "encode_only path enabled: using EncoderExecutor. "
            "Scheduler, sampler, KV cache, and generation-related parameters "
            "(disable_overlap_scheduler, max_tokens, temperature, etc.) "
            "are bypassed. Use llm.encode() for inference."
        )

    def batch_forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Execute a pre-formed batch in one forward pass.

        Args:
            inputs: Dict with 'input_ids' ([total_tokens]) and 'seq_lens'
                ([batch_size]) required. Optional model-specific kwargs
                (token_type_ids, inputs_embeds, etc.) are passed through.

        Returns:
            Dict with 'logits' tensor and any other model outputs.
        """
        return self.model_engine.encoder_forward(inputs)

    def shutdown(self):
        """No background thread to stop — just release model engine resources."""
        del self.model_engine
