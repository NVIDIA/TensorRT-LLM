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

from .engine.runners.interface import PackedInputs


class EncoderExecutor:
    """Executor for models using the encode-only path.

    Primary path: batch_forward(inputs) — synchronous batch execution.
    Hands the already-packed batch to the encoder runner as-is.

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

        self.model_engine.warmup()

    def batch_forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """Execute a pre-formed batch in one forward pass.

        Args:
            inputs: Dict with 'input_ids' ([total_tokens]) and 'seq_lens'
                ([batch_size]) required. Optional model-specific kwargs
                (token_type_ids, inputs_embeds, etc.) are passed through.

        Returns:
            Dict with 'logits' tensor and any other model outputs.
        """
        model_inputs = dict(inputs)
        input_ids = model_inputs.pop("input_ids")
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        packed_inputs = PackedInputs(
            input_ids=input_ids,
            sequence_lengths=[int(length) for length in model_inputs.pop("seq_lens")],
            multi_item_part_lens=model_inputs.pop("multi_item_part_lens", None),
            model_inputs=model_inputs,
            gather_context_logits=kwargs.pop("gather_context_logits", False),
        )
        return self.model_engine.forward(packed_inputs, **kwargs)

    def shutdown(self):
        """No background thread to stop — just release model engine resources."""
        del self.model_engine
