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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger

from .engine.model_setup import build_encoder_runner
from .engine.runners.encoder import EncoderRunner
from .engine.runners.interface import PackedInputs

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    from ..distributed import Distributed
    from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
    from .model_loader import ModelLoader


class EncoderExecutor:
    """Executor for models using the encode-only path.

    Primary path: batch_forward(inputs) — synchronous batch execution.
    It builds its EncoderRunner from the shared model setup steps and calls
    it directly.

    This executor has no background thread, no scheduler, no sampler,
    and no request queue. It runs entirely on the calling thread.
    """

    def __init__(
        self,
        runner: EncoderRunner,
        *,
        model_loader: ModelLoader,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        pretrained_config: PretrainedConfig | None,
        dist: Distributed,
    ) -> None:
        self._runner = runner
        self._model_loader = model_loader
        # The limits that callers validate a batch against.
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.pretrained_config = pretrained_config
        self.dist = dist

        logger.info(
            "encode_only path enabled: using EncoderExecutor. "
            "Scheduler, sampler, KV cache, and generation-related parameters "
            "(disable_overlap_scheduler, max_tokens, temperature, etc.) "
            "are bypassed. Use llm.encode() for inference."
        )

        self._runner.warmup()

    @classmethod
    def create(
        cls,
        llm_args: TorchLlmArgs,
        *,
        model_path: str,
        checkpoint_loader: BaseCheckpointLoader,
        mapping: Mapping,
        dist: Distributed,
    ) -> EncoderExecutor:
        """Load and compile the model, build its runner, and warm it up."""
        if mapping.world_size > 1:
            raise ValueError(
                "encode_only runs in-process on a single GPU and does not support "
                "tensor, pipeline or context parallelism."
            )
        with build_encoder_runner(
            llm_args,
            model_path=model_path,
            checkpoint_loader=checkpoint_loader,
            mapping=mapping,
            dist=dist,
        ) as built:
            return cls(
                built.runner,
                model_loader=built.model_loader,
                max_batch_size=built.max_batch_size,
                max_num_tokens=built.max_num_tokens,
                max_seq_len=built.max_seq_len,
                pretrained_config=built.model_config.pretrained_config,
                dist=dist,
            )

    def batch_forward(
        self, inputs: dict[str, Any], *, gather_context_logits: bool = False
    ) -> dict[str, torch.Tensor]:
        """Execute a pre-formed batch in one forward pass.

        Args:
            inputs: Dict with 'input_ids' ([total_tokens]) and 'seq_lens'
                ([batch_size]) required. Optional model-specific kwargs
                (token_type_ids, inputs_embeds, etc.) are passed through.
            gather_context_logits: Return the logits of every token instead
                of one row per sequence.

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
            gather_context_logits=gather_context_logits,
        )
        return self._runner.forward_packed(packed_inputs)

    def shutdown(self) -> None:
        """No background thread to stop — release the loader, the graphs, then the model."""
        self._model_loader.cleanup()
        self._runner.release_graphs()
        # The runner holds the model and its compiled caller.
        del self._runner, self._model_loader
        release_gc()
