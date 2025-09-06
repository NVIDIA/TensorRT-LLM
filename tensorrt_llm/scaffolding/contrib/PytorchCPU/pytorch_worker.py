# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
from transformers import AutoTokenizer

from tensorrt_llm import sampling_params, scaffolding
from tensorrt_llm._torch.pyexecutor import config
from tensorrt_llm.llmapi import llm

logger = logging.getLogger(__name__)


class PytorchWorker(scaffolding.Worker):
    """
    A worker implementation for CPU inference using PyTorch backend.

    This worker enables CPU-based inference for TensorRT-LLM scaffolding,
    allowing development and testing without GPU requirements.
    """

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 32,
        max_num_tokens: int = 4096,
        trust_remote_code: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize the PyTorch CPU worker.

        Args:
            model_path: Path to the model directory or Hugging Face model name
            max_batch_size: Maximum batch size for inference
            max_num_tokens: Maximum number of tokens to process
            trust_remote_code: Whether to trust remote code in model loading
            device: Device to use for inference (default: "cpu")
        """
        super().__init__()
        self.model_path = model_path
        self.device = device

        # Force CPU usage if CUDA is not available
        if device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
            # Set environment variables to ensure CPU usage
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Using CPU device for inference")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
            logger.info(f"Tokenizer loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_path}: {e}")
            raise

        # Configure PyTorch backend for CPU inference
        pytorch_config = config.PyTorchConfig(
            use_cuda_graph=False,  # Disable CUDA graphs for CPU
            attn_backend='torch',  # Use torch attention backend for CPU
            torch_compile_enabled=False,  # Disable torch compile for CPU
            disable_overlap_scheduler=True,  # Disable overlap scheduler for CPU
            allreduce_strategy="AUTO",  # Use auto strategy for allreduce
        )

        try:
            self.llm = llm.LLM(
                model=model_path,
                tokenizer=self.tokenizer,
                backend='pytorch',
                pytorch_backend_config=pytorch_config,
                max_batch_size=max_batch_size,
                max_num_tokens=max_num_tokens,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=1,  # Force single process for CPU
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def convert_task_params(
            self,
            task: scaffolding.GenerationTask) -> sampling_params.SamplingParams:
        """
        Convert a GenerationTask to SamplingParams for the LLM.

        Args:
            task: The generation task containing sampling parameters.

        Returns:
            SamplingParams object configured for the task.
        """
        try:
            task_sampling_params = sampling_params.SamplingParams(
                max_tokens=task.max_tokens
                if task.max_tokens is not None else 512,
                temperature=task.temperature
                if task.temperature is not None else 1.0,
                top_p=task.top_p if task.top_p is not None else 1.0,
                top_k=task.top_k if task.top_k is not None else -1,
                return_context_logits=(task.return_context_logits
                                       if task.return_context_logits is not None
                                       else False),
                frequency_penalty=(task.frequency_penalty if
                                   task.frequency_penalty is not None else 0.0),
                presence_penalty=(task.presence_penalty if task.presence_penalty
                                  is not None else 0.0),
                stop=task.stop if task.stop is not None else [],
            )
            return task_sampling_params
        except Exception as e:
            logger.error(f"Failed to convert task parameters: {e}")
            # Return default sampling params as fallback
            return sampling_params.SamplingParams(max_tokens=512,
                                                  temperature=1.0)

    async def generation_handler(
            self, task: scaffolding.GenerationTask) -> scaffolding.TaskStatus:
        """
        Handle generation requests for the given task.

        Args:
            task: The generation task to process.

        Returns:
            TaskStatus indicating success or failure.
        """
        try:
            # Convert task parameters to SamplingParams
            task_sampling_params = self.convert_task_params(task)

            # Generate response using the LLM
            result = await self.llm.generate_async(
                task.input_str, sampling_params=task_sampling_params)

            # Extract results and populate task
            if result.outputs and len(result.outputs) > 0:
                output = result.outputs[0]
                task.output_tokens = getattr(output, 'token_ids', None)
                task.cumulative_logprob = getattr(output, 'cumulative_logprob',
                                                  None)
                task.logprobs = getattr(output, 'logprobs', None)
                task.output_str = getattr(output, 'text', None)
                task.context_logits = getattr(result, 'context_logits', None)

                return scaffolding.TaskStatus.SUCCESS
            else:
                logger.error("No outputs generated from LLM")
                return scaffolding.TaskStatus.WORKER_EXCEPTION

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return scaffolding.TaskStatus.WORKER_EXCEPTION

    def shutdown(self):
        """
        Clean up resources when shutting down the worker.
        """
        try:
            if hasattr(self, 'llm') and self.llm:
                self.llm.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    task_handlers = {scaffolding.GenerationTask: generation_handler}
