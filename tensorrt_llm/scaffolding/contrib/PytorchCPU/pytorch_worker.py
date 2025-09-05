import logging
import os

import torch
from transformers import AutoTokenizer

from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.scaffolding import GenerationTask, TaskStatus, Worker
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

logger = logging.getLogger(__name__)


class PytorchWorker(Worker):
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
                legacy=False,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
            logger.info(f"Tokenizer loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_path}: {e}")
            raise

        # Configure PyTorch backend for CPU inference
        pytorch_config = PyTorchConfig(
            use_cuda_graph=False,  # Disable CUDA graphs for CPU
            attn_backend='torch',  # Use torch attention backend for CPU
            torch_compile_enabled=False,  # Disable torch compile for CPU
            disable_overlap_scheduler=True,  # Disable overlap scheduler for CPU
            allreduce_strategy="AUTO",  # Use auto strategy for allreduce
        )

        try:
            self.llm = LLM(
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

    def convert_task_params(self, task: GenerationTask) -> SamplingParams:
        """
        Convert a GenerationTask to SamplingParams for the LLM.

        Args:
            task: The generation task containing sampling parameters

        Returns:
            SamplingParams object configured for the task
        """
        try:
            sampling_params = SamplingParams(
                max_tokens=task.max_tokens or 512,  # Default max tokens
                temperature=task.temperature or 1.0,
                top_p=task.top_p or 1.0,
                top_k=task.top_k or -1,
                return_context_logits=task.return_context_logits or False,
                frequency_penalty=task.frequency_penalty or 0.0,
                presence_penalty=task.presence_penalty or 0.0,
                stop=task.stop or [],
            )
            return sampling_params
        except Exception as e:
            logger.error(f"Failed to convert task parameters: {e}")
            # Return default sampling params as fallback
            return SamplingParams(max_tokens=512, temperature=1.0)

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        """
        Handle generation requests for the given task.

        Args:
            task: The generation task to process

        Returns:
            TaskStatus indicating success or failure
        """
        try:
            # Convert task parameters to SamplingParams
            sampling_params = self.convert_task_params(task)

            # Generate response using the LLM
            result = await self.llm.generate_async(
                task.input_str,
                sampling_params=sampling_params
            )

            # Extract results and populate task
            if result.outputs and len(result.outputs) > 0:
                output = result.outputs[0]
                task.output_tokens = output.token_ids
                task.cumulative_logprob = output.cumulative_logprob
                task.logprobs = output.logprobs
                task.output_str = output.text
                task.context_logits = getattr(result, 'context_logits', None)

                return TaskStatus.SUCCESS
            else:
                logger.error("No outputs generated from LLM")
                return TaskStatus.WORKER_EXECEPTION

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return TaskStatus.WORKER_EXECEPTION

    def shutdown(self):
        """
        Clean up resources when shutting down the worker.
        """
        try:
            if hasattr(self, 'llm') and self.llm:
                self.llm.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    task_handlers = {GenerationTask: generation_handler}
