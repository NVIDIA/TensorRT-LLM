# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyTorch backend worker for TensorRT-LLM Scaffolding.

This module provides a worker implementation that uses native PyTorch and
HuggingFace transformers for inference, enabling easy integration of research
components without requiring TensorRT compilation.
"""

import copy
import traceback
from typing import List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from tensorrt_llm.executor.result import Logprob

from .result import ScaffoldingOutput
from .task import GenerationTask, RewardTask, StreamGenerationTask, TaskStatus
from .worker import Worker


class _StopStringCriteria(StoppingCriteria):
    """Custom stopping criteria that checks for stop strings in decoded output."""

    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: List[str], input_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = self.tokenizer.decode(
            input_ids[0, self.input_length :], skip_special_tokens=True
        )
        return any(s in generated for s in self.stop_strings)


class PyTorchWorker(Worker):
    """Worker that executes tasks using PyTorch and HuggingFace models.

    This worker enables running inference with pure PyTorch models, which is
    useful for:
    - Rapid prototyping without TensorRT compilation
    - Custom research models (reward models, verifiers, etc.)
    - Models with architectures not yet supported by TensorRT
    - Debugging and analysis requiring PyTorch-native tools

    The worker supports generation tasks (using causal LMs), streaming
    generation tasks (with pause/resume/cancel), and reward tasks (using
    sequence classification models).

    Example:
        ```python
        # Create worker with a HuggingFace model
        worker = PyTorchWorker.from_pretrained("gpt2")

        # Use in scaffolding
        llm = ScaffoldingLlm(controller, {"generation": worker})
        ```
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[Union[str, torch.device]] = None,
        max_batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.model.to(self.device)
        self.model.eval()

        self.max_batch_size = max_batch_size

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = False,
        **model_kwargs,
    ) -> "PyTorchWorker":
        """Create a PyTorchWorker from a HuggingFace model name or path.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to run on (default: auto-detect CUDA)
            torch_dtype: Data type for model weights (default: float16)
            trust_remote_code: Whether to trust remote code in model
            **model_kwargs: Additional arguments passed to model loading

        Returns:
            PyTorchWorker instance with loaded model
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        return cls(model, tokenizer, device=device)

    @classmethod
    def from_pretrained_reward_model(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = False,
        **model_kwargs,
    ) -> "PyTorchWorker":
        """Create a PyTorchWorker for reward modeling tasks.

        Loads a sequence classification model suitable for scoring text.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        return cls(model, tokenizer, device=device)

    def _tokenize_input(self, input_str: Optional[str], input_tokens: Optional[list]):
        """Tokenize from input_str or input_tokens, returning model inputs dict.

        Returns None if neither input_str nor input_tokens is provided.
        """
        if input_str is not None:
            return self.tokenizer(input_str, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
        elif input_tokens is not None:
            input_ids = torch.tensor([input_tokens], device=self.device)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return None

    def convert_task_params(self, task: GenerationTask) -> GenerationConfig:
        """Convert task sampling parameters to HuggingFace GenerationConfig."""
        return GenerationConfig(
            max_new_tokens=task.max_tokens if task.max_tokens is not None else 100,
            temperature=task.temperature if task.temperature is not None else 1.0,
            top_p=task.top_p if task.top_p is not None else 1.0,
            top_k=task.top_k if task.top_k is not None else 50,
            do_sample=(task.temperature is not None and task.temperature > 0),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _build_stopping_criteria(self, task: GenerationTask, input_length: int):
        """Build stopping criteria list from task stop sequences."""
        if not task.stop:
            return None
        stop_strings = [task.stop] if isinstance(task.stop, str) else task.stop
        if not stop_strings:
            return None
        return StoppingCriteriaList(
            [_StopStringCriteria(self.tokenizer, stop_strings, input_length)]
        )

    def _strip_stop_sequence(self, text: str, task: GenerationTask) -> str:
        """Remove trailing stop sequence from generated text."""
        if not task.stop:
            return text
        stop_strings = [task.stop] if isinstance(task.stop, str) else task.stop
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1:
                return text[:idx]
        return text

    def _extract_logprobs(self, scores: tuple, generated_tokens: List[int], num_logprobs: int):
        """Extract top-k logprobs from generation scores.

        Args:
            scores: Tuple of [batch_size, vocab_size] tensors per step.
            generated_tokens: List of token IDs that were generated.
            num_logprobs: Number of top logprobs to return per token.

        Returns:
            TokenLogprobs (list[dict[int, Logprob]])
        """
        logprobs_list = []
        for step_idx, step_scores in enumerate(scores):
            log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
            token_dict = {}

            if num_logprobs > 0:
                k = min(num_logprobs, log_probs.shape[0])
                topk_vals, topk_indices = torch.topk(log_probs, k=k)
                for rank, (val, idx) in enumerate(zip(topk_vals, topk_indices)):
                    token_dict[idx.item()] = Logprob(logprob=val.item(), rank=rank + 1)

            # Always include the actually generated token
            if step_idx < len(generated_tokens):
                gen_token_id = generated_tokens[step_idx]
                if gen_token_id not in token_dict:
                    gen_logprob = log_probs[gen_token_id].item()
                    gen_rank = (log_probs > gen_logprob).sum().item() + 1
                    token_dict[gen_token_id] = Logprob(logprob=gen_logprob, rank=gen_rank)

            logprobs_list.append(token_dict)
        return logprobs_list

    def _sample_token(self, logits: torch.Tensor, gen_config: GenerationConfig) -> torch.Tensor:
        """Sample a single token from logits using the generation config."""
        if not gen_config.do_sample:
            return torch.argmax(logits, dim=-1)

        logits = logits / max(gen_config.temperature, 1e-7)

        # Top-k filtering
        if gen_config.top_k > 0:
            topk_vals = torch.topk(logits, gen_config.top_k)[0]
            logits[logits < topk_vals[..., -1, None]] = -float("inf")

        # Top-p (nucleus) filtering
        if gen_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = (cum_probs - torch.softmax(sorted_logits, dim=-1)) >= gen_config.top_p
            sorted_logits[remove_mask] = -float("inf")
            logits.scatter_(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        """Handle text generation task using PyTorch model."""
        try:
            inputs = self._tokenize_input(task.input_str, task.input_tokens)
            if inputs is None:
                return TaskStatus.WORKER_EXCEPTION

            # Ensure inputs is a dict (tokenizer returns BatchEncoding)
            if not isinstance(inputs, dict):
                inputs = dict(inputs)

            gen_config = self.convert_task_params(task)
            input_length = inputs["input_ids"].shape[1]

            stopping_criteria = self._build_stopping_criteria(task, input_length)

            generate_kwargs = {
                "generation_config": gen_config,
                "return_dict_in_generate": True,
                "output_scores": task.num_logprobs is not None,
            }
            if stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            generated_tokens = outputs.sequences[0, input_length:].tolist()
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            output_text = self._strip_stop_sequence(output_text, task)

            task.output_str = output_text
            task.output_tokens = generated_tokens

            if task.num_logprobs is not None and hasattr(outputs, "scores") and outputs.scores:
                task.logprobs = self._extract_logprobs(
                    outputs.scores, generated_tokens, task.num_logprobs
                )

            return TaskStatus.SUCCESS

        except Exception as e:
            print(f"PyTorchWorker generation error: {e}")
            traceback.print_exc()
            return TaskStatus.WORKER_EXCEPTION

    async def stream_generation_handler(self, task: StreamGenerationTask) -> TaskStatus:
        """Handle streaming generation with pause/resume/cancel semantics.

        Uses token-by-token autoregressive decoding with KV cache for
        efficient pause and resume.
        """
        try:
            if task.cancel_flag:
                task.end_flag = True
                return TaskStatus.SUCCESS

            # First invocation: initialize state
            if task.request_handle is None:
                inputs = self._tokenize_input(task.input_str, task.input_tokens)
                if inputs is None:
                    return TaskStatus.WORKER_EXCEPTION

                input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids

                task.request_handle = {
                    "input_ids": input_ids,
                    "generated_ids": [],
                    "past_key_values": None,
                    "done": False,
                }

            handle = task.request_handle
            gen_config = self.convert_task_params(task)
            max_remaining = gen_config.max_new_tokens - len(handle["generated_ids"])
            step_target = task.streaming_step or 1
            steps_done = 0

            # Prepare stop strings
            stop_strings = None
            if task.stop:
                stop_strings = [task.stop] if isinstance(task.stop, str) else task.stop

            with torch.no_grad():
                current_ids = handle["input_ids"]
                past = handle["past_key_values"]

                while steps_done < step_target and max_remaining > 0:
                    if past is not None:
                        model_input = current_ids[:, -1:]
                        model_out = self.model(model_input, past_key_values=past, use_cache=True)
                    else:
                        model_out = self.model(current_ids, use_cache=True)

                    next_token_logits = model_out.logits[:, -1:, :]
                    next_token = self._sample_token(next_token_logits.squeeze(1), gen_config)

                    handle["generated_ids"].append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)
                    past = model_out.past_key_values
                    steps_done += 1
                    max_remaining -= 1

                    # Check EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        handle["done"] = True
                        break

                    # Check stop sequences
                    if stop_strings:
                        partial = self.tokenizer.decode(
                            handle["generated_ids"], skip_special_tokens=True
                        )
                        if any(s in partial for s in stop_strings):
                            handle["done"] = True
                            break

                    # Emit streaming output
                    if task.streaming_output_flag:
                        partial_text = self.tokenizer.decode(
                            handle["generated_ids"], skip_special_tokens=True
                        )
                        task.streaming_output_list.append(
                            ScaffoldingOutput(partial_text, copy.deepcopy(handle["generated_ids"]))
                        )

                handle["input_ids"] = current_ids
                handle["past_key_values"] = past

            # Fill results
            task.output_tokens = list(handle["generated_ids"])
            output_text = self.tokenizer.decode(handle["generated_ids"], skip_special_tokens=True)
            if stop_strings:
                output_text = self._strip_stop_sequence(output_text, task)
            task.output_str = output_text

            if handle["done"] or len(handle["generated_ids"]) >= gen_config.max_new_tokens:
                task.end_flag = True

            return TaskStatus.SUCCESS

        except Exception as e:
            print(f"PyTorchWorker stream generation error: {e}")
            traceback.print_exc()
            return TaskStatus.WORKER_EXCEPTION

    async def reward_handler(self, task: RewardTask) -> TaskStatus:
        """Handle reward/scoring task using sequence classification model."""
        try:
            inputs = self._tokenize_input(task.input_str, task.input_tokens)
            if inputs is None:
                # Fallback to empty string for backward compatibility
                inputs = self.tokenizer(
                    "", return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)

            if not isinstance(inputs, dict):
                inputs = dict(inputs)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0]

            if task.custom_output_params is None:
                task.custom_output_params = {}

            if logits.shape[0] == 2:
                probs = torch.softmax(logits, dim=0)
                task.custom_output_params["score"] = probs[1].item()
            else:
                task.custom_output_params["logits"] = logits.cpu().numpy().tolist()
                task.custom_output_params["score"] = logits.max().item()

            return TaskStatus.SUCCESS

        except Exception as e:
            print(f"PyTorchWorker reward error: {e}")
            traceback.print_exc()
            return TaskStatus.WORKER_EXCEPTION

    def shutdown(self):
        """Clean up resources by moving model to CPU and freeing GPU memory."""
        if hasattr(self.model, "cpu"):
            self.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    task_handlers = {
        GenerationTask: generation_handler,
        StreamGenerationTask: stream_generation_handler,
        RewardTask: reward_handler,
    }
