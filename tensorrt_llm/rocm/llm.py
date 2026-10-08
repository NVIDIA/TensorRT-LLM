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
"""Single-device ROCm LLM execution, independent of TensorRT and native CUDA bindings."""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from trtllm_profile import active_session, component, trace_active

from .runtime import KernelBackend, resolve_device, resolve_dtype
from .sampling import CompletionOutput, RequestOutput, SamplingParams


class _StopStrings(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        strings: list[str],
        prompt_width: int,
        minimum: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.strings = strings
        self.prompt_width = prompt_width
        self.minimum = minimum

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor | None, **kwargs
    ) -> torch.Tensor:
        if input_ids.shape[1] - self.prompt_width < self.minimum:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        texts = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_width :], skip_special_tokens=True
        )
        return torch.tensor(
            [any(stop in text for stop in self.strings) for text in texts],
            dtype=torch.bool,
            device=input_ids.device,
        )


class LLM:
    """Generate text with ROCm PyTorch/Hugging Face on one genuine RDNA4 GPU.

    Unquantized causal language models supported by the installed Transformers
    version can use SDPA or eager attention. ``device='cpu'`` is explicitly for
    reference testing, never an automatic fallback. TensorRT plans, NVIDIA
    quantization plugins, distributed scheduling, and CUDA-specific options are
    rejected instead of being ignored or silently emulated.
    """

    def __init__(
        self,
        model: str | Path | PreTrainedModel,
        tokenizer: str | Path | PreTrainedTokenizerBase | None = None,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "auto",
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        attn_backend: Literal["sdpa", "eager", "hip"] = "sdpa",
        kernels: KernelBackend = "torch",
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **unsupported,
    ) -> None:
        if unsupported:
            raise NotImplementedError(f"Unsupported ROCm options: {', '.join(sorted(unsupported))}")
        if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
            raise NotImplementedError(
                "The ROCm backend currently supports single-device execution only"
            )
        if max_batch_size < 1 or (max_seq_len is not None and max_seq_len < 1):
            raise ValueError("max_batch_size and max_seq_len must be positive")
        if attn_backend not in ("sdpa", "eager", "hip") or kernels not in ("torch", "hip"):
            raise ValueError("Use attn_backend='sdpa'/'eager'/'hip' and kernels='torch'/'hip'")
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        if kernels == "hip" and self.device.type != "cuda":
            raise ValueError(
                "kernels='hip' requires a real RDNA4 GPU; use kernels='torch' for CPU validation"
            )
        if attn_backend == "hip" and kernels != "hip":
            raise ValueError("attn_backend='hip' requires kernels='hip'")
        self._max_batch_size = max_batch_size
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._closed = False
        self.last_stats: dict[str, float | int] = {}
        self.native_norm_count = 0
        self.model_id = (
            str(model)
            if isinstance(model, (str, Path))
            else model.config.name_or_path or type(model).__name__
        )
        loading = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        # Loading must create ordinary versioned parameters, even when called
        # from an outer inference-mode scope (important for dispatch profiling).
        with trace_active(), torch.inference_mode(False):
            if isinstance(model, PreTrainedModel):
                loaded = model
                config = model.config
                if attn_backend == "hip":
                    from .attention import configure_native_attention

                    config._attn_implementation = configure_native_attention(config)
            else:
                path = Path(model)
                if path.suffix in (".engine", ".plan") or (
                    path.is_dir() and not (path / "config.json").is_file()
                ):
                    raise ValueError(
                        "Load a Hugging Face checkpoint, not a TensorRT engine directory/plan"
                    )
                config = AutoConfig.from_pretrained(str(model), **loading)
                self._check_quantization(config)
                implementation = attn_backend
                if attn_backend == "hip":
                    from .attention import configure_native_attention

                    implementation = configure_native_attention(config)
                loaded = AutoModelForCausalLM.from_pretrained(
                    str(model),
                    config=config,
                    torch_dtype=self.dtype,
                    attn_implementation=implementation,
                    **loading,
                )
            self._check_quantization(config)
            if attn_backend == "hip" and not getattr(loaded, "_supports_attention_backend", False):
                raise NotImplementedError(
                    "This Transformers model does not support the native attention interface"
                )
            self.model = loaded.eval().to(device=self.device, dtype=self.dtype)
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = tokenizer
            else:
                source = str(tokenizer) if tokenizer is not None else self.model_id
                self.tokenizer = AutoTokenizer.from_pretrained(source, **loading)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer must define a pad token or an EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            capacities = [
                value
                for value in (
                    getattr(config, "max_position_embeddings", None),
                    getattr(config, "n_positions", None),
                    self.tokenizer.model_max_length,
                )
                if isinstance(value, int) and 0 < value < 100000000
            ]
            capacity = min(capacities) if capacities else None
            if max_seq_len is not None and capacity is not None and max_seq_len > capacity:
                raise ValueError(
                    f"max_seq_len={max_seq_len} exceeds the model/tokenizer capacity {capacity}"
                )
            self._max_seq_len = max_seq_len or capacity
            if kernels == "hip":
                from .kernels import load_kernels
                from .ops import apply_native_norms

                load_kernels(self.device)
                self.native_norm_count = apply_native_norms(self.model)

    @staticmethod
    def _check_quantization(config) -> None:
        if getattr(config, "quantization_config", None):
            raise NotImplementedError(
                "Quantized checkpoints are not enabled by the portable backend. Use FP32/FP16/BF16 weights; "
                "NVIDIA FP4/FP8/bitsandbytes plugins are not ROCm equivalents."
            )

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def _encode(
        self, prompts: list[str] | list[list[int]]
    ) -> tuple[dict[str, torch.Tensor], list[list[int]], list[str]]:
        if isinstance(prompts[0], str):
            if not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("All prompts must have the same type")
            encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
            tokens = [
                ids[mask.bool()].tolist()
                for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            ]
            texts = list(prompts)
            tensors = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        else:
            if not all(
                isinstance(prompt, list)
                and prompt
                and all(isinstance(token, int) and not isinstance(token, bool) for token in prompt)
                for prompt in prompts
            ):
                raise ValueError("Token prompts must be nonempty lists of integer token IDs")
            tokens = [list(prompt) for prompt in prompts]
            width = max(map(len, tokens))
            tensors = {
                "input_ids": torch.tensor(
                    [[self.tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in tokens]
                ),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(ids)) + [1] * len(ids) for ids in tokens]
                ),
            }
            texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=False)
        if any(not ids for ids in tokens):
            raise ValueError("Prompts must contain at least one token")
        vocabulary_size = self.model.get_input_embeddings().num_embeddings
        if any(token < 0 or token >= vocabulary_size for ids in tokens for token in ids):
            raise ValueError("Prompt token ID is outside the model vocabulary")
        with component("transfer"):
            tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}
        return tensors, tokens, texts

    def generate(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        *,
        streaming: bool = False,
    ) -> list[RequestOutput]:
        """Generate a batch, preserving prompt order and excluding EOS from token IDs.

        Stop strings are excluded from text, but their complete boundary tokens
        remain in token_ids (a stop string may end inside a subword token).
        Streaming is intentionally rejected until a compatible streaming API exists.
        """
        if streaming:
            raise NotImplementedError(
                "Streaming on ROCm is not implemented; use non-streaming generate"
            )
        params = sampling_params or SamplingParams()
        if not isinstance(params, SamplingParams):
            raise TypeError("Use tensorrt_llm.rocm.SamplingParams with the ROCm backend")
        if isinstance(prompts, str):
            batches = [prompts]
        elif isinstance(prompts, list) and prompts and isinstance(prompts[0], int):
            batches = [prompts]
        elif isinstance(prompts, list):
            batches = prompts
        else:
            raise TypeError("prompts must be text or a list of text/token prompts")
        with self._lock, trace_active(), torch.inference_mode():
            if self._closed:
                raise RuntimeError("LLM has been shut down")
            results: list[RequestOutput] = []
            started = time.perf_counter()
            model_hooks = nullcontext()
            if active_session() is not None:
                from trtllm_profile.torch_trace import instrument_model

                model_hooks = instrument_model(self.model)
            with model_hooks:
                for offset in range(0, len(batches), self._max_batch_size):
                    batch = batches[offset : offset + self._max_batch_size]
                    encoded, prompt_ids, texts = self._encode(batch)
                    width = encoded["input_ids"].shape[1]
                    if (
                        self._max_seq_len is not None
                        and width + params.max_tokens > self._max_seq_len
                    ):
                        raise ValueError(
                            f"Prompt plus max_tokens exceeds max_seq_len={self._max_seq_len}"
                        )
                    strings = [params.stop] if isinstance(params.stop, str) else (params.stop or [])
                    stopping = (
                        StoppingCriteriaList(
                            [_StopStrings(self.tokenizer, strings, width, params.min_tokens)]
                        )
                        if strings
                        else StoppingCriteriaList()
                    )
                    do_sample = params.temperature > 0
                    end_id = (
                        params.end_id
                        if params.end_id is not None
                        else self.model.generation_config.eos_token_id
                    )
                    if end_id is None:
                        end_id = self.tokenizer.eos_token_id
                    pad_id = (
                        params.pad_id if params.pad_id is not None else self.tokenizer.pad_token_id
                    )
                    vocabulary_size = self.model.get_input_embeddings().num_embeddings
                    eos_ids = (
                        end_id
                        if isinstance(end_id, list)
                        else ([end_id] if end_id is not None else [])
                    )
                    if any(token >= vocabulary_size for token in [pad_id, *eos_ids]):
                        raise ValueError("EOS/pad token ID is outside the model vocabulary")
                    options = {
                        "max_new_tokens": params.max_tokens,
                        "min_new_tokens": params.min_tokens,
                        "do_sample": do_sample,
                        "num_beams": params.beam_width,
                        "num_return_sequences": params.n,
                        "repetition_penalty": params.repetition_penalty,
                        "pad_token_id": pad_id,
                        "eos_token_id": None if params.ignore_eos else end_id,
                        "stopping_criteria": stopping,
                        "use_cache": True,
                        "return_dict_in_generate": False,
                    }
                    if do_sample:
                        options.update(
                            temperature=params.temperature, top_p=params.top_p, top_k=params.top_k
                        )
                    rng = (
                        torch.random.fork_rng(
                            devices=[self.device.index] if self.device.type == "cuda" else []
                        )
                        if params.seed is not None
                        else nullcontext()
                    )
                    with rng:
                        if params.seed is not None:
                            torch.random.default_generator.manual_seed(params.seed)
                            if self.device.type == "cuda":
                                with torch.cuda.device(self.device):
                                    torch.cuda.manual_seed(params.seed)
                        generated = self.model.generate(**encoded, **options)
                    with component("transfer"):
                        generated_ids = generated[:, width:].cpu().tolist()
                    for index, (text, ids) in enumerate(zip(texts, prompt_ids)):
                        outputs = []
                        for completion in range(params.n):
                            tokens = generated_ids[index * params.n + completion]
                            reason = "length"
                            if not params.ignore_eos:
                                endings = [
                                    position
                                    for position, token in enumerate(tokens)
                                    if token in eos_ids
                                ]
                                if endings:
                                    tokens = tokens[: endings[0]]
                                    reason = "stop"
                            output_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                            stop_positions = [
                                output_text.find(stop) for stop in strings if stop in output_text
                            ]
                            if stop_positions:
                                output_text = output_text[: min(stop_positions)]
                                reason = "stop"
                                while tokens and tokens[-1] == pad_id:
                                    tokens = tokens[:-1]
                            outputs.append(
                                CompletionOutput(completion, output_text, tokens, reason)
                            )
                        results.append(RequestOutput(next(self._ids), text, ids, outputs))
            elapsed = time.perf_counter() - started
            input_tokens = sum(len(result.prompt_token_ids) for result in results)
            output_tokens = sum(
                len(output.token_ids) for result in results for output in result.outputs
            )
            self.last_stats = {
                "wall_s": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_s": output_tokens / elapsed if elapsed else 0.0,
            }
            return results

    def shutdown(self) -> None:
        """Release model references; do not clear another application's GPU allocator."""
        with self._lock:
            self._closed = True
            self.model = None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


__all__ = ["LLM"]
