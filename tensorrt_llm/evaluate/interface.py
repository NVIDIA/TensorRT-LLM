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
import copy
import json
import os
import random
from abc import ABC, abstractmethod
from collections import deque
from itertools import islice
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

import tensorrt_llm.profiler as profiler

from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams


def get_chat_template_kwargs(
        template_owner: Any,
        chat_template_kwargs: Optional[dict[str,
                                            Any]] = None) -> dict[str, Any]:
    """Return effective chat template kwargs for evaluation.

    Some chat templates, such as Qwen3-family templates, enable a long-form
    thinking mode by default. For exact-match style benchmarks, that can consume
    the full generation budget before the model reaches its final answer. Keep
    reasoning disabled unless the caller explicitly opts in.
    """
    effective_kwargs = dict(chat_template_kwargs or {})
    owner = getattr(template_owner, "tokenizer", template_owner)
    chat_template = getattr(owner, "chat_template", None)
    if isinstance(chat_template, str) and "enable_thinking" in chat_template:
        effective_kwargs.setdefault("enable_thinking", False)
    return effective_kwargs


def get_model_context(llm: Any) -> tuple[str, str]:
    """Return the HF model directory and model type for an LLM object."""
    model_dir = getattr(llm, "_hf_model_dir", None) or getattr(
        llm, "model", None)
    if model_dir is None:
        raise ValueError("The LLM object does not expose a model directory.")

    config_path = os.path.join(str(model_dir), "config.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_type = config.get("model_type")
    if model_type is None:
        raise KeyError(f"'model_type' is missing from {config_path}.")
    return str(model_dir), str(model_type)


def resolve_in_flight_window(llm: Any, bound_in_flight: bool) -> Optional[int]:
    """In-flight submission cap for `generate_windowed`."""
    if not bound_in_flight:
        return None
    max_batch_size = llm.args.max_batch_size
    if isinstance(max_batch_size, int) and max_batch_size > 0:
        return max_batch_size
    return None


def generate_windowed(
    submit: Callable[[Any], RequestOutput],
    items: Sequence[Any],
    window: Optional[int],
    *,
    submit_desc: str = "Submitting requests",
    fetch_desc: str = "Fetching responses",
    disable_tqdm: bool = False,
) -> List[RequestOutput]:
    """Submit at most `window` items at any given time.

    `submit` maps one item to a `RequestOutput` future (e.g. the return of `llm.generate_async`).
    Results are returned in input order. When `window` is `None` or `>= len(items)` the original
    submit-all-then-fetch flow is used (every request submitted up front).

    Bounding the in-flight window curbs how many `file_system` SharedTensors (multimodal payloads
    crossing to the worker as `/dev/shm/torch_*`) are created / held / torn down concurrently,
    which narrows the teardown race.
    """
    if window is None or window >= len(items):
        # No effective cap: keep the original submit-all-then-fetch flow.
        futures = [
            submit(item)
            for item in tqdm(items, desc=submit_desc, disable=disable_tqdm)
        ]
        return [
            future.result()
            for future in tqdm(futures, desc=fetch_desc, disable=disable_tqdm)
        ]

    # Prime the window, then refill one request each time the oldest in-flight request is retired
    # (FIFO), so at most `window` requests are ever in flight and output order matches input order.
    outputs: List[RequestOutput] = []
    pending: deque = deque()
    item_iter = iter(items)
    for item in islice(item_iter, window):
        pending.append(submit(item))
    with tqdm(total=len(items), desc=fetch_desc,
              disable=disable_tqdm) as progress:
        for item in item_iter:
            outputs.append(pending.popleft().result())
            progress.update(1)
            pending.append(submit(item))
        while pending:
            outputs.append(pending.popleft().result())
            progress.update(1)
    return outputs


class Evaluator(ABC):

    def __init__(self,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 fewshot_as_multiturn: bool = False,
                 system_prompt: Optional[str] = None,
                 chat_template_kwargs: Optional[dict[str, Any]] = None,
                 output_dir: Optional[str] = None,
                 bound_in_flight: bool = False):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.system_prompt = system_prompt
        self.chat_template_kwargs = chat_template_kwargs
        self.output_dir = output_dir
        # When `True`, cap concurrently in-flight requests to the engine's `max_batch_size` instead
        # of submitting every request up front (see `generate_windowed`). Off by default.
        self.bound_in_flight = bound_in_flight

    @abstractmethod
    def generate_samples(self) -> Iterable[tuple]:
        raise NotImplementedError()

    @abstractmethod
    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries) -> float:
        raise NotImplementedError()

    def do_apply_chat_template(self, llm: Any,
                               prompt: Union[str, List[dict]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if self.system_prompt is not None:
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }] + messages
        chat_template_kwargs = get_chat_template_kwargs(
            llm.tokenizer, self.chat_template_kwargs)
        return llm.tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True,
                                                 **chat_template_kwargs)

    def _get_sampline_params(self, sampling_params: Optional[SamplingParams],
                             sampling_args: Optional[dict]) -> SamplingParams:
        if sampling_params is None:
            sampling_params = SamplingParams()
        else:
            sampling_params = copy.deepcopy(sampling_params)

        if sampling_args is not None:
            for key, value in sampling_args.items():
                setattr(sampling_params, key, value)
        return sampling_params

    def evaluate(self,
                 llm: Any,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False) -> float:
        profiler.start("trtllm exec")
        prepared, references, auxiliaries = [], [], []
        for prompt, sampling_args, reference, *aux in tqdm(
                self.generate_samples(), desc="Preparing requests"):
            if self.apply_chat_template:
                prompt = self.do_apply_chat_template(llm, prompt)
            sampling_params = self._get_sampline_params(sampling_params,
                                                        sampling_args)
            prepared.append((prompt, sampling_params))
            references.append(reference)
            auxiliaries.append(aux)

        def _submit(item):
            prompt, params = item
            return llm.generate_async(prompt, params, streaming=streaming)

        window = resolve_in_flight_window(llm, self.bound_in_flight)
        results = generate_windowed(_submit, prepared, window)

        if self.output_dir:
            dump_inference_results(self.output_dir, results,
                                   getattr(llm, 'tokenizer', None))

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        score = self.compute_score(results, references, *zip(*auxiliaries))
        return score

    @staticmethod
    def command(ctx, *args, **kwargs) -> None:
        raise NotImplementedError()


def dump_inference_results(output_dir: str, results: List[dict],
                           tokenizer: Any):
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect results
    results_list = []
    for task_id, result in enumerate(results):
        output_ids = result.outputs[0].token_ids
        output_text = result.outputs[0].text.strip()
        input_text = result.prompt.strip()
        input_ids = tokenizer.encode(input_text)
        results_list.append({
            "task_id": task_id,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "input_text": input_text,
            "output_text": output_text
        })

    # Dump token ids
    ids_path = os.path.join(output_dir, "dumped_ids.json")
    try:
        with open(ids_path, "w") as f:
            for item in results_list:
                data = {
                    "task_id": item["task_id"],
                    "input_ids": item["input_ids"],
                    "output_ids": item["output_ids"],
                    "input_tokens": len(item["input_ids"]),
                    "output_tokens": len(item["output_ids"])
                }
                f.write(json.dumps(data) + "\n")
        logger.info(f"Dumped IDs to {ids_path}")
    except Exception as e:
        logger.warning(f"Failed to dump IDs to {ids_path}: {e}")

    # Dump text
    text_path = os.path.join(output_dir, "dumped_text.json")
    try:
        with open(text_path, "w") as f:
            for item in results_list:
                data = {
                    "task_id": item["task_id"],
                    "input_text": item["input_text"],
                    "output_text": item["output_text"],
                    "input_len": len(item["input_text"]),
                    "output_len": len(item["output_text"])
                }
                f.write(json.dumps(data) + "\n")
        logger.info(f"Dumped text to {text_path}")
    except Exception as e:
        logger.warning(f"Failed to dump text to {text_path}: {e}")
