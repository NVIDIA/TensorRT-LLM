# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from functools import partial
from typing import List, TextIO, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer

from tensorrt_llm.bench.dataclasses.general import (DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs import default_multimodal_input_loader


class DatasetFormatError(ValueError):
    """Raised when the input dataset stream is empty, corrupted, or incorrectly formatted."""


def initialize_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Initialize a tokenizer.

    Args:
        model_name (str): The name of the HuggingFace model to pull a
        tokenizer from.

    Returns:
        PreTrainedTokenizer: An initialized HuggingFace tokenizer.
    """
    # Initialize the tokenizer specific to the model that we are planning
    # to benchmark.
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side="left",
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def create_dataset_from_stream(
    tokenizer: PreTrainedTokenizer,
    stream: TextIO,
    max_input_length: int = 0,
    max_output_length: int = 0,
    num_requests: int = 0,
    model_dir: str = None,
    model_type: str = None,
    modality: str = None,
    image_data_format: str = "pt",
    data_device: str = "cpu",
    max_input_seq_len_for_multimodal: int = 4096,
) -> Tuple[DatasetMetadata, List[InferenceRequest]]:
    """Generate metadata and a list of requests to drive benchmarking.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        stream (TextIO): Stream of input requests.
        max_input_length (int, optional): Maximum input length to cap prompts to. Defaults to 0.
        max_output_length (int, optional): Maximum output length to cap prompts to. Defaults to 0.
        num_requests (int, optional): Number of requests to limit to. Defaults to 0.

    Returns:
        Tuple[DatasetMetadata, List[InferenceRequest]]: A tuple containing a dataclass of dataset
        statistics and a list of inference requests for benchmarking.
    """
    # Initialize dataset list, and metadata tracking variables.
    dataset = []
    max_requests = num_requests if num_requests > 0 else float("inf")

    # If we're limiting the input length to a certain size, then set up
    # a partial to truncate the data down to size. Otherwise, just use the
    # unmodified tokenizer callable.
    tokenize = (partial(
        tokenizer,
        padding="max_length",
        max_length=max_input_length,
        truncation=True,
    ) if max_input_length > 0 else tokenizer)

    # If we need to limit the output length, fill in a partial callable
    # for max, otherwise a lambda that just returns x with no bounds.
    output_limiter = (partial(max, max_output_length)
                      if max_output_length > 0 else lambda x: x)

    # For each line in the standard input, parse out the JSON string we expect
    # to see.
    # Note the := walrus -- we're assigning and checking the condition.
    all_osl = []
    prompts = []
    media_paths = []
    all_logits = []
    task_ids = []
    lora_requests = []
    all_turns = []
    all_categories = []
    all_question_ids = []
    while (line := stream.readline()) and len(task_ids) < max_requests:
        # We support two JSONL formats:
        #
        # 1. Standard single-turn format:
        # {"task_id": 1, "prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000}
        #
        # 2. Multi-turn format (e.g. MT-Bench question.jsonl):
        # {"question_id": 81, "category": "writing", "turns": ["Write a blog post...", "Rewrite..."]}
        # When "turns" is present, the first turn is used as the prompt for
        # tokenization/metadata, and all turns are stored for sequential
        # multi-turn benchmarking.  "output_tokens" is required, same as
        # single-turn format.
        #
        # For multimodal data, the data should be of the form:
        # {"task_id": 1, "prompt": "...", "output_tokens": 1000,
        # "media_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"]}
        #
        # For LoRA data, the data should be of the form:
        # {"task_id": 1, "prompt": "...", "output_tokens": 1000,
        # "lora_request": {"lora_name": "my_lora", "lora_int_id": 1, "lora_path": "/path/to/lora"}}
        #
        # Each line should be a complete JSON dictionary with no indentation
        # or newline characters.
        data = json.loads(line)

        turns = data.get("turns")
        if turns is not None and isinstance(turns, list):
            prompts.append(data.get("prompt") or turns[0])
            media_paths.append(data.get("media_paths", None))
            all_logits.append(data.get("input_ids", data.get("logits", None)))
            all_turns.append(turns)
            all_categories.append(data.get("category"))
            all_question_ids.append(data.get("question_id"))
            all_osl.append(data.get("output_tokens"))
            task_ids.append(
                data.get("task_id", data.get("question_id", len(task_ids))))
        else:
            prompts.append(data.get("prompt"))
            media_paths.append(data.get("media_paths", None))
            all_logits.append(data.get("input_ids", data.get("logits", None)))
            all_turns.append(None)
            all_categories.append(None)
            all_question_ids.append(None)
            all_osl.append(data.get("output_tokens"))
            task_ids.append(data.get("task_id"))

        # Parse LoRA request if present
        lora_data = data.get("lora_request", None)
        if lora_data:
            lora_request = LoRARequest(lora_name=lora_data["lora_name"],
                                       lora_int_id=lora_data["lora_int_id"],
                                       lora_path=lora_data.get("lora_path", ""))
            lora_requests.append(lora_request)
        else:
            lora_requests.append(None)

    # Early validation: check if any data was actually read from the stream
    if len(prompts) == 0:
        raise DatasetFormatError(
            "No data was read from the dataset stream. "
            "The dataset file may be empty, corrupted, or in an incorrect format. "
            "Expected JSON lines with at least 'prompt', 'task_id' and 'output_tokens' fields."
        )

    if modality is not None:
        # Multimodal data need extra preprocessing
        assert modality in [
            "image", "video"
        ], f"Modality must be one of ['image', 'video'] but got {modality}."
        prompts = default_multimodal_input_loader(
            tokenizer=tokenizer,
            model_dir=model_dir,
            model_type=model_type,
            modality=modality,
            prompts=prompts,
            media=media_paths,  # list of dicts
            image_data_format=image_data_format,
            device=data_device)

    all_isl = []
    all_seq_len = []
    for prompt, logits, osl, task_id, lora_request, turns, category, question_id in zip(
            prompts, all_logits, all_osl, task_ids, lora_requests, all_turns,
            all_categories, all_question_ids):
        if modality is not None:
            # NOTE: we cannot tokenize multi-modal data, handled by preprocessor
            #       so the actual sequence length is unknown until the model is run
            logits = None
            cur_isl = max_input_seq_len_for_multimodal
        else:
            # If the request comes in with logits, just use the provided.
            # Otherwise we need to tokenize it.
            logits = tokenize(prompt)["input_ids"] if logits is None else logits
            cur_isl = len(logits)
        all_isl.append(cur_isl)
        num_turns = len(turns) if turns is not None else 1
        all_seq_len.append(cur_isl + num_turns * osl)

        request = InferenceRequest(
            task_id=task_id,
            prompt=prompt,
            output_tokens=output_limiter(osl),
            input_ids=logits,
            lora_request=lora_request,
            turns=turns,
            category=category,
            question_id=question_id,
        )
        dataset.append(request)

    isl_stats = PercentileStats.from_iterable(all_isl)
    osl_stats = PercentileStats.from_iterable(all_osl)
    seq_len_stats = PercentileStats.from_iterable(all_seq_len)

    # Fill in basic dataset metrics here
    metadata = DatasetMetadata(
        isl_stats=isl_stats,
        osl_stats=osl_stats,
        seq_len_stats=seq_len_stats,
        num_requests=len(dataset),
    )

    return metadata, dataset


def update_metadata_for_multimodal(metadata, statistics) -> DatasetMetadata:
    """Update the metadata from benchmark statistics. Only used for multimodal models.

    Args:
        metadata (DatasetMetadata): The metadata to update.
        statistics (StatsKeeper): The statistics to update the metadata with.

    Returns:
        DatasetMetadata: The updated metadata.
    """
    all_isl = []
    all_osl = []
    all_seq_len = []
    for request in statistics.requests.values():
        all_isl.append(request.num_input_tokens)
        all_osl.append(request.num_total_output_tokens)
        all_seq_len.append(request.num_input_tokens +
                           request.num_total_output_tokens)
    isl_stats = PercentileStats.from_iterable(all_isl)
    osl_stats = PercentileStats.from_iterable(all_osl)
    seq_len_stats = PercentileStats.from_iterable(all_seq_len)
    metadata.isl_stats = isl_stats
    metadata.osl_stats = osl_stats
    metadata.seq_len_stats = seq_len_stats

    return metadata
