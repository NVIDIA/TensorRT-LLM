# Adopted from
# https://github.com/vllm-project/vllm/blob/200bbf92e8861e2458a6f90bca73f40cc3b1ad1f/benchmarks/benchmark_dataset.py
# https://github.com/sgl-project/sglang/blob/8321f8e45e07a8539935145d1c76373e457ddc89/python/sglang/bench_serving.py
# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena

TODO: Implement CustomDataset to parse a JSON file and convert its contents into
SampleRequest instances, similar to the approach used in ShareGPT.
"""

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from benchmark_utils import download_and_cache_file
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    # Default values copied from benchmark_serving.py for the random dataset.
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    def __init__(
        self,
        return_text: bool = True,
        sample_from_sharegpt: bool = True,
        download_path: Optional[str] = None,
        download_timeout: int = 180,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_from_sharegpt = sample_from_sharegpt
        if self.sample_from_sharegpt:
            self.load_data(download_path, download_timeout)
        self.return_text = return_text

    def load_data(self, download_path: str, download_timeout: int):
        if self.dataset_path is None:
            logger.warning(
                "Dataset is not provided, downloading sharegpt dataset")
            assert download_path is not None, "Please provide a download path to sample from the ShareGPT dataset for more consistent ISL by specifying it with the `--download-path` option. Alternatively, you can use the `--random-ids` option to skip the sampling, which may introduce some unexpected ISL variation even the range ratio is set to 0."
            self.dataset_path = download_and_cache_file(
                RandomDataset.SHAREGPT_URL, download_path,
                RandomDataset.SHAREGPT_URL.split("/")[-1], download_timeout)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        # Enforce range_ratio < 1
        assert range_ratio < 1.0, (
            "random_range_ratio must be < 1.0 to ensure a valid sampling range")

        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (np.random.randint(
            0, vocab_size, size=prefix_len).tolist() if prefix_len > 0 else [])

        # New sampling logic: [X * (1 - b), X * (1 + b)]
        input_low = int(input_len * (1 - range_ratio))
        input_high = int(input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        # Add logging for debugging
        logger.info("Sampling input_len from [%s, %s]", input_low, input_high)
        logger.info("Sampling output_len from [%s, %s]", output_low,
                    output_high)

        input_lens = np.random.randint(input_low,
                                       input_high + 1,
                                       size=num_requests)
        output_lens = np.random.randint(output_low,
                                        output_high + 1,
                                        size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)

        requests = []
        if self.sample_from_sharegpt:
            with open(self.dataset_path) as f:
                dataset = json.load(f)
            # Filter out the conversations with less than 2 turns.
            dataset = [
                data for data in dataset
                if len(data.get("conversations", data.get("conversation", [])))
                >= 2
            ]
            # Only keep the first turn of each conversation.
            dataset = [
                data.get("conversations", data.get("conversation",
                                                   []))[0]["value"].strip()
                for data in dataset
            ]
            # Shuffle the dataset.
            random.shuffle(dataset)

            # Filter out sequences that are too long or too short
            requests = []
            for prompt in dataset:
                i = len(requests)
                if i == num_requests:
                    break

                # Tokenize the prompts and completions.
                prompt_token_ids = tokenizer.encode(prompt)
                prompt_len = len(prompt_token_ids)

                # Skip empty prompt
                if prompt_len == 0:
                    continue

                if prompt_len > input_lens[i]:
                    input_ids = prompt_token_ids[:input_lens[i]]
                else:
                    # Re-calculate the prompt length to exclude special tokens.
                    prompt_len = len(
                        tokenizer.encode(prompt, add_special_tokens=False))
                    if prompt_len == 0:
                        continue
                    ratio = (input_lens[i] + prompt_len) // prompt_len
                    prompt = " ".join([prompt] * ratio)
                    prompt_token_ids = tokenizer.encode(prompt)
                    while len(prompt_token_ids) < input_lens[i]:
                        prompt += " " + prompt
                        prompt_token_ids = tokenizer.encode(prompt)
                    input_ids = prompt_token_ids[:input_lens[i]]

                prompt = prefix_token_ids + input_ids

                if self.return_text:
                    prompt = tokenizer.decode(prompt)

                total_input_len = prefix_len + int(input_lens[i])
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=total_input_len,
                        expected_output_len=int(output_lens[i]),
                    ))
            assert len(requests) == num_requests, (
                f"Only {len(requests)} requests sampled from sharegpt dataset, {num_requests} requests are needed"
            )
        else:
            for i in range(num_requests):
                inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                             vocab_size).tolist()
                prompt = prefix_token_ids + inner_seq
                if self.return_text:
                    prompt = tokenizer.decode(prompt)
                total_input_len = prefix_len + int(input_lens[i])
                requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=total_input_len,
                        expected_output_len=int(output_lens[i]),
                    ))
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """
    URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

    def __init__(self,
                 download_timeout: int,
                 download_path: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data(download_timeout, download_path)

    def load_data(self,
                  download_timeout: int,
                  download_path: Optional[str] = None) -> None:
        if self.dataset_path is None:
            logger.warning("dataset_path is not provided")
            self.dataset_path = download_and_cache_file(
                ShareGPTDataset.URL, download_path,
                ShareGPTDataset.URL.split("/")[-1], download_timeout)

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = (len(completion_ids)
                              if output_len is None else output_len)
            if not is_valid_sequence(prompt_len,
                                     new_output_len,
                                     skip_min_output_len_check=output_len
                                     is not None):
                continue
            if enable_multimodal_chat:
                raise NotImplementedError
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens)
                      for tokens in tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(base_msg,
                                                 add_generation_prompt=True,
                                                 tokenize=False)
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset}).")

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        while len(samples) < num_requests:
            extra_lines = random.choices(self.data,
                                         k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted
                        if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                    ))
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self, ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int) -> list:
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests,
                                    random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
        **kwargs,
    ) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                ))
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: Union[set[str], dict[str, Callable]] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationDataset(HuggingFaceDataset):
    """Dataset for conversation data with multimodal support."""
    SUPPORTED_DATASET_PATHS = {
        'lmms-lab/LLaVA-OneVision-Data', 'Aeala/ShareGPT_Vicuna_unfiltered'
    }
    IS_MULTIMODAL = True

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               **kwargs) -> list:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        sampled_requests = []
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len):
                continue
            if enable_multimodal_chat:
                raise NotImplementedError
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat":
        lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1":
        lambda x: x["turns"][0][0]["content"]
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)
            if parser_fn is None:
                raise ValueError(
                    f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                raise NotImplementedError
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               **kwargs) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = f"{item['instruction']}:\n{item['input']}"
            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """ # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               enable_multimodal_chat: bool = False,
               **kwargs) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item['turns'][0]

            # apply template
            prompt = tokenizer.apply_chat_template([{
                "role": "user",
                "content": prompt
            }],
                                                   add_generation_prompt=True,
                                                   tokenize=False)

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# AIMO Dataset Implementation
# -----------------------------------------------------------------------------


class AIMODataset(HuggingFaceDataset):
    """
    Dataset class for processing a AIMO dataset with reasoning questions.
    """
    SUPPORTED_DATASET_PATHS = {
        "AI-MO/aimo-validation-aime", "AI-MO/NuminaMath-1.5",
        "AI-MO/NuminaMath-CoT"
    }

    def sample(self,
               tokenizer: PreTrainedTokenizerBase,
               num_requests: int,
               output_len: Optional[int] = None,
               **kwargs) -> list:
        sampled_requests = []
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt, completion = item['problem'], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len,
                                                        max_prompt_len=2048,
                                                        max_total_len=32000):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------


class ASRDataset(HuggingFaceDataset):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              |
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """ # noqa: E501
    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr", "facebook/voxpopuli", "LIUM/tedlium",
        "edinburghcstr/ami", "speechcolab/gigaspeech", "kensho/spgispeech"
    }

    DEFAULT_OUTPUT_LEN = 128
    IS_MULTIMODAL = True

    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = "<|startoftranscript|><|en|><|transcribe|>"\
                              "<|notimestamps|>"
    skip_long_audios: bool = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        import librosa
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests = []
        skipped = 0
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            audio = item["audio"]
            y, sr = audio["array"], audio["sampling_rate"]
            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                ))
        if skipped:
            logger.warning("%d samples discarded from dataset due to" \
                           " their length being greater than" \
                           " what Whisper supports.", skipped)
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests
