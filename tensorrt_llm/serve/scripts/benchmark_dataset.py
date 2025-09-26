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

import base64
import io
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.inputs.utils import convert_image_mode
from tensorrt_llm.serve.scripts.benchmark_utils import download_and_cache_file

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


def timing_decorator(method_name: str):
    """
    Decorator to time method execution and print the results.

    Args:
        method_name: Name to display in timing output (e.g., 'load_data', 'sample')
    """

    def decorator(func):

        def wrapper(self, *args, **kwargs):
            dataset_name = self.__class__.__name__
            start_time = time.perf_counter()
            print(f"{dataset_name}.{method_name}() started...")

            try:
                result = func(self, *args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time
                print(
                    f"{dataset_name}.{method_name}() completed in {duration:.4f} seconds"
                )
                return result
            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time
                print(
                    f"{dataset_name}.{method_name}() failed after {duration:.4f} seconds: {str(e)}"
                )
                raise

        return wrapper

    return decorator


def auto_time_methods(*method_names):
    """
    Class decorator that automatically applies timing to specified methods
    in the class and all its subclasses.

    Usage:
        @auto_time_methods("load_data", "sample")
        class MyDataset(BenchmarkDataset):
            def load_data(self):  # Will be automatically timed
                pass
            def sample(self):     # Will be automatically timed
                pass
    """

    def class_decorator(cls):
        # Store the method names that should be timed
        cls._timed_methods = method_names

        # Override __init_subclass__ to automatically apply timing to subclasses
        original_init_subclass = getattr(cls, '__init_subclass__',
                                         lambda **kwargs: None)

        @classmethod
        def __init_subclass__(subcls, **kwargs):
            original_init_subclass(**kwargs)

            # Apply timing to the specified methods if they exist in the subclass
            for method_name in method_names:
                if hasattr(subcls, method_name):
                    original_method = getattr(subcls, method_name)

                    # Only wrap if not already wrapped (check for our wrapper's signature)
                    if not hasattr(original_method, '_is_timed'):
                        timed_method = timing_decorator(method_name)(
                            original_method)
                        timed_method._is_timed = True
                        setattr(subcls, method_name, timed_method)

        cls.__init_subclass__ = __init_subclass__

        # Also apply timing to methods in the current class
        for method_name in method_names:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                if not hasattr(original_method, '_is_timed'):
                    timed_method = timing_decorator(method_name)(
                        original_method)
                    timed_method._is_timed = True
                    setattr(cls, method_name, timed_method)

        return cls

    return class_decorator


def batch_tokenize_prompts(
        prompts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 1000,
        progress_name: str = "prompts") -> tuple[list[int], list[list[int]]]:
    """
    Efficiently tokenize a list of prompts using batch processing.

    Args:
        prompts: List of text prompts to tokenize
        tokenizer: The tokenizer to use
        batch_size: Number of prompts to process in each batch
        progress_name: Name to show in progress messages

    Returns:
        Tuple of (prompt_lengths, prompt_token_ids) where:
        - prompt_lengths: List of prompt lengths (number of tokens per prompt)
        - prompt_token_ids: List of token ID lists for each prompt
    """
    import time

    if not prompts:
        return [], []

    print(
        f"Batch tokenizing {len(prompts)} {progress_name} (batch_size={batch_size})..."
    )

    prompt_lengths = []
    prompt_token_ids = []
    total_time = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Batch tokenization
        start_time = time.perf_counter()
        batch_encoded = tokenizer(batch_prompts,
                                  padding=False,
                                  truncation=False)
        batch_time = time.perf_counter() - start_time
        total_time += batch_time

        # Extract lengths and token IDs
        for j in range(len(batch_prompts)):
            token_ids = batch_encoded.input_ids[j]
            prompt_lengths.append(len(token_ids))
            prompt_token_ids.append(token_ids)

        # Progress reporting
        if (i + batch_size) % 5000 == 0 or (i + batch_size) >= len(prompts):
            processed = min(i + batch_size, len(prompts))
            avg_time = total_time / processed * 1000
            print(
                f"  Processed {processed}/{len(prompts)} {progress_name} - Avg: {avg_time:.2f}ms per item"
            )

    avg_time_per_prompt = total_time / len(prompts) * 1000
    print(
        f"Batch tokenization completed: {total_time:.4f}s total ({avg_time_per_prompt:.2f}ms per {progress_name[:-1]})"
    )

    return prompt_lengths, prompt_token_ids


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[dict] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


@auto_time_methods("load_data", "sample")
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
        self.data = None
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.rng = torch.Generator()
        self.rng.manual_seed(self.random_seed)
        random.seed(self.random_seed)

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
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)

    def apply_multimodal_chat_transformation(self,
                                             prompt: str,
                                             mm_content: Optional[dict] = None
                                             ) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]


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


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        TypeError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = convert_image_mode(image, "RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise TypeError(f"Invalid image input {image}. Must be a PIL.Image.Image"
                    " or str or dictionary with raw image bytes.")


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
        if range_ratio >= 1.0:
            raise ValueError(
                "random_range_ratio must be < 1.0 to ensure a valid sampling range"
            )

        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (torch.randint(
            0, vocab_size, size=(prefix_len, ), generator=self.rng).tolist()
                            if prefix_len > 0 else [])

        # New sampling logic: [X * (1 - b), X * (1 + b)]
        input_low = int(input_len * (1 - range_ratio))
        input_high = int(input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        # Add logging for debugging
        logger.debug("Sampling input_len from [%s, %s]", input_low, input_high)
        logger.debug("Sampling output_len from [%s, %s]", output_low,
                     output_high)

        input_lens = torch.randint(input_low,
                                   input_high + 1,
                                   size=(num_requests, ),
                                   generator=self.rng).tolist()
        output_lens = torch.randint(output_low,
                                    output_high + 1,
                                    size=(num_requests, ),
                                    generator=self.rng).tolist()
        offsets = torch.randint(0,
                                vocab_size,
                                size=(num_requests, ),
                                generator=self.rng).tolist()

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

            # Batch tokenize all prompts first for efficiency
            prompt_lengths, prompt_token_ids = batch_tokenize_prompts(
                dataset, tokenizer, progress_name="random dataset prompts")

            # Filter out sequences that are too long or too short
            requests = []
            dataset_len = len(dataset)

            for i in range(num_requests):
                # Use modulo to cycle through the dataset when num_requests > dataset_len
                dataset_idx = i % dataset_len
                prompt = dataset[dataset_idx]
                initial_prompt_len = prompt_lengths[dataset_idx]
                cached_token_ids = prompt_token_ids[dataset_idx]

                # Skip empty prompt
                if initial_prompt_len == 0:
                    continue

                if initial_prompt_len > input_lens[i]:
                    # Use cached token IDs to avoid re-encoding
                    input_ids = cached_token_ids[:input_lens[i]]
                else:
                    # Re-calculate the prompt length to exclude special tokens.
                    prompt_len = len(
                        tokenizer.encode(prompt, add_special_tokens=False))
                    if prompt_len == 0:
                        continue
                    ratio = (input_lens[i] + prompt_len) // prompt_len
                    prompt = " ".join([prompt] * ratio)
                    prompt_token_ids_for_truncation = tokenizer.encode(prompt)
                    while len(prompt_token_ids_for_truncation) < input_lens[i]:
                        prompt += " " + prompt
                        prompt_token_ids_for_truncation = tokenizer.encode(
                            prompt)
                    input_ids = prompt_token_ids_for_truncation[:input_lens[i]]

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
# Custom Dataset Implementation
# -----------------------------------------------------------------------------


class RandomImageDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 128
    DEFAULT_OUTPUT_LEN = 128
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    DEFAULT_IMAGE_SIZE = 512
    DEFAULT_NUM_IMAGES = 1
    IS_MULTIMODAL = True

    def __init__(
        self,
        return_text: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.return_text = return_text

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_images: int = DEFAULT_NUM_IMAGES,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list[SampleRequest]:
        # Enforce range_ratio < 1
        if range_ratio >= 1.0:
            raise ValueError(
                "random_range_ratio must be < 1.0 to ensure a valid sampling range"
            )

        vocab_size = tokenizer.vocab_size

        prefix_token_ids = (torch.randint(
            0, vocab_size, size=(prefix_len, ), generator=self.rng).tolist()
                            if prefix_len > 0 else [])

        # New sampling logic: [X * (1 - b), X * (1 + b)]
        input_low = int(input_len * (1 - range_ratio))
        input_high = int(input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        # Add logging for debugging
        logger.debug("Sampling input_len from [%s, %s]", input_low, input_high)
        logger.debug("Sampling output_len from [%s, %s]", output_low,
                     output_high)

        input_lens = torch.randint(input_low,
                                   input_high + 1,
                                   size=(num_requests, ),
                                   generator=self.rng).tolist()
        output_lens = torch.randint(output_low,
                                    output_high + 1,
                                    size=(num_requests, ),
                                    generator=self.rng).tolist()
        offsets = torch.randint(0,
                                vocab_size,
                                size=(num_requests, ),
                                generator=self.rng).tolist()

        # Determine final image dimensions
        # When both width/height and image_size are provided, prioritize width/height
        final_width = width
        final_height = height

        # If width and height are still at default values but image_size is different, use image_size
        if (width == self.DEFAULT_WIDTH and height == self.DEFAULT_HEIGHT
                and image_size != self.DEFAULT_IMAGE_SIZE):
            final_width = image_size
            final_height = image_size
        logger.info("Using width: %s, height: %s for random image dimensions",
                    final_width, final_height)
        logger.info("Generating %d images per request", num_images)

        sampled_requests = []
        for i in range(num_requests):
            # Generate random text prompt
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) %
                         vocab_size).tolist()
            prompt = prefix_token_ids + inner_seq
            if self.return_text:
                prompt = tokenizer.decode(prompt)
            total_input_len = prefix_len + int(input_lens[i])

            # Generate random images (support multiple images per request)
            images = []
            for _ in range(num_images):
                random_image = torch.randint(0,
                                             256,
                                             (final_height, final_width, 3),
                                             dtype=torch.uint8,
                                             generator=self.rng).numpy()
                pil_image = Image.fromarray(random_image)
                images.append(pil_image)

            # Process images for multimodal content
            mm_content = [process_image(img) for img in images]

            # Handle multimodal chat transformation
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=mm_content,
                ))

        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


class CustomDataset(BenchmarkDataset):
    """
    TensorRT LLM customized dataset implementation.
    It assumes the dataset to be consist of several lines of json, each line is a minimal OpenAI API format request.
    Example format of each sample on each line:
    {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": ""
                }
            ],
            "max_tokens": 2048,
        }
    }
    """

    def __init__(self, dataset_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.data = []
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("--dataset-path is not provided")
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        random.shuffle(self.data)

    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Optimized version using batch tokenization for better performance.
        """
        # Collect all prompts and metadata
        prompts = []
        max_tokens_list = []

        for i, entry in enumerate(self.data):
            if len(prompts) >= num_requests:
                break
            prompt = entry["input"]["messages"][1]["content"]
            max_tokens = entry["input"]["max_tokens"]
            prompts.append(prompt)
            max_tokens_list.append(max_tokens)

        # Use batch tokenization utility
        prompt_lengths, _ = batch_tokenize_prompts(
            prompts, tokenizer, progress_name="custom dataset prompts")

        # Create SampleRequest objects
        samples = []
        for prompt, prompt_len, max_tokens in zip(prompts, prompt_lengths,
                                                  max_tokens_list):
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=max_tokens,
                ))

        return samples


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
        if enable_multimodal_chat:
            raise NotImplementedError

        # Collect prompts and completions for batch processing
        prompts = []
        completions = []

        for entry in self.data:
            if len(prompts) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )
            prompts.append(prompt)
            completions.append(completion)

        # Batch tokenize prompts and completions
        prompt_lengths, _ = batch_tokenize_prompts(
            prompts, tokenizer, progress_name="ShareGPT prompts")
        completion_lengths, _ = batch_tokenize_prompts(
            completions, tokenizer, progress_name="ShareGPT completions")

        # Filter and create samples
        samples: list = []
        for prompt, completion, prompt_len, completion_len in zip(
                prompts, completions, prompt_lengths, completion_lengths):
            new_output_len = completion_len if output_len is None else output_len
            if not is_valid_sequence(prompt_len,
                                     new_output_len,
                                     skip_min_output_len_check=output_len
                                     is not None):
                continue

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
        # Calculate average token length for poem lines using batch tokenization
        line_lengths, _ = batch_tokenize_prompts(self.data,
                                                 tokenizer,
                                                 progress_name="sonnet lines")
        avg_len = sum(line_lengths) / len(line_lengths)

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
        if enable_multimodal_chat:
            raise NotImplementedError

        # Filter examples with at least 2 conversations and collect data
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        prompts = []
        completions = []
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(prompts) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]
            prompts.append(prompt)
            completions.append(completion)

        # Batch tokenize prompts and completions
        prompt_lengths, _ = batch_tokenize_prompts(
            prompts, tokenizer, progress_name="conversation prompts")
        completion_lengths, _ = batch_tokenize_prompts(
            completions, tokenizer, progress_name="conversation completions")

        # Filter and create samples
        sampled_requests = []
        for prompt, completion, prompt_len, completion_len in zip(
                prompts, completions, prompt_lengths, completion_lengths):
            current_output_len = completion_len if dynamic_output else output_len
            assert isinstance(current_output_len,
                              int) and current_output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len):
                continue

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=current_output_len,
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

        # Collect prompts for batch processing
        prompts = []
        parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)
        if parser_fn is None:
            raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
        sampled_requests = []
        for item in self.data:
            if len(prompts) >= num_requests:
                break
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
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

        # Collect prompts for batch processing
        prompts = []
        for item in self.data:
            if len(prompts) >= num_requests:
                break
            prompt = f"{item['instruction']}:\n{item['input']}"
            prompts.append(prompt)

        # Batch tokenize prompts
        prompt_lengths, _ = batch_tokenize_prompts(
            prompts, tokenizer, progress_name="instruct coder prompts")

        # Create samples
        sampled_requests = []
        for prompt, prompt_len in zip(prompts, prompt_lengths):
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

        # Collect prompts for batch processing
        prompts = []
        for item in self.data:
            if len(prompts) >= num_requests:
                break
            raw_prompt = item['turns'][0]

            # apply template
            formatted_prompt = tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": raw_prompt
                }],
                add_generation_prompt=True,
                tokenize=False)
            prompts.append(formatted_prompt)

        # Batch tokenize prompts
        prompt_lengths, _ = batch_tokenize_prompts(
            prompts, tokenizer, progress_name="MT-Bench prompts")

        # Create samples
        sampled_requests = []
        for prompt, prompt_len in zip(prompts, prompt_lengths):
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
        dynamic_output = output_len is None

        # Collect prompts and completions for batch processing
        prompts = []
        completions = []
        for item in self.data:
            if len(prompts) >= num_requests:
                break
            prompt, completion = item['problem'], item["solution"]
            prompts.append(prompt)
            completions.append(completion)

        # Batch tokenize prompts and completions
        prompt_lengths, _ = batch_tokenize_prompts(prompts,
                                                   tokenizer,
                                                   progress_name="AIMO prompts")
        completion_lengths, _ = batch_tokenize_prompts(
            completions, tokenizer, progress_name="AIMO completions")

        # Filter and create samples
        sampled_requests = []
        for prompt, completion, prompt_len, completion_len in zip(
                prompts, completions, prompt_lengths, completion_lengths):
            current_output_len = completion_len if dynamic_output else output_len
            assert isinstance(current_output_len,
                              int) and current_output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len,
                                                        completion_len,
                                                        max_prompt_len=2048,
                                                        max_total_len=32000):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=current_output_len,
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
