import json
import sys
from functools import partial
from pathlib import Path
from select import select
from typing import List, TextIO, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer

from tensorrt_llm.bench.dataclasses import DatasetMetadata, InferenceRequest


def generate_dataset_from_stream(dataset_path: Path,
                                 model: str,
                                 num_requests: int = 0):
    # Check for data on stdin.
    data_on_stdin: bool = bool(len(select([
        sys.stdin,
    ], [], [], 0.0)[0]))

    # Cannot set the data file path and pipe in from stdin. Choose one.
    if dataset_path is not None and data_on_stdin:
        raise ValueError(
            "Cannot provide a dataset on both stdin and by --dataset option. "
            "Please pick one.")
    # If we are receiving data from a path or stdin, parse and gather metadata.
    stream = sys.stdin if data_on_stdin else open(dataset_path, "r")
    tokenizer = initialize_tokenizer(model)
    # Parse the dataset from stdin and return it plus its metadata.
    metadata, requests = \
        create_dataset_from_stream(
            tokenizer,
            stream=stream,
            num_requests=num_requests
        )

    return metadata, requests


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
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def create_dataset_from_stream(
    tokenizer: PreTrainedTokenizer,
    max_input_length: int = 0,
    max_output_length: int = 0,
    stream: TextIO = sys.stdin,
    num_requests: int = 0,
) -> Tuple[DatasetMetadata, List[InferenceRequest]]:
    """Generate metadata and a list of requests to drive benchmarking.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        max_input_length (int): Maximum input length to cap prompts to.

    Returns:
        DatasetMetadata: Dataclass of dataset statistics.
        List[InferenceRequest]: A list of inference requests for benchmarking.
    """
    # Initialize dataset list, and metadata tracking variables.
    dataset = []
    max_isl = 0
    max_osl = 0
    max_sequence = 0
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
    while (line := stream.readline()) and len(dataset) < max_requests:
        # We expect the data to come in as a JSON string.
        # For example:
        # {"prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000}
        # Each line should be a complete JSON dictionary with no indentation
        # or newline characters.
        data = json.loads(line)
        logits = data.get("logits", None)
        prompt = data.get("prompt", None)
        task_id = data["task_id"]
        osl = data["output_tokens"]
        # If the request comes in with logits, just use the provided.
        # Otherwise we need to tokenize it.
        logits = tokenize(prompt)["input_ids"] if logits is None else logits

        request = InferenceRequest(
            task_id=task_id,
            prompt=prompt,
            output_tokens=output_limiter(osl),
            logits=logits,
        )
        max_isl = max(max_isl, len(logits))
        max_osl = max(max_osl, osl)
        max_sequence = max(max_sequence, len(logits) + osl)
        dataset.append(request)

    # Fill in basic dataset metrics here
    # TODO: Maybe fill this out to be more complete?
    metadata = DatasetMetadata(
        max_isl=max_isl,
        max_osl=max_osl,
        max_sequence_length=max_sequence,
        num_requests=len(dataset),
    )

    return metadata, dataset
