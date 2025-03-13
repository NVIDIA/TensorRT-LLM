import json
from functools import partial
from typing import List, TextIO, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer

from tensorrt_llm.bench.dataclasses.general import (DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats


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
) -> Tuple[DatasetMetadata, List[InferenceRequest]]:
    """Generate metadata and a list of requests to drive benchmarking.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        stream (TextIO): Stream of input requests.
        max_input_length (int, optional): Maximum input length to cap prompts to. Defaults to 0.
        max_output_length (int, optional): Maximum output length to cap prompts to.. Defaults to 0.
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
    all_isl = []
    all_osl = []
    all_seq_len = []
    while (line := stream.readline()) and len(dataset) < max_requests:
        # We expect the data to come in as a JSON string.
        # For example:
        # {"prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000}
        # Each line should be a complete JSON dictionary with no indentation
        # or newline characters.
        data = json.loads(line)
        logits = data.get("input_ids", data.get("logits", None))
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
            input_ids=logits,
        )
        all_isl.append(len(logits))
        all_osl.append(osl)
        all_seq_len.append(len(logits) + osl)
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
