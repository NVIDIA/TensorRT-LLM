import json
from functools import partial
from typing import List, TextIO, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizer

from tensorrt_llm.bench.dataclasses.general import (DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs import default_multimodal_input_loader


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
    all_osl = []
    prompts = []
    media_paths = []
    all_logits = []
    task_ids = []
    lora_requests = []
    while (line := stream.readline()) and len(task_ids) < max_requests:
        # We expect the data to come in as a JSON string.
        # For example:
        # {"prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000}
        #
        # For multimodal data, the data should be of the form:
        # {"prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000,
        # "media_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"]}
        #
        # For LoRA data, the data should be of the form:
        # {"prompt": "Generate an infinite response to the following:
        # There once was a man who.", "output_tokens": 1000,
        # "lora_request": {"lora_name": "my_lora", "lora_int_id": 1, "lora_path": "/path/to/lora"}}
        #
        # Each line should be a complete JSON dictionary with no indentation
        # or newline characters.
        data = json.loads(line)
        prompts.append(data.get("prompt"))
        media_paths.append(data.get("media_paths", None))
        all_logits.append(data.get("input_ids", data.get("logits", None)))
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
    for prompt, logits, osl, task_id, lora_request in zip(
            prompts, all_logits, all_osl, task_ids, lora_requests):
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
        all_seq_len.append(cur_isl + osl)

        request = InferenceRequest(
            task_id=task_id,
            prompt=prompt,
            output_tokens=output_limiter(osl),
            input_ids=logits,
            lora_request=lora_request,
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
