"""A simple config for Llama-2 building and generating scripts.

Modify directly if you want to change settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union


# TODO: remove and unify with _AutoDeployLlmArgs
@dataclass
class SimpleConfig:
    """Experiment Configuration."""

    ### MODEL ARG #############################################################################
    # Path or repo_id for a HF model directory
    # The model directory should contain model weights and tokenizer configs. Model weights can be
    # provided as either of the following:
    # 1. Sharded checkpoint (multiple files) in the safetensors format
    # 2. Single, unsharded checkpoint in the safetensors format
    # 3. Single, unsharded checkpoint in the pytorch format (.pt/.pth) file ending.
    model: str
    # same as model. None defaults to model. Only used if customize_tokenizer is True
    tokenizer: Optional[str] = None
    model_factory: Literal["AutoModelForCausalLM", "AutoModelForImageTextToText"] = (
        "AutoModelForCausalLM"
    )
    skip_loading_weights: bool = False  # only load the architecture, not the weights
    customize_tokenizer: bool = False  # True: tokenizer from the model factory, False: from LLM api

    ### MODEL EXTRA KWARGS #########################################################################
    # Extra kwargs for the model config class to customize the model config. Those arguments will
    # take precedence over the default values or config values in the model config file in the HF
    # directory. Arguments are resolved in the following order:
    # 1. Default values in the model config class
    # 2. Values in the model config file in the HF directory
    # 3. Values in the model_kwargs
    # Note that that if the kwarg does not exist in the model config class, it will be ignored.
    # An example model config class can be found [here](https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/models/llama/configuration_llama.py#L26).
    model_kwargs: Dict = field(default_factory=dict)

    ### TOKENIZER EXTRA KWARGS #####################################################################
    # Extra kwargs for the tokenizer class to customize the tokenizer. Same as model_kwargs.
    # For example, the default HF Llama tokenizer can be initialized with the arguments specified
    # [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py#L127).
    # NOTE: This is only used if customize_tokenizer is True
    tokenizer_kwargs: Dict = field(default_factory=dict)

    ### CONFIGURE BACKEND, RUNTIME, AND WORLD SIZE ##################################
    world_size: int = 1  # choose from number of GPUs for TP (0--> no TP, no spawned processes)
    runtime: Literal["demollm", "trtllm"] = "trtllm"
    compile_backend: Literal["torch-simple", "torch-compile", "torch-cudagraph", "torch-opt"] = (
        "torch-compile"
    )
    attn_backend: Literal["TritonWithFlattenedInputs", "FlashInfer"] = "FlashInfer"
    mla_backend: Literal["MultiHeadLatentAttention"] = "MultiHeadLatentAttention"
    max_seq_len: int = 512  # max sequence length for inference/cache
    max_batch_size: int = 8  # max dimension for statically allocated kv cache
    attn_page_size: int = 64  # page size for attention
    simple_shard_only: bool = False  # if True, force simple sharding(all_gather) in TP;
    # otherwise auto-detect and use column+row (all_reduce) sharding

    ### SOME SIMPLE PROMPTING CONFIG ###############################################################
    batch_size: int = 2  # example input shape
    device: str = "cuda"
    prompt: Union[str, List[str]] = field(
        default_factory=lambda: [
            "How big is the universe? ",
            "In simple words and in a single sentence, explain the concept of gravity: ",
            "How to fix slicing in golf? ",
            "Where is the capital of Iceland? ",
            "How big is the universe? ",
            "In simple words and in a single sentence, explain the concept of gravity: ",
            "How to fix slicing in golf? ",
            "Where is the capital of Iceland? ",
        ]
    )
    max_tokens: int = 100
    top_k: int = 200
    temperature: float = 1.0
    visualize: bool = False

    ### BENCHMARKING CONFIG ########################################################################
    free_mem_ratio: float = 0.0  # specifies the fraction of available memory to occupy for cache
    benchmark: bool = False  # If true, set ISO to 2048 random int and OSL to 128
    benchmark_num: int = 10  # By default run 10 times and get average
    benchmark_isl: int = 2048  # input seq length for benchmarking
    benchmark_osl: int = 128  # output seq length for benchmarking
    benchmark_bs: int = 1  # batch size for benchmarking
    benchmark_results_path: Optional[str] = "./benchmark_results.json"
    benchmark_store_results: bool = False  # if True, store benchmark res in benchmark_results_path

    ### POST INITIALIZATION ########################################################################
    def __post_init__(self):
        # check if model was supplied
        assert self.model, "model must be supplied!"

        # NEVER use cache
        self.model_kwargs["use_cache"] = False

        # special handling for torch_dtype in model_kwargs since HF does not correctly update
        # torch_dtype string to an actual torch.dtype object (only with default)
        if "torch_dtype" in self.model_kwargs:
            import torch

            dtype = self.model_kwargs["torch_dtype"]
            if isinstance(dtype, str):
                dtype = getattr(torch, self.model_kwargs["torch_dtype"])
            assert isinstance(dtype, torch.dtype), f"Invalid dtype: {dtype}"
            self.model_kwargs["torch_dtype"] = dtype

        self.max_batch_size = max(self.max_batch_size, self.batch_size)

        # make sure benchmark isl/osl/bs fits into available resources
        if self.benchmark:
            self.max_batch_size = max(self.benchmark_bs, self.max_batch_size)
            self.max_seq_len = max(self.max_seq_len, self.benchmark_isl + self.benchmark_osl)

        # No paging allowed in TritonWithFlattenedInputs
        if self.attn_backend in ["TritonWithFlattenedInputs"]:
            self.attn_page_size = self.max_seq_len

        # use min instead of max to avoid OOM for large batch size
        self.model_kwargs["max_position_embeddings"] = min(
            self.max_seq_len,
            self.model_kwargs.get("max_position_embeddings", self.max_seq_len),
        )

        if isinstance(self.prompt, str):
            self.prompt = [self.prompt]

        # replicate prompts to get to batch_size
        prompts = self.prompt * (self.batch_size // len(self.prompt) + 1)
        self.prompt = prompts[: self.batch_size]
