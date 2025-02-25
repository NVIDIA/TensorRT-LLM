"""A simple config for Llama-2 building and generating scripts.

Modify directly if you want to change settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


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
    # If no `model` argument is provided, the checkpoint directory is used to infer the model
    # architecture.
    model: Optional[str] = None
    skip_loading_weights: bool = False

    ### MODEL EXTRA KWARGS #########################################################################
    # Extra kwargs for the model config class to customize the model config. Those arguments will
    # take precedence over the default values or config values in the model config file in the HF
    # directory. Arguments are resolved in the following order:
    # 1. Default values in the model config class
    # 2. Values in the model config file in the HF directory
    # 3. Values in the model_kwargs
    # Note that that if the kwarg does not exist in the model config class, it will be ignored.
    # An example model config class can be found [here](https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/models/llama/configuration_llama.py#L26).
    model_kwargs: Dict = field(
        default_factory=lambda: {
            "max_position_embeddings": 4096,  # to save on memory
            "use_cache": False,
        }
    )

    ### CONFIGURE MODEL FACTORY, BACKEND, RUNTIME, AND WORLD SIZE ##################################
    world_size: int = 1  # choose from number of GPUs for TP (0--> no TP, no spawned processes)
    runtime: str = "demollm"  # chose from "demollm" or "trtllm" (production-grade runtime)
    compile_backend: str = "torch-opt"  # choose from "torch-simple", "torch-opt"
    attn_backend: str = "TritonWithFlattenedInputs"  # "TritonWithFlattenedInputs" or "FlashInfer"
    max_seq_len: int = 512  # max sequence length for inference/cache
    max_batch_size: int = 8  # max dimension for statically allocated kv cache
    page_size: int = 64  # page size for attention

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
    benchmark: bool = False  # If true, set ISO to 2048 random int and OSL to 128
    benchmark_num: int = 10  # By default run 10 times and get average
    benchmark_isl: int = 2048  # input seq length for benchmarking
    benchmark_osl: int = 128  # output seq length for benchmarking
    benchmark_bs: int = 1  # batch size for benchmarking
    benchmark_results_path: Optional[str] = "./benchmark_results.json"

    ### POST INITIALIZATION ########################################################################
    def __post_init__(self):
        # check if model was supplied
        assert self.model, "model must be supplied!"

        # we don't want to loose the default values for model_kwargs unless explicitly set by the
        # user. They are not preserved by the standard initialization process since they whole dict
        # gets replaced by the user provided one. We don't want that though.
        f_default = self.__dataclass_fields__["model_kwargs"].default_factory()
        setattr(self, "model_kwargs", {**f_default, **getattr(self, "model_kwargs")})

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
        if self.attn_backend == "TritonWithFlattenedInputs":
            self.page_size = self.max_seq_len

        # use min instead of max to avoid OOM for large batch size
        self.model_kwargs["max_position_embeddings"] = min(
            self.max_seq_len,
            self.model_kwargs["max_position_embeddings"],
        )

        if isinstance(self.prompt, str):
            self.prompt = [self.prompt]

        # replicate prompts to get to batch_size
        prompts = self.prompt * (self.batch_size // len(self.prompt) + 1)
        self.prompt = prompts[: self.batch_size]
