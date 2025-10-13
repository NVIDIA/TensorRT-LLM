from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import click
from click_option_group import OptionGroup, optgroup

from tensorrt_llm.llmapi.llm_args import BaseLlmArgs
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory


class ChoiceWithAlias(click.Choice):

    def __init__(self,
                 choices: Sequence[str],
                 aliases: Mapping[str, str],
                 case_sensitive: bool = True) -> None:
        super().__init__(choices, case_sensitive)
        self.aliases = aliases

    def to_info_dict(self) -> Dict[str, Any]:
        info_dict = super().to_info_dict()
        info_dict["aliases"] = self.aliases
        return info_dict

    def convert(self, value: Any, param: Optional["click.Parameter"],
                ctx: Optional["click.Context"]) -> Any:
        if value in self.aliases:
            value = self.aliases[value]
        return super().convert(value, param, ctx)


def common_llm_options(f: Callable) -> Callable:
    """Apply all common LLM API options to a command.

    This decorator adds all LLM API-related configuration options and organizes them into
    logical groups.
    """
    # Model and backend configuration
    f = optgroup.group("Model Configuration",
                       help="Model, tokenizer, and backend settings.")(f)
    f = optgroup.option(
        "--tokenizer",
        type=str,
        default=None,
        help="Path | Name of the tokenizer. "
        "Specify this value only if using TensorRT engine as model.")(f)
    f = optgroup.option(
        "--backend",
        type=ChoiceWithAlias(["pytorch", "tensorrt", "_autodeploy"],
                             {"trt": "tensorrt"}),
        default="pytorch",
        help="The backend to use. Default is pytorch backend.")(f)
    f = optgroup.option(
        "--trust_remote_code",
        is_flag=True,
        default=False,
        help="Flag for HF transformers to trust remote code.")(f)
    f = optgroup.option(
        "--extra_llm_api_options",
        type=str,
        default=None,
        help=
        "Path to a YAML file that overwrites the LLM API configuration for serving the model."
    )(f)

    # Parallelism configuration
    f = optgroup.group("Parallelism Configuration",
                       help="Multi-GPU and distributed execution settings.",
                       cls=OptionGroup)(f)
    f = optgroup.option("--tp_size",
                        "--tp",
                        "tensor_parallel_size",
                        type=int,
                        default=1,
                        help="Tensor parallelism size.")(f)
    f = optgroup.option("--pp_size",
                        "--pp",
                        "pipeline_parallel_size",
                        type=int,
                        default=1,
                        help="Pipeline parallelism size.")(f)
    f = optgroup.option("--ep_size",
                        "--ep",
                        "moe_expert_parallel_size",
                        type=int,
                        default=None,
                        help="Expert parallelism size for MoE models.")(f)
    f = optgroup.option(
        "--cluster_size",
        "moe_cluster_parallel_size",
        type=int,
        default=None,
        help="Expert cluster parallelism size for MoE models.")(f)
    f = optgroup.option(
        "--gpus_per_node",
        type=int,
        default=None,
        help=
        "Number of GPUs per node. Default to None, and it will be detected automatically."
    )(f)

    # Build and runtime limits
    f = optgroup.group(
        "Build and Runtime Limits",
        help="Maximum batch size, sequence length, and token limits.",
        cls=OptionGroup)(f)
    f = optgroup.option(
        "--max_batch_size",
        type=int,
        default=BaseLlmArgs.model_fields["max_batch_size"].default,
        help="Maximum number of requests that the engine can schedule.")(f)
    f = optgroup.option(
        "--max_num_tokens",
        type=int,
        default=BaseLlmArgs.model_fields["max_num_tokens"].default,
        help=
        "Maximum number of batched input tokens after padding is removed in each batch."
    )(f)
    f = optgroup.option(
        "--max_seq_len",
        type=int,
        default=BaseLlmArgs.model_fields["max_seq_len"].default,
        help="Maximum total length of one request, including prompt and outputs. "
        "If unspecified, the value is deduced from the model config.")(f)
    f = optgroup.option(
        "--max_beam_width",
        type=int,
        default=BaseLlmArgs.model_fields["max_beam_width"].default,
        help="Maximum number of beams for beam search decoding.")(f)
    f = optgroup.option(
        "--max_input_len",
        type=int,
        default=BaseLlmArgs.model_fields["max_input_len"].default,
        help="Maximum input sequence length.")(f)

    # KV cache configuration
    f = optgroup.group("KV Cache Configuration",
                       help="KV cache memory and reuse settings.",
                       cls=OptionGroup)(f)
    f = optgroup.option(
        "--kv_cache_free_gpu_memory_fraction",
        "--kv_cache_free_gpu_mem_fraction",
        "kv_cache_free_gpu_memory_fraction",
        type=float,
        default=0.9,
        help=
        "Free GPU memory fraction reserved for KV Cache, after allocating model weights and buffers."
    )(f)
    f = optgroup.option("--disable_kv_cache_reuse",
                        is_flag=True,
                        default=False,
                        help="Flag for disabling KV cache reuse.")(f)

    # Postprocessing options
    f = optgroup.group("Postprocessing Options",
                       help="Output processing and formatting settings.",
                       cls=OptionGroup)(f)
    f = optgroup.option(
        "--num_postprocess_workers",
        type=int,
        default=0,
        help="[Experimental] Number of workers to postprocess raw responses.")(
            f)
    f = optgroup.option(
        "--reasoning_parser",
        type=click.Choice(ReasoningParserFactory.parsers.keys()),
        default=None,
        help="[Experimental] Specify the parser for reasoning models.")(f)

    # Advanced options
    f = optgroup.group("Advanced Options",
                       help="Advanced configuration and debugging settings.",
                       cls=OptionGroup)(f)
    f = optgroup.option(
        "--fail_fast_on_attention_window_too_large",
        is_flag=True,
        default=False,
        help=
        "Exit with runtime error when attention window is too large to fit even a single sequence in the KV cache."
    )(f)
    f = optgroup.option(
        "--skip_tokenizer_init",
        is_flag=True,
        default=False,
        help="Skip tokenizer initialization when loading the model.")(f)

    return f
