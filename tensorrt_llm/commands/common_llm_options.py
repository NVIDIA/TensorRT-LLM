import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type

import click
from click_option_group import OptionGroup, optgroup
from pydantic import BaseModel

from tensorrt_llm.llmapi.llm_args import BaseLlmArgs, KvCacheConfig
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory


def option_from_field(
    field_name: str,
    option_names: Optional[List[str]] = None,
    model_class: Type[BaseModel] = BaseLlmArgs,
) -> Callable:
    """Create a CLI option decorator from a field in a Pydantic model.

    Args:
        field_name: Name of the field in the pydantic model
        option_name: CLI option name (default: --{field_name})
        model_class: Pydantic model class containing the field

    Returns:
        A decorator function that adds the option
    """
    if option_names is None:
        option_names = [f"--{field_name}"]

    field_info = model_class.model_fields[field_name]

    field_type = field_info.annotation
    # Handle Optional types
    if typing.get_origin(field_type) is typing.Union:
        args = typing.get_args(field_type)
        # Get the non-None type
        field_type = next((arg for arg in args if arg is not type(None)),
                          args[0])

    elif field_type not in (int, float, str, bool):
        raise ValueError(f"Unsupported field type: {field_type}")

    return optgroup.option(*option_names,
                           type=field_type,
                           is_flag=field_type is bool,
                           default=field_info.default,
                           help=field_info.description)


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
    f = option_from_field("tokenizer")(f)
    f = optgroup.option(
        "--backend",
        type=ChoiceWithAlias(["pytorch", "tensorrt", "_autodeploy"],
                             {"trt": "tensorrt"}),
        default="pytorch",
        help=BaseLlmArgs.model_fields["backend"].description)(f)
    f = option_from_field("trust_remote_code")(f)
    f = optgroup.option(
        "--extra_llm_api_options",
        type=str,
        default=None,
        help=
        "Path to a YAML file with LLM API configuration options for serving the model."
    )(f)

    # Parallelism configuration
    f = optgroup.group("Parallelism Configuration",
                       help="Multi-GPU and distributed execution settings.",
                       cls=OptionGroup)(f)
    f = option_from_field("tensor_parallel_size",
                          option_names=["--tp_size", "--tp"])(f)
    f = option_from_field("pipeline_parallel_size",
                          option_names=["--pp_size", "--pp"])(f)
    f = option_from_field("moe_expert_parallel_size",
                          option_names=["--ep_size", "--ep"])(f)
    f = option_from_field("moe_cluster_parallel_size",
                          option_names=["--cluster_size"])(f)

    # Build and runtime limits
    f = optgroup.group(
        "Build and Runtime Limits",
        help="Maximum batch size, sequence length, and token limits.",
        cls=OptionGroup)(f)
    f = option_from_field("max_batch_size")(f)
    f = option_from_field("max_num_tokens")(f)
    f = option_from_field("max_seq_len")(f)
    f = option_from_field("max_beam_width")(f)
    f = option_from_field("max_input_len")(f)

    # KV cache configuration
    f = optgroup.group("KV Cache Configuration",
                       help="KV cache memory and reuse settings.",
                       cls=OptionGroup)(f)
    f = option_from_field("free_gpu_memory_fraction",
                          option_names=[
                              "--kv_cache_free_gpu_memory_fraction",
                              "--kv_cache_free_gpu_mem_fraction"
                          ],
                          model_class=KvCacheConfig)(f)
    f = optgroup.option("--disable_kv_cache_reuse",
                        is_flag=True,
                        default=False,
                        help="Flag for disabling KV cache reuse.")(f)

    # Postprocessing options
    f = optgroup.group("Postprocessing Options",
                       help="Output processing and formatting settings.",
                       cls=OptionGroup)(f)
    f = option_from_field("num_postprocess_workers")(f)
    f = optgroup.option(
        "--reasoning_parser",
        type=click.Choice(ReasoningParserFactory.parsers.keys()),
        default=None,
        help=BaseLlmArgs.model_fields["reasoning_parser"].description)(f)

    # Advanced options
    f = optgroup.group("Advanced Options",
                       help="Advanced configuration and debugging settings.",
                       cls=OptionGroup)(f)
    f = option_from_field("fail_fast_on_attention_window_too_large")(f)
    f = option_from_field("skip_tokenizer_init")(f)

    return f
