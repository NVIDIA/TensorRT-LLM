"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliImplicitFlag,
    CliUnknownArgs,
    SettingsConfigDict,
)

from tensorrt_llm._torch.auto_deploy import LLM, DemoLLM
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.utils._config import (
    DynamicYamlMixInForSettings,
    deep_merge_dicts,
)
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

# Registry paths
_REGISTRY_DIR = Path(__file__).resolve().parent / "model_registry"
_REGISTRY_YAML = _REGISTRY_DIR / "models.yaml"
_REGISTRY_CONFIGS_DIR = _REGISTRY_DIR / "configs"

# Global torch config, set the torch compile cache to fix up to llama 405B
torch._dynamo.config.cache_size_limit = 20


# A single query is either a plain string or a full HF chat message template.
PromptInput = Union[str, List[Dict]]


class PromptConfig(BaseModel):
    """Prompt configuration.

    This configuration class can be used for this example script to configure the example prompts
    and the sampling parameters.

    Queries can be plain strings or HF-style chat message lists
    (``[{"role": "user", "content": "..."}]``). Plain-string queries are automatically wrapped in
    a chat template when the model's tokenizer supports one.
    """

    batch_size: int = Field(default=10, description="Number of queries")
    queries: Union[PromptInput, List[PromptInput]] = Field(
        default_factory=lambda: [
            "How big is the universe? ",
            "In simple words and a single sentence, explain the concept of gravity: ",
            "How to fix slicing in golf? ",
            "Where is the capital of Iceland? ",
            "What are the three laws of thermodynamics? ",
            "Summarize the plot of Romeo and Juliet in two sentences: ",
            "Write a Python function that checks if a number is prime.",
            "Explain the difference between a compiler and an interpreter: ",
            "What causes the northern lights? ",
            "What are the health benefits of drinking green tea?",
        ],
        description="Plain-text queries or HF-style chat message lists. Plain strings are "
        "automatically wrapped as chat messages when the model's tokenizer has a chat template.",
    )
    sp_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"max_tokens": 100, "top_k": None, "temperature": 1.0},
        description="Sampling parameter kwargs passed on the SamplingParams class. "
        "Defaults are set to the values used in the original model.",
    )

    def model_post_init(self, __context: Any):
        """Repeat and truncate queries to match batch_size.

        NOTE (lucaslie): has to be done with model_post_init to ensure it's always run. field
        validators are only run if a value is provided.
        """
        queries = self.queries
        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, list) and queries and isinstance(queries[0], dict):
            # single HF message template, e.g. [{"role": "user", "content": "..."}]
            queries = [queries]
        queries = queries * (self.batch_size // len(queries) + 1)
        self.queries = queries[: self.batch_size]

    @field_validator("sp_kwargs", mode="after")
    @classmethod
    def validate_sp_kwargs(cls, sp_kwargs):
        """Insert desired defaults for sampling params and try parsing string values as JSON."""
        default = cls.model_fields["sp_kwargs"].get_default(call_default_factory=True)
        return deep_merge_dicts(default, sp_kwargs)


class BenchmarkConfig(BaseModel):
    """Configuration for storing results."""

    results_path: Optional[str] = Field(default="./benchmark_results.json")
    store_results: bool = Field(default=False, description="If True, store results to results_path")


class ExperimentConfig(DynamicYamlMixInForSettings, BaseSettings):
    """Experiment Configuration for the example script.

    This configuration aggregates all relevant configurations for this example script. It is also
    used to auto-generate the CLI interface.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        cli_kebab_case=True,
        cli_ignore_unknown_args=True,
        nested_model_default_partial_update=True,
    )
    extra_cli_args: CliUnknownArgs

    ### CORE ARGS ##################################################################################
    # The main AutoDeploy arguments - contains model, tokenizer, backend configs, etc.
    args: LlmArgs = Field(
        description="The main AutoDeploy arguments containing model, tokenizer, backend configs, etc. "
        "Please check `tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs` for more details."
    )

    # Optional model field for convenience - if provided, will be used to initialize args.model
    model: Optional[str] = Field(
        default=None,
        description="The path to the model checkpoint or the model name from the Hugging Face Hub. "
        "If provided, will be passed through to initialize args.model",
    )

    ### SIMPLE PROMPTING CONFIG ####################################################################
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    ### BENCHMARKING CONFIG ########################################################################
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    ### MODEL REGISTRY CONFIG ####################################################################
    use_registry: CliImplicitFlag[bool] = Field(
        default=False,
        description="Resolve args.yaml_extra from examples/auto_deploy/model_registry/models.yaml for --model.",
    )
    registry_config_id: Optional[str] = Field(
        default=None,
        description="Optional config_id selector used with --use-registry when a model has multiple registry entries.",
    )

    ### CONFIG DEBUG FLAG ##########################################################################
    dry_run: CliImplicitFlag[bool] = Field(default=False, description="Show final config and exit")

    ### VALIDATION #################################################################################
    @model_validator(mode="before")
    @classmethod
    def setup_args_from_model(cls, data: Dict) -> Dict:
        """Check for model being provided directly or via args.model."""
        msg = '"model" must be provided directly or via "args.model"'
        if not isinstance(data, dict):
            raise ValueError(msg)
        if not ("model" in data or "model" in data.get("args", {})):
            raise ValueError(msg)

        data["args"] = data.get("args", {})
        if "model" in data:
            data["args"]["model"] = data["model"]
        return data

    @model_validator(mode="before")
    @classmethod
    def process_extra_cli_args(cls, data: Dict) -> Dict:
        """Process extra CLI args.

        This model validator enables the user to provide additional CLI args that may not be
        auto-generated by the CLI app. A common use case for this would to modify graph transforms
        dynamically via CLI arguments.

        For example, the user can provide a CLI argument for raw dictionaries like this, e.g., for
        ``model_kwargs``: ``--args.model-kwargs.num-hidden-layers=10``.
        """
        # build a clean dotlist: ["a.b=1","c.d.e=foo",…]
        raw: List[str] = data.pop("extra_cli_args", [])
        dotlist = []
        it: Iterator[str] = iter(raw)
        for tok in it:
            if not tok.startswith("--"):
                continue
            body = tok[2:]
            if "=" in body:
                body, val = body.split("=", 1)
            else:
                # flag + separate value
                val = next(it, None)
            # ensure kebab-case is converted to snake_case
            dotlist.append(f"{body.replace('-', '_')}={val}")

        # Merge unknown dotted CLI overrides (captured in extra_cli_args)
        # into the config before nested pydantic models validate.
        data = deep_merge_dicts(data, OmegaConf.from_dotlist(dotlist))

        # Resolve registry configs before nested LlmArgs validation runs.
        if data.get("use_registry", False):
            model_name = data.get("model") or data.get("args", {}).get("model")
            if not model_name:
                raise ValueError("--use-registry requires --model or --args.model to be specified.")

            config_id = data.get("registry_config_id")
            registry_yaml_extra = get_registry_yaml_extra(model_name, config_id)

            data.setdefault("args", {})
            existing_yaml_extra = list(data["args"].get("yaml_extra", []) or [])
            # Registry defaults go first so explicit user --args.yaml-extra can override.
            data["args"]["yaml_extra"] = [*registry_yaml_extra, *existing_yaml_extra]

            merged_cfg = _merge_yaml_files(data["args"]["yaml_extra"])
            merged_max_batch_size = merged_cfg.get("max_batch_size")
            merged_cg_cfg = merged_cfg.get("cuda_graph_config")
            merged_cg_max = (
                merged_cg_cfg.get("max_batch_size") if isinstance(merged_cg_cfg, dict) else None
            )

            explicit_cg_cfg = data["args"].get("cuda_graph_config")
            explicit_cg_max = (
                explicit_cg_cfg.get("max_batch_size") if isinstance(explicit_cg_cfg, dict) else None
            )

            # If registry sets a smaller top-level max_batch_size without an explicit
            # cuda_graph max, synchronize to avoid LlmArgs validation failures.
            if (
                explicit_cg_max is None
                and isinstance(merged_max_batch_size, int)
                and (merged_cg_max is None or merged_cg_max > merged_max_batch_size)
            ):
                data["args"].setdefault("cuda_graph_config", {})
                data["args"]["cuda_graph_config"]["max_batch_size"] = merged_max_batch_size

        return data

    @field_validator("model", mode="after")
    @classmethod
    def sync_model_with_args(cls, model_value, info):
        if "args" not in info.data:
            return model_value
        args: LlmArgs = info.data["args"]
        return args.model

    @field_validator("prompt", mode="after")
    @classmethod
    def sync_prompt_batch_size_with_args_max_batch_size(cls, prompt: PromptConfig, info):
        if "args" not in info.data:
            return prompt
        args: LlmArgs = info.data["args"]
        if args.max_batch_size < prompt.batch_size:
            args.max_batch_size = prompt.batch_size
        return prompt


def get_registry_yaml_extra(model_name: str, config_id: Optional[str] = None) -> List[str]:
    """Look up a model in the registry and return its resolved yaml_extra config paths.

    Args:
        model_name: HuggingFace model id as listed in the registry (e.g. ``meta-llama/Llama-3.1-8B-Instruct``).

    Returns:
        List of absolute paths to the yaml config files for the model.

    Raises:
        KeyError: If the model/config is not found or is ambiguous without ``config_id``.
    """
    with open(_REGISTRY_YAML) as f:
        registry = yaml.safe_load(f)

    matches = [entry for entry in registry.get("models", []) if entry.get("name") == model_name]

    if config_id is not None:
        matches = [entry for entry in matches if entry.get("config_id", "default") == config_id]
        if not matches:
            raise KeyError(
                f"Model '{model_name}' with config_id '{config_id}' not found in AutoDeploy model registry "
                f"({_REGISTRY_YAML})."
            )
    elif len(matches) > 1:
        default_matches = [
            entry for entry in matches if entry.get("config_id", "default") == "default"
        ]
        if len(default_matches) == 1:
            matches = default_matches
        else:
            available = sorted({entry.get("config_id", "default") for entry in matches})
            raise KeyError(
                f"Model '{model_name}' has multiple registry entries with config_id values {available}. "
                "Provide --registry-config-id to select one."
            )

    if not matches:
        raise KeyError(
            f"Model '{model_name}' not found in the AutoDeploy model registry ({_REGISTRY_YAML}). "
            "Either add it to the registry or provide --yaml-extra directly."
        )

    selected = matches[0]
    return [str(_REGISTRY_CONFIGS_DIR / cfg) for cfg in selected.get("yaml_extra", [])]


def _merge_yaml_files(yaml_paths: List[str]) -> Dict[str, Any]:
    """Load and deep-merge YAML files in order."""
    merged: Dict[str, Any] = {}
    for yaml_path in yaml_paths:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        merged = deep_merge_dicts(merged, data)
    return merged


def build_llm_from_config(config: ExperimentConfig) -> LLM:
    """Builds a LLM object from our config."""
    # construct llm high-level interface object
    llm_lookup = {
        "demollm": DemoLLM,
        "trtllm": LLM,
    }
    llm = llm_lookup[config.args.runtime](**config.args.model_dump(exclude_unset=True))
    return llm


def prepare_queries(queries: List[PromptInput], tokenizer=None) -> List[Dict]:
    """Prepare queries for the LLM API.

    Queries that are already HF-style message lists (``List[Dict]``) are passed through directly.
    Plain-string queries are wrapped as HF chat messages when the tokenizer has a chat template,
    or passed as plain text prompts otherwise.
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    prepared = []
    for query in queries:
        if isinstance(query, list):
            prepared.append(
                {
                    "prompt": query[0].get("content", "") if query else "",
                    "messages": query,
                }
            )
        elif has_chat_template:
            prepared.append(
                {
                    "prompt": query,
                    "messages": [{"role": "user", "content": query}],
                }
            )
        else:
            prepared.append({"prompt": query})
    return prepared


def print_outputs(outs: Union[RequestOutput, List[RequestOutput]]) -> List[List[str]]:
    prompts_and_outputs: List[List[str]] = []
    if isinstance(outs, RequestOutput):
        outs = [outs]
    for i, out in enumerate(outs):
        prompt, output = out.prompt, out.outputs[0].text
        ad_logger.info(f"[PROMPT {i}] {prompt}: {output}")
        prompts_and_outputs.append([prompt, output])
    return prompts_and_outputs


def main(config: Optional[ExperimentConfig] = None):
    if config is None:
        config: ExperimentConfig = CliApp.run(ExperimentConfig)

    ad_logger.info(f"AutoDeploy Experiment Config:\n{yaml.dump(config.model_dump())}")

    if config.dry_run:
        return

    llm = build_llm_from_config(config)

    # prompt the model and print its output
    ad_logger.info("Running example prompts...")
    hf_tokenizer = getattr(llm.tokenizer, "tokenizer", None)
    queries = prepare_queries(config.prompt.queries, hf_tokenizer)
    outs = llm.generate(
        queries,
        sampling_params=SamplingParams(**config.prompt.sp_kwargs),
    )
    results = {
        "prompts_and_outputs": print_outputs(outs),
    }
    # Add config values so they get logged to JET extra
    results.update(config.model_dump(mode="json"))

    if config.benchmark.store_results:
        results_path = Path(config.benchmark.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

    llm.shutdown()
    return results


if __name__ == "__main__":
    main()
