"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

import json
import sys
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

        return deep_merge_dicts(data, OmegaConf.from_dotlist(dotlist))

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


def get_registry_yaml_extra(model_name: str) -> List[str]:
    """Look up a model in the registry and return its resolved yaml_extra config paths.

    Args:
        model_name: HuggingFace model id as listed in the registry (e.g. ``meta-llama/Llama-3.1-8B-Instruct``).

    Returns:
        List of absolute paths to the yaml config files for the model.

    Raises:
        KeyError: If the model is not found in the registry.
    """
    with open(_REGISTRY_YAML) as f:
        registry = yaml.safe_load(f)

    for entry in registry.get("models", []):
        if entry["name"] == model_name:
            return [str(_REGISTRY_CONFIGS_DIR / cfg) for cfg in entry.get("yaml_extra", [])]

    raise KeyError(
        f"Model '{model_name}' not found in the AutoDeploy model registry ({_REGISTRY_YAML}). "
        "Either add it to the registry or provide --yaml-extra directly."
    )


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


def _inject_registry_yaml_extra() -> None:
    """If ``--use-registry`` is in sys.argv, replace it with the resolved ``--yaml-extra`` entries.

    This allows callers to simply run::

        python build_and_run_ad.py --model <hf_model_id> --use-registry

    instead of manually specifying every ``--yaml-extra`` path.  The flag is consumed here and the
    resolved paths are injected back into ``sys.argv`` before pydantic-settings parses them.
    """
    if "--use-registry" not in sys.argv:
        return

    # Extract model name from argv (support both --model=X and --model X forms)
    model_name: Optional[str] = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--model="):
            model_name = arg.split("=", 1)[1]
            break
        if arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            break

    if model_name is None:
        raise ValueError("--use-registry requires --model to be specified.")

    yaml_extra_paths = get_registry_yaml_extra(model_name)

    # Remove --use-registry and inject --yaml-extra <path0> --yaml-extra <path1> ...
    # Each path needs its own flag because pydantic-settings CLI only captures one value per flag.
    argv = [a for a in sys.argv if a != "--use-registry"]
    for path in yaml_extra_paths:
        argv += ["--args.yaml-extra", path]
    sys.argv = argv


def main(config: Optional[ExperimentConfig] = None):
    if config is None:
        _inject_registry_yaml_extra()
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
