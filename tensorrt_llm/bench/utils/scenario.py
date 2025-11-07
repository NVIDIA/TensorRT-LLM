"""Utilities for extracting and processing recipe scenario parameters.

This module provides functions to extract scenario information from recipe YAML
files and merge them with CLI parameters for trtllm-bench commands.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import yaml

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.bench.benchmark import GeneralExecSettings
    from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment


def extract_scenario_from_recipe(recipe_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract scenario section from a recipe YAML file.

    Args:
        recipe_path: Path to recipe YAML file, or None

    Returns:
        Dictionary containing scenario parameters, or None if not a recipe format
        or if recipe_path is None

    Example:
        >>> scenario = extract_scenario_from_recipe("recipe.yaml")
        >>> print(scenario["target_isl"])
        8192
    """
    if recipe_path is None:
        return None

    try:
        with open(recipe_path, "r") as f:
            loaded_data = yaml.safe_load(f)

        # Check if this is a recipe format (has 'scenario' and 'llm_api_options' keys)
        if (
            isinstance(loaded_data, dict)
            and "scenario" in loaded_data
            and "llm_api_options" in loaded_data
        ):
            return loaded_data["scenario"]

        return None
    except (FileNotFoundError, yaml.YAMLError, KeyError):
        return None


def merge_params_with_priority(
    cli_params: Dict[str, Any],
    scenario: Optional[Dict[str, Any]],
    cli_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge CLI parameters with scenario values, with CLI taking precedence.

    Priority order (highest to lowest):
    1. Explicitly set CLI parameters (different from default)
    2. Scenario values from recipe
    3. CLI default values

    Args:
        cli_params: Parameters from CLI arguments
        scenario: Scenario dict from recipe (or None)
        cli_defaults: Default values for CLI args (used to detect explicit values)

    Returns:
        Merged parameter dictionary

    Example:
        >>> cli = {"concurrency": 128, "tp": 1}
        >>> scenario = {"target_concurrency": 256, "tp_size": 4}
        >>> defaults = {"concurrency": -1, "tp": 1}
        >>> merged = merge_params_with_priority(cli, scenario, defaults)
        >>> print(merged["concurrency"])  # CLI explicitly set
        128
        >>> print(merged["tp"])  # From scenario (tp_size -> tp)
        4
    """
    if scenario is None:
        return cli_params.copy()

    merged = cli_params.copy()

    # Mapping from scenario keys to CLI parameter keys
    # Note: 'model' is excluded because it's a required top-level trtllm-bench parameter
    param_mapping = {
        "target_concurrency": "concurrency",
        "target_isl": "target_input_len",
        "target_osl": "target_output_len",
        "num_requests": "num_requests",
        "tp_size": "tp",
        "ep_size": "ep",
        "pp_size": "pp",
        "streaming": "streaming",
    }

    for scenario_key, cli_key in param_mapping.items():
        if scenario_key in scenario:
            scenario_value = scenario[scenario_key]

            # Check if CLI value was explicitly set (differs from default)
            cli_value = cli_params.get(cli_key)
            default_value = cli_defaults.get(cli_key) if cli_defaults else None

            # Use scenario value if:
            # 1. CLI value is None/not set, OR
            # 2. CLI value equals the default (not explicitly set by user)
            if cli_value is None or (default_value is not None and cli_value == default_value):
                merged[cli_key] = scenario_value

    return merged


def validate_scenario_params(scenario: Dict[str, Any]) -> None:
    """Validate scenario parameters.

    Args:
        scenario: Scenario dictionary to validate

    Raises:
        ValueError: If scenario parameters are invalid
    """
    required_fields = ["target_isl", "target_osl", "target_concurrency"]

    # Check required fields
    for field in required_fields:
        if field not in scenario:
            raise ValueError(f"Scenario missing required field: {field}")

    # Validate numeric fields
    if scenario["target_isl"] <= 0:
        raise ValueError(f"target_isl must be positive, got: {scenario['target_isl']}")

    if scenario["target_osl"] <= 0:
        raise ValueError(f"target_osl must be positive, got: {scenario['target_osl']}")

    if scenario["target_concurrency"] <= 0:
        raise ValueError(
            f"target_concurrency must be positive, got: {scenario['target_concurrency']}"
        )

    # Validate optional stdev fields
    if "isl_stdev" in scenario:
        if scenario["isl_stdev"] < 0:
            raise ValueError(f"isl_stdev must be non-negative, got: {scenario['isl_stdev']}")

    if "osl_stdev" in scenario:
        if scenario["osl_stdev"] < 0:
            raise ValueError(f"osl_stdev must be non-negative, got: {scenario['osl_stdev']}")

    # Validate num_requests
    if "num_requests" in scenario:
        if scenario["num_requests"] <= 0:
            raise ValueError(f"num_requests must be positive, got: {scenario['num_requests']}")


def prepare_llm_api_options_for_recipe(
    extra_llm_api_options_path: Optional[str], scenario: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Prepare llm_api_options for LLM constructor when using recipe format.

    When a recipe format is detected (scenario is not None), this function extracts
    only the llm_api_options section and writes it to a temporary file. This prevents
    the scenario section from being passed to the LLM constructor, which would cause
    an "invalid argument" error.

    Args:
        extra_llm_api_options_path: Path to recipe/config YAML file
        scenario: Scenario dict from recipe (None if not recipe format)

    Returns:
        Path to temporary file with llm_api_options (if recipe format), or
        original path (if not recipe format), or None (if no path provided)

    Example:
        >>> scenario = extract_scenario_from_recipe("recipe.yaml")
        >>> config_path = prepare_llm_api_options_for_recipe("recipe.yaml", scenario)
        # config_path now points to temp file with only llm_api_options section
    """
    if extra_llm_api_options_path is None:
        return None

    # If not a recipe format, return original path
    if scenario is None:
        return extra_llm_api_options_path

    # Recipe format detected - extract llm_api_options only
    logger.info("Recipe format detected - extracting llm_api_options for LLM constructor")

    try:
        with open(extra_llm_api_options_path, "r") as f:
            full_recipe = yaml.safe_load(f)

        # Extract only the llm_api_options section
        llm_api_options_only = full_recipe.get("llm_api_options", {})

        # Create temporary file with only llm_api_options
        temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", text=True)
        with os.fdopen(temp_fd, "w") as f:
            yaml.safe_dump(llm_api_options_only, f)

        logger.info(f"Created temporary config file with llm_api_options at: {temp_path}")
        return temp_path

    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.warning(f"Failed to process recipe file for llm_api_options: {e}")
        return extra_llm_api_options_path


def auto_generate_dataset(
    scenario: Dict[str, Any],
    workspace: Path,
    tokenizer: str,
    output_filename: str = "auto_generated_dataset.json",
) -> Path:
    """Generate a synthetic dataset from scenario parameters.

    Args:
        scenario: Scenario dictionary with ISL/OSL/concurrency parameters
        workspace: Workspace directory to write dataset
        tokenizer: Tokenizer name or path for dataset generation
        output_filename: Name of output dataset file

    Returns:
        Path to generated dataset file

    Raises:
        ValueError: If required scenario parameters are missing
    """
    validate_scenario_params(scenario)

    dataset_path = workspace / output_filename

    # Extract parameters
    target_isl = scenario["target_isl"]
    target_osl = scenario["target_osl"]
    num_requests = scenario.get("num_requests", 512)
    isl_stdev = scenario.get("isl_stdev", 0)
    osl_stdev = scenario.get("osl_stdev", 0)

    # Generate synthetic dataset using prepare_dataset.py logic
    # For now, create a simple JSON format that benchmarks can consume
    #
    # TODO: This is a simplified implementation. In production, should either:
    # 1. Call prepare_dataset.py as a subprocess
    # 2. Import and use prepare_dataset.py's generation logic
    # 3. Use the dataset generation utilities from benchmarks/cpp/

    import numpy as np

    requests = []
    for i in range(num_requests):
        # Generate input/output lengths with normal distribution
        if isl_stdev > 0:
            input_len = int(max(1, np.random.normal(target_isl, isl_stdev)))
        else:
            input_len = target_isl

        if osl_stdev > 0:
            output_len = int(max(1, np.random.normal(target_osl, osl_stdev)))
        else:
            output_len = target_osl

        # Create request in format expected by benchmarks
        request = {
            "task_id": i,
            "prompt": " ".join(["word"] * input_len),  # Placeholder tokens
            "output_tokens": output_len,
            "input_len": input_len,
        }
        requests.append(request)

    # Write to JSON Lines file (one JSON object per line)
    # This is the format expected by trtllm-bench
    workspace.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")

    return dataset_path


def process_recipe_scenario(
    params: Dict[str, Any],
    options: "GeneralExecSettings",
    bench_env: "BenchmarkEnvironment",
    cli_defaults: Dict[str, Any],
) -> Tuple[Dict[str, Any], "GeneralExecSettings", Optional[Dict[str, Any]]]:
    """Process recipe scenario: extract, merge params, and auto-generate dataset.

    This is a unified helper for throughput and low_latency benchmarks to handle
    recipe-based configuration. It:
    1. Extracts scenario from recipe file (if present)
    2. Merges CLI params with scenario (CLI takes precedence)
    3. Auto-generates dataset if needed based on scenario ISL/OSL

    Args:
        params: CLI parameters dictionary (will be modified in-place)
        options: General execution settings from get_general_cli_options
        bench_env: Benchmark environment object
        cli_defaults: Default values for CLI args (used to detect explicit values)
                     Should vary by benchmark type (e.g., concurrency differs)

    Returns:
        Tuple of (updated_params, updated_options, scenario)
        - updated_params: params dict with merged scenario values
        - updated_options: regenerated options if dataset was auto-generated
        - scenario: extracted scenario dict (or None if not recipe format)
    """
    # Import here to avoid circular dependency
    from tensorrt_llm.bench.benchmark import get_general_cli_options

    # Extract scenario from recipe
    # Priority: --recipe > --extra_llm_api_options
    recipe_path = params.get("recipe")
    extra_llm_api_options_path = params.get("extra_llm_api_options")
    config_path = recipe_path if recipe_path else extra_llm_api_options_path
    scenario = extract_scenario_from_recipe(config_path)

    if not scenario:
        return params, options, None

    logger.info("Detected recipe format with scenario parameters")

    # Merge CLI params with scenario (CLI explicitly set takes precedence)
    merged_params = merge_params_with_priority(params, scenario, cli_defaults)

    # Update params with merged values
    params.update(merged_params)

    # Auto-generate dataset if not provided
    if params.get("dataset") is None and scenario.get("target_isl") and scenario.get("target_osl"):
        logger.info("No dataset provided, auto-generating from scenario parameters")
        workspace = Path.cwd() / ".trtllm_bench_workspace"
        auto_dataset_path = auto_generate_dataset(
            scenario, workspace, tokenizer=str(options.checkpoint_path)
        )
        params["dataset"] = auto_dataset_path
        logger.info(f"Generated dataset at {auto_dataset_path}")

        # Update options with auto-generated dataset
        options = get_general_cli_options(params, bench_env)

    return params, options, scenario
