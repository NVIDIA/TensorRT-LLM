"""TensorRT-LLM configuration generator CLI.

This CLI tool generates optimized TensorRT-LLM recipe files from high-level
inference scenario constraints.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from tensorrt_llm.recipes import find_all_matching_recipes, validate_config, validate_scenario
from tensorrt_llm.recipes.matcher import merge_overrides


def format_env_vars(env: Dict[str, str]) -> str:
    """Format environment variables for shell command.

    Args:
        env: Dictionary of environment variables

    Returns:
        Formatted string like "VAR1=value1 VAR2=value2"
    """
    if not env:
        return ""
    return " ".join(f"{k}={v}" for k, v in env.items())


def generate_bench_command(recipe_path: str, model: str) -> str:
    """Generate the trtllm-bench command line.

    Args:
        recipe_path: Path to the recipe YAML file
        model: Model name from the scenario

    Returns:
        Formatted trtllm-bench command template
    """
    return (
        f"# For throughput benchmarking:\n"
        f"trtllm-bench --model {model} throughput --extra_llm_api_options {recipe_path}\n\n"
        f"# For latency benchmarking:\n"
        f"trtllm-bench --model {model} latency --extra_llm_api_options {recipe_path}\n\n"
        f"# For building only:\n"
        f"trtllm-bench --model {model} build --extra_llm_api_options {recipe_path}"
    )


def print_result(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    env: Dict[str, str],
    output_path: str,
    profile_name: str,
) -> None:
    """Print formatted result to stdout.

    Args:
        scenario: Scenario parameters
        config: Generated configuration
        env: Environment variables
        output_path: Path where recipe was written
        profile_name: Name of the profile used
    """
    click.echo(
        click.style(
            "\nGenerated optimized recipe for the specified scenario:", fg="green", bold=True
        )
    )
    click.echo(f"Profile: {profile_name}\n")

    # Print scenario
    click.echo(click.style("scenario:", fg="cyan", bold=True))
    scenario_yaml = yaml.dump(scenario, default_flow_style=False, sort_keys=False)
    for line in scenario_yaml.splitlines():
        click.echo(f"  {line}")
    click.echo()

    # Print environment variables if any
    if env:
        click.echo(click.style("env:", fg="cyan", bold=True))
        for key, value in env.items():
            click.echo(f"  {key}: {value}")
        click.echo()

    # Print configuration
    click.echo(click.style("llm_api_config:", fg="cyan", bold=True))
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    for line in config_yaml.splitlines():
        click.echo(f"  {line}")
    click.echo()

    # Print file write confirmation
    click.echo(click.style(f"Wrote recipe to {output_path}.", fg="green"))
    click.echo()

    # Print bench command
    click.echo(click.style("To run benchmarks with this recipe, use:", fg="yellow", bold=True))
    click.echo()

    bench_cmd = generate_bench_command(output_path, scenario.get("model", "<model>"))
    click.echo(bench_cmd)
    click.echo()


@click.command("configure")
@click.option(
    "--model",
    type=str,
    required=True,
    help="Model name or HuggingFace path (e.g., 'nvidia/DeepSeek-R1-0528-FP4')",
)
@click.option("--gpu", type=str, default=None, help="GPU type (e.g., 'H100_SXM', 'B200')")
@click.option("--num-gpus", type=int, default=None, help="Number of GPUs to use")
@click.option("--target-isl", type=int, required=True, help="Target input sequence length")
@click.option("--target-osl", type=int, required=True, help="Target output sequence length")
@click.option(
    "--target-concurrency",
    type=int,
    required=True,
    help="Target concurrency (number of concurrent requests)",
)
@click.option(
    "--tp-size",
    type=int,
    default=None,
    help="Tensor parallelism size (for matching existing recipes)",
)
@click.option(
    "--ep-size",
    type=int,
    default=None,
    help="Expert parallelism size (for matching existing recipes)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for the generated recipe YAML file",
)
@click.option(
    "--no-validate", is_flag=True, default=False, help="Skip validation of scenario constraints"
)
def configure(
    model: str,
    gpu: Optional[str],
    num_gpus: Optional[int],
    target_isl: int,
    target_osl: int,
    target_concurrency: int,
    tp_size: Optional[int],
    ep_size: Optional[int],
    output: str,
    no_validate: bool,
):
    r"""Retrieve an exact matching recipe from the database.

    This tool searches for an exact match in tensorrt_llm/recipes/db/ based on
    the provided scenario parameters and outputs the matching recipe to a file.

    The tool performs exact matching on: model, target_isl, target_osl, and
    target_concurrency. If no exact match is found, or if multiple matches are
    found, an error is returned.

    Examples:
    \b
    # Find and retrieve recipe for DeepSeek-R1 FP4 on B200
    trtllm-configure \\
        --model nvidia/DeepSeek-R1-0528-FP4 \\
        --target-isl 8192 \\
        --target-osl 1024 \\
        --target-concurrency 256 \\
        --output my-recipe.yaml

    \b
    # Find recipe for GPT-OSS on H100
    trtllm-configure \\
        --model openai/gpt-oss-120b \\
        --target-isl 8000 \\
        --target-osl 1000 \\
        --target-concurrency 256 \\
        --output recipe.yaml
    """
    try:
        # Build scenario from CLI arguments
        scenario = {
            "model": model,
            "target_isl": target_isl,
            "target_osl": target_osl,
            "target_concurrency": target_concurrency,
        }

        if gpu:
            scenario["gpu"] = gpu
        if num_gpus is not None:
            scenario["num_gpus"] = num_gpus
        if tp_size is not None:
            scenario["tp_size"] = tp_size
        if ep_size is not None:
            scenario["ep_size"] = ep_size

        # Find all matching recipes in the database
        matches = find_all_matching_recipes(scenario)

        if len(matches) == 0:
            # No exact match found
            error_msg = (
                f"No matching recipe found in database for scenario:\n"
                f"  model: {model}\n"
                f"  target_isl: {target_isl}\n"
                f"  target_osl: {target_osl}\n"
                f"  target_concurrency: {target_concurrency}\n\n"
                f"Please ensure an exact matching recipe exists in tensorrt_llm/recipes/db/"
            )
            raise ValueError(error_msg)

        elif len(matches) > 1:
            # Multiple matches found - ambiguous
            recipe_names = [match[0].name for match in matches]
            error_msg = (
                f"Multiple matching recipes found for scenario:\n"
                f"  model: {model}\n"
                f"  target_isl: {target_isl}\n"
                f"  target_osl: {target_osl}\n"
                f"  target_concurrency: {target_concurrency}\n\n"
                f"Matching recipes:\n"
                + "\n".join(f"  - {name}" for name in recipe_names)
                + "\n\nPlease refine your scenario to match exactly one recipe."
            )
            raise ValueError(error_msg)

        # Exactly one match - use it
        recipe_path, matched_recipe = matches[0]
        click.echo(click.style(f"Found matching recipe: {recipe_path.name}", fg="green"))

        config = matched_recipe.get("llm_api_config", {})
        env = matched_recipe.get("env", {})
        overrides = matched_recipe.get("overrides", {})
        if overrides:
            config = merge_overrides(config, overrides)

        # Use the matched recipe's scenario (preserves all fields)
        matched_scenario = matched_recipe.get("scenario", {})

        # Validate matched recipe unless disabled
        if not no_validate:
            warnings = validate_scenario(matched_scenario, strict=True)
            for warning in warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

            # Validate config from recipe
            config_warnings = validate_config(config)
            for warning in config_warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

            # TODO: Add llm_api_config validation once PR #8331 merges
            # (standardizes LlmArgs with Pydantic - validation will happen automatically)

        # Build complete recipe structure (use matched scenario to preserve all fields)
        recipe_data = {
            "scenario": matched_scenario,
            "env": env,
            "llm_api_config": config,
        }

        # Write recipe to file
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(recipe_data, f, default_flow_style=False, sort_keys=False)

        # Get profile name from matched recipe scenario (if present)
        profile_name = matched_scenario.get("profile", "N/A")

        # Print result
        print_result(matched_scenario, config, env, str(output_path), profile_name)

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


def main():
    """Main entry point for trtllm-configure CLI."""
    configure()


if __name__ == "__main__":
    main()
