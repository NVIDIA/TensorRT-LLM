"""TensorRT-LLM configuration generator CLI.

This CLI tool generates optimized TensorRT-LLM recipe files from high-level
inference scenario constraints.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from tensorrt_llm.recipes import (
    compute_from_scenario,
    detect_profile,
    match_recipe,
    validate_config,
    validate_scenario,
)
from tensorrt_llm.recipes.matcher import merge_overrides
from tensorrt_llm.recipes.profiles import PROFILE_REGISTRY


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


def generate_bench_command(recipe_path: str) -> str:
    """Generate the trtllm-bench command line.

    Args:
        recipe_path: Path to the recipe YAML file

    Returns:
        Formatted trtllm-bench command
    """
    return f"trtllm-bench --recipe {recipe_path}"


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
    click.echo(click.style("config:", fg="cyan", bold=True))
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

    bench_cmd = generate_bench_command(output_path)
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
    help="Tensor parallelism size (overrides auto-computed value)",
)
@click.option(
    "--ep-size",
    type=int,
    default=None,
    help="Expert parallelism size (overrides auto-computed value)",
)
@click.option(
    "--profile",
    type=click.Choice(list(PROFILE_REGISTRY.keys())),
    default=None,
    help="Profile to use (auto-detected from model name if not specified)",
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
    profile: Optional[str],
    output: str,
    no_validate: bool,
):
    r"""Generate optimized TensorRT-LLM recipe from scenario constraints.

    This tool takes high-level inference scenario parameters and generates a
    complete recipe YAML file (scenario + config + env) that can be used with
    trtllm-bench's --recipe flag.

    Examples:
    \b
    # Generate recipe from scenario parameters
    trtllm-configure \\
        --model nvidia/DeepSeek-R1-0528-FP4 \\
        --gpu B200 \\
        --num-gpus 8 \\
        --target-isl 8192 \\
        --target-osl 1024 \\
        --target-concurrency 256 \\
        --output my-recipe.yaml

    \b
    # Override TP/EP sizes
    trtllm-configure \\
        --model openai/gpt-oss-120b \\
        --target-isl 8000 \\
        --target-osl 1000 \\
        --target-concurrency 256 \\
        --tp-size 4 \\
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

        # Try to match against existing recipes first
        matched_recipe = match_recipe(scenario)
        if matched_recipe:
            click.echo(click.style("Found matching recipe in database!", fg="green"))
            config = matched_recipe.get("config", {})
            env = matched_recipe.get("env", {})
            overrides = matched_recipe.get("overrides", {})
            if overrides:
                config = merge_overrides(config, overrides)
        else:
            # Compute from scenario using profile
            result = compute_from_scenario(scenario, profile)
            config = result["config"]
            env = result.get("env", {})

        # Validate scenario unless disabled
        if not no_validate:
            warnings = validate_scenario(scenario, strict=True)
            for warning in warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

            # Validate generated config
            config_warnings = validate_config(config)
            for warning in config_warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

        # Build complete recipe structure
        recipe_data = {
            "scenario": scenario,
            "env": env,
            "config": config,
        }

        # Write recipe to file
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(recipe_data, f, default_flow_style=False, sort_keys=False)

        # Determine which profile was used
        profile_name = profile or scenario.get("profile") or detect_profile(model)
        if not profile_name:
            profile_name = "custom"

        # Print result
        print_result(scenario, config, env, str(output_path), profile_name)

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
