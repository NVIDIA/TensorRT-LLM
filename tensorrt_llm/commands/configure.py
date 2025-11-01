"""TensorRT-LLM configuration generator CLI.

This CLI tool generates optimized TensorRT-LLM configurations from high-level
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
from tensorrt_llm.recipes.matcher import load_recipe_file, merge_overrides
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


def generate_serve_command(
    scenario: Dict[str, Any], cli_args: Dict[str, Any], env: Dict[str, str], config_path: str
) -> str:
    """Generate the trtllm-serve command line.

    Args:
        scenario: Scenario parameters
        cli_args: CLI arguments computed from profile
        env: Environment variables
        config_path: Path to the config YAML file

    Returns:
        Formatted trtllm-serve command
    """
    model = scenario.get("model", "MODEL_PATH")
    tp_size = cli_args.get("tp_size", 1)
    ep_size = cli_args.get("ep_size", 1)
    max_num_tokens = cli_args.get("max_num_tokens")
    max_batch_size = cli_args.get("max_batch_size")

    # Build command parts
    parts = []

    # Environment variables
    env_str = format_env_vars(env)
    if env_str:
        parts.append(env_str)

    # Base command
    parts.append("trtllm-serve")
    parts.append(model)

    # CLI arguments
    parts.append(f"--tp_size {tp_size}")
    if ep_size > 1:
        parts.append(f"--ep_size {ep_size}")

    if max_num_tokens is not None:
        parts.append(f"--max_num_tokens {max_num_tokens}")

    if max_batch_size is not None:
        parts.append(f"--max_batch_size {max_batch_size}")

    parts.append(f"--extra_llm_api_options {config_path}")

    return " \\\n    ".join(parts)


def print_result(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    env: Dict[str, str],
    cli_args: Dict[str, Any],
    output_path: str,
    profile_name: str,
) -> None:
    """Print formatted result to stdout.

    Args:
        scenario: Scenario parameters
        config: Generated configuration
        env: Environment variables
        cli_args: CLI arguments
        output_path: Path where config was written
        profile_name: Name of the profile used
    """
    click.echo(
        click.style(
            "\nFound optimized configuration for the specified scenario:", fg="green", bold=True
        )
    )
    click.echo(f"Profile: {profile_name}\n")

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
    click.echo(click.style(f"Wrote config to {output_path}.", fg="green"))
    click.echo()

    # Print serve command
    click.echo(
        click.style(
            "To serve the model with optimized settings, run the following command:",
            fg="yellow",
            bold=True,
        )
    )
    click.echo()

    serve_cmd = generate_serve_command(scenario, cli_args, env, output_path)
    click.echo(serve_cmd)
    click.echo()


@click.command("configure")
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name or HuggingFace path (e.g., 'nvidia/DeepSeek-R1-0528-FP4')",
)
@click.option("--gpu", type=str, default=None, help="GPU type (e.g., 'H100_SXM', 'B200')")
@click.option("--num-gpus", type=int, default=None, help="Number of GPUs to use")
@click.option("--target-isl", type=int, default=None, help="Target input sequence length")
@click.option("--target-osl", type=int, default=None, help="Target output sequence length")
@click.option(
    "--target-concurrency",
    type=int,
    default=None,
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
    "--recipe",
    type=click.Path(exists=True),
    default=None,
    help="Path to a recipe YAML file to load",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output path for the generated config YAML file",
)
@click.option(
    "--no-validate", is_flag=True, default=False, help="Skip validation of scenario constraints"
)
def configure(
    model: Optional[str],
    gpu: Optional[str],
    num_gpus: Optional[int],
    target_isl: Optional[int],
    target_osl: Optional[int],
    target_concurrency: Optional[int],
    tp_size: Optional[int],
    ep_size: Optional[int],
    profile: Optional[str],
    recipe: Optional[str],
    output: str,
    no_validate: bool,
):
    r"""Generate optimized TensorRT-LLM configuration from scenario constraints.

    This tool takes high-level inference scenario parameters and generates an
    optimized configuration file that can be used with trtllm-serve's
    --extra_llm_api_options flag.

    Examples:
    \b
    # Generate config from scenario parameters
    trtllm-configure \\
        --model nvidia/DeepSeek-R1-0528-FP4 \\
        --gpu B200 \\
        --num-gpus 8 \\
        --target-isl 8192 \\
        --target-osl 1024 \\
        --target-concurrency 256 \\
        --output config.yaml

    \b
    # Load from an existing recipe file
    trtllm-configure \\
        --recipe examples/gptoss-fp4-h100.yaml \\
        --output config.yaml
    """
    try:
        # Load from recipe file if provided
        if recipe:
            recipe_data = load_recipe_file(recipe)
            scenario = recipe_data.get("scenario", {})
            env_from_recipe = recipe_data.get("env", {})
            config_from_recipe = recipe_data.get("config", {})
            overrides = recipe_data.get("overrides", {})

            # Use recipe data as base, but allow CLI overrides
            if model:
                scenario["model"] = model
            if gpu:
                scenario["gpu"] = gpu
            if num_gpus is not None:
                scenario["num_gpus"] = num_gpus
            if target_isl is not None:
                scenario["target_isl"] = target_isl
            if target_osl is not None:
                scenario["target_osl"] = target_osl
            if target_concurrency is not None:
                scenario["target_concurrency"] = target_concurrency
            if tp_size is not None:
                scenario["tp_size"] = tp_size
            if ep_size is not None:
                scenario["ep_size"] = ep_size

            # If recipe already has config, use it
            if config_from_recipe:
                config = config_from_recipe
                env = env_from_recipe
                # Compute CLI args from scenario for the serve command
                profile_name = (
                    profile or scenario.get("profile") or detect_profile(scenario.get("model", ""))
                )
                if profile_name:
                    result = compute_from_scenario(scenario, profile_name)
                    cli_args = result.get("cli_args", {})
                else:
                    cli_args = {}
            else:
                # Recipe only has scenario, compute config
                result = compute_from_scenario(scenario, profile)
                config = result["config"]
                env = result.get("env", {})
                cli_args = result.get("cli_args", {})

            # Apply overrides
            if overrides:
                config = merge_overrides(config, overrides)
        else:
            # Build scenario from CLI arguments
            if not all([model, target_isl, target_osl, target_concurrency]):
                click.echo(
                    click.style(
                        "Error: When not using --recipe, you must specify: "
                        "--model, --target-isl, --target-osl, --target-concurrency",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)

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
                click.echo(click.style("Found matching recipe!", fg="green"))
                config = matched_recipe.get("config", {})
                env = matched_recipe.get("env", {})
                overrides = matched_recipe.get("overrides", {})
                if overrides:
                    config = merge_overrides(config, overrides)

                # Compute CLI args
                profile_name = profile or detect_profile(model)
                result = compute_from_scenario(scenario, profile_name)
                cli_args = result.get("cli_args", {})
            else:
                # Compute from scenario
                result = compute_from_scenario(scenario, profile)
                config = result["config"]
                env = result.get("env", {})
                cli_args = result.get("cli_args", {})

        # Validate scenario unless disabled
        if not no_validate:
            warnings = validate_scenario(scenario, strict=True)
            for warning in warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

            # Validate generated config
            config_warnings = validate_config(config)
            for warning in config_warnings:
                click.echo(click.style(str(warning), fg="yellow"), err=True)

        # Apply CLI overrides to cli_args
        if tp_size is not None:
            cli_args["tp_size"] = tp_size
        if ep_size is not None:
            cli_args["ep_size"] = ep_size

        # Write config to file
        output_path = Path(output)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Determine which profile was used
        profile_name = (
            profile or scenario.get("profile") or detect_profile(scenario.get("model", ""))
        )
        if not profile_name:
            profile_name = "custom"

        # Print result
        print_result(scenario, config, env, cli_args, str(output_path), profile_name)

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
