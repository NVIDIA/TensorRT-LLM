from enum import StrEnum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, CliSubCommand, SettingsConfigDict, get_subcommand

from .constraints import BaseConstraints, BenchmarkConstraints
from .profile import PROFILE_REGISTRY


def generate_subcommand_description(constraints_cls: type[BaseConstraints]) -> str:
    """Generate a description of the subcommand for the given constraints class."""
    profiles = PROFILE_REGISTRY[constraints_cls]
    description = constraints_cls._get_cli_description() + "\n\n"

    description += (
        "The --profile flag can be used to specify which profile to use. A profile defines the strategy used to "
        "generate the optimized config. The available profiles are:\n\n"
    )

    for profile in profiles:
        metadata = profile._get_metadata()
        description += f"- {metadata.cli_name}: {metadata.description}\n"

    return description


def create_subcommand(constraints_cls: type[BaseConstraints]) -> CliSubCommand:
    """Create a Pydantic CLI subcommand for the given constraints class."""
    profiles = PROFILE_REGISTRY[constraints_cls]
    default_profile = next(profile for profile in profiles if profile._get_metadata().is_default)
    ProfileEnum = StrEnum(
        "ProfileEnum",
        [profile._get_metadata().cli_name for profile in profiles],
    )

    class SubCommand(constraints_cls):
        # The docstring is shown as the help message for the subcommand
        __doc__ = generate_subcommand_description(constraints_cls)

        # Common options for all subcommands
        output: Optional[Path] = Field(
            default=None,
            description="YAML file path where the optimized config will be written.",
            validation_alias=AliasChoices("output", "o"),
        )

        profile: ProfileEnum = Field(
            default=default_profile._get_metadata().cli_name,
            description="Name of the profile to use, which defines the strategy used to generate the optimized config. "
            "See above for a description of the available profiles.",
        )

        @model_validator(mode="after")
        def validate_output(self) -> "SubCommand":
            """Verify that output file is a valid YAML file path and does not already exist."""
            if self.output is not None:
                if self.output.suffix != ".yaml":
                    raise ValueError(f"Output file must be a YAML file. Got '{self.output}'.")
                if self.output.exists():
                    print(f"Output file '{self.output}' already exists, will overwrite it.")
            return self

        def run(self) -> None:
            # Dispatch to the appropriate profile
            profiles = PROFILE_REGISTRY[constraints_cls]
            profile_cls = next(
                profile for profile in profiles if profile._get_metadata().cli_name == self.profile
            )
            config = profile_cls().get_config(self)
            print(f"Found optimized config: \n\n{yaml.safe_dump(config)}")

            if self.output is None:
                print(
                    "No output file specified. To write the optimized config to a file, use the --output / -o flag."
                )
            else:
                with open(self.output, "w") as f:
                    f.write(yaml.safe_dump(config))
                print(f"Optimized config written to {self.output}")
                print("To serve the model with optimized settings, run the following command:\n")
                print(f"trtllm-serve {self.model} --config {self.output}")

    return CliSubCommand[SubCommand]


BenchmarkSubCommand = create_subcommand(BenchmarkConstraints)
# TODO: add support for throughput/latency subcommand
# ThroughputLatencySubCommand = create_subcommand(ThroughputLatencyConstraints)


class TRTLLMConfigure(BaseSettings):
    # The docstring below is used to generate the CLI help message
    """The trtllm-configure CLI tool allows you to optimize the configuration of TensorRT LLM for your specific
    inference scenario.
    """  # noqa: D205

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="trtllm-configure",
        cli_enforce_required=True,  # Make required fields enforced at CLI level
        cli_implicit_flags=True,  # Boolean fields will be exposed as e.g. --flag and --no-flag
        cli_avoid_json=True,  # Do not expose JSON string options for nested models
    )

    benchmark: BenchmarkSubCommand = Field(description=BenchmarkConstraints._get_cli_description())
    # TODO: add support for throughput/latency SLA subcommand
    # throughput_latency: CliSubCommand[ThroughputLatencySubCommand] = Field(
    #     description=ThroughputLatencySubCommand.__doc__
    # )

    def run(self) -> None:
        """Main entrypoint for the trtllm-configure CLI tool."""
        subcommand = get_subcommand(self)
        subcommand.run()


def main():
    TRTLLMConfigure().run()


if __name__ == "__main__":
    main()
