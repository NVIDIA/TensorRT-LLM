from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, CliSubCommand, SettingsConfigDict, get_subcommand

from tensorrt_llm.configure.profile import InferenceMaxProfile, ThroughputLatencySLAProfile
from tensorrt_llm.logger import logger


class CommonOptions(BaseModel):
    """Common options for all subcommands of the trtllm-configure CLI tool."""

    output: Optional[Path] = Field(
        default=None,
        description="YAML file path where the optimized config will be written.",
        validation_alias=AliasChoices("output", "o"),
    )

    @model_validator(mode="after")
    def validate_output(self) -> "CommonOptions":
        """Verify that output file is a valid YAML file path and does not already exist."""
        if self.output is not None:
            if self.output.suffix != ".yaml":
                raise ValueError(f"Output file must be a YAML file. Got '{self.output}'.")
            if self.output.exists():
                raise ValueError(f"Output file '{self.output}' already exists.")
        return self


class InferenceMaxSubCommand(InferenceMaxProfile, CommonOptions):
    """Optimize TensorRT LLM for an InferenceMax benchmark workload with a specific number of concurrent requests."""


class ThroughputLatencySLASubCommand(ThroughputLatencySLAProfile, CommonOptions):
    """Optimize TensorRT LLM to meet a throughput and latency SLA."""


TRTLLMConfigureSubCommand = Union[InferenceMaxSubCommand, ThroughputLatencySLASubCommand]


class TRTLLMConfigure(BaseSettings):
    # NOTE: the docstring below is used to generate the CLI help message
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

    inferencemax: CliSubCommand[InferenceMaxSubCommand] = Field(
        description=InferenceMaxSubCommand.__doc__
    )
    # TODO: add support for throughput/latency SLA subcommand
    # throughput_latency: CliSubCommand[ThroughputLatencySLASubCommand] = Field(
    #     description=ThroughputLatencySLASubCommand.__doc__
    # )

    def run(self) -> None:
        """Main entrypoint for the trtllm-configure CLI tool."""
        subcommand: TRTLLMConfigureSubCommand = get_subcommand(self)

        config = subcommand.get_config()

        # exclude_unset and exclude_default are explicitly used to avoid including default values
        config_dict = config.model_dump(exclude_unset=True, exclude_default=True)
        logger.info(f"Optimized config: \n\n{yaml.safe_dump(config_dict)}")

        if subcommand.output is None:
            logger.info(
                "No output file specified. To write the optimized config to a file, use the --output / -o flag."
            )
        else:
            with open(subcommand.output, "w") as f:
                f.write(yaml.safe_dump(config_dict))
            logger.info(f"Optimized config written to {subcommand.output}")
            logger.info("To serve the model with optimized settings, run the following command:")
            logger.info(f"    trtllm-serve {subcommand.model} --config {subcommand.output}")


def main():
    TRTLLMConfigure().run()


if __name__ == "__main__":
    main()
