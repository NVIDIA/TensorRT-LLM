from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from tensorrt_llm.configure.constraints import BaseConstraints, BenchmarkConstraints


class ProfileMetadata(BaseModel):
    """Metadata about a profile."""

    constraints_cls: type[BaseConstraints] = Field(
        description="The constraints class that this profile is compatible with."
    )
    cli_name: str = Field(description="The name of the profile as it will be exposed in the CLI.")
    description: str = Field(
        description="A description of the profile which will be shown in the CLI help message."
    )
    is_default: bool = Field(
        default=False,
        description="Whether this profile is the default profile for the constraints class. There can only be one "
        "default profile per constraints class.",
    )


class BaseProfile(ABC):
    """Base class for all profiles.

    A profile defines a particular strategy used to find an optimized config for a given scenario
    (e.g. database lookup, heuristics, etc.)

    Each profile is compatible with a specific type of constraints.
    """

    @classmethod
    @abstractmethod
    def _get_metadata(cls) -> ProfileMetadata:
        """Get the metadata associated with this profile."""

    @abstractmethod
    def get_config(self, constraints: BaseConstraints) -> dict[str, Any]:
        """Retrieve or generate the optimal config for the given constraints."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate that there is only one default profile per constraints class
        metadata = cls._get_metadata()
        if metadata.is_default:
            for other_profile in PROFILE_REGISTRY[metadata.constraints_cls]:
                other_metadata = other_profile._get_metadata()
                if other_metadata.is_default:
                    raise ValueError(
                        f"Multiple default profiles found for constraints class {metadata.constraints_cls.__name__}: "
                        f"{other_profile.__name__} and {cls.__name__}"
                    )

        # Add this class to the profile registry
        PROFILE_REGISTRY[metadata.constraints_cls].append(cls)


# Maps constraints classes to the list of profiles compatible with those constraints
PROFILE_REGISTRY: defaultdict[type[BaseConstraints], list[type[BaseProfile]]] = defaultdict(list)


class InferenceMaxProfile(BaseProfile):
    @classmethod
    def _get_metadata(cls) -> ProfileMetadata:
        return ProfileMetadata(
            constraints_cls=BenchmarkConstraints,
            cli_name="inferencemax",
            description=(
                "Retrieve optimized settings from a database of configs used for SemiAnalysis InferenceMax "
                "benchmarks."
            ),
            is_default=True,
        )

    def get_config(self, constraints: BenchmarkConstraints) -> dict[str, Any]:
        # TODO: add logic to retrieve optimal config from database
        return {}
