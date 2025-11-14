from abc import ABC, abstractmethod

from pydantic import BaseModel, validate_call

from tensorrt_llm.configure.scenario import BenchmarkScenario, ThroughputLatencySLAScenario
from tensorrt_llm.llmapi.llm_args import LlmArgs


class BaseProfile(BaseModel, ABC):
    """Base class for all profiles.

    A profile defines a particular strategy used to find an optimized config for a given scenario
    (e.g. database lookup, heuristics, etc.)

    Each profile is compatible with a specific scenario type.
    """

    @abstractmethod
    def get_config(self) -> LlmArgs: ...


class InferenceMaxProfile(BaseProfile, BenchmarkScenario):
    @validate_call
    def get_config(self) -> LlmArgs:
        # TODO: add logic to retrieve optimal recipe from database
        return LlmArgs()


class ThroughputLatencySLAProfile(BaseProfile, ThroughputLatencySLAScenario):
    @validate_call
    def get_config(self) -> LlmArgs:
        # TODO: add logic to retrieve optimal recipe
        return LlmArgs()
