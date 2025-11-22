"""Pydantic schemas for recipe validation.

This module provides the single source of truth for recipe file structure.
Recipes are YAML files that combine scenario parameters (benchmark settings)
with LLM API options (model configuration).
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ScenarioConfig(BaseModel):
    """Scenario parameters for benchmark configuration.

    Defines the target workload characteristics for performance testing.
    """

    model_config = {"extra": "forbid"}  # Strict validation - only known fields allowed

    # Required fields
    model: str = Field(description="Model identifier (e.g., 'tinyllama', 'llama-7b')")
    target_isl: int = Field(gt=0, description="Target input sequence length (must be positive)")
    target_osl: int = Field(gt=0, description="Target output sequence length (must be positive)")
    target_concurrency: int = Field(gt=0, description="Target concurrency rate (must be positive)")

    # Optional benchmark-specific fields for trtllm-bench auto-dataset generation
    isl_stdev: int = Field(
        default=0,
        ge=0,
        description="ISL standard deviation for auto-dataset generation (0=exact, for trtllm-bench)",
    )
    osl_stdev: int = Field(
        default=0,
        ge=0,
        description="OSL standard deviation for auto-dataset generation (0=exact, for trtllm-bench)",
    )
    num_requests: int = Field(
        default=512,
        gt=0,
        description="Number of requests for auto-dataset generation (consumed by trtllm-bench)",
    )

    # Metadata (optional, not validated beyond type)
    gpu: Optional[str] = Field(default=None, description="GPU type metadata (e.g., 'H100', 'A100')")
    num_gpus: Optional[int] = Field(default=None, ge=1, description="Number of GPUs (metadata)")


class RecipeConfig(BaseModel):
    """Complete recipe configuration.

    A recipe combines:
    - scenario: Benchmark workload parameters
    - llm_api_options: LLM API configuration (validated separately by LlmArgs)
    - env: Environment variables to set
    """

    model_config = {"extra": "forbid"}  # Strict validation at top level

    # Required
    scenario: ScenarioConfig = Field(description="Benchmark scenario parameters")

    # Optional
    env: Dict[str, Any] = Field(default_factory=dict, description="Environment variables")
    llm_api_options: Dict[str, Any] = Field(
        default_factory=dict, description="LLM API configuration"
    )
