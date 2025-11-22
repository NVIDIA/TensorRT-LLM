"""Recipe validation and configuration schemas.

This package provides Pydantic schemas for validating recipe YAML files.
Recipes combine scenario parameters (benchmark settings) with LLM API
configuration for reproducible performance testing.
"""

from .schema import RecipeConfig, ScenarioConfig

__all__ = ["RecipeConfig", "ScenarioConfig"]
