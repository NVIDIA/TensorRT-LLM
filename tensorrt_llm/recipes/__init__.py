"""TensorRT-LLM Recipe System for Optimized Inference Configurations.

This module provides a recipe-based configuration system for TensorRT-LLM,
allowing users to generate optimized configurations for specific inference
scenarios.
"""

from .matcher import compute_from_scenario, detect_profile, match_recipe
from .profiles import PROFILE_REGISTRY, ProfileBase, get_profile, register_profile
from .validator import validate_config, validate_scenario

__all__ = [
    "PROFILE_REGISTRY",
    "ProfileBase",
    "get_profile",
    "register_profile",
    "detect_profile",
    "match_recipe",
    "compute_from_scenario",
    "validate_scenario",
    "validate_config",
]
