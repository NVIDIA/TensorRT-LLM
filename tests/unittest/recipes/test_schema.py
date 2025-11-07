"""Unit tests for recipe schema validation.

These tests verify that Pydantic schemas correctly validate recipe YAML files.
Minimal tests are needed since Pydantic handles validation automatically.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tensorrt_llm.recipes import RecipeConfig, ScenarioConfig


def test_tinyllama_recipe_validates():
    """Test that the tinyllama recipe file validates successfully."""
    recipe_path = Path(__file__).parents[3] / "tensorrt_llm/recipes/db/tinyllama-test.yaml"

    with open(recipe_path) as f:
        data = yaml.safe_load(f)

    # Should not raise ValidationError
    recipe = RecipeConfig(**data)

    # Verify basic fields
    assert recipe.scenario.model == "tinyllama"
    assert recipe.scenario.target_isl == 1024
    assert recipe.scenario.target_osl == 256
    assert recipe.scenario.target_concurrency == 32


def test_all_recipes_in_db_validate():
    """Test that all recipe files in db/ directory validate successfully."""
    recipes_dir = Path(__file__).parents[3] / "tensorrt_llm/recipes/db"

    recipe_files = list(recipes_dir.glob("*.yaml"))
    assert len(recipe_files) > 0, "No recipe files found in db/ directory"

    for recipe_file in recipe_files:
        with open(recipe_file) as f:
            data = yaml.safe_load(f)

        # Should not raise ValidationError
        RecipeConfig(**data)


def test_invalid_scenario_caught():
    """Test that Pydantic catches invalid scenario parameters."""
    # Negative target_isl should be caught
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            model="test",
            target_isl=-1,  # Invalid: must be positive
            target_osl=256,
            target_concurrency=32,
        )

    # Verify the error is about target_isl constraint
    assert "target_isl" in str(exc_info.value)


def test_missing_required_fields():
    """Test that missing required fields are caught."""
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            model="test",
            target_isl=1024,
            # Missing target_osl and target_concurrency
        )

    error_str = str(exc_info.value)
    assert "target_osl" in error_str or "target_concurrency" in error_str


def test_optional_fields_have_defaults():
    """Test that optional fields have correct default values."""
    scenario = ScenarioConfig(model="test", target_isl=1024, target_osl=256, target_concurrency=32)

    assert scenario.isl_stdev == 0
    assert scenario.osl_stdev == 0
    assert scenario.num_requests == 512
