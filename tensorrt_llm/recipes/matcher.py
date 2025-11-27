"""Recipe matching and profile detection logic."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .profiles import PROFILE_REGISTRY, get_profile


def detect_profile(model: str) -> Optional[str]:
    """Detect profile from model name using substring matching.

    Args:
        model: Model name or path (e.g., "nvidia/DeepSeek-R1-0528-FP4")

    Returns:
        Profile name if detected, None otherwise

    Examples:
        >>> detect_profile("nvidia/DeepSeek-R1-0528-FP4")
        'dsr1-fp4'
        >>> detect_profile("deepseek-ai/DeepSeek-R1-FP8")
        'dsr1-fp8'
        >>> detect_profile("openai/gpt-oss-120b")
        'gptoss-fp4'
    """
    model_lower = model.lower()

    # DeepSeek-R1 detection
    if "deepseek" in model_lower and "r1" in model_lower:
        if "fp4" in model_lower:
            return "dsr1-fp4"
        elif "fp8" in model_lower:
            return "dsr1-fp8"
        # Default to FP4 if precision not specified
        return "dsr1-fp4"

    # GPT-OSS detection
    if "gpt-oss" in model_lower or "gptoss" in model_lower:
        # Default to FP4 for GPT-OSS
        return "gptoss-fp4"

    return None


def load_recipe_file(recipe_path: str) -> Dict[str, Any]:
    """Load a recipe YAML file.

    Args:
        recipe_path: Path to the recipe YAML file

    Returns:
        Dictionary containing the recipe data

    Raises:
        FileNotFoundError: If recipe file doesn't exist
        yaml.YAMLError: If recipe file is invalid YAML
    """
    path = Path(recipe_path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(path, "r") as f:
        recipe = yaml.safe_load(f)

    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe file must contain a YAML dictionary, got: {type(recipe)}")

    return recipe


def find_recipe_files() -> list[Path]:
    """Find all recipe YAML files in the db directory.

    Returns:
        List of Path objects pointing to recipe files
    """
    # Get the directory where this file is located
    recipes_dir = Path(__file__).parent / "db"

    if not recipes_dir.exists():
        return []

    # Find all .yaml and .yml files
    recipe_files = list(recipes_dir.glob("*.yaml")) + list(recipes_dir.glob("*.yml"))
    return recipe_files


def find_all_matching_recipes(scenario: Dict[str, Any]) -> list[tuple[Path, Dict[str, Any]]]:
    """Find all recipes that exactly match the scenario parameters.

    Args:
        scenario: Dictionary containing scenario parameters

    Returns:
        List of tuples (recipe_path, recipe_dict) for all matching recipes
    """
    recipe_files = find_recipe_files()
    matches = []

    for recipe_path in recipe_files:
        try:
            recipe = load_recipe_file(str(recipe_path))

            # Check if recipe has a scenario section
            if "scenario" not in recipe:
                continue

            recipe_scenario = recipe["scenario"]

            # Try to match key parameters (exact match required)
            match_keys = ["model", "target_isl", "target_osl", "target_concurrency"]
            if all(
                scenario.get(key) == recipe_scenario.get(key)
                for key in match_keys
                if key in scenario
            ):
                # Found a match - add to list
                matches.append((recipe_path, recipe))

        except Exception:
            # Skip invalid recipe files
            continue

    return matches


def match_recipe(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to match scenario against existing recipe files.

    Args:
        scenario: Dictionary containing scenario parameters

    Returns:
        Matched recipe dictionary if found, None otherwise

    Note: This function returns the first match. Use find_all_matching_recipes()
    to get all matches and detect ambiguous scenarios.
    """
    matches = find_all_matching_recipes(scenario)
    return matches[0][1] if matches else None


def compute_from_scenario(
    scenario: Dict[str, Any], profile: Optional[str] = None
) -> Dict[str, Any]:
    """Compute configuration from scenario using profile logic.

    Args:
        scenario: Dictionary containing scenario parameters
        profile: Profile name to use (if None, will check scenario['profile'] then auto-detect)

    Returns:
        Dictionary with 'config', 'env', and 'cli_args' keys

    Raises:
        ValueError: If profile cannot be determined or is invalid
    """
    # Use profile from arguments, then scenario dict, then auto-detect
    if profile is None:
        profile = scenario.get("profile")

    if profile is None:
        profile = detect_profile(scenario.get("model", ""))
        if profile is None:
            raise ValueError(
                f"Could not auto-detect profile from model '{scenario.get('model')}'. "
                f"Please specify --profile explicitly or set 'profile' in the scenario. "
                f"Available profiles: {', '.join(PROFILE_REGISTRY.keys())}"
            )

    # Get profile instance and compute configuration
    profile_obj = get_profile(profile)
    result = profile_obj.compute_config(scenario)

    return result


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override values into configuration.

    Args:
        config: Base configuration dictionary
        overrides: Override values to apply

    Returns:
        Merged configuration dictionary
    """
    result = config.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_overrides(result[key], value)
        else:
            # Override value
            result[key] = value

    return result
