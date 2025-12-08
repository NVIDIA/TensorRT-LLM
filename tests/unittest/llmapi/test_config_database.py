"""L0 tests for validating config database YAML files against LLM API pydantic models.

This module validates that all YAML configuration files in examples/configs/database/
conform to the TorchLlmArgs pydantic model structure. This provides an early
validation check without the cost of actually running trtllm-serve.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, update_llm_args_with_extra_dict

# Path to config directories
# File is at: tests/unittest/llmapi/test_config_database.py
# parents[0]=llmapi, parents[1]=unittest, parents[2]=tests, parents[3]=tensorrt_llm
CONFIG_ROOT = Path(__file__).parents[3] / "examples" / "configs"
DATABASE_DIR = CONFIG_ROOT / "database"

# Collect all database config files (excluding lookup.yaml which is an index file)
DATABASE_CONFIGS = (
    [c for c in DATABASE_DIR.rglob("*.yaml") if c.name != "lookup.yaml"]
    if DATABASE_DIR.exists()
    else []
)


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only test execution.

    The TorchLlmArgs validators include GPU-dependent code:
    - validate_gpus_per_node: calls torch.cuda.device_count()
    - validate_dtype: calls torch.cuda.get_device_properties(0).major

    This fixture mocks these functions to allow validation tests to run
    without requiring GPU access.
    """
    mock_props = Mock()
    mock_props.major = 8  # Simulate SM80+ GPU (Ampere or newer)
    mock_props.name = "Mock GPU"

    with patch("torch.cuda.device_count", return_value=8):
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            with patch("torch.cuda.is_available", return_value=True):
                yield


def get_config_id(config_path: Path) -> str:
    """Generate a readable test ID from config path."""
    try:
        return str(config_path.relative_to(DATABASE_DIR))
    except ValueError:
        return str(config_path.name)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DATABASE_CONFIGS, ids=get_config_id)
def test_config_validates_against_llm_args(config_path: Path):
    """Validate config YAML against TorchLlmArgs pydantic model.

    This test:
    1. Loads the YAML config file
    2. Creates a base TorchLlmArgs with dummy model
    3. Merges the config using update_llm_args_with_extra_dict
    4. Validates by reconstructing TorchLlmArgs

    Any invalid fields will cause a ValidationError.
    """
    # Load config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Handle empty configs
    if config_dict is None:
        config_dict = {}

    # Create base args with dummy model and skip tokenizer init
    # skip_tokenizer_init=True prevents attempting to load a real tokenizer
    base_args = TorchLlmArgs(model="dummy/model", skip_tokenizer_init=True)
    base_dict = base_args.model_dump()

    # Merge config using the existing update function
    merged = update_llm_args_with_extra_dict(base_dict, config_dict)

    # Validate by reconstructing TorchLlmArgs
    # This will raise ValidationError if config has invalid fields
    TorchLlmArgs(**merged)


@pytest.mark.part0
def test_database_config_count():
    """Sanity check: ensure we found the expected config files."""
    assert len(DATABASE_CONFIGS) > 0, "No database config files found"
    print(f"\nFound {len(DATABASE_CONFIGS)} database configs")
