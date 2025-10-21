"""Test suite for DynamicYamlMixInForSettings utility class."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from tensorrt_llm._torch.auto_deploy.utils._config import DynamicYamlMixInForSettings


class SimpleModel(BaseModel):
    """Simple model for testing."""

    value: int
    name: str
    flag: bool = False


class OptionModel(BaseModel):
    """Model with literal options."""

    name: str
    option: Literal["on", "off"] = "off"


class BasicSettings(DynamicYamlMixInForSettings, BaseSettings):
    """Basic settings class for testing."""

    simple: SimpleModel
    option: OptionModel


def create_settings_with_default_yaml(default_yaml_path: Path):
    """Create a settings class with a specific default yaml file path."""

    class SettingsWithDefaultYaml(DynamicYamlMixInForSettings, BaseSettings):
        """Settings class with default yaml file."""

        yaml_default: str = str(default_yaml_path)

        simple: SimpleModel
        option: OptionModel

    return SettingsWithDefaultYaml


def create_nested_settings(nested_default_yaml_path: Path):
    """Create a nested settings class with a specific default yaml file path."""

    class NestedSettings(DynamicYamlMixInForSettings, BaseSettings):
        """Nested settings class for testing precedence."""

        yaml_default: str = str(nested_default_yaml_path)

        args: BasicSettings
        extra_field: str = "default"

    return NestedSettings


def create_mode_based_settings(
    train_yaml: Path,
    eval_yaml: Path,
    default_mode: str = "",
    default_yaml_default: str = "",
):
    """Create a settings class that selects default yaml via a mode mapping.

    The returned class overrides _get_yaml_default_from_mode to map
    mode -> yaml file path and optionally sets default values for
    `mode` and `yaml_default` to enable precedence testing.
    """

    class ModeBasedSettings(DynamicYamlMixInForSettings, BaseSettings):
        """Settings class whose default yaml is determined by `mode`."""

        # Allow explicit defaults for precedence testing
        mode: Literal["train", "eval", ""] = default_mode
        yaml_default: str = default_yaml_default

        simple: SimpleModel
        option: OptionModel

        @classmethod
        def _get_yaml_default_from_mode(cls, mode: str | None) -> str | None:
            mapping = {
                "train": str(train_yaml),
                "eval": str(eval_yaml),
            }
            return mapping.get((mode or ""))

    return ModeBasedSettings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def basic_yaml_files(temp_dir):
    """Create basic yaml test files."""
    files = {}

    # Default config
    files["default"] = temp_dir / "default.yaml"
    files["default"].write_text("""
simple:
  value: 100
  name: "default"
  flag: true
option:
  name: "default_option"
  option: "on"
""")

    # Override config 1
    files["config1"] = temp_dir / "config1.yaml"
    files["config1"].write_text("""
simple:
  value: 200
  name: "config1"
option:
  name: "config1_option"
""")

    # Override config 2
    files["config2"] = temp_dir / "config2.yaml"
    files["config2"].write_text("""
simple:
  flag: false
  name: "config2"
option:
  option: "off"
""")

    # Partial config
    files["partial"] = temp_dir / "partial.yaml"
    files["partial"].write_text("""
simple:
  value: 999
""")

    return files


@pytest.fixture
def nested_yaml_files(temp_dir):
    """Create nested yaml test files."""
    files = {}

    # Nested default
    files["nested_default"] = temp_dir / "nested_default.yaml"
    files["nested_default"].write_text("""
args:
  simple:
    value: 50
    name: "nested_default"
    flag: true
  option:
    name: "nested_default_option"
    option: "on"
extra_field: "nested_default_extra"
""")

    # Nested override 1
    files["nested_override1"] = temp_dir / "nested_override1.yaml"
    files["nested_override1"].write_text("""
args:
  simple:
    value: 150
    name: "nested_override1"
  option:
    name: "nested_override1_option"
extra_field: "nested_override1_extra"
""")

    # Nested override 2
    files["nested_override2"] = temp_dir / "nested_override2.yaml"
    files["nested_override2"].write_text("""
args:
  simple:
    flag: false
    name: "nested_override2"
  option:
    option: "off"
""")

    # Inner config (for args.yaml_extra)
    files["inner_config"] = temp_dir / "inner_config.yaml"
    files["inner_config"].write_text("""
simple:
  value: 300
  name: "inner_config"
option:
  name: "inner_config_option"
  option: "on"
""")

    return files


# Basic YAML loading tests
def test_no_yaml_configs():
    """Test settings without any yaml configs."""
    with pytest.raises(ValidationError):
        # Should fail because required fields are missing
        BasicSettings()


def test_single_yaml_config(basic_yaml_files):
    """Test loading a single yaml config file."""
    settings = BasicSettings(yaml_extra=[basic_yaml_files["config1"]])

    assert settings.simple.value == 200
    assert settings.simple.name == "config1"
    assert settings.simple.flag is False  # default value
    assert settings.option.name == "config1_option"
    assert settings.option.option == "off"  # default value


def test_multiple_yaml_configs_merging(basic_yaml_files):
    """Test merging multiple yaml configs in order."""
    # Order: config1, config2 (config2 should override config1)
    settings = BasicSettings(yaml_extra=[basic_yaml_files["config1"], basic_yaml_files["config2"]])

    assert settings.simple.value == 200  # from config1
    assert settings.simple.name == "config2"  # overridden by config2
    assert settings.simple.flag is False  # from config2
    assert settings.option.name == "config1_option"  # from config1
    assert settings.option.option == "off"  # from config2


def test_partial_yaml_config(basic_yaml_files):
    """Test partial yaml config with some missing fields."""
    with pytest.raises(ValidationError):
        # Should fail because 'name' is missing from simple
        BasicSettings(yaml_extra=[basic_yaml_files["partial"]])


# Default YAML file tests
def test_default_yaml_file_loading(basic_yaml_files):
    """Test loading default yaml file from model_config."""
    SettingsWithDefaultYaml = create_settings_with_default_yaml(basic_yaml_files["default"])
    settings = SettingsWithDefaultYaml()

    assert settings.simple.value == 100
    assert settings.simple.name == "default"
    assert settings.simple.flag is True
    assert settings.option.name == "default_option"
    assert settings.option.option == "on"


def test_default_yaml_with_additional_configs(basic_yaml_files):
    """Test default yaml file with additional configs."""
    SettingsWithDefaultYaml = create_settings_with_default_yaml(basic_yaml_files["default"])
    settings = SettingsWithDefaultYaml(yaml_extra=[basic_yaml_files["config1"]])

    # Additional configs should override default
    assert settings.simple.value == 200  # from config1
    assert settings.simple.name == "config1"  # from config1
    assert settings.simple.flag is True  # from default
    assert settings.option.name == "config1_option"  # from config1
    assert settings.option.option == "on"  # from default


def test_multiple_additional_configs_with_default(basic_yaml_files):
    """Test multiple additional configs with default yaml file."""
    SettingsWithDefaultYaml = create_settings_with_default_yaml(basic_yaml_files["default"])
    settings = SettingsWithDefaultYaml(
        yaml_extra=[basic_yaml_files["config1"], basic_yaml_files["config2"]]
    )

    # Order: default.yaml, config1.yaml, config2.yaml
    assert settings.simple.value == 200  # from config1
    assert settings.simple.name == "config2"  # from config2 (last override)
    assert settings.simple.flag is False  # from config2
    assert settings.option.name == "config1_option"  # from config1
    assert settings.option.option == "off"  # from config2


# Nested settings tests
def test_nested_default_yaml(nested_yaml_files):
    """Test nested settings with default yaml file."""
    NestedSettings = create_nested_settings(nested_yaml_files["nested_default"])
    settings = NestedSettings()

    assert settings.args.simple.value == 50
    assert settings.args.simple.name == "nested_default"
    assert settings.args.simple.flag is True
    assert settings.args.option.name == "nested_default_option"
    assert settings.args.option.option == "on"
    assert settings.extra_field == "nested_default_extra"


def test_nested_with_outer_yaml_configs(nested_yaml_files):
    """Test nested settings with yaml configs at outer level."""
    NestedSettings = create_nested_settings(nested_yaml_files["nested_default"])
    settings = NestedSettings(yaml_extra=[nested_yaml_files["nested_override1"]])

    # Outer config should override inner defaults
    assert settings.args.simple.value == 150
    assert settings.args.simple.name == "nested_override1"
    assert settings.args.simple.flag is True  # from default
    assert settings.args.option.name == "nested_override1_option"
    assert settings.args.option.option == "on"  # from default
    assert settings.extra_field == "nested_override1_extra"


def test_nested_with_inner_yaml_configs(nested_yaml_files):
    """Test nested settings with yaml configs at inner level."""
    NestedSettings = create_nested_settings(nested_yaml_files["nested_default"])
    # Create nested settings with inner yaml configs
    settings = NestedSettings(args=BasicSettings(yaml_extra=[nested_yaml_files["inner_config"]]))

    # Inner yaml configs should be processed
    assert settings.args.simple.value == 300
    assert settings.args.simple.name == "inner_config"
    assert settings.args.simple.flag is False  # default
    assert settings.args.option.name == "inner_config_option"
    assert settings.args.option.option == "on"
    assert settings.extra_field == "nested_default_extra"  # from outer default


def test_nested_precedence_outer_over_inner(nested_yaml_files):
    """Test precedence: outer yaml configs override inner yaml configs."""
    NestedSettings = create_nested_settings(nested_yaml_files["nested_default"])
    # Both outer and inner yaml configs
    # Outer yaml config gets converted to init arguments for inner settings ("args")
    # The yaml_extra for the inner settings are passed in as yaml setting with lower precedence
    settings = NestedSettings(
        yaml_extra=[nested_yaml_files["nested_override1"]],
        args={"yaml_extra": [nested_yaml_files["inner_config"]]},
    )

    # Outer should take precedence over inner
    assert settings.args.simple.value == 150  # from outer (nested_override1)
    assert settings.args.simple.name == "nested_override1"  # from outer
    assert settings.args.simple.flag is True  # from outer default
    assert settings.args.option.name == "nested_override1_option"  # from outer
    assert settings.args.option.option == "on"  # from outer default
    assert settings.extra_field == "nested_override1_extra"


def test_inner_init_precedence_over_outer_yaml(nested_yaml_files):
    """Test precedence: outer yaml configs override inner yaml configs."""
    NestedSettings = create_nested_settings(nested_yaml_files["nested_default"])
    # Both outer and inner yaml configs
    settings = NestedSettings(
        yaml_extra=[nested_yaml_files["nested_override1"]],
        args=BasicSettings(yaml_extra=[nested_yaml_files["inner_config"]]),
    )

    # Initialized BasicSettings takes precedence over yaml since it's a init argument
    assert settings.args.simple.value == 300
    assert settings.args.simple.name == "inner_config"  # from inner yaml
    assert settings.args.simple.flag is False  # from inner yaml
    assert settings.args.option.name == "inner_config_option"  # from inner yaml
    assert settings.args.option.option == "on"  # from inner yaml
    assert settings.extra_field == "nested_override1_extra"


# Precedence order tests
def test_init_overrides_yaml(basic_yaml_files):
    """Test that init values override yaml configs."""
    init_simple = SimpleModel(value=999, name="init_value", flag=True)
    init_option = OptionModel(name="init_option", option="on")

    settings = BasicSettings(
        simple=init_simple, option=init_option, yaml_extra=[basic_yaml_files["config1"]]
    )

    # Init values should override yaml
    assert settings.simple.value == 999
    assert settings.simple.name == "init_value"
    assert settings.simple.flag is True
    assert settings.option.name == "init_option"
    assert settings.option.option == "on"


def test_env_overrides_yaml(basic_yaml_files):
    """Test that environment variables override yaml configs."""
    with patch.dict(
        os.environ,
        {"SIMPLE": '{"value": 888, "name": "env_value"}', "OPTION": '{"name": "env_option"}'},
    ):
        settings = BasicSettings(yaml_extra=[basic_yaml_files["config1"]])

        # Environment should override yaml
        assert settings.simple.value == 888
        assert settings.simple.name == "env_value"
        assert settings.simple.flag is False  # from yaml (no env override)
        assert settings.option.name == "env_option"
        assert settings.option.option == "off"  # from yaml default


def test_partial_env_override(basic_yaml_files):
    """Test partial environment variable override."""
    with patch.dict(os.environ, {"SIMPLE": '{"flag": true}', "OPTION": '{"option": "on"}'}):
        settings = BasicSettings(yaml_extra=[basic_yaml_files["config1"]])

        # Mix of env and yaml values
        assert settings.simple.value == 200  # from yaml
        assert settings.simple.name == "config1"  # from yaml
        assert settings.simple.flag is True  # from env
        assert settings.option.name == "config1_option"  # from yaml
        assert settings.option.option == "on"  # from env


def test_missing_yaml_file(temp_dir):
    """Test handling of missing yaml file."""
    missing_file = temp_dir / "missing.yaml"

    # Should raise error for missing file
    with pytest.raises(ValueError):
        BasicSettings(yaml_extra=[missing_file])


def test_invalid_yaml_syntax(temp_dir):
    """Test handling of invalid yaml syntax."""
    invalid_yaml = temp_dir / "invalid.yaml"
    invalid_yaml.write_text("""
simple:
  value: 100
  name: "test"
  flag: true
option:
  name: "test_option"
  option: invalid_option  # This should cause validation error
""")

    with pytest.raises(ValidationError):
        BasicSettings(yaml_extra=[invalid_yaml])


def test_malformed_yaml_file(temp_dir):
    """Test handling of malformed yaml file."""
    malformed_yaml = temp_dir / "malformed.yaml"
    malformed_yaml.write_text("""
simple:
  value: 100
  name: "test"
  flag: true
option:
  name: "test_option"
  option: "on"
  invalid_structure: {
    missing_close_brace: "value"
""")

    with pytest.raises(Exception):  # Should raise yaml parsing error
        BasicSettings(yaml_extra=[malformed_yaml])


# Deep merging tests
def test_deep_merge_nested_dicts(temp_dir):
    """Test deep merging of nested dictionaries."""
    base_yaml = temp_dir / "base.yaml"
    base_yaml.write_text("""
simple:
  value: 100
  name: "base"
  flag: true
option:
  name: "base_option"
  option: "on"
""")

    override_yaml = temp_dir / "override.yaml"
    override_yaml.write_text("""
simple:
  value: 200
  # name should remain from base
  # flag should remain from base
option:
  option: "off"
  # name should remain from base
""")

    settings = BasicSettings(yaml_extra=[base_yaml, override_yaml])

    # Deep merge should preserve non-overridden values
    assert settings.simple.value == 200  # overridden
    assert settings.simple.name == "base"  # preserved
    assert settings.simple.flag is True  # preserved
    assert settings.option.name == "base_option"  # preserved
    assert settings.option.option == "off"  # overridden


def test_complex_deep_merge_order(temp_dir):
    """Test complex deep merge with multiple files."""
    # Create three files with overlapping but different fields
    yaml1 = temp_dir / "yaml1.yaml"
    yaml1.write_text("""
simple:
  value: 100
  name: "yaml1"
  flag: true
option:
  name: "yaml1_option"
  option: "on"
""")

    yaml2 = temp_dir / "yaml2.yaml"
    yaml2.write_text("""
simple:
  value: 200
  name: "yaml2"
  # flag not specified, should remain from yaml1
option:
  name: "yaml2_option"
  # option not specified, should remain from yaml1
""")

    yaml3 = temp_dir / "yaml3.yaml"
    yaml3.write_text("""
simple:
  # value not specified, should remain from yaml2
  # name not specified, should remain from yaml2
  flag: false
option:
  # name not specified, should remain from yaml2
  option: "off"
""")

    settings = BasicSettings(yaml_extra=[yaml1, yaml2, yaml3])

    # Final result should be deep merge of all three
    assert settings.simple.value == 200  # from yaml2
    assert settings.simple.name == "yaml2"  # from yaml2
    assert settings.simple.flag is False  # from yaml3
    assert settings.option.name == "yaml2_option"  # from yaml2
    assert settings.option.option == "off"  # from yaml3


# New test case for nested dictionary deep merging
class SomeConfigModel(BaseModel):
    """Model representing a configuration entry."""

    param1: str
    param2: int = 42
    param3: bool = False


class SomeSettings(DynamicYamlMixInForSettings, BaseSettings):
    """Settings with a dictionary of config models."""

    configs: Dict[str, SomeConfigModel]


class SomeNestedSettings(DynamicYamlMixInForSettings, BaseSettings):
    """Nested settings containing SomeSettings."""

    args: SomeSettings
    extra_field: str = "default_extra"


def create_some_nested_settings_with_default_yaml(default_yaml_path: Path):
    """Create SomeNestedSettings with a default yaml file."""

    class SomeNestedSettingsWithDefaultYaml(DynamicYamlMixInForSettings, BaseSettings):
        """Nested settings with default yaml file."""

        yaml_default: str = str(default_yaml_path)

        args: SomeSettings
        extra_field: str = "default_extra"

    return SomeNestedSettingsWithDefaultYaml


@pytest.fixture
def dict_config_yaml_files(temp_dir):
    """Create yaml files for testing dictionary config deep merging."""
    files = {}

    # Inner settings config (for SomeSettings)
    files["inner_config"] = temp_dir / "inner_config.yaml"
    files["inner_config"].write_text("""
configs:
  k1:
    param1: "inner_k1_value"
    param2: 100
    param3: true
  k2:
    param1: "inner_k2_value"
    param2: 200
    param3: false
""")

    # Outer settings config (for SomeNestedSettings)
    files["outer_config"] = temp_dir / "outer_config.yaml"
    files["outer_config"].write_text("""
args:
  configs:
    k1:
      param1: "outer_k1_value"
      param2: 150
      # param3 not specified, should remain from inner
    k3:
      param1: "outer_k3_value"
      param2: 300
      param3: true
extra_field: "outer_extra_value"
""")

    # Default config for nested settings
    files["nested_default"] = temp_dir / "nested_default.yaml"
    files["nested_default"].write_text("""
args:
  configs:
    k1:
      param1: "default_k1_value"
      param2: 50
      param3: false
    k4:
      param1: "default_k4_value"
      param2: 400
      param3: true
extra_field: "default_extra_value"
""")

    return files


def test_nested_dict_deep_merge_basic(dict_config_yaml_files):
    """Test basic deep merging of nested dictionaries."""
    # Test with only inner config
    settings = SomeNestedSettings(args={"yaml_extra": [dict_config_yaml_files["inner_config"]]})

    # Should have k1 and k2 from inner config
    assert len(settings.args.configs) == 2
    assert "k1" in settings.args.configs
    assert "k2" in settings.args.configs

    # Check k1 values
    k1_config = settings.args.configs["k1"]
    assert k1_config.param1 == "inner_k1_value"
    assert k1_config.param2 == 100
    assert k1_config.param3 is True

    # Check k2 values
    k2_config = settings.args.configs["k2"]
    assert k2_config.param1 == "inner_k2_value"
    assert k2_config.param2 == 200
    assert k2_config.param3 is False

    # Check default extra field
    assert settings.extra_field == "default_extra"


def test_nested_dict_deep_merge_with_outer_yaml(dict_config_yaml_files):
    """Test deep merging when outer YAML contains nested dictionary configs."""
    # Create settings with both inner and outer configs
    # Use args as dict to allow deep merging, not as explicitly initialized object
    settings = SomeNestedSettings(
        yaml_extra=[dict_config_yaml_files["outer_config"]],
        args={"yaml_extra": [dict_config_yaml_files["inner_config"]]},
    )

    # Should have k1 (merged), k2 (from inner), and k3 (from outer)
    assert len(settings.args.configs) == 3
    assert "k1" in settings.args.configs
    assert "k2" in settings.args.configs
    assert "k3" in settings.args.configs

    # Check k1 values - outer should override inner for specified fields
    k1_config = settings.args.configs["k1"]
    assert k1_config.param1 == "outer_k1_value"  # from outer
    assert k1_config.param2 == 150  # from outer
    assert k1_config.param3 is True  # from inner (not overridden by outer)

    # Check k2 values - should remain from inner
    k2_config = settings.args.configs["k2"]
    assert k2_config.param1 == "inner_k2_value"
    assert k2_config.param2 == 200
    assert k2_config.param3 is False

    # Check k3 values - should be from outer
    k3_config = settings.args.configs["k3"]
    assert k3_config.param1 == "outer_k3_value"
    assert k3_config.param2 == 300
    assert k3_config.param3 is True

    # Check extra field from outer
    assert settings.extra_field == "outer_extra_value"


def test_nested_dict_deep_merge_with_default_yaml(dict_config_yaml_files):
    """Test deep merging with default yaml file and additional configs."""
    SomeNestedSettingsWithDefaultYaml = create_some_nested_settings_with_default_yaml(
        dict_config_yaml_files["nested_default"]
    )

    # Create settings with default yaml and additional outer config
    settings = SomeNestedSettingsWithDefaultYaml(
        yaml_extra=[dict_config_yaml_files["outer_config"]],
        args={"yaml_extra": [dict_config_yaml_files["inner_config"]]},
    )

    # Should have k1 (from outer, overriding both default and inner),
    # k2 (from inner), k3 (from outer), and k4 (from default)
    assert len(settings.args.configs) == 4
    assert "k1" in settings.args.configs
    assert "k2" in settings.args.configs
    assert "k3" in settings.args.configs
    assert "k4" in settings.args.configs

    # Check k1 values - outer should have highest precedence
    k1_config = settings.args.configs["k1"]
    assert k1_config.param1 == "outer_k1_value"  # from outer
    assert k1_config.param2 == 150  # from outer
    assert (
        k1_config.param3 is False
    )  # from default (outer config takes precedence over inner for k1)

    # Check k2 values - should be from inner
    k2_config = settings.args.configs["k2"]
    assert k2_config.param1 == "inner_k2_value"
    assert k2_config.param2 == 200
    assert k2_config.param3 is False

    # Check k3 values - should be from outer
    k3_config = settings.args.configs["k3"]
    assert k3_config.param1 == "outer_k3_value"
    assert k3_config.param2 == 300
    assert k3_config.param3 is True

    # Check k4 values - should be from default
    k4_config = settings.args.configs["k4"]
    assert k4_config.param1 == "default_k4_value"
    assert k4_config.param2 == 400
    assert k4_config.param3 is True

    # Check extra field from outer
    assert settings.extra_field == "outer_extra_value"


def test_nested_dict_deep_merge_precedence_order(dict_config_yaml_files):
    """Test the complete precedence order for nested dictionary deep merging."""
    SomeNestedSettingsWithDefaultYaml = create_some_nested_settings_with_default_yaml(
        dict_config_yaml_files["nested_default"]
    )

    # Create additional yaml file that partially overrides outer config
    partial_override = dict_config_yaml_files["outer_config"].parent / "partial_override.yaml"
    partial_override.write_text("""
args:
  configs:
    k1:
      param2: 999  # Override just param2
    k2:
      param1: "partial_k2_value"  # Add k2 config at outer level
extra_field: "partial_extra_value"
""")

    # Test with multiple yaml configs: default -> outer -> partial_override
    # and inner config for args
    settings = SomeNestedSettingsWithDefaultYaml(
        yaml_extra=[dict_config_yaml_files["outer_config"], partial_override],
        args={"yaml_extra": [dict_config_yaml_files["inner_config"]]},
    )

    # Should have all keys
    assert len(settings.args.configs) == 4

    # Check k1 - should be combination of all sources with proper precedence
    k1_config = settings.args.configs["k1"]
    assert k1_config.param1 == "outer_k1_value"  # from outer (not overridden by partial)
    assert k1_config.param2 == 999  # from partial_override (highest precedence)
    assert (
        k1_config.param3 is False
    )  # from default (outer config takes precedence over inner for k1)

    # Check k2 - should be from inner with partial outer override
    k2_config = settings.args.configs["k2"]
    assert k2_config.param1 == "partial_k2_value"  # from partial_override
    assert k2_config.param2 == 200  # from inner
    assert k2_config.param3 is False  # from inner

    # Check extra field from partial (highest precedence)
    assert settings.extra_field == "partial_extra_value"


def test_nested_dict_explicit_init_vs_yaml_precedence(dict_config_yaml_files):
    """Test that explicitly initialized objects take precedence over yaml configs."""
    # When we pass an explicitly initialized SomeSettings object,
    # it should take precedence over outer yaml configs
    settings = SomeNestedSettings(
        yaml_extra=[dict_config_yaml_files["outer_config"]],
        args=SomeSettings(yaml_extra=[dict_config_yaml_files["inner_config"]]),
    )

    # Should only have k1 and k2 from inner config (explicit init takes precedence)
    assert len(settings.args.configs) == 2
    assert "k1" in settings.args.configs
    assert "k2" in settings.args.configs
    assert "k3" not in settings.args.configs  # k3 from outer is ignored

    # Check k1 values - should be from inner only
    k1_config = settings.args.configs["k1"]
    assert k1_config.param1 == "inner_k1_value"  # from inner
    assert k1_config.param2 == 100  # from inner
    assert k1_config.param3 is True  # from inner

    # Check k2 values - should be from inner
    k2_config = settings.args.configs["k2"]
    assert k2_config.param1 == "inner_k2_value"
    assert k2_config.param2 == 200
    assert k2_config.param3 is False

    # Check extra field from outer (this still works at the top level)
    assert settings.extra_field == "outer_extra_value"


# Real world scenario tests
def test_cli_like_usage(temp_dir):
    """Test CLI-like usage with multiple config levels."""
    # Create a realistic scenario with default config and user overrides
    default_config = temp_dir / "default.yaml"
    default_config.write_text("""
simple:
  value: 42
  name: "default_model"
  flag: false
option:
  name: "default_option"
  option: "off"
""")

    user_config = temp_dir / "user.yaml"
    user_config.write_text("""
simple:
  value: 100
  flag: true
option:
  option: "on"
""")

    experiment_config = temp_dir / "experiment.yaml"
    experiment_config.write_text("""
simple:
  value: 999
  name: "experiment_model"
""")

    SettingsWithDefaultYaml = create_settings_with_default_yaml(default_config)
    # Simulate CLI usage: default + user + experiment configs
    settings = SettingsWithDefaultYaml(yaml_extra=[user_config, experiment_config])

    # Should have proper precedence
    assert settings.simple.value == 999  # from experiment (highest priority)
    assert settings.simple.name == "experiment_model"  # from experiment
    assert settings.simple.flag is True  # from user
    assert settings.option.name == "default_option"  # from default
    assert settings.option.option == "on"  # from user


def test_empty_yaml_configs_list():
    """Test with empty yaml_extra list."""
    # Should behave same as no yaml_extra
    with pytest.raises(ValidationError):
        BasicSettings(yaml_extra=[])


def test_relative_and_absolute_paths(basic_yaml_files, temp_dir):
    """Test with both relative and absolute paths."""
    # Create a relative path test using current working directory
    relative_config = temp_dir / "relative_config.yaml"
    relative_config.write_text(basic_yaml_files["config1"].read_text())

    # Test with a settings class that uses relative path for default
    relative_default = temp_dir / "relative_default.yaml"
    relative_default.write_text(basic_yaml_files["default"].read_text())

    # Use absolute path for the settings class
    SettingsWithDefaultYaml = create_settings_with_default_yaml(relative_default)

    settings = SettingsWithDefaultYaml(
        yaml_extra=[
            relative_config,  # absolute path (Path object)
            basic_yaml_files["config2"],  # absolute path (Path object)
        ]
    )

    # Should work with both path types
    assert settings.simple.value == 200  # from relative_config (same as config1)
    assert settings.simple.name == "config2"  # from config2


# Error for deprecated model_config.yaml_file
def test_model_config_yaml_file_raises_error(temp_dir):
    """Using model_config.yaml_file must raise a clear error."""
    mc_default = temp_dir / "mc_default.yaml"
    mc_default.write_text(
        """
simple:
  value: 1
  name: "mc"
option:
  name: "mc_option"
  option: "on"
        """
    )

    class SettingsUsingModelConfig(DynamicYamlMixInForSettings, BaseSettings):
        # Intentionally use deprecated mechanism to trigger the error
        model_config = ConfigDict(yaml_file=str(mc_default))

        simple: SimpleModel
        option: OptionModel

    with pytest.raises(
        ValueError, match=r"Static yaml config via yaml_file in config is not supported"
    ):
        SettingsUsingModelConfig()


# =============================================================
# Additional tests for mode/yaml_default handling and precedence
# =============================================================


@pytest.fixture
def mode_yaml_files(temp_dir):
    """Create yaml files for testing mode and yaml_default precedence."""
    files = {}

    files["train"] = temp_dir / "train.yaml"
    files["train"].write_text(
        """
simple:
  value: 111
  name: "train"
  flag: true
option:
  name: "train_option"
  option: "on"
        """
    )

    files["eval"] = temp_dir / "eval.yaml"
    files["eval"].write_text(
        """
simple:
  value: 222
  name: "eval"
  flag: false
option:
  name: "eval_option"
  option: "off"
        """
    )

    return files


# 1) Handling when mode and/or yaml_default is provided
def test_error_when_both_mode_and_yaml_default_are_provided(mode_yaml_files):
    """Providing both mode and yaml_default must raise a ValueError."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"], mode_yaml_files["eval"]
    )

    with pytest.raises(
        ValueError, match=r"Cannot provide both 'mode' and 'yaml_default' simultaneously"
    ):
        ModeBasedSettings(mode="train", yaml_default=str(mode_yaml_files["eval"]))


def test_only_mode_uses_mode_selected_default_yaml(mode_yaml_files):
    """Providing only mode picks the default yaml based on the mode mapping."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"], mode_yaml_files["eval"]
    )

    settings = ModeBasedSettings(mode="eval")
    assert settings.simple.value == 222
    assert settings.simple.name == "eval"
    assert settings.simple.flag is False
    assert settings.option.name == "eval_option"
    assert settings.option.option == "off"


def test_only_yaml_default_uses_that_file(mode_yaml_files):
    """Providing only yaml_default picks that yaml file as default."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"], mode_yaml_files["eval"]
    )

    settings = ModeBasedSettings(yaml_default=str(mode_yaml_files["train"]))
    assert settings.simple.value == 111
    assert settings.simple.name == "train"
    assert settings.simple.flag is True
    assert settings.option.name == "train_option"
    assert settings.option.option == "on"


# 2) Precedence order for determining the default yaml
# Order (highest to lowest):
# 1. provided mode
# 2. provided yaml_default
# 3. yaml_default from default mode
# 4. default yaml_default


def test_default_selection_prefers_provided_mode_over_class_defaults(mode_yaml_files):
    """Provided mode overrides any class default yaml_default or default mode mapping."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"],
        mode_yaml_files["eval"],
        default_mode="train",
        default_yaml_default=str(mode_yaml_files["train"]),
    )

    settings = ModeBasedSettings(mode="eval")
    # Must select eval.yaml despite class defaults pointing to train
    assert settings.simple.name == "eval"
    assert settings.option.name == "eval_option"


def test_default_selection_prefers_provided_yaml_default_over_default_mode(mode_yaml_files):
    """Provided yaml_default takes precedence over the class default mode mapping."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"],
        mode_yaml_files["eval"],
        default_mode="train",
        default_yaml_default="",
    )

    settings = ModeBasedSettings(yaml_default=str(mode_yaml_files["eval"]))
    assert settings.simple.name == "eval"
    assert settings.option.name == "eval_option"


def test_default_selection_uses_default_mode_mapping_when_no_args(mode_yaml_files):
    """If nothing is provided, the class default mode decides the default yaml."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"],
        mode_yaml_files["eval"],
        default_mode="train",
        default_yaml_default=str(mode_yaml_files["eval"]),  # different on purpose
    )

    settings = ModeBasedSettings()
    # Must select train.yaml because default_mode maps to train
    assert settings.simple.name == "train"
    assert settings.option.name == "train_option"


def test_default_selection_falls_back_to_default_yaml_default(mode_yaml_files):
    """If no mode and no provided yaml_default, use the class default yaml_default."""
    ModeBasedSettings = create_mode_based_settings(
        mode_yaml_files["train"],
        mode_yaml_files["eval"],
        default_mode="",
        default_yaml_default=str(mode_yaml_files["eval"]),  # only yaml_default set
    )

    settings = ModeBasedSettings()
    assert settings.simple.name == "eval"
    assert settings.option.name == "eval_option"
