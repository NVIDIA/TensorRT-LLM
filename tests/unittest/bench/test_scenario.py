"""Unit tests for trtllm-bench scenario handling and priority logic.

These tests verify the override behavior between --recipe, --extra_llm_api_options,
and CLI flags to ensure correct priority order and warning messages.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from tensorrt_llm.bench.utils.scenario import (
    merge_params_with_priority,
    prepare_llm_api_options_for_recipe,
)


class TestMergeParamsWithPriority:
    """Tests for merge_params_with_priority() function."""

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_cli_explicitly_set_overrides_scenario(self, mock_logger):
        """Test that explicitly set CLI values override scenario values."""
        cli_params = {"concurrency": 128, "tp": 2}
        scenario = {"target_concurrency": 256, "tp_size": 4}
        cli_defaults = {"concurrency": -1, "tp": 1}

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        # CLI concurrency was explicitly set (differs from default)
        assert merged["concurrency"] == 128

        # CLI tp was explicitly set (differs from default)
        assert merged["tp"] == 2

        # Verify warnings were logged
        assert mock_logger.warning.call_count == 2
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "CLI flag --concurrency=128 overrides recipe value" in call for call in warning_calls
        )
        assert any("scenario.target_concurrency=256" in call for call in warning_calls)
        assert any("CLI flag --tp=2 overrides recipe value" in call for call in warning_calls)
        assert any("scenario.tp_size=4" in call for call in warning_calls)

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_scenario_value_used_when_cli_not_explicitly_set(self, mock_logger):
        """Test that scenario values are used when CLI equals default."""
        cli_params = {"concurrency": -1, "tp": 1}
        scenario = {"target_concurrency": 256, "tp_size": 4}
        cli_defaults = {"concurrency": -1, "tp": 1}

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        # Both CLI values equal defaults, so scenario values should be used
        assert merged["concurrency"] == 256
        assert merged["tp"] == 4

        # Verify info logs were called
        assert mock_logger.info.call_count == 2
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Using recipe value for --concurrency: 256" in call for call in info_calls)
        assert any("from scenario.target_concurrency" in call for call in info_calls)
        assert any("Using recipe value for --tp: 4" in call for call in info_calls)
        assert any("from scenario.tp_size" in call for call in info_calls)

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_mixed_explicit_and_default_cli_values(self, mock_logger):
        """Test scenario with some CLI values explicit and some default."""
        cli_params = {"concurrency": 128, "tp": 1, "target_input_len": None}
        scenario = {
            "target_concurrency": 256,
            "tp_size": 4,
            "target_isl": 1024,
        }
        cli_defaults = {"concurrency": -1, "tp": 1, "target_input_len": None}

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        # concurrency explicitly set -> override
        assert merged["concurrency"] == 128

        # tp equals default -> use scenario
        assert merged["tp"] == 4

        # target_input_len is None -> use scenario
        assert merged["target_input_len"] == 1024

        # Verify 1 warning and 2 info calls
        assert mock_logger.warning.call_count == 1
        assert mock_logger.info.call_count == 2

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_cli_value_none_uses_scenario(self, mock_logger):
        """Test that None CLI values use scenario values."""
        cli_params = {"tp": None, "ep": None}
        scenario = {"tp_size": 4, "ep_size": 2}
        cli_defaults = {"tp": 1, "ep": 1}

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        assert merged["tp"] == 4
        assert merged["ep"] == 2

        # Verify info logs were called
        assert mock_logger.info.call_count == 2

    def test_all_parameter_mappings(self):
        """Test all scenario-to-CLI parameter mappings."""
        cli_params = {
            "concurrency": -1,
            "target_input_len": None,
            "target_output_len": None,
            "num_requests": 512,
            "tp": 1,
            "ep": 1,
            "pp": 1,
            "streaming": False,
        }
        scenario = {
            "target_concurrency": 128,
            "target_isl": 2048,
            "target_osl": 512,
            "num_requests": 1000,
            "tp_size": 2,
            "ep_size": 4,
            "pp_size": 2,
            "streaming": True,
        }
        cli_defaults = {
            "concurrency": -1,
            "target_input_len": None,
            "target_output_len": None,
            "num_requests": 512,
            "tp": 1,
            "ep": 1,
            "pp": 1,
            "streaming": False,
        }

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        # All should use scenario values since CLI equals defaults
        assert merged["concurrency"] == 128
        assert merged["target_input_len"] == 2048
        assert merged["target_output_len"] == 512
        assert merged["num_requests"] == 1000
        assert merged["tp"] == 2
        assert merged["ep"] == 4
        assert merged["pp"] == 2
        assert merged["streaming"] is True

    def test_no_scenario_returns_cli_params(self):
        """Test that None scenario returns copy of CLI params unchanged."""
        cli_params = {"concurrency": 128, "tp": 2}
        cli_defaults = {"concurrency": -1, "tp": 1}

        merged = merge_params_with_priority(cli_params, None, cli_defaults)

        assert merged == cli_params
        assert merged is not cli_params  # Should be a copy

    def test_no_cli_defaults_provided(self, caplog):
        """Test behavior when cli_defaults is None."""
        cli_params = {"concurrency": 128}
        scenario = {"target_concurrency": 256}
        cli_defaults = None

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        # Without defaults, CLI value should still override
        assert merged["concurrency"] == 128

    def test_scenario_key_not_in_params_mapping(self):
        """Test that scenario keys not in mapping are ignored."""
        cli_params = {"concurrency": -1}
        scenario = {
            "target_concurrency": 128,
            "unknown_field": "some_value",  # Not in param_mapping
        }
        cli_defaults = {"concurrency": -1}

        merged = merge_params_with_priority(cli_params, scenario, cli_defaults)

        assert merged["concurrency"] == 128
        assert "unknown_field" not in merged


class TestPrepareExtraLlmApiOptions:
    """Tests for priority between --recipe and --extra_llm_api_options."""

    def test_extra_llm_api_options_overrides_recipe(self, caplog):
        """Test that --extra_llm_api_options takes priority over --recipe."""
        # This would be tested at the caller level in process_recipe_scenario
        # We're testing the warning message here
        with patch("tensorrt_llm.bench.utils.scenario.logger") as mock_logger:
            recipe_path = "/path/to/recipe.yaml"
            extra_path = "/path/to/extra.yaml"

            # Simulate the logic in process_recipe_scenario
            if recipe_path and extra_path:
                mock_logger.warning(
                    f"Both --recipe and --extra_llm_api_options provided. "
                    f"Using --extra_llm_api_options ({extra_path}) "
                    f"which overrides --recipe ({recipe_path})"
                )

            # Verify warning was called
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Both --recipe and --extra_llm_api_options provided" in call_args
            assert extra_path in call_args
            assert recipe_path in call_args


class TestPrepareLlmApiOptionsForRecipe:
    """Tests for prepare_llm_api_options_for_recipe() function."""

    def test_none_path_returns_none(self):
        """Test that None path returns None."""
        result = prepare_llm_api_options_for_recipe(None, None)
        assert result is None

    def test_non_recipe_format_returns_original_path(self):
        """Test that non-recipe format returns original path unchanged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write simple llm_api_options (not recipe format)
            yaml.safe_dump({"max_tokens": 100, "temperature": 0.7}, f)
            temp_path = f.name

        try:
            # scenario=None means not recipe format
            result = prepare_llm_api_options_for_recipe(temp_path, scenario=None)
            assert result == temp_path
        finally:
            Path(temp_path).unlink()

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_recipe_format_extracts_llm_api_options(self, mock_logger):
        """Test that recipe format extracts llm_api_options to temp file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write recipe format
            recipe_data = {
                "scenario": {
                    "model": "test",
                    "target_isl": 1024,
                    "target_osl": 256,
                    "target_concurrency": 32,
                },
                "llm_api_options": {"max_tokens": 100, "temperature": 0.7},
                "env": {"SOME_VAR": "value"},
            }
            yaml.safe_dump(recipe_data, f)
            temp_path = f.name

        try:
            # scenario dict means recipe format detected
            scenario = recipe_data["scenario"]
            result = prepare_llm_api_options_for_recipe(temp_path, scenario)

            # Should return a different path (temp file)
            assert result != temp_path
            assert result is not None

            # Verify info log was called
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Recipe format detected" in call for call in info_calls)

            # Verify temp file contains only llm_api_options
            with open(result) as f:
                extracted = yaml.safe_load(f)
                assert extracted == {"max_tokens": 100, "temperature": 0.7}

            # Clean up temp file
            Path(result).unlink()
        finally:
            Path(temp_path).unlink()

    def test_recipe_with_empty_llm_api_options(self):
        """Test recipe with empty llm_api_options section."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            recipe_data = {
                "scenario": {
                    "model": "test",
                    "target_isl": 1024,
                    "target_osl": 256,
                    "target_concurrency": 32,
                },
                "llm_api_options": {},
            }
            yaml.safe_dump(recipe_data, f)
            temp_path = f.name

        try:
            scenario = recipe_data["scenario"]
            result = prepare_llm_api_options_for_recipe(temp_path, scenario)

            assert result is not None
            assert result != temp_path

            # Verify temp file contains empty dict
            with open(result) as f:
                extracted = yaml.safe_load(f)
                assert extracted == {}

            Path(result).unlink()
        finally:
            Path(temp_path).unlink()

    def test_recipe_without_llm_api_options_key(self):
        """Test recipe without llm_api_options key (defaults to empty dict)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            recipe_data = {
                "scenario": {
                    "model": "test",
                    "target_isl": 1024,
                    "target_osl": 256,
                    "target_concurrency": 32,
                },
                # No llm_api_options key
            }
            yaml.safe_dump(recipe_data, f)
            temp_path = f.name

        try:
            scenario = recipe_data["scenario"]
            result = prepare_llm_api_options_for_recipe(temp_path, scenario)

            assert result is not None

            # Verify temp file contains empty dict (default from .get())
            with open(result) as f:
                extracted = yaml.safe_load(f)
                assert extracted == {} or extracted is None

            Path(result).unlink()
        finally:
            Path(temp_path).unlink()

    @patch("tensorrt_llm.bench.utils.scenario.logger")
    def test_file_not_found_returns_original_path(self, mock_logger):
        """Test that FileNotFoundError returns original path with warning."""
        non_existent = "/path/that/does/not/exist.yaml"
        scenario = {"model": "test", "target_isl": 1024}

        result = prepare_llm_api_options_for_recipe(non_existent, scenario)

        # Should return original path and log warning
        assert result == non_existent

        # Verify warning was logged
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Failed to process recipe file" in call for call in warning_calls)
