import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root

# Constants for test configuration
_DEFAULT_NUM_REQUESTS = 3
_DEFAULT_INPUT_MEAN = 100
_DEFAULT_INPUT_STDEV = 10
_DEFAULT_OUTPUT_MEAN = 100
_DEFAULT_OUTPUT_STDEV = 10
_TEST_TASK_IDS = [0, 1, 2]
_TOKENIZER_SUBPATH = "llama-models-v2/tinyllama-tarot-v1/"
_PREPARE_DATASET_SCRIPT_PATH = "benchmarks/cpp/prepare_dataset.py"


class TestPrepareDatasetLora:
    """
    Test suite for prepare_dataset.py CLI tool LoRA metadata generation
    functionality.

    This test class validates that the prepare_dataset.py script correctly
    generates LoRA request metadata when LoRA-specific parameters are provided.
    It covers both fixed task ID and random task ID scenarios.
    """

    @pytest.fixture
    def temp_lora_dir(self) -> str:
        """
        Create a temporary LoRA directory structure for testing.

        Creates a temporary directory with subdirectories for each test task
        ID, simulating the expected LoRA adapter directory structure.

        Returns:
            str: Path to the temporary LoRA directory
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_dir = Path(temp_dir) / "loras"
            # Create dummy LoRA adapter directories for each test task ID
            for task_id in _TEST_TASK_IDS:
                task_dir = lora_dir / str(task_id)
                task_dir.mkdir(parents=True, exist_ok=True)
            yield str(lora_dir)

    def _build_base_command(self, llm_root: Path) -> List[str]:
        """
        Build the base command for running prepare_dataset.py.

        Args:
            llm_root: Path to the TensorRT LLM root directory

        Returns:
            List[str]: Base command components

        Raises:
            pytest.skip: If LLM_MODELS_ROOT is not available
        """
        script_path = llm_root / _PREPARE_DATASET_SCRIPT_PATH
        cmd = ["python3", str(script_path)]

        # Add required tokenizer argument
        model_cache = llm_models_root()
        if model_cache is None:
            pytest.skip("LLM_MODELS_ROOT not available")

        tokenizer_dir = model_cache / _TOKENIZER_SUBPATH
        cmd.extend(["--tokenizer", str(tokenizer_dir)])

        # Always add --stdout flag since we parse stdout output
        cmd.extend(["--stdout"])

        return cmd

    def _add_lora_arguments(self, cmd: List[str], **kwargs) -> None:
        """
        Add LoRA-specific arguments to the command.

        Args:
            cmd: Command list to modify in-place
            **kwargs: Keyword arguments containing LoRA parameters
        """
        if "lora_dir" in kwargs:
            cmd.extend(["--lora-dir", kwargs["lora_dir"]])
        if "task_id" in kwargs:
            cmd.extend(["--task-id", str(kwargs["task_id"])])
        if "rand_task_id" in kwargs:
            min_id, max_id = kwargs["rand_task_id"]
            cmd.extend(["--rand-task-id", str(min_id), str(max_id)])

    def _add_synthetic_data_arguments(self, cmd: List[str]) -> None:
        """
        Add synthetic data generation arguments to the command.

        Args:
            cmd: Command list to modify in-place
        """
        cmd.extend([
            "token-norm-dist", "--num-requests",
            str(_DEFAULT_NUM_REQUESTS), "--input-mean",
            str(_DEFAULT_INPUT_MEAN), "--input-stdev",
            str(_DEFAULT_INPUT_STDEV), "--output-mean",
            str(_DEFAULT_OUTPUT_MEAN), "--output-stdev",
            str(_DEFAULT_OUTPUT_STDEV)
        ])

    def _run_prepare_dataset(self, llm_root: Path, **kwargs) -> str:
        """
        Execute prepare_dataset.py with specified parameters and capture
        output.

        Args:
            llm_root: Path to the TensorRT LLM root directory
            **kwargs: Keyword arguments for LoRA configuration

        Returns:
            str: Standard output from the executed command

        Raises:
            subprocess.CalledProcessError: If the command execution fails
        """
        cmd = self._build_base_command(llm_root)
        self._add_lora_arguments(cmd, **kwargs)
        self._add_synthetic_data_arguments(cmd)

        # Execute command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout

    def _parse_json_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse JSON lines from prepare_dataset.py output.

        Args:
            output: Raw stdout output containing JSON lines

        Returns:
            List[Dict[str, Any]]: Parsed JSON data objects
        """
        lines = output.strip().split('\n')
        json_data = []

        for line in lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    json_data.append(data)
                except json.JSONDecodeError:
                    # Skip non-JSON lines (such as debug prints)
                    continue

        return json_data

    def _validate_lora_request(self,
                               lora_request: Dict[str, Any],
                               expected_lora_dir: str,
                               task_id_range: Tuple[int, int] = None) -> None:
        """Validate LoRA request structure and content."""
        # Check required fields
        required_fields = ["lora_name", "lora_int_id", "lora_path"]
        for field in required_fields:
            assert field in lora_request, f"Missing '{field}' in LoRA request"

        task_id = lora_request["lora_int_id"]

        # Validate task ID range if specified
        if task_id_range:
            min_id, max_id = task_id_range
            assert min_id <= task_id <= max_id, (
                f"Task ID {task_id} out of range [{min_id}, {max_id}]")

        # Validate structure
        expected_name = f"lora_{task_id}"
        expected_path = os.path.join(expected_lora_dir, str(task_id))

        assert lora_request["lora_name"] == expected_name, (
            f"Expected LoRA name '{expected_name}', "
            f"got '{lora_request['lora_name']}'")
        assert lora_request["lora_path"] == expected_path, (
            f"Expected LoRA path '{expected_path}', "
            f"got '{lora_request['lora_path']}'")

    @pytest.mark.parametrize("test_params", [
        pytest.param({
            "task_id": 1,
            "description": "fixed task ID"
        },
                     id="fixed_task_id"),
        pytest.param(
            {
                "rand_task_id": (0, 2),
                "description": "random task ID range"
            },
            id="random_task_id")
    ])
    def test_lora_metadata_generation(self, llm_root: Path, temp_lora_dir: str,
                                      test_params: Dict) -> None:
        """Test LoRA metadata generation with various configurations."""
        # Extract test parameters
        task_id = test_params.get("task_id")
        rand_task_id = test_params.get("rand_task_id")
        description = test_params["description"]

        # Run prepare_dataset
        kwargs = {"lora_dir": temp_lora_dir}
        if task_id is not None:
            kwargs["task_id"] = task_id
        if rand_task_id is not None:
            kwargs["rand_task_id"] = rand_task_id

        output = self._run_prepare_dataset(llm_root, **kwargs)
        json_data = self._parse_json_output(output)

        assert len(json_data) > 0, f"No JSON data generated for {description}"

        # Validate LoRA requests
        for i, item in enumerate(json_data):
            assert "lora_request" in item, (
                f"Missing 'lora_request' in JSON entry {i} for {description}")
            self._validate_lora_request(item["lora_request"],
                                        temp_lora_dir,
                                        task_id_range=rand_task_id)
