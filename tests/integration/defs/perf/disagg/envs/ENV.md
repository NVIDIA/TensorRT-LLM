# Environment Configuration Guide

Please put your own test env files here. Create a file like `env_<yourenv>.sh` in this directory.

## Required Environment Variables

The env file should be as follows:

```bash
export CONTAINER_IMAGE="<your_sqsh_image>"
export WORK_DIR="<your_work_directory>"
export SCRIPT_DIR="<your_scripts_dir>"
export REPO_DIR="<your_tensorrt_llm_repo_directory>"
export TRTLLM_WHEEL_PATH="<your_tensorrt_llm_wheel_path>"
export GPU_TYPE="<your_gpu_type>"
export SLURM_PARTITION="<your_slurm_cluster_partition>"
export SLURM_ACCOUNT="<your_slurm_cluster_account>"
export MODEL_DIR="<your_model_path>"
export DATASET_DIR="<your_dataset_path>"
export OUTPUT_PATH="<your_html_and_csv_output_path>"
export PATH="<please_add_poetry_binary_to_your_path>"
export XDG_CACHE_HOME="<your_xdg_cache_home>"
export PIP_CACHE_DIR="<your_pip_cache_dir>"
export INSTALL_MODE="<install_mode>"
export DEBUG_MODE=<debug_mode>
export DEBUG_JOB_ID=<your_debug_job_id>
```

## Variable Descriptions

### `CONTAINER_IMAGE`
Path to the Enroot/Pyxis container image (usually a `.sqsh` file) containing TensorRT-LLM and dependencies.
- **Format**: Absolute path to `.sqsh` file
- **Example**: `/path/to/container/trtllm-<version>-<arch>-<date>.sqsh`

### `WORK_DIR`
Working directory where test configurations and scripts are located. This should point to this disagg test directory.
- **Format**: Absolute path
- **Example**: `/path/to/tensorrt_llm/tests/integration/defs/perf/disagg`

### `SCRIPT_DIR`
Directory containing the benchmark execution scripts for disaggregated serving.
- **Format**: Absolute path
- **Example**: `/path/to/tensorrt_llm/examples/disaggregated/slurm/benchmark`

### `REPO_DIR`
Root directory of the TensorRT-LLM repository. Leave empty if not needed.
- **Format**: Absolute path or empty string
- **Example**: `/path/to/tensorrt_llm` or `""`
    When setting to empty string, means no need to build from source. 
    You should use none/wheel install_mode.

### `TRTLLM_WHEEL_PATH`
Path to TensorRT-LLM wheel file(s) for installation. Supports wildcards.
- **Format**: Absolute path with optional wildcards
- **Example**: `/path/to/build/*.whl`

### `GPU_TYPE`
GPU architecture type for test filtering. Tests will only run on matching hardware.
- **Format**: String (currently only `GB200` is supported)
- **Example**: `GB200`

### `SLURM_PARTITION`
SLURM partition name where jobs will be submitted.
- **Format**: String
- **Example**: `batch` or `interactive`

### `SLURM_ACCOUNT`
SLURM account name for job billing and resource allocation.
- **Format**: String
- **Example**: `your_project_account`

### `MODEL_DIR`
Base directory containing models. This path will be used to locate model checkpoints.
- **Format**: Absolute path
- **Example**: `/shared/models/common`

### `DATASET_DIR`
Base directory containing dataset files. This path will be used to locate dataset files.
- **Format**: Absolute path
- **Example**: `/shared/datasets/common`

### `OUTPUT_PATH`
Directory where test results, HTML reports, and CSV files will be saved.
- **Format**: Absolute path
- **Example**: `/path/to/output`

### `PATH`
System PATH with Poetry binary location. Poetry is required for dependency management.
- **Format**: PATH string with Poetry binary directory
- **Example**: `/path/to/home/.local/bin:$PATH`

### `XDG_CACHE_HOME`
Custom cache directory to avoid home directory quota issues.
- **Format**: Absolute path
- **Example**: `/path/to/user`

### `PIP_CACHE_DIR`
Pip cache directory for package downloads. Should be in a location with sufficient space.
- **Format**: Absolute path
- **Example**: `${XDG_CACHE_HOME}/pip`

### `INSTALL_MODE`
TensorRT-LLM installation mode. Controls whether to install from wheel or use existing installation.
- **Options**: `wheel`, `none`, or `source`
- **Example**: `none` (use existing installation)

### `DEBUG_MODE`
Enable debug mode to skip SLURM job submission for local testing.
- **Options**: `0` (disabled) or `1` (enabled)
- **Default**: `0`

### `DEBUG_JOB_ID`
Mock job ID to use when DEBUG_MODE is enabled. Only used for testing without SLURM.
- **Format**: Integer
- **Example**: `12345`

## Usage

1. Copy the example above to a new file, e.g., `env_myenv.sh`
2. Replace all path placeholders with your actual paths
3. Source the environment file before running tests:
   ```bash
   source envs/env_myenv.sh
   poetry run pytest --disagg test_disagg.py -s -vv
   ```

## Notes

- Make sure all directories exist before running tests
- Ensure you have appropriate permissions for SLURM_PARTITION and SLURM_ACCOUNT
- The OUTPUT_PATH directory should have sufficient space for test results
- XDG_CACHE_HOME should be on a filesystem with adequate quota
- Keep your env file private if it contains sensitive information
- Add your env file to `.gitignore` if needed
