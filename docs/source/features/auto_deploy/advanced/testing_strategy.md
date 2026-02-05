# Testing Strategy

This document describes the testing strategy for AutoDeploy, covering the multi-tiered approach used to ensure quality and reliability.

## Testing Philosophy

AutoDeploy uses a multi-tiered testing approach that balances fast feedback with comprehensive coverage:

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard                            │
│         (Broad model coverage + performance)            │
├─────────────────────────────────────────────────────────┤
│                 Integration Tests                       │
│            (Accuracy tests, CI-registered)              │
├─────────────────────────────────────────────────────────┤
│                  E2E Mini Tests                         │
│           (Compile + prompt workflows)                  │
├─────────────────────────────────────────────────────────┤
│                    Unit Tests                           │
│      (Component testing: patches, transforms, etc.)     │
└─────────────────────────────────────────────────────────┘
```

- **Unit Tests**: Fast, isolated tests for individual components (patches, transforms, custom ops)
- **E2E Mini Tests**: End-to-end workflows testing compile + prompt for unique model combinations
- **Integration Tests**: Important accuracy tests registered individually in CI
- **Dashboard**: Broad model coverage and performance testing across all supported models

## Unit Tests

Unit tests verify individual components like patches, transformations, custom operations, and utilities.

### Location

All unit tests are located in `tests/unittest/auto_deploy/`:

```
tests/unittest/auto_deploy/
├── _utils_test/                    # Shared test utilities
├── singlegpu/                      # Single GPU tests
│   ├── compile/                    # Compilation tests
│   ├── custom_ops/                 # Custom operations tests
│   ├── models/                     # Model-specific patch tests
│   ├── shim/                       # Executor/engine tests
│   ├── smoke/                      # E2E mini tests (see below)
│   ├── transformations/            # Graph transformation tests
│   └── utils/                      # Utility function tests
└── multigpu/                       # Multi-GPU tests
    ├── custom_ops/                 # Multi-GPU custom ops
    ├── smoke/                      # Multi-GPU E2E mini tests
    └── transformations/            # Multi-GPU transformation tests
```

### CI Registration

Tests are automatically run in CI once registered. New test files and functions are picked up automatically **if they are in an existing registered folder**.

Tests are registered in `tests/integration/test_lists/test-db/l0_*.yml` files under the `backend: autodeploy` section:

```yaml
backend: autodeploy
tests:
- unittest/auto_deploy/singlegpu/compile
- unittest/auto_deploy/singlegpu/custom_ops
- unittest/auto_deploy/singlegpu/models
- unittest/auto_deploy/singlegpu/shim
- unittest/auto_deploy/singlegpu/smoke
- unittest/auto_deploy/singlegpu/transformations
- unittest/auto_deploy/singlegpu/utils
```

#### Adding a New Folder

If you create a **new folder** (not just a new file in an existing folder), you must register it in the appropriate YAML files:

1. Edit `tests/integration/test_lists/test-db/l0_a30.yml` (and other GPU-specific files as needed)
1. Add the new folder path under the `backend: autodeploy` section
1. Example: `- unittest/auto_deploy/singlegpu/my_new_folder`

### Parallel Execution

Most unit tests run in parallel using pytest-xdist for faster execution. The exception is the `smoke/` subfolders, which run sequentially (see E2E Mini Tests below).

## E2E Mini Tests (Smoke Tests)

E2E mini tests verify complete end-to-end workflows including model compilation and prompt execution for unique model combinations.

### Location

- **Single GPU**: `tests/unittest/auto_deploy/singlegpu/smoke/`
- **Multi GPU**: `tests/unittest/auto_deploy/multigpu/smoke/`

### Purpose

These tests ensure that the full AutoDeploy pipeline works correctly for various model architectures and configurations:

- `test_ad_build_small_single.py` - Tests multiple model configurations (Llama, Mixtral, Qwen, Phi-3, DeepSeek, Mistral, Nemotron)
- `test_ad_trtllm_bench.py` - Benchmarking functionality
- `test_ad_trtllm_serve.py` - Serving functionality
- `test_ad_speculative_decoding.py` - Speculative decoding
- `test_ad_export_onnx.py` - ONNX export functionality

### Execution

Smoke tests are **not executed in parallel** to avoid resource contention during full model compilation and execution. They run sequentially within the CI pipeline.

## Integration Tests

Integration tests cover important accuracy tests and other scenarios that require explicit CI registration.

### Registration

Unlike unit tests (where new files in existing folders are auto-discovered), **each individual integration test case must be explicitly registered** in the CI YAML files.

Format: `path/to/test_file.py::test_function_name[param_id]`

Example from `l0_a30.yml`:

```yaml
- accuracy/test_cli_flow.py::TestLlama3_1_8BInstruct::test_medusa_fp8_prequantized
- examples/test_multimodal.py::test_llm_multimodal_general[Qwen2-VL-7B-Instruct-pp:1-tp:1-float16-bs:1-cpp_e2e:False-nb:4]
```

### Example: Adding an Accuracy Test

For reference, see [PR #10717](https://github.com/NVIDIA/TensorRT-LLM/pull/10717) which added a Nemotron 3 super accuracy test. The workflow is:

1. Create the test function in the appropriate test file
1. Register the specific test case in the relevant `l0_*.yml` file(s)
1. Ensure the test passes locally before submitting

### Location

Integration tests are typically located in:

- `examples/` - Model-specific integration tests
- `accuracy/` - Accuracy validation tests

## Dashboard (Model Coverage Testing)

The dashboard provides broad model coverage and performance testing for all supported models in AutoDeploy.

### Model Registry

Models are registered in `examples/auto_deploy/model_registry/models.yaml`. For detailed instructions, see the [Model Registry README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/auto_deploy/model_registry).

### Format (Version 2.0)

The registry uses a flat list format with composable configurations:

```yaml
version: '2.0'
description: AutoDeploy Model Registry - Flat format with composable configs
models:
- name: meta-llama/Llama-3.1-8B-Instruct
  yaml_extra: [dashboard_default.yaml, world_size_2.yaml]

- name: meta-llama/Llama-3.3-70B-Instruct
  yaml_extra: [dashboard_default.yaml, world_size_4.yaml, llama3_3_70b.yaml]
```

### Key Concepts

- **Flat list**: Models are in a single list (not grouped)
- **Composable configs**: Each model references YAML config files via `yaml_extra`
- **Deep merging**: Config files are merged in order (later files override earlier ones)

### Configuration Files

Config files are stored in `examples/auto_deploy/model_registry/configs/`:

| File | Purpose |
|------|---------|
| `dashboard_default.yaml` | Baseline settings for all models |
| `world_size_N.yaml` | GPU count configuration (1, 2, 4, or 8) |
| `multimodal.yaml` | Vision + text models |
| `demollm_triton.yaml` | DemoLLM runtime with Triton backend |
| Model-specific configs | Custom settings for specific models |

### World Size Guidelines

| World Size | Model Size Range | Example Models |
|------------|------------------|----------------|
| 1 | \< 2B params | TinyLlama, Qwen 0.5B, Phi-4-mini |
| 2 | 2-15B params | Llama 3.1 8B, Qwen 7B, Mistral 7B |
| 4 | 20-80B params | Llama 3.3 70B, QwQ 32B, Gemma 27B |
| 8 | 80B+ params | DeepSeek V3, Llama 405B, Nemotron Ultra |

### Adding a New Model

1. Add the model entry to `models.yaml`:

```yaml
- name: organization/my-new-model-7b
  yaml_extra: [dashboard_default.yaml, world_size_2.yaml]
```

2. For models with special requirements, create a custom config in `configs/` and reference it:

```yaml
- name: organization/my-custom-model
  yaml_extra: [dashboard_default.yaml, world_size_4.yaml, my_model.yaml]
```

3. Validate with `prepare_model_coverage_v2.py` from the autodeploy-dashboard repository

The model will be automatically picked up by the dashboard testing infrastructure on the next run.
