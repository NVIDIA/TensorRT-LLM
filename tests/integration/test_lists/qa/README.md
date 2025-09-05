# Description

This folder contains QA test definitions for TensorRT-LLM, which are executed on a daily/release schedule. These tests focus on end-to-end validation, accuracy verification, disaggregated testing, and performance benchmarking.

## Test Categories

QA tests are organized into three main categories:

### 1. Functional Tests
Functional tests include E2E (end-to-end), accuracy, and disaggregated test cases:

- **E2E Tests**: Complete workflow validation from model loading to inference output
- **Accuracy Tests**: Model accuracy verification against reference implementations
- **Disaggregated Tests**: Distributed deployment and multi-node scenario validation

### 2. Performance Tests
Performance tests focus on benchmarking and performance validation:
- Baseline performance measurements
- Performance regression detection
- Throughput and latency benchmarking
- Resource utilization analysis

### 3. Triton Backend Tests
Triton backend tests validate the integration with NVIDIA Triton Inference Server:
- Backend functionality validation
- Model serving capabilities
- API compatibility testing
- Integration performance testing

## Dependencies

The following Python packages are required for running QA tests:

```bash
pip3 install -r ${TensorRT-LLM_PATH}/requirements-dev.txt
```

### Dependency Details

- **mako**: Template engine for test generation and configuration
- **oyaml**: YAML parser with ordered dictionary support
- **rouge_score**: ROUGE evaluation metrics for text generation quality assessment
- **lm_eval**: Language model evaluation framework

## Test Files

This directory contains various test configuration files:

### Functional Test Lists
- `llm_function_core.txt` - Primary test list for single node multi-GPU scenarios (all new test cases should be added here)
- `llm_function_core_sanity.txt` - Subset of examples for quick torch flow validation
- `llm_function_nim.txt` - NIM-specific functional test cases
- `llm_function_multinode.txt` - Multi-node functional test cases
- `llm_function_gb20x.txt` - GB20X release test cases
- `llm_function_rtx6k.txt` - RTX 6000 series specific tests
- `llm_function_l20.txt` - L20 specific tests, only contains single gpu cases

### Performance Test Files
- `llm_perf_full.yml` - Main performance test configuration
- `llm_perf_cluster.yml` - Cluster-based performance tests
- `llm_perf_sanity.yml` - Performance sanity checks
- `llm_perf_nim.yml` - NIM-specific performance tests
- `llm_trt_integration_perf.yml` - Integration performance tests
- `llm_trt_integration_perf_sanity.yml` - Integration performance sanity checks

### Triton Backend Tests
- `llm_triton_integration.txt` - Triton backend integration tests

### Release-Specific Tests
- `llm_digits_func.txt` - Functional tests for DIGITS release
- `llm_digits_perf.txt` - Performance tests for DIGITS release

## Test Execution Schedule

QA tests are executed on a regular schedule:

- **Weekly**: Automated regression testing
- **Release**: Comprehensive validation before each release
   - **Full Cycle Testing**:
        run all gpu with llm_function_core.txt + run NIM specific gpu with llm_function_nim.txt
    - **Sanity Cycle Testing**:
        run all gpu with llm_function_core_sanity.txt
    - **NIM Cycle Testing**:
        run all gpu with llm_function_core_sanity.txt + run NIM specific gpu with llm_function_nim.txt
- **On-demand**: Manual execution for specific validation needs

## Running Tests

### Manual Execution

To run specific test categories:

```bash
# direct to defs folder
cd tests/integration/defs
# Run all fp8 functional test
pytest --no-header -vs --test-list=../test_lists/qa/llm_function_full.txt -k fp8
# Run a single test case
pytest -vs accuracy/test_cli_flow.py::TestLlama3_1_8B::test_auto_dtype
```

### Automated Execution

QA tests are typically executed through CI/CD pipelines with appropriate test selection based on:

- Release requirements
- Hardware availability
- Test priority and scope

## Test Guidelines

### Adding New Test Cases
- **Primary Location**: For functional testing, new test cases should be added to `llm_function_full.txt` first
- **Categorization**: Test cases should be categorized based on their scope and execution time
- **Validation**: Ensure test cases are properly validated before adding to any test list
