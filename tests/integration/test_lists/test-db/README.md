# Description

This folder contains test definition which is consumed by `trt-test-db` tool based on system specifications.

## Installation

Install `trt-test-db` using the following command:

```bash
pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple --ignore-installed trt-test-db==1.8.5+bc6df7
```

## Test Definition

Test definitions are stored in YAML files located in `${TRT_LLM_ROOT}/tests/integration/test_lists/test-db/`. These files define test conditions and the tests to be executed.

### Example YAML Structure

```yaml
version: 0.0.1
l0_e2e:
  - condition:
      terms:
        supports_fp8: true
      ranges:
        system_gpu_count:
          gte: 4
          lte: 4
      wildcards:
        gpu:
          - '*h100*'
        linux_distribution_name: ubuntu*
    tests:
      - examples/test_llama.py::test_llm_llama_v3_1_1node_multi_gpus[llama-3.1-8b-enable_fp8]
      - examples/test_llama.py::test_llm_llama_v3_1_1node_multi_gpus[llama-3.1-70b-enable_fp8]
```

## Generating Test Lists

Use `trt-test-db` to generate a test list based on the system configuration:

```bash
trt-test-db -d /TensorRT-LLM/src/tests/integration/test_lists/test-db \
            --context l0_e2e \
            --test-names \
            --output /TensorRT-LLM/src/l0_e2e.txt \
            --match-exact '{"chip":"ga102gl-a","compute_capability":"8.6","cpu":"x86_64","gpu":"A10","gpu_memory":"23028.0","host_mem_available_mib":"989937","host_mem_total_mib":"1031949","is_aarch64":false,"is_linux":true,"linux_distribution_name":"ubuntu","linux_version":"22.04","supports_fp8":false,"supports_int8":true,"supports_tf32":true,"sysname":"Linux","system_gpu_count":"1",...}'
```
This command generates a test list file (`l0_e2e.txt`) based on the specified context and system configuration.

## Running Tests

Execute the tests using `pytest` with the generated test list:

```bash
pytest -v --test-list=/TensorRT-LLM/src/l0_e2e.txt --output-dir=/tmp/logs
```

This command runs the tests specified in the test list and outputs the results to the specified directory.

## Additional Information
- The `--context` parameter in the `trt-test-db` command specifies which context to search in the YAML files.
- The `--match-exact` parameter provides system information used to filter tests based on the conditions defined in the YAML files.
- Modify the YAML files to add or update test conditions and test cases as needed.
For more detailed information on `trt-test-db` and `pytest` usage, refer to their respective documentation.
