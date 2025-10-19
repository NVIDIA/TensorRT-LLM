# How to run TRT-LLM tests

## 1. Unit test (Python)

All the tests contained in the `unittest` directory folder are considered as "unit test" in this doc, these tests can use the python standard [unittests](https://docs.python.org/3/library/unittest.html) and [pytest](https://docs.pytest.org/en/stable/). Since pytest are compatible with the unittest framework, we use pytest to launch these in the CI.

Unit test should be small, fast, and test only for specific function.

If you need to run them locally, the only dependencies are `requirements-dev.txt`.

```bash
# in TensorRT LLM source repo root dir
# use editable install, such that your local changes will be used immedietely in the tests w/o another install
# see https://setuptools.pypa.io/en/latest/userguide/development_mode.html
pip install -e ./

# the pytest and required plugins used are listed in the requirements-dev.txt
pip install -r requirements-dev.txt

cd tests/
## There are multiple ways to tell pytest to launch a subset of the targeted test cases

# example 1: runs all the tests under this directory, ignores the integration. WARNING: this can takes a very long time
pytest ./

# example 2: run a single test file
pytest ./test_builder.py

# example 3: run a test in a subfolder
pytest ./functional

# example 4: run a test with a substr
pytest -k test_basic_builder_flow
```

## 2. Integration test (Python)

All the integration tests are launched by pytest. The integration tests are currently all located [tests/integration/defs](./integration/defs/).

You can read the pytest official doc for details, https://docs.pytest.org/en/stable/

### Prepare model files (Non-NVIDIA developers)

Many integration tests rely on real model data. To correctly run the integration test, you must place all needed models in a directory and set environment variable `LLM_MODELS_ROOT` to it.

The subdirectory hierarchy of each model can be found in the codebase. For example, `bert_example_root` in `integration/defs/conftest.py`.

Examples to run integration test locally.

```bash
export LLM_MODELS_ROOT=/path-to-models

# in root dir
pip install -r requirements-dev.txt
cd tests/integration/defs

# example 1: run a case
pytest "accuracy/test_llm_api_pytorch.py::TestLlama3_1_8B::test_auto_dtype"

# example 2: run a test list
pytest --rootdir . --test-list=<a txt file contains on test case per line>

# example 3: list all the cases.
pytest --co -q

# example 4: run all the tests which contains this sub string
pytest -k test_llm_gpt2_medium_bad_words_1gpu

# example 5: run all tests which match this regexp
pytest -R ".*test_llm_gpt2_medium_bad_words_1gpu.*non.*py.*"

# example 6: list all the cases contains a sub string
pytest -k llmapi --co -q
```

You can set the output directory for logs/runtime data using the --output-dir flag.
For more options, refer to pytest --help, paying attention to Custom options added for TRT-LLM.

### Common issues:

1. `trtllm-build: not found`

    Many of the test cases use `trtllm-build` command to build engines.
    If you meet the error of `trtllm-build: not found`, you should add the `trtllm-build` path into your `PATH` env before launchig pytest. Normally if you install trtllm in the `$HOME/.local` or use `pip install -e ./` to install trtllm in-place, the trtllm-build command should be located in `$HOME/.local/bin`.

    Thus you should do `export PATH=$HOME/.local/bin:$PATH` before running the pytest

2. The `LLM_MODELS_ROOT` is not set correctly

    ```bash
        AssertionError: ...llm-models/gpt2-medium does not exist, and fail_if_path_is_invalid is True, please check the cache directory
        assert False

      conftest.py:149: AssertionError
    ```
    If you see above failures when running pytest locally, its likely that you didn't set the `LLM_MODELS_ROOT` env correctly. The default value is a NVIDIA internal path that is used in CI environment.

    When you finish setup the model directory, remember to mount it in the docker container.


## 3. C++ runtime test

TRT-LLM C++ runtime tests are using [google-test](https://github.com/google/googletest) framework, and Pytest is used to run sets of these tests.

The C++ runtime relies on TRT-LLM python frontend to generate engines as test data, so there are scripts to generate the engines in the C++ test [resources directory](../cpp/tests/resources/).
Pytest calls these scripts from fixtures prior to launching the test cases.

Details on usage of the resources scripts can be found in the [C++ Test document](../cpp/tests/README.md).

## 4. Performance regression test

For performance regression testing in QA and CI, see the [performance test guide](./integration/README.md).

# How to add test to CI

## 1. How does the CI work

Due to CI hardware resource limitation, and some cases only run on specific GPUs, the test cases are managed based on GPU type.

In directory `integration/test_lists/test-db`, each yml file corresponds to a GPU type.

In file `jenkins/L0_Test.groovy`, the variables `x86TestConfigs`, `SBSATestConfigs`, `x86SlurmTestConfigs` and `SBSASlurmTestConfigs` map yml files to CI stages according to platforms and launch methods.

Currently the yml files are manually maintained, which requires developer to update them when new test cases are added.

### How to choose GPU type

The CI resource of each GPU type is different. Usually you should choose the cheapest GPU that fulfills test requirements. In most cases, an integration test case should only run on one GPU type, unless it's very important or has different behaviours on different GPUs.

The priority is A10 > A30 > L40s > A100 > H100 > B200.

## 2. Add an integration test

Integrations tests usually run entire workflow, containing checkpoint converting, engine building and evaluating, to check functional and accuracy.

Integration tests are stored in [`integration/defs`](./integration/defs). In particular, please see [`integration/defs/accuracy`](./integration/defs/accuracy) for more detailed guidance to add accuracy tests.

Once a new integration test case is added, the yml files must be updated to contain the newly added case. Otherwise, the CI will not be able to collect and run this case.

## 3. Add a unit test

A unit test are used to test a standalone feature or building block, and only runs partial workflow.

For legacy and case management reason, the CI doesn't run unit tests directly. It uses a bridge to map multiple unit test cases into one integration test case, and manages these bridged cases.
The bridge is implemented in `integration/defs/test_unittests.py` and `pytest_generate_tests` function in `tests/integration/defs/conftest.py`.

In `integration/test_lists/test-db`, cases with prefix `unittest/` are treated as unit test bridges. Each of them generates an instance of `test_unittests_v2` which executes a `pytest` subprocess in `tests/unittest` directory.
The entire line will be passed as commandline arguments of `pytest` subprocess.

For example, `unittest/trt/attention/test_gpt_attention.py -k "partition0"` is equivalent to `cd tests; pytest unittest/trt/attention/test_gpt_attention.py -k "partition0"`.

New unit tests can be added to CI as follows:

1. Determine the commandline to run desired cases. In working directory `tests`, the command usually looks like one of them:

```bash
pytest unittest/_torch/my_new_folder # run all cases in a directory
pytest unittest/_torch/my_new_file.py # run all cases in a file
pytest unittest/an_existing_file.py -k "some_keyword or another_keyword" # run some cases in a file, filtered by keywords
pytest unittest/an_existing_file.py -m "part0 and gpu2" # run some cases in a file, filtered by pytest mark
```

2. Check existing bridge cases and make sure your cases are not covered by an existing one.
For example, you may want to add `pytest unittest/an_existing_file.py -k "some_keyword or another_keyword"`, but there is already `pytest unittest/an_existing_file.py -k "not thrid_keyword"` which covers your filter.

3. Choose a suitable GPU and add a line of your cases. For example, adding `unittest/an_existing_file.py -k "some_keyword or another_keyword"` to `tests/integration/test_lists/test-db/l0_a10.yml`.

## 4. Run a CI stage locally

Each yml file in `integration/test_lists/test-db` corresponds to a CI stage. You can run a stage locally, e.g. `l0_a10.yml`, as follows.

1. Open `l0_a10.yml`, it should look like:

```yaml
version: 0.0.1
l0_a10:
- condition:
    ranges:
      system_gpu_count:
        gte: 1
        lte: 1
    wildcards:
      gpu:
      - '*a10*'
      linux_distribution_name: ubuntu*
  tests:
  # ------------- PyTorch tests ---------------
  - disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0]
  - disaggregated/test_disaggregated.py::test_disaggregated_cuda_graph[TinyLlama-1.1B-Chat-v1.0]
  - disaggregated/test_disaggregated.py::test_disaggregated_mixed[TinyLlama-1.1B-Chat-v1.0]
  - disaggregated/test_disaggregated.py::test_disaggregated_overlap[TinyLlama-1.1B-Chat-v1.0]
  # ------------- CPP tests ---------------
  - cpp/test_e2e.py::test_model[medusa-86]
  - cpp/test_e2e.py::test_model[redrafter-86]
  - cpp/test_e2e.py::test_model[mamba-86]
  - cpp/test_e2e.py::test_model[recurrentgemma-86]
  - cpp/test_e2e.py::test_model[eagle-86]
```

2. Copy all items in `tests` field to a text file, for example, `a10_list.txt`. Don't forget to remove extra characters like comments and the dash marks.

```
disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0]
disaggregated/test_disaggregated.py::test_disaggregated_cuda_graph[TinyLlama-1.1B-Chat-v1.0]
disaggregated/test_disaggregated.py::test_disaggregated_mixed[TinyLlama-1.1B-Chat-v1.0]
disaggregated/test_disaggregated.py::test_disaggregated_overlap[TinyLlama-1.1B-Chat-v1.0]
cpp/test_e2e.py::test_model[medusa-86]
cpp/test_e2e.py::test_model[redrafter-86]
cpp/test_e2e.py::test_model[mamba-86]
cpp/test_e2e.py::test_model[recurrentgemma-86]
cpp/test_e2e.py::test_model[eagle-86]
```

3. Invoke `pytest` with TRT-LLM custom option `--test-list`:

```shell
cd tests/integration/defs
pytest . --test-list="a10_list.txt" --output-dir=/tmp/llm_integration_test
```

## 5. Set timeout for some long cases individually
To set a timeout for specific long-running test cases, follow these steps:

### For CI (test-db YAML files):
1. Locate the test case line in the corresponding test-db YAML file (e.g., `tests/integration/test_lists/test-db/l0_a10.yml`).
2. Append `TIMEOUT (...)` to the test case line, as shown below:
   ```yaml
   - disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] TIMEOUT (30)
   ```
   - Ensure there is **at least one space** before and after the `TIMEOUT` keyword.
   - The time value inside the parentheses `()` must be a **number** representing the timeout in **minutes**.

### For Local Testing (TXT files):
1. If you are running the tests locally using a prepared `.txt` file (e.g., `a10_list.txt`), append the `TIMEOUT` setting to the test case line in the same way:
   ```
   disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] TIMEOUT (30)
   ```

## 6. Set isolated execution for cases individually

Some test cases may experience intermittent failures due to resource conflicts, memory leaks, or state pollution when run together with other tests. The `ISOLATION` marker ensures these cases run in a separate pytest process, avoiding such issues.

### When to use the `ISOLATION` marker:
- Tests that modify global state or environment variables
- Tests with memory-intensive operations that may affect subsequent tests
- Tests that experience intermittent failures only when run with other tests
- Tests that require exclusive access to certain resources (GPU memory, files, etc.)

### Usage:
Add `ISOLATION` to the test case line with proper spacing:

**For CI (test-db YAML files):**
```yaml
- disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] ISOLATION
```

**For Local Testing (TXT files):**
```
disaggregated/test_disaggregated.py::test_disaggregated_single_gpu_with_mpirun[TinyLlama-1.1B-Chat-v1.0] ISOLATION
```

## 7. Combining test markers

Multiple markers can be combined for the same test case using commas. Both formats are valid:

```yaml
- test_case.py::test_function[param] ISOLATION, TIMEOUT (90)
- test_case.py::test_function[param] TIMEOUT (90), ISOLATION
```

### Example:
```yaml
# Regular test (runs with other tests)
- accuracy/test_llm_api.py::test_basic_functionality[gpt2]

# Test with timeout only
- accuracy/test_llm_api.py::test_long_running[model] TIMEOUT (60)

# Isolated test (runs in separate process)
- accuracy/test_llm_api.py::test_memory_intensive[large_model] ISOLATION

# Isolated test with timeout
- accuracy/test_llm_api.py::test_complex_workflow[model] ISOLATION, TIMEOUT (120)
```

### Important Notes:
- **TIMEOUT**: Ensures the test terminates if it exceeds the specified time limit (in minutes). Useful for preventing stuck tests from blocking the pipeline.
- **ISOLATION**: Runs the test in a separate pytest process to avoid resource conflicts and state pollution. Use sparingly as it increases execution time.
- Ensure there is **at least one space** before and after each marker keyword
- Both markers are case-sensitive and must be written exactly as `TIMEOUT` and `ISOLATION`
