# How to run TRT-LLM tests

## 1. Unit test (Python)

All the tests contained in the `unittest` directory folder are considered as "unit test" in this doc, these tests can use the python standard [unittests](https://docs.python.org/3/library/unittest.html) and [pytest](https://docs.pytest.org/en/stable/). Since pytest are compatible with the unittest framework, we use pytest to launch these in the CI.

Unit test should be small, fast, and test only for specific function.

If you need to run them locally, the only dependencies are `requirements-dev.txt`.

```bash
# in tensorrt-llm source repo root dir
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

Examples to run integration test locally.

```bash
# The integration tests will read the models weights data from path specified LLM_MODELS_ROOT env
# Test would fail or be skipped if LLM_MODELS_ROOT is not set to a correct directory.
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models

# in root dir
pip install -r requirements-dev.txt
cd tests/integration/defs

# example 1: run a case
pytest "accuracy/test_accuracy.py::TestGpt2CnnDailymail::test_auto_dtype"

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
        AssertionError: /scratch.trt_llm_data/llm-models/gpt2-medium does not exist, and fail_if_path_is_invalid is True, please check the cache directory
        assert False

      conftest.py:149: AssertionError
    ```
    If you see above failures when running pytest locally, its likely that you didn't set the `LLM_MODELS_ROOT` env correctly. The default one is `/scratch.trt_llm_data`, since this is the one used in CI env.

    You should set this `LLM_MODELS_ROOT` correctly to the path where you mount the IT scratch `dc2-cdot87-swgpu01-lif1b:/vol/scratch19/scratch.trt_llm_data/llm-models`.
    If you are running in computelab nodes, the scratch path is `/home/scratch.trt_llm_data/llm-models`, and you should add `-v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro` when starting your container, and `export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models` before running the pytest.


## 4. C++ runtime test

TRT-LLM C++ runtime tests are using [google-test](https://github.com/google/googletest) framework, and Pytest is used to run sets of these tests.

The C++ runtime relies on TRT-LLM python frontend to generate engines as test data, so there are scripts to generate the engines in the C++ test [resources directory](../cpp/tests/resources/).
Pytest calls these scripts from fixtures prior to launching the test cases.

Details on usage of the resources scripts can be found in the [C++ Test document](../cpp/tests/README.md).

## 5. Performance regression test

Porformance in QA tests and CI are still using TURTLE, see [legacy turtle perf test readme](./integration/README.md)

# How to add test to CI

## 1. How does the CI work

Due to CI hardware resource limitation, and some cases only run on specific GPUs, the test cases are managed based on GPU type.

In directory `integration/test_lists/test-db`, each yml file corresponds to a GPU type.

In file `jenkins/L0_Test.groovy`, the variable `turtleConfigs` maps yml files to CI stages.

Currently the yml files are manually maintained, which requires developer to update them when new test cases are added.

### How to choose GPU type

The CI resource of each GPU type is different. Usually you should choose the cheapest GPU that fulfills test requirements. In most cases, an integration test case should only run on one GPU type, unless it's very important or has different behaviours on different GPUs.

The priority is A10 > A30 > L40s > A100 > H100 > B200.

## 2. Add an integration test

Integrations tests usually run entire workflow, containing checkpoint converting, engine building and evaluating, to check functional and accuracy.

Integration tests are stored in `integration/defs`. Once a new integration test case is added, the yml files must be updated to contain the newly added case. Otherwise, the CI will not be able to collect and run this case.

## 3. Add a unit test

A unit test are used to test a standalone feature or building block, and only runs partial workflow.

For legacy and case management reason, the CI doesn't run unit tests directly. It uses a bridge to map multiple unit test cases into one integration test case, and manages these bridged cases.
The bridge is implemented in `integration/defs/test_unittests.py` and `pytest_generate_tests` function in `tests/integration/defs/conftest.py`.

In `integration/test_lists/test-db`, cases with prefix `unittest/` are treated as unit test bridges. Each of them generates an instance of `test_unittests_v2` which executes a `pytest` subprocess in `tests/unittest` directory.
The entire line will be passed as commandline arguments of `pytest` subprocess.

For example, `unittest/attention/test_gpt_attention.py -k "partition0"` is equivalent to `cd tests; pytest unittest/attention/test_gpt_attention.py -k "partition0"`.

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
