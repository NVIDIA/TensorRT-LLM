# How to run TRT-LLM tests

## 1. Unit test

All the tests contained in the current directory (include the recursively sub directories) except the [llm-test-defs](./llm-test-defs) folder are considered as "unit test" in this doc, these tests can use the python standard [unittests](https://docs.python.org/3/library/unittest.html) and [pytest](https://docs.pytest.org/en/stable/). Since pytest are compatible with the unittest framework, we use pytest to launch these in the CI.

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

# example 1: runs all the tests under this directory, ignores the llm-test-defs. WARNING: this can takes a very long time
pytest ./

# example 2: run a single test file
pytest ./test_builder.py

# example 3: run a test in a subfolder
pytest ./functional

# example 4: run a test with a substr
pytest -k test_basic_builder_flow
```

## 2. Integration test

There are 2 ways to launch the integration tests, TURTLE and pytest.

- TURTLE is used in CI and QA env as of Jul 2024.
  TRT-LLM integration tests use TURTLE for a legacy reason, because when the TRT-LLM project started TURTLE was already verified and integrate to various test env like TRT-LLM QA env, TRT CI, TRT QA env.  For detailed usages, see [TURTLE test readme](./tests/llm-test-defs/README.md), this doc just listed a few simple usages.

- pytest support was added by MR 4974, note the pytest way is still in alpha, and will add more features by JIRA EPIC TRTLLM-945.
  TURTLE will be deprecated and replaced by pytest in various CI envs after the TRTLLM-945 EPIC finished.
  Please report issues in the `#swdl-trt-llm-dev` channel if you meet issues using pytest.
  You can read the pytest official doc for details, https://docs.pytest.org/en/stable/


It's recommended to use pytest to local functional testing purpose, since it has less dependencies.
But if you want to reproduce the CI or QA bugs before they migrate to pytest, you should use TURTLE. You should find the TURTLE test cases from the failure logs of the bug.


### Recommended way for local test: use pytest

Using pytest, there is no need to clone or install anything, since pytest is already included in the requirements-dev.txt

```bash
# The integration tests will read the models weights data from path specified LLM_MODELS_ROOT env
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models

# in root dir
pip install -r requirements-dev.txt

cd tests/llm-test-defs/turtle/defs
# example 1: run a case
pytest "test_accuracy.py::test_accuracy_gpt[gpt-context-fmha-enabled]"

# example 2: run a test list
pytest --test-list ../test_lists/bloom/l0_accuracy_1.txt

# example 3: list all the cases
pytest --co -q

# example 4: run a test with a sub string
pytest -k test_llm_gpt2_medium_bad_words_1gpu
```

### Legacy way: use TURTLE

TURTLE itself has many options and need some additional setup, thus a shortcut script [run_turtle.py](./tests/llm-test-defs/run_turtle.py) was developed to help developers who is not familiar with TURTLE

```bash
# execute these in the tensorrt-llm source repo root dir.
# install dependencies, do not need to do it every time if already installled.
pip install -r requirements-dev.txt

# example 1: run turtle with only one test case
# run_turtle.py internally will clone TURTLE repo to local, and prepare an virtualenv for TURTLE tool (called turtle-venv),
# all the test will run in another python env (called test-venv).
python3 tests/llm-test-defs/run_turtle.py -t "test_accuracy.py::test_accuracy_gpt[gpt-context-fmha-enabled]"

# example 2: run turtle with a test list file contains a subset of test cases
python3 tests/llm-test-defs/run_turtle.py -f tests/llm-test-defs/turtle/test_lists/bloom/l0_accuracy_1.txt

# example 3: list all available the test cases
python3 tests/llm-test-defs/run_turtle.py -l

```

Note: the folder structure `tests/llm-test-defs/turtle/defs` exists for TURTLE purpose, the integration test will be eventually restructured to something like `tests/integration` after migrate to pytest.

## 3. Performance test

TRT-LLM performance tests are also using TURTLE as of Jul 2024, and will be replaced by pytest eventually after pytest method supports all the functionalities. See [TURTLE test](./llm-test-defs/README.md) on how to run perf test.


## 4. C++ runtime test

TRT-LLM C++ runtime tests are using [google-test](https://github.com/google/googletest) framework. The C++ runtime relies on TRT-LLM python frontend to generate engines as test data, so there is a python script to wrap the engine generation and google test launch command line, `../cpp/tests/resources/scripts/test_cpp.py`.
Details usages, see [C++ Test document](../cpp/tests/README.md).
