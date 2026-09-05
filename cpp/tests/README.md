# C++ Tests

This document explains how to build and run the C++ tests, and the included [resources](resources).

## Pytest Scripts

The unit tests can be launched via the Pytest script in [test_unit_tests.py](../../tests/integration/defs/cpp/test_unit_tests.py). These do not require engines to be built. The Pytest script will also build TRT-LLM.

The Pytest script in [test_multi_gpu.py](../../tests/integration/defs/cpp/test_multi_gpu.py) builds TRT-LLM and runs the multi-GPU C++ tests.

To get an overview of the tests and their parameterization, call:

```bash
pytest tests/integration/defs/cpp/test_unit_tests.py --collect-only
pytest tests/integration/defs/cpp/test_multi_gpu.py --collect-only
```

All tests take the number of the CUDA architecture of the GPU you wish to use as a parameter e.g. 90 for Hopper.

It is possible to choose unit tests or a multi-GPU test by name.
Example calls could look like this:

```bash
export LLM_MODELS_ROOT="/path/to/model_cache"

pytest tests/integration/defs/cpp/test_unit_tests.py::test_unit_tests[runtime-90]

pytest tests/integration/defs/cpp/test_multi_gpu.py::test_mpi_utils[90]

pytest tests/integration/defs/cpp/test_multi_gpu.py::test_cache_transceiver[90-nixl_kvcache-2proc]
```

## Manual steps

### Compile

From the top-level directory call:

```bash
CPP_BUILD_DIR=cpp/build
python3 scripts/build_wheel.py -a "80-real;86-real" --build_dir ${CPP_BUILD_DIR}
pip install -r requirements-dev.txt
pip install build/tensorrt_llm*.whl
cd $CPP_BUILD_DIR && make -j$(nproc) google-tests
```

Single tests can be executed from `$CPP_BUILD_DIR/tests/unit_tests/<group>`, e.g.

```bash
./$CPP_BUILD_DIR/tests/unit_tests/common/loggerTest
```

### Run all tests with ctest

To run all tests and produce an xml report, call

```bash
./$CPP_BUILD_DIR/ctest --output-on-failure --output-junit "cpp-test-report.xml"
```