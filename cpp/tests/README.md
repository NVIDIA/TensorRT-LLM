# C++ Tests

This document explains how to build and run the C++ tests, and the included [resources](resources).

## Compile

From the top-level directory call:

```bash
CPP_BUILD_DIR=cpp/build
python3 scripts/build_wheel.py -a "80-real;86-real" --build_dir ${CPP_BUILD_DIR}
pip install -r requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com
pip install build/tensorrt_llm*.whl
cd $CPP_BUILD_DIR && make -j$(nproc) google-tests
```

Single tests can be executed from `CPP_BUILD_DIR/tests`, e.g.

```bash
./$CPP_BUILD_DIR/tests/allocatorTest
```

## End-to-end tests

`gptSessionTest`, `gptManagerTest` and `trtGptModelRealDecoderTest` require pre-built TensorRT engines, which are loaded in the tests. They also require data files which are stored in [cpp/tests/resources/data](resources/data).

### Build engines

To avoid discrepancy in the reference and tests data set `SKIP_GEMM_PLUGIN_PROFILINGS=1` to disable GEMM tactic profiling in GEMM plugins.

```bash
export SKIP_GEMM_PLUGIN_PROFILINGS=1
```

[Scripts](resources/scripts) are provided that download the GPT2 and GPT-J models from Huggingface and convert them to TensorRT engines.
The weights and built engines are stored under [cpp/tests/resources/models](resources/models).
To build the engines from the top-level directory:

```bash
PYTHONPATH=examples/gpt python3 cpp/tests/resources/scripts/build_gpt_engines.py
PYTHONPATH=examples/gptj python3 cpp/tests/resources/scripts/build_gptj_engines.py
PYTHONPATH=examples/llama python3 cpp/tests/resources/scripts/build_llama_engines.py
```

### Generate expected output

End-to-end tests read inputs and expected outputs from Numpy files located at [cpp/tests/resources/data](resources/data). The expected outputs can be generated using [scripts](resources/scripts) which employ the Python runtime to run the built engines:

```bash
PYTHONPATH=examples/gpt python3 cpp/tests/resources/scripts/generate_expected_gpt_output.py
PYTHONPATH=examples/gptj python3 cpp/tests/resources/scripts/generate_expected_gptj_output.py
PYTHONPATH=examples/llama python3 cpp/tests/resources/scripts/generate_expected_llama_output.py
```

### Run test

After building the engines and generating the expected output execute the tests

```bash
./$CPP_BUILD_DIR/tests/gptSessionTest
```

## Run all tests with ctest

To run all tests and produce an xml report, call

```bash
./$CPP_BUILD_DIR/ctest --output-on-failure --output-junit "cpp-test-report.xml"
```
