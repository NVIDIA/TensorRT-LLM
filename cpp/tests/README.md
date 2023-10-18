# C++ Tests

This document explains how to build and run the C++ tests, and the included [resources](resources).

Windows users: Be sure to set DLL paths as specified in [Extra Steps for C++ Runtime Usage](../../windows/README.md#extra-steps-for-c-runtime-usage).

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

[Scripts](resources/scripts) are provided that download the GPT2 and GPT-J models from Huggingface and convert them to TensorRT engines.
The weights and built engines are stored under [cpp/tests/resources/models](resources/models).
To build the engines from the top-level directory:

```bash
PYTHONPATH=examples/gpt:$PYTHONPATH python3 cpp/tests/resources/scripts/build_gpt_engines.py
PYTHONPATH=examples/gptj:$PYTHONPATH python3 cpp/tests/resources/scripts/build_gptj_engines.py
PYTHONPATH=examples/llama:$PYTHONPATH python3 cpp/tests/resources/scripts/build_llama_engines.py
```

It is possible to build engines with tensor and pipeline parallelism for LLaMA using 4 GPUs.

```bash
PYTHONPATH=examples/llama python3 cpp/tests/resources/scripts/build_llama_engines.py --only_multi_gpu
```

### Generate expected output

End-to-end tests read inputs and expected outputs from Numpy files located at [cpp/tests/resources/data](resources/data). The expected outputs can be generated using [scripts](resources/scripts) which employ the Python runtime to run the built engines:

```bash
PYTHONPATH=examples/gpt:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_gpt_output.py
PYTHONPATH=examples/gptj:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_gptj_output.py
PYTHONPATH=examples/llama:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_llama_output.py
PYTHONPATH=examples/CHATGLM6B:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_CHATGLM6B_output.py
```

### Generate data with tensor and pipeline parallelism

It is possible to generate tensor and pipeline parallelism data for LLaMA using 4 GPUs. To generate results from the top-level directory:

```bash
PYTHONPATH=examples/llama mpirun -n 4 python3 cpp/tests/resources/scripts/generate_expected_llama_output.py --only_multi_gpu
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
