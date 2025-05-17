# C++ Tests

This document explains how to build and run the C++ tests, and the included [resources](resources).

## Pytest Scripts

The unit tests can be launched via the Pytest script in [test_unit_tests.py](../../tests/integration/defs/cpp/test_unit_tests.py). These do not require engines to be built. The Pytest script will also build TRT-LLM.

The Pytest scripts in [test_e2e.py](../../tests/integration/defs/cpp/test_e2e.py) and [test_multi_gpu.py](../../tests/integration/defs/cpp/test_multi_gpu.py) build TRT-LLM, build engines, and generate expected outputs and execute the end-to-end C++ tests all in one go.
`test_e2e.py` and `test_multi_gpu.py` contain single and multi-device tests, respectively.

To get an overview of the tests and their parameterization, call:

```bash
pytest tests/integration/defs/cpp/test_unit_tests.py --collect-only
pytest tests/integration/defs/cpp/test_e2e.py --collect-only
pytest tests/integration/defs/cpp/test_multi_gpu.py --collect-only
```

All tests take the number of the CUDA architecture of the GPU you wish to use as a parameter e.g. 90 for Hopper.

It is possible to choose unit tests or a single model for end-to-end tests.
Example calls could look like this:

```bash
export LLM_MODELS_ROOT="/path/to/model_cache"

pytest tests/integration/defs/cpp/test_unit_tests.py::test_unit_tests[runtime-90]

pytest tests/integration/defs/cpp/test_e2e.py::test_model[llama-90]

pytest tests/integration/defs/cpp/test_e2e.py::test_benchmarks[gpt-90]

pytest tests/integration/defs/cpp/test_multi_gpu.py::TestDisagg::test_symmetric_executor[gpt-mpi_kvcache-90]
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

Single tests can be executed from `CPP_BUILD_DIR/tests`, e.g.

```bash
./$CPP_BUILD_DIR/tests/allocatorTest
```

### End-to-end tests

`trtGptModelRealDecoderTest` and `executorTest` require pre-built TensorRT engines, which are loaded in the tests. They also require data files which are stored in [cpp/tests/resources/data](resources/data).

#### Build engines

[Scripts](resources/scripts) are provided that download the GPT2 and GPT-J models from Huggingface and convert them to TensorRT engines.
The weights and built engines are stored under [cpp/tests/resources/models](resources/models).
To build the engines from the top-level directory:

```bash
PYTHONPATH=examples/models/core/gpt:$PYTHONPATH python3 cpp/tests/resources/scripts/build_gpt_engines.py
PYTHONPATH=examples/models/core/llama:$PYTHONPATH python3 cpp/tests/resources/scripts/build_llama_engines.py
PYTHONPATH=examples/medusa:$PYTHONPATH python3 cpp/tests/resources/scripts/build_medusa_engines.py
PYTHONPATH=examples/eagle:$PYTHONPATH python3 cpp/tests/resources/scripts/build_eagle_engines.py
PYTHONPATH=examples/redrafter:$PYTHONPATH python3 cpp/tests/resources/scripts/build_redrafter_engines.py
```

It is possible to build engines with tensor and pipeline parallelism for LLaMA using 4 GPUs.

```bash
PYTHONPATH=examples/models/core/llama python3 cpp/tests/resources/scripts/build_llama_engines.py --only_multi_gpu
```

#### Generate expected output

End-to-end tests read inputs and expected outputs from Numpy files located at [cpp/tests/resources/data](resources/data). The expected outputs can be generated using [scripts](resources/scripts) which employ the Python runtime to run the built engines:

```bash
PYTHONPATH=examples:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_gpt_output.py
PYTHONPATH=examples:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_llama_output.py
PYTHONPATH=examples:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_medusa_output.py
PYTHONPATH=examples:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_eagle_output.py
PYTHONPATH=examples:$PYTHONPATH python3 cpp/tests/resources/scripts/generate_expected_redrafter_output.py
```

#### Generate data with tensor and pipeline parallelism

It is possible to generate tensor and pipeline parallelism data for LLaMA using 4 GPUs. To generate results from the top-level directory:

```bash
PYTHONPATH=examples mpirun -n 4 python3 cpp/tests/resources/scripts/generate_expected_llama_output.py --only_multi_gpu
```

#### Run test

After building the engines and generating the expected output execute the tests

```bash
./$CPP_BUILD_DIR/tests/batch_manager/trtGptModelRealDecoderTest
```

### Run all tests with ctest

To run all tests and produce an xml report, call

```bash
./$CPP_BUILD_DIR/ctest --output-on-failure --output-junit "cpp-test-report.xml"
```
