# TensorRT LLM test definitions

The following subfolder contains test definitions for Tensorrt LLM.


## Directory structure

~~~
.
└── turtle              # Root directory
    ├── defs            #     Tiest definitions
    ├── perf_configs    #     Configs for perf tests
    └── test_lists      #     Test lists
        ├── bloom       #         Legacy test lists used by TURTLE (Do not add any new test lists here)
        ├── test-db     #         Test-DB (New test list convention adopted by pytest)
        ├── dev         #         Other test lists used by TRT LLM developers
        ├── qa          #         Test lists used by QA
        └── waives.txt  #         Test waive list
~~~

- To run perf tests, you also need to first build the cpp benchmark by calling `build_wheel.py` with `--benchmarks` flag.

## Run perf tests

All the perf test names are in the form of `perf/test_perf.py::test_perf[...]` where the `...` part is the test parameters.

TRT-LLM performance tests is still using TURTLE as of Dec 2024, and will be replaced by pytest eventually after pytest method supports all the functionalities. See [TURTLE test](./llm-test-defs/README.md) on how to run perf test.  TURTLE itself has many options and need some additional setup, thus a shortcut script [run_turtle.py](./tests/llm-test-defs/run_turtle.py) was developed to help developers who is not familiar with TURTLE

```bash
# execute these in the tensorrt-llm source repo root dir.
# install dependencies, do not need to do it every time if already installed.
pip install -r requirements-dev.txt

# example 1: list all available the test cases
python3 tests/llm-test-defs/run_turtle.py -l

# example 2: run a test case
# For example, if QA reports a perf bug for `perf/test_perf.py::test_perf[llama_7b-cppmanager-exe-plugin_ifb-float16-input_output_len:128,128,+512,32]`, then you can repro it by running:
python3 tests/llm-test-defs/run_turtle.py -t "perf/test_perf.py::test_perf[llama_7b-cppmanager-exe-plugin_ifb-float16-input_output_len:128,128,+512,32]"

```

The captured perf metrics will be saved in `build/turtle_output/perf_scripts_test_results.csv` and the test logs are saved in `build/turtle_output/workspace/logs`. Currently, we capture these perf metrics:

1. `test_perf_metric_build_time`: The engine building time in seconds.
2. `test_perf_metric_build_peak_cpu_memory`: The build-phase peak CPU mem usage in MB.
3. `test_perf_metric_build_peak_gpu_memory`: The build-phase peak GPU mem usage in MB.
4. `test_perf_metric_inference_time`: The inference latency in ms.
5. `test_perf_metric_inference_peak_gpu_memory`: The inference-phase peak GPU mem usage in GB.
6. `test_perf_metric_context_gpu_memory`: The context GPU mem usage in MB.

## Common Issues and solutions

1. No package 'libffi' found
Install libffi by `sudo apt-get install libffi-dev` and then remove the turtle-venv by `rm -fr build/turtle_venv`, and rerun.

2. ModuleNotFoundError: No module named 'nrsu'
Install nrsu by `python3 -m pip install --extra-index-url https://sc-hw-artf.nvidia.com/api/pypi/compute-pypi-local/simple/ nrsu==1.0.231107094326` and rerun.
