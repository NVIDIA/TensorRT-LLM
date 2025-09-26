# TensorRT LLM test definitions

The following subfolder contains test definitions for Tensorrt LLM.


## Directory structure

~~~
.
└── integration              # Root directory for integration tests
    ├── defs            #     Test definitions
    ├── perf_configs    #     Configs for perf tests
    └── test_lists      #     Test lists
        ├── test-db     #         Test-DB that is the test list convention adopted by CI
        ├── dev         #         Other test lists used by TRT LLM developers
        ├── qa          #         Test lists used by QA
        └── waives.txt  #         Test waive list
~~~

- To run perf tests, you also need to first build the cpp benchmark by calling `build_wheel.py` with `--benchmarks` flag.

## Run perf tests

All the perf test names are in the form of `perf/test_perf.py::test_perf[...]` where the `...` part is the test parameters.

Below are some specific pytest options used for perf tests

```bash
# execute these in the TensorRT LLM source repo root dir.
# install dependencies, do not need to do it every time if already installed.
pip install -r requirements-dev.txt

# example 1: run a test case
# For example, if QA reports a perf bug for `perf/test_perf.py::test_perf[llama_7b-cppmanager-exe-plugin_ifb-float16-input_output_len:128,128,+512,32]`, then you can repro it by running:
cd LLM_ROOT/tests/integration/defs
echo "perf/test_perf.py::test_perf[llama_7b-cppmanager-exe-plugin_ifb-float16-input_output_len:128,128,+512,32]" > perf.txt
pytest --perf --test-list=perf.txt --output-dir=/workspace/test-log --perf-log-formats csv --perf-log-formats yaml

```

The captured perf metrics will be saved in `/workspace/test-log/perf_scripts_test_results.csv` or `/workspace/test-log/perf_scripts_test_results.yaml` depends on the option `--perf-log-formats`, and the test logs are saved in `/workspace/test-log/result.xmk`. Currently, we capture these perf metrics:

1. `test_perf_metric_build_time`: The engine building time in seconds.
2. `test_perf_metric_build_peak_cpu_memory`: The build-phase peak CPU mem usage in MB.
3. `test_perf_metric_build_peak_gpu_memory`: The build-phase peak GPU mem usage in MB.
4. `test_perf_metric_inference_time`: The inference latency in ms.
5. `test_perf_metric_inference_peak_gpu_memory`: The inference-phase peak GPU mem usage in GB.
6. `test_perf_metric_context_gpu_memory`: The context GPU mem usage in MB.

## Common Issues and solutions

1. No package 'libffi' found
Install libffi by `sudo apt-get install libffi-dev` and rerun.
