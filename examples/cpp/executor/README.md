# Executor API examples

This directory contains two examples that demonstrate how to use the `Executor` API. The first example defined in `executorExampleBasic.cpp` shows how you can generate output tokens for a single prompt in only a few lines of code. The second example defined in `executorExampleAdvanced.cpp` supports more options such as providing an arbitrary number of input requests with arbitrary tokens per request and running in streaming mode.

## Building the examples

To build the examples, you first need to build the TensorRT-LLM C++ shared libraries (`libtensorrt_llmm.so` and `libnvinfer_plugin_tensorrt_llm.so`) using the [`build_wheel.py`](source:scripts/build_wheel.py) script. Alternatively, if you have already build the TensorRT-LLM libraries, you can modify the provided `CMakeLists.txt` such that the `libtensorrt_llm.so` and `libnvinfer_plugin_tensorrt_llm.so` are imported properly.

Once the TensorRT-LLM libraries are built, you can run

```
mkdir build
cd build
cmake ..
make -j
```
from the `./examples/cpp/executor/` folder to build the basic and advanced examples.

## Preparing the TensorRT-LLM engine(s)

Before you run the examples, please make sure that you have already built engine(s) using the TensorRT-LLM API.

Use `trtllm-build` to build the TRT-LLM engine.

## Running the examples

### executorExampleBasic

From the `examples/cpp/executor/build` folder, you can get run the `executorExampleBasic` example with:

```
./executorExampleBasic <path_to_engine_dir>
```
where `<path_to_engine_dir>` is the path to the directly containing the TensorRT engine files.

### executorExampleAdvanced

From the `examples/cpp/executor/build` folder, you can also run the `executorExampleAdvanced` example. To get the full list of supported input arguments, type

```
./executorExampleAdvanced -h
```

For example, you can run:

```
./executorExampleAdvanced --engine_dir <path_to_engine_dir>  --input_tokens_csv_file ../inputTokens.csv
```

to run with the provided dummy input tokens from `inputTokens.csv`. Upon successful completion, you should see the following in the logs:
```
[TensorRT-LLM][INFO] Creating request with 6 input tokens
[TensorRT-LLM][INFO] Creating request with 4 input tokens
[TensorRT-LLM][INFO] Creating request with 10 input tokens
[TensorRT-LLM][INFO] Got 20 tokens for beam 0 for requestId 3
[TensorRT-LLM][INFO] Request id 3 is completed.
[TensorRT-LLM][INFO] Got 14 tokens for beam 0 for requestId 2
[TensorRT-LLM][INFO] Request id 2 is completed.
[TensorRT-LLM][INFO] Got 16 tokens for beam 0 for requestId 1
[TensorRT-LLM][INFO] Request id 1 is completed.
[TensorRT-LLM][INFO] Writing output tokens to outputTokens.csv
[TensorRT-LLM][INFO] Exiting.
```

#### Multi-GPU run

To run the `executorExampleAdvanced` on models that require multiple GPUs, you can run the example using MPI as follows:

```
mpirun -n <num_ranks> --allow-run-as-root ./executorExampleAdvanced --engine_dir <path_to_engine_dir>  --input_tokens_csv_file ../inputTokens.csv
```
where `<num_ranks>` must equal to `tp*pp` for the TensorRT engine. By default GPU device IDs `[0...(num_ranks-1)]` will be used.

Alternatively, it's also possible to run multi-GPU model by using the so-called `Orchestrator` communication mode, where the `Executor` instance will automatically spawn additional processes to run the model on multiple GPUs. To use the `Orchestrator` communication mode, you can run the example with:

```
./executorExampleAdvanced --engine_dir <path_to_engine_dir>  --input_tokens_csv_file ../inputTokens.csv --use_orchestrator_mode --worker_executable_path <path_to_executor_worker>
```
where `<path_to_executor_worker>` is the absolute path to the stand-alone executor worker executable, located at`cpp/build/tensorrt_llm/executor_worker/executorWorker` by default.
