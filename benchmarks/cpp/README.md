# Benchmark for C++ Runtime

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Usage

### 1. Build TensorRT-LLM and benchmarking source code

Please follow the [`installation document`](../../../README.md) to build TensorRT-LLM.

After that, you can build benchmarking source code for C++ runtime
```
cd cpp/build
make -j benchmarks
```

### 2. Launch C++ benchmarking

Before you launch C++ benchmarking, please make sure that you have already built engine(s) using TensorRT-LLM API, C++ benchmarking code cannot generate engine(s) for you.

You can reuse the engine built by benchmarking code for Python Runtime, please see that [`document`](../python/README.md).

For detailed usage, you can do the following
```
cd cpp/build

# You can directly execute the binary for help information
./benchmarks/gptSessionBenchmark --help
./benchmarks/bertBenchmark --help
```

Take GPT-350M as an example for single GPU

```
./benchmarks/gptSessionBenchmark \
    --model gpt_350m \
    --engine_dir "../../benchmarks/gpt_350m/" \
    --batch_size "1" \
    --input_output_len "60,20"

# Expected ouput:
# [BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 40.81
```
Take GPT-175B as an example for multiple GPUs
```
mpirun -n 8 ./benchmarks/gptSessionBenchmark \
    --model gpt_175b \
    --engine_dir "../../benchmarks/gpt_175b/" \
    --batch_size "1" \
    --input_output_len "60,20"

# Expected ouput:
# [BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 792.14
```

*Please note that the expected outputs in that document are only for reference, specific performance numbers depend on the GPU you're using.*
