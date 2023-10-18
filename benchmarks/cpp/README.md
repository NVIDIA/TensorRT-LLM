# Benchmark for C++ Runtime

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Usage

### 1. Build TensorRT-LLM and benchmarking source code

Please follow the [`installation document`](../../../README.md) to build TensorRT-LLM.

Windows users: Follow the
[`Windows installation document`](../../../windows/README.md)
instead, and be sure to set DLL paths as specified in
[Extra Steps for C++ Runtime Usage](../../../windows/README.md#extra-steps-for-c-runtime-usage).

After that, you can build benchmarking source code for C++ runtime
```
cd cpp/build
make -j benchmarks
```

### 2. Launch C++ benchmarking (Fixed BatchSize/InputLen/OutputLen)

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

### 3. Launch Batch Manager benchmarking (Inflight/V1 batching)

#### Prepare dataset

Run a preprocessing script to prepare dataset. This script converts the prompts(string) in the dataset to input_ids.
```
python3 prepare_dataset.py \
    --dataset <path/to/dataset> \
    --max_input_len 300 \
    --tokenizer_dir <path/to/tokenizer> \
    --tokenizer_type auto \
    --output preprocessed_dataset.json
```
For `tokenizer_dir`, specifying the path to the local tokenizer that have already been downloaded, or simply the name of the tokenizer from HuggingFace like `gpt2` will both work. The tokenizer will be downloaded automatically for the latter case.

#### Prepare TensorRT-LLM engines
Please make sure that the engines are built with argument `--use_inflight_batching` and `--remove_input_padding` if you'd like to benchmark inflight batching, for more details, please see the document in TensorRT-LLM examples.

#### Launch benchmarking

For detailed usage, you can do the following
```
cd cpp/build

# You can directly execute the binary for help information
./benchmarks/gptManagerBenchmark --help
```

Take GPT-350M as an example for single GPU V1 batching
```
./benchmarks/gptManagerBenchmark \
    --model gpt \
    --engine_dir ../../examples/gpt/trt_engine/gpt2/fp16/1-gpu/ \
    --type V1 \
    --dataset ../../benchmarks/cpp/preprocessed_dataset.json
```

Take GPT-350M as an example for 2-GPU inflight batching
```
mpirun -n 2 ./benchmarks/gptManagerBenchmark \
    --model gpt \
    --engine_dir ../../examples/gpt/trt_engine/gpt2-ib/fp16/2-gpu/ \
    --type IFB \
    --dataset ../../benchmarks/cpp/preprocessed_dataset.json
```
