# Benchmark for C++ Runtime

This document explains how to benchmark the models supported by TensorRT-LLM on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Usage

### 1. Build TensorRT-LLM and benchmarking source code

Please follow the [`installation document`](../../README.md#installation) to build TensorRT-LLM.

Note that the benchmarking source code for C++ runtime is not built by default, you can use the argument `--benchmarks` in [`build_wheel.py`](source:scripts/build_wheel.py) to build the corresponding executable.

Windows users: Follow the
[`Windows installation document`](../../windows/README.md)
instead, and be sure to set DLL paths as specified in
[Extra Steps for C++ Runtime Usage](../../windows/README.md#extra-steps-for-c-runtime-usage).

### 2. Launch C++ benchmarking (Fixed BatchSize/InputLen/OutputLen)

#### Prepare TensorRT-LLM engine(s)

Before you launch C++ benchmarking, please make sure that you have already built engine(s) using TensorRT-LLM API, C++ benchmarking code cannot generate engine(s) for you.

You can use the [`build.py`](source:benchmarks/python/build.py) script to build the engine(s). Alternatively, if you have already benchmarked Python Runtime, you can reuse the engine(s) built previously, please see that [`document`](../python/README.md).

####  Launch benchmarking

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

# Expected output:
# [BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 40.81
```
Take GPT-175B as an example for multiple GPUs
```
mpirun -n 8 ./benchmarks/gptSessionBenchmark \
    --model gpt_175b \
    --engine_dir "../../benchmarks/gpt_175b/" \
    --batch_size "1" \
    --input_output_len "60,20"

# Expected output:
# [BENCHMARK] batch_size 1 input_length 60 output_length 20 latency(ms) 792.14
```

If you want to obtain context and generation logits, you could build an enigne with `--gather_context_logits` and `--gather_generation_logits`, respectively. Enable `--gather_all_token_logits` will enable both of them.

If you want to get the logits, you could run gptSessionBenchmark with `--print_all_logits`. This will print a large number of logit values and has a certain impact on performance.

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
