(weight-streaming)=

## Running With Weight Streaming to Reduce GPU Memory Consumption

TensorRT Weight Streaming can offload some weights to the CPU memory and stream them to the GPU memory during runtime.
This can reduce the weights size in GPU memory, therefore, we can run larger models or larger batch sizes in the same GPU memory budget.


During build time, build the engine with `--weight-streaming --gemm_plugin disable` since Weight Streaming only supports non-plugin weights. During runtime, run with `--gpu_weights_percent x` to config the percent of weights that remained on the GPU. `x` can be a value from `0.0` to `1.0`.

Here is an example to run llama-7b with Weight Streaming:
```bash

# Convert model as normal. Assume hugging face model is in llama-7b-hf/
python3 examples/llama/convert_checkpoint.py \
    --model_dir llama-7b-hf/ \
    --output_dir /tmp/llama_7b/trt_ckpt/fp16/1-gpu/ \
    --dtype float16

# Build engine that enabled Weight Streaming.
trtllm-build \
    --checkpoint_dir /tmp/llama_7b/trt_ckpt/fp16/1-gpu/ \
    --output_dir /tmp/llama_7b/trt_engines/fp16/1-gpu/ \
    --weight_streaming \
    --gemm_plugin disable \
    --max_batch_size 128 \
    --max_input_len 512 \
    --max_seq_len 562

# Run the engine with 20% weights in GPU memory.
python3 examples/summarize.py \
    --engine_dir /tmp/llama_7b/trt_engines/fp16/1-gpu/ \
    --batch_size 1 \
    --test_trt_llm \
    --hf_model_dir llama-7b-hf/ \
    --data_type fp16 \
    --gpu_weights_percent 0.2

```

We can also benchmark the efficiency of Weight Streaming. Here is an example:
```bash
python3 benchmarks/python/benchmark.py \
    --engine_dir /tmp/llama_7b/trt_engines/fp16/1-gpu/ \
    --batch_size "1;32" \
    --input_output_len "256,32" \
    --gpu_weights_percent "0.0;0.3;0.6;1.0" \
    --dtype float16 \
    --csv \
    --log_level verbose
```


### API Changes

To build engines with Weight Streaming enabled, some API changes are needed for the builder:
- Added a new bool member `weight_streaming` to class `BuildConfig`.
- Added a new bool parameter `weight_streaming` to method `create_builder_config` of class `Builder`.

To run with Weight Streaming with `Executor`, there are some API change to its config `ExecutorConfig`:
- Added a new float parameter `gpuWeightsPercent` to the constructor of `ExecutorConfig`.
- Added two member functions `setGpuWeightsPercent` and `getGpuWeightsPercent` to set and get the GPU weights percentage.

Here is an example to create an `Executor` with Weight Streaming:
```c++
...
auto executorConfig = tle::ExecutorConfig(gpuWeightsPercent=0.5);
auto executor = tle::Executor("model_path", tensorrt_llm::executor::ModelType::kDECODER_ONLY, executorConfig);
...
```
