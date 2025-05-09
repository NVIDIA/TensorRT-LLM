(troubleshooting)=

# Troubleshooting

This document describes some of the frequently asked questions and their solutions in TensorRT-LLM, including problems of installation, model-building, model-execution, and input / output size.

## Installation Errors

During compilation and installation of TensorRT-LLM, many build errors can be resolved by simply deleting the build tree and rebuilding again.

In most occasions, these problems are caused by the workflow like: an old compilation -> some code change (update of the repo or users' writing) -> a later compilation.

Solution: try running build script with `--clean`, or try running `rm -r build cpp/build` before running build script again.

## Debug on Unit Tests

Here is an example to print the values of the MLP output tensor in the a unit test ([full example](../../../tests/test_debugging_api.py)).

1. Register the intermediate tensors as the network outputs with `register_network_output` API.

```python
class MLP(Module):

    def __init__(self, ...):
        super().__init__()
        # Do not modify the definition in `__init__` method
        self.fc = ...
        self.proj = ...

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = tensorrt_llm.functional.relu(inter)
        # Here register the tensor `inter` as our debug output tensor
        self.register_network_output('inter', inter)
        output = self.proj(inter)
        return output
```

2. Mark the intermediate tensors as network outputs.

```python
for k, v in gm.named_network_outputs():
    net._mark_output(v, k, dtype)
```

3. Print the tensors at runtime.

```python
print(outputs.keys())
print(outputs['inter'])
```

## Debug on E2E Models

Here is an example to print the values of the MLP output tensor in the GPT model.

1. Register the MLP output tensor in `tensorrt_llm/models/gpt/model.py`.

```python
        hidden_states = residual + attention_output.data

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        # Register as model output
        # ------------------------------------------------------
        self.register_network_output('mlp_output', hidden_states)
        # ------------------------------------------------------

        hidden_states = residual + hidden_states
```

2. Build the TensorRT engine of the model.

Enable the `--enable_debug_output` option when building engines with `trtllm-build`

```bash
cd examples/gpt

# Download hf gpt2 model
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# Convert to TensorRT-LLM checkpoint
python3 convert_checkpoint.py \
    --model_dir gpt2 \
    --dtype float16 \
    --output_dir gpt2/trt_ckpt/fp16/1-gpu

# Build TensorRT-LLM engines with --enable_debug_output
trtllm-build \
    --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu \
    --enable_debug_output \
    --output_dir gpt2/trt_engines/fp16/1-gpu
```

3. Print the intermediate output tensors.

Add debug info in `tensorrt_llm/runtime/generation.py`.

```python
        stream = torch.cuda.current_stream().cuda_stream
        instance_idx = step % 2
        if self.cuda_graph_mode and self.runtime.cuda_graph_instances[
                instance_idx] is not None:
            # launch cuda graph
            CUASSERT(
                cudart.cudaGraphLaunch(
                    self.runtime.cuda_graph_instances[instance_idx], stream))
            ok = True
        else:
            ok = self.runtime._run(context, stream)

        if not ok:
            raise RuntimeError(f"Executing TRT engine failed step={step}!")
        if self.debug_mode:
            torch.cuda.synchronize()
            # -------------------------------------------
            if step == 0:
                print(self.debug_buffer.keys())
            print(f"Step: {step}")
            print(self.debug_buffer['transformer.layers.6.mlp_output'])
            # -------------------------------------------
```

4. Run `../run.py` with `--debug_mode` and `--use_py_session`.

```bash
python3 ../run.py \
    --engine_dir gpt2/trt_engines/fp16/1-gpu \
    --tokenizer_dir gpt2 \
    --max_output_len 8 \
    --debug_mode \
    --use_py_session
```

5. See the value of the tensor.

```txt
......
dict_keys(['context_lengths', 'cache_indirection', 'position_ids', 'logits', 'last_token_ids', 'input_ids', 'kv_cache_block_pointers', 'host_kv_cache_block_pointers', 'sequence_length', 'host_past_key_value_lengths', 'host_sink_token_length', 'host_request_types', 'host_max_attention_window_sizes', 'host_context_lengths', 'transformer.layers.0.mlp_output', 'transformer.layers.1.mlp_output', 'transformer.layers.2.mlp_output', 'transformer.layers.3.mlp_output', 'transformer.layers.4.mlp_output', 'transformer.layers.5.mlp_output', 'transformer.layers.6.mlp_output', 'transformer.layers.7.mlp_output', 'transformer.layers.8.mlp_output', 'transformer.layers.9.mlp_output', 'transformer.layers.10.mlp_output', 'transformer.layers.11.mlp_output', 'transformer.layers.12.mlp_output', 'transformer.layers.13.mlp_output', 'transformer.layers.14.mlp_output', 'transformer.layers.15.mlp_output', 'transformer.layers.16.mlp_output', 'transformer.layers.17.mlp_output', 'transformer.layers.18.mlp_output', 'transformer.layers.19.mlp_output', 'transformer.layers.20.mlp_output', 'transformer.layers.21.mlp_output', 'transformer.layers.22.mlp_output', 'transformer.layers.23.mlp_output'])
Step: 0
tensor([[ 0.0294, -0.0260, -0.0776,  ..., -0.0560, -0.0235,  0.0273],
        [-0.0071,  0.5879,  0.1993,  ..., -1.0449, -0.6299,  0.5957],
        [-0.8779,  0.1050,  0.7090,  ...,  0.0910,  1.0713, -0.2939],
        ...,
        [ 0.1212, -0.0903, -0.5918,  ..., -0.1045, -0.3445,  0.1082],
        [-1.0723, -0.0732,  0.6157,  ...,  0.3452,  0.2998,  0.2649],
        [-0.7134,  0.9692, -0.1141,  ..., -0.0096,  0.9521,  0.1437]],
       device='cuda:0', dtype=torch.float16)
Step: 1
tensor([[-0.2107,  0.5874,  0.8179,  ...,  0.7900, -0.6890,  0.6064]],
       device='cuda:0', dtype=torch.float16)
Step: 2
tensor([[ 0.4192, -0.0047,  1.3887,  ..., -0.9028, -0.0682, -0.2820]],
       device='cuda:0', dtype=torch.float16)
Step: 3
tensor([[-0.7949, -0.5073, -0.1721,  ..., -0.5830, -0.1378, -0.0070]],
       device='cuda:0', dtype=torch.float16)
Step: 4
tensor([[-0.0804,  0.1272, -0.6255,  ..., -0.1072, -0.0523,  0.7144]],
       device='cuda:0', dtype=torch.float16)
Step: 5
tensor([[-0.3328, -0.8828,  0.3442,  ...,  0.8149, -0.0630,  1.2305]],
       device='cuda:0', dtype=torch.float16)
Step: 6
tensor([[-0.2225, -0.2079, -0.1459,  ..., -0.3555, -0.1672,  0.1135]],
       device='cuda:0', dtype=torch.float16)
Step: 7
tensor([[ 0.1290, -0.1556,  0.3977,  ..., -0.8218, -0.3291, -0.8672]],
       device='cuda:0', dtype=torch.float16)
Input [Text 0]: "Born in north-east France, Soyer trained as a"
Output [Text 0 Beam 0]: " chef before moving to London in the early"
```

## Debug Execution Errors

If problems come from plugins, try setting the environment variable `CUDA_LAUNCH_BLOCKING=1` to make kernels launch synchronously with their return status checked immediately.

If problems come from runtime-shape of the input tensors, double-check the shape (rank and length of each rank) and location (CPU / GPU) of input tensors for the engine obey the build-time setting.

For example, one possible reason of getting the error information like below is, we use mismatched configuration between engine building and running, including code change (update of repo or users' rewrting), too large or too small input shape, etc..

```txt
unexpected shape for input 'XXX' for model 'YYY'. Expected [-1,-1,-1], got [8,16]. NOTE: Setting a non-zero max_batch_size in the model config requires a batch dimension to be prepended to each input shape. If you want to specify the full shape including the batch dim in your input dims config, try setting max_batch_size to zero. See the model configuration docs for more info on max_batch_size.

[TensorRT-LLM][ERROR] Assertion failed: Tensor 'input_ids' has invalid shape (8192), expected (-1) (/code/tensorrt_llm/cpp/tensorrt_llm/runtime/tllmRuntime.cpp:149)

RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 8192 but got size 1024 for tensor number 1 in the list.
```

By setting environment variable `export TLLM_LOG_LEVEL=TRACE`, we can get more information about the TensorRT engine and context at runtime.

Before the first forward computation, the shapes of all input / output tensors and their corresponding allowed ranges are provided in a table like:

```txt
[TensorRT-LLM][TRACE] Information of engine input / output.
[TensorRT-LLM][TRACE] =====================================================================
[TensorRT-LLM][TRACE]              Name              |I/O|Location|DataType|    Shape     |
[TensorRT-LLM][TRACE] ---------------------------------------------------------------------
[TensorRT-LLM][TRACE] input_ids                      | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] position_ids                   | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] last_token_ids                 | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] kv_cache_block_offsets         | I |  GPU   | INT32  |(1, -1, 2, -1)|
[TensorRT-LLM][TRACE] host_kv_cache_block_offsets    | I |  GPU   | INT32  |(1, -1, 2, -1)|
[TensorRT-LLM][TRACE] host_kv_cache_pool_pointers    | I |  GPU   | INT64  |    (1, 2)    |
[TensorRT-LLM][TRACE] host_kv_cache_pool_mapping     | I |  GPU   | INT32  |     (28)     |
[TensorRT-LLM][TRACE] sequence_length                | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] host_request_types             | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] host_past_key_value_lengths    | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] context_lengths                | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] host_runtime_perf_knobs        | I |  GPU   | INT64  |     (16)     |
[TensorRT-LLM][TRACE] host_context_lengths           | I |  GPU   | INT32  |     (-1)     |
[TensorRT-LLM][TRACE] host_max_attention_window_sizes| I |  GPU   | INT32  |     (28)     |
[TensorRT-LLM][TRACE] host_sink_token_length         | I |  GPU   | INT32  |     (1)      |
[TensorRT-LLM][TRACE] cache_indirection              | I |  GPU   | INT32  | (-1, 1, -1)  |
[TensorRT-LLM][TRACE] logits                         | O |  GPU   |  FP32  | (-1, 65024)  |
[TensorRT-LLM][TRACE] =====================================================================
[TensorRT-LLM][TRACE] Information of optimization profile.
[TensorRT-LLM][TRACE] Optimization Profile 0:
[TensorRT-LLM][TRACE] =============================================================================
[TensorRT-LLM][TRACE]              Name              |     Min      |     Opt      |     Max      |
[TensorRT-LLM][TRACE] -----------------------------------------------------------------------------
[TensorRT-LLM][TRACE] input_ids                      |     (1)      |     (8)      |    (8192)    |
[TensorRT-LLM][TRACE] position_ids                   |     (1)      |     (8)      |    (8192)    |
[TensorRT-LLM][TRACE] last_token_ids                 |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] kv_cache_block_offsets         | (1, 1, 2, 1) |(1, 4, 2, 16) |(1, 8, 2, 32) |
[TensorRT-LLM][TRACE] host_kv_cache_block_offsets    | (1, 1, 2, 1) |(1, 4, 2, 16) |(1, 8, 2, 32) |
[TensorRT-LLM][TRACE] host_kv_cache_pool_pointers    |    (1, 2)    |    (1, 2)    |    (1, 2)    |
[TensorRT-LLM][TRACE] host_kv_cache_pool_mapping     |     (28)     |     (28)     |     (28)     |
[TensorRT-LLM][TRACE] sequence_length                |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] host_request_types             |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] host_past_key_value_lengths    |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] context_lengths                |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] host_runtime_perf_knobs        |     (16)     |     (16)     |     (16)     |
[TensorRT-LLM][TRACE] host_context_lengths           |     (1)      |     (4)      |     (8)      |
[TensorRT-LLM][TRACE] host_max_attention_window_sizes|     (28)     |     (28)     |     (28)     |
[TensorRT-LLM][TRACE] host_sink_token_length         |     (1)      |     (1)      |     (1)      |
[TensorRT-LLM][TRACE] cache_indirection              |  (1, 1, 1)   | (4, 1, 1024) | (8, 1, 2048) |
[TensorRT-LLM][TRACE] logits                         |  (1, 65024)  |  (4, 65024)  |  (8, 65024)  |
[TensorRT-LLM][TRACE] =============================================================================
```

Before each forward computation, the real shapes of all input / output tensors for TRT engine are provided by a table like:

```txt
[TensorRT-LLM][TRACE] Information of context input / output.
[TensorRT-LLM][TRACE] Using Optimization Profile: 0
[TensorRT-LLM][TRACE] =================================================
[TensorRT-LLM][TRACE]              Name              |I/O|   Shape    |
[TensorRT-LLM][TRACE] -------------------------------------------------
[TensorRT-LLM][TRACE] input_ids                      | I |    (33)    |
[TensorRT-LLM][TRACE] position_ids                   | I |    (33)    |
[TensorRT-LLM][TRACE] last_token_ids                 | I |    (3)     |
[TensorRT-LLM][TRACE] kv_cache_block_offsets         | I |(1, 3, 2, 4)|
[TensorRT-LLM][TRACE] host_kv_cache_block_offsets    | I |(1, 3, 2, 4)|
[TensorRT-LLM][TRACE] host_kv_cache_pool_pointers    | I |   (1, 2)   |
[TensorRT-LLM][TRACE] host_kv_cache_pool_mapping     | I |    (28)    |
[TensorRT-LLM][TRACE] sequence_length                | I |    (3)     |
[TensorRT-LLM][TRACE] host_request_types             | I |    (3)     |
[TensorRT-LLM][TRACE] host_past_key_value_lengths    | I |    (3)     |
[TensorRT-LLM][TRACE] context_lengths                | I |    (3)     |
[TensorRT-LLM][TRACE] host_runtime_perf_knobs        | I |    (16)    |
[TensorRT-LLM][TRACE] host_context_progress          | I |    (1)     |
[TensorRT-LLM][TRACE] host_context_lengths           | I |    (3)     |
[TensorRT-LLM][TRACE] host_max_attention_window_sizes| I |    (28)    |
[TensorRT-LLM][TRACE] host_sink_token_length         | I |    (1)     |
[TensorRT-LLM][TRACE] cache_indirection              | I |(3, 2, 256) |
[TensorRT-LLM][TRACE] logits                         | O | (3, 65024) |
[TensorRT-LLM][TRACE] =================================================
```

## Tips

* It's recommended to add options `–shm-size=1g –ulimit memlock=-1` to the
  docker or nvidia-docker run command.  Otherwise you may see NCCL errors when
  running multiple GPU inferences. See
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#errors
  for details.

* When building models, memory-related issues such as
```
[09/23/2023-03:13:00] [TRT] [E] 9: GPTLMHeadModel/layers/0/attention/qkv/PLUGIN_V2_Gemm_0: could not find any supported formats consistent with input/output data types
[09/23/2023-03:13:00] [TRT] [E] 9: [pluginV2Builder.cpp::reportPluginError::24] Error Code 9: Internal Error (GPTLMHeadModel/layers/0/attention/qkv/PLUGIN_V2_Gemm_0: could not find any supported formats consistent with input/output data types)
```
may happen. One possible solution is to reduce the amount of memory needed by
reducing the maximum batch size, input and output lengths. Another option is to
enable plugins, for example: `--gpt_attention_plugin`.

* MPI + Slurm

TensorRT-LLM is a
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)-aware package
that uses [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/). If you are
running scripts in a [Slurm](https://slurm.schedmd.com/) environment, you might
encounter interferences:
```
--------------------------------------------------------------------------
PMI2_Init failed to initialize.  Return code: 14
--------------------------------------------------------------------------
--------------------------------------------------------------------------
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support and therefore cannot
execute. There are several options for building PMI support under
SLURM, depending upon the SLURM version you are using:

  version 16.05 or later: you can use SLURM's PMIx support. This
  requires that you configure and build SLURM --with-pmix.

  Versions earlier than 16.05: you must use either SLURM's PMI-1 or
  PMI-2 support. SLURM builds PMI-1 by default, or you can manually
  install PMI-2. You must then build Open MPI using --with-pmi pointing
  to the SLURM PMI library location.

Please configure as appropriate and try again.
--------------------------------------------------------------------------
```

You may experience other problems like hanging on the program startup.

As a rule of thumb, if you are running TensorRT-LLM interactively on a Slurm
node, prefix your commands with `mpirun -n 1` to run TensorRT-LLM in a
dedicated MPI environment, not the one provided by your Slurm allocation.

For example: `mpirun -n 1 python3 examples/gpt/build.py ...`

It's critical that it's always `-n 1` regardless of how many GPUs are being used. If you'd use `-n 2` for a 2 GPU program it will not work. `mpirun` here isn't being used to orchestrate multiple processes, but to invoke the right environment on SLURM. The internal MPI implementation deals with spawning the additional processes.
