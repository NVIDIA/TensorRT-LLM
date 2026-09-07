(troubleshooting)=

# Troubleshooting

```{caution}
The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.
```

This document describes some of the frequently asked questions and their solutions in TensorRT-LLM, including problems of installation, model-building, model-execution, and input / output size.

## Installation Errors

During compilation and installation of TensorRT-LLM, many build errors can be resolved by simply deleting the build tree and rebuilding again.

In most occasions, these problems are caused by the workflow like: an old compilation -> some code change (update of the repo or users' writing) -> a later compilation.

Solution: try running build script with `--clean`, or try running `rm -r build cpp/build` before running build script again.

## Debug on Unit Tests

Here is an example to print the values of the MLP output tensor in a unit test. The former `tests/unittest/others/test_debugging_api.py` sample was removed with the TensorRT backend; see [TensorRT Backend Removal](../tensorrt-backend-removal.md).

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

The end-to-end TensorRT engine debug flow documented here
(`examples/models/core/gpt/convert_checkpoint.py`, `trtllm-build --enable_debug_output`,
and `run.py --debug_mode`) was removed with the TensorRT backend. HuggingFace
checkpoints load directly through the PyTorch backend (`trtllm-serve` / the LLM
API); there is no separate convert/build step. See
[TensorRT Backend Removal](../tensorrt-backend-removal.md).

## Debug Execution Errors

If problems come from plugins, try setting the environment variable `CUDA_LAUNCH_BLOCKING=1` to make kernels launch synchronously with their return status checked immediately.

If problems come from runtime-shape of the input tensors, double-check the shape (rank and length of each rank) and location (CPU / GPU) of input tensors for the engine obey the build-time setting.

For example, one possible reason of getting the error information like below is, we use mismatched configuration between engine building and running, including code change (update of repo or users' rewriting), too large or too small input shape, etc..

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
reduce other build-time memory pressure (the legacy `--gpt_attention_plugin` `trtllm-build` flag was removed with the TensorRT backend; see [TensorRT Backend Removal](../tensorrt-backend-removal.md)).

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

For example: `mpirun -n 1 trtllm-serve <hf_model> ...` (the legacy `examples/models/core/gpt/build.py` path was removed with the TensorRT backend).

It's critical that it's always `-n 1` regardless of how many GPUs are being used. If you'd use `-n 2` for a 2 GPU program it will not work. `mpirun` here isn't being used to orchestrate multiple processes, but to invoke the right environment on SLURM. The internal MPI implementation deals with spawning the additional processes.
