# How to debug

This document describes how to debug in TensorRT-LLM.

## Overview

Usually, we want to print the intermediate tensor values when debugging a TensorRT-LLM model.
TensorRT-LLM obeys define-and-run paradigm, we should mark the interested intermediate tensors as the network outputs.
Then, we print the values at runtime.

## Debug on unit tests

1. Register the intermediate tensors as the network outputs with `register_network_output` API.


```python
class MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 bias=True,
                 tp_group=None,
                 tp_size=1):
        super().__init__()
        self.fc = tensorrt_llm.layers.ColumnLinear(hidden_size,
                                                   ffn_hidden_size,
                                                   bias=bias,
                                                   tp_group=tp_group,
                                                   tp_size=tp_size,
                                                   gather_output=False)
        self.proj = tensorrt_llm.layers.RowLinear(ffn_hidden_size,
                                                  hidden_size,
                                                  bias=bias,
                                                  tp_group=tp_group,
                                                  tp_size=tp_size)

    def forward(self, hidden_states):
        inter = self.fc(hidden_states)
        inter = tensorrt_llm.functional.relu(inter)
        # Here, we want to print the tensor value after relu
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

Here is the [full example](source:tests/test_debugging_api.py).


## Debug on E2E models

Here is an example to print the values of the MLP output tensor in the GPT model.


1. In `tensorrt_llm/models/gpt/model.py`, we register the MLP output tensor:

```python
        hidden_states = residual + attention_output.data

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        # register as model output
        # ------------------------------------------------------
        self.register_network_output('mlp_output', hidden_states)
        # ------------------------------------------------------

        hidden_states = residual + hidden_states
```

2. Build the TensorRT engine of the model:

When building engines with `trtllm-build`, enable the `--enable_debug_output` option.

```bash
cd examples/gpt

# Download hf gpt2 model
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# Convert to TensorRT-LLM checkpoint
python3 convert_checkpoint.py --model_dir gpt2 \
        --dtype float16 \
        --output_dir gpt2/trt_ckpt/fp16/1-gpu

# Build TensorRT-LLM engines with --enable_debug_output
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --enable_debug_output \
        --output_dir gpt2/trt_engines/fp16/1-gpu
```

3. Print the intermediate output tensors:

In `tensorrt_llm/runtime/generation.py`, we print the debug info:

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

Then, run `../run.py` with `--debug_mode` and `--use_py_session`:

```bash
python3 ../run.py --engine_dir gpt2/trt_engines/fp16/1-gpu \
        --tokenizer_dir gpt2 \
        --max_output_len 8 \
        --debug_mode \
        --use_py_session
```

We will see the tensor values:

```
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

## Debug execution errors

- If you use plugins, use can set the environment variable `CUDA_LAUNCH_BLOCKING=1` so that kernels are launch synchronously, with their return status checked immediately.
- If you see memory errors, make sure that the engine inputs respect the build-time shapes and that they reside **on the correct device** (CPU/GPU).
