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

2. In `examples/gpt/build.py`, we mark it as a TensorRT network output:

```python
    with net_guard(network):
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        inputs = tensorrt_llm_gpt.prepare_inputs(args.max_batch_size,
                                                 args.max_input_len,
                                                 args.max_output_len, True,
                                                 args.max_beam_width)
        tensorrt_llm_gpt(*inputs)

        # mark as TRT network output
        # ----------------------------------------------------------------
        for k, v in tensorrt_llm_gpt.named_network_outputs():
            network._mark_output(v, k,
                                 tensorrt_llm.str_dtype_to_trt(args.dtype))
        # ----------------------------------------------------------------
```


3. Build the TensorRT engine of the model:

```bash
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin
```

4. Print the intermediate output tensors:


In `examples/gpt/run.py`, we open the debug mode:

```python
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=True)
```

In `tensorrt_llm/runtime/generation.py`, we print the debug info:

```python
        if step == 0:
            ...
            ctx_shape, ctx_buffer = self._get_context_shape_buffer(
                    input_ids, max_input_length, step,
                    input_lengths, position_ids, last_token_ids, attention_mask,
                    this_src_cache_indirection)
            self.runtime._set_shape(context, ctx_shape)
            self.runtime._set_buffer(context, ctx_buffer)
            # -------------------------------------------
            debug_buffer = ctx_buffer
            # -------------------------------------------

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.runtime._run(context, stream)
        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        if self.debug_mode:
            torch.cuda.synchronize()
            # -------------------------------------------
            if step == 0:
                print(debug_buffer.keys())
            print(step, debug_buffer['layers.6.mlp_output'])
            # -------------------------------------------

        if not step == self.max_new_tokens - 1:
            ...
            next_step_shape, next_step_buffer = self._get_next_step_shape_buffer(
                    batch_size, scfg.num_beams, max_input_length, step,
                    input_lengths, position_ids, last_token_ids,
                    attention_mask, next_src_cache_indirection)
            self.runtime._set_shape(next_context, next_step_shape)
            self.runtime._set_buffer(next_context, next_step_buffer)
            # -------------------------------------------
            debug_buffer = next_step_buffer
            # -------------------------------------------

```

Then, we will see the tensor values:

```bash
python run.py --max_output_len=8
dict_keys(['input_ids', 'logits', 'input_lengths', 'position_ids', 'last_token_ids', 'max_input_length', 'cache_indirection', 'past_key_0', 'past_value_0', 'present_key_0', 'present_value_0', 'past_key_1', 'past_value_1', 'present_key_1', 'present_value_1', 'past_key_2', 'past_value_2', 'present_key_2', 'present_value_2', 'past_key_3', 'past_value_3', 'present_key_3', 'present_value_3', 'past_key_4', 'past_value_4', 'present_key_4', 'present_value_4', 'past_key_5', 'past_value_5', 'present_key_5', 'present_value_5', 'past_key_6', 'past_value_6', 'present_key_6', 'present_value_6', 'past_key_7', 'past_value_7', 'present_key_7', 'present_value_7', 'past_key_8', 'past_value_8', 'present_key_8', 'present_value_8', 'past_key_9', 'past_value_9', 'present_key_9', 'present_value_9', 'past_key_10', 'past_value_10', 'present_key_10', 'present_value_10', 'past_key_11', 'past_value_11', 'present_key_11', 'present_value_11', 'past_key_12', 'past_value_12', 'present_key_12', 'present_value_12', 'past_key_13', 'past_value_13', 'present_key_13', 'present_value_13', 'past_key_14', 'past_value_14', 'present_key_14', 'present_value_14', 'past_key_15', 'past_value_15', 'present_key_15', 'present_value_15', 'past_key_16', 'past_value_16', 'present_key_16', 'present_value_16', 'past_key_17', 'past_value_17', 'present_key_17', 'present_value_17', 'past_key_18', 'past_value_18', 'present_key_18', 'present_value_18', 'past_key_19', 'past_value_19', 'present_key_19', 'present_value_19', 'past_key_20', 'past_value_20', 'present_key_20', 'present_value_20', 'past_key_21', 'past_value_21', 'present_key_21', 'present_value_21', 'past_key_22', 'past_value_22', 'present_key_22', 'present_value_22', 'past_key_23', 'past_value_23', 'present_key_23', 'present_value_23', 'sequence_length', 'past_key_value_length', 'layers.0.mlp_output', 'layers.1.mlp_output', 'layers.2.mlp_output', 'layers.3.mlp_output', 'layers.4.mlp_output', 'layers.5.mlp_output', 'layers.6.mlp_output', 'layers.7.mlp_output', 'layers.8.mlp_output', 'layers.9.mlp_output', 'layers.10.mlp_output', 'layers.11.mlp_output', 'layers.12.mlp_output', 'layers.13.mlp_output', 'layers.14.mlp_output', 'layers.15.mlp_output', 'layers.16.mlp_output', 'layers.17.mlp_output', 'layers.18.mlp_output', 'layers.19.mlp_output', 'layers.20.mlp_output', 'layers.21.mlp_output', 'layers.22.mlp_output', 'layers.23.mlp_output'])
0 tensor([[[ 0.0295, -0.0256, -0.0780,  ..., -0.0562, -0.0241,  0.0273],
         [-0.0089,  0.5882,  0.1989,  ..., -1.0464, -0.6305,  0.5967],
         [-0.8793,  0.1056,  0.7083,  ...,  0.0889,  1.0714, -0.2931],
         ...,
         [ 0.1209, -0.0886, -0.5927,  ..., -0.1048, -0.3437,  0.1085],
         [-1.0752, -0.0739,  0.6156,  ...,  0.3454,  0.3014,  0.2653],
         [-0.7126,  0.9685, -0.1145,  ..., -0.0084,  0.9521,  0.1425]]],
       device='cuda:0')
1 tensor([[[-0.2129,  0.5879,  0.8172,  ...,  0.7892, -0.6887,  0.6063]]],
       device='cuda:0')
2 tensor([[[ 0.4184, -0.0066,  1.3895,  ..., -0.9023, -0.0686, -0.2831]]],
       device='cuda:0')
3 tensor([[[-0.7935, -0.5085, -0.1696,  ..., -0.5839, -0.1375, -0.0078]]],
       device='cuda:0')
4 tensor([[[-0.0810,  0.1262, -0.6260,  ..., -0.1065, -0.0529,  0.7143]]],
       device='cuda:0')
5 tensor([[[-0.3322, -0.8835,  0.3427,  ...,  0.8159, -0.0622,  1.2327]]],
       device='cuda:0')
6 tensor([[[-0.2217, -0.2057, -0.1475,  ..., -0.3545, -0.1673,  0.1131]]],
       device='cuda:0')
7 tensor([[[ 0.1268, -0.1570,  0.3972,  ..., -0.8213, -0.3282, -0.8672]]],
       device='cuda:0')
Input: Born in north-east France, Soyer trained as a
Output:  chef before moving to London in the early
```

## Debug execution errors

- If you use plugins, use can set the environment variable `CUDA_LAUNCH_BLOCKING=1` so that kernels are launch synchronously, with their return status checked immediately.
- If you see memory errors, make sure that the engine inputs respect the build-time shapes and that they reside **on the correct device** (CPU/GPU).
