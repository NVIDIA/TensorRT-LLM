(add-model)=

# Adding a Model

This document describes how to add a typical decoder-only model in TensorRT-LLM.

## Step 1. Write Modeling Part

TensorRT-LLM provides different levels of APIs:

- Low-level functions, for example, `concat`, `add`, and `sum`.
- Basic layers, such as, `Linear` and `LayerNorm`.
- High-level layers, such as, `MLP` and `Attention`.
- Base class for typical decoder-only models, such as, `DecoderModelForCausalLM`.

1. Create a model directory in `tensorrt_llm/models`, for example `my_model`.
2. Write a `model.py` with TensorRT-LLM's APIs

```python
class MyDecoderLayer(Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        self.layer_idx = layer_idx
        self.config = config
        self.input_layernorm = LayerNorm(...)
        self.attention = Attention(...)
        self.post_layernorm = LayerNorm(...)
        self.mlp = MLP(...)

    def forward(self, hidden_states, ...):
        # decoder layer forward
        return hidden_states

class MyModel(Module):
    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.vocab_embedding = Embedding(...)
        self.layers = DecoderLayerList(MyDecoderLayer, config)
        self.ln_f = LayerNorm(...)

    def forward(self, input_ids, ...):
        # model forward
        return hidden_states


class MyModelForCausalLM(DecoderModelForCausalLM):
    def __init__(self, config: PretrainedConfig):
        transformer = MyModel(config)
        lm_head = ColumnLinear(...)
        super().__init__(config, transformer, lm_head)
```


## Step 2. Implement Weight Conversion

The weights from source framework need to be converted and bound to the new added TensorRT-LLM model. Here is an example of converting HuggingFace weights:

```python
class MyModelForCausalLM(DecoderModelForCausalLM):
    @classmethod
    def from_hugging_face(
            cls,
            hf_model_dir,
            dtype='float16',
            mapping: Optional[Mapping] = None) -> MyModelForCausalLM
        # create a TensorRT-LLM MyModelForCausalLM model object
        # convert HuggingFace checkpoint to TensorRT-LLM expected weights dict
        # load the weights to MyModelForCausalLM object
```

It's optional to develop a `convert_checkpoint.py` script in the `examples/my_model/` directory for the convenience of offline weights conversion.

## Step 3. Register New Model

Please register the new model class `MyModelForCausalLM` in `tensorrt_llm/models/__init__.py`.

## Step 4. Verify New Model

At last, let's verify the new model. The typical commands are as following:

```bash
cd examples/my_model/

python convert_checkpoint.py --model_dir hf_model_dir --output_dir tllm_ckpt_dir

trtllm-build --checkpoint_dir tllm_ckpt_dir --output_dir tllm_engine_dir

# try the model with a single prompt
python ../run.py --engine_dir tllm_engine_dir --tokenizer_dir hf_model_dir --input_text "Born in north-east France, Soyer trained as a"
# run summarization task
python ../summarize.py --engine_dir tllm_engine_dir --hf_model_dir hf_model_dir --test_trt_llm
```

## Reference

It's recommended to read the workflow[./workflow.md] and checkpoint[./checkpoint.md] documents for more details.
