# TensorRT-LLM Build Workflow

## Overview


The build workflow contains two major steps.

1. Create TensorRT-LLM models from existing model checkpoints exported by the training framework.
2. Build the TensorRT-LLM models to TensorRT-LLM engines.

To generalize the TensorRT-LLM optimization features to all models, and to share the same workflow between different models for TensorRT-LLM users, TensorRT-LLM has conventions about how the models shall be defined and how the models shall be imported.

TensorRT-LLM checkpoint convention is documented in [](checkpoint.md) and all decoder-only models had been migrated to adopt the convention. Model-specific convert_checkpoint.py scripts are shipped as source code in example directories, and a trtllm-build CLI tool had been added. However, there are some disadvantages of providing convert checkpoint scripts outside the core TensorRT-LLM lib as example:

1. TensorRT-LLM evolves so quickly that the model's definition code might have changed for better performance; which means the `convert_checkpoint.py` is out of date.


2. TensorRT-LLM is creating a new set of high-level APIs which handle model conversion, engine building, and inference in one class for easier-of-use. Thus, the high-level APIs need to call the weights conversion code, which shall be part of TensorRT-LLM core lib, not the example. And the conversion code of different models shall have same interface such that the high-level APIs do not need to add many ad-hoc code for different models.

To mitigate these issues, the model specific `convert_checkpoint.py` scripts are being refactored. Most of the conversion code will be moved into core lib, sitting next to the model definition. Refer to `tensorrt_llm/models/llama/` as an example. There is a new set of APIs for importing models and converting weights. The 0.9 release refactored the LLaMA model class to adopt the new APIs, others models' refactor work is ongoing.


## Conversion APIs

The API for weight conversion of the LLaMA model looks like this. A `TopModelMixin` class is introduced, `from_hugging_face()` interface is declared, the `LLaMAForCausalLM` class inherits `TopModelMixin` (not direct parent class, but in its base class hierarchy), and implements the interface.

```python
class TopModelMixin
    @classmethod
    def from_hugging_face(cls,
                          hf_model_dir: str,
                          dtype: Optional[str] = 'float16',
                          mapping: Optional[Mapping] = None,
                          **kwargs):
        raise NotImplementedError("Subclass shall override this")

# TopModelMixin is in the part of base class hierarchy
class LLaMAForCausalLM (DecoderModelForCausalLM):
    @classmethod
    def from_hugging_face(cls,
             hf_model_dir,
             dtype='float16',
             mapping: Optional[Mapping] = None) -> LLaMAForCausalLM:
        # creating a TensorRT-LLM llama model object
        # converting HuggingFace checkpoint to TensorRT-LLM expected weights dict
        # Load the weights to llama model object
```


Then, in the convert_checkpoint.py script in the
[`examples/models/core/llama/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama/) directory of the GitHub repo,
the logic can be greatly simplified. Even if the model definition code of TensorRT-LLM LLaMA class is changed due to some reason, the `from_hugging_face` API will keep the same, thus the existing workflow using this interface will not be affected.


```python
#other args omitted for simplicity here.
llama = LLaMAForCausalLM.from_hugging_face(model_dir, dtype, mapping=mapping)
llama.save_checkpoint(output_dir, save_config=(rank==0))
```

The `from_hugging_face` API does not save the checkpoint into disk intentionally, instead it returns an in-memory object. Call `save_checkpoint` to save the models. This keeps the flexibility and makes the flow of convert->build in one process faster. Typically, saving and loading disk for large models are slower and thus should be avoided.


Since LLaMA models were also released with different formats, such as the Meta checkpoint, the `LLaMAForCausalLM` class has a `from_meta_ckpt` function for that. This function is not declared in the `TopModelMixin` class due to it being LLaMA specific, and therefore, other models don't use it.


In the 0.9 release, only LLaMA is refactored. Since popular LLaMA (and its variants) models are released by Hugging Face and Meta checkpoint formats, only these two functions are implemented.


In future releases, there might be `from_jax`, `from_nemo`, `from_keras` or other factory methods for different training checkpoints added.
For example, the Gemma 2B model and the convert_checkpoint.py file in the [`examples/models/core/gemma`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma/)
directory support JAX and Keras formats in addition to Hugging Face. The model developers can choose to implement **any subset** of these factory methods for the models they contributed to TensorRT-LLM.


For some formats which are not supported by TensorRT-LLM model developers, you still have the freedom to implement your own weights conversion outside the core lib; the flow will look like this:


```python
config = read_config_from_the_custom_training_checkpoint(model_dir)
llama = LLaMAForCausalLM(config)

# option 1:
# Create a weights dict and then calls LLaMAForCausalLM.load
weights_dict = convert_weights_from_custom_training_checkpoint(model_dir)
llama.load(weights_dict)

# option 2:
# Internally assign the model parameters directly
convert_and_load_weights_into_trtllm_llama(llama, model_dir)
# Use the llama object as usual, to save the checkpoint or build engines
```

Though there are some limitations and pitfalls of doing these custom weights loading, if the model definition is inside TensorRT-LLM core lib, and the weights loading/conversion are outside the core lib, the conversion code might need to be updated when new TensorRT-LLM is released.


## Quantization APIs

TensorRT-LLM relies on NVIDIA Modelopt toolkit to support some of the quantization like: FP8, W4A16_AWQ, W4A8_AWQ, while it also has some its own quantization implementation for Smooth Quant, INT8 KV cache, and INT4/INT8 weight only.


In TensorRT-LLM 0.8 version:

* For Modelopt-supported quantization algorithms, a standalone script,
  [example/quantization/quantize.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/quantize.py)
  can export TensorRT-LLM checkpoints, and the trtllm-build command needs to be executed to build the checkpoints to engines.

* For the non-Modelopt quantization algorithms, users need to use the per-model convert_checkpoint.py scripts to export TensorRT-LLM checkpoints.

Use the `quantize()` interface to unify the different quantization flows. The default implementation is added in the `PretrainedModel` class.


```python
class PretrainedModel:
    @classmethod
    def quantize(
        cls,
        hf_model_dir,
        output_dir,
        quant_config: QuantConfig,
        mapping: Optional[Mapping] = None): #some args are omitted here
        # Internally quantize the given hugging face models using Modelopt
        # and save the checkpoint to output_dir
```

* The default implementation only handles the Modelopt supported quantization. The LLaMA class then inherits this `PretrainedModel` and dispatches the Modelopt quantization to the super class's default implementation.
* The model developer raises errors in the sub-class implementation if the new model is not supported by Modelopt yet.


```python
class LLaMAForCausalLM:
    @classmethod
    def quantize(
        cls,
        hf_model_dir,
        output_dir,
        quant_config: QuantiConfig,
        mapping: Optional[Mapping] = None): #some args are omitted here
        use_modelopt_quantization = ... # determine if to use Modelopt or use native
        if use_modelopt_quantization:
            super().quantize(hf_model_dir,
                             output_dir,
                             quant_config)
        else:
            # handles TensorRT-LLM native model specific quantization
            # or raise exceptions if not supported
```


The `quantize` API is designed to take multi-GPU resources internally to make quantization. For example, a LLaMA 70B BF16 takes 140G memory, if we make FP8 quantization, then, another 70G is needed. So, we need at least 210G, 4 * A100(H100) is needed to quantize the LLaMA 70B model. If you want to call `quantize` API inside a MPI program, be cautious and ensure the quantize API is only called by rank 0.


Usage of the `quantize` API in an MPI program looks like this, only rank 0 calls it. In an non-MPI program, the `if rank == 0` and the `mpi_barrier()` are not needed.

```python
quant_config = QuantConfig()
quant_config.quant_algo = quant_mode.W4A16_AWQ
mapping = Mapping(world_size=tp_size, tp_size=tp_size)
if rank == 0:
    LLaMAForCausalLM.quantize(hf_model_dir,
                          checkpoint_dir,
                          quant_config=quant_config)
mpi_barrier() # wait for rank-o finishes the quantization
llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir, rank)
engine = build(llama, build_config)
engine.save(engine_dir)
```


The `examples/quantization/quantize.py` is kept for backward compatibility.


## Build APIs


The `tensorrt_llm.build` API builds the TensorRT-LLM model object to TensorRT-LLM engine. This new API replaced the older flow: creating a builder, creating a network object, tracing the model to the network, and building TensorRT engines.
The usage of this API looks like this:

```python
llama = ... # create LLaMAForCausalLM object
build_config = BuildConfig(max_batch_size=1)
engine = tensorrt_llm.build(llama, build_config)
engine.save(engine_dir)
```


The Llama object can be created by any method mentioned in the [](#conversion-apis) or [](#quantization-apis) sections.


The `trtllm-build` CLI tool is a thin wrapper around this `tensorrt_llm.build` API. The flags of the CLI tool are kept close to the fields of the `BuildConfig` class.


If a model were to be saved into disk and then built to the engine later, TensorRT-LLM provides a `from_checkpoint` API to deserialize the checkpoint.

```python
## TensorRT-LLM code
class PretrainedModel:
    @classmethod
    def from_checkpoint(cls,
                    ckpt_dir: str,
                    rank: int = 0,
                    config: PretrainedConfig = None):
        # Internally load the model weights from a given checkpoint directory
```


The `from_checkpoint` API is called to deserialize the checkpoint to a model object.  The `tensorrt_llm.build` API can be called to build the engine.


```python
llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)
engine = build(llama, build_config)
engine.save(engine_dir)
```

## CLI Tools

All the weights conversion, quantization, and build APIs mentioned above have corresponding CLI tools for convenience.

* Model specific `convert_checkpoint.py` scripts are inside the `examples/<model xxx>/` folder.
* A unified quantization script is inside the `examples/quantization/quantize.py` and can be shared by all **supported** models.
* A `trtllm-build` CLI tool builds all models from TensorRT-LLM checkpoint.

Refer to the following considerations for the CLI tools:

* These scripts and tools should be used for scripting. Do not import the Python functions/class defined in these tools. TensorRT-LLM does not promise the content of these scripts can be compatible with previous versions. The options of these tools may also be changed when it’s not avoidable.

* These scripts in the example folder may use TensorRT-LLM internal/unstable APIs, which is not guaranteed to work if the examples’ version and the TensorRT-LLM install version are mismatched. There are some GitHub issues caused by version mismatch.
    - https://github.com/NVIDIA/TensorRT-LLM/issues/1293
    - https://github.com/NVIDIA/TensorRT-LLM/issues/1252
    - https://github.com/NVIDIA/TensorRT-LLM/issues/1079

    You should always install the same TensorRT-LLM version specified in `examples/<model xxx>/requirements.txt`.

* In the future, the per-model conversion script may or may not be unified to one single script shared by models, given the nature of different models’ attributes may be different. However, the TensorRT-LLM team will try to make sure the flags for the same feature are consistent between different scripts.

* The TensorRT-LLM team encourages use of the new low-level conversion/quantization/build API instead of these scripts. The conversion APIs will be added model-by-model gradually, which may span a few releases.
