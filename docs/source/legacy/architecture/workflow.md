# TensorRT-LLM Build Workflow

```{caution}
The legacy TensorRT backend has been removed and is no longer supported. This page is retained for cross-reference only.
```

> [!WARNING]
> This page describes the **legacy** TensorRT engine-build workflow.
> For new projects, use [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
> or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html) instead.

## Overview

Historically, the **legacy** TensorRT build workflow contained two major steps:

1. Create TensorRT-LLM checkpoint models from existing checkpoints exported by the training framework.
2. Build those TensorRT-LLM models into TensorRT engines (via `trtllm-build` / `tensorrt_llm.build`).

Both steps, the per-model `convert_checkpoint.py` scripts, the `tensorrt_llm/models/<name>/` convert packages, and the `trtllm-build` CLI were **removed with the TensorRT backend**. Do not run convert/build commands from this page. For serving today, use [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html) or the [LLM Python API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html) with a Hugging Face checkpoint (see also [TensorRT Backend Removed](../tensorrt-backend-removal.md) and [](checkpoint.md)).

The remainder of this page is retained only to document how that legacy convert→build flow used to work and why conversion code was moved toward the core lib before the backend was retired.


## Conversion APIs (legacy)

The conversion APIs below (`TopModelMixin.from_hugging_face`, `LLaMAForCausalLM`, and the deleted `tensorrt_llm/models/llama/` package) were part of the removed TensorRT engine workflow. They are **not** present on current `main`.

Historically, the weight-conversion API for the LLaMA model looked like this. A `TopModelMixin` class declared `from_hugging_face()`; `LLaMAForCausalLM` inherited `TopModelMixin` (not a direct parent, but in its base class hierarchy) and implemented the interface:

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


Historically, a thin `convert_checkpoint.py` wrapper lived under the deleted `examples/models/core/llama/` tree and called this API. That example directory and script are gone with the TensorRT backend; the `from_hugging_face` / `save_checkpoint` helpers are likewise absent.


```python
#other args omitted for simplicity here.
llama = LLaMAForCausalLM.from_hugging_face(model_dir, dtype, mapping=mapping)
llama.save_checkpoint(output_dir, save_config=(rank==0))
```

The `from_hugging_face` API does not save the checkpoint into disk intentionally, instead it returns an in-memory object. Call `save_checkpoint` to save the models. This keeps the flexibility and makes the flow of convert->build in one process faster. Typically, saving and loading disk for large models are slower and thus should be avoided.


Since LLaMA models were also released with different formats, such as the Meta checkpoint, the `LLaMAForCausalLM` class has a `from_meta_ckpt` function for that. This function is not declared in the `TopModelMixin` class due to it being LLaMA specific, and therefore, other models don't use it.


In the 0.9 release, only LLaMA is refactored. Since popular LLaMA (and its variants) models are released by Hugging Face and Meta checkpoint formats, only these two functions are implemented.


In future releases, there might be `from_jax`, `from_nemo`, `from_keras` or other factory methods for different training checkpoints added.
Historically, Gemma also shipped a per-model `convert_checkpoint.py` under [`examples/models/core/gemma`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/gemma/) (that script is gone; the directory now only documents the PyTorch path). Additional factory methods (`from_jax`, `from_nemo`, …) were planned for the TensorRT convert packages and were never completed before the backend was removed.


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

* For Modelopt-supported quantization algorithms, a standalone script
  `examples/quantization/quantize.py` historically exported TensorRT-LLM checkpoints, and `trtllm-build` built those checkpoints into engines. That `quantize.py` entry point is **no longer in the tree** (the `examples/quantization/` folder now documents loading pre-quantized HF checkpoints on the PyTorch backend).

* For non-Modelopt quantization algorithms, users historically used the per-model `convert_checkpoint.py` scripts (also removed) to export TensorRT-LLM checkpoints.

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
mpi_barrier() # wait for rank-0 to finish the quantization
llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir, rank)
engine = build(llama, build_config)
engine.save(engine_dir)
```


The `examples/quantization/quantize.py` helper was kept for a while for backward compatibility and has since been removed with the TensorRT backend.


## Build APIs (legacy)

The `tensorrt_llm.build` API (and the removed `tensorrt_llm/builder.py` / `BuildConfig` engine-build path) built a TensorRT-LLM model object into a TensorRT engine. It replaced an even older flow that created a builder, network object, traced the model, and built engines. That API is **not** shipped on current `main` (`setup.py` console scripts are only `trtllm-bench`, `trtllm-serve`, and `trtllm-eval`).

Historically, usage looked like this:

```python
llama = ... # create LLaMAForCausalLM object
build_config = BuildConfig(max_batch_size=1)
engine = tensorrt_llm.build(llama, build_config)
engine.save(engine_dir)
```


The Llama object can be created by any method mentioned in the [](#conversion-apis-legacy) or [](#quantization-apis) sections.


The removed `trtllm-build` CLI was a thin wrapper around that `tensorrt_llm.build` API; its flags mirrored the fields of the legacy `BuildConfig` class.


If a model had been saved to disk and then built to an engine later, the legacy stack provided a `from_checkpoint` API to deserialize the checkpoint.

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


Historically, `from_checkpoint` deserialized the checkpoint to a model object, and `tensorrt_llm.build` built the engine.


```python
llama = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)
engine = build(llama, build_config)
engine.save(engine_dir)
```

## CLI Tools (removed)

The conversion / quantization / build CLIs documented historically on this page are **gone** with the TensorRT backend:

* Per-model `convert_checkpoint.py` scripts under `examples/<model>/` — deleted (many model example trees were removed or reduced to PyTorch READMEs).
* Unified `examples/quantization/quantize.py` — deleted.
* `trtllm-build` — no longer a console script (`setup.py` ships only `trtllm-bench`, `trtllm-serve`, `trtllm-eval`).

Do not import or run those tools. For current workflows, load Hugging Face (or other supported) checkpoints directly with `trtllm-serve` / the LLM API, and evaluate with `trtllm-eval`. See [TensorRT Backend Removed](../tensorrt-backend-removal.md).
