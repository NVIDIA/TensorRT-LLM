# TRT-LLM Low-Level API Examples

The folder's example scripts are designed to demonstrate the usage of low-level workflow (compared to the LLM high-level API, which builds and runs all in one step) for the TRT-LLM.

* (1) Use `LLaMAForCausalLM.from_hugging_face` to create a TRT-LLM model from a Hugging Face model. Alternatively, use the `LLaMAForCausalLM.quantize` API to save a quantized TRT-LLM model from a Hugging Face model. The `LLaMAForCausalLM` can be replaced by other model classes which implemented the APIs, including the following as of version 0.12. The list is not exclusive, and the classes implementing these APIs can be added in the future.
    - `GPTForCausalLM`
    - `PhiForCausalLM`
    - `Phi3ForCausalLM`
    - `QWenForCausalLM`
    - `FalconForCausalLM`
    - `GPTJForCausalLM`

* (2) Use the low-level `tensorrt_llm.build` API to build the TRT-LLM model into a TRT-LLM engine.
* (3) Use the `GenerationExecutor` API to load the engine and start generating.

The high-level `LLM` API provides the most simplified workflow. The APIs in this example are backbones of the high-level API and provide more flexibility for advanced users. It's recommended to use the high-level API whenever possible.

See [workflow](../../docs/source/architecture/workflow.md) for the design and explanation of these APIs.

The `<hf llama dir>` placeholder in the following examples is the path to your local clone of the LLaMA model. For example, if you clone the model using `git clone https://huggingface.co/meta-llama/Llama-2-7b-hf /tmp/llama-2-7b-hf`, then you should replace `<hf llama dir>` with `/tmp/llama-2-7b-hf` below.


## Single GPU
```bash
python ./llama.py --hf_model_dir <hf llama dir> --engine_dir ./llama.engine
```

## Multi-GPU

Using multiple GPUs with tensor parallelism to build and run LLaMA, and then generate text on a pre-defined input.
```bash
python ./llama_multi_gpu.py --hf_model_dir <hf llama path> --engine_dir ./llama.engine.tp2 --tp_size 2
```

## Quantization
Using AWQ INT4 weight only algorithm to quantize the given hugging llama model first and save as TRT-LLM checkpoint, and then build TRT-LLM engine from that checkpoint and serve

```bash
python ./llama_quantize.py --hf_model_dir <hf llama path> --cache_dir ./llama.awq/
```


## AutoModelForCausalLM

The API `tensorrt_llm.AutoModelForCausalLM` can read from a Hugging Face model directory, find the correct TRT-LLM model class and dispatch the `from_hugging_face` mothod to the correct TRT-LLM class.

The following code snippets demonstrated the usage of the `AutoModelForCausalLM` class.

```python
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp, pp_size=pp)
    trtllm_model = AutoModelForCausalLM.from_hugging_face(hf_model_dir, mapping=mapping)
    engine = build(trtllm_model, build_config)
    executor = GenerationExecutor.create(engine)
```

## AutoConfig

The API `tensorrt_llm.AutoConfig` can read the configuration from a Hugging Face model directory, find and return the correct TRT-LLM configuration class if it's supported, and raise a `NotImplementedError` if not supported. This API is useful when one needs to create a TRT-LLM model object using dummy weights, for things like workflow testing, benchmarks, without reading the real weights from storage, since reading the weights for large models can take significant amount of time. The usage looks like below snippets:

```python
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp, pp_size=pp)
    trtllm_config = AutoConfig.from_hugging_face(hf_model_dir, dtype='float16', mapping=mapping)

    # Use the __init__ constructor directly to create a TRT-LLM model object
    # instead of using from_hugging_face class method, since from_hugging_face will read the weights
    trtllm_model_fake_weights = AutoModelForCausalLM.get_trtllm_model_class(hf_model_dir)(trtllm_config)
```
