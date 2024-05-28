# High-level API
We are working on a Python high-level API(HLAPI) for LLM workflow, which is still in incubation and may change later.
Here we show you a preview of how it works and how to use it.

Note that the APIs are not stable and only support the LLaMA model. We appreciate your patience and understanding as we improve this API.

Please install the required packages first:

```bash
pip install -r requirements.txt
```

You can refer to [llm_examples.py](llm_examples.py) for all of the examples, and run it with the [run_examples.py](./run_examples.py) script, the command is as follows:

```sh
# To run examples with single GPU:
python3 ./run_examples.py run_single_gpu --model_dir <llama-model-path>

# Run the multi-GPU examples
python3 ./run_examples.py run_multi_gpu --model_dir <llama-model-path>

# Run the quantization examples
python3 ./run_examples.py run_quant --model_dir <llama-model-path>
```

For 7B, 13B models those could be held in a single GPU, it should run all the examples automatically and print the results.

For larger models, such as LLaMA v2 70B, at least two H100/A100 cards are required, the following command could be used to start a parallel task with Tensor Parallelism enabled.

``` sh
python3 llm_examples.py --task run_llm_on_tensor_parallel \
    --prompt="<prompt>" \
    --hf_model_dir=<llama-model-path>
```

## Model preparation
The HLAPI supports three kinds of model formats:

1. HuggingFace models
2. TensorRT-LLM engine built by trtllm-build tool or saved by the HLAPI
3. TensorRT-LLM checkpoints, converted by `convert_checkpoint.py` in examples

All kinds of models could be used directly by the HLAPI, and the `ModelConfig(<any-model-path>)` could accept any kind of them.

Let's elaborate on the preparation of the three kinds of model formats.

### Option 1: From HuggingFace models

Given its popularity, the TRT-LLM HLAPI chooses to support HuggingFace format as one of the start points, to use the HLAPI on LLaMA models, you need to run the following conversion script provided in [transformers/llama](https://huggingface.co/docs/transformers/main/model_doc/llama) or [transformers/llama2](https://huggingface.co/docs/transformers/main/model_doc/llama2) to convert the Meta checkpoint to HuggingFace format.

For instance, when targeting the LLaMA2 7B model, the official way to retrieve the model is to visit the [LLaMA2 7B model page](https://huggingface.co/transformers/llama2-7B), normally you need to submit a request for the model file.
For a quick start, you can also download the model checkpoint files directly from [modelscope.cn](https://www.modelscope.cn/models/shakechen/Llama-2-7b/files), the command is as follows:

``` sh
git clone https://www.modelscope.cn/shakechen/Llama-2-7b.git
```

To convert the checkpoint files, a script from transformers is required, thus please also clone the transformers repo with the following code:

```sh
git clone https://github.com/huggingface/transformers.git
```

Finally, the command to convert the checkpoint files to HuggingFace format is as follows:

``` sh
python <transformers-dir>/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir Llama-2-7b --model_size 7B --output_dir llama-hf-7b
```

That should produce a HuggingFace format model in `./llama-hf-7b`, which could be used by the HLAPI.

### Option 2: From TensorRT-LLM engine
There are two ways to build the TensorRT-LLM engine:

1. You can build the TensorRT-LLM engine from the HuggingFace model directly with the `trtllm-build` tool, and save the engine to disk for later use.  Please consult the LLaMA's [README](../llama/README.md).
2. Use the HLAPI to save one:

```python
config = ModelConfig(<model-path>)
llm = LLM(config)

# Save engine to local disk
llm.save(<engine-dir>)
```

### Option 3: From TensorRT-LLM checkpoint
In each model example, there is a `convert_checkpoint.py` to convert third-party models to TensorRT-LLM checkpoint for further usage.
The HLAPI could seamlessly accept the checkpoint, and build the engine in the backend.
For step-by-step guidance on checkpoint conversion, please refer to the LLaMA's [README](../llama/README.md).


## Basic usage
To use the API, import the `LLM` and `ModelConfig` from the `tensorrt_llm` package and create an LLM object with a HuggingFace model directly.
For example:

``` python
from tensorrt_llm import LLM, ModelConfig

config = ModelConfig(model_dir=<llama_model_path>)
llm = LLM(config)
```

It will trigger TRT-LLM engine building in the backend, and create a HuggingFace tokenizer by default to support an end-to-end generation.

To generate text, use the `generate` method of the `LLM` object directly with a batch of prompts, for example:

``` python
prompts = ["To tell a story"]
for output in llm.generate(prompts):
    print(output)
```

The output might be something like:

``` python
GenerationOutput(text="with a picture.\nI'm a writer, but I'm also a photographer.")
```

You can also dump the runtime engine to disk, and load from the engine file directly in the next run to save the engine building time from the HuggingFace model.

``` python
# dump the llm
llm.save(<engine-path>)

# next time
config = ModelConfig(model_dir=<engine-path>)
llm = LLM(config)
```

In other words, the `model_dir` could accept either a HugggingFace model, a built TensorRT-LLM engine, or a TensorRT-LLM checkpoint, and the `LLM()` will do the rest work silently for end-to-end execution.

## Quantization

By simply setting several flags in the ModelConfig, TensorRT-LLM can quantize the HuggingFace model automatically. For example, to perform an Int4 AWQ quantization, the following code will trigger the model quantization.


``` python
from tensorrt_llm.quantization import QuantAlgo

config = ModelConfig(model_dir=<llama_model_path>)

config.quant_config.quant_algo = QuantAlgo.W4A16_AWQ

llm = LLM(config)
```

## Parallelism

### Tensor Parallelism
It is easy to enable Tensor Parallelism in the HLAPI. For example, setting `parallel_config.tp_size=2` to perform a 2-way parallelism:

```python
from tensorrt_llm import LLM, ModelConfig

config = ModelConfig(model_dir=<llama_model_path>)
config.parallel_config.tp_size = 2
```

### Pipeline Parallelism
Similar to Tensor Parallelism, you can enable Pipeline Parallelism in the HLAPI with following code:

```python
config.parallel_config.pp_size = 4
# you can also mix TP and PP
# config.parallel_config.tp_size = 2
```

### Automatic Parallelism (in preview)

By simply enabling `parallel_config.auto_parallel` in the ModelConfig, TensorRT-LLM can parallelize the model automatically. For example, setting `parallel_config.world_size` to perform a 2-way parallelism:

``` python
from tensorrt_llm import LLM, ModelConfig

config = ModelConfig(model_dir=<llama_model_path>)
config.parallel_config.auto_parallel = True
config.parallel_config.world_size = 2
```

### Multi-GPU multi-node (MGMN) support
By default, the HLAPI will spawn MPI processes under the hood to support single-node-multi-gpu scenarios, what you need to do is to set the `parallel_config.tp_size/pp_size` to the number of GPUs you want to use.

But for MGMN scenarios, since the jobs are managed by some cluster management systems, such as [Slurm](https://slurm.schedmd.com/documentation.html), you need to submit HLAPI tasks differently.

Firstly, it is suggested to build the engine offline with the `trtllm-build` tools, please refer to [LLaMA's README](../llama/README.md) for more details.

Secondly, you need to prepare a Python file containing the HLAPI task, a naive example is as below, note that, this Python script will be executed only once on MPI rank0, and it looks nothing special compared to the single-node-multi-gpu scenario, such as TP or PP.

```python
config = ModelConfig(model_dir=<llama_model_path>)
# Set the parallel_config to the number of GPUs you want to use
config.parallel_config.tp_size = 4
config.parallel_config.pp_size = 2

llm = LLM(config)
for output in llm.generate([[32, 12]]):
    print(output)
```

Thirdly, you need to prepare a Slurm script to submit the task, the script contains the following lines:

```sh
#SBATCH -N 2                                 # number of nodes
#SBATCH --ntasks-per-node=4

srun --container-image="<docker-image>" \
     --mpi=pmix \
     ... \ # much details here
     trtllm-hlapi-launch python3 <your-python-script>.py
```

The `trtllm-hlapi-launch` is a script provided by the HLAPI to launch the task and take care of the MPI runtime, you can find it in the local `bin` directory once you install the TensorRT-LLM package.

Finally, you can submit the task with `sbatch <your-slurm-script>.sh`.

Considering the Slurm or other cluster management systems may be highly customized and the task-submit command may be variant, the forementioned example is just a naive example. The key point is to submit the Python script with the MPI runtime, and the HLAPI will take care of the rest.

## Generation
### `asyncio`-based generation
With the high-level API, you can also perform asynchronous generation with the `generate_async` method. For example:

```python
config = ModelConfig(model_dir=<llama_model_path>)

llm = LLM(config, async_mode=True)

async for output in llm.generate_async(<prompt>, streaming=True):
    print(output)
```

When the `streaming` flag is set to `True`, the `generate_async` method will return a generator that yields the token results as soon as they are available. Otherwise, it will return a generator that yields the final results only.

### Future-style generation
The result of the `generate_async` method is a Future-like object, it doesn't block the thread unless the `.result()` is called.

```python
# This will not block the main thread
generation = llm.generate_async(<prompt>)
# Do something else here
# call .result() to explicitly block the main thread and wait for the result when needed
output = generation.result()
```

The `.result()` method works like the [result](https://docs.python.org/zh-cn/3/library/asyncio-future.html#asyncio.Future.result) method in the Python Future, you can specify a timeout to wait for the result.

```python
output = generation.result(timeout=10)
```

There is an async version, where the `.aresult()` is used.

```python
generation = llm.generate_async(<prompt>)
output = await generation.aresult()
```

### Customizing sampling with `SamplingConfig`
With SamplingConfig, you can customize the sampling strategy, such as beam search, temperature, and so on.

To enable beam search with a beam size of 4, set the `sampling_config` as follows:

```python
from tensorrt_llm import ModelConfig, LLM
from tensorrt_llm.hlapi import SamplingConfig

config = ModelConfig(model_dir=<llama_model_path>, max_beam_width=4)

llm = LLM(config)
# Let the LLM object generate text with the default sampling strategy, or
# you can create a SamplingConfig object as well with several fields set manually
sampling_config = llm.get_default_sampling_config()
sampling_config.beam_width = 4 # current limitation: beam_width should be equal to max_beam_width

for output in llm.generate(<prompt>, sampling_config=sampling_config):
    print(output)
```

You can set other fields in the `SamplingConfig` object to customize the sampling strategy, please refer to the [SamplingConfig](https://nvidia.github.io/TensorRT-LLM/_cpp_gen/runtime.html#_CPPv4N12tensorrt_llm7runtime14SamplingConfigE) class for more details.

## LLM pipeline configuration

### Runtime customization

For `kv_cache_config`, `capacity_scheduling_policy` and `streaming_llm` features, please refer to LLaMA's [README](../llama/README.md) for more details, the high-level API supports these features as well by setting the corresponding fields in the `LLM()` constructor.

```python
from tensorrt_llm import ModelConfig, LLM
from tensorrt_llm.hlapi import KvCacheConfig, CapacitySchedulerPolicy

config = ModelConfig(model_dir=<llama_model_path>)
llm = LLM(config,
          kv_cache_config=KvCacheConfig(
                            max_new_tokens=128,
                            free_gpu_memory_fraction=0.8),
          capacity_scheduling_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT)
```

### Tokenizer customization

By default, the high-level API uses transformers’ `AutoTokenizer`. You can override it with your own tokenizer by passing it when creating the LLM object. For example:

```python
llm = LLM(config, tokenizer=<my_faster_one>)
```

The LLM() workflow should use your tokenizer instead.

It is also possible to input token IDs directly without Tokenizers with the following code, note that the result will be also IDs without text since the tokenizer is not used.

``` python
llm = LLM(config)

for output in llm.generate([32, 12]): ...
```

### Disabling tokenizer
For performance considerations, you can disable the tokenizer by passing the token ID list to the `LLM.generate/_async` method. For example:

```python
config = ModelConfig(model_dir=<llama_model_path>)

llm = LLM(config)
for output in llm.generate([[32, 12]]):
    print(output)
```

You will get something like `GenerationResult(text='', token_ids=[23, 14, 3, 29871, 3], ...)`, note that the `text` field is empty since the tokenizer is not activated.
