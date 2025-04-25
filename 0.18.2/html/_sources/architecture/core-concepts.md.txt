(core-concepts)=

# Model Definition

TensorRT-LLM has a Model Definition API that can be used to define
Large Language Models. This API is built on top of the powerful
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html#)
to create graph representations of deep neural networks in TensorRT. To become
familiar with the core concepts of the TensorRT API, refer to the
[Core Concepts](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/coreConcepts.html)
section of the TensorRT documentation before proceeding further.

In TensorRT-LLM, the [`tensorrt_llm.Builder`](source:tensorrt_llm/builder.py) class
contains a
[`tensorrt.Builder`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder)
object. That instance is used in the `tensorrt_llm.Builder.create_network`
method to create an instance of the
[`tensorrt.INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#tensorrt.INetworkDefinition)
class. The `INetworkDefinition` object can then be populated using the free
functions defined in the
[`tensorrt_llm.functional`](source:tensorrt_llm/functional.py).

A simple example of such a free function is `tensorrt_llm.activation` that inserts a
[`tensorrt.IActivationLayer`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Layers.html#tensorrt.IActivationLayer)
node in the graph of the model:

```python
# In tensorrt_llm.functional:

def activation(input: Tensor, act_type: trt.ActivationType) -> Tensor:
    layer = default_trtnet().add_activation(input.trt_tensor, act_type)   # default_trtnet() -> INetworkDefinition
    return _create_tensor(layer.get_output(0), layer)
```

To make it even easier for users, a few of the most standard activation
functions found in LLMs are derived from that function:

```python
# In tensorrt_llm.functional:

relu    = partial(activation, act_type=trt.ActivationType.RELU)
sigmoid = partial(activation, act_type=trt.ActivationType.SIGMOID)

```

Specialized activation functions can be used to assemble more advanced
functions such as the `silu` activation:

```python
# In tensorrt_llm.functional:

def silu(input: Tensor) -> Tensor:
    return input * sigmoid(input)
```

When the TensorRT-LLM's Model Definition API is utilized, a graph of the network is
assembled.  The graph can later be traversed or transformed using the graph
traversal API exposed by the
[`tensorrt.ILayer`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#tensorrt.ILayer)
class. That graph will also be optimized by TensorRT during the compilation of
the engine, as explained in the next section.

# Compilation

Once populated, the instance of the
[`tensorrt.INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#tensorrt.INetworkDefinition),
can be compiled into an efficient engine by the
[`tensorrt.Builder`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder)
In TensorRT-LLM, it is done through the `build_engine` member function of the
`tensorrt_llm.Builder` class that calls the
[`build_serialized_network`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder.build_serialized_network)
method of the
[`tensorrt.Builder`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder)
object. That call, if everything works as expected, produces an instance of the
[`tensorrt.IHostMemory`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/HostMemory.html#tensorrt.IHostMemory)
class. That object is an optimized TensorRT engine that can be stored as a
binary file.

## TensorRT Compiler

The TensorRT compiler can sweep through the graph to choose the best kernel for each operation and available GPU. Crucially, it can also identify patterns in the graph where multiple operations are good candidates for being fused into a single kernel. This reduces the required amount of memory movement and the overhead of launching multiple GPU kernels.

TensorRT also compiles the graph of operations into a single [CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/) that can be launched all at one time, further reducing the kernel launch overhead.

The TensorRT compiler is extremely powerful for fusing layers and increasing execution speed, but there are some complex layer fusions—like [FlashAttention](https://arxiv.org/abs/2307.08691) — that involve interleaving many operations together and which can’t be automatically discovered. For those, you can explicitly replace parts of the graph with [plugins](#plugins) at compile time.

## Model Engine

The engine file contains the information that you need for executing the model, but LLM usage in practice requires much more than a single forward pass through the model. TensorRT-LLM includes a highly optimized C++ runtime for executing built LLM engines and managing processes like sampling tokens from the model output, managing the KV cache, and batching requests together.

You can use that runtime directly to execute the model locally, or you can use the TensorRT-LLM runtime backend for NVIDIA Triton Inference Server to serve the model for multiple users.

## Weight Bindings

TensorRT engines embed the network weights, that must be known for compilation.
For that reason, the weights must be bound to parameters in the model
definition before calling `tensorrt_llm.Builder.build_engine`. It leads to code like:

```python
# The Linear operator exposes two parameters (see tensorrt_llm/layers/linear.py):
class Linear(Module):
    def __init__(self, ...):
        self.weight = Parameter(shape=(self.out_features, self.in_features), dtype=dtype)
        self.bias   = Parameter(shape=(self.out_features, ), dtype=dtype)

# The parameters are bound to the weights before compiling the model. See examples/gpt/weight.py:
tensorrt_llm_gpt.layers[i].mlp.fc.weight.value = fromfile(...)
tensorrt_llm_gpt.layers[i].mlp.fc.bias.value   = fromfile(...)
```

Note that TensorRT can also
[refit](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#refitting-engine-c)
engines to update the weights after compilation. This feature is available to
TensorRT-LLM users through the `refit_engine` method in the
`tensorrt_llm.Builder` class.

## Pattern-Matching and Fusion

One of the key steps performed by TensorRT when it compiles the network graph
is the fusion of operations. Fusion is a well-known technique to improve the
efficiency when executing LLMs. It helps reduce the amount of data transferred
between the memory (DRAM) and the compute cores (CUDA cores as well as Tensor
Cores located on the [Streaming
Multiprocessors](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)
of a GPU). It also removes kernel launch overhead (each time a kernel is
launched on the GPU, there is a small additional CPU cost that is called the
launch overhead). A classical example is the fusion of the activation function
with the matrix multiplication (matmul) that usually precedes it in the
network.

In TensorRT-LLM, when defining the model, such a sequence can be written as:

```python
c = tensorrt_llm.functional.matmul(a, b)
c = tensorrt_llm.functional.relu(c)
```

During inference, if the above sequence is executed without fusion, the `c`
tensor has to be written to global memory at the end of the `matmul`, read from
that same memory in `relu` and written again after `relu`. If no other
operation uses the intermediate values between `matmul` and `relu`, it is
suboptimal.  That is why, during compilation, TensorRT will identify that
pattern and automatically produce a GPU kernel that applies `relu` at the end
of `matmul` without an intermediate step through global memory. With that
optimization, the `c` tensor is written only once (after `relu`) instead of
twice, and is not read between the two operations.

The process of identifying the sequences of operations that can be fused is
called _pattern-matching_. TensorRT has a powerful pattern-matching algorithm
that can identify a lot of possible fusions. All the identified patterns are
converted into more efficient kernels by an advanced kernel compiler.

## Plugins

The number of possible fusions is almost infinite and some useful fusions
involve very advanced modifications of the graph. A well-known example
is the [Flash-Attention](https://arxiv.org/abs/2205.14135) technique to
optimize the [Multihead-Attention](https://arxiv.org/abs/1706.03762) block
found in many LLMs. Flash-Attention requires modifications to the arithmetic
performed in the sequence `BMM-Softmax-BMM` (where `BMM` stands for Batched
Matrix-Matrix product) and the interleaving of the `for`-loops of the two
batched matrix products.  That's non-trivial and not necessarily something
you can expect a compiler to "discover" on its own (or it might require the
support for a [polyhedral
model](https://en.wikipedia.org/wiki/Polytope_model)).

As a result, even if TensorRT has a powerful pattern-matching algorithm and
supports a lot of possible fusions, there is always the risk that it cannot
identify uncommon and/or very advanced patterns. To overcome that inevitable
limitation, TensorRT offers a powerful mechanism known as
[plugins](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Plugin/pyPlugin.html).

The plugins are nodes inserted in the network graph definition that map to user-defined
GPU kernels. TensorRT-LLM uses a number of such plugins. They can be found in
the [`cpp/tensorrt_llm/plugins`](source:/cpp/tensorrt_llm/plugins) directory.

Plugins are written in C++ and follow a well-defined interface described in the
[Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
section of the TensorRT
[Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).
When executed within a TensorRT engine, plugins trigger the execution of
their encapsulated GPU kernels. A fairly simple example of plugins is the
[`QuantizeTensorPlugin`](source:/cpp/tensorrt_llm/plugins/quantizeTensorPlugin) that
triggers a CUDA kernel in the `QuantizeTensorPlugin::enqueue` member function:

```cpp
// In cpp/tensorrt_llm/plugins/quantizeTensorPlugin/quantizeTensorPlugin.cpp:

int QuantizeTensorPlugin::enqueue(...) {
    if (inputDesc[0].type == DataType::kFLOAT) {
        invokeQuantization<float>(...);
    } else {
        invokeQuantization<half>(...);
    }
    return 0;
}

// In cpp/tensorrt_llm/kernels/quantization.cu:

template <typename T>
void invokeQuantization(...) {
    // The standard <<< >>> construct to launch CUDA kernels
    quantizedKernel<<<grid, block, 0, stream>>>(...);
}
```

For more details on how TensorRT-LLM implements the GPT Attention operator, see
the [Multi-head, Multi-query and Group-query Attention](../advanced/gpt-attention.md) document.

# Runtime

TensorRT-LLM includes an API to implement Python and C++ runtimes. The role of
the runtime components is to load the TensorRT engines and drive their
execution. Typically, for an auto-regressive model like GPT, the runtime is in
charge of loading the engine that implements both the processing of the input
sequence as well as the body of the generation loop. See the [GPT C++
Runtime](../advanced/gpt-runtime.md) document for details on the C++ Runtime.

(multi-gpu-multi-node)=

# Multi-GPU and Multi-Node Support

Even if TensorRT is designed for single-GPU systems, TensorRT-LLM adds the
support for systems with multiple GPUs and nodes. It is enabled
using TensorRT plugins that wrap communication primitives from the
[NCCL](https://developer.nvidia.com/nccl) library as well as a custom
plugin that optimize the All-Reduce primitive in the presence of All-to-all
connections between GPUs (through NVSwitch in DGX systems).

The communication plugins can be found in
[cpp/tensorrt_llm/plugins/ncclPlugin](source:cpp/tensorrt_llm/plugins/ncclPlugin)
and the multi-GPU functions are exposed in the TensorRT-LLM Model Definition API
as:

```python
# In tensorrt_llm/functional.py:

# Collectives.
def allreduce(tensor: Tensor, group: List[int]) -> Tensor
def allgather(tensor: Tensor, group: List[int], gather_dim: int = 0) -> Tensor

# Point-to-point communication primitives.
def send(tensor: Tensor, tgt: int) -> Tensor
def recv(tensor: Tensor, src: int) -> Tensor
```

The multi-GPU support can be enabled through two different modes of model
parallelism: Tensor Parallelism and Pipeline Parallelism. The former mode
splits the different layers of a model across the GPUs. Each GPU runs the
entire network and synchronizes with its siblings when needed. The Pipeline
Parallelism distributes the different layers to the GPUs. Each GPU runs a
subset of the entire model and communications happen at the boundary of those
subsets of layers. Tensor Parallelism usually leads to more balanced executions
but requires more memory bandwidth between the GPUs. Pipeline Parallelism
reduces the need for high-bandwidth communication but may incur load-balancing
issues and may be less efficient in terms of GPU utilization.

## Examples

Here are examples of Llama 3.1 70B and Llama 3.1 405B showing how to perform multi-GPU and multi-node inference in TensorRT-LLM. The example of Llama 3.1 70B performs multi-GPU inference on a single node, while the example of Llama 3.1 405B performs multi-node inference.

### Llama 3.1 70B

The following sample commands build an engine for running the Llama 3.1 70B model with tensor parallelism (TP=4) using 4 GPUs on a single node.

```bash
folder_trt_llm=../TensorRT-LLM
model_dir=Llama-3.1-70B
ckpt_dir=ckpt_llama_3.1_70b
engine_dir=engine_llama_3.1_70b
dtype=bfloat16
tp_size=4
pp_size=1
kv_cache_type=paged
max_input_len=128
max_output_len=128
max_batch_size=4
workers=$(( tp_size * pp_size ))

python ${folder_trt_llm}/examples/llama/convert_checkpoint.py \
    --output_dir ${ckpt_dir} \
    --model_dir ${model_dir} \
    --dtype ${dtype} \
    --tp_size ${tp_size} \
    --pp_size ${pp_size} \
    --workers ${workers} \
    --use_parallel_embedding

trtllm-build \
    --output_dir ${engine_dir} \
    --checkpoint_dir ${ckpt_dir} \
    --gemm_plugin ${dtype} \
    --gpt_attention_plugin ${dtype} \
    --kv_cache_type ${kv_cache_type} \
    --max_input_len ${max_input_len} \
    --max_seq_len $(( max_input_len + max_output_len )) \
    --max_batch_size ${max_batch_size} \
    --workers ${workers}
```

The following sample commands perform inference using 4 GPUs on a single node by running `examples/run.py`.

```bash
input_text="Born in north-east France, Soyer trained as a"

mpirun -n $(( tp_size * pp_size )) \
    python ${folder_trt_llm}/examples/run.py \
        --engine_dir ${engine_dir} \
        --tokenizer_dir ${model_dir} \
        --input_text "${input_text}" \
        --max_output_len ${max_output_len}
```

### Llama 3.1 405B

The following sample commands build an engine for running the Llama 3.1 405B model with tensor parallelism (TP=16) on 2 nodes that each have 8 GPUs. Although the model runs on multiple nodes, you can build the engine on a single node.

```bash
folder_trt_llm=../TensorRT-LLM
model_dir=Llama-3.1-405B
ckpt_dir=ckpt_llama_3.1_405b
engine_dir=engine_llama_3.1_405b
dtype=bfloat16
tp_size=16
pp_size=1
kv_cache_type=paged
max_input_len=128
max_output_len=128
max_batch_size=4
workers=8

python ${folder_trt_llm}/examples/llama/convert_checkpoint.py \
    --output_dir ${ckpt_dir} \
    --model_dir ${model_dir} \
    --dtype ${dtype} \
    --tp_size ${tp_size} \
    --pp_size ${pp_size} \
    --workers ${workers} \
    --use_parallel_embedding

trtllm-build \
    --output_dir ${engine_dir} \
    --checkpoint_dir ${ckpt_dir} \
    --gemm_plugin ${dtype} \
    --gpt_attention_plugin ${dtype} \
    --kv_cache_type ${kv_cache_type} \
    --max_input_len ${max_input_len} \
    --max_seq_len $(( max_input_len + max_output_len )) \
    --max_batch_size ${max_batch_size} \
    --workers ${workers}
```

The following sample script, `launch_llama_3.1_405b.sh`, shows how to perform inference with Slurm on 2 nodes that each have 8 GPUs. If you use a different workload management software, the key concern is to run the `examples/run.py` command.

```bash
#!/bin/bash
#SBATCH --account account
#SBATCH --partition partition
#SBATCH --job-name job-name
#SBATCH --time 1:00:00
#SBATCH --nodes 2

folder_trt_llm=../TensorRT-LLM
engine_dir=engine_llama_3.1_405b
model_dir=Llama-3.1-405B
max_output_len=128

input_text="Born in north-east France, Soyer trained as a"

srun \
    --ntasks-per-node 8 \
    --mpi pmix \
    python ${folder_trt_llm}/examples/run.py \
        --engine_dir ${engine_dir} \
        --tokenizer_dir ${model_dir} \
        --input_text "${input_text}" \
        --max_output_len ${max_output_len}
```

You can perform inference by running the script on the Slurm cluster.

```bash
sbatch launch_llama_3.1_405b.sh
```
