# TensorRT-LLM Architecture

TensorRT-LLM is a toolkit to assemble optimized solutions to perform Large
Language Model (LLM) inference. It offers a Python API to define models and
compile efficient [TensorRT](https://developer.nvidia.com/tensorrt) engines for
NVIDIA GPUs. It also contains Python and C++ components to build runtimes to
execute those engines as well as backends for the [Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server) to easily
create web-based services for LLMs. TensorRT-LLM supports multi-GPU and
multi-node configurations (through MPI).

As a user, the very first step to create an inference solution is to either
define your own model or select a pre-defined network architecture (see
[here]() for the list of models supported by TensorRT-LLM). Once defined, that
model must be trained using a training framework (training is outside of the
scope of TensorRT-LLM). For pre-defined models, checkpoints can be downloaded
from various providers.  To illustrate that point, a lot of examples in
TensorRT-LLM use model weights obtained from the
[HuggingFace](https://huggingface.co) hub and trained using [NVIDIA
Nemo](https://developer.nvidia.com/nemo) or [PyTorch](https://pytorch.org).

Equipped with the model definition and the weights, a user must use
TensorRT-LLM's Python API to recreate the model in a way that can be compiled
by TensorRT into an efficient engine. For ease of use, TensorRT-LLM already
supports a handful of standard models.

Together with the Python API to describe models, TensorRT-LLM provides users
with components to create a runtime that executes the efficient TensorRT
engine. Runtime components offer beam-search, along with extensive sampling
functionalities such as top-K and top-P sampling. The exhaustive list can be
found in the documentation of the [Runtime](./gpt_runtime.md). The C++ runtime
is the recommended runtime.

TensorRT-LLM also includes Python and C++ backends for NVIDIA Triton Inference
Server to assemble solutions for LLM online serving. The C++ backend implements
in-flight batching as explained in the [Batch Manager](./batch_manager.md)
documentation and is the recommended backend.

## Model Definition

As mentioned above, TensorRT-LLM has a Python API that can be used to define
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

When the TensorRT-LLM's Python API is utilized, a graph of the network is
assembled.  The graph can later be traversed or transformed using the graph
traversal API exposed by the
[`tensorrt.ILayer`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/LayerBase.html#tensorrt.ILayer)
class. That graph will also be optimized by TensorRT during the compilation of
the engine, as explained in the next section.

## Compilation

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

### Weight Bindings

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

### Pattern-Matching and Fusion

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

### Plugins

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
the [Multi-head, Multi-query and Group-query Attention](gpt_attention.md) document.

## Runtime

TensorRT-LLM includes an API to implement Python and C++ runtimes. The role of
the runtime components is to load the TensorRT engines and drive their
execution. Typically, for an auto-regressive model like GPT, the runtime is in
charge of loading the engine that implements both the processing of the input
sequence as well as the body of the generation loop. See the [GPT C++
Runtime](gpt_runtime.md) document for details on the C++ Runtime.

### Multi-GPU and Multi-Node Support

Even if TensorRT is designed for single-GPU systems, TensorRT-LLM adds the
support for systems with multiple GPUs and nodes. It is enabled
using TensorRT plugins that wrap communication primitives from the
[NCCL](https://developer.nvidia.com/nccl) library as well as a custom
plugin that optimize the All-Reduce primitive in the presence of All-to-all
connections between GPUs (through NVSwitch in DGX systems).

The communication plugins can be found in
[cpp/tensorrt_llm/plugins/ncclPlugin](source:cpp/tensorrt_llm/plugins/ncclPlugin)
and the multi-GPU functions are exposed in the TensorRT-LLM Python API
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

## In-flight Batching

TensorRT-LLM supports in-flight batching of requests (also known as continuous
batching or iteration-level batching) for higher serving throughput. See the
[Batch Manager](./batch_manager.md) document for more details.
