(gpt-runtime)=

# C++ GPT Runtime

TensorRT-LLM includes a C++ component to execute TensorRT engines built with
the Python API as described in the {ref}`architecture-overview` section.
That component is called the C++ runtime.

The API of the C++ runtime is composed of the classes declared in
[`cpp/include/tensorrt_llm/runtime`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/include/tensorrt_llm/runtime) and
implemented in [`cpp/tensorrt_llm/runtime`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/runtime).

Even if the different components described in that document mention GPT in
their name, they are not restricted to this specific model. Those classes can
be used to implement auto-regressive models like BLOOM, GPT-J, GPT-NeoX or
LLaMA, for example.

Complete support of encoder-decoder models, like T5, will be added to
TensorRT-LLM in a future release. An experimental version, only in Python for
now, can be found in the [`examples/enc_dec`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/enc_dec) folder.

## Overview

Runtime models are described by an instance of the
[`ModelConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h)
class and a pointer to the TensorRT engine that must be
executed to perform the inference.
The environment is configured through the
[`WorldConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/worldConfig.h)
(that name comes from
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) and its "famous"
`MPI_COMM_WORLD` default communicator).
The [`SamplingConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/samplingConfig.h)
class encapsulates parameters that control the
[generation](https://huggingface.co/blog/how-to-generate) of new tokens.

### Model Configuration

The model configuration is an instance of the
[`ModelConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h) class.
That class encapsulates the following parameters (they are declared as private
member variables and exposed through getters and setters):

 * `vocabSize`, the size of the vocabulary,
 * `numLayers`, the number of layers in the model,
 * `numHeads`, the number of heads in the attention block,
 * `numKvHeads`, the number of heads for K and V in the attention component.
   When the number of K/V heads is the same as the number of (Q) heads, the
   model uses multi-head attention. When the number of K/V heads is 1, it uses
   multi-query attention. Otherwise, it uses group-query attention. Refer to {ref}`gpt-attention` for more information,
 * `hiddenSize`, the size of the hidden dimension,
 * `dataType`, the datatype that was used to build the TensorRT engine and that
   must be used to run the model during inference,
 * `useGptAttentionPlugin`, indicates if the {ref}`gpt-attention` operator was compiled using the
   [GPT Attention plugin](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/plugins/gptAttentionPlugin),
 * `inputPacked`, indicates that the input must be packed (or padded when set
   to `false`). For performance reasons, it is recommended to always use packed,
   even if its default is set to `false` (will be changed in a future release).
   Refer to {ref}`gpt-attention` for more information,
 * `pagedKvCache`, indicates if the K/V cache uses paging.
   Refer to {ref}`gpt-attention` for more information,
 * `tokensPerBlock`, is the number of tokens in each block of the K/V cache.
   It's relevant when the paged K/V cache is enabled. By default, the value is
   64. Refer to {ref}`gpt-attention` for more information,
 * `quantMode`, controls the quantization method. Refer to {ref}`precision` for more information.
 * `maxBatchSize`, indicates the maximum batch size that the TensorRT engine
   was built for,
 * `maxInputLen`, the maximum size of the input sequences,
 * `maxSequenceLen`, the maximum total size (input+output) of the sequences.

### World Configuration

Familiarity with
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), is not required
to utilize the TensorRT-LMM C++ runtime. There are two main things
you need to know:
* The C++ Runtime in TensorRT-LLM uses
[processes](https://en.wikipedia.org/wiki/Process_(computing)) to execute
TensorRT engines on the different GPUs. Those GPUs can be located on a single
node as well as on different nodes in a cluster. Each process is called a
*rank* in MPI.
* The ranks are grouped in communication groups. The
TensorRT-LLM C++ Runtime calls that group the *world*.

The world configuration is an instance of the
[`WorldConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/worldConfig.h)
class, which encapsulates the following parameters:

* `tensorParallelism`, the number of ranks that collaborate together to
  implement Tensor Parallelism (TP). With TP, each GPU performs computations for
  all the layers of the model. Some of those computations are distributed
  across the GPU. TP is more balanced than Pipeline Parallelism (PP), in most cases, but
  requires higher bandwidth between the GPUs. It is the recommended setting in
  the presence of NVLINK between GPUs,
* `pipelineParallelism`, the number of ranks that collaborate together to
  implement Pipeline Parallelism (PP). With PP, each GPU works on a subset of
  consecutive layers. Communications between the GPUs happen only at the
  boundaries of the subsets of layers. It is harder to guarantee the full
  utilization of the GPUs with PP but it requires less memory bandwidth. It
  is the recommended setting in the absence of NVLINK between GPUs,
* `rank`, the unique identifier of the rank,
* `gpusPerNode`, indicates the number of GPUs on each node. Having that
  information allows the C++ runtime to optimize communications between GPUs in
  a node (like taking advantage of the
  [NVLINK](https://www.nvidia.com/en-us/data-center/nvlink/)
  interconnect between GPUs of an A100
  [DGX](https://www.nvidia.com/en-us/data-center/dgx-platform/)
  node).

### Sampling Parameters

The [`SamplingConfig`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/samplingConfig.h)
class encapsulates parameters that control the
[generation](https://huggingface.co/blog/how-to-generate) of new tokens.
Except for the `beamWidth` parameter, all the fields are optional and the
runtime will use a default value if no values are provided by the user. For
vector fields, the TensorRT-LLM runtime supports one value per sequence (that is,
the vector contains `batchSize` values). If all the sequences use the same
value for a given parameter, the vector can be limited to a single element
(that is, `size() == 1`).

***General***

 * `temperature`, a vector of floating-point numbers to control the
   modulation of logits when sampling new tokens. It can have any value `>= 0.0f`. The default value is `1.0f`(no modulation).
 * `minLength`, a vector of integers to set a lower-bound on the number of tokens
   generated. It can have any value `>= 0`. Value `0` has no effect, the first generated token can be EOS. The default value is `1` (at least one non-EOS token is generated).
 * `repetitionPenalty`, a vector of float-point numbers to penalize tokens
    (irrespective of the number of appearances). It is multiplicative penalty. It can have any value `> 0.0f`. Repetition penalty `< 1.0f` encourages repetition, `> 1.0f` discourages it. The default value is `1.0f` (no effect).
 * `presencePenalty`, a vector of float-point numbers to penalize tokens
   already present in the sequence (irrespective of the number of appearances). It is additive penalty.
   It can have any value, values `< 0.0f` encourage repetition, `> 0.f` discourage it. The default value is `0.0f` (no effect).
 * `frequencyPenalty`, a vector of float-point numbers to penalize tokens
   already present in the sequence (dependent on the number of appearances). It is additive penalty. It can have any value, values `< 0.0f` encourage repetition, `> 0.0f` discourage it.
   The default value is `0.0f`(no effect).
 * `noRepeatNgramSize`, a vector of integers. It can have any value `> 0`. If set to int `> 0`, all ngrams of that size can only occur once.

The parameters `repetitionPenalty`, `presencePenalty`, and `frequencyPenalty` are not mutually
exclusive.

***Sampling***

 * `randomSeed`, a vector of 64-bit integers to control the random seed used by
   the random number generator in sampling. Its default value is `0`,
 * `topK`, a vector of integers to control the number of logits to sample from.
   Must be in range of `[0, 1024]`. Its default value is `0`.
   Note that if different values are provided for the
   different sequences in the batch, the performance of the implementation will
   depend on the largest value. For efficiency reasons, we recommend to batch
   requests with similar `topK` values together,
 * `topP`, a vector of floating-point values to control the top-P probability
   to sample from. Must be in range of `[0.f, 1.f]`. Its default value is `0.f`,
 * `topPDecay`, `topPMin` and `topPResetIds`, vectors to control the decay in
   the `topP` algorithm. The `topP` values are modulated by
   a decay that exponentially depends on the length of the sequence as explained in
   [_Factuality Enhanced Language Models for Open-Ended Text Generation_](https://arxiv.org/abs/2206.04624).
   `topPDecay` is the decay, `topPMin` is the lower-bound and `topPResetIds`
   indicates where to reset the decay.
   `topPDecay`, `topPMin` must be in ranges of `(0.f, 1.f]` and `(0.f, 1.f]` respectively.
   Defaults are `1.f`, `1.0e-6,f` and `-1`,

If both `topK` and `topP` fields are set, the `topK` method will be run for
sequences with a `topK` value greater than `0.f`. In that case, the `topP`
value for that sequence also influences the result. If the `topK` values for
some sequences are `0.f`, the `topP` method will be used for those remaining
sequences. If both `topK` and `topP` are zero, greedy search is performed.

***Beam-search***

 * `beamWidth`, is the width used for the [beam
   search](https://en.wikipedia.org/wiki/Beam_search) sampling algorithm. There
   is no explicit upper-bound on the beam width but increasing the beam width
   will likely increase the latency. Use `1` to disable beam-search,
 * `beamSearchDiversityRate`, a floating-point value that controls the
   diversity in beam-search. It can have any value  `>= 0.0f`. The default value is `0.f`,
 * `lengthPenalty`, a floating-point value that controls how to penalize the
   longer sequences in beam-search (the log-probability of a sequence will be
   penalized by a factor that depends on `1.f / (length ^ lengthPenalty)`). The
   default is value `0.f`,
 * `earlyStopping`, a integer value that controls whether the generation process
   finishes once `beamWidth` sentences are generated (end up with `end_token`).
   Default value `1` means `earlyStopping` is enabled, value `0` means `earlyStopping`
   is disable, other values  means the generation process is depended on
   `length_penalty`.
The `beamWidth` parameter is a scalar value. It means that in this release of
TensorRT-LLM, it is not possible to specify a different width for each input
sequence. This limitation is likely to be removed in a future release.

## The Session

*The runtime session is deprecated in favor of the {ref}`executor`.
 It will be removed in a future release of TensorRT-LLM.*

An example of how to use the `GptSession` to run a GPT-like auto-regressive model can be found in
[`cpp/tests/runtime/gptSessionTest.cpp`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tests/runtime/gptSessionTest.cpp).

### Internal Components

The `GptSession` class encapsulates two main components. The
[`TllmRuntime`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/tllmRuntime.h) is in charge of the
execution of the TensorRT engine. The
[`GptDecoder`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/gptDecoder.h)
does the generation of the tokens from the logits.  The `TllmRuntime` class is
an internal component and you are not expected to use that class directly.
The `GptDecoder` can be used directly to implement custom generation loop
and for use cases that cannot be satisfied by the implementation in
`GptSession`.

## In-flight Batching Support

In-flight batching is supported using separate decoders per
request. The biggest difference compared to using a single decoder is in how
the token generation from logits is managed. A batch is split into `batchSize`
individual requests and kernels are issued using separated CUDA streams.
This behavior may be revisited in a future release to maintain the structure
of the batch and improve efficiency.

## Know Issues and Future Changes

 * In the current release of TensorRT-LLM, the C++ and Python runtimes are two
   separate software components and the C++ runtime is being more actively
   developed (with features like in-flight batching). An objective, for a
   future release, could be to rebuild the Python runtime on top of the C++
   one.
