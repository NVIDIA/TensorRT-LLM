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
now, can be found in the [`examples/models/core/enc_dec`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/enc_dec) folder.

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
A comparison of selecting decoding method is listed as the table below (`X` means it is not supported yet).
Except for the `beamWidth` parameter, all the fields are optional and the
runtime will use a default value if no values are provided by the user. For
vector fields, the TensorRT-LLM runtime supports one value per sequence (that is,
the vector contains `batchSize` values). If all the sequences use the same
value for a given parameter, the vector can be limited to a single element
(that is, `size() == 1`).

|        Method name in HF         |                    Condition in HF                    | Method name in TRT-LLM |              Condition in TRT-LLM              |
| :------------------------------: | :---------------------------------------------------: | :--------------------: | :--------------------------------------------: |
|        assisted decoding         | `assistant_model` or `prompt_lookup_num_tokens!=None` |           X            |                                                |
|       beam-search decoding       |          `num_beams>1` and `do_sample=False`          |      beam search       |                `beamWidth > 1`                 |
| beam-search multinomial sampling |          `num_beams>1` and `do_sample=True`           |           X            |                                                |
| constrained beam-search decoding |    `constraints!=None` or `force_words_ids!=None`     |           X            |                                                |
|        contrastive search        |            `penalty_alpha>0` and `top_k>1`            |           X            |                                                |
|   diverse beam-search decoding   |         `num_beams>1` and `num_beam_groups>1`         |           X            |                                                |
|         greedy decoding          |          `num_beams=1` and `do_sample=False`          |        sampling        | `beamWidth == 1` and `topK=0` and `topP=0.0f`  |
|       multinomial sampling       |          `num_beams=1` and `do_sample=True`           |        sampling        | `beamWidth == 1` and (`topK>0` or `topP>0.0f`) |

***General***

|   Name in TRT-LLM   |                                    Description                                    |   Data type   |                                      Range of value                                       |                     Default value                     |       Name in HF       |
| :-----------------: | :-------------------------------------------------------------------------------: | :-----------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------: | :--------------------: |
|    `temperature`    |                     modulation of logits in sampling workflow                     | List\[Float\] |                                    \[0.0f, $+\infty$\)                                    |                `1.0f` (no modulation)                 |     `temperature`      |
|     `minLength`     |                   lower-bound on the number of tokens generated                   |  List\[Int\]  |                                     \[0, $+\infty$\)                                      | `0` (no effect (the first generated token can be EOS) |      `min_length`      |
| `repetitionPenalty` | penalize repetitive tokens <br> multiplicative, irrespective of appearances count | List\[Float\] |   \[0.0f, $+\infty$\) <br> `< 1.0f` encourages repetition <br> `> 1.0f` discourages it    |                  `1.0f` (no effect)                   |  `repetition_penalty`  |
|  `presencePenalty`  |     penalize existed tokens <br> additive, irrespective of appearances count      | List\[Float\] | \($-\infty$, $+\infty$\) <br> `< 0.0f` encourages repetition <br> `> 0.0f` discourages it |                  `0.0f` (no effect)                   |           no           |
| `frequencyPenalty`  |       penalize existed tokens <br> additive, dependent on appearances count       | List\[Float\] | \($-\infty$, $+\infty$\) <br> `< 0.0f` encourages repetition <br> `> 0.0f` discourages it |                  `0.0f` (no effect)                   |           no           |
| `noRepeatNgramSize` |                                                                                   |  List\[Int\]  |          \[0, $+\infty$\) <br> `> 0` all ngrams of that size can only occur once          |                    `0` (no effect)                    | `no_repeat_ngram_size` |

* The tokens of input prompt are included during adopting `repetitionPenalty`, `presencePenalty`, and `frequencyPenalty` onto logits.

* The parameters `repetitionPenalty`, `presencePenalty`, and `frequencyPenalty` are not mutually exclusive.

***Sampling***

| Name in TRT-LLM |               Description                                               |   Data type   |  Range of value   |  Default value   | Name in HF |
| :-------------: | :---------------------------------------------------------------------: | :-----------: | :---------------: | :--------------: | :--------: |
|  `randomSeed`   | random seed for random number generator                                 |     Int64     |   \[0, 2^64-1\]   |       `0`        |     no     |
|     `topK`      |   the number of logits to sample from                                   |  List\[Int\]  |    \[0, 1024\]    |       `0`        |  `top_k`   |
|     `topP`      |  the top-P probability to sample from                                   | List\[Float\] |  \[0.0f, 1.0f\]   |      `0.0f`      |  `top_p`   |
|   `topPDecay`   |    the decay in the `topP` algorithm                                    | List\[Float\] |  \(0.0f, 1.0f\]   |      `1.0f`      |     no     |
|    `topPMin`    |    the decay in the `topP` algorithm                                    | List\[Float\] |  \(0.0f, 1.0f\]   |    `1.0e-6,f`    |     no     |
| `topPResetIds`  |    the decay in the `topP` algorithm                                    |  List\[Int\]  | \[-1, $+\infty$\) | `-1` (no effect) |     no     |
|     `minP`      | scale the most likely token to determine the minimum token probability. |  List\[Float\]  | \[0.0f, 1.0f\] | `0.0` (no effect) |     `min_p`     |

 * If setting `topK = 0` and `topP = 0.0f`, greedy search is performed.
 * If setting `topK > 0` and `topP = 0.0f`, `topK` tokens of highest probabilities will become the candidates of sampling (named `TopK sampling` in TRT-LLM).
 * If setting `topK = 0` and `topP > 0.0f`, tokens will be sorted with probability descendly, then the tokens with highest probabilities which the accumulated probability larger than `topP` will become the candidates of sampling (named `TopP sampling` in TRT-LLM).
 * If setting `topK > 0` and `topP > 0.0f`, `topK` tokens of highest probabilities will be selected, then those selected tokens will be sorted with probability descendly and their probability will be normalized, then the tokens with highest normalized probabilities which the accumulated probability larger than `topP` will become the candidates of sampling (named `TopKTopP sampling` in TRT-LLM)

 * If different `topK` values are provided for the different sequences in the batch, the performance of the implementation will depend on the largest value. For efficiency reasons, we recommend to batch requests with similar `topK` values together.

 * `topPDecay`, `topPMin` and `topPResetIds` are explained in
   [_Factuality Enhanced Language Models for Open-Ended Text Generation_](https://arxiv.org/abs/2206.04624).
   `topPDecay` is the decay, `topPMin` is the lower-bound and `topPResetIds` indicates where to reset the decay.

 * `minP` is explained in [_Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs_](https://arxiv.org/abs/2407.01082).

 * TensorRT-LLM does not generate all possible tokenizations of a word. Therefore, stop words may appear in the output if there are multiple ways to tokenize a stop word and the token sequence in the output differs from the one in `stopWords`.

***Beam-search***

|      Name in TRT-LLM      |           Description           |      Data type      |      Range of value      |       Default value       |     Name in HF      |
| :-----------------------: | :-----------------------------: | :-----------------: | :----------------------: | :-----------------------: | :-----------------: |
|        `beamWidth`        | width for beam-search algorithm |         Int         |       \[0, 1024\]        | `0` (disable beam search) |    `beam_width`     |
| `beamSearchDiversityRate` |  diversity of generated tokens  |    List\[Float\]    |     \[0, $+\infty$\)     |          `0.0f`           | `diversity_penalty` |
|      `lengthPenalty`      |    penalize longer sequences    |    List\[Float\]    |     \[0, $+\infty$\)     |          `0.0f`           |  `length_penalty`   |
|      `earlyStopping`      |      see description below      |     List\[Int\]     | \($-\infty$, $+\infty$\) |            `0`            |  `early_stopping`   |
|     `beamWidthArray`      |      see description below      | List\[List\[Int\]\] |       \[0, 1024\]        |            ``             |         no          |

 * Beam-search algorithm: [beam search](https://en.wikipedia.org/wiki/Beam_search).
 * Parameter `diversity_penalty` in HF is only used for `diverse beam-search decoding` (or named `Group-Beam-Search`), which is not supported by TRT-LLM yet.
 * If setting `earlyStopping = 1`, decoding will stop once `beamWidth` finished sentences are generated.
 * If setting `earlyStopping = 0`, decoding will keep going until no better sentences (with better score) can be generated.
 * If setting `earlyStopping` to other values, decoding will stop only depending on `lengthlengthPenalty`.
 * `beamWidthArray` is a list of beam width for each step. Using `beamWidthArray = [20,40,80]` as an example,
beam width will be 20 for the first step, 40 for second step, 80 for the later all steps.
 * The `beamWidth` parameter is a scalar value. It means that in this release of
TensorRT-LLM, it is not possible to specify a different width for each input
sequence. This limitation is likely to be removed in a future release.

### Internal Components

The [`TllmRuntime`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/tllmRuntime.h) is in charge of the execution of the TensorRT engine.
The `TllmRuntime` class is an internal component and you are not expected to use that class directly.
The [`GptDecoder`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime/gptDecoder.h) generates tokens from the logits.
The `GptDecoder` can be used directly to implement a custom generation loop and for use cases that cannot be satisfied by the TRT-LLM implementation.
