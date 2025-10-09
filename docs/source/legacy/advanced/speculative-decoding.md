# Speculative Sampling

- [About Speculative Sampling](#about-speculative-sampling)
- [Performance Improvements](#Performance-improvements)
- [Draft-Target-Model](#Draft-Target-Model)
- [NGram](#ngram)
- [Medusa](#medusa)
  - [Medusa Tree](#medusa-tree)
  - [Using Medusa with TensorRT-LLM](#using-medusa-with-tensorrt-llm)
    - [Limitations](#limitations)
- [ReDrafter](#redrafter)
- [EAGLE](#eagle)
    - [Disaggregated Serving](#disaggregated-serving)
- [Lookahead decoding](#lookahead-decoding)

## About Speculative Sampling

Speculative Sampling (also referred to as Speculative Decoding) is a set of techniques designed to allow generation of more than one token per forward pass iteration. This can lead to a reduction in the average per-token latency **in situations where the GPU
is underutilized due to small batch sizes.**

Speculative Sampling involves predicting a sequence of future tokens, referred to as draft tokens, using a method
that is substantially more efficient than repeatedly executing the target Large Language Model (LLM).
These draft tokens are then collectively validated by processing them through the target LLM in a single forward pass.
The underlying assumptions are twofold:

1. processing multiple draft tokens concurrently will be as rapid as processing a single token
2. multiple draft tokens will be validated successfully over the course of the full generation

If the first assumption holds true, the latency of speculative decoding will no worse than the standard approach. If the second holds, output token generation advances by statistically more than one token per forward pass.
The combination of both these allows speculative decoding to result in reduced latency.

TensorRT-LLM supports several approaches for generating draft tokens, including:

1. Utilizing a smaller, auxiliary model, known as the draft model approach. For more information, refer to the [Fast Inference from Transformers via Speculative Decoding paper](https://arxiv.org/pdf/2211.17192.pdf).
2. Implementing additional language model heads that predict tokens for future positions:
    1. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads paper](https://arxiv.org/abs/2401.10774).
    2. [Recurrent Drafter for Fast Speculative Decoding in Large Language Models](https://arxiv.org/html/2403.09919v1).
    3. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077).
3. Utilizing prompt tokens as draft tokens. For more information, refer to [NGram](https://github.com/apoorvumang/prompt-lookup-decoding/).
4. Utilizing Jacobi-like decoding to predict and verify draft tokens using the same model which does not need additional fine-tuning. Refer to [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/pdf/2402.02057).


## Performance Improvements

It's important to note that the effectiveness of speculative decoding techniques is highly dependent
on the specific task at hand. For instance, forecasting subsequent tokens in a code-completion scenario
may prove simpler than generating a summary for an article.

Furthermore, when integrating Medusa with a standard PyTorch model implementation which may not be as finely
tuned as TensorRT-LLM, the potential time savings are more pronounced.

## Draft-Target-Model

The Draft-Target-Model involves the use of two distinct models (a smaller Draft model and a larger Target model) trained independently but sharing the same vocabulary. For example, GPT 125M / 6.7B models serve as the Draft / Target model.

The management of Draft and Target models is facilitated through two separate `Executor` instances.
It is essential that you to coordinate the interactions between the Draft and Target models effectively.
Initially, the Draft model is queried to generate up to `K` draft tokens.
These tokens are then forwarded to the Target model for verification.
Upon verification, the Target model may return up to `K+1` tokens.
Subsequently, the prompt, now updated with the accepted tokens, is sent back to the Draft model to initiate the generation of new draft tokens.
This iterative process continues until a predefined stop conditions are met.
An example orchestration script is available in the Triton backend repository’s
[draft-target-model client example](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/client/python/draft_target_model_client.py).

We provide two styles of running Draft-Target-Model now: using TensorRT-LLM-BLS in Triton Inference Server, or using TensorRT-LLM directly. Detailed steps of running can be found in [examples/draft_target_model/README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/draft_target_model/README.md) and the code can be found in [examples/ngram/run_dtm_ngram.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/ngram/run_dtm_ngram.py).

## NGram

The NGram speculative decoding directly copies from the input prompt and previous generated output as draft tokens while generating the later output. It works like Draft-Target-Model but involves only one Target LLM model without further fine-tuning. The NGram profit from the scenarios which have high n-gram overlap between input prompt and output, such as summarization, document QA, multi-turn chat, code editing, etc.

See document in [examples/ngram/README.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/ngram/README.md) and the code can be found in [examples/ngram/run_dtm_ngram.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/ngram/run_dtm_ngram.py).

## Medusa

This approach leverages a single model to both generate and verify draft tokens.
It enhances the existing model by adding multiple extra language model heads, known as Medusa heads.
These additional heads are trained to predict future tokens while the base model remains unchanged.
Specifically, the first Medusa head is tasked with predicting the immediate next token,
the second head predicts the token after that, and so on.
With `K` Medusa heads, the model can forecast up to `K` tokens ahead.
The draft tokens generated by the Medusa heads during iteration `i`
are then verified and potentially accepted in the subsequent iteration, `i+1`.

The true potential of the Medusa strategy is realized when more than one token per head is used,
employing a TopK approach to create multiple potential paths, essentially forming a tree, rather than
a single linear path as seen in the Draft model approach. To reduce redundant computations, many of these paths,
which often share common prefixes, are consolidated into a single path.
This is achieved by applying attention with a sparse mask that represents the various paths. Sparse mask formed by Medusa tree is described in detail later.

By validating multiple paths simultaneously, there is an increased likelihood of accepting more than one token per iteration,
albeit at the expense of additional computational effort.

It is crucial to recognize that as the number of potential paths grows exponentially with `K`,
it is not necessary to explore or validate all of them. A recommended strategy for managing this complexity is to prune the tree
by focusing only on the paths with higher-probability tokens.

You must strike a balance between the breadth and depth of the tree you want to explore and the impact of a larger tree on the overall
performance for your specific application.

In the TensorRT-LLM implementation of Medusa, the configuration of the tree is a runtime parameter.
This flexibility allows you to experiment and identify the optimal tree structure for your use case,
which can then be utilized in a production environment.

### Medusa Tree

Consider the following diagram, which illustrates how the hidden states from the last layer of the base model
are passed to the base model's language model (LM) head and to four Medusa heads (MHs).

<p align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/blob/rel/docs/source/media/medusa_tree.svg?raw=true" alt="Example Medusa Tree" width="auto" height="auto">
</p>

In this example:

1. The token <code>l<sub>0</sub></code> represents the actual token generated by the model.
All other tokens, denoted as <code>p<sub>hk</sub></code>, are predictions from the MHs,
where `h` indicates the Medusa head index (1-based) and `k` represents the TopK choice index (0-based).
1. Four MHs are used, which means the model is predicting four future tokens.
2. The first two MHs utilize Top-2 predictions, while the last two use Top-1.
For instance, <code>p<sub>10</sub></code> and <code>p<sub>11</sub></code> are the top and
second top predictions from the first Medusa Head (MH1).
1. A total of four paths are explored, which is fewer than the 16 that would be examined
if a complete binary tree were used (assuming Top-2 predictions for all MHs).
1. As some of these paths may be accepted, there are ten potential candidates, referred to as `medusa_choices`.
The number of tokens that can be accepted at each step, including the true token,
ranges from 1 (if all Medusa predictions are incorrect) to 5 (if all are correct).

During the generation phase, the model receives an input of 10 tokens,
which corresponds to the last tokens of each candidate path, rather than just one.

In TensorRT-LLM, you have the option to define such trees by providing all the Medusa choices
or by simply specifying the unique paths.

- Since each candidate/path begins with the true token (<code>l<sub>0</sub></code>),
there is no need to specify it separately. For the predicted tokens, only the TopK indices are required.
- For example, to specify the path <code>l<sub>0</sub>p<sub>10</sub>p<sub>21</sub>p<sub>30</sub></code>,
one would use `[0,1,0]`. And
to specify the path <code>l<sub>0</sub>p<sub>11</sub>p<sub>20</sub></code>,
one would use `[1,0]`.
- To specify all 4 paths in the example, use `medusa_choices=[[0,0,0,0], [0,1,0], [1,0], [1,1]]`.
- It's also possible to specify all candidates explicitly, similar to the Medusa repository.
For instance, `medusa_choices=[[0], [0,0], [0,0,0], [0,0,0,0], [0,1],
[0,1,0], [1], [1,0], [1,1]]`. Note that when specifying all the candidates explicitly, **we don't include
the empty `[]` candidate** for the case where only the true token is accepted, that is, all the predictions from MHs are wrong.
So, only `9` candidates are specified.

**Specifying paths-only instead of all choices is currently supported only in the Python runtime.**

### Using Medusa with TensorRT-LLM

For guidance on constructing and executing Medusa with the Python runtime, consult the [Medusa README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/medusa/README.md). When utilizing the Inflight Fused Batching (IFB) with the C++ API, it is necessary to define the `medusa_choices` explicitly within the model configuration. For detailed instructions, refer to the [model configuration in TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#modify-the-model-configuration) for more details.

#### Limitations

- TensorRT-LLM supports Medusa only for Vicuna (fine tuned LLaMA).
However, similar to any new model, you can follow the same approach to define your own Medusa model and deploy with TensorRT-LLM.
- We match only tokens during the validation phase that is `medusa_temperature=0`.
- Beam search is **not** compatible with Medusa.


## ReDrafter

The ReDrafter approach enhances the single-model Medusa method by predicting and verifying tokens using the same model. However, unlike Medusa, it predicts draft tokens using a recurrent predictor, where each draft token depends on the previous one. This method also allows the use of beam search to identify more prominent draft tokens. For more details, please read [the ReDrafter paper](https://arxiv.org/html/2403.09919v1).

TensorRT-LLM implements the ReDrafter model such that logits prediction, beam search, and draft token acceptance are performed inside the TensorRT engine. This contrasts with standard model inference, which only predicts logits and performs decoding outside the engine. Since the engine predicts explicit draft tokens instead of implicit tokens decoded from logits, we categorize this speculative decoding method as `explicit_draft_tokens`. Please, visit the [ReDrafter README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/redrafter/README.md) for information about building and running the model. ReDrafter supports both Inflight Fused Batching runtime and Python static batching runtime.

## EAGLE

The EAGLE approach enhances the single-model Medusa method by predicting and verifying tokens using the same model. Similarly to ReDrafter, it predicts draft tokens using a recurrent predictor where each draft token depends on the previous one. However, unlike ReDrafter, it uses a single-layer transformer model to predict draft tokens from previous hidden states and decoded tokens. In the EAGLE-1 decoding tree needs to be known during the decoding. In the EAGLE-2 this tree is asssembled during the execution by searching for the most probable hypothesis along the beam.

Similarly to ReDrafter, TensorRT-LLM implements the EAGLE model such that logits prediction, draft tokens acceptance and draft token generation are performed inside of the TensorRT engine(EAGLE-1 and EAGLE-2 are both supported). Please, visit the [EAGLE README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/eagle/README.md) for information about building and running the model.

### Disaggregated Serving

[Disaggregated Serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disaggregated-service.md) with EAGLE3 using the two model approach is supported in the Pytorch backend. Please refer to the following [Dynamo example](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/llama4_plus_eagle.md) on how to run EAGLE3 with Disaggregated Serving for Llama 4 Maverick.

## Lookahead Decoding

Lookahead decoding algorithm operates through two parallel computation branches within the same model: a lookahead branch that generates n-grams using a fixed-sized 2D window, and a verification branch that validates promising n-gram candidates. This approach eliminates the necessity for additional model training or fine-tuning and can be enabled for any autoregressive model. Refer to the [Lookahead decoding README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/lookahead/README.md) for information about building and running the model.
