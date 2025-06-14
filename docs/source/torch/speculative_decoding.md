# Speculative Decoding

Two flavors of speculative decoding are currently supported in the PyTorch backend:
- A variant which inserts a drafter directly into the model code as a submodule. We refer to this as the "one model" implementation.
- A variant which produces draft tokens in the `PyExecutor`. The draft tokens are attached to requests before they are passed
into the target model's `ModelEngine`. We refer to this as the "two model" implementation.

In general, the one model implementation is faster. It's able to achieve better performance in extreme low latency
scenarios because it can launch the entire drafting loop as a single CUDA graph. The trade off is flexibility. The one model implementation
does not support dynamic draft lengths. Additionally, only a subset of models/speculative decoding algorithms support the one model implementation.
The table below enumerates all of the algorithm/model combinations that are supported.

| Speculative Decoding Algorithm | Model                          |
| ------------------------------ | ------------------------------ |
| EAGLE 3                        | Llama 4 Scout/Maverick         |
| MTP                            | Deepseek V3/R1                 |
| EAGLE-style MTP                | Deepseek V3/R1                 |

The two model implementation supports the following speculative decoding algorithms:

| Speculative Decoding Algorithm                        | Model                                                 |
| ----------------------------------------------------- | ----------------------------------------------------- |
| EAGLE 3                                               | Llama 4 Scout/Maverick *more general support planned* |
| Draft/target                                          | All models                                            |
| NGram                                                 | All models                                            |

The goal of this document is to describe the *general architecture* for speculative decoding,
not any individual algorithm. Thus, algorithm-specific details will be omitted except where
they are necessary to explain architecture decisions.

## Usage

To use speculative decoding, one simply has to pass the appropriate `speculative_config` when
creating an `LLM`. For example, to use EAGLE 3:

```python
from tensorrt_llm.llmapi import EagleDecodingConfig

speculative_config = EagleDecodingConfig(max_draft_len=3, pytoch_eagle_weights_path=/path/to/draft_model)

llm = LLM(..., speculative_config=speculative_config)
```

In general, the speculative config objects let you configure properties like the maximum draft
length. If the speculation algorithm supports both the one model and two model flavors, the
config will contain a flag to switch between the two implementations. Details on each speculative
algorithm can be found in `tensorrt_llm/llmapi/llm_args.py`.

## Components of a Speculative Decoding

Most of the components are defined in `_torch/speculative/interface.py`.

1. `SpeculativeDecodingMode`: this is a simple `IntEnum`, one for each supported algorithm. There are a few
nontrivial methods, however.
- `needs_kv_cache_rewind`. See "KV Cache Rewind" below. In general, this will be true for all two model speculative
decoding algorithms.
- `extend_ctx`: If true, the speculative decoding will dispatch requests with `py_draft_tokens` attached to them
to the *prefill* version of the attention kernels. This usually needs to be true. The exception is when you're on
Blackwell using the TRTLLM attention backend. In that case, we can use the generation kernels for better performance.
This optimized kernel has one limitation; all draft lengths must be the same (or padding must be used) in this case.

> *It is highly likely that these will be refactored in the near future to reduce the difficulty of adding a new speculative
decoding algorithm. `extend_ctx` in particular is problematic. Ideally, we would move all of the kernel dispatching logic
to a lower level of abstraction.*

2. `SpecConfig`: Basic configuration derived from the `speculative_config` given to the `LLM`. Each speculative decoding
algorithm defines a subclass of `SpecConfig`. There are two methods that can be overridden:

- `update_from_model_config`: This takes a `ModelConfig` from the target model. By default, it does nothing. It can
be used to set special attributes that cannot be known until the target model configuration is known. For example,
`Eagle3Config` overrides this to store the target model's `hidden_size` (this is later passed to the `SpecMetadata` to facilitate
hidden state capture).

- `get_draft_model_prompt`: This takes a list of tokens and returns a list of tokens. It defines the prompt that the draft model should
get the first time it is run for a particular request. By default, it just returns its input. Usually, you won't want to mess with
this - we only need it for special algorithms like EAGLE 3 which throw away the first token of each prompt.

`SpecConfig` objects will always have a corresponding `SpeculativeDecodingMode` instance attached to them.

3. `SpecMetadata`: Defines all metadata that should be passed to the model during the forward pass to facilitate speculative decoding.
Each speculative decoding algorithm defines a subclass of `SpecMetadata`. Similar to `AttentionMetadata`, each `CUDAGraphRunner` owns
its own `SpecMetadata`, and CUDA-graph compatible `SpecMetadata` objects may be created by invoking `create_cuda_graph_metadata(batch_size)`.
`SpecMetadata` has lots of fields. Many of them are exclusively used by the one model implementation. For the two model implementation, the
main purpose of `SpecMetadata` is to facilitate the capture of hidden states. In EAGLE 3, we need to capture hidden states from the
target model to use as draft model inputs. The `SpecMetadata` stores a list of layers to capture and the model calls
`maybe_capture_hidden_states(layer_id, hidden_states, residual)` during its forward pass. If the layer ID is in the list of layers to capture,
the hidden states are saved. For CUDA graph compatibility, these may be saved in pre-allocated buffers.

`SpecMetadata` is derived from a `SpecConfig` object in `_torch/speculative/utils.py`. There are a few other optional components created in
this file too:

4. `ResourceManager`: We can create a custom resource manager to prepare and free resources before and after target forward passes; see
the section on `ResourceManager` in `arch.md`. This is used by the n-gram method to manage its pool. The one model implementation also uses
`ResourceManager`s to manage hidden states.

5. `Sampler`: Each speculative decoding algorithm can optionally create its own sampler. This is mostly used by the one model implementation.
The default `TorchSampler` is used as a fallback if no custom sampler is provided. EAGLE 3 two model also has a simple custom decoder to handle
differences in the draft/target model vocab sizes.

6. `Worker`: This is exclusive to the one-model implementation. The `Worker` is the object that gets injected into the target model as a
submodule.

## Two Model Speculative Decoding Architecture

Note that there are currently a few limitations on the two model implementation:
* KV cache reuse must be disabled.
* Overlap scheduling must be disabled.

In this approach, we introduce two new steps to the `PyExecutor`'s `_executor_loop`.
* `_prepare_draft_requests`
* `_prepare_draft_tokens`

### `_prepare_draft_requests`

This stage occurs for all speculative decoding algorithms before scheduling. The purpose
of this stage is to make the KV cache and scheduler aware of the fact that speculative decoding
will occur. Draft tokens take up extra KV cache pages and count towards the executor's
`max_num_tokens` limit. Thus, we need a way to tell the scheduler that drafting will occur
**before we do the scheduling**.

To achieve this, we simply attach the maximum number of draft tokens to each request. The
scheduler and KV cache manager will automatically account for tokens attached to the
`py_draft_tokens` attribute.

```python
for req in self.active_requests:
    req.py_draft_tokens = [0] * max_draft_len
```

### `_prepare_draft_tokens`

This stage occurs after scheduling and KV cache allocation. The purpose of this stage
is to attach draft tokens to the `py_draft_tokens` attribute. The target `ModelEngine` and
`Sampler` will handle the verification after this stage. This stage involves invoking a draft
`ModelEngine` one or more times.

*Note for n-gram speculative decoding*: This is a special case because the draft tokens
are not produced by a draft model. In this case, there is a `ResourceManager` that attaches
`py_draft_tokens` to each request. `_prepare_draft_tokens` is skipped, but the target model
and sampler verification infra are the same.

The `_prepare_draft_tokens` stage breaks down into a few steps:

1. Pre-process the requests. We first produce a `ScheduledRequests` object derived from the
target model's `ScheduledRequests`. The rules are simple in the general case: if it's the
first time the draft model is seeing the request, the draft model will get a context request.
Otherwise, it will get a generation request. There is one special case - for EAGLE 3, we need
to recompute KV cache for tokens that were accepted by the draft model on the previous iteration.
Hence, if there were any accepted tokens for this case, the draft model gets a chunked prefill
request containing all of the previously accepted draft tokens plus the new token.

2. Run the draft model with the pre-processed batch. This may involve querying the last used
`SpecMetadata` object to collect captured hidden states if they are required for drafting.

3. Run the sampler on the draft model's outputs. Add the decoded tokens to the corresponding
target model's `request.py_draft_tokens`. Remove any finished requests (e.g. due to EOS) from
the draft model batch.

4. Repeat steps 2-3 to run the draft model autoregressively until all requests are complete
(emitted EOS, hit the draft model's max sequence length, or hit the maximum draft length).

In addition to producing all "real" draft tokens, `_prepare_draft_tokens` currently must also pad
all `py_draft_tokens` to the maximum draft length. This is a CUDA graph limitation - the target
model will capture its CUDA graphs using the maximum number of draft tokens on each request.

### Verification and Sampling

Once the draft tokens are obtained, the target model runs a forward pass through the usual flow.
Everything is the same, except that the logits for all the draft tokens are returned and passed
to the sampler.

Currently, only greedy sampling is supported for speculative decoding. A draft token is accepted if
matches the previously decoded token exactly. For example, suppose we have a generation request
`[t, d1, d2, d3]`, where `d1`, `d2`, and `d3` are drat tokens. Suppose the token after `t` is `d1`
(determined with the `argmax` of the logits). `d1` is then accepted. If the token after `d1` is `d2`,
then we can accept `d2`. And so on until we can't accept draft tokens anymore.

### KV Cache Rewind

KV cache space allocated to rejected tokens is freed before the next iteration. This is achieved by setting
the `request.py_rewind_len` attribute to `num_draft_tokens_allocated - num_accepted_tokens`. The pages are
freed as part of the `resource_manager.free_resources` routine.

The purpose of KV cache rewind is to avoid complicated page reuse logic in the KV cache manager's `prepare_resources`
function. In practice, this is very cheap since we're just marking blocks as available and not actually
invoking the CUDA allocator.
