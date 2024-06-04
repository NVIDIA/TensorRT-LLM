(inference-request)=

# Inference Request

The main class to describe requests to `GptManager` is `InferenceRequest`. This is structured as a map of tensors and a `uint64_t requestId`.
The mandatory input tensors to create a valid `InferenceRequest` object are described below. Sampling config params are documented in the {ref}`gpt-runtime` section. Descriptions have been omitted in the table.

| Name | Shape | Type | Description |
| :----------------------: | :----------------------------: | :-----------------------------: | :-----------------------------: |
| `request_output_len` | [1,1] | `int32_t` | Max number of output tokens |
| `input_ids` | [1, num_input_tokens] | `int32_t` | Tensor of input tokens |

Optional tensors that can be supplied to `InferenceRequest` are shown below. Default values, where applicable are specified.:
| Name | Shape | Type | Description |
| :----------------------: | :----------------------------: | :-----------------------------: | :-----------------------------: |
| `streaming` | [1] | `bool` | (Default=`false`). When `true`, stream out tokens as they are generated. When `false` return only when the full generation has completed.  |
| `beam_width` | [1] | `int32_t` | (Default=1) Beam width for this request; set to 1 for greedy sampling |
| `temperature` | [1] | `float` | Sampling Config param: `temperature` |
| `runtime_top_k` | [1] | `int32_t` | Sampling Config param: `topK` |
| `runtime_top_p` | [1] | `float` | Sampling Config param: `topP` |
| `len_penalty` | [1] | `float` | Sampling Config param: `lengthPenalty` |
| `early_stopping` | [1] | `int` | Sampling Config param: `earlyStopping` |
| `repetition_penalty` | [1] | `float` | Sampling Config param: `repetitionPenalty` |
| `min_length` | [1] | `int32_t` | Sampling Config param: `minLength` |
| `presence_penalty` | [1] | `float` | Sampling Config param: `presencePenalty` |
| `frequency_penalty` | [1] | `float` | Sampling Config param: `frequencyPenalty` |
| `no_repeat_ngram_size` | [1] | `int32_t` | Sampling Config param: `noRepeatNgramSize` |
| `random_seed` | [1] | `uint64_t` | Sampling Config param: `randomSeed` |
| `end_id` | [1] | `int32_t` | End token Id. If not specified, defaults to -1 |
| `pad_id` | [1] | `int32_t` | Pad token Id |
| `embedding_bias` | [1, vocab_size] | `float` | The bias is added to the logits for each token in the vocabulary before decoding occurs. Positive values in the bias encourage the sampling of tokens, while negative values discourage it. A value of `0.f` leaves the logit value unchanged. |
| `bad_words_list` | [2, num_bad_words] | `int32_t` | Bad words list |
| `stop_words_list` | [2, num_stop_words] | `int32_t` | Stop words list |
| `prompt_embedding_table` | [1] | `float16` | P-tuning prompt embedding table |
| `prompt_vocab_size` | [1] | `int32_t` | P-tuning prompt vocab size |
| `lora_task_id` | [1] | `uint64_t` | Task ID for the given lora_weights.  This ID is expected to be globally unique.  To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.  The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`. If the cache is full the oldest LoRA will be evicted to make space for new ones.  An error is returned if `lora_task_id` is not cached |
| `lora_weights` | [num_lora_modules_layers, D x Hi + Ho x D] | `float` (model data type) | weights for a LoRA adapter. Refer to {ref}`lora` for more information. |
| `lora_config` | [num_lora_modules_layers, 3] | `int32_t` | LoRA configuration tensor. `[ module_id, layer_idx, adapter_size (D aka R value) ]` Refer to {ref}`lora` for more information. |
| `return_log_probs` | [1] | `bool` | When `true`, include log probs in the output |
| `return_context_logits` | [1] | `bool` | When `true`, include context logits in the output |
| `return_generation_logits` | [1] | `bool` | When `true`, include generation logits in the output |
| `draft_input_ids` | [num_draft_tokens] | `int32_t` | Draft tokens to be leveraged in generation phase to potentially generate multiple output tokens in one inflight batching iteration |
| `draft_logits` | [num_draft_tokens, vocab_size] | `float` | Draft logits associated with `draft_input_ids` to be leveraged in generation phase to potentially generate multiple output tokens in one inflight batching iteration |

# Responses

Responses from GptManager are formatted as a list of tensors. The table below shows the set of output tensors returned by `GptManager` (via the `SendResponseCallback`):
| Name | Shape | Type | Description |
| :----------------------: | :----------------------------: | :-----------------------------: | :-----------------------------: |
| `output_ids` | [beam_width, num_output_tokens] | `int32_t` | Tensor of output tokens. When `streaming` is enabled, this is a single token. |
| `sequence_length` | [beam_width] | `int32_t` | Number of output tokens. When `streaming` is set, this will be 1. |
| `output_log_probs` | [1, beam_width, num_output_tokens] | `float` | Only if `return_log_probs` is set on input. Tensor of log probabilities of output token logits. |
| `cum_log_probs` | [1, beam_width] | `float` | Only if `return_log_probs` is set on input. Cumulative log probability of the sequence generated. |
| `context_logits` | [1, num_input_tokens, vocab_size] | `float` | Only if `return_context_logits` is set on input. Tensor of input token logits. |
| `generation_logits` | [1, beam_width, num_output_tokens, vocab_size] | `float` | Only if `return_generation_logits` is set on input. Tensor of output token logits. |
