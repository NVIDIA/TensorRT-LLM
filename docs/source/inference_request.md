# Inference Request

The main class to describe requests to `GptManager` is `InferenceRequest`. This is structured as a map of tensors and a `uint64_t requestId`.
The mandatory tensors to create a valid `InferenceRequest` object are described below. Sampling Config params are documented in more detail [here](gpt_runtime.md#sampling-parameters), and descriptions are omitted in the table:
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
| `repetition_penalty` | [1] | `float` | Sampling Config param: `repetitionPenalty` |
| `min_length` | [1] | `int32_t` | Sampling Config param: `minLength` |
| `presence_penalty` | [1] | `float` | Sampling Config param: `presencePenalty` |
| `frequency_penalty` | [1] | `float` | Sampling Config param: `frequencyPenalty` |
| `random_seed` | [1] | `uint64_t` | Sampling Config param: `randomSeed` |
| `end_id` | [1] | `int32_t` | End token Id |
| `pad_id` | [1] | `int32_t` | Pad token Id |
| `embedding_bias` | [1] | `float` | Embedding bias |
| `bad_words_list` | [2, num_bad_words] | `int32_t` | Bad words list |
| `stop_words_list` | [2, num_stop_words] | `int32_t` | Stop words list |
| `prompt_embedding_table` | [1] | `float16` | P-tuning prompt embedding table |
| `prompt_vocab_size` | [1] | `int32_t` | P-tuning prompt vocab size |
| `return_log_probs` | [1] | `bool` | When `true`, include log probs in the output |
| `draft_input_ids` | [num_draft_tokens] | `int32_t` | Draft tokens to be leveraged in generation phase to potentially generate multiple output tokens in one inflight batching iteration |
