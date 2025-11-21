(additional-outputs)=

# Additional Outputs

TensorRT LLM provides several options to return additional outputs from the model during inference. These options can be specified in the `SamplingParams` object and control what extra information is returned for each generated sequence.
For an example showing how to set the parameters and how to access the results, see [examples/llm-api/quickstart_advanced.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/quickstart_advanced.py).

## Options

### `return_context_logits`

- **Description**: If set to `True`, the logits (raw model outputs before softmax) for the context (input prompt) tokens are returned for each sequence.
- **Usage**: Useful for tasks such as scoring the likelihood of the input prompt or for advanced post-processing.
- **Default**: `False`

### `return_generation_logits`

- **Description**: If set to `True`, the logits for the generated tokens (tokens produced during generation) are returned for each sequence.
- **Usage**: Enables advanced sampling, custom decoding, or analysis of the model's output probabilities for generated tokens.
- **Default**: `False`

### `prompt_logprobs`

- **Description**: If set to an integer value `N`, the top-`N` log probabilities for each prompt token are returned, along with the corresponding token IDs.
- **Usage**: Useful for analyzing how likely the model considers each input token, scoring prompts, or for applications that require access to the token-level log probability of the prompt.
- **Default**: `None`

### `logprobs`

- **Description**: If set to an integer value `N`, the top-`N` log probabilities for each generated token are returned, along with the corresponding token IDs.
- **Usage**: Useful for uncertainty estimation, sampling analysis, or for applications that require access to the probability distribution over tokens at each generation step.
- **Default**: `None` (no log probabilities returned)

### `additional_model_outputs`

- **Description**: Specifies extra outputs to return from the model during inference. This should be a list of strings, where each string corresponds to the name of a supported additional output (such as "hidden_states" or "attentions").
- **Usage**: Allows retrieval of intermediate model results like hidden states, attentions, or any other auxiliary outputs supported by the model. This can be useful for debugging, interpretability, or advanced research applications.
- **How to use**:
  - Provide a list of supported output names, e.g.:

    ```python
    additional_model_outputs=["hidden_states", "attentions"]
    ```

  - Pass this list to the `additional_model_outputs` parameter of `SamplingParams`.
  - After generation, access the results per sequence via `sequence.additional_context_outputs` (for context outputs)
  and `sequence.additional_generation_outputs` (for generation outputs).
- **Default**: `None` (no additional outputs returned)

**Note:** The available output names depend on the model implementation. The model forward function is expected to return a dictionary of model outputs including the `"logits"` and any additional output that should be attached to responses.
