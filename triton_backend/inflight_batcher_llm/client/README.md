# Sample TRT-LLM backend clients
Three sample TRT-LLM Triton clients are provided with the TRT-LLM Triton backend implementation.
* `e2e_grpc_speculative_decoding_client.py`: Demonstrates how to orchestrate between two independent TRT-LLM models - a draft model and a target model to achiever faster inferencing using speculative decoding. The high level design involves the client making a call to the draft model requesting a certain number of draft tokens, and then associating those draft tokens with a request to the target model. The target model returns some number of completion tokens internally leveraging the draft tokens to speed up inference. The client wraps these back-to-back calls to draft and target models in a loop to complete the full generation.
Example command:
```
python3 e2e_grpc_speculative_decoding_client.py -p "The only thing we have to fear is" \
              --url-draft ${DRAFT_MODEL_URL} \
              --url-target ${TARGET_MODEL_URL}
```
To get draft model draft tokens's logits, you need to enable `gather_generation_logits` when building then engine, and add `--return-draft-model-draft-logits` when running `e2e_grpc_speculative_decoding_client.py`.

To get the target model accepted tokens's logits, you need to enable `gather_generation_logits` when building the engine, and add `--return-target-model-accepted-token-logits` when running `e2e_grpc_speculative_decoding_client.py`.


* `end_to_end_grpc_client.py`: Demonstrates sending a single request to a tritonserver running an ensemble including preprocessor (tokenizer), TRT-LLM model and postprocessor (detokenizer) and getting back a completion from it.
Example command:
```
python3 end_to_end_grpc_client.py \
        --streaming --output-len 10 \
        --prompt "The only thing we have to fear is"

```
* `inflight_batcher_llm_client.py`: Isolates queries and responses to the TRT-LLM model alone. Invokes tokenizer and detokenizer in the client script i.e. outside the server running inference.
Example command:
```
python3 inflight_batcher_llm_client.py \
            --tokenizer-dir ${TOKENIZER_PATH} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --input-tokens-csv=${LOGDIR}/prompts.csv \
            --output-tokens-csv=${LOGDIR}/completions.csv
```
