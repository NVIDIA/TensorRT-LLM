# Mistral Large V3

* Setup the model path

```bash
export mistral_large_3_model_path=<mistral_large_3_model_path>
export mistral_large_3_eagle_model_path=<mistral_large_3_eagle_model_path>
```

## Multimodal run

* Run the Mistral Large V3 by `quickstart_multimodal.py`

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_multimodal.py \
    --model_dir ${mistral_large_3_model_path} \
    --tp_size 4 \
    --moe_ep_size 4 \
    --max_tokens 100 \
    --checkpoint_format mistral \
    --model_type mistral_large_3 \
    --moe_backend TRTLLM \
    --image_format pil
```

## LLM-only run

* Run the Mistral Large V3 by `quickstart_advanced.py`

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py \
    --model_dir ${mistral_large_3_model_path} \
    --tp_size 4 \
    --moe_ep_size 4 \
    --max_tokens 100 \
    --checkpoint_format mistral \
    --moe_backend TRTLLM
```

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py \
    --model_dir ${mistral_large_3_model_path} \
    --tp_size 4 \
    --moe_ep_size 4 \
    --max_tokens 100 \
    --checkpoint_format mistral \
    --spec_decode_algo EAGLE3 \
    --spec_decode_max_draft_len 1 \
    --use_one_model \
    --draft_model_dir ${mistral_large_3_eagle_model_path} \
    --eagle3_model_arch mistral_large3 \
    --moe_backend TRTLLM
```


* Launch the trtllm-serve and send a request

```bash
echo "
backend: pytorch
tensor_parallel_size: 4
moe_expert_parallel_size: 4
checkpoint_format: mistral
" > serve.yml
mpirun -n 1 --allow-run-as-root --oversubscribe python3 -m tensorrt_llm.commands.serve serve \
    ${mistral_large_3_model_path} \
    --host localhost --port 8001 --backend pytorch \
    --config serve.yml \
    --tokenizer ${mistral_large_3_model_path} \
    2>&1 | tee serve_debug.log &

curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "${mistral_large_3_model_path}",
      "prompt": "The capital of France is",
      "max_tokens": 16,
      "top_k": 16
  }'

# The result would be like
{"id":"cmpl-7e342c1d722d4226a1bf3ed35d762c35","object":"text_completion","created":1764061351,"model":"${mistral_large_3_model_path}","choices":[{"index":0,"text":"The capital of France is **Paris**.\n\nParis is the largest city in France and","token_ids":null,"logprobs":null,"context_logits":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null,"avg_decoded_tokens_per_iter":1.0}],"usage":{"prompt_tokens":7,"total_tokens":23,"completion_tokens":16,"prompt_tokens_details":{"cached_tokens":1}},"prompt_token_ids":null}
```
