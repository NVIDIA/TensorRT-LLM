# Mistral Large V3

* Setup the model path

```bash
export mistral_large_3_model_path=<mistral_large_3_model_path>
export mistral_large_3_eagle_model_path=<mistral_large_3_eagle_model_path>
```

## LLM-only run

* Run the Mistral Large V3 by `quickstart_advanced.py`

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py \
    --model_dir ${mistral_large_3_model_path} \
    --tp_size 4 \
    --moe_ep_size 4 \
    --max_tokens 100 \
    --checkpoint_format mistral_large_3 \
    --kv_cache_fraction 0.25 \
    --moe_backend TRTLLM # optional
```

* Run the Mistral Large V3 by `quickstart_advanced.py` with Eagle3. 

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py \
    --model_dir ${mistral_large_3_model_path} \
    --tp_size 4 \
    --moe_ep_size 4 \
    --max_tokens 10 \
    --checkpoint_format mistral_large_3 \
    --kv_cache_fraction 0.25 \
    --disable_kv_cache_reuse \
    --spec_decode_algo EAGLE3 \
    --spec_decode_max_draft_len 1 \
    --use_one_model \
    --draft_model_dir ${mistral_large_3_eagle_model_path} \
    --moe_backend TRTLLM \
    --print_iter_log \
    2>&1 | tee debug.log
```

* Launch the trtllm-serve and send a request

```bash
echo "
backend: pytorch
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
kv_cache_config:
  free_gpu_memory_fraction: 0.25
  enable_block_reuse: true
checkpoint_format: mistral_large_3
" > serve.yml
mpirun -n 1 --allow-run-as-root --oversubscribe python3 -m tensorrt_llm.commands.serve serve \
    ${mistral_large_3_model_path} \
    --host localhost --port 8001 --backend pytorch \
    --extra_llm_api_options serve.yml \
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

* Launch the trtllm-serve with eagle3 and send a request

```bash
echo "
backend: pytorch
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
kv_cache_config:
  free_gpu_memory_fraction: 0.25
  enable_block_reuse: true
checkpoint_format: mistral_large_3
speculative_config:
    decoding_type: Eagle
    max_draft_len: 1
    speculative_model_dir: ${mistral_large_3_eagle_model_path}
    eagle3_one_model: true
" > serve.yml
mpirun -n 1 --allow-run-as-root --oversubscribe python3 -m tensorrt_llm.commands.serve serve \
    ${mistral_large_3_model_path} \
    --host localhost --port 8001 --backend pytorch \
    --extra_llm_api_options serve.yml \
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
```

## How to use the modules

The following explains how to use the different modules of Mistral Large V3.

```python
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3ForCausalLM
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM
from tensorrt_llm.llmapi.tokenizer import MistralTokenizer
from tensorrt_llm._torch.models.checkpoints.mistral.checkpoint_loader import MistralCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import MistralLarge3WeightMapper
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
```

### Tokenizer
```python
mtok = MistralTokenizer.from_pretrained(TOKENIZER_DIR)
```

### Config and model instance
```python
config_loader = MistralConfigLoader()
config = config_loader.load(MODEL_DIR)

model = Mistral3VLM(model_config=config)
assert isinstance(model.llm, DeepseekV3ForCausalLM)
```

### Checkpoint loading
```python
weight_mapper=MistralLarge3WeightMapper()
loader = MistralCheckpointLoader(weight_mapper=weight_mapper)

weights_dict = loader.load_weights(MODEL_DIR)
```

### Weight loading
#### E2E
```python
model.load_weights(weights_dict, weight_mapper=weight_mapper) # target usage
```
#### By module
```python
def _filter_weights(weights, prefix):
    return {
        name[len(prefix):]: weight
        for name, weight in weights.items() if name.startswith(prefix)
    }

llm_weights = weight_mapper.rename_by_params_map(
    params_map=weight_mapper.mistral_llm_mapping,
    weights=_filter_weights(weights_dict, "language_model."))
model.llm.load_weights(llm_weights, weight_mapper=weight_mapper)
```
