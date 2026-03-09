# Architecture-Class Config Templates

Basic aggregate (IFB, single node, PyTorch, non-speculative) starting-point `extra_llm_api_options` YAML configs. Use when no database or curated config matches. Classify the model first (see `architecture.md`), then copy the matching template and adjust the 4 critical knobs per `knob-heuristics.md`.

## MoE + MLA (DeepSeek-R1, DeepSeek-V3, Kimi-K2)

```yaml
backend: pytorch
max_num_tokens: 2048          # adjust: see knob-heuristics.md §max_num_tokens
max_seq_len: 2068             # >= ISL + OSL + 68
tensor_parallel_size: 8       # your TP
moe_expert_parallel_size: 8   # must equal TP
moe_config:
  backend: TRTLLM             # adjust: see knob-heuristics.md §moe_backend
enable_attention_dp: false     # adjust: see knob-heuristics.md §attention_dp
kv_cache_config:
  free_gpu_memory_fraction: 0.80  # adjust: see knob-heuristics.md §kv_fraction
  dtype: fp8
cuda_graph_config:
  enable_padding: true
  max_batch_size: 512          # = target concurrency (cap 512 for MoE)
trust_remote_code: true
stream_interval: 10
```

## MoE + GQA (Qwen3-235B-A22B)

```yaml
backend: pytorch
max_num_tokens: 2048
max_seq_len: 2068
tensor_parallel_size: 8
moe_expert_parallel_size: 8
moe_config:
  backend: TRTLLM
enable_attention_dp: false
kv_cache_config:
  free_gpu_memory_fraction: 0.80
  dtype: fp8
cuda_graph_config:
  enable_padding: true
  max_batch_size: 256
trust_remote_code: true
stream_interval: 10
```

## Dense + GQA (Llama-3.3-70B, GPT-OSS-120B, Nemotron)

```yaml
backend: pytorch
max_num_tokens: 2048
max_seq_len: 2068
tensor_parallel_size: 4
enable_attention_dp: false
kv_cache_config:
  free_gpu_memory_fraction: 0.85
  dtype: fp8
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024
trust_remote_code: true
stream_interval: 10
```

Note: Dense models omit `moe_config` and `moe_expert_parallel_size` entirely.
