# TensorRT-LLM Autodeploy Config Examples

## Required Fields

All autodeploy configs must contain:

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
```

## Basic Config (Llama-like Models)

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 64
max_seq_len: 4096
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.9
```

## Config with FP8 GEMM Fusion

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 64
max_seq_len: 4096
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.9
transforms:
  fuse_fp8_gemms:
    stage: post_load_fusion
    run_shape_prop: false
    run_graph_cleanup: false
    enabled: true
```

## Nano v3 Config (Mamba + Attention Hybrid)

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 384
max_seq_len: 65536
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384]
kv_cache_config:
  free_gpu_memory_fraction: 0.88
  mamba_ssm_cache_dtype: auto  # or float32 for accuracy
transforms:
  detect_sharding:
    allreduce_strategy: SYMM_MEM
    sharding_dims: ['ep', 'bmm']
    manual_config:
      head_dim: 128
      tp_plan:
        # Mamba SSM layer
        "in_proj": "mamba"
        "out_proj": "rowwise"
        # Attention layer
        "q_proj": "colwise"
        "k_proj": "colwise"
        "v_proj": "colwise"
        "o_proj": "rowwise"
        # MoE layer: shared experts
        "up_proj": "colwise"
        "down_proj": "rowwise"
        # MoLE: latent projections
        "fc1_latent_proj": "gather"
        "fc2_latent_proj": "gather"
  multi_stream_moe:
    stage: compile
    enabled: true
  gather_logits_before_lm_head:
    enabled: true
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
  insert_cached_ssm_attention:
    backend: flashinfer_ssm
```

## Super v3 Config (Similar to Nano but Single Stream MoE)

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 384
max_seq_len: 65536
enable_chunked_prefill: true
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
skip_loading_weights: false
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384]
kv_cache_config:
  mamba_ssm_cache_dtype: auto
transforms:
  detect_sharding:
    allreduce_strategy: SYMM_MEM
    sharding_dims: ['ep', 'bmm']
    manual_config:
      head_dim: 128
      tp_plan:
        "in_proj": "mamba"
        "out_proj": "rowwise"
        "q_proj": "colwise"
        "k_proj": "colwise"
        "v_proj": "colwise"
        "o_proj": "rowwise"
        "up_proj": "colwise"
        "down_proj": "rowwise"
        "fc1_latent_proj": "gather"
        "fc2_latent_proj": "gather"
  multi_stream_moe:
    stage: compile
    enabled: false  # Key difference from nano_v3
  gather_logits_before_lm_head:
    enabled: true
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
  insert_cached_ssm_attention:
    backend: flashinfer_ssm
```

## Common Transform Options

### FP8 GEMM Fusion
```yaml
transforms:
  fuse_fp8_gemms:
    stage: post_load_fusion
    run_shape_prop: false  # optional
    run_graph_cleanup: false  # optional
    enabled: true
```

### Multi-Stream MoE
```yaml
transforms:
  multi_stream_moe:
    stage: compile
    enabled: true
```

### Mamba A-Log Fusion
```yaml
transforms:
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
```

### Tensor Parallelism Sharding
```yaml
transforms:
  detect_sharding:
    allreduce_strategy: SYMM_MEM
    sharding_dims: ['ep', 'bmm']
    manual_config:
      head_dim: 128
      tp_plan:
        "q_proj": "colwise"
        "k_proj": "colwise"
        "v_proj": "colwise"
        "o_proj": "rowwise"
```

## KV Cache Config Options

```yaml
kv_cache_config:
  dtype: fp8  # or auto, float16, int8
  free_gpu_memory_fraction: 0.9  # 0.0-1.0
  mamba_ssm_cache_dtype: auto  # or float32 (for Mamba models)
```

## Multi-GPU Config

For multi-GPU deployments, add `world_size` to the config:

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
world_size: 8  # Number of GPUs
max_batch_size: 384
# ... rest of config
```
