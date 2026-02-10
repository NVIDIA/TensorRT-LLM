---
name: trtllm-autodeploy
description: Benchmark and deploy LLMs using TensorRT-LLM's autodeploy backend (_autodeploy) with graph transformations and optimizations. Use when working with TensorRT-LLM and the user needs to (1) benchmark models with autodeploy backend, (2) configure autodeploy graph transformations, (3) run online benchmarks with trtllm-serve + aiperf, (4) run offline benchmarks with trtllm-bench, or (5) compare autodeploy vs pytorch backends.
---

# TensorRT-LLM Autodeploy Backend

TensorRT-LLM's autodeploy backend (`_autodeploy`) provides graph transformations and optimizations for LLM inference. Unlike the standard `pytorch` backend, autodeploy applies compile-time optimizations through configurable transforms.

## Quick Start

### Online Benchmarking (Server + Client)

```bash
# Start server with autodeploy backend
trtllm-serve <checkpoint_id> \
  --backend _autodeploy \
  --extra_llm_api_options config.yaml \
  --trust_remote_code

# Run benchmark with aiperf
aiperf profile \
  --model <checkpoint_id> \
  --url 0.0.0.0:8123 \
  --endpoint-type chat \
  --streaming \
  --concurrency 256 \
  --isl 1024 --osl 1024
```

### Offline Benchmarking (Maximum Throughput)

```bash
# Prepare dataset
python3 benchmarks/cpp/prepare_dataset.py \
  --stdout --tokenizer <checkpoint_id> \
  token-norm-dist --input-mean 1024 --output-mean 1024 \
  --num-requests 256 > dataset.txt

# Run benchmark
trtllm-bench --model <checkpoint_id> throughput \
  --dataset dataset.txt \
  --backend _autodeploy \
  --extra_llm_api_options config.yaml \
  --max_batch_size 256
```

## Configuration Requirements

All autodeploy configs **must** include:

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
```

**Note**: 100+ models have pre-configured settings at `examples/auto_deploy/model_registry/`. See [references/model_registry.md](references/model_registry.md) for details.

Minimal working config:

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
max_batch_size: 256
max_seq_len: 65536
attn_backend: flashinfer
model_factory: AutoModelForCausalLM
kv_cache_config:
  dtype: fp8
  free_gpu_memory_fraction: 0.9
```

## Configuration Workflow

### 1. Choose Base Configuration

**Option A: Use Model Registry** (if your model is supported)

TensorRT-LLM provides 100+ pre-configured models at `examples/auto_deploy/model_registry/`:

```bash
# Check if your model is in the registry
grep "your-model" examples/auto_deploy/model_registry/models.yaml

# Use registry configs as templates or starting points
# See references/model_registry.md for details
```

**Option B: Generate from Scratch**

Use `scripts/generate_config.py` to create initial configs:

```bash
# Basic transformer model
python scripts/generate_config.py -o config.yaml

# Add transform
python scripts/generate_config.py --add-transform multi_stream_moe enabled=true stage=compile -o config_moe_multistream.yaml

# Set any config with key=value
python scripts/generate_config.py basic -o config.yaml
    --set max_batch_size=256 \
    --set enable_chunked_prefill=true \
    --set cuda_graph_batch_sizes=[1,2,4,8,16,32,64,128,256]

```

### 2. Select Graph Transforms

Common transforms to enable in the `transforms:` section:

**Mamba Optimizations** (for Mamba models):
```yaml
transforms:
  fuse_mamba_a_log:
    stage: post_load_fusion
    enabled: true
  insert_cached_ssm_attention:
    backend: flashinfer_ssm
```

**Tensor Parallelism Sharding** (for multi-GPU):
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

See [references/config_examples.md](references/config_examples.md) for complete examples.

### 3. Adjust Performance Parameters

Key tuning parameters:

```yaml
max_batch_size: 256        # Higher for throughput, lower for latency
max_seq_len: 4096          # Maximum context length
enable_chunked_prefill: true  # Improves latency for long prompts

kv_cache_config:
  dtype: fp8               # fp8, fp16, auto
  free_gpu_memory_fraction: 0.9  # GPU memory for KV cache
  mamba_ssm_cache_dtype: auto    # Mamba only: auto or float32

# Optional: specific CUDA graph batch sizes
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
```

## Benchmarking Workflows

### Online Benchmarking

Online benchmarking measures real-world serving performance with concurrent requests.

**Using bench_server.py** (automated):
```bash
python bench_server.py \
  --model meta-llama/Llama-3.1-70B \
  --config-path config.yaml \
  --server-type trtllm-autodeploy \
  --concurrencies "1 8 32 64" \
  --isl 1024 --osl 1024 \
  --world-size 8
```

**Manual workflow**:
1. Start server: `trtllm-serve MODEL --backend _autodeploy --extra_llm_api_options CONFIG`
2. Wait for "Application startup complete."
3. Run aiperf with desired concurrency levels
4. Analyze results in artifact directories

See [references/online_benchmarking.md](references/online_benchmarking.md) for details.

### Offline Benchmarking

Offline benchmarking measures maximum throughput without network overhead.

**Workflow**:
1. **Prepare dataset** with specific ISL/OSL:
   ```bash
   trtllm-bench --model MODEL prepare-dataset \
     --output dataset.txt \
     token-norm-dist \
     --input-mean 1024 --output-mean 1024 \
     --input-stdev 0 --output-stdev 0 \
     --num-requests 256
   ```

2. **Run benchmark**:
   ```bash
   trtllm-bench --model MODEL throughput \
     --dataset dataset.txt \
     --backend _autodeploy \
     --extra_llm_api_options config.yaml \
     --max_batch_size 256
   ```

3. **Analyze metrics**: throughput (tokens/s), TTFT, ITL, GPU utilization

See [references/offline_benchmarking.md](references/offline_benchmarking.md) for examples.

## Backend Comparison

Compare autodeploy vs pytorch backends:

```bash
# Autodeploy
trtllm-bench --model MODEL throughput \
  --dataset dataset.txt \
  --backend _autodeploy \
  --extra_llm_api_options ad_config.yaml

# PyTorch baseline
trtllm-bench --model MODEL throughput \
  --dataset dataset.txt \
  --backend pytorch \
  --extra_llm_api_options pt_config.yaml
```

Key differences:
- **Autodeploy**: Applies graph transforms at compile time
- **PyTorch**: Standard PyTorch execution without transforms
- **Config structure**: Autodeploy uses `transforms:` dict, PyTorch does not

## Multi-GPU Deployment

For multi-GPU setups with autodeploy:

1. **Add world_size to config**:
   ```yaml
   runtime: trtllm
   compile_backend: torch-cudagraph
   world_size: 8  # Number of GPUs
   ```

2. **Add tensor parallelism sharding** (see config_examples.md)

3. **Run commands**:

   **For trtllm-serve (online benchmarking)**:
   ```bash
   # NO CLI flag needed! world_size from config is used automatically
   trtllm-serve MODEL \
     --backend _autodeploy \
     --extra_llm_api_options config.yaml \
     --trust_remote_code
   ```

   **For trtllm-bench (offline benchmarking)**:
   ```bash
   # NO --tp flag needed! world_size from config is used automatically
   trtllm-bench --model MODEL throughput \
     --backend _autodeploy \
     --extra_llm_api_options config.yaml
   ```

**IMPORTANT**: For autodeploy backend, `world_size` is specified in the config YAML and read automatically by both `trtllm-serve` and `trtllm-bench`. Do NOT use CLI flags like `--tp_size` or `--tp` - they are not needed and may cause errors.

## Profiling

Enable nsys profiling for performance analysis:

**With bench_server.py**:
```bash
python bench_server.py \
  --model MODEL \
  --config-path config.yaml \
  --profile  # Adds nsys profiling
```

**Manual**:
```bash
nsys profile -o trace \
  -t cuda,cublas,nvtx \
  --cuda-graph-trace node \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1 \
  trtllm-serve MODEL --backend _autodeploy --extra_llm_api_options config.yaml
```

## Common Patterns

### ISL/OSL Sweep
Test performance across sequence lengths:
```bash
for isl in 128 512 1024 2048; do
  for osl in 128 512 1024; do
    # Generate dataset
    python3 benchmarks/cpp/prepare_dataset.py \
      --stdout --tokenizer MODEL \
      token-norm-dist \
      --input-mean $isl --output-mean $osl \
      --num-requests 256 > dataset_${isl}_${osl}.txt

    # Benchmark
    trtllm-bench --model MODEL throughput \
      --dataset dataset_${isl}_${osl}.txt \
      --backend _autodeploy \
      --extra_llm_api_options config.yaml
  done
done
```

### Concurrency Sweep
Test online performance across loads:
```bash
for conc in 1 4 8 16 32 64 128; do
  aiperf profile \
    --model MODEL --url 0.0.0.0:8123 \
    --endpoint-type chat --streaming \
    --concurrency $conc \
    --isl 1024 --osl 1024 \
    --artifact-dir results_conc_${conc}
done
```

### Transform Comparison
Compare different transform configurations:
```bash
# Baseline (no transforms)
trtllm-bench --model <checkpoint_id> throughput --dataset dataset.txt --backend _autodeploy --extra_llm_api_options base_config.yaml

# With FP8 GEMM fusion
trtllm-bench --model <checkpoint_id> throughput --dataset dataset.txt --backend _autodeploy --extra_llm_api_options fused_config.yaml

# With multi-stream MoE
trtllm-bench --model <checkpoint_id> throughput --dataset dataset.txt --backend _autodeploy --extra_llm_api_options moe_config.yaml
```

## Troubleshooting

### Server startup timeout
- Autodeploy compiles graphs on first run (takes longer than pytorch)
- Default timeout: 1800s for autodeploy, 600s for pytorch
- Use `--server-startup-timeout` to adjust

### Transform errors
- Ensure `runtime: trtllm` and `compile_backend: torch-cudagraph` are set
- Check transform stage: `post_load_fusion` vs `compile`
- Verify model architecture matches transform (e.g., Mamba transforms for Mamba models)

### Multi-GPU issues
- Ensure `world_size` in config matches the number of available GPUs
- Check CUDA_VISIBLE_DEVICES to verify GPU visibility
- Verify NCCL environment for multi-node setups
- Remember: world_size is read from config automatically, no CLI flags needed

### Memory issues
- Reduce `max_batch_size`
- Lower `free_gpu_memory_fraction`
- Use lower precision KV cache (`kv_cache_config.dtype: fp8`)

## Development Workflow

### Fast Iteration for Debugging

**⚠️ WARNING**: These options are for **debugging/development ONLY**. DO NOT use for:
- Accuracy tests
- Performance benchmarks
- Functionality tests

Always run full model for production testing.

#### Option 1: Skip Loading Weights (Random Weights)

Uses random weights instead of loading from checkpoint:

```yaml
skip_loading_weights: true
```

**Use case**: Test graph transformations, compilation, or runtime issues without weight loading overhead.

**Example**:
```bash
python generate_config.py -o debug_config.yaml \
  --set skip_loading_weights=true
```

#### Option 2: Reduce Number of Layers

Use fewer layers to speed up build time:

```yaml
model_kwargs:
  num_hidden_layers: 2  # or 10, depending on model size
```

**Important for hybrid models**: Include at least one layer of each type (e.g., for Mamba: at least one attention layer and one SSM layer).

**Example**:
```bash
# Standard model - 2 layers
python generate_config.py -o debug_config.yaml \
  --set model_kwargs={num_hidden_layers:2}

# Hybrid model - 10 layers (captures multiple layer types)
python generate_config.py -o debug_config.yaml \
  --set model_kwargs={num_hidden_layers:10}
```

#### Combined Fast Debug Config

```yaml
runtime: trtllm
compile_backend: torch-cudagraph
skip_loading_weights: true
model_kwargs:
  num_hidden_layers: 2
max_seq_len: 512
max_batch_size: 8
```

**Full command**:
```bash
python generate_config.py -o debug_fast.yaml \
  --set skip_loading_weights=true \
  --set model_kwargs={num_hidden_layers:2} \
  --set max_batch_size=8
```

#### Development Cycle

1. **Fast iteration** (debugging transforms, graph issues):
   ```bash
   trtllm-serve MODEL --backend _autodeploy \
     --extra_llm_api_options debug_fast.yaml
   ```

2. **Validate fix** (full model):
   ```bash
   trtllm-serve MODEL --backend _autodeploy \
     --extra_llm_api_options production_config.yaml
   ```

3. **Run tests** (accuracy, performance):
   ```bash
   # Use full model with real weights
   trtllm-bench --model MODEL --backend _autodeploy \
     --extra_llm_api_options production_config.yaml
   ```

#### Debugging tools

1. **Examine autodeploy logs
   AutoDeploy prints at least 4 log lines per transformation.
   The logs can be used to determine which transformations were applied, detailed on applied transformations and summary of transformation timing data.
   Example:
   [timestamp] [TRT-LLM AUTO-DEPLOY] [RANK <GPU ID>] [I] [stage=<transoformation stage>, transform=<transformation name>] [PRE-CLEANUP] <data>
   [timestamp] [TRT-LLM AUTO-DEPLOY] [RANK <GPU ID>] [I] [stage=post_export, transform=cleanup_input_constraints] [APPLY] <data>
   [timestamp] [TRT-LLM AUTO-DEPLOY] [RANK <GPU ID>] [I] [stage=post_export, transform=cleanup_input_constraints] [POST-CLEANUP] <data>
   [timestamp] [TRT-LLM AUTO-DEPLOY] [RANK <GPU ID>] [I] [stage=post_export, transform=cleanup_input_constraints] [SUMMARY] matches=<num matches> | time: <>s (pre=<>ss, apply=<>s, post=<>s)

2. **Dump graph IR after every transform:
   ```bash
   AD_DUMP_GRAPHS_DIR=<path to folder> trtllm-serve ... 
   ```
   - Generates a textual representation of the graph after every transformation.
   - Genrerates one file per transformation, even if the transformation is disabled or skipped
   - The files are numbered according to transformation order ddd_transformation_name.txt where ddd is a 3-digit index

## Resources

- **Config Generator**: `scripts/generate_config.py` - Create initial configs
- **Config Examples**: `references/config_examples.md` - Complete config templates
- **Model Registry**: `references/model_registry.md` - Pre-configured settings for 100+ models
- **Online Benchmarking**: `references/online_benchmarking.md` - Server + aiperf details
- **Offline Benchmarking**: `references/offline_benchmarking.md` - trtllm-bench details
