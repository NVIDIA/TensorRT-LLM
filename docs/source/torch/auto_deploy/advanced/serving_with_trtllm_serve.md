# Serving with trtllm-serve

AutoDeploy integrates with the OpenAI-compatible `trtllm-serve` CLI so you can expose AutoDeploy-optimized models over HTTP without writing server code. This page shows how to launch the server with the AutoDeploy backend, configure it via YAML, and validate with a simple request.

## Quick start

Launch `trtllm-serve` with the AutoDeploy backend by setting `--backend _autodeploy`:

```bash
trtllm-serve \
  meta-llama/Llama-3.1-8B-Instruct \
  --backend _autodeploy \
```

- `model`: HF name or local path
- `--backend _autodeploy`: uses AutoDeploy runtime

Once the server is ready, test with an OpenAI-compatible request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages":[{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Where is New York? Tell me in a single sentence."}],
    "max_tokens": 32
  }'
```

## Configuration via YAML

Use `--extra_llm_api_options` to supply a YAML file that augments or overrides server/runtime settings.

```bash
trtllm-serve \
  meta-llama/Llama-3.1-8B \
  --backend _autodeploy \
  --extra_llm_api_options autodeploy_config.yaml
```

Example `autodeploy_config.yaml`:

```yaml
# Compilation backend for AutoDeploy
compile_backend: torch-opt  # options: torch-simple, torch-compile, torch-cudagraph, torch-opt

# Runtime engine
runtime: trtllm                # options: trtllm, demollm

# Model loading
skip_loading_weights: false    # set true for architecture-only perf runs

# KV cache memory
free_mem_ratio: 0.8            # fraction of free GPU mem for KV cache

# CUDA graph optimization
cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64]

# Attention backend
attn_backend: flashinfer       # recommended for best performance
```

## Limitations and tips

- KV cache block reuse is disabled automatically for AutoDeploy backend
- AutoDeploy backend doesn't yet support disaggregated serving. WIP
- For best performance:
  - Prefer `compile_backend: torch-opt`
  - Use `attn_backend: flashinfer`
  - Set realistic `cuda_graph_batch_sizes` that match expected traffic
  - Tune `free_mem_ratio` to 0.8–0.9

## See also

- [AutoDeploy overview](../auto-deploy.md)
- [Benchmarking with trtllm-bench](./benchmarking_with_trtllm_bench.md)

