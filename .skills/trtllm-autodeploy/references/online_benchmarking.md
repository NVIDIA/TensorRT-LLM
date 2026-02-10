# Online Benchmarking with Autodeploy Backend

Online benchmarking measures inference performance of a running server under concurrent requests.

## Using bench_server.py Script

The `bench_server.py` script (if available) automates server launch and benchmarking:

```bash
python bench_server.py \
  --model meta-llama/Llama-3.1-70B \
  --config-path config.yaml \
  --server-type trtllm-autodeploy \
  --concurrencies "1 4 8 16 32 64" \
  --isl 1024 \
  --osl 1024 \
  --world-size 8
```

Key options:
- `--server-type trtllm-autodeploy`: Use autodeploy backend
- `--config-path`: Path to autodeploy config YAML
- `--world-size`: Number of GPUs (adds to config automatically)
- `--concurrencies`: Space-separated list of concurrent request levels
- `--isl/--osl`: Input/output sequence lengths
- `--profile`: Enable nsys profiling
- `--env-vars`: Additional environment variables (KEY=VALUE,KEY2=VALUE2)

## Manual Server Launch

Start the server with autodeploy backend:

```bash
trtllm-serve meta-llama/Llama-3.1-70B \
  --host 0.0.0.0 \
  --port 8123 \
  --trust_remote_code \
  --backend _autodeploy \
  --extra_llm_api_options config.yaml
```

## Manual Client Benchmarking with aiperf

After server is running, benchmark with aiperf:

```bash
aiperf profile \
  --model meta-llama/Llama-3.1-70B \
  --url 0.0.0.0:8123 \
  --endpoint-type chat \
  --streaming \
  --concurrency 32 \
  --request-count 160 \
  --isl 1024 \
  --osl 1024 \
  --artifact-dir ./results \
  --num-warmup-requests 1 \
  --extra-inputs '{"ignore_eos": true, "min_tokens": 1024}' \
  --request-timeout-seconds 1800
```

Key aiperf options:
- `--concurrency`: Number of concurrent requests
- `--request-count`: Total requests (typically concurrency × rounds)
- `--isl/--osl`: Input/output sequence lengths
- `--streaming`: Enable streaming responses
- `--extra-inputs`: JSON with additional parameters (ignore_eos, min_tokens)
- `--request-timeout-seconds`: Timeout per request

## Common Concurrency Patterns

Default concurrencies for sweep testing:
```
1, 4, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 224, 256, 384
```

Quick test:
```
1, 8, 32, 64
```

## Profiling with nsys

Enable profiling during server launch:

```bash
nsys profile \
  -o trace_$(date +%y%m%d_%H%M)_ad_serving \
  -f true \
  -t cuda,cublas,nvtx \
  --cuda-graph-trace node \
  --trace-fork-before-exec=true \
  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json \
  trtllm-serve meta-llama/Llama-3.1-70B \
    --backend _autodeploy \
    --extra_llm_api_options config.yaml
```

Or use bench_server.py with `--profile` flag.

## Environment Variables

Useful environment variables for autodeploy:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TLLM_LLMAPI_ENABLE_NVTX=1  # Enable NVTX markers
export TLLM_PROFILE_RECORD_GC=1   # Profile garbage collection
```

## Expected Workflow

1. Create autodeploy config YAML with transforms
2. Launch server with `--backend _autodeploy`
3. Wait for server startup (check for "Application startup complete.")
4. Run aiperf with desired concurrency levels
5. Analyze results in artifact directories

## Results Structure

Each concurrency level creates a directory:
```
results/
├── isl_1024_osl_1024_conc_1/
├── isl_1024_osl_1024_conc_8/
├── isl_1024_osl_1024_conc_32/
└── ...
```

Each directory contains:
- `profile_export.json`: Detailed metrics
- `profile_export_genai_perf.csv`: CSV summary
- Request/response logs
