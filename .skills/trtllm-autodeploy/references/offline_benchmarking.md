# Offline Benchmarking with Autodeploy Backend

Offline benchmarking measures maximum throughput without a server, useful for performance characterization.

## Basic Usage

```bash
trtllm-bench \
  --model meta-llama/Llama-3.1-8b \
  throughput \
  --dataset /path/to/dataset.txt \
  --backend _autodeploy \
  --max_batch_size 256 \
  --extra_llm_api_options config.yaml \
  --tp 1
```

Key flags:
- `--backend _autodeploy`: Use autodeploy backend
- `--extra_llm_api_options`: Path to autodeploy config YAML
- `--tp`: Tensor parallelism size (number of GPUs)
- `--max_batch_size`: Maximum batch size for throughput testing
- `--dataset`: Path to dataset file with prompts

## Dataset Preparation

Generate synthetic datasets with `prepare_dataset.py`:

```bash
python3 benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer meta-llama/Llama-3.1-8b \
  token-norm-dist \
  --input-mean 128 \
  --output-mean 128 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 256 \
  > /tmp/dataset.txt
```

Parameters:
- `--tokenizer`: HuggingFace model for tokenization
- `--input-mean`: Mean input sequence length
- `--output-mean`: Mean output sequence length
- `--input-stdev`: Standard deviation for input (0 = fixed length)
- `--output-stdev`: Standard deviation for output (0 = fixed length)
- `--num-requests`: Number of requests in dataset

## Complete Example

```bash
# 1. Prepare dataset
python3 benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --tokenizer meta-llama/Llama-3.1-8b \
  token-norm-dist \
  --input-mean 1024 \
  --output-mean 1024 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 256 \
  > dataset_1024_1024.txt

# 2. Run benchmark
trtllm-bench \
  --model meta-llama/Llama-3.1-8b \
  throughput \
  --dataset dataset_1024_1024.txt \
  --warmup 0 \
  --backend _autodeploy \
  --max_batch_size 256 \
  --extra_llm_api_options config.yaml \
  --tp 1
```

## Multi-GPU Benchmarking

For multi-GPU setups, set `--tp` to the number of GPUs:

```bash
trtllm-bench \
  --model meta-llama/Llama-3.1-70B \
  throughput \
  --dataset dataset.txt \
  --backend _autodeploy \
  --max_batch_size 256 \
  --extra_llm_api_options config.yaml \
  --tp 8  # 8 GPUs
```

Ensure the config includes `world_size: 8` or the appropriate GPU count.

## Comparing Backends

Test both autodeploy and pytorch backends:

```bash
# Autodeploy backend
trtllm-bench \
  --model meta-llama/Llama-3.1-8b \
  throughput \
  --dataset dataset.txt \
  --backend _autodeploy \
  --extra_llm_api_options ad_config.yaml \
  --max_batch_size 256

# PyTorch backend (for comparison)
trtllm-bench \
  --model meta-llama/Llama-3.1-8b \
  throughput \
  --dataset dataset.txt \
  --backend pytorch \
  --extra_llm_api_options pytorch_config.yaml \
  --max_batch_size 256
```

## Additional Options

### Warmup Runs
```bash
--warmup 5  # Number of warmup iterations
```

### Output Control
```bash
--output results.json  # Save results to file
```

### Verbose Logging
```bash
--verbose  # Enable detailed logging
```

## Metrics to Expect

trtllm-bench reports:
- **Throughput**: Tokens per second
- **Latency**: Time per request (ms)
  - First token latency (TTFT)
  - Inter-token latency (ITL)
- **GPU Utilization**: Percentage
- **Memory Usage**: GPU memory consumed

## Common Patterns

### ISL/OSL Sweep
Test different input/output lengths:

```bash
for isl in 128 512 1024 2048; do
  for osl in 128 512 1024 2048; do
    python3 benchmarks/cpp/prepare_dataset.py \
      --stdout \
      --tokenizer meta-llama/Llama-3.1-8b \
      token-norm-dist \
      --input-mean $isl \
      --output-mean $osl \
      --input-stdev 0 \
      --output-stdev 0 \
      --num-requests 256 \
      > dataset_${isl}_${osl}.txt

    trtllm-bench \
      --model meta-llama/Llama-3.1-8b \
      throughput \
      --dataset dataset_${isl}_${osl}.txt \
      --backend _autodeploy \
      --extra_llm_api_options config.yaml \
      --output results_${isl}_${osl}.json
  done
done
```

### Batch Size Sweep
Test different batch sizes:

```bash
for bs in 32 64 128 256 384; do
  trtllm-bench \
    --model meta-llama/Llama-3.1-8b \
    throughput \
    --dataset dataset.txt \
    --backend _autodeploy \
    --max_batch_size $bs \
    --extra_llm_api_options config.yaml \
    --output results_bs${bs}.json
done
```
