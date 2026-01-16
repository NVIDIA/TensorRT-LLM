# RL Framework Integration Tests

This directory contains integration tests for TensorRT-LLM with [Ray orchestrator](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/ray_orchestrator), specifically designed to cover usage patterns from various RL (Reinforcement Learning) frameworks such as VeRL and NeMo RL. 

## Available Scripts

| Script | Description |
|--------|-------------|
| `run_rl_perf_reproduce.py` | Emulates RL workload performance with multiple AsyncLLM instances distributed across GPUs using Ray placement groups |

## Usage Examples

### RL Performance Reproduction

The `run_rl_perf_reproduce.py` script creates multiple TensorRT-LLM instances in parallel to simulate RL rollout workloads.

**TP=4 with 2 instances (8 GPUs total):**

```bash
python run_rl_perf_reproduce.py \
    --model_dir /path/to/model_dir \
    --data_path /path/to/prompts.json \
    --num_instances 2 \
    --tp_size 4 \
    --top_p 1 \
    --logprobs 1 \
    --max_batch_size 1024 \
    --enable_cuda_graph_padding
```

**TP=1 with 8 instances (8 GPUs total):**

```bash
python run_rl_perf_reproduce.py \
    --model_dir /path/to/model_dir \
    --data_path /path/to/prompts.json \
    --num_instances 8 \
    --tp_size 1 \
    --top_p 1 \
    --logprobs 1 \
    --max_batch_size 384 \
    --enable_cuda_graph_padding
```

## Data Format

The `--data_path` should point to a JSON file containing a list of prompts, where each prompt is a list of token IDs:

```json
[
    [1, 2345, 6789, ...],
    [1, 3456, 7890, ...],
    ...
]
```

## Notes

- RL Perf reproduction scripts support single-node execution only (max 8 GPUs)
