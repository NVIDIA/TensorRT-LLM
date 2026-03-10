# Helix Parallelism

Helix is a context parallelism (CP) technique for the decode/generation phase of LLM inference. Unlike traditional attention-FFN disaggregation (AFD) techniques, which spatially separate attention and FFN blocks onto different GPUs, Helix temporally separates them by reconfiguring the same GPUs.

For all details, see the original paper:
[Helix Parallelism: Rethinking Sharding Strategies for
Interactive Multi-Million-Token LLM Decoding](https://arxiv.org/pdf/2507.07120)

## How Helix Works

In Helix parallelism:

- **KV cache distribution**: The KV cache is partitioned across CP ranks during generation, with each rank responsible for a portion of the cached context
- **Attention computation**: Each rank computes partial attention over its local KV cache shard
- **Attention postprocessing**: Partial results are combined / corrected across ranks to produce the final attention output
- **FFN layers**: CP ranks are repurposed as tensor parallelism (TP) ranks for FFN/MoE layers, maximizing GPU utilization

## When to Use Helix

Helix parallelism provides performance benefits when **all** of the following conditions apply:

1. **Disaggregated serving**: Helix is designed for generation servers in a disaggregated (prefill/decode split) deployment architecture
2. **Long input sequences**: Performance gains typically appear with input sequence lengths **>64K tokens** or more
3. **Low batch sizes**: Optimal for latency-sensitive workloads with high tokens/second/user requirements

On a typical latency vs. throughput Pareto curve, Helix targets operating points toward the right side (low latency, high per-user throughput).

## Supported Models

Helix parallelism currently supports models using **Multi-head Latent Attention (MLA)** on Blackwell GPU architecture:

- DeepSeek-V3 / DeepSeek-V3-Lite

## Configuration

### Configuration Parameters

Please set the following parameters for the generation servers in disaggregated mode. Example can be seen in the e2e accuracy test mentioned below. 

| Parameter | Description | Required |
|-----------|-------------|----------|
| `context_parallel_size` | Number of GPUs for context parallelism (≥2 for Helix) | Yes |
| `cp_config.cp_type` | Must be `"HELIX"` or `CpType.HELIX` | Yes |
| `cp_config.tokens_per_block` | Tokens per KV cache block | Yes |
| `kv_cache_config.tokens_per_block` | Must match `cp_config.tokens_per_block` | Yes |

### JSON Configuration (for YAML/JSON configs)

```json
{
    "context_parallel_size": 2,
    "cp_config": {
        "cp_type": "HELIX",
        "tokens_per_block": 32
    },
    "kv_cache_config": {
        "tokens_per_block": 32
    }
}
```

## Testing Helix with TensorRT-LLM

### Unit Test: MLA Module Correctness

The simplest correctness test validates the [MLA attention module](../../../tensorrt_llm/_torch/modules/attention.py) with Helix enabled:

```bash
# Run the MLA Helix unit test
pytest tests/unittest/_torch/modules/test_mla_helix.py -v
```

This test verifies that attention outputs match between single-GPU and Helix-parallelized execution.

### End-to-End Accuracy test

For end-to-end validation, the accuracy benchmark evaluates DeepSeek-V3-Lite in disaggregated mode on MMLU and GSM8K benchmarks:

Test location: `tests/integration/defs/accuracy/test_disaggregated_serving.py`  
Test name: `TestDeepSeekV3Lite::test_auto_dtype_with_helix`

This test demonstrates proper disaggregated server configuration with Helix.
