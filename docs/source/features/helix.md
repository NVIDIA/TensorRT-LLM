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

1. **Disaggregated serving**: Helix is designed for generation servers in a disaggregated (prefill/decode split) deployment architecture. Aggregated Helix, which runs prefill and decode on the same CP ranks, is now supported for dense MLA layers; see [Aggregated Helix](#aggregated-helix)
2. **Long input sequences**: Performance gains typically appear with input sequence lengths **>64K tokens** or more
3. **Low batch sizes**: Optimal for latency-sensitive workloads with high tokens/second/user requirements

On a typical latency vs. throughput Pareto curve, Helix targets operating points toward the right side (low latency, high per-user throughput).

## Supported Models

Helix parallelism currently supports models using **Multi-head Latent Attention (MLA)** on Blackwell GPU architecture:

- DeepSeek-V3 / DeepSeek-V3-Lite
- Kimi K3 (aggregated Helix)

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

## Aggregated Helix

Without a cache transceiver, Helix runs aggregated: one executor runs both the
context and generation phases on the CP ranks. Aggregated Helix has these
requirements:

- Blackwell (SM100) or newer.
- MLA models with dense MLA attention only. Every sparse attention algorithm is
  rejected, including `skip_softmax`; Helix with sparse attention requires
  disaggregated serving.
- The TRTLLM attention backend.
- Chunked prefill (`enable_chunked_prefill=True`).
- KVCacheManagerV2 (`kv_cache_config.use_kv_cache_manager_v2=True`).

The following are not supported with aggregated Helix:

- An FP4 KV cache. Use a BF16 or FP8 KV cache.
- A disk cache tier, a remote cache level, or a KV cache connector.
- KV pool rebalance (`kv_cache_config.enable_kv_pool_rebalance`).
- Attention DP with `pipeline_parallel_size > 1`.
- A separate draft KV cache manager or draft-owned K/V layers for speculative decoding.
- KV cache compression.

### KV Cache Block Reuse

Block reuse (`kv_cache_config.enable_block_reuse`) is supported with aggregated Helix
and not with disaggregated Helix. Under aggregated Helix:

- Reuse happens in whole ledger blocks of `tokens_per_block * context_parallel_size`
  tokens, so every CP rank reuses the same prefix. Partial-block reuse is turned off.
- Retention priorities are supported. A retention `duration_ms` (`decode_duration_ms` or a
  token range's `duration_ms`) is rejected, because each CP rank would expire blocks on its
  own clock.
- For models with recurrent-state layers, such as Kimi K3, snapshot points are moved down
  to ledger-block boundaries. Set `mamba_state_config.periodic_snapshot_interval` to a
  multiple of the ledger block so that no snapshot is lost.

## Testing Helix with TensorRT-LLM

### Unit Test: MLA Module Correctness

This developer test is not included in TensorRT-LLM release containers. Follow the
[build-from-source workflow](../installation/build-from-source.md), then install the test dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

The test requires at least two Blackwell GPUs; pytest skips it on unsupported hardware.

The simplest correctness test validates the [MLA attention module](../../../tensorrt_llm/_torch/attention/mla.py) with Helix enabled:

```bash
# Run the MLA Helix unit test
python3 -m pytest tests/unittest/_torch/attention/multi_gpu/test_mla_helix.py -v
```

This test verifies that attention outputs match between single-GPU and Helix-parallelized execution.

### End-to-End Accuracy test

For end-to-end validation, the accuracy benchmark evaluates DeepSeek-V3-Lite in disaggregated mode on MMLU and GSM8K benchmarks:

Test location: `tests/integration/defs/accuracy/test_disaggregated_serving.py`  
Test name: `TestDeepSeekV3Lite::test_auto_dtype_with_helix`

This test demonstrates proper disaggregated server configuration with Helix.
