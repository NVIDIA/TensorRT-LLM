# Deployment Guide for DeepSeek R1 on TensorRT LLM - Blackwell & Hopper Hardware

(Trimmed excerpt for this exercise.)

## Models

* FP8 model: DeepSeek-R1-0528
* NVFP4 model: DeepSeek-R1-0528-FP4

## MoE Backend Support Matrix

| device | Checkpoint | Supported moe_backend |
|----------|----------|----------|
| H100/H200 | FP8 | CUTLASS |
| B200/GB200 EP<=8 | NVFP4 | CUTLASS, TRTLLM |
| B200/GB200 EP<=8 | FP8 | DEEPGEMM |

The default MoE backend is `CUTLASS`; for combinations `CUTLASS` does not support,
set `moe_config.backend` explicitly.

## Choosing a config

Benchmark-tuned configs for B200 live under `examples/configs/database/` and are keyed
by ISL/OSL, concurrency and GPU count. All of them enable MTP speculative decoding.
Low-concurrency (latency-oriented) configs use tensor parallelism for attention;
high-concurrency (throughput-oriented) configs enable attention data parallelism
(`enable_attention_dp: true`).

## Launch

```shell
trtllm-serve deepseek-ai/DeepSeek-R1-0528 --host 0.0.0.0 --port 8000 --config ${EXTRA_LLM_API_FILE}
```
