# Knob Heuristics — Decision Tables

The 4 knobs that materially affect basic aggregate (IFB) single-node throughput. All heuristics distilled from 72K+ benchmark records (aggregate/inflight batching, PyTorch backend, non-speculative).

## moe_backend: `moe_config.backend` (MoE models only)

Controls which GEMM kernel dispatches expert computations. Up to 2x throughput impact.

| Quantization | GPU | Rule |
|---|---|---|
| FP4 | Any | CUTLASS at high concurrency (>=64), TRTLLM at low |
| FP8 | Blackwell (B200/B300/GB200/GB300) | DEEPGEMM at high concurrency, TRTLLM at low |
| FP8 | Hopper (H100/H200) | CUTLASS always (DEEPGEMM unavailable on SM90) |

**If wrong**: TRTLLM at high concurrency leaves 30-50% throughput on the table. CUTLASS/DEEPGEMM at conc<=4 can be slightly worse due to dispatch overhead.

Dense models: omit `moe_config` entirely.

## attention_dp: `enable_attention_dp`

Replicates attention weights across TP ranks for data-parallel attention. Trades GPU memory for attention throughput at high concurrency.

| Architecture | When to enable |
|---|---|
| MoE + MLA | Concurrency >= 128 for short context (ISL ~1k), >= 512 for long context (ISL ~8k) on Blackwell (192 GiB). Never on H100 (80 GiB) — empirically ADP never helps for MoE+MLA on H100. |
| MoE + GQA | Concurrency >= 32 (adds 15-20 GiB overhead from large attention matrices). |
| Dense + GQA | Concurrency >= 256 (rarely helps below this). |

**If wrong**: OOM at server startup (ADP on + high KV fraction). Throughput plateau at high concurrency (ADP off when it should be on). GQA models pay 15-20 GiB/GPU for ADP vs 1-3 GiB for MLA.

## kv_fraction: `kv_cache_config.free_gpu_memory_fraction`

Fraction of free GPU memory allocated to KV cache after model loading. Too high = OOM; too low = wasted capacity and premature preemption.

| Architecture | ADP off | ADP on |
|---|---|---|
| MoE + MLA (192 GiB GPU) | 0.80-0.90 | 0.75-0.85 |
| MoE + MLA (80 GiB GPU) | 0.85-0.90 | 0.80 |
| MoE + GQA | 0.80 | 0.65-0.75 |
| Dense + GQA | 0.85-0.90 | 0.80-0.85 |

**If wrong**: OOM on startup or first request (too high). TTFT spikes from premature preemption (too low). MoE GEMM autotuner failure (`Can't allocate profile workspace`) when fraction >= 0.80 with ADP on large MoE.

## max_num_tokens: `max_num_tokens`

Scheduler token budget — how many tokens the engine processes per iteration. Sets the prefill budget.

**Rules**:
- Must be >= ISL + 64 (chat template overhead) when chunked prefill is disabled, or requests get HTTP 400.
- Sweet spot: ISL to 2x ISL.
- Low concurrency (<=32): `max(ISL, 2048)`.
- High concurrency (>=64): `max(ISL * 2, 4096)`.
- Never > ISL * 4 — wastes activation buffer, counterintuitively shrinks KV cache.
- Check database configs first: some models use fixed values (e.g., 1152 for DeepSeek-R1).

**If wrong**: HTTP 400 ("prompt length exceeds max_num_tokens") if too low. 2-5% throughput loss if excessively high.
