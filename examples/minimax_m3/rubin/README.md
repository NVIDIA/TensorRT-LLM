<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M3 on Rubin (SM107): disaggregated serving

| File | Role |
|---|---|
| `ctx_agentx_c24.yaml` | Context worker for multi-turn agentic traffic (AgentX, 24 concurrent conversations). |
| `fp4_moe_tactics_tp2ctx_v1.json` | FP4 MoE tactic table for the TP2/EP2 context worker (`TLLM_FP4_MOE_TACTIC_TABLE`). |
| `setup_fa4.sh` (applies `3rdparty/patches/fa4_fp8_rescale0.patch`) | FA4 for the FP8 dense-layer prefill; prepend its directory to `PYTHONPATH` in every context worker. |

## FP8 dense prefill on FA4

On SM107 the dense layers' FP8 prefill runs FA4 (`flash_attn.cute`, from
flash-attn-4 4.0.0b19) when the installed FA4 carries `3rdparty/patches/fa4_fp8_rescale0.patch`.
That one-line change makes the FP8 forward rescale the row maximum on every increase,
so the softmax numerators stay within E4M3 range. `bash setup_fa4.sh <dir>` makes a
patched copy of the installed package (site-packages is left unchanged); prepend
`<dir>` to `PYTHONPATH`. Without the change the dense layers keep fmha_sm100 and
log `FA4 lacks the FP8 rescale_threshold 0.0 change`. Set
`TRTLLM_MSA_DISABLE_DENSE_FA4=1` to keep fmha_sm100 deliberately.

## Agentic multi-turn traffic

Each turn of an agentic conversation resends the previous turn's prompt plus a
short suffix, so time to first token and context-worker throughput depend on how
much of that prefix is still cached. Measured with AgentX at 24 concurrent
conversations on 1 x TP2 context + 4 x TP2 generation workers, this setup reached
a 97.4% prefix hit rate on the context worker.

Context worker: `ctx_agentx_c24.yaml`.

- **One KV pool.** TP2 attention instead of attention DP keeps every conversation
  in a single pool, so a conversation's prefix is never on the other rank.
- **Capacity.** `free_gpu_memory_fraction: 0.95` and a 100 GiB per-rank host tier
  (`host_cache_size`, KV cache manager V2). GPU-evicted blocks move to host memory
  and are copied back on the next turn instead of being recomputed. Size the host
  tier from the node's free host memory, and keep at least 25% of it free.
- **`block_reuse_config.policy: per_conversation`.** Reuse is tracked per
  conversation id.
- **`enable_tokenization_cache: true`.** Repeated prompt prefixes are not
  re-tokenized.
- **`max_num_tokens: 16384`.** This is the prefill chunk.
- **`enable_iter_perf_stats: true`.** The per-iteration KV statistics then include
  host-tier traffic.
- **Append reserve.** Set `TLLM_CTX_APPEND_RESERVE=8064`. Each iteration then chunks
  the first context request against `max_num_tokens - 8064`, so short follow-up
  turns run in the same iteration as a long cold prefill instead of waiting for it.
  The leftover budget goes to the shortest uncached request first
  (`TLLM_CTX_APPEND_RESERVE_SJF=0` keeps arrival order). The reserve is held back
  only while another context request is waiting. It applies only when
  `max_num_tokens > 2 * R`.
- **Deterministic MoE tactics.** Set `TLLM_FP4_MOE_TACTIC_TABLE` to
  `fp4_moe_tactics_tp2ctx_v1.json`. The FP4 MoE runner then uses the tabled tactic
  for each token bucket instead of a per-start autotuned one, so two server starts
  produce identical prefill output. Each entry is the tactic autotuning chose most
  often, so speed is unchanged. The table only applies to MoE shape `4,3072,64,0`
  (TP2/EP2).

Generation workers (TP2 MiniMax-M3 decode workers) need:

- `tensor_parallel_size: 2`.
- `enable_block_reuse: true`, `use_kv_cache_manager_v2: true` and
  `block_reuse_config.policy: per_conversation`, so a conversation's next turn
  only transfers the new suffix from the context worker.

Disaggregated router: `type: conversation`. Each conversation stays on the same
generation worker, so that worker's cached prefix is reused.
