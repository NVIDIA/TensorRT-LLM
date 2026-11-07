<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax-M3 on Rubin (SM107): aggregated TP2 serving

One TP2 engine per pair of GPUs serves both prefill and decode, for multi-turn
agentic traffic (AgentX). Pick the configuration by concurrency:

| File | Concurrent conversations | Requests per engine |
|---|---|---|
| `agg_c14.yaml` | 14 | 2 |
| `agg_c20.yaml` | 20 | 3 |
| `agg_c46.yaml` | 46 | 14 |
| `agg_c56.yaml` | 56 | 24 |

Source `agg_env.sh` in every worker, set `speculative_model` to the
MiniMax-M3 Eagle3 drafter checkpoint, and prepend the `setup_fa4.sh` directory
(`../README.md`) to `PYTHONPATH`.

## Decode target

`modeling_v2: require` builds the MiniMax-M3 sm_107 TP2 modeling_v2 target
(`minimax_m3_nvfp4__sm_107__tp2`) for every configuration: it runs decode steps
of 1 to 192 verify rows (up to 48 requests with three drafts), and its prefill
steps of at least 4096 context rows take the built-in model's context path.

## Scheduling and memory

- **Batch wait.** `batch_wait_timeout_iters` holds a new context request for up
  to N iterations so it joins a fuller step: 32 at 46 conversations, 8 at 56.
- **CUDA graphs.** `agg_c56.yaml` captures every batch size 1..24, so a step is
  never padded to the next captured size.
- **KV capacity.** `free_gpu_memory_fraction: 0.97` and an explicit per-rank
  host tier (`host_cache_size`, KV cache manager V2): 295 GiB at 14 / 20
  conversations, 355 GiB at 46 / 56. Size it from the node's free host memory
  (at most a quarter per GPU), and keep the host tier on the GPU's local NUMA
  node: pages that spill to the remote node stall the copy-back.
- **Prefill.** Breakable prefill CUDA graphs captured up to 4096 tokens, and
  16k-token chunks.

## Launch layout at 56 conversations

Put the two ranks of an engine on different CPU sockets (for example GPUs 0
and 2 on a four-GPU node), bind each rank's memory to its GPU's NUMA node with
`numactl -m <node>`, and set `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` so each rank's
worker threads run on its GPU's local CPUs. Do not use `numactl --cpunodebind`:
TensorRT-LLM resets the CPU affinity unless `TLLM_NUMA_AWARE_WORKER_AFFINITY=1`.
