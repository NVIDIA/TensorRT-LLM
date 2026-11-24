<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kimi K3.1 Pro ULL on VR200 (sm_107): disaggregated recipes for C1, C4, and C8

`submit.py` configs (`examples/disaggregated/slurm/benchmark/submit.py`) for the ultra-low-latency points of
Kimi K3.1 Pro: the ModelOpt NVFP4 checkpoint, FP8 KV cache, FP8 DSA index keys, FP32 KDA state and the DFlash2
drafter at draft length 7. Fill in the `<...>` placeholders before submitting.

| Point | Layout: CTX + GEN | GPUs | Lanes |
|---|---|---|---|
| C1 | 1 × tep8 + 1 × helix16, GEN batch 1, k7 | 24 | 1 |
| C4 | 1 × tep8 + 1 × helix16, GEN batch 4, k7 | 24 | 4 |

- **tep8** (CTX): TP 8, MoE EP 8, no attention DP; 8 GPUs on 2 nodes. Target `kimi_k31_pro_nvfp4__sm_107__tep8`.
- **helix16** (GEN): Helix context parallel 16, TP 1, MoE TP 16 × EP 1, no attention DP; 16 GPUs on 4 nodes. Target
  `kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`.
- Measurement settings: AgentX (Workload, below), forced acceptance 2.68 on GEN, HBM clock locked at its maximum
  supported memory clock, SM clock free. Forced acceptance applies to real-draft rows only.

## Requirements

The helix16 GEN targets (`kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1` and its `TLLM_K31_HELIX16_WIDE` fork
`kimi_k31_pro_nvfp4__sm_107__helix16_wide`) build their MLA decode kernels on the
[tokenspeed-mla](https://github.com/lightseekorg/tokenspeed) package (MIT), which TensorRT-LLM does not install.
Install it in the GEN workers' environment, e.g. `pip install tokenspeed-mla==2.0.2`; without it, the GEN target
fails with `ModuleNotFoundError: No module named 'tokenspeed_mla'` when it builds its first MLA decode kernel. The CTX
targets (`tep8`, `dep16`) do not need it.

## Files

| File | Case |
|---|---|
| `C1_e2e.yaml`, `C4_e2e.yaml`, `C8_e2e.yaml` | performance: AgentX, forced acceptance on GEN |
| `C1_gsm.yaml`, `C4_gsm.yaml` | accuracy on the same servers without forced acceptance: needle at 131,072 tokens, GSM8K, prefix-reuse check |

C4 = C1 with GEN `max_batch_size` 4, `max_num_tokens` 32, CUDA graphs for 1..4 requests, concurrency 4, and the CTX
KV settings below. Do not use GEN batch 7 or 8: 56/64 rows × 4 exceed the 212 SMs and the DSA select/pack drops to
cluster size 1.

C8 uses the same worker configs as C4 with 8 AgentX client lanes and a GEN first-forward queue threshold of 4.
The reference run used a 3600 s AgentX window, per-rank NUMA binding and page-cache dropping before replay.

## Launch

1. Work dir: this repository's launcher plus the benchmark clients that the yamls select with `K31_CLIENT`: an
   AgentX client for `*_e2e.yaml`, and for `*_gsm.yaml` a long-context needle test and a prefix-reuse check. Provide
   a `run_benchmark.sh` that runs the selected client from the work dir:

   ```
   mkdir -p <WORK_DIR>
   cp examples/disaggregated/slurm/benchmark/* <WORK_DIR>/
   ```

   Keep this repository's `start_worker.sh` and `start_server.sh`: hostnames, the checkpoint's default tool and
   reasoning parsers (`kimi_k3`), and the `slurm.numa_local` binding (NUMA, below).
   `start_worker.sh` passes no `--server_role`, so the yamls set the role in the environment: `TRTLLM_DISAGG_ROLE`
   is `context` in `ctx_worker_env_var` and `generation` in `gen_worker_env_var` (our runs passed `--server_role`,
   which sets the same variable). Without it the CTX is routed as a generation server; keep both entries if you
   edit the env lines.
2. Fill in the placeholders: `<PARTITION>`, `<ACCOUNT>`, `<EXTRA_SBATCH_ARGS>`, `<IMAGE>`, `<CONTAINER_MOUNTS>`,
   `<MODEL_PATH>`, `<DRAFTER_PATH>`, `<TRTLLM_REPO>`, `<TRTLLM_WHEEL>`, `<WORK_DIR>`, `<LOG_DIR>`,
   `<TRTLLM_CACHE_DIR>`, and `<HF_HOME>`. With no extra sbatch arguments, set
   `<EXTRA_SBATCH_ARGS>` to `""`: an empty value is YAML null and `submit.py` fails on it.
3. `python3 <WORK_DIR>/submit.py -c examples/kimi_k31/ull_disagg/C1_e2e.yaml` (add `--dry-run` to inspect the
   generated ctx/gen configs).
4. Lock HBM at its maximum supported memory clock on every node
   (`nvidia-smi -lmc <max_mem_clock>,<max_mem_clock>`; read the value from `nvidia-smi -q -d SUPPORTED_CLOCKS`),
   SM clock free.
5. `job_time` 01:40:00 covers the 3600 s window plus model load and warm-up.

Checkpoint `nvidia/kimi_k31_pro_nvfp4` @ 1d820323a55c, drafter `RadixArk/Kimi-K3.1-Dflash2-Preview` @ d0d8bc2e3c6d,
AgentX traces `semianalysis_cc_traces_weka_062126`.

## Settings that matter

- **Routing (CTX):** the CTX sets `disable_mm_encoder: true` and `speculative_config.context_kv_dtype: fp8`.
  The specialized sm_107 targets (`kimi_k31_pro_nvfp4__sm_107__tep8` here) take only a text-only deployment
  with explicit FP8 drafter history. Without either key the CTX starts without an error on the configurable
  `kimi_k31_pro_nvfp4__sm_10x__default` target instead, which is not the recorded configuration. With
  `disable_mm_encoder` the CTX builds no vision tower, so these recipes serve text only. The GEN layout (Helix
  CP16, MoE TP16 × EP1) routes to the ULL target before that check and needs neither key.
- **Forced acceptance** `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=2.68` (k7 golden AL 3.68) on **GEN only**; the
  CTX never verifies drafts. The `*_gsm.yaml` cases drop it.
- **First block from the CTX:** the CTX loads the whole drafter (no `context_kv_only`; about 1 GiB more weights per
  rank than the history-only drafter, plus the plain capture buffer) and, on each request's last prefill chunk,
  drafts the block a generation step would draft next (the selector's greedy walk; `K31_CTX_DRAFTS` lines on the
  CTX). It sends the 7 drafts with the first token; a request it cannot draft for sends none. The GEN verifies them
  on the request's first step as real drafts (forced acceptance applies; `K31_CTX_FIRST_DRAFTS ... real=` lines),
  and a sampled first step verifies them as a point mass.
  For a mixed-version rollout, upgrade all GEN receivers before upgrading the TEP8 CTX servers that send these
  blocks. The GEN receive hook clears the slot's previous proposal before verifying the imported greedy block.
- **GEN env:** `TRTLLM_SPEC_SAMPLER_FUSED_SCATTER=1`, `TRTLLM_STATE_INDEX_COPY_KERNEL=1`,
  `TRTLLM_KV_BLOCK_OFFSETS_TRITON=2`; `TLLM_BENCHMARK_REQ_QUEUES_SIZE` = the first-forward queue threshold (1 for C1, 4 for C4 and C8).
- **All workers:** `TRTLLM_K3_ULL_STEP_PREP=1`, `TRTLLM_K3_ULL_PIN_EXECUTOR=auto`, `TRTLLM_K3_ULL_GIL_SWITCH_US=200`,
  `TLLM_KV_CACHE_MANAGER_V2_BACKEND=cpp`.
- **NUMA:** all five yamls set `slurm.numa_bind: true`, `slurm.numa_local: true` and
  `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` (CTX and GEN).
  - `numa_bind` alone runs each worker under `numactl -m 0,1`, the NVL72 host-memory policy: host allocations
    come from NUMA nodes 0 and 1 (the two CPU sockets) only. It does not bind CPUs.
  - `numa_local` starts each rank, before any Python runs, under `numactl --cpunodebind=<n> --membind=<n>`, where
    `<n>` is the NUMA node of the rank's GPU, and replaces `numa_bind`'s `-m 0,1`. Each rank logs
    `NUMA_LOCAL rank=... gpu=... numa=... cpus=... mems=... membind=...`; `NUMA_LOCAL_SKIP` means it started
    unbound (then `numa_bind` applies).
  - `TLLM_NUMA_AWARE_WORKER_AFFINITY=1` keeps TensorRT-LLM from clearing that CPU mask; it then binds the worker
    threads to the same node's CPUs.
  - Why: without it, TensorRT-LLM binds CPUs only after it selects the GPU, and sets no memory policy. A rank
    that starts on the other socket keeps its import-time heap there. On VR200 the GEN ranks then split into
    a fast and a slow tier (small pageable copies between CUDA graph replays), and every step waits for
    the slow tier.
  - Measured with an equivalent per-rank wrapper (CPUs bound to the GPU's node before Python starts, memory
    by first touch) on C1, short 600 s AgentX: the spread between the two GEN tiers nearly closed. The p90
    gain of this alone was not isolated.
  - The C1 and C4 reference runs did not use `numa_bind` or `numa_local`; their effect on these points
    is unmeasured under the workload below (Workload).
  - The trtllm-serve frontend on each worker's first rank inherits that rank's binding.
- **C4 and C8 CTX KV** (C1 keeps fraction 0.45 and no `avg_seq_len`):
  - `free_gpu_memory_fraction` 0.6 and `avg_seq_len` 80000, `max_num_tokens` 8192, no `host_cache_size` (KV cache
    manager v2 then adds a host tier the size of the GPU quota).
  - CTX env: `TRTLLM_KV_V2_WAIT_FOR_INFLIGHT_PREFIX=1 TRTLLM_KDA_MIDCHUNK_SNAPSHOT=1 TLLM_V2_CTX_DEMOTE_EXACT=1
    TLLM_V2_CTX_DEMOTE_MAX_ITERS=32 TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS=16384
    TLLM_V2_LOW_PRIORITY_SNAPSHOT_PROMPT_LEN=16384`.
  - Why: without `avg_seq_len` the pool split assumes half of `max_seq_len` per sequence and the KDA state pool gets
    7 slots, so at 4 lanes the CTX keeps losing prefix snapshots and recomputes whole prefixes. With 80000 it gets 57
    slots: in a 480 s C4 test the prompt cache hit rose and TTFT p90 fell sharply. `avg_seq_len` 240000 gave 19 slots
    and a higher TTFT p90.
  - Do not raise the fraction to 0.9 with 16384-token chunks: the DSA indexer's long-context prefill workspace is not
    covered by the startup memory profile and the CTX runs out of memory.
  - `C4_gsm.yaml` runs the accuracy checks (GSM8K, needle, prefix reuse) on this CTX setting.
- **Rejection sampling (off):** the recipes verify DFlash2 drafts by exact match. To verify sampled requests by block
  verification instead (the selector's sampled walk, its proposal probabilities kept per request), add
  `use_rejection_sampling: true` to the **GEN** `speculative_config` only (the CTX does not verify). It needs
  `modeling_v2: require` (already set) and no guided decoding. Greedy requests (temperature 0) keep the exact-match
  path either way. Each GEN rank logs `K31_RS_PATH block_verify=1` once and `K31_RS_AGREE ... mismatched=0` every 200
  steps (rows whose proposal differed from rank 0's before the ranks agreed on it).
  - Limitation: with `advanced_sampling_mode: no_topk_no_topp` (these recipes) a request's `top_k` / `top_p` /
    `min_p` are ignored, with or without rejection sampling: the target distribution is the temperature softmax only.
    `no_topk` honours `top_p`, `no_topp` honours `top_k`, and only `full` honours all three. The drafter's proposal is
    always the temperature softmax over its 32 candidates; block verification corrects it to the filtered target, so
    the output follows the target in every mode.
  - With forced acceptance on, the tokens after the verified prefix are the drafts, so use that combination for step
    timing only, not for output quality.
- **Workload:** AgentX (3600 s window, max context 996,579, 3 warm-up requests per lane). Drop the page cache of
  the model and drafter files on every node before the client if your filesystem evicts them during the run.
