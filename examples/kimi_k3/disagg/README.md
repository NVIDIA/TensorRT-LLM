<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kimi K3 disaggregated serving (ctx/gen split)

Configuration pair + deployment wiring for running Kimi K3 with separate
context (prefill) and generation (decode) servers. Status: validated
end-to-end on GB300 NVL72 with the current configs (1 ctx + 1 gen, DEP16
both sides, KVCacheManagerV2); see the `gen_config_no_sa.yaml` row below
for GSM8K accuracy and throughput. See the caveats section for constraints.

## Files

| File | Purpose |
|---|---|
| `ctx_config.yaml` | Context-server extra LLM-API options (DEP16, overlap scheduler off, no spec decode) |
| `gen_config.yaml` | Generation-server options WITH suffix-automaton (SA) speculative decoding (DEP16, eager) |
| `gen_config_no_sa.yaml` | Generation-server options WITHOUT spec decode — use this first (CUDA graphs ON by default: GSM8K 96.5, ~950/2043 output tok/s @c64/c256 on 8k/1k random prompts; null `cuda_graph_config` for token-parity debugging) |
| `disagg_proxy_config.yaml` | `trtllm-serve disaggregated` proxy config (1 ctx + 1 gen) |
| `benchmark_kimi_k3_dep16.yaml` | Config for the SLURM benchmark harness (`examples/disaggregated/slurm/benchmark/submit.py`) |

## K3 constraints baked into the configs

- **EP-only parallelism on BOTH sides**: `ep_size == tp_size`, no PP, no
  TP on linears. Deployed as DEP-N (`enable_attention_dp: true`).
- **Matched ctx/gen parallelism (DEP16 = DEP16)** for now: only this
  geometry is validated end-to-end on hardware. Heterogeneous ctx/gen
  parallelism passes peer validation and single-node loopback transfer
  tests (with attention-DP off the KDA state is head-sharded across TP,
  so the TP-mismatch mappers re-tile it; with attention-DP on it is
  replicated), but no hetero geometry has been validated at scale.
- **Ctx sizing = DEP16**: DEP16 is the smallest verified fit
  (~193 GiB weights/rank on GB300). A DEP8 ctx is estimated at
  ~273 GiB/rank for weights alone (extrapolating the 1.5 TB checkpoint:
  replicated share ~113 GiB + experts/8), leaving no activation headroom
  on GB300 (288 GiB) and not fitting GB200 (186 GiB). Treat DEP8-ctx as
  ruled out on GB200 and an open (likely negative) question on GB300.
- **Python transceiver** (`backend: NIXL`, `transceiver_runtime: PYTHON`):
  only it can move the KDA state. It is the default (and `auto` selects it
  for K3 too); the configs set it explicitly.
- `disable_overlap_scheduler: true` on the ctx server (disagg
  requirement) and on the gen server (SA runs eager; also keeps the
  SA-off smoke maximally comparable).
- `enable_block_reuse: false`, no chunked prefill, beam width 1 (model
  requirements); `tokens_per_block: 64` is kept for historical reasons
  (older TRTLLM-Gen MLA kernels required it).
- The Python-transceiver recipes do not set `max_tokens_in_buffer`.
  Size `max_num_tokens` and `max_seq_len` for the target input/output lengths.
- **No bounce buffer**: leave `kv_cache_bounce_size_mb` unset. K3 runs on
  KVCacheManagerV2, which keeps the MLA KV cache and the KDA state in its
  pools. On GB300 NVL72 these pools are fabric memory, and standard NIXL
  writes them pool to pool across nodes over MNNVL (cuda_ipc). With its
  default gate a bounce buffer would stay unused: the gate only accepts
  writes between mismatched head layouts, and K3's matched DEP16 layout
  produces none.
- **Optional, shorter KV transfer**: for latency-sensitive or short-ISL
  serving, set `kv_cache_bounce_size_mb: 1024` and
  `agent_bounce_params: {min_descriptor_count: "1", max_average_descriptor_size: "4MB"}`
  on both servers to send K3's writes through the C++ transfer-agent
  bounce buffer. On GB300 NVL72 this cut the per-request 8k KV transfer
  about 4x (~40 ms to ~10 ms) for 1 GiB per GPU; end-to-end throughput
  was unchanged.

## KDA state payload size

Per-request recurrent-state payload (fixed, token-count independent):
69 KDA layers x (conv `[3*96*128, 4]` bf16 + delta `[96, 128, 128]` fp32)
= 454,459,392 bytes (~433 MiB). For synthetic-KV harnesses that size
transfers per token: at K3's per-rank 211,968 B/token
(69 layers x kvFactor 2 x 6 kv heads/rank x head_dim 128 x 2 B), 2144
tokens reproduce this payload exactly.

## Launch sequence (manual, single ctx + single gen)

Each K3 worker spans 16 GPUs (4 NVL72 nodes at 4 GPUs/node). Environment
prerequisites for every worker shell (see caveats below for why):

```bash
export UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc   # on clusters where verbs cannot
                                                # initialize; a container-default
                                                # UCX_TLS=tcp breaks NIXL setup
```

1. Start the context server (16-rank MPI world across its 4 nodes):

   ```bash
   trtllm-llmapi-launch trtllm-serve $MODEL_PATH \
       --host <ctx_head_node> --port 8001 \
       --config examples/kimi_k3/disagg/ctx_config.yaml
   ```

2. Start the generation server (SA off first):

   ```bash
   trtllm-llmapi-launch trtllm-serve $MODEL_PATH \
       --host <gen_head_node> --port 8002 \
       --config examples/kimi_k3/disagg/gen_config_no_sa.yaml
   ```

3. Edit `disagg_proxy_config.yaml` (worker URLs = the head nodes above),
   then start the proxy:

   ```bash
   trtllm-serve disaggregated -c examples/kimi_k3/disagg/disagg_proxy_config.yaml
   ```

4. Send OpenAI-compatible requests to the proxy (port 8000). Once the
   SA-off path is parity-validated, restart the gen server with
   `gen_config.yaml` to enable SA.

## SLURM benchmark harness

`benchmark_kimi_k3_dep16.yaml` drives the full orchestration (worker
config generation, node allocation, proxy, benchmark client):

```bash
python3 examples/disaggregated/slurm/benchmark/submit.py \
    -c examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml --dry-run  # inspect
python3 examples/disaggregated/slurm/benchmark/submit.py \
    -c examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml            # submit
```

- Set `benchmark.dataset_file` before an e2e submission.
- **Gen-only baseline**: set `benchmark.mode: gen_only_no_context`
  (submit.py exports `TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1` to the
  workers) to measure the decode-side ceiling without KV transfer.
- The harness's `start_worker.sh` clears `UCX_TLS`; the config carries
  the transport pin via `TRTLLM_WORKER_UCX_TLS`, which `start_worker.sh`
  re-exports as `UCX_TLS` after the clear.
- pyxis/enroot resets image-defined variables (notably `PATH`) at
  container start, so the config injects the in-place TRT-LLM venv via
  `TRTLLM_PATH_PREPEND` / `TRTLLM_PYTHONPATH_PREPEND`, applied inside
  the container by `start_worker.sh` / `start_server.sh` /
  `run_benchmark.sh`.

## Current caveats (read before running)

1. **Python-transceiver "MPI hang" — root-caused, environmental (RESOLVED
   with the env pins).** A reported multi-node hang in
   `KvCacheTransceiverV2._exchange_rank_info` → `mpi_allgather` was a
   downstream symptom of `UCX_TLS=all` on nodes where `ud_verbs` cannot
   initialize: the broken transport wedges native NIXL/UCX agent init
   asymmetrically per rank, and the healthy ranks park forever in the
   setup MPI collectives. Not an MPI/pmix or transceiver code bug; with
   `UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` the Python transceiver passes
   multi-node with no code change.
2. **SA ships eager here.** SA speculative decoding in disagg was
   validated for accuracy (GSM8K parity with aggregated serving) with CUDA
   graphs disabled, as configured in `gen_config.yaml`; that validation
   predates the current configs and has not been repeated with them. SA
   with CUDA graphs is functional (the MLA
   latent-cache append under CUDA graphs handles spec-dec verification),
   but the disagg SA + graphs perf points have not been re-measured yet,
   so `gen_config.yaml` keeps graphs off. Start with
   `gen_config_no_sa.yaml` for the first bring-up on a new cluster, then
   switch to `gen_config.yaml`.
3. **Matched-DP only.** Keep ctx and gen at identical DEP16 with
   attention-DP on both sides; heterogeneous parallelism passes peer
   validation but is not validated end-to-end (see constraints above).
4. **Cluster environment** (NVL72 nodes): on clusters where verbs
   transports cannot initialize, pin
   `UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` (`UCX_TLS=all` hangs setup,
   see caveat 1) and never run the Python transceiver with a
   container-default `UCX_TLS=tcp` (breaks NIXL VRAM registration) —
   unset/override it. Keep `cuda_ipc` in any pin; UCX uses it for
   transfers over MNNVL.
5. **Transfer payload**: each request moves a fixed 433.4 MiB (~454.5 MB)
   KDA state blob ctx → gen in addition to the MLA latent KV
   (~27 KB/token). Keep ctx and gen inside one NVL72 domain so these
   writes can go over MNNVL.
6. **KV cache manager V2 only.** Only the Python transceiver can move the
   KDA state, and it rejects KV cache manager V1 at startup. So
   `use_kv_cache_manager_v2: false` fails, and so does `"auto"` with a
   text-only K3 checkpoint (`KimiLinearForCausalLM` declares no
   preference, so `auto` picks V1). Use the released checkpoint with the
   default KV cache manager.
7. **SA caps gen-side batch size.** SA requires `max_batch_size` ≤ 8 on
   the generation server, which bounds per-instance concurrency at
   `8 × dp_size` (128 with DEP16). Plan instance counts accordingly.
8. **Prefill capacity and TTFT under burst.** Without chunked prefill,
   context-server throughput is limited and queued prefills grow TTFT
   roughly linearly under closed-loop bursts. Rate-match the ctx:gen
   instance ratio to the expected traffic instead of oversubscribing a
   single context server.
9. **Startup time.** Weight loading takes tens of minutes per 16-GPU
   instance before the first token; set health-check, idle-reaper, and
   job time limits accordingly. The disaggregated proxy does not serve
   `/v1/models` (404) — point readiness probes at a different endpoint.
10. **`max_num_tokens` coupling.** The generation side must cover
    `max_batch_size × (1 + max_draft_len)`; with chunked prefill disabled,
    the context side must accommodate the prompt in `max_num_tokens` and
    `max_seq_len`.
