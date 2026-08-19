# Kimi K3 disaggregated serving (ctx/gen split)

Configuration pair + deployment wiring for running Kimi K3 with separate
context (prefill) and generation (decode) servers. Status: **validated
end-to-end on hardware** (GB300 NVL72, 1 ctx + 1 gen, DEP16 both sides,
GSM8K accuracy parity with aggregated serving) — see the caveats section
for constraints.

## Files

| File | Purpose |
|---|---|
| `ctx_config.yaml` | Context-server extra LLM-API options (DEP16, overlap scheduler off, no spec decode) |
| `gen_config.yaml` | Generation-server options WITH suffix-automaton (SA) speculative decoding (DEP16, eager) |
| `gen_config_no_sa.yaml` | Generation-server options WITHOUT spec decode — use this first (CUDA graphs ON by default: GSM8K 96.89, 765/2138 tok/s @c64/c256 vs aggregated 643/1972; null `cuda_graph_config` for token-parity debugging) |
| `disagg_proxy_config.yaml` | `trtllm-serve disaggregated` proxy config (1 ctx + 1 gen) |
| `benchmark_kimi_k3_dep16.yaml` | Config for the SLURM benchmark harness (`examples/disaggregated/slurm/benchmark/submit.py`) |

## K3 constraints baked into the configs

- **EP-only parallelism on BOTH sides**: `ep_size == tp_size`, no PP, no
  TP on linears. Deployed as DEP-N (`enable_attention_dp: true`).
- **Matched ctx/gen parallelism (DEP16 = DEP16)** for now. Heterogeneous
  ctx/gen TP with attention-DP *off* would silently corrupt memory for
  K3's replicated KDA state and is rejected at peer registration — do
  not deviate. Hetero DEP with attention-DP on both sides is believed
  correct but unvalidated.
- **Ctx sizing = DEP16**: DEP16 is the smallest verified fit
  (~193 GiB weights/rank on GB300). A DEP8 ctx is estimated at
  ~273 GiB/rank for weights alone (extrapolating the 1.5 TB checkpoint:
  replicated share ~113 GiB + experts/8), leaving no activation headroom
  on GB300 (288 GiB) and not fitting GB200 (186 GiB). Treat DEP8-ctx as
  ruled out on GB200 and an open (likely negative) question on GB300.
- **`transceiver_runtime: PYTHON` is mandatory**: `auto` resolves to the
  C++ transceiver, which throws at construction for K3's
  `MixedMambaHybridCacheManager`.
- `disable_overlap_scheduler: true` on the ctx server (disagg
  requirement) and on the gen server (SA runs eager; also keeps the
  SA-off smoke maximally comparable).
- `enable_block_reuse: false`, `tokens_per_block: 64`, no chunked
  prefill, beam width 1 (model requirements).
- `max_tokens_in_buffer: 8448` covers the target max ISL of 8192; raise
  it together with `max_num_tokens`/`max_seq_len` for longer ISL.
- **`kv_cache_bounce_size_mb: 1024` on both sides**: the V2 transceiver's default
  pool-to-pool path cannot use inter-node cuda_ipc on MNNVL (the KV pool
  is a plain, non-fabric allocation) and falls back to ~0.4 GB/s
  host-staged tcp; the fabric-VMM bounce buffer restores cuda_ipc/MNNVL
  eligibility (measured ~455 GB/s/GPU). Bounce engages automatically for payloads above `TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES` (default 2 MiB) — always true for K3's ~433 MiB per-request state.
  The region must fit ONE request's full payload: fixed 433 MiB KDA
  state + ~27 KB/token MLA latent (649 MiB at 8k ISL). A
  512 MiB value makes every 8k transfer fall back to the per-fragment
  tcp path (`[kv-bounce] in-place: transfer 649MiB exceeds the 512MiB
  bounce region`).

## Launch sequence (manual, single ctx + single gen)

Each K3 worker spans 16 GPUs (4 NVL72 nodes at 4 GPUs/node). Environment
prerequisites for every worker shell (see caveats below for why):

```bash
export UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc   # on clusters where verbs cannot
                                                # initialize; a container-default
                                                # UCX_TLS=tcp breaks V2 NIXL
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

1. **V2 transceiver "MPI hang" — root-caused, environmental (RESOLVED
   with the env pins).** A reported multi-node hang in
   `KvCacheTransceiverV2._exchange_rank_info` → `mpi_allgather` was a
   downstream symptom of `UCX_TLS=all` on nodes where `ud_verbs` cannot
   initialize: the broken transport wedges native NIXL/UCX agent init
   asymmetrically per rank, and the healthy ranks park forever in the
   setup MPI collectives. Not an MPI/pmix or V2 code bug; with
   `UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` V2 NIXL passes multi-node
   with no code change.
2. **SA ships eager here.** SA speculative decoding in disagg is
   validated for accuracy (GSM8K parity with aggregated serving) with
   CUDA graphs disabled, as configured in `gen_config.yaml`. SA with
   CUDA graphs is functional (the MLA latent-cache append under CUDA
   graphs handles spec-dec verification), but the disagg SA + graphs
   perf points have not been re-measured yet, so `gen_config.yaml`
   keeps graphs off. Start with `gen_config_no_sa.yaml` for the first
   bring-up on a new cluster, then switch to `gen_config.yaml`.
3. **Matched-DP only.** Keep ctx and gen at identical DEP16 with
   attention-DP on both sides; heterogeneous parallelism with
   attention-DP off is rejected (see constraints above).
4. **Cluster environment** (NVL72 nodes): on clusters where verbs
   transports cannot initialize, pin
   `UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc` (`UCX_TLS=all` hangs setup,
   see caveat 1) and never run V2 NIXL with a container-default
   `UCX_TLS=tcp` (breaks V2 NIXL VRAM registration) — unset/override it.
   No bounce env override is needed: the byte gate
   (`TRTLLM_KV_CACHE_BOUNCE_MIN_BYTES`, default 2 MiB) is always cleared
   by K3 payloads (constraints section above).
5. **Transfer payload**: each request moves a fixed 433.4 MiB (~454.5 MB)
   KDA state blob ctx → gen in addition to the MLA latent KV
   (~27 KB/token).
   Within an NVL72 domain this is ~0.9 ms/request (measured; not a
   bottleneck), but off-fabric paths would pay 11–23 ms — keep ctx and
   gen inside one NVL72 domain.
6. **Bounce-buffer sizing cliff (silent).** Size `kv_cache_bounce_size_mb`
   to the largest single request's full KV payload (fixed KDA state plus
   the per-token MLA latent; ≥1024 MB for 8k ISL). An undersized region
   does not error — every transfer silently falls back to a much slower
   host-staged TCP path.
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
    `max_batch_size × (1 + max_draft_len)`; the context side needs
    `max_tokens_in_buffer` ≥ max ISL (see constraints above).
