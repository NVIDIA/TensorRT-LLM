# Cookbook vs. in-repo gpt-oss-120b configs: knob-by-knob comparison

Compares the cookbook lifetime configs we are reproducing
([repro_agg_tp1_eagle3.yaml](configs/repro_agg_tp1_eagle3.yaml),
[repro_disagg_ctx_tp1.yaml](configs/repro_disagg_ctx_tp1.yaml),
[repro_disagg_gen_tp1.yaml](configs/repro_disagg_gen_tp1.yaml)) against the
in-tree configs for `openai/gpt-oss-120b`:

- Aggregated curated configs: `examples/configs/curated/gpt-oss-120b-latency.yaml`, `gpt-oss-120b-throughput.yaml`
- Database (pareto-tuned, per-GPU/conc) configs: `examples/configs/database/openai/gpt-oss-120b/B200/*.yaml`, `H200/*.yaml`
- Perf-sanity disaggregated configs: `tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_*_ctx1_tp1_gen1_*_ccb-UCX.yaml`
- Perf-sanity aggregated FP4 configs: `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml`
- Deployment guide: `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md`
- Eagle3 blog: `docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md`

Defaults referenced from `tensorrt_llm/llmapi/llm_args.py`.

## Notable differences (candidates for ablation)

| # | Knob | Cookbook | In-repo norm | Note |
|---|---|---|---|---|
| 1 | **`TRTLLM_ENABLE_PDL`** env | not set in `gpt_oss_120b_disagg_env_lifetime_20260512.yaml` | `=1` everywhere (database `env_overrides`, perf-sanity disagg `worker_env_var`, deployment guide, blog11, blog09) | PDL is part of the recommended Blackwell perf path. blog01 reports ~173 ms of decode-iteration savings on DeepSeek-R1 from PDL alone. Worth testing whether enabling it shifts ctx forward time or scheduler tick. |
| 2 | **`num_postprocess_workers`** (gen) | **0** | **4** in every in-repo gpt-oss-120b config that sets it | With value > 0, detokenization runs in separate processes (IPC over ZMQ). With 0 it runs synchronously in the executor thread on every token. For streaming decode, in-process detok can extend `executor → server token` if the tokenizer is slow on a single thread. The cookbook keeps it at 4 for ctx and 0 for gen; the in-repo configs use 4 in both roles. |
| 3 | **`stream_interval`** | ctx=**10**, gen=**100** | **20** uniformly across curated, database, perf-sanity (agg and disagg) | The schema doc: "Set this to a larger value when the batch size is large, which helps reduce the streaming overhead." `stream_interval=100` on gen means the worker batches up to 100 decode iterations between streaming chunks. At conc=1 / draft=3 the end-to-end token count is unaffected, but the first SSE chunk after TTFT carries ~100 iters of latency, which inflates ITL noticeably. |
| 4 | **`cache_transceiver_config.max_tokens_in_buffer`** | **131072** | **1024** (1k1k) / **8448** (8k1k) in perf-sanity disagg | Cookbook sizes the KV-transfer pinned buffer to `max_seq_len`; perf-sanity sizes it to the ISL. Larger value costs more host pinned memory and may slow NIXL/UCX ring init at startup; per-transfer time should be the same. |
| 5 | **`moe_config.backend`** on ctx | **TRTLLM** | **CUTEDSL** with `use_low_precision_moe_combine: true` in perf-sanity disagg ctx (`tests/scripts/perf-sanity/disaggregated/gb200_gpt-oss-120b-fp4_1k1k_con64_ctx1_tp1_gen1_tp4_eplb0_mtp0_ccb-UCX.yaml`) | Perf-sanity ctx uses CUTEDSL on B200/GB200; cookbook uses TRTLLM for both ctx and gen. With the cookbook context-compute bucket already at 27 ms, this is a small-magnitude change. |
| 6 | **`disable_overlap_scheduler`** on ctx | **false** | **true** in `examples/disaggregated/README.md`, every perf-sanity disagg ctx, every disagg example yaml | The in-repo README states: *"The overlap scheduler for context servers is currently disabled, as it is not yet supported in disaggregated context server architectures."* Cookbook sets `disable_overlap_scheduler: false` on the ctx role. Worth confirming whether overlap is functional on the ctx path with current TRT-LLM. |
| 7 | **`torch_compile_config` + `enable_piecewise_cuda_graph: true`** (with 157-entry ctx `capture_num_tokens`, 128-entry gen list) | enabled with large `capture_num_tokens` lists | not used in any in-repo gpt-oss-120b config (curated, database, perf-sanity) | Piecewise CUDA graph + torch.compile is documented in `docs/source/features/torch_compile_and_piecewise_cuda_graph.md`; the in-repo gpt-oss-120b path uses plain CUDA graphs from `cuda_graph_config` only. Adds first-run warmup time and graph cache memory. Worth testing whether removing it changes the steady-state numbers. |
| 8 | **`eagle3_layers_to_capture: [23, 29, 35]`** | custom mid-layers | `[-1]` (just post-norm) in `gpt_oss_120b_fp4_grace_blackwell.yaml` perf-sanity Eagle test; deployment-guide / blog11 do not enumerate | **Matches the checkpoint.** "Eagle3-next" is the pre-release name of [`nvidia/gpt-oss-120b-Eagle3-v3`](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-v3). Its `config.json` declares `eagle_aux_hidden_state_layer_ids: [24, 30, 36]` (1-based) ≡ TRT-LLM `[23, 29, 35]` (0-based). A shared local copy is at `/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3` and is the default for `scripts/launch_agg.sh` / `scripts/launch_disagg.sh`. |

## Other differences (likely benign)

| Knob | Cookbook | In-repo | Note |
|---|---|---|---|
| `kv_cache_config.enable_block_reuse` | `true` (= class default) | `false` in perf-sanity (avoids skewing synthetic benches) | Reasonable for a real serving workload; perf-sanity intentionally disables to keep prefill cost stable across runs. |
| `kv_cache_config.free_gpu_memory_fraction` | 0.9 | 0.85 (database B200), 0.9 (perf-sanity disagg), 0.8 (perf-sanity agg) | Within normal range. |
| `max_batch_size` (ctx) | 128 | 32 (perf-sanity disagg ctx for both 1k1k and 8k1k) | For conc=1 this just allocates unused KV slots. |
| `max_batch_size` (gen) | 128 | 256 (perf-sanity disagg gen, 1k1k tp4 conc64) | Smaller than perf-sanity but fine for conc=1. |
| `max_seq_len` | 131072 | 2068 (1k1k), 9236 (8k1k) | Cookbook keeps the full model context; perf-sanity sizes to ISL+OSL only. Affects KV cache budget. |
| `max_num_tokens` (gen) | 512 | 20000 (perf-sanity disagg gen) | Cookbook gen sets a small prefill budget appropriate for decode-only; perf-sanity uses a generic large value. |
| `cache_transceiver_config.backend` | `DEFAULT` (= NIXL with UCX sub-backend) | `UCX` direct (perf-sanity) | Both end up on UCX wire format; DEFAULT adds the NIXL shim layer (very small overhead). |
| `scheduler_config.capacity_scheduler_policy` | `MAX_UTILIZATION` | not set (defaults to `MAX_UTILIZATION`) | No effective difference. |
| `gpus_per_node` | 8 (agg), unset (ctx/gen) | 4 (perf-sanity disagg, GB200 chip count) | Affects accounting only. |
| `attn_backend` | not set (default `TRTLLM`) | explicitly `TRTLLM` in some perf-sanity agg | No effective difference. |

## Suggested ablation order

Each row is a one-knob change against the baseline configs. Suggested order (cheap-to-test → expensive):

| Order | Variation | Stage to watch in the breakdown |
|---|---|---|
| A | Set `TRTLLM_ENABLE_PDL=1` env var on all three processes | `context compute`, `decode compute` |
| B | `num_postprocess_workers: 4` on gen | `decode → server token`, `disagg egress` |
| C | `stream_interval: 1` on ctx and gen | `decode → server token`, `executor → server token` |
| D | `disable_overlap_scheduler: true` on ctx | `prefill finalize` |
| E | `moe_config.backend: CUTEDSL` + `use_low_precision_moe_combine: true` on ctx | `context compute` |
| F | Drop `torch_compile_config` + `enable_piecewise_cuda_graph` | `prefill finalize`, `context compute` |
| G | Drop `speculative_config` entirely on agg | `executor → server token` |

Pre-built ablation YAMLs already in this dir:
[`configs/repro_agg_tp1_nospec.yaml`](configs/repro_agg_tp1_nospec.yaml),
[`configs/repro_disagg_ctx_tp1_nospec.yaml`](configs/repro_disagg_ctx_tp1_nospec.yaml),
[`configs/repro_disagg_gen_tp1_nospec.yaml`](configs/repro_disagg_gen_tp1_nospec.yaml),
[`configs/repro_disagg_proxy_round_robin.yaml`](configs/repro_disagg_proxy_round_robin.yaml).
Remaining variations are 2-line sed edits on the baseline YAMLs at ablation
time.
