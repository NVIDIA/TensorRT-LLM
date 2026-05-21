# gpt-oss-120b agg vs disagg TTFT reproduction

Independent reproduction of the agg-vs-disagg TTFT comparison reported by
[`simengl/trtllm-cookbook .../aligned_request_lifetime_report_20260512.md`](https://gitlab-master.nvidia.com/simengl/trtllm-cookbook/-/blob/main/projects/aarwlt-gptoss/agg-vs-disagg-lifetime-20260512/analysis/aligned_request_lifetime_report_20260512.md)
and the [`ai-dynamo` TTFT breakdown](https://github.com/ai-dynamo/dynamo/blob/rihuo/check_benchmark/benchmark_result/ttft_breakdown.md).
Uses only `trtllm-serve` + `benchmark_serving.py` + server-side
`print_iter_log` (no SFlow / Dynamo / nsys required).

## TL;DR finding

| metric         | this repro p50 | cookbook (single request) |
|----------------|---------------:|--------------------------:|
| agg TTFT       | 166 ms         | 950 ms (server lane)      |
| disagg TTFT    | 202 ms         | 2629 ms                   |
| disagg − agg   | +36 ms         | +963 ms                   |

See [results/agg/FINDINGS.md](results/agg/FINDINGS.md) and
[results/disagg/FINDINGS.md](results/disagg/FINDINGS.md) for the per-stage
breakdowns and a side-by-side discussion of the differences (cross-process
clock alignment, kv_cache_aware router, KV-cache reuse from the cookbook's
warmup pass, etc.).

## Layout

```
configs/                      1:1 copies of the cookbook YAMLs plus a few
                              ablation variants. Eagle3-next checkpoint path
                              is sed-substituted in by the launch scripts.
scripts/launch_agg.sh         backgrounds trtllm-serve agg on 1 GPU, polls /health
scripts/launch_disagg.sh      backgrounds ctx (8001) + gen (8002) + proxy (8000)
scripts/stop_agg.sh           tear down via logs/pids/agg.pid
scripts/stop_disagg.sh        tear down via logs/pids/{ctx,gen,proxy}.pid
scripts/bench_ttft.sh         benchmark_serving.py wrapper (conc=1, 32 prompts)
scripts/cookbook_breakdown.py reproduces the cookbook stage table from
                              the *-perf_metrics.json benchmark_serving writes
scripts/parse_iter_log.py     iter-level analyzer that needs no /perf_metrics
                              endpoint (works straight off server logs)
results/                      bench output + FINDINGS.md per layout
COOKBOOK_VS_INREPO_CONFIG_DIFF.md  config audit vs in-repo gpt-oss-120b configs
```

## Quick workflow

```bash
# Aggregated
CUDA_VISIBLE_DEVICES=0 scripts/launch_agg.sh repro_agg_tp1_eagle3
scripts/bench_ttft.sh agg http://localhost:8000
scripts/stop_agg.sh
python3 scripts/parse_iter_log.py logs/agg.log

# Disaggregated 1ctx+1gen
CTX_GPU=0 GEN_GPU=1 scripts/launch_disagg.sh \
    PROXY_CONFIG_BASE=repro_disagg_proxy_round_robin
scripts/bench_ttft.sh disagg http://localhost:8000
scripts/stop_disagg.sh
python3 scripts/parse_iter_log.py logs/gen.log
python3 scripts/parse_iter_log.py logs/ctx.log
```

Defaults (override via env vars before invocation):

| env var | default |
|---|---|
| `MODEL`               | `/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b` |
| `EAGLE_CKPT`          | `/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3` |
| `SERVED_MODEL_NAME`   | `openai/gpt-oss-120b` (so bench client model id is stable) |
| `NUM_PROMPTS`         | 32 |
| `CONCURRENCY`         | 1 |
| `ISL` / `OSL`         | 13576 / 144 (matches the cookbook prompt size) |

## Caveats

- `HF_HUB_OFFLINE=1` is set by the launchers, so the proxy's
  `kv_cache_aware` router cannot load `openai/gpt-oss-120b` from HF and
  crashes on first request. Use `repro_disagg_proxy_round_robin.yaml`
  instead, or pass `--tokenizer <local path>` to `trtllm-serve disaggregated`.
- `/perf_metrics` on the disagg proxy needs `TRTLLM_KVCACHE_TIME_OUTPUT_PATH`
  set on **all three** processes plus the env-gated per-request
  `sampling_params.return_perf_metrics=True` path (chat completions
  doesn't set it; completions does — see
  `tensorrt_llm/serve/openai_server.py:1383`).
- benchmark_serving's "Initial test run" pre-sends prompt 0 as a warmup;
  with `enable_block_reuse: true` the first measured request is a 100% KV
  hit. `scripts/parse_iter_log.py` and `scripts/cookbook_breakdown.py`
  both drop the warmup request(s) by default.
