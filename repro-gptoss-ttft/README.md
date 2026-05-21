# gpt-oss-120b agg vs disagg TTFT reproduction

Independent reproduction of the agg-vs-disagg TTFT comparison reported by
[`simengl/trtllm-cookbook .../aligned_request_lifetime_report_20260512.md`](https://gitlab-master.nvidia.com/simengl/trtllm-cookbook/-/blob/main/projects/aarwlt-gptoss/agg-vs-disagg-lifetime-20260512/analysis/aligned_request_lifetime_report_20260512.md)
and the [`ai-dynamo` TTFT breakdown](https://github.com/ai-dynamo/dynamo/blob/rihuo/check_benchmark/benchmark_result/ttft_breakdown.md).
Uses only `trtllm-serve` + `benchmark_serving.py` + server-side
`print_iter_log` (no SFlow / Dynamo / nsys required).

## TL;DR finding

| run                           | TTFT p50 | TTFT p99 | TPOT  | E2EL  | notes |
|-------------------------------|---------:|---------:|------:|------:|---|
| agg, TP1, 1 GPU               | 166 ms   | 187 ms   | 1.09 ms | 373 ms | fresh ctx, seed=42 |
| disagg round_robin, 2 GPUs    | 202 ms   | 223 ms   | 2.61 ms | 575 ms | fresh ctx, seed=42 |
| disagg kv_cache_aware, 2 GPUs | 202 ms   | 207 ms   | 2.40 ms | 547 ms | fresh ctx, seed=43 |
| cookbook agg (rep1, 1 GPU)    | 950 ms (server lane) | n/a | n/a | 1680 ms | single request, KV reuse from warmup |
| cookbook disagg (rep6, 2 GPUs)| 2629 ms  | n/a      | n/a   | n/a   | single request, separate steady-clock alignment across 3 processes |

At conc=1 with the cookbook configs (TP1, Eagle3-v3, fp8 KV, torch.compile
+ piecewise CUDA graph, default DEFAULT/NIXL transceiver), this repro
measures **+36 ms TTFT going from agg to disagg**, and **router choice
makes no measurable TTFT difference** at the 1ctx+1gen topology.

See [results/agg/FINDINGS.md](results/agg/FINDINGS.md) and
[results/disagg/FINDINGS.md](results/disagg/FINDINGS.md) for the per-stage
breakdowns plus side-by-side discussion of differences vs the cookbook
report (cross-process clock alignment in the cookbook's measurement
framework, KV-cache reuse pattern on the cookbook's warmup pass, etc.).

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
| `SERVED_MODEL_NAME`   | `openai/gpt-oss-120b` (stable bench-client model id) |
| `NUM_PROMPTS`         | 32 |
| `CONCURRENCY`         | 1 |
| `ISL` / `OSL`         | 13576 / 144 (matches the cookbook prompt size) |
| `SEED`                | 42 (override per run when chaining benches — see Caveats) |

## Caveats

### KV cache reuse — change the seed between runs

`benchmark_serving.py --random-ids --seed N` generates 32 prompts
deterministically from N. With `enable_block_reuse: true` on the ctx
worker, two back-to-back runs against the same server with the same seed
make the second run hit the KV cache on every prompt and report an
artificially low TTFT.

Confirmed in this experiment: a kv_aware disagg run that reused
`seed=42` immediately after a `seed=42` round_robin run reported
TTFT p50 = 76 ms with `num_ctx_tokens = 1` per ctx iter (i.e. 100% KV
reuse). Re-running with `SEED=43` produced an uncached TTFT p50 = 202 ms.

**Rule**: when chaining benches against the same workers, pass a new
`SEED` per invocation (`SEED=43 scripts/bench_ttft.sh ...`) or restart
the workers first. `benchmark_serving.py`'s "Initial test run" also
sends the first prompt as a warmup before measurement, so the first
main-loop request always hits cache; `scripts/parse_iter_log.py` and
`scripts/cookbook_breakdown.py` drop the warmup request(s) by default.

### Router (`kv_cache_aware` vs `round_robin`)

- The `kv_cache_aware` router tries `AutoTokenizer.from_pretrained(model)`
  on the proxy where `model` is the request body's `model` field. Under
  `HF_HUB_OFFLINE=1` it fails if the requested id is an HF-only name like
  `openai/gpt-oss-120b`. Workarounds: send the bench with
  `MODEL=/local/checkpoint` so the request's `model` field is a local
  path the proxy can resolve, or use `repro_disagg_proxy_round_robin.yaml`.
- At conc=1 with 1 ctx + 1 gen server, the router choice has no
  measurable TTFT impact (we tested it). With multiple ctx servers it
  would, but that's not exercised here.

### `/perf_metrics` plumbing

- `/perf_metrics` on the disagg proxy needs `TRTLLM_KVCACHE_TIME_OUTPUT_PATH`
  set on **all three** processes plus the env-gated per-request
  `sampling_params.return_perf_metrics=True` path (chat completions
  doesn't set it; completions does — see
  `tensorrt_llm/serve/openai_server.py:1383`). If you manually restart
  the proxy without re-exporting the env var, the endpoint returns an
  empty list.
