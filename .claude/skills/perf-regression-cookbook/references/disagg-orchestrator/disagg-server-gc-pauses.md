---
id: case-disagg-server-gc-pauses
type: regression-case
family: execution-and-graph
module: disagg-orchestrator
maturity: full
regression_class: [host-work-added]
signals: [throughput-drop, gpu-idle-between-steps, host-time-increase]
subsystems: [serve-endpoint, runtime-python, scheduler-executor]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5522847"]
commits: ["740340dd17e9"]
success_prs: [7858]
failed_prs: []
---

# Python GC pauses in the disagg router starved CTX/GEN workers at high concurrency

> Part of the [Disagg orchestrator regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5522847` · commit `740340dd17e9` · PR #7858 —
  "[https://nvbugs/5522847][fix] Disable GC on disagg server and client".
- **Symptom:** a performance drop caused by disagg-server GC in
  high-concurrency scenarios. The PR carries real numbers — config
  `ctx4_gen1_dep32_batch256_eplb0_mtp1`, Output token throughput **187,466.23**
  (GC enabled) vs **240,964.24** (GC disabled), i.e. **−22 %** with GC on,
  reproduced at commit `8881f125f8b8` (one commit ahead of `release/1.1.0rc2`).
  **No model and no GPU type are stated** — cite the config string and the delta,
  not a platform.
- **Root cause:** the disaggregated server's router is **pure Python** and sits
  between clients and the CTX/GEN workers. At high concurrency it allocates
  enough short-lived objects to trigger frequent generational collections, and —
  per the comment the PR adds — during a long GC pause "requests are not
  immediately forwarded to CTX workers and GEN workers, causing them to run with
  small batch sizes". So the GPU loss is indirect and second-order: the pause does
  not slow any kernel, it *starves batching*, and a smaller batch is a permanently
  worse operating point for the duration. The PR also records the allocation
  evidence: `count0` "increases by fewer than 1,000 after every 200,000 requests,
  while the maximum value of `count0` exceeded 3,000,000" — i.e. the surviving-object
  count barely grows while gen-0 churn is enormous, which is precisely the profile
  where collection cost is pure waste.
- **How introduced:** `pre-existing-gap` — no culprit commit. This is what the
  Python router has always done; it becomes visible only above a concurrency
  threshold, which is why it reads as a "drop" without a bisect.
- **Fix mechanism:** disable GC on both ends of the disagg path.
  Server: `if int(os.getenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "1")): gc.disable()`
  immediately before `asyncio.run(server(...))` — **default ON**, with the env var
  as the escape hatch. Client: `benchmark_serving.py`'s existing
  `gc.collect(); gc.freeze()` becomes `gc.disable()`. Note what this trades: with
  GC off, reference cycles are never reclaimed, so the fix is safe only because the
  router's surviving-object count is flat (the `count0` evidence above). Do not
  copy `gc.disable()` into a component that accumulates cycles.
- **Detection signal:** A/B the switch — run the same concurrency with
  `TRTLLM_DISAGG_SERVER_DISABLE_GC=0` and `=1`; a large gap at high concurrency
  and none at low concurrency is this. On a pre-fix tree,
  `grep -n "gc.freeze" tensorrt_llm/serve/scripts/benchmark_serving.py` matches
  (post-fix it is `gc.disable()`). In-process, `gc.set_debug(gc.DEBUG_STATS)` or
  sampling `gc.get_count()` shows gen-0 churn with a flat survivor count. The
  workload-side signature is **worker batch sizes smaller than the offered
  concurrency implies** with no queueing error — check batch size before checking
  kernels.
- **Prevention/guard:** **no test, no guard, no log line.** The env var is the only
  control, and it defaults to the fixed behaviour, so a regression here would be
  someone flipping the default back. Rule: any pure-Python component on the request
  path at 4-digit concurrency should be evaluated for GC pauses before its logic is
  optimized — the pause is invisible in every per-kernel profile.
- **Generalizes to:** `pattern-host-work-on-hot-path`; carries to routers,
  proxies, tokenizer servers and benchmark clients — anything Python between the
  client and the executor. Related but distinct mechanisms live in the
  **instability** cookbook, where the same collector produces variance rather than
  a level shift: `pattern-gc-pause-on-decode-path` and
  `pattern-gc-driven-collective-destruction`. If the symptom is rep-to-rep
  variance or a cross-rank hang rather than a steady throughput deficit, read those
  instead of this case.
