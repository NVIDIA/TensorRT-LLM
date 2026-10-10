---
id: case-kv-aware-adp-cold-start
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [cold-start-imbalance]
signals: [first-iter-spike, rep-to-rep-variance]
subsystems: [adp-router, kv-cache]
introduced_via: [preexisting]
phase: [prefill]
patterns: [pattern-cold-start-worker-imbalance]
nvbugs: []
commits: ["e796f16c81bc"]
success_prs: [14307]
failed_prs: []
---

# KV-aware ADP router pins shared prefix to one rank until traffic spike hits cold ranks

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `e796f16c81bc` · PR #14307 — cold-start warmup for
  KV-aware ADP router. Merged to `main` 2026-05-22. No NVBug: the PR is titled
  `[None]` and its body names none, so this is PR-only provenance.
- **Symptom (variance signature):** with `match_rate_threshold=0.1` the
  cache-affinity gate stays ON for any system prompt longer than ~10% of a
  request. On low-traffic sequential arrivals, each new request pins to
  the first warm rank; other ranks never see the prompt. When an eventual
  traffic spike arrives, cold ranks pay the full prefill cost re-computing
  the same shared prompt — the resulting spike is a first-N-request
  latency outlier that trips perf-CI and produces rep-to-rep variance
  depending on the arrival pattern.
- **Root cause:** cache-affinity scoring uses `(req_tokens - match_len)` as
  the dominant term once a rank has cached the shared prompt, so a fresh
  ADP router state combined with sequential arrivals never spreads the
  shared prefix.
- **How introduced:** the KV-aware routing policy was authored to optimize
  steady-state locality; the transient before every rank sees the shared
  prefix was not accounted for.
- **Fix mechanism:** track `_pending_warmup_ranks: Set[int]` in
  `KVCacheAwareADPRouter`, initialised to `{0, …, tp_size-1}`; relaxed requests
  with no explicit target synthesise `min(_pending_warmup_ranks)` while the set
  is non-empty, and any assigned `target_dp_rank` (explicit-strict or
  synthesised) discards its rank, so the first `tp_size` requests spread across
  ranks and every rank caches the (assumed shared) system prompt before affinity
  scoring pins traffic. Cap-saturated ranks also discard, so the synthesiser
  cannot loop on a busy rank; once the set empties, warmup is dormant forever and
  routing is pure scoring.
  **The mitigation is opt-in and OFF by default** —
  `AttentionDpConfig.kv_cache_routing_cold_start_warmup` defaults to `False`,
  which leaves `_pending_warmup_ranks` empty and preserves pre-warmup routing.
  This matters twice over: an agent checking "is the fix present in this tree?"
  must check the *config value*, not just the code, because a tree containing
  #14307 still exhibits the full cold-start pinning unless the flag is set; and
  the default is off deliberately, since the PR notes the warmup "can scatter
  requests onto cold ranks and bypass cache-affinity scoring entirely for the
  first `tp_size` requests — wasting prefill work the scoring path would
  otherwise consolidate" on diverse-prompt workloads. So this is a *conditional*
  fix for shared-long-prompt traffic, not a universal one.
- **Detection signal:** first N requests all landing on rank 0 in a
  gen-only / ADP trace; cold ranks reporting zero prompt cache hits after
  the warm-up window; `grep -n '_pending_warmup_ranks\|kv_cache_routing_cold_start_warmup\|KVCacheAwareADPRouter' tensorrt_llm/_torch/pyexecutor/`
  to confirm the guard is present **and** that the config flag is enabled — the
  code being present proves nothing here.
- **Prevention/guard:** any affinity-scoring routing policy must have a
  cold-start ramp; a unit test that submits N=world_size identical prompts
  and asserts each rank sees it exactly once. #14307's own suite is the model:
  `TestKVCacheAwareADPRouterWarmup` builds its router with
  `cold_start_warmup=True` to reach the new path, plus a `test_disabled_by_default`
  asserting the empty pending set when the flag is absent — a default-off
  mitigation needs both halves tested or the off-by-default case rots silently.
- **Generalizes to:** `pattern-cold-start-worker-imbalance`; carries to any
  KV-cache-aware routing (disaggregated serve gen-side, prefix-caching
  DP), MoE routing that reserves per-expert warm state, and consistent-
  hashing routers where sequential arrivals produce degenerate hot-node
  distributions.
