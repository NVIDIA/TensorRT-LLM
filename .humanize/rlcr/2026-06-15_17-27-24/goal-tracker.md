# Goal Tracker

<!--
RULES:
- IMMUTABLE SECTION: Do not modify after Round 0 initialization
- MUTABLE SECTION: Update each round, document all changes
-->

## IMMUTABLE SECTION

### Ultimate Goal
Implement and validate an evidence-backed, shape-gated dispatch path for FP8 block-scaling GEMM across the target devices H100 and H200 (SM90/Hopper) and B200 (SM100/Blackwell), selecting per shape and per architecture between the TRT-LLM default kernel and Direct DeepGEMM through a correctness-gated, persistent dispatch cache plus hard static guards, with no online profiling on the production inference path. On SM90 the DeepGEMM candidate is the existing Hopper 1d2d path; on SM100/B200 it is DeepGEMM's `sm100_fp8_gemm_1d1d` (1d1d, a distinct scale contract), which is NOT yet wired into the TRT thop and has no existing local evidence — wiring and validating it is in scope (DEC-8, resolved full-target). TRT-LLM default behavior is preserved for LLM decode / small-M shapes and for out-of-scope architectures (SM120 GeForce Blackwell, Ada). The work produces local correctness + op-level evidence plus end-to-end evidence (synthetic shape-frequency replay AND a full model run) sufficient to decide whether an upstream TRT-LLM issue/PR is justified. DeepGEMM is never made a global unconditional default. Upstream issue/PR submission stays blocked on local perf validation and TRT-LLM owner feedback.

### Acceptance Criteria
Full positive/negative tests are in `fp8-blockscale-gemm-dispatch.plan.md` (repo root of TensorRT-LLM). Titles:
- AC-1: Platform scope safety — only target arches (SM90 H100/H200, SM100 B200) route shape-gated; out-of-scope arches (SM120, Ada) unchanged; forcing DeepGEMM where no validated path exists raises.
- AC-2: Small-M hard guard — `M <= M_SMALL` (default 512) routes TRT, no tuning/cache lookup.
- AC-3: Counterexample denylist — exact shapes (e.g. `65536x3072x3072`) route TRT, overriding any bucket.
- AC-4: Deterministic + observable routing — unit-testable via synthetic cache; distinct large M (same N,K) must not collapse.
- AC-5: Correctness gate — BF16 allclose passes before any timing row is cached.
- AC-6: Cache identity + invalidation — key embeds SM, device class, build id, DeepGEMM version+availability, policy version, backend candidate set; stale/cross-arch entries ignored.
- AC-7: Conservative fallback incl. capture — cache-miss / CUDA-graph-capture / global-autotune → TRT, no online profiling, no cache mutation.
- AC-8: Evidence coverage (deliverable) — small-M LLM, mixed M, diffusion large-M, counterexample on SM90 (H100/H200) and SM100 (B200), PLUS synthetic shape-frequency replay AND full model run, each with correctness + median.
- AC-9: Layout revalidation — non-contiguous/unsupported-layout DeepGEMM cache entry revalidates and falls back to TRT.
- AC-10: DeepGEMM-disabled build — `TRTLLM_ENABLE_DEEP_GEMM_THOP` off → TRT-only; forced DeepGEMM raises; DeepGEMM cache entries ignored on availability mismatch.

---

## MUTABLE SECTION

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan from gen-plan (converged) | - | - |
| 0 | Folded multi-arch re-review (task1 analysis) into task8 contract: SM100 path needs a dedicated internal op, `can_use_deep_gemm_blackwell`, SM100-only capability gate, packed UE8M0 1d1d scale conversion, major-order/stride/contiguity checks, SM100 JIT/runtime init; relax `backend==DeepGemm => sm==90` to an arch-specific gate (SM90 Hopper path, SM100 Blackwell path, SM103/SM120/Ada unchanged); `fp8_block_scale_gemm_blackwell` is the mandatory B200 fallback; cache `device_class` split per H100/H200/B200; cache identity carries concrete backend IDs (sm90_trt, sm90_deepgemm_1d2d, sm100_trt_blackwell, sm100_deepgemm_1d1d) | Codex multi-arch re-review (bnwp9izb8) found the multi-arch delta not yet converged; DEC-3 added B200 | refines AC-1, AC-8, AC-10 (SM100 cases already in plan AC text); no AC removed |

#### Active Tasks
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1 design lock (precedence, per-arch candidates, cache identity, capture rule, Blackwell 1d1d contract) | AC-1,3,4,6,7 | done | analyze | codex | covered by 2 gen-plan rounds + multi-arch re-review (bnwp9izb8); findings in Evolution Log |
| task2 SM90 Hopper internal forced-backend ops (public op unchanged) + DeepGEMM JIT IncludeParser init fix | AC-1,10 | done | coding | claude | R0: compiled+linked on GH200; BOTH ops numerically verified — trtllm op exact==env `trtllm`, deep_gemm op exact==env `direct_deep_gemm`, both match BF16 ref (6.84e-04). Committed. |
| task8 Blackwell SM100/B200 internal op + 1d1d scale conv + SM100 gate (needs B200 node) | AC-1,5,9,10 | pending | coding | claude | queued out of Round 0 (B200 node) |
| task3 arch-generic unified runner + static guards + layout revalidation | AC-1,2,3,9,10 | pending | coding | claude | depends task2,task8 |
| task4 offline correctness+profiling cache builder (BF16 gate) | AC-5,6 | pending | coding | claude | depends task3 |
| task5 custom runtime dispatcher + debug harness + routing unit tests | AC-4,6,7,9 | pending | coding | claude | depends task3,task4 |
| task6 evidence sweep (SM90+B200) + synthetic replay + full model run | AC-8 | pending | coding | claude | depends task4,task5 |
| task7 upstream justification + owner sync prep | AC-8 | pending | analyze | codex | depends task6 |

### Blocking Side Issues
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| Exact SM100 TRT→DeepGEMM scale recipe (dtype/shape, FP32→packed-UE8M0, rounding); does model-level accuracy need more than per-op BF16 allclose | 0 | task8/task6 detail; SM90 mainline first | starting task8 (B200) |
| device_class canonicalization (H100 PCIe/SXM/NVL, H200, B200/GB200) | 0 | task4/task5 cache-identity detail | starting task4 |
| Full-model-run definition + whether synthetic replay derives from same trace | 0 | task6 deliverable detail | starting task6 |
| ~~DeepGEMM validation harness / JIT IncludeParser uninitialized~~ RESOLVED R0: added `IncludeParser::prepare_init` to thop; run with venv python + DeepGEMM root = venv `site-packages/deep_gemm` (bundles cutlass). Recipe reusable for task4/task6. | 0 | resolved | done |

### Completed and Verified
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
