# Round 0 Summary

## Mainline objective (from round-0-contract)
Lock the multi-arch design (task1) and implement task2: internal forced-backend ops for the SM90/Hopper path, public op unchanged; build on the GH200 node; confirm the SM90 paths still work. Target AC-1, AC-10.

## What Was Implemented
- **task2 (coding) — internal forced-backend ops** in `cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp`:
  - `fp8_block_scaling_gemm_trtllm(mat1, mat2, mat1Scale, mat2Scale)` — forces the TRT-LLM default kernel for the current arch (SM switch, never DeepGEMM).
  - `fp8_block_scaling_gemm_deep_gemm(mat1, mat2, mat1Scale, mat2Scale)` — forces Direct DeepGEMM; SM90 only today (forwards to the existing `fp8_block_scaling_gemm_hopper_deep_gemm`); raises on non-SM90 and on builds without `TRTLLM_ENABLE_DEEP_GEMM_THOP`.
  - Both bypass the process-global `TRTLLM_FP8_BLOCK_SCALING_GEMM_BACKEND` env var → deterministic per-call selection (concurrency / CUDA-graph safe).
  - Registered both ops in `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL`.
  - Public `fp8_block_scaling_gemm_impl` schema and its env-var path: **unchanged**. Copyright year updated to 2026.
- **task1 (analyze)** — design locked via 2 gen-plan convergence rounds + the multi-arch re-review; SM100/B200 refinements folded into goal-tracker Plan Evolution Log.

## Files Changed
- `cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp` — two new ops + registration + copyright (the only source change).
- `.humanize/rlcr/2026-06-15_17-27-24/{goal-tracker.md, round-0-contract.md, round-0-summary.md}`.
- `.humanize/bitlesson.md` — added BL entry (DeepGEMM JIT IncludeParser init).
- (project-root test scaffolding, outside this git repo: `round0_new_ops_sanity.py`, `round0_run_sanity.sh`.)

## Validation (GH200, alloc `sc-2653078`, node `lego-cg1-qs-16`, rc17 container)
- **Compile + link: PASS.** `th_common` built cleanly with the change (`fp8BlockScalingGemm.cpp.o` compiled, `libth_common.so` linked, 0 errors). The build script's later `import tensorrt_llm` failure is an unrelated rc17 packaging gap (`kv_cache_manager_v2.rawref._rawref` missing), not this change.
- **`fp8_block_scaling_gemm_trtllm`: PASS.** On `11250x5120x5120`, output is exact-equal to `fp8_block_scaling_gemm_impl` (env `trtllm`) and matches the BF16 reference → AC-1 SM90 default routing.
- **`fp8_block_scaling_gemm_deep_gemm`: routing verified; numerical sanity BLOCKED.** The op correctly dispatches into the existing Hopper DeepGEMM path (reaches the DeepGEMM C++ JIT). The JIT then fails to open `deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` because the DeepGEMM `IncludeParser` is only initialized as a side effect of `import deep_gemm`; `init_deep_gemm_runtime_once` initializes `Compiler`+`KernelRuntime` but not `IncludeParser`, and `deep_gemm` is not importable in the minimal standalone harness used here. This equally affects the pre-existing `impl`+`direct_deep_gemm` path — environment/harness issue, not a defect in the new op.

## Remaining Items
- Stand up a proper DeepGEMM validation harness (importable `deep_gemm` via the build's `.venv-3.12`, or self-init the JIT IncludeParser) and re-run the DeepGEMM-forced numerical sanity. Queued; also required for task4/task6.
- task8 (Blackwell SM100/B200 op + 1d1d scale conversion) — needs a B200 node.
- task3–task7 — not started. This round intentionally scoped to task2 (see round-0-contract); the overall plan is NOT complete.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260615-deepgemm-jit-includeparser-init
Notes: The C++ thop DeepGEMM direct path depends on `import deep_gemm` having run to initialize the DeepGEMM JIT `IncludeParser` (kernel-source include root). `init_deep_gemm_runtime_once` initializes `Compiler`/`KernelRuntime` but not `IncludeParser`, so a minimal harness that loads `libth_common.so` without importing `deep_gemm` fails with "Failed to open: deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh". Fix in harnesses: import `deep_gemm` first (as `bench_trtllm_deepgemm_three_way.py` does) or run with the build's venv where it is available.
