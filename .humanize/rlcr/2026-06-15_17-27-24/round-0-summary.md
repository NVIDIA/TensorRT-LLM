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
- **DeepGEMM JIT init fix (coding)** in the same file: `init_deep_gemm_runtime_once` now also calls `deep_gemm::IncludeParser::prepare_init(deep_gemm_root)` (previously missing) and includes `jit/include_parser.hpp`, so the C++ DeepGEMM direct path self-initializes its JIT include root instead of depending on a separate `import deep_gemm`. Fixes the new DeepGEMM op and the pre-existing `direct_deep_gemm` path.
- **task1 (analyze)** — design locked via 2 gen-plan convergence rounds + the multi-arch re-review; SM100/B200 refinements folded into goal-tracker Plan Evolution Log.

## Files Changed
- `cpp/tensorrt_llm/thop/fp8BlockScalingGemm.cpp` — two new ops + registration + copyright (the only source change).
- `.humanize/rlcr/2026-06-15_17-27-24/{goal-tracker.md, round-0-contract.md, round-0-summary.md}`.
- `.humanize/bitlesson.md` — added BL entry (DeepGEMM JIT IncludeParser init).
- (project-root test scaffolding, outside this git repo: `round0_new_ops_sanity.py`, `round0_run_sanity.sh`.)

## Validation (GH200, alloc `sc-2653078`, node `lego-cg1-qs-16`, rc17 container)
- **Compile + link: PASS.** `th_common` built cleanly with the change (`fp8BlockScalingGemm.cpp.o` compiled, `libth_common.so` linked, 0 errors). The build script's later `import tensorrt_llm` failure is an unrelated rc17 packaging gap (`kv_cache_manager_v2.rawref._rawref` missing), not this change.
- **`fp8_block_scaling_gemm_trtllm`: PASS.** On `11250x5120x5120`, output is exact-equal to `fp8_block_scaling_gemm_impl` (env `trtllm`) and matches the BF16 reference → AC-1 SM90 default routing.
- **`fp8_block_scaling_gemm_deep_gemm`: PASS.** Output is exact-equal to `fp8_block_scaling_gemm_impl` (env `direct_deep_gemm`) and matches the BF16 reference. Reaching this required a real code fix plus a harness/env correction:
  - **Code fix (committed):** `init_deep_gemm_runtime_once` initialized `Compiler` + `KernelRuntime` but NOT `IncludeParser`, whose static include root was left empty → JIT failed to open `deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`. DeepGEMM's own canonical init (`csrc/apis/runtime.hpp`) inits all three; added `IncludeParser::prepare_init(deep_gemm_root)` to match. This also fixes the pre-existing `impl`+`direct_deep_gemm` path when run from a process that has not imported the standalone `deep_gemm` (separate C++ runtime).
  - **Env correction:** the DeepGEMM JIT root must be a packaged deep_gemm whose `include/` bundles CUTLASS (e.g. the build's venv `site-packages/deep_gemm`), not the raw `DeepGEMM/deep_gemm` (no `include/cutlass`), or NVCC fails on `cutlass/arch/barrier.h`.

## Remaining Items
- DeepGEMM validation harness: RESOLVED this round (IncludeParser code fix + venv python `/code/TensorRT-LLM/.venv-3.12/bin/python3` + DeepGEMM root = venv `site-packages/deep_gemm` which bundles CUTLASS under `include/`). This run recipe is reusable for task4/task6.
- task8 (Blackwell SM100/B200 op + 1d1d scale conversion) — needs a B200 node.
- task3–task7 — not started. This round intentionally scoped to task2 (see round-0-contract); the overall plan is NOT complete.

## BitLesson Delta
Action: add
Lesson ID(s): BL-20260615-deepgemm-jit-includeparser-init, BL-20260615-deepgemm-jit-root-needs-cutlass
Notes:
- BL-20260615-deepgemm-jit-includeparser-init: `init_deep_gemm_runtime_once` must call `IncludeParser::prepare_init` (in addition to `Compiler`/`KernelRuntime`); otherwise the JIT include root is empty and kernel `.cuh` files fail to open. Fixed in code this round (the standalone `import deep_gemm` only inits a SEPARATE C++ runtime's IncludeParser, not libth_common's).
- BL-20260615-deepgemm-jit-root-needs-cutlass: `TRTLLM_DEEP_GEMM_ROOT` must point to a packaged deep_gemm whose `include/` bundles CUTLASS (e.g. the build venv `site-packages/deep_gemm`), not the raw `DeepGEMM/deep_gemm` (no `include/cutlass`), or NVCC JIT fails on `cutlass/arch/barrier.h`.
