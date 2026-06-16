# Round 0 Contract

## Mainline Objective
Lock the multi-arch design (task1, done via the gen-plan convergence + multi-arch re-review) and implement task2: add **internal forced-backend ops** for the SM90/Hopper path so the runner can call TRT-default and Direct-DeepGEMM deterministically per call, with the public `fp8_block_scaling_gemm_impl` schema and its env-var path left unchanged. Build the C++ on the allocated GH200 node and confirm the existing SM90 default and direct paths still produce identical output (and match the BF16 reference) on a sanity shape.

## Target ACs
- AC-1 (platform scope safety: SM90 target arch routes correctly; out-of-scope arches unchanged; forcing DeepGEMM where unsupported raises)
- AC-10 (DeepGEMM-disabled build behaves TRT-only; forced DeepGEMM raises)

## Blocking Side Issues In Scope
- None identified yet.

## Queued / Out Of Scope This Round
- task8 (Blackwell SM100/B200 internal op + 1d1d scale conversion) — needs a B200 node; SM90 mainline first.
- task3–task7 (runner, cache builder, dispatcher, evidence, upstream) — depend on task2/task8.
- Open design questions (SM100 scale recipe, device_class canonicalization, full-model-run definition) — queued in goal-tracker.

## Round Success Criteria
- Internal SM90 forced-backend ops added (TRT default + Hopper DeepGEMM) callable deterministically; public op unchanged.
- C++ builds clean on the GH200 node (compute node `lego-cg1-qs-16`, alloc `sc-2653078`).
- On a sanity shape (e.g. `11250x5120x5120`), SM90 default and forced-DeepGEMM outputs match each other (exact) and match the BF16 reference within tolerance.
- Forcing DeepGEMM on a build without `TRTLLM_ENABLE_DEEP_GEMM_THOP` raises clearly (AC-10 negative).
- goal-tracker.md and this contract committed; round-0-summary.md written.

## Environment
- Git repo / project root for RLCR: `TensorRT-LLM/` (top-level `.git` is empty; real repo is the vendored TRT-LLM checkout). Working branch `fp8-blockscale-gemm-dispatch`, base `main`.
- Compute: `ssh-gw` alloc `sc-2653078`, node `lego-cg1-qs-16` (GH200, SM90, 4h). Build/run via Docker release container on the node (see `run_deepgemm_cxx_validation_container.sh`).
