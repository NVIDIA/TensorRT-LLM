# FMHA precompiled kernels

## Intent

Eliminate request-time compilation of supported SM100/SM103 FMHA variants in
TensorRT-LLM wheels. A cold-cache MiniMax-M3 request currently blocks the
executor while `fmha_sm100` compiles a previously unseen variant; in a
four-rank main reproduction, that compilation consumed 139.614 s of a
139.862 s mixed prefill/decode step. The change must address all supported
variants, not the observed request shape alone.

## Approaches

1. Build and ship the complete supported variant matrix with the wheel, then
   load the matching artifact directly. This removes first-use compilation for
   packaged variants, at the cost of wheel build time and size. This is the
   proposed approach, subject to measuring those costs and validating both CPU
   architectures.
2. Compile ahead during warmup or serve startup. This shifts the delay and
   remains vulnerable to incomplete shape coverage; it is excluded by the
   requested scope.
3. Substitute a different kernel or change dispatch for the one observed
   shape. This changes execution and may leave other variants unresolved; it
   does not satisfy the requested general fix.

## Requirements and constraints

- Normal Linux wheels that support SM100/SM103 include usable precompiled
  artifacts for the complete legal FMHA dispatch matrix and required planning,
  reduction, and top-k helper kernels. Verify the matrix against the same
  dispatch definitions used at runtime; do not maintain a hand-picked list.
- A normal cold-cache request selects the packaged artifact without spawning a
  compiler. Retain a documented source-build fallback for environments that do
  not ship the artifacts, and preserve explicit diagnostic build modes.
- Never use an incompatible artifact after source or build-mode changes. Ensure
  the wheel's binary is built from the exact sources and CUDA/CUTLASS/TVM FFI
  dependencies staged in that wheel; account for editable installations. The
  existing cache digest covers FMHA sources and `jit.py`, but does not cover
  CUTLASS, the CUDA toolchain, or the TVM FFI ABI and is insufficient by itself.
- Do not require a GPU to build the artifacts. Avoid deleting unrelated
  compiler temporary files or relying on a user's cache directory.
- Support the project's x86-64 and ARM64 wheel pipelines and honor supported
  CUDA architectures. Preserve SM90-only, Windows, older-toolkit, editable,
  precompiled-wheel extraction, and out-of-tree build behavior. Do not change
  release/1.3 or increase runtime warmup.
- Measure build time, peak resource use, and size before deciding build
  parallelism and packaging. Reject incomplete or incompatible artifacts.
- Build in an isolated fresh directory. The existing helper's aggregate can be
  incomplete when per-variant cache entries already exist, and its global
  `/tmp/tmpxft_*` cleanup is unsuitable for a shared build node. Validate all
  350 runtime variants and the three helper artifacts before publication.

## Validation

- Unit tests cover all dispatch variants, artifact selection, missing/stale
  artifacts, diagnostic-mode isolation, and no compiler spawn on an artifact
  hit. Include a test that a normal build failure cannot silently produce a
  wheel that claims complete coverage.
- Build and inspect actual wheels, and run numerical attention coverage on
  supported hardware against the existing backend. Compare clean-cache
  baseline and fixed runs on the same B300 and GB300 hardware, commit-derived
  environment, request shape, and measurement settings. Capture the exact
  mixed step and verify the target JIT is absent without relocating it into
  startup. Keep the full 1024/1024 original case as the end-to-end check.
- The diagnostic probe mutates `jit.py` and therefore changes the existing
  source fingerprint. Test the final wheel without that mutation, or explicitly
  verify that the diagnostic does not invalidate its packaged artifacts.
- Compare the reproducible 1024/128 diagnostic only with the same reduced
  case; do not use it as a throughput substitute for the original report.
