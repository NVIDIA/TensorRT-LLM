# FMHA precompiled kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> or superpowers:subagent-driven-development. Track the steps below.

**Goal:** Remove first-use FMHA compiler stalls for every supported variant in
compatible Linux wheels, and verify the reported MiniMax workload on B300 and
GB300 without increasing runtime warmup.

**Architecture:** Generate the full matrix and helper modules in an isolated
wheel-build directory, ship a validated artifact set and compatibility
manifest, and make `fmha_sm100.jit` load that set before consulting its existing
JIT cache. Source/editable installs and diagnostic modes retain the existing
fallback. Build and runtime must use the same source/headers fingerprint and
TVM FFI version; runtime must not require nvcc.

**Tech Stack:** Python 3.10+, CUDA/nvcc, Ninja, TVM FFI, setuptools, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-fmha-precompiled-kernels-design.md`

## Global Constraints

- Target main only; do not increase warmup, pre-seed a node cache, or hardcode
  the triggering variant as the fix.
- Cover all 350 legal dispatch variants and the plan, top-k, and reduction
  helpers for the supported architecture set.
- Preserve debug modes, source/editable installs, unsupported build modes,
  out-of-tree builds, x86-64 and ARM64, and precompiled-wheel extraction.
- Use private temporary directories; never delete another process's files.
- Run clean-cache B300 and GB300 correctness and performance validation,
  including the original 1024/1024 case, before claiming a fix.

## Review Focus

- Source or CUTLASS changes must invalidate packaged binaries; exercise both.
- Runtime FFI ABI or host-architecture mismatch must not load a stale binary.
- All debug flags must bypass production artifacts, including with a custom
  cache root.
- A missing variant or helper must fail wheel generation rather than produce a
  falsely complete artifact set.
- CUDA 12.8 SM100, CUDA 12.9+ SM103, non-SM10x, Windows, and editable modes
  must retain compatible build behavior.

## Task 1: Prove packaged-artifact selection

**Files:** `3rdparty/patches/msa_strided_paged_kv.patch`;
`tests/unittest/_torch/attention/sparse/msa/test_msa_precompiled.py`.

- [ ] Add tests that construct a temporary matching artifact manifest and
      exercise the actual MSA variant manager and the plan, top-k, and
      reduction loaders. A matching artifact must be selected without calling
      a compiler or writing to the user cache.
- [ ] Add negative tests for missing/incomplete artifacts, modified FMHA and
      CUTLASS sources, mismatched TVM FFI/host architecture, and each debug
      flag. Check that source/editable fallback remains available.
- [ ] Run the tests against the existing patched MSA source and confirm they
      fail because packaged loading is absent.
- [ ] Implement manifest validation and packaged loading in the MSA patch,
      preserving existing cache locks and JIT fallback. Re-run the tests.

## Task 2: Generate complete artifacts in wheel builds

**Files:** `scripts/build_wheel.py`, `scripts/build_msa_aot.py`, `setup.py`,
`3rdparty/patches/msa_strided_paged_kv.patch`;
`tests/unittest/scripts/test_build_msa_aot.py`.

- [ ] Add tests for Linux/SM architecture and nvcc capability gating, complete
      dispatch enumeration, expected TVM export names and helper artifacts,
      private temporary/cache directories, and failure on missing binaries.
- [ ] Run the new tests before implementation and verify the expected failures.
- [ ] Add a safe build entry point based on the vendored MSA compiler. Select
      nvcc-supported SM100/SM103 images requested by the wheel; do not force
      unsupported targets. Build all variants and helpers from the staged
      package, without global temporary-file cleanup or GPU access.
- [ ] Validate all expected exported variant symbols and helper binaries,
      produce the manifest, stage the binaries, and include them in wheel and
      precompiled/editable extraction paths. Re-run the tests.
- [ ] Measure build wall time, peak memory, wheel size, and first load latency.
      If the full aggregate exceeds linker/resource limits, replace it with
      bounded shards and repeat all coverage checks before proceeding.

## Task 3: Validate and submit

**Files:** implementation above; test artifacts in the isolated diagnostics
workspace rather than the source tree.

- [ ] Build x86-64 and ARM64 wheels. Verify the packaged artifact manifest,
      every exported variant and helper, no compiler invocation on artifact
      hits, and numerical MSA tests on actual supported GPUs.
- [ ] Compare unmodified and fixed wheels under matched clean-cache B300 and
      GB300 runs, both the reduced diagnostic and original 1024/1024 case.
      Capture the triggering mixed step and check no target compiler process is
      spawned. Measure startup and first-load cost as well as throughput.
- [ ] Re-fetch upstream main, run relevant tests and checks, independently
      review the final diff, commit with DCO sign-off, and open a PR targeting
      NVIDIA/TensorRT-LLM `main` with the bug and evidence linked.
