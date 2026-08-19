---
id: case-pdl
type: case
family: runtime-execution
maturity: full
bottleneck: [launch]
signals: [many-small-kernels, small-batch-decode]
architectures: [sm90, sm100]
model_scope: [model-agnostic, dense, moe]
phase: [any-phase]
patterns: [pattern-pdl-producer-consumer-overlap]
accuracy_risk: lossy
apply_via_kind: [default-on, env-var, kernel-change]
knobs: [TRTLLM_ENABLE_PDL]
specialists: [kernel-cuda-specialist]
commits: ['8462cf6c96f1', '84d2f1281857', '21a93fbf9d10', '9e7b50aefb', '6ee8dbfe0b', 'ba25b6afae08', 'bf7142f8d1', '1c4dacb19a52', '34e2fa5c96']
eligibility:
  - "SM90+ only (gated on getSMVersion() >= 90)"
measured: []
---

# Enable Programmatic Dependent Launch (PDL) to hide kernel-launch/dependency latency

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `8462cf6c96f1` [TRTLLM-9578][feat] make PDL enabled by default (#9695); related: `84d2f1281857` add PDL support for more kernels (#7977, `launchWithPdlWhenEnabled`), `21a93fbf9d10` PDL for CuteDSL + overlap MoeOutputMemset (#10043). Per-kernel accuracy WARs (PDL disabled): `9e7b50aefb` quant kernels (#10285), `6ee8dbfe0b`/`ba25b6afae08` TinyGEMM, `bf7142f8d1` TRTLLM-GEN routing NaN (#11994), `1c4dacb19a52` dsv3 MoE (#9799), `34e2fa5c96` Qwen3-235B ATP (#9530).
- **Applies when:** launch-bound on SM≥90 — back-to-back dependent kernels (MoE comm→GEMM, layernorm→GEMM, router/quant→GEMM) where each waits for the prior to drain and launch overhead / grid-tail is exposed; chains of many small kernels at low concurrency. Not available below SM90 (gated on `getSMVersion() >= 90`).
- **Mechanism:** PDL lets a dependent kernel begin grid setup/preamble while the producer's tail still drains, via `cudaLaunchAttributeProgrammaticStreamSerialization`, overlapping launch/tail latency. The default-on change flips `enablePDL` to `true`, making `TRTLLM_ENABLE_PDL=0` an explicit opt-out (still SM≥90 gated).
- **Generalizes to:** producer-tail / consumer-preamble overlap for any dependent-kernel pair; carries to new kernels via the shared helper, to CuteDSL/trtllm-gen kernels, and to MoE all-to-all comm kernels; adapt by adding the launch attribute (or `launchWithPdlWhenEnabled`), gating arch ≥ sm90, AND accuracy-checking each kernel (PDL is not universally safe).
- **Apply via:** env `TRTLLM_ENABLE_PDL` (default `1` on SM≥90; `0` disables globally). New kernels: `tensorrt_llm::common::launchWithPdlWhenEnabled(...)` or set the launch attribute guarded by `getEnvEnablePDL()`. Delegate to **kernel-cuda-specialist**.
- **Expected effect:** lower exposed launch/dependency latency between kernels → higher throughput in kernel-chain-bound regions; direction only — measured Δ to be recorded from run.
- **Accuracy risk:** lossy/risky per-kernel — PDL changes inter-kernel ordering/visibility and HAS produced real accuracy/NaN regressions (TinyGEMM, quant, trtllm-gen FMHA/routing, dsv3 MoE, Qwen3-235B). Needs an on-disk accuracy record + per-kernel rollback; some kernels must keep PDL off.
- **Verify:** kernel-chain region latency / inter-kernel gap drops in nsys; AND an accuracy/parity eval (WAR commits exist because kernels regressed) — no NaN, score within tolerance vs `TRTLLM_ENABLE_PDL=0`.
- **Rollback:** global `TRTLLM_ENABLE_PDL=0`; per-kernel disable the attribute. Trigger: accuracy drop or NaN in any affected module.
- **Prior art:** PRs #9695, #7977, #10043 (+WARs). Files: `cpp/.../common/envUtils.{cpp,h}` (`getEnvEnablePDL`, `launchWithPdlWhenEnabled`), `cutlass_kernels/moe_gemm/moe_kernels.cu`, `fusedLayernormKernels/*`, `mlaKernels.cu`, `fusedMoeCommKernels.cu`. Owning specialist: **kernel-cuda-specialist**.
