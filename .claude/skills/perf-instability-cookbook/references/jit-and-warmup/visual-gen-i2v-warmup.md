---
id: case-visual-gen-i2v-warmup
type: instability-case
family: warmup-and-jit
module: jit-and-warmup
maturity: full
instability_class: [warmup-path-mismatch]
signals: [first-iter-spike, midrun-stall]
subsystems: [visual-gen-pipeline]
introduced_via: [incomplete-coverage]
phase: [prefill]
patterns: [pattern-warmup-path-mismatch, pattern-jit-on-hot-path]
nvbugs: []
commits: ["7bb916d2c7d9"]
success_prs: [12351]
failed_prs: []
---

# LTX-2 I2V requests trigger torch.compile recompilation because warmup only exercised T2V

> Part of the [JIT & warmup instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `7bb916d2c7d9` · PR #12351 — Add I2V warmup to
  prevent torch.compile recompilation. Merged to `main` 2026-03-20. No NVBug:
  the PR is tracked as JIRA **TRTLLM-11497** (its baseline scaling table cites
  TRTLLM-11367) and names no nvbug, so this is PR-only provenance — a
  `gh pr list --search "<bugid>"` sweep will never surface it.
- **Symptom (variance signature):** LTX-2 image-to-video (I2V) requests
  suffer a torch.compile recompilation on the first inference step because
  the warmup path only exercised the T2V (text-to-video) path. The metric
  is therefore stable across T2V-only reps but has a large first-iter
  outlier the first time an I2V request lands.
- **Root cause:** warmup runs `forward(image=None)` → `v_timestep` shape
  `(B,)` (scalar, all tokens share same timestep). I2V inference runs
  `forward(image=...)` → `v_timestep` shape `(B, T)` (per-token, first
  frame conditioned at timestep≈0) because of the denoise mask. torch.compile
  sees a new shape and silently recompiles the entire transformer graph on
  `gen_step1`.
- **How introduced:** the warmup was authored around the T2V path;
  I2V-vs-T2V shape divergence was not tracked as a warmup axis.
- **Fix mechanism:** add a second warmup pass with `image=torch.zeros(...)` so
  that both T2V and I2V compiled graphs are cached before serving. The extra
  warmup costs ~5 s (1-GPU) to ~3 s (8-GPU) because most graph fragments are
  already cached from the T2V pass — i.e. the fix trades a few seconds of
  startup for the recompile, it does not eliminate work.
  Measured per PR (B200, NVFP4, VANILLA attn, 768×1280, 121 frames, 40 steps):
  first I2V step `gen_step1` **3.25 s → 0.78 s** (1-GPU) and
  **3.67 s → 0.35 s** (8-GPU); I2V denoise per-step 0.85 → 0.79 s (1-GPU) and
  0.30 → 0.21 s (8-GPU, **1.43×**) — the per-step average moves because the
  step-1 recompile was dragging the 40-step mean up. Reported 8-GPU scaling
  efficiency 1.66× → 3.76×.
- **Detection signal:** the first I2V-serving iter shows a torch.compile /
  inductor compile span in nsys; grep the pipeline for the number of
  distinct compiled entries: `grep -n '_run_warmup\|image=None' tensorrt_llm/`.
- **Prevention/guard:** enumerate every served forward signature that
  affects the compiled graph shape (image=None vs image=…, mask on/off,
  caching flags) and require a warmup pass for each; a warmup-completeness
  assertion at serve-start.
- **Generalizes to:** `pattern-warmup-path-mismatch`; carries to any
  compiled pipeline whose input signature varies at runtime (mask on/off,
  streaming vs non-streaming, tool-call variants, dual-encoder vs single-
  encoder), video / audio pipelines with mode-conditional shapes, and
  torch.compile graphs whose inductor cache is keyed on tensor shape.
