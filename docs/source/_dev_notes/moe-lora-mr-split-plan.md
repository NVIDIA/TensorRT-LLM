# MoE LoRA review-split plan

This is a coordination note for splitting the work on
`user/brb/moe-multi-lora` into reviewable upstream MRs. It is intentionally
checked into `_dev_notes` rather than user-facing docs because it is purely
PR-process scaffolding; once all referenced MRs are merged it can be
deleted.

## Goals

- Split the eager-mode portion of `user/brb/moe-multi-lora` into 2 MRs
  (originally 3 -- helpers and sidecar were folded into a single
  Python-infrastructure MR; see "Consolidation" below).
- Maximize reviewer parallelism: the two MRs are reviewable by different
  reviewer pools simultaneously.
- Keep CI tax bounded: only one MR should pull a full GPU-test cycle; the
  others should land Python-unit-only changes.
- Defer all CUDA-graph capture work to a follow-up MR series ("MR #2"),
  not covered by this plan.

## Consolidation (post-original split)

The original split had three MRs: helpers (#1.1), sidecar (#1.2), and the
feature stack (#1.3). After pushing all three, helpers and sidecar were
folded into the single `user/brb/moe-lora-helpers` branch on the rationale
that both are pure-Python pre-feature plumbing reviewed by the same kind
of reviewer, and that two near-trivial Python-only MRs is more PR overhead
than warranted. The remote `user/brb/moe-lora-sidecar` branch has been
deleted; its commit is reachable via `user/brb/moe-multi-lora` and via the
combined helpers branch. References below to "MR #1.1" now describe the
combined helpers + sidecar MR; references to "MR #1.2" are removed; "MR
#1.3" is renumbered to "MR #1.2" but still refers to
`user/brb/moe-lora-feature`.

## Branch graph

```
origin/main
├── user/brb/moe-lora-helpers   ← MR #1.1: layout + validation helpers + sidecar (Python only)
│   └── user/brb/moe-lora-feature
│                                ← MR #1.2: kernel + integration + reference suite
```

`user/brb/moe-multi-lora` (this branch) is the development trunk that
keeps the entire history including the investigation pair and the
post-mortem regression suite. None of the MR branches reference this
branch directly.

## MR #1.1 — `[TRTLLM-12507][feat] add MoE LoRA layout, validation helpers, and lora_layout.json sidecar`

**Source:** subset of `4407871ce4` (the routed-expert LoRA core) + all of
`6e9b10de96` (sidecar). Two commits on the branch.

**Scope (9 files):**

Helper portion (carved from `4407871ce4`):
- `tensorrt_llm/_torch/peft/lora/moe_layout.py` (new)
- `tensorrt_llm/_torch/peft/lora/validation.py` (new)
- `tests/unittest/_torch/lora/test_moe_layout.py` (new)
- `tests/unittest/_torch/lora/test_moe_lora_validator.py` (new)

Sidecar portion (verbatim from `6e9b10de96` minus its two doc edits):
- `tensorrt_llm/lora_layout_sidecar.py` (new)
- `tensorrt_llm/lora_manager.py` (sidecar discovery + plumbing)
- `tensorrt_llm/_torch/pyexecutor/model_engine.py` (engine wiring)
- `tests/unittest/_torch/lora/test_lora_layout_sidecar.py` (new)
- `tests/unittest/others/test_lora_manager.py` (LoraManager sidecar tests)

The two doc edits inside `6e9b10de96` are intentionally dropped and
land instead in MR #1.2: the dev-note edit because the file does not
yet exist on `origin/main`, and the user-facing `docs/source/features/lora.md`
edit because its "lora_layout.json" subsection wraps the wider
"Routed-Expert MoE LoRA" section that is added by `4407871ce4` in
MR #1.2.

**Self-containment:**
- `moe_layout.py` imports only `torch` + stdlib.
- `validation.py` imports only `tensorrt_llm.lora_helper.LoraConfig` + stdlib.
- `lora_layout_sidecar.py` imports only stdlib.
- The sidecar plumbing in `lora_manager.py` and `model_engine.py` is
  metadata-only -- the C++ LoRA cache row size is determined by
  `LoraModule::localTotalSize`, so the loader still replicates the
  shared tensor across `num_experts`. The flag plumbing is wired
  end-to-end so MR #1.2's kernel honors it; until MR #1.2 lands, the
  six kernel flags are accepted but mathematically a no-op against
  replicated storage.

**Reviewer pool:** Python reviewer with LoRA familiarity. No GPU, no
kernel knowledge required. Sidecar plumbing additionally touches
`LoraManager` and `ModelEngine`, which may pull a model-loading reviewer.

**CI footprint:** Python unit tests only. Fast.

**Recipe:**

```bash
git fetch origin main
git checkout -b user/brb/moe-lora-helpers origin/main

# Commit 1: helpers (subset of 4407871ce4).
git cherry-pick -n 4407871ce4
KEEP='^(tensorrt_llm/_torch/peft/lora/(moe_layout|validation)\.py'
KEEP+="|tests/unittest/_torch/lora/test_moe_(layout|lora_validator)\.py)$"
git diff --staged --name-only | grep -vE "$KEEP" \
  | xargs --no-run-if-empty git restore --staged --worktree --
git commit -s -m "[TRTLLM-12507][feat] add MoE LoRA layout and validation helpers" \
  -m "<body>"

# Commit 2: sidecar (6e9b10de96 minus both doc edits).
git cherry-pick 6e9b10de96
git rm -f docs/source/_dev_notes/moe-lora-preflight.md
git checkout origin/main -- docs/source/features/lora.md
git -c core.editor=true cherry-pick --continue
```

## MR #1.2 — `[TRTLLM-12507][feat] add routed-expert LoRA support to fused MoE (eager)`

**Source:** the rest of the eager-mode work, stacked on MR #1.1.

**Scope:**
- Rest of `4407871ce4` after MR #1.1's carve-out: `cpp/.../moeOp.cpp`,
  `_torch/custom_ops/torch_custom_ops.py`,
  `_torch/modules/fused_moe/{create_moe,fused_moe_cutlass}.py`,
  `_torch/models/modeling_qwen_moe.py`,
  `_torch/peft/lora/{layer,cuda_graph_lora_params}.py`,
  `tests/unittest/_torch/lora/{test_moe_lora_op,test_lora}.py`,
  `docs/source/features/lora.md`,
  `docs/source/_dev_notes/moe-lora-preflight.md`.
- `8a9a3746e6` — native shared-outer flag in CUTLASS MoE.
- `3c9060e2dd` — `reference_swiglu_moe_lora` helper + 10
  reference/bisection tests.

**Reviewer pool:** kernel reviewer + PyTorch backend reviewer.

**CI footprint:** full GPU CI.

**Bundling rationale (regression suite stays here):** `3c9060e2dd` is the
correctness justification for the kernel feature. Splitting it into a
separate MR would make MR #1.2 reviewers take the kernel correctness
on faith from same-kernel tests alone. Keeping it bundled gives reviewers
"the kernel is correct because here is the fp32 reference suite that
agrees with it" inside one PR boundary.

**Cherry-pick recipe:**

```bash
git checkout -b user/brb/moe-lora-feature user/brb/moe-lora-helpers
git cherry-pick 4407871ce4   # helper files already in HEAD; produces a reduced diff
git cherry-pick 8a9a3746e6
git cherry-pick 3c9060e2dd   # may conflict on test_moe_lora_op.py; resolve manually
```

Expected conflicts at `git cherry-pick 3c9060e2dd` are both in
`tests/unittest/_torch/lora/test_moe_lora_op.py`:

1. **`@pytest.mark.skip` reason** on
   `test_moe_lora_slot_indexed_cuda_graph_replay_matches_eager`.
   Counter-intuitively, **take HEAD's wording** (the `4407871ce4` short
   reason about the Phase 6 kernel patch). The incoming `3c9060e2dd`
   wording references a `moe_device_lora_env` fixture defined by the
   device-path commits (`83a7e5210c`..`be86076e81`), which are not in
   MR #1.2. Picking `3c9060e2dd`'s wording (with its function signature
   `(moe_device_lora_env)`) would leave the test referencing an
   undefined fixture.

2. **Trailing block** that adds `test_moe_lora_device_path_matches_host_path`
   plus the `_make_per_expert_lora_scaled` helper and the 10 reference /
   bisection tests. Drop the device-path test in its entirety -- it
   reads `TLLM_MOE_LORA_USE_DEVICE_PATH`, which is wired up by the
   device-path commits, not MR #1.2. Keep everything from
   `_LORA_REFERENCE_SCALE` onward. Tidy up the surrounding section
   comment that originally referenced the dropped device-vs-host
   comparison.

## What stays only on `user/brb/moe-multi-lora`

The following commits are **not** cherry-picked into any MR branch. They
live on `user/brb/moe-multi-lora` for archival and can be reached via
`git cherry-pick <sha>` for any future debugging:

- `40b3a19ddc` — workaround investigation (force-Ampere GEMM2,
  pre-sum + FINALIZE, two C++ workarounds).
- `41d09cbf03` — revert of the investigation.
- `0ca7be555a` — Phase 6b.E findings docs (refuted by the regression
  suite; do not ship).
- `ff249e3732` — eager-mode numerical caveat docs (refuted by the
  regression suite; do not ship).

## Sequencing and merge order

1. Open MR #1.1 against `main`. Python-only, fast review.
2. Open MR #1.2 stacked on MR #1.1 (target = `user/brb/moe-lora-helpers`
   on the fork until MR #1.1 lands; re-target to `main` afterward).
   Reviewers can begin reading it immediately; merge will block on
   MR #1.1 landing.
3. Land MR #1.1 → MR #1.2.

## Follow-up: MR #2 (CUDA-graph capture)

Out of scope for this plan. Will be a stack of 5 commits
(`83a7e5210c`, `22f348ef5d`, `7412cdf34e`, `63998d4252`, `be86076e81`)
plus an audit of the dev note for any leftover Phase 6b.E references,
opened against `main` after MR #1.2 lands.
