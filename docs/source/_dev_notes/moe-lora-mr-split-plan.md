# MoE LoRA review-split plan

This is a coordination note for splitting the work on
`user/brb/moe-multi-lora` into reviewable upstream MRs. It is intentionally
checked into `_dev_notes` rather than user-facing docs because it is purely
PR-process scaffolding; once all referenced MRs are merged it can be
deleted.

## Goals

- Split the eager-mode portion of `user/brb/moe-multi-lora` into 3 MRs.
- Maximize reviewer parallelism: at least two of the three MRs reviewable
  by different reviewer pools simultaneously.
- Keep CI tax bounded: only one MR should pull a full GPU-test cycle; the
  others should land Python-unit-only changes.
- Defer all CUDA-graph capture work to a follow-up MR series ("MR #2"),
  not covered by this plan.

## Branch graph

```
origin/main
├── user/brb/moe-lora-helpers   ← MR #1.1: layout + validation helpers (Python only)
│   └── user/brb/moe-lora-feature
│                                ← MR #1.3: kernel + integration + reference suite
└── user/brb/moe-lora-sidecar   ← MR #1.2: lora_layout.json sidecar (Python only)
```

`user/brb/moe-multi-lora` (this branch) is the development trunk that
keeps the entire history including the investigation pair and the
post-mortem regression suite. None of the MR branches reference this
branch directly.

## MR #1.1 — `[TRTLLM-12507][feat] add MoE LoRA layout and validation helpers`

**Source:** subset of `4407871ce4` (the routed-expert LoRA core).

**Scope (4 files):**
- `tensorrt_llm/_torch/peft/lora/moe_layout.py` (new)
- `tensorrt_llm/_torch/peft/lora/validation.py` (new)
- `tests/unittest/_torch/lora/test_moe_layout.py` (new)
- `tests/unittest/_torch/lora/test_moe_lora_validator.py` (new)

**Self-containment:**
- `moe_layout.py` imports only `torch` + stdlib.
- `validation.py` imports only `tensorrt_llm.lora_helper.LoraConfig` +
  stdlib.
- Both unit-test files import only the helper they exercise + pytest.

**Reviewer pool:** Python reviewer with LoRA familiarity. No GPU, no
kernel knowledge needed.

**CI footprint:** Python unit tests only. Fast.

**Cherry-pick recipe:**

```bash
git fetch origin main
git checkout -b user/brb/moe-lora-helpers origin/main
git cherry-pick -n 4407871ce4
# Restore everything except the 4 helper files:
KEEP='^(tensorrt_llm/_torch/peft/lora/(moe_layout|validation)\.py'
KEEP+="|tests/unittest/_torch/lora/test_moe_(layout|lora_validator)\.py)$"
git diff --staged --name-only | grep -vE "$KEEP" \
  | xargs --no-run-if-empty git restore --staged --worktree --
git status   # should show only the 4 kept files staged
git commit -s -m "[TRTLLM-12507][feat] add MoE LoRA layout and validation helpers" \
  -m "<body>"
```

## MR #1.2 — `[TRTLLM-12507][feat] add lora_layout.json sidecar for shared-outer MoE LoRA`

**Source:** `6e9b10de96` cherry-picked verbatim, with the dev-note edit
(40 lines in `docs/source/_dev_notes/moe-lora-preflight.md`) dropped if
the file does not yet exist on `origin/main`. The dev-note bring-up is
owned by MR #1.3 (which adds the file via `4407871ce4`), so the sidecar
landing without those 40 lines is intentional.

**Scope:**
- `tensorrt_llm/lora_layout_sidecar.py` (new)
- `tensorrt_llm/lora_manager.py` (sidecar discovery + plumbing)
- `tensorrt_llm/_torch/pyexecutor/model_engine.py` (engine wiring)
- `tests/unittest/_torch/lora/test_lora_layout_sidecar.py` (new)
- `tests/unittest/others/test_lora_manager.py` (LoraManager sidecar tests)
- `docs/source/features/lora.md` (~21-line user-facing doc addition)

**Self-containment:** the sidecar lives at the package root
(`tensorrt_llm/`), not in `_torch/peft/lora/`, so it does not import
anything from MR #1.1. It can be reviewed before, in parallel with, or
after MR #1.1.

**Caveat:** the sidecar's *runtime* usefulness (the kernel actually
honoring the shared-outer flag) requires `8a9a3746e6` from MR #1.3.
Until that lands the sidecar is metadata only, but its parsing and
validation are fully self-contained and unit-tested in this MR.

**Reviewer pool:** model-loading / `LoraManager` owner. Different person
from the MR #1.1 reviewer in practice.

**CI footprint:** Python unit tests only. Fast.

**Cherry-pick recipe:**

```bash
git checkout -b user/brb/moe-lora-sidecar origin/main
git cherry-pick 6e9b10de96
# If the dev-note edit conflicts (file doesn't exist on main):
git restore --staged --worktree -- docs/source/_dev_notes/moe-lora-preflight.md
git -c core.editor=true cherry-pick --continue
```

## MR #1.3 — `[TRTLLM-12507][feat] add routed-expert LoRA support to fused MoE (eager)`

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
separate MR #1.4 would make MR #1.3 reviewers take the kernel correctness
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

Expected conflict at `git cherry-pick 3c9060e2dd`: small (~21 lines) edit
to the `@pytest.mark.skip` reason on
`test_moe_lora_slot_indexed_cuda_graph_replay_matches_eager`. Take
`3c9060e2dd`'s wording — it is the post-investigation version that
matches the new reference suite.

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

1. Open MR #1.1 and MR #1.2 simultaneously. They can be reviewed by
   different reviewers in parallel.
2. Open MR #1.3 stacked on MR #1.1. Reviewers can begin reading it
   immediately; merge will block on MR #1.1 landing.
3. Land in any of these orders: MR #1.1 → MR #1.3 (sequential due to
   stacking), MR #1.2 (anywhere). MR #1.2 can land before, between, or
   after the others.

## Follow-up: MR #2 (CUDA-graph capture)

Out of scope for this plan. Will be a stack of 5 commits
(`83a7e5210c`, `22f348ef5d`, `7412cdf34e`, `63998d4252`, `be86076e81`)
plus an audit of the dev note for any leftover Phase 6b.E references,
opened against `main` after MR #1.3 lands.
