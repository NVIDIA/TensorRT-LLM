# F011 — the gates measure the working tree, and nothing owns it

- **Found:** 2026-09-23 02:05 PDT, by the multimodal reviewer, minutes before
  three MMMU accuracy gates would have run
- **Component:** the gate commands generally; `workflow.py`'s unit loop
- **Severity:** a gate can measure code nobody reviewed, and report the number
  as the reviewed unit's
- **Status:** caught and worked around by hand; fix proposed

## What happened

The multimodal unit was assembled (commit `36ea38b940`), reviewed, and
APPROVEd. The orchestrator was about to run its accuracy gate on three members
— roughly 15 minutes each plus engine start.

Meanwhile a **separate agent**, implementing the audio input path, was editing
the same `modeling.py`. It had gone from the committed 2827 lines to 3199.

Nothing would have stopped the gate. `run_accuracy_gate` builds a command
naming `checkpoint_path` and an options file; the *code* it measures is
whatever is on disk under `trtllm_repo_path` when the engine imports it. There
is no pin, no check, and no record of which tree state produced a number.

The reviewer caught it only because it happened to re-read the file's line
count and noticed it had changed underneath its own review.

## Why this is worse than a flaky run

Three separate failures were queued up:

1. **The gate would have measured a half-written vision path.** The in-flight
   edits renamed `_VIT_CALL_CONSTANTS` / `_VIT_MASK_PADDING`, removed the
   vision full-attention call site's arguments, and added an unconditional
   `assert isinstance(audio_cfg, dict)` to `ModelingV2Core.__init__` — which
   would break the *three already-green units* to add a fourth.
2. **A failure would have burned assembler turns.** The unit budget is six;
   three failed members is three turns spent on a defect that does not exist
   in the reviewed commit.
3. **A pass would have been worse.** Had it somehow passed, the run would hold
   an MMMU number attributed to `36ea38b940` that was measured on something
   else. Nothing downstream could tell.

Point 3 is the real defect. This workflow's whole claim is that a target is
trusted because its gates passed — and a gate that cannot say *what it
measured* cannot support that claim.

## Proposed change

1. **Record the tree state in the verdict.** `GateResult` gains the `git`
   description of the repo at gate time — `git rev-parse HEAD` plus a dirty
   flag, or better `git describe --always --dirty`. One subprocess call per
   gate, and every number in `progress.yaml` and the journal becomes
   attributable. This alone would have turned a silent hazard into a visible
   one.
2. **Refuse to gate a dirty tree under the target's own path**, unless the run
   says otherwise. The assembler writes and commits *before* handing back —
   that is already the protocol — so at gate time the target's files should be
   clean. A dirty target path at gate time means someone else is writing, and
   that is worth a hard stop with a message naming the files.
3. **Say the invariant out loud in the role prompts.** Nothing currently tells
   an agent that the repo is shared mutable state during a gate. The commit
   discipline block warns about `git add -A` sweeping in a sibling's
   half-written files; this is the same hazard one level up, and it is not
   mentioned.

Point 2 is the one with teeth. Point 1 is cheap and should land regardless —
even with the guard, a recorded tree state is what lets a number be defended
six months later.

## The operator error underneath it

This was not the workflow's fault alone: **I dispatched the audio agent in
parallel with the multimodal unit's gating, onto the same file.** The
parallelism was deliberate — closing the audio gap while gates ran — and the
collision was foreseeable.

That is worth recording as its own lesson, because the run's standing
instruction is to maximise parallelism, and this is the boundary: **agents may
run in parallel across *different* files, never across the same one while it
is being gated.** The smith phase gets this right by construction — nine
smiths touched nine disjoint entries — and the assembler phase does not,
because there is only one `modeling.py`.

## What was done

The in-flight audio work was saved to
`05-units/audio-wip-mainagent.patch` (652 lines), the target files were
restored to the reviewed commit with `git checkout --`, and the audio agent
was told to stop, save its own patch, and wait for the gates to finish before
resuming. Nothing was lost.
