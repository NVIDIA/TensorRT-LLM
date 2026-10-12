# Findings — what a real onboarding run taught this workflow

Each file here was written while something was going wrong, during the
onboarding of `MiMo-V2.6-Flash-RL` to 4x GB300 (2026-09-22/23). They are kept
because **every one of them presented as a model or target defect while it was
happening**, and none of them was.

They are evidence, not a checklist. A finding names what was measured, what it
cost, and what was changed — so a later reader can tell a fixed problem from a
recurring one.

## Read these two first

- **`RETROSPECTIVE-2026-09-mimo-v2.6-flash.md`** — the whole run reflected on:
  execution efficiency, flow length, inter-agent coordination, and agents'
  own trial-and-error, each argued from measured numbers. Its five
  highest-return changes are all implemented on this branch.
- **`20260922-F002-unified-targets-are-gated-at-tp-only.md`** — the one that
  changed the numbers. A `unified` target was served at pure tensor
  parallelism on every member, so the expert-parallel and attention-DP
  branches were exercised by no gate while the target claimed all three.

## The shape most of them share

Two thirds are **environment failures wearing a model's clothes**: an MPI
launcher that makes every rank a leader and hangs forever with no error, a
`TMPDIR` too long for a unix socket, a supply survey that measured nothing and
printed a check mark, a gate that measured a tree somebody else was editing.

The remaining third are **silent-correctness** problems, which on the
checkpoint that produced them were the dominant hazard: a wrong MXFP4 MoE
kernel does not raise, it serves fluent token soup; a missing input processor
answers text-only questions correctly, having never looked at the image. Most
of the defensive machinery on this branch — the boot gate asking a question
with a checkable answer, the insistence on negative controls, the tree-state
recording — follows from that.

## Numbering

`F001`..`F011` in the order found. A `-fix-report` file beside a finding is
what was actually implemented, which is not always what the finding proposed —
`F011`'s proposal to use `git describe --always --dirty` turned out to take
over 60 seconds on a lustre checkout and was replaced by something that takes
0.14 s.
