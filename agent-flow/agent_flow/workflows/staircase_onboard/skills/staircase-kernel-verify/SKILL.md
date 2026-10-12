---
name: staircase-kernel-verify
description: Load when smith must check an existing catalog entry against the values a checkpoint will pass — confirm the receipt's arch still holds and still post-dates every file of the entry, and where the certified column does not cover a value, extend the test, rewrite the contract's not-certified prose from observed runs, and revalidate the receipt.
---

# staircase-kernel-verify

One run, one entry, one surface. The entry already exists; the question is
whether what this checkpoint passes is inside what the receipt certifies.

The spec is shared: read
`.claude/skills/staircase-kernel-onboard/references/entry-spec.md` (the three
pieces, receipt discipline, test recipe) and
`.claude/skills/staircase-kernel-onboard/references/contract-evidence.md`
(what makes a contract sentence defensible). This skill only says what
differs.

## Two outcomes

**Covered.** The contract's certified column already includes these values,
and the receipt's `sm_<arch>` key matches the architecture this run targets.
Change nothing. Report what you checked and against which receipt.

That check is not a formality: receipts are three-state, and **absent means
unknown**. Receipts are keyed by GPU architecture and nothing else — there is
no version pin in tree, because a target moves with the trunk — so the two
things that can make a green receipt a claim about a run nobody made are the
wrong arch and a stale file. **A receipt is valid only if it post-dates the
last write to every file of its entry**, and an entry spans two trees (the
contract and wrapper in the package, the GPU test in the tests tree), so check
both, mechanically, against mtimes. All of that costs a file read and no GPU.

What replaced the old import-time pin assert is not on this entry at all:
each target declares `REQUIRED_TRTLLM_OPS`, and
`tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py`
asserts every name in it exists — which turns an engine drifting under a
target from a wrong answer into a loud failure.

**Not covered.** Extend it:

1. **Probe first where the semantics are undocumented**, then freeze what the
   probes pinned into test cases. Do not write the assertion before you know
   what the op does at this value
2. **Extend the existing
   `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py`**
   with the new coverage. Same rules as a new test: the reference is built on
   the spot from native torch ops, never from the wrapper or another catalog
   entry; randomness is seeded; `torch.testing.assert_close` at dtype-aware
   tolerances, and any loosening carries an inline justification
3. **Rewrite the contract's not-certified annotations for this surface** into
   certified semantics, from observed runs only. This is the part that gets
   skipped and it is the part the next target reads — a receipt nobody can
   interpret is worth little
4. **Rerun the entry's whole test file**, not just the new case, and record
   the receipt from that run. A multi-rank entry records `world_size: <N>`;
   absent means 1
5. Do not format or lint during iteration — the repo's `pre-commit` runs once
   at delivery. Revalidate that the receipt post-dates the last write to every
   file of the entry, in both trees, then commit pathspec-scoped:

```bash
git add tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.md \
        tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py \
        tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml
git commit -m "feat(catalog): certify <name> at <surface>"
```

The wrapper usually stays untouched. If it does not — if covering this surface
needs a signature change — that is worth saying loudly in the report, because
every existing caller is affected.

**The op may simply fail at this value.** That is a valid product: record
`status: failed` with a short note, land it, and report it. It is a real
constraint on every target that would have passed that value, and finding it
here is far cheaper than finding it in an accuracy gate.

## When the surface neighbours a known intermittent defect

`entry-spec.md` states the rule and the case that produced it: prove the
harness can *see* the defect before reporting a clean result. It applies with
full force here — extending coverage next to a known-flaky surface is exactly
where a blind harness returns a confident, meaningless green.

## Report

- entry path, the surface requested, and the verdict: already covered / newly
  certified / failed
- the receipt as it now reads, and what the new coverage pinned
- **what remains uncertified** on this entry — the next caller reads this
- every claim matched against disk before sending
