# F004 — `source_embargo` cannot name the trees most likely to be copied from

- **Found:** 2026-09-22 04:10 PDT, while choosing the accuracy-anchor engine
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/task_schema.py`,
  `_validate_source_embargo`
- **Severity:** the run's central claim — "this pipeline *derived* the target" — is unenforceable
  for this checkpoint
- **Status:** reported; fix batched with F001/F003

## What happens

`_validate_source_embargo` rejects an absolute path outright:

> `'source_embargo'` entries are repo-relative; got absolute ...

and requires each entry to exist under `trtllm_repo_path`. Both halves are
sound for the case the field was written for: TensorRT-LLM already ships an
implementation of this checkpoint, name it, and the writing roles cannot open
it.

## Why it does not cover this run

TensorRT-LLM ships **no** MiMo implementation — `grep -rl "MiMoV2\|mimo_v2"`
over `tensorrt_llm/`, `tests/` and `examples/` returns nothing. So there is
nothing in-repo to embargo, and `source_embargo: []` is correct.

But two other engines on **this same filesystem** do implement it:

```
/lustre/<fs>/portfolios/<org>/users/<user>/vllm
/lustre/<fs>/portfolios/<org>/users/<user>/sglang
```

and vLLM's is the only implementation of the vision tower available anywhere
here. Those paths are an ordinary `Read` away from every agent in the run,
they are discoverable by a single `grep -rl mimo_v2 /lustre/...`, and a smith
looking for "how does anyone do asymmetric qk/v head dims on Blackwell" is
*exactly* the agent who will find them.

`_common.py`'s `SOURCE_BOUNDARY` does say "Other inference engines' modeling
code stays out entirely". That sentence is a discipline in a prompt. For the
in-repo case the project decided a discipline was not enough and built a
mechanism — `source_embargo_block`'s own docstring says so:

> "read to understand an interface, never to copy a structure" is a line
> nobody can audit from the outside: the target would be correct, and whether
> the pipeline could have produced it would be unknowable.

That reasoning does not depend on which repository the tempting file is in.
It depends on the file being reachable. Here, the reachable ones are the ones
the field cannot name.

## Why it bites harder than usual on this run

The accuracy anchor for this checkpoint has to come from one of those two
engines — nothing else on this filesystem can load a 310B mxfp4 MoE, and the
checkpoint's own HF implementation cannot load its own quantized weights. So
the run will *deliberately start a vLLM or SGLang server*, which puts those
checkouts in the run's own working set. The black-box use is correct and
necessary; the read access that comes with it is incidental and unwanted.

## Proposed change

Widen the field rather than adding a second one:

1. Accept an **absolute path** when it resolves outside `trtllm_repo_path`,
   keeping the existing must-exist check — an embargo naming a path that is
   not there reads as protection while protecting nothing, and that argument
   is path-shape-independent.
2. Keep rejecting an absolute path that resolves *inside* the repo, with a
   message saying to spell it repo-relative. That is the case the current
   error is actually about, and it stays caught.
3. A directory entry embargoes the tree beneath it. Today's entries are
   files; a sibling engine checkout is a tree, and asking a spec to enumerate
   its model files is asking it to go stale.
4. `source_embargo_block` already renders the list into the prompt verbatim,
   so nothing downstream changes. Its wording says "This checkpoint already
   has an implementation in this repository", which becomes false for an
   out-of-repo entry — split that sentence on where the paths point.

## What this does not claim

An embargo in a prompt is still a prompt. The project removed the per-read
`PreToolUse` hook that made it mechanical, and `README.md` records that
decision and what carries the duty instead. This finding is not asking for
the hook back; it is asking that the list a future hook would read be *able
to contain the right paths*.
