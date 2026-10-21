---
name: trtllm-modeling-v2-parallel
description: Load when a target runs above world size 1, or when one modeling codebase must cover a family of parallel configurations (a `unified` segment) — what a rank does differently, where the collectives belong, and how a unified target declares and gates its member set.
---

# trtllm-modeling-v2-parallel

Loaded whenever the target is not a plain `tp1`: a multi-rank single
configuration, or a `unified` segment covering a family.

| reference | read it when |
|---|---|
| `references/multi-rank.md` | world size > 1 — sharding, collectives, `lm_head` vocab parallelism, manifest `src` transforms, what attention DP changes on top of TP |
| `references/unified.md` | the segment is `unified` — config folding, the degenerate path, the Coverage section, member selection |

Both, for a unified target at world size > 1, which is the normal case.

## What changes above world size 1

The short version; `references/multi-rank.md` is normative.

- **The manifest gains a `src` transform column.** A rank loads a slice, and
  which slice is a function of the rank. Its ordering rule matters as much as
  its content
- **`lm_head` vocab parallelism is a manifest obligation**, not something the
  runtime does for you
- **Collectives are catalog entries like everything else** and they sit at
  specific points in the forward. Which points is a property of the topology,
  not of the model — the HF reference says nothing about it, which is why the
  pre-audit walks the segment separately
- **Coverage asserts change under EP**: a rank holds a window of experts, so
  "every checkpoint key consumed" becomes "every key in my window consumed,
  and the rest declared off-window"
- **Attention DP replaces several TP conclusions rather than adding to them.**
  Read that part of `multi-rank.md` before assuming a TP fact carries over
- **Every rank must see the same batch composition** when attention DP is off.
  The schedulers stay in step by construction, and a forward that branches on
  anything rank-local breaks that silently

## Unified targets

A `unified` segment is **one modeling codebase over a family of parallel
configurations**, folded at runtime from the `mapping` object.
`references/unified.md` has the folding, the degenerate path, member
selection and gating. The segment name promises structure, not coverage: which
members work is the run journal's **Coverage** section (below).

A member is not a config file. Topology comes from the routing decision and
from the gate's own arguments: `--tp_size` / `--moe_ep_size` /
`--enable_attention_dp` on the boot gate
(`examples/llm-api/quickstart_advanced.py`), and the same three as LLM-API
arguments — `tensor_parallel_size`, `moe_expert_parallel_size`,
`enable_attention_dp` — in the accuracy case. Which member is default is the
first row of Coverage. Every gate run of every member passes
`modeling_v2=require`; under `auto` a member that
missed the target's criteria would quietly be the built-in implementation, and
its row would be a record of the wrong system.

<!-- PORT: verify — no `unified` target has migrated in tree (only
`models/gpt_oss/.../tp1` and `models/deepseek_v3/.../dep4` exist), so how the
`<ckpt>__<arch>__<par>` directory name and `routing.py` name a family
rather than one member is not established here. -->

## Coverage section

```markdown
## Coverage

One modeling codebase, over the configurations below. Each member carries its
own gate records.

| member | topology | boot | accuracy | note |
|---|---|---|---|---|
| tep4 | `--tp_size 4 --moe_ep_size 4`                          | PASS | GSM8K 63.91 | default member |
| dep4 | `--tp_size 4 --moe_ep_size 4 --enable_attention_dp`     | PASS | GSM8K 63.38 | |
| tp1  | —                                                      | —    | —           | excluded: ~420 GiB, does not fit one card |
```
