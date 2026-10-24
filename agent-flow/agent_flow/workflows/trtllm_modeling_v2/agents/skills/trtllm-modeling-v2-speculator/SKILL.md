---
name: trtllm-modeling-v2-speculator
description: Load when adding speculative decoding to a target — MTP, Eagle3, DFlash, DSpark and the other one-engine drafting modes. It changes the forward and the weight-loading path together, and it needs a third gate, because rejection sampling makes a miscomputed draft produce correct text.
---

# trtllm-modeling-v2-speculator

Covers every `--features` value that makes the target draft its own tokens:
`mtp`, `eagle3`, `dflash`, `dspark`, and the rest of the one-engine family. One
skill because they share a binding surface and a gate; they differ in what the
draft path computes, which is `references/algorithms.md`.

**A checkpoint's `mtp.*` namespace does not mean MTP.** A checkpoint may store
a DSpark block drafter there, with `num_nextn_predict_layers` beside it.
Read what the stages compute before choosing the feature — the DSpark section
of `references/algorithms.md` says how to tell — because `mtp` over DSpark
weights either fails to load or drafts garbage that verification quietly
rejects, and the second case passes accuracy. The schema refuses `mtp` and
`dspark` in one spec for this reason.

| reference | read it when |
|---|---|
| `references/runtime-boundary.md` | always — what the runtime owns of the drafting loop, what the target owes, and where the two meet |
| `references/algorithms.md` | choosing or implementing a specific mode |
| `references/multi-token-prediction.md` | the mode is MTP — mechanism, the concat-order trap, checkpoint layout, HBM arithmetic |
| `references/acceptance-gate.md` | **always, before claiming the feature works** |

## What this feature changes

Unlike most features it touches **two** paths, and forgetting the second is
the common failure:

1. **the forward** — the draft module runs after the trunk, `max_draft_len`
   times, and its output feeds verification
2. **weight loading** — the draft module's weights stop being a
   known-not-loaded key set and start being loaded. On a DeepSeek-family
   checkpoint that is hundreds of keys and several GiB per rank

It also changes what the engine allocates: the KV pool grows by the draft
module's layers. That is why the variant's arguments often carry a second key
that is not a tuning knob — see below.

## Attachment

The draft path is written **flat into `modeling.py`**, not into an adapter.
There is no runtime-owned service here: drafting is a forward-shaped change,
and the runtime's part (the drafting loop, rejection sampling) already exists
and needs nothing registered.

Concretely, per `references/runtime-boundary.md`: the shell gains the binding
the runtime looks for, the core gains the draft container, **each of the two
targets gains its own `mtp_forward`** making the draft ops' calls with every
argument at the call site, and the draft layer's `forward` is one routing line —
`core._select_target(md).mtp_forward(...)`, read per call. The no-feature
path must stay **bit-identical** — when the feature is off, both targets'
`forward` are exactly what the recorded gate measured.

Do not inherit the upstream speculative-decoding base class. It reaches the
draft model through a module-level function with no override point, which
would construct the engine's own in-tree draft module — the thing this project
exists not to use.

## The arguments assembler owns

The variant is activated through the LLM API, not through a config file:
`speculative_config=MTPDecodingConfig(max_draft_len=N)` for MTP,
`DSparkDecodingConfig(max_draft_len=<block size>)` for DSpark, and the
corresponding config for the other modes. It is not a tuning variant: it
selects a second forward path *and* a second weight-loading path. Record it in
the run journal and in the accuracy case that gates it, so the two agree.

It commonly carries a **boot requirement** alongside the activation keys. The
draft module's weights plus the enlarged KV pool can push the engine past what
the default memory fraction leaves, and the failure is not an OOM exception —
measured on one target: three of four ranks spinning in `cudaFree` inside the
caching allocator while the fourth waits at an MPI barrier, no error, no
progress. The dep4 MTP variant carries
`kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75)` for exactly
that reason. Discovering it is part of gating this feature; recording it, with
the evidence, is part of the deliverable. **tuner may not change either key.**

`max_draft_len` is spelled out on purpose. Left unset it resolves to 1, not to
anything workload-derived — and it is a **certification boundary before it is
a knob**: it sets the attention entries' `predicted_tokens_per_seq`, so a
value past what those entries certify needs `trtllm-modeling-v2-kernel-verify` first,
not a bigger number.

## Gates

boot with the variant's arguments → accuracy → **acceptance**, all three with
`modeling_v2=require` — except the stock leg of the third, which asks for
`off` on purpose.

The third one is not optional and not a performance measurement.
`references/acceptance-gate.md` has the protocol; the reason is this:

> Rejection sampling holds the emitted distribution to the target model's. A
> *miscomputed* draft path therefore produces **correct text**, just more
> slowly — every draft rejected. The boot gate passes. Accuracy passes. Only
> speed moves.

So the first two gates are structurally blind here, and `acceptance_length`
against a stock reference is the only detector. `1.0` means total rejection.
Clearly above 1.0 but well below the reference means subtly wrong — which is
the case that looks like success.

The reference is the same checkpoint under **stock in-tree modeling** at the
same workload and the same `speculative_config` — that is
`modeling_v2=off` against `=require`, **one variable rather than two
harnesses**.

Accuracy still runs, and it must be compared against an identity run **in the
same session**: one target measured 94.7688 and 95.0720 on a bit-identical
forward a day apart.

## Report additions

- the mode, and `max_draft_len` with the certification bound that permits it
- the acceptance number, the reference it was measured against, and the
  workload both saw
- the boot requirement, if one was found, with the evidence that established it
- confirmation that the no-feature path is unchanged, and how that was checked
