---
name: staircase-modeling-disagg
description: Load when a target must work under disaggregated serving — prefill on one engine, decode on another, with per-sequence state transferred between them. The modeling duty is that EVERY piece of per-sequence state prefill produces is in the transferable set, which for a hybrid model is not just the attention KV pages.
---

# staircase-modeling-disagg

`--features disagg`. Context and generation run as **separate engines**. What
prefill computed has to reach the decode engine over the cache transceiver,
and whatever does not reach it is silently recomputed as zero.

This is a serving topology, so it is tempting to read it as an operations
concern. It is not, and the reason is specific:

> **A single-engine target can keep per-sequence state anywhere and still pass
> every gate.** Under disaggregation, only state the transceiver knows about
> survives the hop.

That makes disagg support a property of the *modeling code*, and the only
feature in this workflow whose defect mode is "correct in every single-engine
measurement, wrong across the hop".

## The one question to answer first

**Enumerate every tensor whose value at the end of prefill affects the first
generated token, and say where each one lives.**

For an ordinary attention-only model the answer is short: the paged KV cache,
and the transceiver moves it. For this run's checkpoint it is not:

| state | produced by | in the KV pool? |
|---|---|---|
| attention K/V pages | the 12 `full_attention` layers | yes |
| GDN conv state (`number_of_conv_states: 3`, kernel 4) | the 36 `linear_attention` layers | **has to be — check** |
| GDN recurrent/SSM state (fp32) | the 36 `linear_attention` layers | **has to be — check** |
| sparse-attention indexer state (`indexer_budget: 2048`) | the 12 attention layers | **has to be — check** |
| MTP draft-layer state | the speculator, if present | ask separately |

The three marked rows are the whole feature. If any of them is held in a module
attribute, a Python-side buffer, or a cache the transceiver does not enumerate,
the target is *not* disagg-capable however green its single-engine gates are.

`tensorrt_llm/_torch/pyexecutor/resource_manager.py` is where a state plane
either is or is not part of the managed pool. Start there, and answer per state
kind rather than per model.

## What the runtime gives you

| surface | what it is |
|---|---|
| `tensorrt_llm/_torch/disaggregation/` | the transceiver: `KvCacheTransceiver`, `create_kv_cache_transceiver`, backends under `native/` and `nixl/` |
| `tests/unittest/_torch/disaggregation/` | the contract tests, including `test_kv_cache_transceiver_contract.py` and a `fake_kv_cache_transceiver.py` worth reading before writing anything |
| `examples/disaggregated/disagg_config.yaml` | the deployment shape: `context_servers` / `generation_servers`, each with its own parallel sizes and `cache_transceiver_config` |
| `docs/source/features/disagg-serving.md` | the feature doc |

Nothing there is model-specific. The transceiver moves what the resource
managers registered; it has no way to discover state you kept elsewhere.

## Device budget

A disagg run needs **at least two engines**, therefore at least two devices,
*even when the target's own parallel segment is `tp1`*. A target declared
`tp1` and a disagg witness needing two GPUs is not a contradiction — the
segment describes one engine.

Say so in the plan. A run that assumed one device and then could not place the
generation server reads as a scheduling failure and is a specification one.

## Preconditions that hang instead of raising

Shared with `staircase-modeling-dwdp`, and worth re-checking here because a
disagg launch that stalls looks exactly like a modeling deadlock:

- the **re-executed interpreter** must be the one the run declares. The disagg
  leader re-execs a literal `python3`; if that resolves elsewhere the workers
  come up without trtllm and the launch stalls with no message;
- the **cache transceiver must be usable**. Pick the backend explicitly rather
  than taking a default, and confirm it in the log rather than assuming;
- the **single-process worker flag** applies to each engine's ranks.

Turning each of these into a named boot failure is the boot gate's job. That is
what a minutes-scale gate is for.

## Gates

boot (aggregated) → accuracy (aggregated) → **equivalence witness
(disaggregated)**, all under `modeling_v2=require`.

The witness is the only measurement that can see the feature, and it is a
*comparison*, not a smoke test:

- same prompts, **greedy** decoding, same seed, same `max_tokens`;
- leg A: one engine, aggregated;
- leg B: context server + generation server, everything else identical;
- a code path the disaggregated engines refuse or guard off (a bounded replay,
  say) is pinned to the same setting on BOTH legs with the block's `legs_env`
  (leg B's launcher gets it as `STAIRCASE_WITNESS_LEGS_ENV` and applies it to
  both engines); otherwise the legs differ in that path as well as in the hop;
- one request at a time on both legs (`max_batch_size: 1` in the witness
  block; leg B gets it as `STAIRCASE_WITNESS_MAX_BATCH_SIZE`): an aggregated
  engine and a disaggregated pair never pack rows alike, and a row's bits
  depend on the rows beside it;
- assert the outputs are **token-identical**.

Not "similar", not "both coherent". Missing recurrent state usually yields text
that is fluent and wrong, and a short benchmark under a 5-point tolerance will
not catch it. Token identity is the property that distinguishes "the state
crossed" from "the state was recomputed from zero and the model recovered".

Run both legs **in the same session**. A cross-session comparison carries
session variance into a judgement that is supposed to be exact.

### Prefix length matters

Use at least one prompt long enough to fill several KV blocks *and* to put the
recurrent state well away from its initial value — a 2048-token prompt on this
checkpoint exercises `indexer_budget` too. A 10-token prompt can be
token-identical with the recurrent state dropped entirely.

## Report additions

- the state enumeration above, filled in, with the file and symbol that owns
  each row
- the transceiver backend actually used, quoted from the log
- the two-leg comparison: prompt lengths, the number of tokens compared, and
  whether identity held
- which of the three hang-preconditions the boot gate now asserts
- device count used by the witness, stated next to the target's `tp1` segment
  so nobody reads them as contradictory
