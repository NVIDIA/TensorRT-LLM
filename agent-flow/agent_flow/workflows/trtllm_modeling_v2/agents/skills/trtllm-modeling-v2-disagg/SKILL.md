---
name: trtllm-modeling-v2-disagg
description: Load when a target must work under disaggregated serving — prefill on one engine, decode on another, with per-sequence state transferred between them. The modeling duty is that EVERY piece of per-sequence state prefill produces is in the transferable set, which for a hybrid model is not just the attention KV pages.
---

# trtllm-modeling-v2-disagg

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
and the transceiver moves it. For a hybrid checkpoint it is not:

| state | produced by | in the KV pool? |
|---|---|---|
| attention K/V pages | the full-attention layers | yes |
| conv window state | every recurrent (linear-attention / SSM) layer | **has to be — check** |
| recurrent / SSM state | every recurrent layer | **has to be — check** |
| sparse-attention indexer state | the attention layers, if the model has an indexer | **has to be — check** |
| draft-layer state | the speculator, if present | ask separately |

Read the counts, the conv kernel width and the state dtypes off the
checkpoint's `config.json`. The three marked rows are the whole feature. If any of them is held in a module
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

Worth re-checking, because a disagg launch that stalls looks exactly like a
modeling deadlock:

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
- leg B: a context engine and a generation engine, everything else identical.
  The flow provides it (`checks/harness/disagg_leg_b.py`): leg A's engine,
  built by the tree's own `quickstart_advanced.py` from leg A's argv, twice,
  on two halves of the node, each prompt sent context-only and then
  generation-only with the `DisaggregatedParams` that came back. Write no
  launcher; the witness block needs no key at all. Declare
  `cache_transceiver_config` (default `{backend: DEFAULT,
  transceiver_runtime: auto}`) when the model needs another transceiver, and
  confirm in the log which one each engine brought up. A run that names its
  own `contrast_command` (and `contrast_env`) takes over leg B and owes
  everything below itself;
- a code path the disaggregated engines refuse or guard off (a bounded replay,
  say) is pinned to the same setting on BOTH legs with the block's `legs_env`
  (leg B's launcher gets it as `MV2FLOW_WITNESS_LEGS_ENV` and applies it to
  both engines); otherwise the legs differ in that path as well as in the hop;
- one request at a time on both legs (`max_batch_size: 1` in the witness
  block; leg B gets it as `MV2FLOW_WITNESS_MAX_BATCH_SIZE`): an aggregated
  engine and a disaggregated pair never pack rows alike, and a row's bits
  depend on the rows beside it;
- both legs print, per continuation, quickstart's `[i] Prompt: ...,
  Generated text: ...` line and, under `TRTLLM_WITNESS_FULL_OUTPUT=1`,
  `[i] Witness token ids: [...]` and `[i] Witness text: '...'` after it --
  leg A from the tree's `examples/llm-api/quickstart_advanced.py`, leg B from
  its launcher. Text that agrees without token ids is not a pass;
- assert the outputs are **token-identical**;
- then prove the comparison could have failed: leg B runs once more on the
  longest prompt with `MV2FLOW_WITNESS_ABLATE=state`, destroying on the
  generation engine the state each request received (every layer's
  recurrent state, else its KV pages) before the first decode step, and
  printing `Witness ablated: <what>`. That continuation must DIFFER from
  leg A's; identical, the verdict is `INSENSITIVE, not a pass`. The flow's
  leg B does this where the executor takes a received request up
  (`PyExecutor._prepare_disagg_gen_transmission_complete`), through the
  cache managers' `get_conv_states` / `get_ssm_states` at the request's slot;
  a target that keeps its recurrent state anywhere those do not reach gets
  `Witness ablation refused: ...` -- the same place the transceiver cannot
  reach, which is the defect this skill is about;
- at a `dep` member, pin every hop: leg B gets `MV2FLOW_WITNESS_RANK_PAIRS`
  (`0:0,1:1,0:1,1:0` at dep2, one pair per prompt), pins prompt `i`'s
  context-only and generation-only requests to the `i`-th pair with
  `SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`, and
  prints `[i] Witness ranks: ctx=<c> gen=<g>`. Unpinned, the router breaks
  ties by rank and every hop goes from rank 0 to rank 0, so a defect on any
  other rank never reaches the comparison.

Not "similar", not "both coherent". Missing recurrent state usually yields text
that is fluent and wrong, and a short benchmark under a 5-point tolerance will
not catch it. Token identity is the property that distinguishes "the state
crossed" from "the state was recomputed from zero and the model recovered".

Run both legs **in the same session**. A cross-session comparison carries
session variance into a judgement that is supposed to be exact.

### Prefix length matters

Use at least one prompt long enough to fill several KV blocks *and* to put the
recurrent state well away from its initial value, and past any sparse-attention
indexer's budget when the model has one. A 10-token prompt can be
token-identical with the recurrent state dropped entirely.

So can a long prompt the model continues without reading it. With the
recurrent state and the conv window zeroed after the hop, `one two ... ten`
repeated to 131, 2050 and 4097 tokens stayed token-identical: the attention
pages alone continue a pattern. Prose diverged within a dozen tokens. The
witness generates its `prompt_tokens` prompts as prose for that reason; a
hand-written long prompt must be text the model has to read (not repeated,
not memorized). The self-test above checks the longest prompt of the
workload, so make that one count.

## Report additions

- the state enumeration above, filled in, with the file and symbol that owns
  each row
- the transceiver backend actually used, quoted from each engine's log
- the two-leg comparison: prompt lengths, the number of tokens compared, and
  whether identity held
- the self-test: what leg B destroyed and the token at which that
  continuation first differed; at a `dep` member, the rank pairs crossed
- which of the three hang-preconditions the boot gate now asserts
- device count used by the witness, stated next to the target's `tp1` segment
  so nobody reads them as contradictory
