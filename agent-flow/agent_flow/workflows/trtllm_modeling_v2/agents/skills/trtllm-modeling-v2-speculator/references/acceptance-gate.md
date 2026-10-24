# The acceptance gate

The third gate for speculative decoding. It exists because the first two are
structurally incapable of seeing a defect in the draft path.

## Why the boot and accuracy gates are blind

Rejection sampling holds the emitted distribution to the **target** model's,
whatever the draft proposes. So a draft module that computes the wrong thing
does not produce wrong text — it produces the same text, with every draft
rejected, more slowly.

| gate | what a broken draft does to it |
|---|---|
| boot | passes: the continuations are the target model's |
| accuracy | passes: the distribution is the target model's, to the last decimal |
| throughput | drops — but you have nothing to compare it against |

The failure mode is not "it crashes" or "it scores badly". It is **it works,
and the feature is worthless**, which reads as success.

The same reasoning applies to any future feature whose output is
distribution-preserving by construction. That is the rule to check a new
feature against, not the specific mechanism. A feature earns a third gate by
exactly one property: a *miscomputed* implementation still produces a correct
answer.

## The measurement

`acceptance_length` — mean accepted tokens per engine step, including the
always-accepted trunk token. A run with `max_draft_len = k` has a ceiling of
`k + 1`.

In tree it is not a separate harness. The gate is a case in the accuracy
suite, beside the model family's other gates:

```python
with LLM(
    MODEL_PATH,
    speculative_config=...,
    kv_cache_config=...,
    cuda_graph_config=CudaGraphConfig(),
    enable_iter_perf_stats=True,
    **TOPOLOGY,
) as llm:
    task.evaluate(llm)  # the same workload
    al = compute_acceptance_length(llm)  # accuracy_core.py
    assert_acceptance_length(ACCEPTANCE_KEY, al)  # references/acceptance_length.yaml
```

`compute_acceptance_length` reads `llm.get_stats()` and folds the iterations
that carried draft tokens:
`(numAcceptedTokens + numRequestsWithDraftTokens) / numRequestsWithDraftTokens`
— which needs `enable_iter_perf_stats=True` on the LLM.
`assert_acceptance_length` checks the observed value against `min_al` in
`tests/integration/defs/accuracy/references/acceptance_length.yaml`.

**The two legs are one case parametrized over the mode.** The whole comparison
is `modeling_v2=require` against `=off` on the same checkpoint, same
workload, same `speculative_config` — one variable, not two harnesses. Because
a multi-rank job's mode is fixed when its ranks start, each leg asserts the
environment it was given rather than choosing one, and skips otherwise.

Both legs read the **same** recorded minimum, and that is what makes the pair
informative:

| outcome | reading |
|---|---|
| modeling_v2 fails, stock passes | the draft path regressed |
| both fail | the anchor is stale — re-derive it, do not blame the target |

**The anchor is populated from the stock leg.** Populating it from the
target's own number would make the gate self-referential.

Reading a single number alone tells you almost nothing:

| value | meaning |
|---|---|
| `1.0` | total rejection — the draft path is wrong, or off |
| clearly above 1.0, well below the reference | **subtly wrong** — the dangerous case |
| at the reference | the draft path computes what the checkpoint says |

### Absence of measurement is not a low number

A `None`, a zero-iteration result or an empty stats list is **absence of
measurement**, never a failing gate; do not read it as 1.0.

The trap: for a **one-model** mode such as `MTP_EAGLE_ONE_MODEL`, the engine's
stats aggregation reads each request's `py_draft_tokens`, which the one-model
path resets to `[0] * max_total_draft_tokens` before the forward while the
real ids stay in device tensors. A harness that counts from it reports zero
drafted requests on the stock model and the target alike. Run the **stock**
leg first: nothing reported on *both* is upstream's instrumentation, nothing
on ours alone is a defect in the target.

### A short answer is not a measurement either

The orchestrator's gate drives `quickstart_advanced.py` on the task's
`accuracy.acceptance.prompts` (or `prompts_file`), falling back to the boot
prompts. An answer of two tokens and EOS is a context step plus one generation
step, and neither verifies a draft: the rate reads 1.00 for any drafter. The
gate therefore fails a workload with fewer than `min_draft_iterations`
generation iterations as **"workload too short to measure acceptance"**, and
every verdict lists the per-prompt rates beside the weighted value. That
verdict is about the workload, not the target: it is fixed in `task.yaml`,
never in the draft path.

## The reference is the whole gate

An acceptance number is only interpretable against **the same checkpoint under
stock in-tree modeling, at the same workload, with the same
`speculative_config`**. Four things must match, and the last two are the ones
that get dropped:

1. same checkpoint
2. stock TensorRT-LLM's own modeling, not ours — i.e. `modeling_v2=off`
3. **same workload** — both legs are the same case running the same evaluation
   task with the same protocol, so both systems see identical requests. Do not
   compare across workloads
4. **same draft length.** A reference at a different `max_draft_len` answers a
   different question

The stock leg's own requirements are assembler's to establish, because the
gate is assembler's to pass. They are typically whatever stock trtllm needs in
order to boot this checkpoint at all — boot knobs, which change batching
capacity rather than what the model predicts. **Acceptance is a property of the
model**, so a reference forced onto different memory knobs is still the right
comparison; only the workload has to match.

A reference may also need something no argument covers — an environment
variable that steers a transport, for instance. The in-tree instance: the stock
leg of the dep4 MTP gate requires `TRTLLM_CAN_USE_DEEP_EP=0` exported,
because under attention DP + EP stock lands on DeepEPLowLatency, whose dispatch
takes only NVFP4 uint8 hidden states while the MTP layer is bf16; disabling it
lands stock on AllGatherReduceScatter — the strategy the modeling_v2 target
implements by hand, which makes the two *more* comparable rather than less. If
a reference needs such a variable, it goes in the case that runs it, in full,
because nothing else will carry it, and it is exported in the launching
environment like the mode switch.

## What a passing gate looked like

Recorded for deepseek-r1-0528-nvfp4 / sm_103 / dep4, `mtp3`, on 4 GB300s:
`acceptance_length` **3.3514** against stock's **3.2752**, ratio **1.023**.
That row is the one that matters for a speculative variant — boot and accuracy
were both green on the same target and are blind to the draft path.

<!-- PORT: verify — `references/acceptance_length.yaml`'s
`ModelingV2DeepseekR10528Nvfp4Sm103Dep4::mtp3` entry records a different pair
for what looks like the same gate (stock `ref_al` 3.020, hand-set `min_al` 2.5,
with modeling_v2 measured 2.921-2.938 over three runs). Which workload each
pair was taken on, and therefore which belongs in this file, was not resolved
in the port. -->

Draft lengths 1 and 2 remain ungated on that target — they are the dominated
end of the measured draft-length axis.

## Cost, and the free proxy

Acceptance instrumentation is not free: measured 1.4% / 3.5% / 4.0% of
throughput at concurrency 1 / 32 / 256 (deepseek-r1-0528-nvfp4 `dep4` on B200 —
the instrumentation is upstream's, so re-measure the cost when it changes),
which also makes a label **non-comparable** to one measured without it. Never
mix the two in a figure. In tree the same warning applies to
`enable_iter_perf_stats=True`, which the gate turns on.

For a sweep — and whenever the instrumented number is unavailable — use the
proxy instead: a response body already carries
`mean_avg_decoded_tokens_per_iter`, agreeing with the engine's acceptance
length to 0.99–1.02x. Reserve the instrumented measurement for the gate itself.
The sweep half of this is the later perf workflow's, not onboarding's.

## What the gate does not tell you

That the feature is *worth it*. Acceptance says the draft path computes the
right thing; whether drafting pays depends on the break-even rate, which is
the extra HBM traffic per step divided by the traffic of a step. That is a
performance question and it belongs to `/tuning`.

It also flatters easily: a random-prompt harness gives the model its own
repetitive continuation to draft, which drafts near-perfectly. When quoting a
speedup, quote the one from the real-text gate runs too, and say which is
which.
