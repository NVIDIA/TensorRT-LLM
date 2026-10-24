# Modeling iterations: the mechanics

Everything a tuner turn needs to go from a hypothesis to a `CANDIDATE` decision. The request
names: the target dir, the worktree, the dev spool, the env script, the campaign workdir and the
current best measurements. Paths below use those names.

## The modeling_v2 contract you are optimizing inside

Read `tensorrt_llm/_torch/_experimental/modeling_v2/README.md` and `catalog/_op.py` first.

- An op entry is a stateless `OpWrapper` subclass: `__call__` (reads nothing off `self`),
  `reference`, `is_valid`, `CELLS`, `ARCHS`, `note`, and a module-level instance. The tolerance band
  lives in the entry's GPU test.
- Ops one target calls live in that target's `ops/<category>/`; ops two targets call live in
  `catalog/<category>/`. Promote only when a second caller exists -- except a thin entry over an
  installed FlashInfer API (tier 3): it goes straight to `catalog/<category>/`, shape-generic (shapes
  come from the cells), so the next target reaches it with one import (e.g.
  `catalog/ssm/gated_delta_rule_decode_pretranspose.py`, `catalog/gemm/low_m_bf16_mm.py`). No version
  pin in the entry; the version raced against lives in `kernel_choices.json`.
- The forward passes every argument at the call site (per-layer weights as rows of the target's
  `_LayerWeights`). `PrefillTarget` handles any batch with context rows; `DecodeTarget` is
  generation-only and never reads `num_contexts`/`num_ctx_tokens`.
- CPU-only claims tests (`tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py`) enforce: no
  `self.` in `__call__`, every entry with cells has a test naming `.CELLS`, targets import only the
  catalog and their own files, `route()` imports each target on one line. Run them after any edit.
- Weight preprocessing (concatenating projections, re-laying out scales) belongs in the target's
  `weights.py` / `post_load_weights`, done once at load, never per step.

## Shapes worth certifying for a perf entry

A perf entry is certified at the shapes the forward reaches *under the campaign's config*:
decode M in `1..max_con` (at least 1, a mid value, max_con, and whatever padding rounds to), the
prefill chunk up to `max_num_tokens`, and the real checkpoint widths. A cell per shape class, not
per integer. Note the integer-width limits the entry crosses.

## Fusion patterns that recur

| pattern | what is saved | how |
|---|---|---|
| projection concat | N GEMM launches -> 1; better small-M efficiency | concatenate weights at load (`weights.py`), one GEMM, split views downstream (check every consumer accepts the strides, or fuse the split into the consumer) |
| norm into producer/consumer | an HBM round-trip of the activation | the residual-add / injection that produces the stream also normalizes it (one Triton row kernel writing both outputs) |
| GEMM epilogue | the `[M, N]` intermediate write + read and its elementwise launches | Triton GEMM with the activation / gate / reduction in the epilogue; at decode M the GEMM is weight-bandwidth bound, so the epilogue is free |
| router + top-k into the MoE runner | the logits round-trip and a launch | pass `routing_logits` and the routing method the runner implements (check tie-breaking and renormalization semantics against the current op's reference) |
| shared expert on a side stream | serialized independent work | fork on an aux stream with events; capture-safe; measure at con 1 where it matters |
| quantize in the producer | a separate quantize kernel and its read | the kernel producing the activation writes FP4/FP8 + scales directly |

## Ordering

Move work that does not depend on the previous op later or earlier to overlap with it (side
streams), hoist per-step invariants out of the layer loop (computed once per forward), and remove
copies that only make a tensor contiguous when the consumer accepts strides. Prove independence
from the data flow, not from the stream labels.

Benchmark an overlap at the scope it competes in: a whole decoder layer under CUDA-graph replay,
with every other user of the side stream (a MoE branch, a shared expert) in it. A block-only bench
won -5.1% and -2.7% for two side-stream GEMMs that moved e2e by +0.47% and -0.3%. The work moved to
the side stream should use few, wide CTAs (cuBLAS tiles); a GEMV that launches thousands of small CTAs
fills every SM and delays the latency-bound chain it was meant to run beside.

## Worktree, GPU jobs and tests

You work in the worktree you were given (`<wt>`), on its `tune/<slug>` branch. GPU work goes
through the dev spool -- never a blocking submit (a tool call longer than ~10 minutes aborts), so
launch, then poll:

```bash
S=<dev spool>; PKG=<trtllm_modeling_v2 package dir>
# queue: 1 GPU, 30 min, run against the worktree (cwd + first PYTHONPATH entry)
python3 $PKG/infra/substrate/spoolctl.py --spool $S launch --gpus 1 --wall 30m --tree <wt> <name> \
  'source <env script>; cd <wt>/tests/unittest && python -m pytest -q -p no:cacheprovider <test path>'
# poll (returns when done; keep each poll short)
python3 $PKG/infra/substrate/spoolctl.py --spool $S status
cat $S/done/<name>.rc 2>/dev/null; tail -n 40 $S/logs/<name>.log
```

A spool that will not start the job soon (in hand-over, ending before `--wall`, or with no live
watcher) prints a `warning:` naming the dev spools that can take it now; `launch --pick` sends the
job to whichever dev spool can start it soonest and prints which, so poll that one.

Job names: letters, digits, `-`, `_`; unique per request (prefix them with your slug). A result
whose processes imported `tensorrt_llm` from another tree is refused (`mixed_checkout`): that is the
worktree not being first on the path -- always pass `--tree`.

Commit in the worktree with `git -C <wt> commit -s -m "<title>"` (DCO sign-off, no co-author
lines). One hypothesis per branch; several commits are fine.

## Micro-bench

`tune/microbench.py <case file> --out <json>` runs inside a GPU job. A case file yields `Case`
objects: one shape, implementations called as `fn(i)` with a rotating index so per-layer weight
copies are cycled (cold L2, as in a real step). Report both `graph_us` (decode under CUDA graphs)
and `eager_us` (prefill, and launch overhead). Include the current implementation as `reference`
and a `check` against it. Put the case file and its JSON under
`<campaign workdir>/microbench/<slug>/`.

Decode-step arithmetic to project an e2e effect: per-call savings x call sites per step (layers x
calls per layer) / current TPOT. State the projection as a projection.

## Kernel choices record

Append to `<campaign workdir>/kernel_choices.json` (a JSON list) one object per op you sourced:

```json
{"op": "gdn decode recurrence", "workload": "decode M=1..128, fp32 state, 48 v-heads",
 "chosen": "fused_sigmoid_gating_delta_rule_update (target ops, Triton)", "source": "tier 5",
 "alternatives": [{"name": "flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose", "source": "tier 3 (0.6.18)", "us": 41.2, "note": "fp32 pool path"}],
 "evidence": "<campaign workdir>/microbench/<slug>/result.json", "decided": "<iso time>"}
```
