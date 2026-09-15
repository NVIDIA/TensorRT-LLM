---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 39}
---

<!--
`tests:` is filled from the observed pytest count of the certifying run, never
from arithmetic.
-->

# mhc_split_sinkhorn

**Wraps** `torch.ops.trtllm.mhc_split_sinkhorn` (one call).

## Semantics

Splits one Hyper-Connection coefficient projection into its three coefficient
sets and Sinkhorn-normalizes the combination matrix, in a single kernel. The
result is doubly stochastic at the certified `sinkhorn_repeat = 20`; see the
qualifier below for what a smaller count gives instead.

```
rstd     = rsqrt(r_acc / k + rms_eps)              # one statistic per token
m        = y_acc * rstd                            # the normalized mixes
pre [j]  = sigmoid(m[j]        * scale[0] + base[j])        + hc_pre_eps
post[j]  = hc_post_mult_value * sigmoid(m[hc+j] * scale[1] + base[hc+j])
comb[j,k]= m[2hc + j*hc + k] * scale[2] + base[2hc + j*hc + k]

comb = softmax(comb, dim=-1) + hc_sinkhorn_eps      # eps AFTER the division
comb = comb / (comb.sum(dim=-2) + hc_sinkhorn_eps)  # first column pass
repeat (sinkhorn_repeat - 1) times:
    comb = comb / (comb.sum(dim=-1) + hc_sinkhorn_eps)
    comb = comb / (comb.sum(dim=-2) + hc_sinkhorn_eps)
```

**The fusion boundary.** The call owns the normalization of the mixes, both
sigmoid gates, and the whole Sinkhorn loop. It does **not** own the projection
that produces `y_acc`/`r_acc` (`gemm/mhc_gemm_sqrsum_fma`), and it does **not**
apply any of the three coefficient sets to a residual stream — `pre` is applied
by `mhc_pre_mapping` and `post`/`comb` by `mhc_post_mapping`. Keeping the
application outside is what makes this op usable by a model whose `pre` is
consumed by a *later* sublayer; see "Why this op and not the fused ones".

**Three details a reading of the algebra loses**, each pinned by its own case:

* `pre` gets `+ eps` and `post` gets a **multiplier** — the corrections are
  different shapes, not the same one written twice.
* the first normalization is a row **softmax** with `+ eps` applied *after* the
  division, while every later one divides by `sum + eps`.
* the loop body runs `sinkhorn_repeat - 1` times, because the row softmax **and**
  the first column pass happen before it. The sequence therefore **ends on a
  column pass**, which is why at the certified 20 passes column sums land on 1 to
  fp32 precision and row sums only approach it — and it is also why
  `sinkhorn_repeat <= 1` is not "a call that skips the column normalization" but
  simply the 1-pass call.

**"Doubly stochastic" is a property of the certified `sinkhorn_repeat = 20`, not
of the kernel at any count.** At one pass a column whose four entries are all
near zero lands at `4/5`, because the column divide is by `cs + eps`; the
measurement and the mechanism are under "Accepted silently and NOT guarded" in
Preconditions.

**`hc` is a compile-time 4.** `mhcKernels.cu` declares `constexpr int
HC_MULT = 4`, so every buffer width here is fixed rather than derived from an
argument. A checkpoint with a different hyper-connection multiplier needs a
different kernel, not a different call.

## Signature

```python
def mhc_split_sinkhorn(
    y_acc, r_acc, hc_scale, hc_base,
    k, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
) -> tuple[Tensor, Tensor, Tensor]
```

| name | shape | dtype | layout / device | notes |
|---|---|---|---|---|
| `y_acc` | `[M, 24]` | float32 | contiguous, CUDA | the unnormalized projection; `24 = (2+4)*4` |
| `r_acc` | `[M]` | float32 | contiguous, CUDA | the square sum over the flattened stream |
| `hc_scale` | `[3]` | float32 | contiguous, CUDA | pre / post / comb, in that order |
| `hc_base` | `[24]` | float32 | contiguous, CUDA | same layout as `y_acc`'s row |
| `k` | scalar | Python int | — | the width `r_acc` spans; divides it |
| `rms_eps` | scalar | Python float | — | inside the rsqrt |
| `hc_pre_eps` | scalar | Python float | — | added after `pre`'s sigmoid |
| `hc_sinkhorn_eps` | scalar | Python float | — | the Sinkhorn's |
| `hc_post_mult_value` | scalar | Python float | — | `post`'s multiplier |
| `sinkhorn_repeat` | scalar | Python int | — | total passes; the loop runs this minus one |
| **returns** | `[M,4]`, `[M,4]`, `[M,4,4]` | float32 | contiguous, CUDA | `(pre, post, comb)`, freshly allocated |

`M` is **derived from `y_acc`** and the outputs are allocated here. The op takes
`M` as an argument and writes that many rows into whatever buffers it is given:
driven on sm_103, `M = 2 * rows` against correctly sized outputs was accepted and
wrote past their end. A wrapper that allocates and derives cannot express that,
which is better than guarding it.

## Certified coverage

What the receipt covers. Every row is one or more cases in
`tests/unittest/_torch/staircase/activation/test_staircase_mhc_split_sinkhorn.py`.

| axis | certified values | cases |
|---|---|---|
| **V4.1 column x rows** | `k=20480`, `rms_eps=1e-20`, `hc_pre_eps=hc_sinkhorn_eps=1e-6`, `hc_post_mult_value=2.0`, `sinkhorn_repeat=20`, x 17 row buckets | 17 |
| accuracy row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| **`M`, as a proved interval** | `M ∈ [0, 16384]` — see "The certified `M` interval" below | 1 |
| zero logical rows | `M = 0` returns `[0,4]`, `[0,4]`, `[0,4,4]` with no queued CUDA error | 1 |
| doubly stochastic | column sums to 1 at `atol=1e-5`, row sums looser, and the asymmetry asserted | 1 |
| eps / multiplier placement | `pre_eps` shifts `pre` and leaves `post` bit-identical; `post_mult` scales `post` linearly and `post/mult <= 1` | 1 |
| `sinkhorn_repeat` | 1 against 20: `comb` moves, and row sums converge | 1 |
| `sinkhorn_repeat <= 1` | 1, 0 and -1 are bit-equal and match the 1-pass reference; their column sums are 0.2 off while 20's are under 1e-5 | 1 |
| comb orientation | transposing the comb block of `y_acc` and `hc_base` changes the result | 1 |
| loud rejection | non-fp32 `y_acc` | 1 |
| wrapper guards (metadata only) | 8 branches, each raw-driven to accepted/finite/attributably-wrong first | 8 |
| scalar domain: `k` 0 / negative | raw-driven to accepted-and-collapsed / accepted-and-all-NaN, then **passed through** by the wrapper | 2 |
| scalar domain: `sinkhorn_repeat` 0 / -1 | raw-driven to accepted and **clamped to the 1-pass result**, then **passed through** | 2 |
| guards that must NOT fire | oversized `hc_scale` / `hc_base` / `r_acc` with **poisoned** tails, each bit-equal to correct | 1 |
| CUDA-graph replay | the **wrapper** captured and replayed twice with different inputs, both matching eager | 1 |
| total | | **39**, all passing; `tests: 39` in the receipt |

The tolerance is `torch.testing.assert_close`'s fp32 default, **unchanged**.
Both sides are fp32 and the op measures ~2e-7 against the reference, so nothing
here needs a widened floor.

### The certified `M` interval

A sampled bucket list cannot certify `M`: the target has not been assembled yet,
so the prefill chunk lengths and decode capture buckets it will emit are not
knowable from inside this entry, and an enumeration would silently exclude
ordinary values (`M = 4` was absent from the list above). The claim is therefore
made as an **invariant with a proof**, not as a list.

*The rule.* The result for row `i` does not depend on `M`. The kernel's only
cross-lane traffic is the Sinkhorn column sum, a `__shfl_xor` over the token's
own aligned quad — `BLOCK_SIZE` is a multiple of 32 so a quad never straddles a
warp — and the tail is an early `token >= M` return. Every row offset is formed
in 64 bits.

*The evidence.* `test_row_results_are_independent_of_m` checks one `M = 16384`
run against the native-torch reference, then requires every `M` in a **dense
sweep of 0..129** plus the neighbourhoods of 256, 512, 1024, 2048, 4096, 8192
and 16384 to reproduce that run's first `M` rows **bit-exactly**. 150 values of
`M` in all.

So **any `M` in `[0, 16384]` is certified**, including zero logical rows (AC4)
and every small decode batch. `M > 16384` is not claimed — the rule predicts it
holds, but the prediction is not a receipt.

Uncertified and therefore not claimed: `M > 16384`, `hc_mult != 4` (impossible —
compile-time constant), non-fp32 inputs, and any arch other than sm_103.

## Why this op and not the fused ones

The same build registers `mhc_big_fuse` and `mhc_fused_hc`, which do more per
call and would normally be preferred. Both are **driven** in
`staircase_v41_hc_probe.py` section 7 — an earlier revision of this entry
rejected them from a source reading alone, which is a guess, not a result.
Measured at V4.1 geometry, against the reference for both readings of `pre`:

| op | post/comb | layer_input vs IMMEDIATE | layer_input vs DELAYED |
|---|---|---|---|
| `mhc_big_fuse` | 1.2e-07 / 1.8e-07 | **3.38e-03** | 8.84e-01 |
| `mhc_fused_hc` (backend 3) | 2.7e-03 / 2.4e-03 | **3.31e-03** | 7.94e-01 |

So both compute the coefficients correctly and both collapse the stream with the
`pre` they just derived: the immediate reading matches to ~3e-03 while the
delayed one is 240x further away. That separation is the rejection, and it is a
number rather than an argument.

`mhc_fused_hc` carries a second, independent bar. Its MMA backends are
statically instantiated per hidden size, and driving backend 2 at this
checkpoint's 5120 raises
`mhcFusedHcAllInOneLaunch: unsupported hidden_size=5120; supported hidden sizes
are 4096 and 7168`. Its FMA path also refuses the heuristic tactic default that
the other backends accept (`SHAPE_N=24 not divisible by tile_n=0`), so a caller
must pick a compiled `(tile_n, ks, tile_m)` triple by hand.

`mhc_fused_hc_mma_enabled()` is a **capability query**, not a tensor-compute
candidate; it returns `True` here and is listed separately for that reason.

Both fused ops derive `pre` and apply it to the residual stream **inside the
same call**, from the same `(y_acc, r_acc)` that produce `post`/`comb`.
DeepSeek-V4.1-Flash does not do that. Its `Block.forward` uses the *incoming* `pre_mix` for the
attention sublayer and the *attention's* `pre` for the FFN sublayer, returning
the FFN's `pre` for the next block — so it needs the **current** sublayer's
`post`/`comb` together with the **previous** sublayer's `pre`. One call reading
one `(y_acc, r_acc)` cannot express that pairing, and it is not a flag the
caller can set.

`mhc_hc_head_apply` is a genuine third candidate and it **matches**: it derives
`pre` from whatever `(mixes, sqrsum)` it is handed and applies it to whatever
stream it is handed, so the delayed schedule *is* expressible by threading the
previous sublayer's accumulators in — measured on both the immediate and the
delayed drive at `rel_scale` 2.07e-03 and 1.95e-03. It loses to
`mhc_pre_mapping` on **state**: carrying `pre` is `[M,4]` fp32, carrying
`(y_acc, r_acc)` is `[M,24]` plus `[M]`, and this op has to run anyway for
`post`/`comb` — so the fusion saves no work, it only recomputes a `pre` the
split already produced.

## CUDA-graph replay

**The wrapper itself is capture-safe, and the target calls it normally inside a
captured region.** No caller-owned output buffers, and no reason to reach past
the catalog surface to the raw op.

An earlier revision of this entry claimed the opposite — that capture required
pre-allocated outputs the wrapper does not provide, so Goal 1.7 would have to
call `torch.ops.trtllm.mhc_split_sinkhorn` directly. That was wrong, and it was
wrong in the expensive direction: it certified a surface no caller of this entry
uses and left the wrapper unproven under capture. The wrapper's three
`torch.empty` calls are made *inside* `torch.cuda.graph`, so they come from the
graph's private memory pool; the tensors the capture call returned are the same
memory every replay rewrites, and holding those three references is all the
caller has to do.

`test_cuda_graph_replay_through_the_wrapper_matches_eager` captures
`mhc_split_sinkhorn(...)` and replays **twice with different inputs**, checking
both against the native-torch reference. One replay would not distinguish a
working graph from one that silently reproduced its captured input.

## Metadata consumed

None. Stateless — no runtime, attention metadata, or workspace.

## Preconditions

Every statement below was driven on **sm_103 / trtllm 1.3.0rc26**.

### Rejected loudly by the op — the wrapper does not repeat it

| violation | observed message |
|---|---|
| `y_acc` not float32 | `RuntimeError: expected scalar type Float but found BFloat16` |

### Accepted silently AND guarded by the wrapper — eight metadata violations

A wrapper assert is authorized only for a **pure tensor-metadata** violation
(shape / stride / dtype / device) that the op was observed to accept silently.
Both halves are required, and silence alone is not enough — which is why the
four scalar domains in the next section are documented here instead of being
rejected. An earlier revision of this wrapper asserted `k >= 1` and
`sinkhorn_repeat >= 1`; those are computation values, and the asserts were a
wrapper-contract violation rather than a safety feature.

Every row below was driven through the raw op in
`test_staircase_mhc_split_sinkhorn.py`, which asserts acceptance **before** it
asserts the wrapper's guard. Each malformed argument differs from a correct one
in **metadata only** — the same logical values, with any storage the kernel then
reads past them poisoned by the test — so the damage is deterministic and
attributable to the shape or the stride rather than to the values. (An earlier
revision built these out of fresh random or uninitialized tensors and out of a
fresh short `r_acc` whose tail happened to hold the right bytes; that one
printed *accepted and bit-equal*, which reads as harmlessness and is not.)

All eight are accepted, finite, and wrong **exactly where the metadata reaches**.
The attribution column is asserted, not described:

| violation | bit-equal to correct | differs |
|---|---|---|
| `y_acc` width other than 24 (row stride is a compile-time 24) | row 0 of all three | rows 1.. |
| `y_acc` non-contiguous (a column slice of a wider buffer) | row 0 of all three | rows 1.. |
| `r_acc` shorter than `M` | rows `0..M/2` of all three | rows `M/2..M` |
| `r_acc` non-contiguous | row 0 of all three | rows 1.. |
| `hc_scale` shorter than 3 | `pre`, `post` | `comb` |
| `hc_scale` non-contiguous | `pre` | `post`, `comb` |
| `hc_base` shorter than 24 | `pre`, `post` | `comb` |
| `hc_base` non-contiguous | — | `pre`, `post`, `comb` |

### Accepted silently and NOT guarded — four scalar domains the caller owns

`k` divides the square sum and `sinkhorn_repeat` is a loop bound, so both are
**computation values, not metadata**. The wrapper passes them through untouched
and this section is the only place a caller can learn what they do — which is
the trade the metadata-only rule makes deliberately. Each is measured through
the raw op, and the wrapper is then asserted to return the same thing rather
than to reject it.

**`k = 0`** divides the square sum by zero, so `rstd = rsqrt(inf)` is 0 and the
normalized mixes collapse. All three outputs are **finite**, plausible, and not
the caller's result.

**`k < 0`** makes `r_acc / k + rms_eps` negative, so `rsqrtf` returns NaN and
**every** element of **all three** outputs — `pre`, `post` and `comb` — is NaN.
A caller who checks `isfinite` catches this one and none of the others. (An
earlier revision of the test asserted only `pre` and `comb` while this sentence
claimed all three; `post` is asserted now.)

**`sinkhorn_repeat` of 0 or a negative value** returns **bit-equal to
`sinkhorn_repeat = 1`**, because the row softmax and the first column
normalization sit outside the kernel's `for (it = 1; it < sinkhorn_repeat; ...)`
loop. The argument is ignored rather than honoured, and the returned `comb` is a
plausible non-negative matrix, so nothing about it looks wrong. (An earlier
revision of this entry said 0 "returns comb without the column normalization".
It does not — the column pass is unconditional.)

**Why one pass is nonetheless not doubly stochastic**, which is the same fact
from the other side and is the reason `sinkhorn_repeat` matters at all: the
column divide is by `cs + hc_sinkhorn_eps`, so a column whose four entries are
all near zero — every row's softmax put its mass elsewhere — has `cs ≈ 4·eps`
and lands at `4/5` rather than 1. Measured on sm_103 at one pass: column sums
**0.2** away from 1. At the certified `sinkhorn_repeat = 20` the alternating
passes have removed those degenerate columns and the sums are within `1e-5`.
The doubly-stochastic guarantee in "Semantics" is therefore a property of the
certified iteration count, not of the kernel at any count.

**OVERSIZED buffers are harmless and are NOT guarded.** `hc_scale` longer than
3, `hc_base` longer than 24, and `r_acc` longer than `M` are each accepted and
**bit-equal** to the correct result, with their tails poisoned, so the kernel
demonstrably indexes what it needs and ignores the rest. An earlier revision of
the wrapper asserted exact sizes and rejected all three, which is a defect in
the wrapper rather than a safety feature. The guards are minimum-length checks,
and `test_oversized_buffers_are_harmless_and_not_guarded` keeps the exact-size
form from coming back.

A caller inside the coverage table above, violating none of the above, gets a
result within `torch.testing.assert_close`'s fp32 defaults of a native-torch
reference built from the algebra at the top.

## Notes

* **The source for this op was recovered onto this branch, and the built library
  was already ahead of it.** Before recovery the mounted `.so` exported
  `mhc_split_sinkhorn` while `cpp/tensorrt_llm/thop/mhcOp.cpp` did not declare
  it — so the op ran, and the next rebuild from this branch would have removed
  it. `cpp/.../mhcOp.cpp`, `cpp/.../mhcKernels/mhcKernels.{cu,h}` were recovered
  from `wip/v41-flash-catalog-20260914` (purely additive: 300 lines added, 0
  deleted), and `staircase_v41_hc_probe.py` section 0 now reports
  `declared_in_cpp=True` for all eight `mhc_*` names.
* **The recovered sources have since been compiled on this branch, and the gap
  is closed.** Review job 749657 ran the mandated rebuild: clang-format clean,
  Ninja rebuilt the mHC CUDA/C++ objects, the checkout libraries linked, the
  wheel built, `mhc_split_sinkhorn` was found in the checkout's
  `libth_common.so`, and the expected schema registered on import. Every receipt
  from that point on is against a library built from this branch's own sources.
* The op's own schema names `sinkhorn_repeat`, and it is the TOTAL number of
  passes: the kernel runs the loop body `sinkhorn_repeat - 1` times because the
  row softmax and the first column pass are outside it. A caller reading the
  name as "extra iterations" would be off by one, and a caller reading 0 as "no
  normalization" would be wrong in a different way — see the scalar-domain
  section of Preconditions.
