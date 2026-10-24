# Attention sinks

A learned per-query-head logit that joins the attention softmax's
denominator and is then dropped from the output, so each row's weights
sum to less than 1. Written from the gpt-oss-120b/sm_100/tp1 onboard
(`self_attn.sinks`, one `[num_heads]` tensor per layer); the mechanism is
not specific to that family.

## What it computes

```
out[i, h] = softmax(concat(logits[i, h, :], sink[h]))[:-1] @ V
mass[i, h] = sigmoid(logsumexp(logits[i, h, :]) - sink[h])   # the row's weight sum, < 1
```

The sink is one extra column in the softmax, never a value contribution.
A large sink drains the row toward zero output; a sufficiently negative
one (`-100.0` or `-inf` measured) is bitwise identical to passing no sink
at all, which makes the inert path easy to verify.

## The two facts a target gets wrong by guessing

**The sink lives in the scaled-score domain.** It is compared against
`q @ K.T / (q_scaling * sqrt(D))`, not the raw dot products. At `D = 64`
the scale is 1/8, so the two readings are far apart — the wrong one
measured 12.8x to 41x outside the tolerance band on every case. Do not
pre-scale the checkpoint's sink values.

**It must hold in both phases, and a kernel that honours only one fails
silently.** Context prefill and generation decode take different FMHA
kernels, and long-history decode takes a third variant that folds partial
softmax states across CTAs (`...MultiCtasKvCga...`) and has to apply the
sink inside that reduction. Certify each separately, and gate every case
against a "sink ignored" reference as well as a correct one — otherwise a
passing test proves only that the numbers are plausible.

## Handling requirements

- **fp32**, whatever the checkpoint says. gpt-oss stores `sinks` as bf16;
  the op raises on anything but float. Promote at load, alongside the
  expert biases.
- **Contiguous, exactly `num_heads` elements.** The op validates the
  dtype and nothing else: a short tensor makes the kernel read past the
  end of the allocation, a longer one has its tail ignored, a strided
  view whose *values* are right still reads the interleaved memory and
  returns wrong attention, and an empty tensor silently disables the
  sink. Shape beyond the element count is irrelevant (`[H]`, `[1, H]`,
  `[H, 1]` all behave identically). Pure metadata, so the wrapper carries
  the guard the op lacks — `thop_attention.py` asserts contiguity and
  `numel() == num_heads`.
- **Per query head**, not per kv head. A checkerboard sink (large on even
  heads, very negative on odd) drains exactly the even head columns while
  leaving the odd ones bitwise identical to the no-sink call — that is
  the cheapest way to pin the indexing.
- The sink perturbs nothing but the denominator: the paged KV append
  stays bit-exact and no cache bytes move.
