# Multi-token prediction (MTP)

What a target assembled on a checkpoint that ships an MTP module had to
establish about **what the extra layer computes**, because no contract
states it and the checkpoint ships no reference implementation for it.
Written from the deepseek-r1-0528-nvfp4/sm_100/dep4 MTP increment
(DeepSeek-R1-0528, 61 trunk layers, hidden 7168, 256 routed experts top-8,
one MTP module at layer 61). The mechanism — an extra decoder layer that
consumes *both* the trunk's hidden state and the next token's embedding, and
is replayed to produce a draft — recurs across the DeepSeek family and its
derivatives; this file is about the mechanism, not that checkpoint.

The runtime half of the picture — how the engine finds the drafter, what
calls the layer, and how the batch shape changes between draft steps — is
`docs/references/trtllm-runtime-integration.md` §13. This file is only the
computation and the weights.

**The checkpoint is the whole specification here, and it is a partial one.**
DeepSeek's published `modeling_deepseek.py` does **not** implement the MTP
module (verified across four R1/R1-0528 checkpoints on disk, quantized and
not), and `transformers`' own `deepseek_v3` does not either. So the graph
below was reconstructed from the checkpoint's key names, shapes and weight
statistics. Every claim that rests on the reconstruction rather than on a
shape is marked.

## The mechanism

The module is one extra decoder layer, structurally identical to a trunk MoE
layer, with a front end bolted on that mixes in the embedding of the token
being predicted:

```
e = embed_tokens(input_ids)                     # embedding of the NEXT token
x = eh_proj( concat( enorm(e), hnorm(h) ) )     # [T, 2*hidden] -> [T, hidden]
x = x + MLA( input_layernorm(x) )               # same block as a trunk layer
x = x + MoE( post_attention_layernorm(x) )      # same structure, different dtype
# shared_head, called separately by the runtime:
logits = lm_head( shared_head.norm(x) )
```

`h` is the trunk's final hidden state for the same position — under
one-model MTP-Eagle the runtime hands the layer the target model's own
output, with no projection in between.

**"How many MTP layers do we enable" is not the knob.** A checkpoint with
one MTP module (`num_nextn_predict_layers: 1`) is replayed autoregressively:
the same layer runs `max_draft_len` times, each step feeding it the previous
step's output token and hidden state. The draft length is a serving knob,
not a checkpoint property. §13 has the mode-selection rule.

## The concat order — the one silent-wrong-answer trap

`eh_proj` is `[hidden, 2*hidden]`. Which half multiplies the embedding
branch and which multiplies the hidden branch is **not** determined by
anything in the checkpoint's metadata, and getting it wrong is a pure
numerical error: no shape mismatch, no assert, no crash. Under rejection
sampling it does not even corrupt output — the drafts are simply always
rejected (see the last section).

**Two sources disagree, and the naming is the one that is right.**

* The DeepSeek-V3 report writes the projection as `M_k [ RMSNorm(h) ;
  RMSNorm(Emb(t)) ]` — **hidden first**.
* The parameter is named `eh_proj`, with its two gains named `enorm` and
  `hnorm` — **embedding first**.

**Measured: the embedding block is first.** Two independent statistics over
the checkpoint's own weights, each controlled:

*Column-norm profile.* Take the per-input-dimension L2 norm of each half of
`eh_proj` and correlate it with each RMSNorm gain. The pairing is exclusive:

| | vs `\|enorm\|` | vs `\|hnorm\|` | vs `\|model.norm\|` |
|---|---|---|---|
| first half `eh_proj[:, :hidden]` | **+0.904** | +0.037 | +0.026 |
| second half `eh_proj[:, hidden:]` | −0.036 | −0.207 | **+0.544** |

`model.norm` is the trunk's final RMSNorm, which gates the same residual
stream `hnorm` does, so the second half tracking it is the same statement as
the second half being the hidden branch.

*Functional response.* Feed the embedding branch's actual input,
`enorm(Emb(t))`, through each half and out through `shared_head`, and
measure the entropy of the resulting distribution. **Compare each half
against its own controls, not against the other half** — the first half is
uniformly sharper by ~0.9 nats on *any* input, so a raw cross-half
comparison is confounded:

| input | via first half | via second half |
|---|---|---|
| `enorm(Emb(t))` — the real embedding-branch input | **2.62 nats** | 6.22 nats |
| `hnorm(gaussian)` | 5.33 | 6.24 |
| gaussian, no gain (control) | 5.36 | 6.23 |
| `enorm(gaussian)` — right gain, wrong vector (control) | 5.09 | 6.17 |

The first half drops **2.5 nats** on the real embedding and on nothing else.
The second half does not move at all: through it, a real token embedding is
indistinguishable from noise. (Uniform over a 129,280 vocabulary is 11.77
nats.)

So: **`eh_proj` consumes `concat(enorm(e), hnorm(h))`.** This is a
reconstruction from weight statistics, not a reading of reference code — it
is strong enough to implement against, and the acceptance rate is what
confirms it end to end. Keep the order as a single named constant at one
place in the layer so flipping it is a one-line experiment.

## Checkpoint layout

The module is stored as one more entry in the layer list, at index
`num_hidden_layers`. On this checkpoint that is 790 keys under
`model.layers.61.`:

| key | count | dtype | shape | load? |
|---|---|---|---|---|
| `enorm.weight`, `hnorm.weight` | 2 | bf16 | `[7168]` | yes |
| `eh_proj.weight` | 1 | bf16 | `[7168, 14336]` | yes |
| `input_layernorm`, `post_attention_layernorm` | 2 | bf16 | `[7168]` | yes |
| `self_attn.*` | 7 | bf16 | **identical to a trunk layer's** | yes |
| `self_attn.{k_proj.k_scale, v_proj.v_scale}` | 2 | fp32 | `[]` | value is **1.0**, as in the trunk |
| `mlp.gate.weight` | 1 | bf16 | `[256, 7168]` | yes |
| `mlp.gate.e_score_correction_bias` | 1 | **fp32** | `[256]` | yes |
| `mlp.experts.{0..255}.{gate,up,down}_proj` | 768 | bf16 | `[2048, 7168]` ×2, `[7168, 2048]` | yes, EP-windowed |
| `mlp.shared_experts.{gate,up,down}_proj` | 3 | bf16 | same | yes |
| `shared_head.norm.weight` | 1 | bf16 | `[7168]` | yes — **a distinct norm**, not `model.norm` |
| `embed_tokens.weight` | 1 | bf16 | `[129280, 7168]` | **no** |
| `shared_head.head.weight` | 1 | bf16 | `[129280, 7168]` | **no** |

**The last two are bit-identical copies of the trunk's** (`torch.equal`
against `model.embed_tokens.weight` and `lm_head.weight`: both True). The
runtime hands the layer whichever `embed_tokens` and `lm_head` the target's
draft-model container exposes, so pointing them at the trunk's is correct
and saves 1.85 GB per rank. They stay in the weight manifest's *predicted
non-load* set even with MTP on; the other 788 keys flip from non-load to
consumed.

**The MTP layer's attention geometry is byte-identical to a trunk layer's**
— same `q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` /
`o_proj` shapes, same two LayerNorms, same `q_lora_rank`. Whatever load-time
derivation the trunk's attention needs (absorption operands, row regrouping,
the rope table) applies unchanged.

## The MTP layer is not quantized, and that is deliberate

On a quantized export the MTP module can be excluded from quantization
wholesale. Here `hf_quant_config.json` carries `model.layers.61*` as one
wildcard entry in a 63-entry `exclude_modules` list, so **every weight in
the module is bf16** while the trunk's MLP path is NVFP4.

**Do not re-quantize it at load time to reuse the trunk's expert
vocabulary.** The exclusion is the export's choice about where accuracy is
worth the bytes; quantizing it anyway changes what the checkpoint means, and
the acceptance rate — the only signal that can see the difference — would
absorb the damage silently as a lower draft quality rather than reporting
it.

The consequence is a vocabulary consequence: the MTP layer's routed experts
need a **bf16** grouped-expert entry, at whatever `(local_experts, hidden,
intermediate)` the parallel split produces, while the trunk's use the NVFP4
runner. Everything else in the graph — both norms, the concat, the
projection GEMM, the whole MLA block, the router, the shared expert,
`shared_head` — maps onto entries a dense-plus-MoE target already carries.

## The HBM arithmetic, and the break-even acceptance rate

A decode step is weight-bandwidth-bound: each rank reads every weight byte
it holds, once. So the cost of drafting is exactly the MTP layer's byte
count, times the number of draft steps.

Per rank on this checkpoint at `dep4` (64 local experts of 256):

| | bytes per draft step |
|---|---|
| routed experts, **bf16**: `3 x 2048 x 7168 x 64 x 2` | **5.637 GB** |
| attention, bf16: 187.1 M params x 2 | 374 MB |
| `eh_proj`: 102.8 M x 2 | 206 MB |
| shared expert: `3 x 2048 x 7168 x 2` | 88 MB |
| router | 4 MB |
| **total** | **≈ 6.31 GB** |

Against a trunk step of ≈ 115 GB per rank (58 NVFP4 MoE layers at 1.585 GB +
61 bf16 attention blocks at 374 MB), that is **+5.5% per draft step**. Note
what the first row means: **the bf16 MTP layer's experts cost 3.56 times
what one NVFP4 trunk MoE layer costs**, purely from the dtype.

A step now produces `acceptance_length` tokens instead of 1, so drafting
pays for itself when

```
acceptance_length  >=  1 + max_draft_len * 0.055
```

| `max_draft_len` | extra HBM/step | break-even `acceptance_length` |
|---|---|---|
| 1 | +5.5% | **1.055** |
| 2 | +11.0% | 1.110 |
| 3 | +16.4% | **1.164** |
| 4 | +21.9% | 1.219 |

**That table is the weight-bytes model, and measurement says it is a lower
bound that stops holding as the batch grows.** Measured on this checkpoint
at `max_draft_len 3`, decomposing each engine step against a non-drafting
one:

| con | acceptance | step cost — predicted | step cost — **measured** | decode speedup |
|---|---|---|---|---|
| 1 | 3.4904 | 1.164 | **1.511** | 2.311x |
| 32 | 3.3356 | 1.164 | **2.198** | 1.518x |
| 256 | 3.4033 | 1.164 | **2.600** | 1.309x |

The marginal cost of each successive draft step at con=256 is **+0.645,
+0.569, +0.386**, against con=1's **+0.200, +0.173, +0.137**. The growing
term tracks the **rows** a step carries, not the layer's fixed weight bytes
— it falls off in the same shape the marginal row count does (1→2 rows is
+100%, 2→3 is +50%, 3→4 is +33%). **That is the
memory-bound-to-compute-bound crossover this file calls the genuinely
uncertain end, now measured rather than predicted.**

So read the weight-bytes table as the floor it is: right at low concurrency,
where a step is bandwidth-bound and drafting really is nearly free, and
roughly 2.2x optimistic at the throughput end. What survives is the
conclusion, because the cost never grows fast enough to catch acceptance:
**at every concurrency measured, acceptance stayed at 2.9x or more of even
the measured break-even**, and no draft length in the certified range lost.

**The break-even is still very low against the predicted cost** — at
`max_draft_len` 3 it takes only 0.164 extra tokens per step on average — and
two costs that look like they should matter do not:

* **Collectives.** The MTP layer is an MoE layer, so it adds two per draft
  step, and the trunk's grow by `draft_len + 1` in bytes. But at decode
  sizes these calls are latency-bound rather than bandwidth-bound (measured
  on this target: 2.753 MB in 25.472 µs = 108 GB/s, an order of magnitude
  under NVLink), so multiplying the byte count barely moves the time.
* **KV capacity.** One extra layer on a 61-layer pool is **+1.6%**, plus
  `max_draft_len - 1` extra tokens per sequence.

The end that is genuinely uncertain was the **high-concurrency** end — once
a rank's batch already activates all of its local experts and the routed
GEMM is near the HBM roofline, `draft_len + 1`× the rows means the same
weight bytes with several times the arithmetic. It is measured above: the
crossover is real, it costs 2.2x the predicted step cost at con=256, and it
still does not overtake acceptance.

**Acceptance itself is near-flat in batch size on this mechanism, which is
the other half of why no draft length loses.** Measured across three draft
lengths and nine concurrencies: `max_draft_len` 1 stayed in 1.9446–1.9673 of
a 2.0 ceiling, 2 in 2.6945–2.7707 of 3.0, and 3 in 3.3356–3.4904 of 4.0.
**So a "shorten the draft as the batch grows" schedule has no
acceptance-side motivation here** — any case for one has to be made on cost,
and on this checkpoint the cost side does not invert. (It is also unusable
under attention DP for a structural reason — see
`docs/references/trtllm-runtime-integration.md` §13.)

## What only the acceptance rate can tell you

**An MTP layer that computes the wrong thing does not produce wrong
output.** Rejection sampling guarantees the emitted distribution is the
target model's regardless of draft quality. A layer with the concat order
reversed, a norm on the wrong operand, or the expert stack packed in the
wrong interleave produces **bit-correct text, more slowly**, with every
draft rejected. An accuracy benchmark cannot see it. Neither can a smoke
gate.

`acceptance_length` — the mean tokens emitted per step by requests that
carried a draft, `1.0` meaning total rejection — is the only detector, and
it needs a reference to be read against:

* **≈ 1.0** at `max_draft_len` 3: the layer is wrong outright.
* **clearly above 1.0 but well below the reference**: the layer is *subtly*
  wrong. This is the band the concat order lands in, and nothing else
  distinguishes it from "this model just drafts poorly".
* **at the reference**: the layer computes what the checkpoint says.

The reference is the same checkpoint under stock in-tree modeling, at the
same load and the same speculative configuration. Acceptance is a property
of the model, so a reference forced onto different memory knobs to boot is
still comparable.

**One trap when choosing the workload, and it does not point the way it
looks like it should.** A serving harness that builds prompts from uniformly
random token ids seems like it must *understate* acceptance — the
continuation of a nonsense prefix is unpredictable, so the drafts should
miss. Measured, it runs the other way, and strongly.

Only the **prompt** is random. Generation then runs for a fixed output
length with EOS ignored, so what MTP is drafting is the model's own
continuation of a nonsense prefix — which degenerates into repetition, and
repetition is the easiest thing in the world to draft. Measured on this
checkpoint at `max_draft_len 3` (ceiling 4.0): **`acceptance_length` 3.93 at
concurrency 1 and 3.46 at 32**, i.e. 97.6% and 82.1% of proposed draft
tokens accepted. Real text does not do that.

So the random-prompt harness **flatters** MTP, and two things follow. A
throughput curve measured on it overstates the gain, so it answers "did
anything get slower" and not "is MTP worth enabling" — that one needs a
real-text workload, and an accuracy-gate run already produces one. And a
future run that sees a *low* acceptance number here must **not** excuse it
as "well, the workload is random": on this harness a low number means the
layer is wrong.
