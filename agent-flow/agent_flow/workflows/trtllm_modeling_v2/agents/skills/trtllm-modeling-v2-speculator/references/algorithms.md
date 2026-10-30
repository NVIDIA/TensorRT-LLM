# Drafting algorithms — what differs between the modes

One skill covers several `--features` values because they share a binding
surface and a gate. This file is the part that differs: what each mode's draft
path computes, and therefore what the target has to write.

The binding surface itself — where the draft module attaches, what the shell
gains, how the weight-loading path changes — is `runtime-boundary.md` and is
the same for all of them.

## The family

Every mode here is **one-engine**: one `PyTorchModelEngine` does target
verification and the next round's drafting in a single model forward. Two-model
draft-target is a different shape and is not covered.

| mode | draft path | weights it turns on | notes |
|---|---|---|---|
| **MTP** | the checkpoint's own MTP module, replayed autoregressively `max_draft_len` times | the MTP module in the checkpoint | `references/multi-token-prediction.md` is the depth. `num_nextn_predict_layers` is a module count, **not** the draft-length knob |
| **Eagle3** | a separate small drafter reading the target's hidden states | a separate drafter checkpoint | needs hidden states exported from the trunk — a forward-shape change the other modes do not require |
| **DFlash** | as upstream `_torch/speculative/dflash.py` | | not yet exercised by any target here |
| **DSpark** | the checkpoint's own block drafter: one 3-stage pass drafts a whole block, then a sequential Markov refine | the drafter stages, which a checkpoint may store as `mtp.*` | **not MTP, whatever the namespace says.** See the DSpark section below |
| n-gram, suffix automaton, PARD | no weights at all — drafts from the emitted prefix | none | no weight-loading change, so the gate reduces to acceptance alone |

Only MTP has been carried end to end here. Everything about the others below is
read from the in-tree sources and the checkpoints' own reference code, not
measured — say so if you rely on it.

## What decides the draft length

`speculative_config.max_draft_len`, spelled out. Two things make it more than
a knob:

- **left unset it resolves to 1**, not to anything workload-derived
- it becomes the decode kernel's `maxSeqLenQ` — on an MLA layer through the
  attention entries' `predicted_tokens_per_seq`, on a standard layer through the
  **generation row count** instead, since `predicted_tokens_per_seq` is inert
  there (`runtime-boundary.md`). So its ceiling is a **certification** bound:
  going past what the MLA/attention entries certify needs
  `trtllm-modeling-v2-kernel-verify` first. The JIT cost is **not** one compile per
  distinct value: measured on sm_100, a row count of 2 adds one entry and 3 and
  4 reuse it, so drafting costs exactly one extra first-call compile whatever
  the draft length — boot time, not a failure

The checkpoint's module count does not bound it. On the MTP-Eagle path a
single module is replayed `max_draft_len` times.

## Choosing, when the checkpoint offers a choice

If the checkpoint ships a draft module, use it — MTP. Its weights were trained
against the target and its acceptance ceiling is far above what a
weightless mode reaches.

Reach for a weightless mode when the checkpoint has no draft module and the
workload is repetitive enough that prefix matching pays. It costs no weights,
no KV pool growth and no boot-time JIT, so the gate is cheap; the acceptance
reference is still required, because "drafts nothing, accepts nothing" and
"drafts wrongly, accepts nothing" look identical without one.

## DSpark — a block drafter that lives where MTP would

A checkpoint can ship a DSpark drafter under `mtp.*` with
`num_nextn_predict_layers` in its config, which is exactly what makes the
wrong feature easy to pick. Its tech report says which drafter it trained;
quote that in the plan.

### Telling it from MTP

Any one of these settles it:

- the config carries `dspark_*` keys — `dspark_block_size`,
  `dspark_noise_token_id`, `dspark_target_layer_ids`, `dspark_markov_rank`
  (a multimodal config may nest them under `text_config`);
- the `mtp.*` weights include `main_proj`, `markov_head`, `confidence_head`
  and lack MTP's signature: no `e_proj` / `h_proj` / `enorm` / `hnorm` /
  `shared_head`;
- the reference forward drafts a block in one call rather than one token per
  module.

Then the feature is `dspark` and never `mtp`. `MTPDecodingConfig` would read
`num_nextn_predict_layers: N`, pick vanilla sequential MTP, and build N
chained one-token heads over weights trained as one N-stage block pass: it
either fails to load the missing MTP keys or drafts garbage that verification
rejects — and the second passes accuracy. If a request says "MTP" for such a
checkpoint, it means "speculate with the bundled drafter", which is `dspark`;
write that down in the spec so a later reader does not "fix" it.

### What the draft path computes

From the checkpoint's reference forward. Each item is a place a transliteration
from an MTP or Eagle mental model goes wrong **and still drafts**, so the only
symptom is a lower acceptance length.

1. **One pass drafts the whole block.** The input is a `block_size`-token
   block `[x, NOISE, ..., NOISE]`: `x` is the token the target just emitted,
   NOISE is `dspark_noise_token_id`. The stages run **once**, in sequence,
   over every block position — each stage a full transformer block as the
   reference defines it — and the head produces base logits for all positions
   in parallel. Not one pass per position; not chained heads.
2. **Attention inside the block is non-causal.** Every block position attends
   to every other block position plus the filled ring slots. A causal mask
   here is the most likely error to carry over from a one-token drafter.
3. **The ring holds TARGET state.** Each stage owns a sliding-window ring,
   filled from the target's hidden states at the layers
   `dspark_target_layer_ids` names, combined and projected as the reference
   does it. Read from the reference whether a ring takes each layer's
   **input or output**, and how parallel residual streams are reduced: that is
   a per-checkpoint contract, and another engine's choice for a different
   checkpoint is no evidence. The block's own K/V are transient and never
   enter the ring. After a multi-token acceptance, every newly committed
   position's target state must reach the ring before the next draft, not
   just the last one.
4. **The Markov head is applied sequentially.** After the parallel pass, for
   each position in order: add a low-rank (`dspark_markov_rank`) bigram bias
   computed from the *previous* position's token, then choose this position's token. Applied in
   parallel it would condition on NOISE tokens.
5. **The confidence head only schedules.** It predicts per-position acceptance
   probabilities so a scheduler can choose how many drafts each request
   verifies. It never changes which tokens are accepted, so greedy output is
   identical with or without it. TRT does not wire it — `DSparkDecodingConfig`
   omits the knobs on purpose and the worker drafts with a zero threshold —
   so the full block is always verified.

**The acceptance ceiling is `block_size + 1`** — 6 at block size 5 — since
verifying k drafts accepts at most k plus the bonus token. Set
`accuracy.acceptance.dspark.min_absolute` against that ceiling, never against
what the target happens to score.

### What exists in TRT, and where the line falls

| piece | where | owner |
|---|---|---|
| `SpeculativeDecodingMode.DSPARK` | `tensorrt_llm/_torch/speculative/interface.py` | runtime |
| `DSparkDecodingConfig`, and its `TorchLlmArgs` validation: an unset `speculative_model` means the target's own `mtp.*`, and `block_size` must equal `max_draft_len` | `tensorrt_llm/llmapi/llm_args.py` | runtime |
| `DSv4DSparkWorker`: the drafting loop, the per-request rolling windows, strict or rejection-sampling acceptance, back-filling accepted positions | `tensorrt_llm/_torch/speculative/dspark.py` | runtime |
| `DSparkSpecMetadata`: captures the target layers' hidden states for the worker | `tensorrt_llm/_torch/speculative/dspark.py` | runtime |
| the three stages' forward, the ring-write projection, the Markov chain | the target | **target** |
| what the worker reaches through on the draft container: `block_size`, `num_stages`, `_attn_params`, `write_context_windows`, `write_context_windows_batched`, `forward_batched` | the target | **target** |

**The draft forward must be the target's own; the runtime owns the loop.** The
stock embedded draft is `DSv4DSparkDraftModel` in
`tensorrt_llm/_torch/models/modeling_dspark.py`. Read it for the interface the
worker expects, never to import or to copy — the rule `runtime-boundary.md`
states for MTP, for the same reason. And do not write a drafting loop, an
acceptance rule or a window manager in the target: those exist.

### Gate spelling

`features: [dspark]`. The boot and acceptance gates emit
`--spec_decode_algo=DSPARK --spec_decode_max_draft_len=<N>`, and `N` comes
from the unit's options YAML — `speculative_config.max_draft_len`, or
`block_size` — never from a default: quickstart builds the config from that
flag alone, and a depth of 1 against a drafter trained for a 5-token block
fails validation before the engine starts.

```yaml
# accuracy.llm_api_options.dspark
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
```

For a new architecture the acceptance gate's `=off` leg cannot load at all, so
declare `accuracy.acceptance.dspark.min_absolute` (1.2 is the smoke floor
vLLM's DeepSeek-V4 DSpark recipe uses) and the gate records the weaker,
absolute claim.

### Open, read from source and not measured

- The `TorchLlmArgs` DSpark validation reads `dspark_*` at the top level of the
  drafter's `config.json`, or inside `dspark_config` / `dflash_config` — **not
  under `text_config`**, where a multimodal checkpoint may keep them. So
  `target_layer_ids`, `mask_token_id` and `markov_rank` stay unset unless the
  spec states them, and with `target_layer_ids` unset `DSparkSpecMetadata`
  captures nothing: the drafter sees no target context and acceptance falls
  toward 1.0, which the acceptance gate reports as a broken draft path. The
  accuracy gate's options YAML can state them; `quickstart_advanced.py`, which
  the boot and acceptance gates drive, cannot. Decide how they reach the
  runtime before the first DSpark boot, and record the decision.
- TRT's own DSpark tests run with chunked prefill, the overlap scheduler and
  block reuse all off. A unit gated with them on is the first to exercise
  that combination; a failure there is not automatically the target's.
