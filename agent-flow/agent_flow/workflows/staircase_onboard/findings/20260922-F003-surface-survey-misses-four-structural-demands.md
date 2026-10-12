# F003 — the surface survey reports four structural obligations as "unrecognized"

- **Found:** 2026-09-22 03:55 PDT, by running `survey.read_checkpoint_surface`
  against this checkpoint's `config.json` before preflight
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/survey.py`
- **Severity:** the plan drafter is handed prose where it should be handed a demand
- **Status:** reported; fix deferred until the architecture analysis confirms semantics

## How this was found

The survey is pure Python and takes a config mapping, so it can be run on a
checkpoint before committing a single GPU-hour:

```
.venv-orch/bin/python -c "
import json
from agent_flow.workflows.staircase_onboard import survey
cfg = json.load(open('<checkpoint>/config.json'))
s = survey.read_checkpoint_surface(cfg)
..."
```

That is worth knowing on its own: **the first half of preflight is a
CPU-only, seconds-long dry run of what the drafter will see**, and nothing
in the docs points at it as a standalone tool.

## What it got right

Ten demands, all correct, including two that matter a lot here:

```
speculation.heads: 3 in-checkpoint predictor layer(s) — weights exist for a
  speculator, so MTP is the checkpoint's own and not a bolt-on
multimodal.tower: vision, audio tower in front of the language model;
  placeholder tokens declared
```

## What it missed

Twenty-three keys came back `unrecognized`. Most of those are genuinely
inert (`processor_config`, the token ids, `layernorm_epsilon`). **Four are
not** — each is a structural obligation with a kernel consequence, and each
has a skill reference already written for it:

| key(s) | what it means | the skill that already covers it |
|---|---|---|
| `hybrid_layer_pattern` (48 entries), `attention_chunk_size` | the layer stack **alternates** local and global attention; the two kinds have *different* head counts | `staircase-modeling-core/references/sliding-window-attention.md` |
| `add_swa_attention_sink_bias`, `add_full_attention_sink_bias` | a learned sink term in the softmax denominator, on the local layers only | `staircase-modeling-core/references/attention-sinks.md` |
| `v_head_dim` (128) against `head_dim` (192) | **asymmetric qk / v head dimensions** — the single hardest kernel-feasibility question this checkpoint asks | `staircase-kernel-feasibility` |
| `swa_num_key_value_heads` (8) against `num_key_value_heads` (4), `swa_rope_theta` (1e4) against `rope_theta` (1e7) | the two attention kinds are different *shapes*, not the same shape with a mask | — |

The windowing demand that *is* emitted says `sliding_window=128` and nothing
else, which reads as "one sliding window over a uniform stack". That is the
wrong picture: it is a 48-layer stack where every fifth or sixth layer is
global, the global layers have 4 KV heads and the local ones have 8, and the
two use RoPE bases three orders of magnitude apart.

## Why this matters more than a missing line of prose

`survey.py`'s own comment is explicit that the unrecognized list is allowed to
be incomplete, and that its incompleteness "costs a line of prose rather than
a wrong plan". That reasoning holds for `processor_config`. It does not hold
for these four, because each one changes what the *vocabulary* must contain:

- asymmetric qk/v head dims decide whether any existing paged-attention entry
  can be reused at all, or whether the target needs a padded-V fallback;
- the sink bias decides whether the attention entry is the stock one;
- the alternating pattern decides whether the KV cache manager is asked for
  one pool or two, with different per-layer geometry;
- the differing KV head counts decide the weight manifest's shard math.

A demand the drafter does not see is a queue item the smith phase never
creates, and that surfaces as a `BLOCKED: vocabulary_gap` from the assembler
hours later — the expensive end of the run.

## Proposed change

Four new probes in `survey.py`, in the same shape as the existing ones:

1. `attention.hybrid` — fires on a per-layer pattern key
   (`hybrid_layer_pattern`, `layer_types`, `full_attention_idx`, ...). Reports
   the period, the counts of each kind, and whether the two kinds declare
   different head geometry (scanning for the `swa_*` / `*_local_*` prefix
   family against their global counterparts).
2. `attention.sink` — fires on any `*attention_sink*` key, reporting which
   layer kind carries it.
3. `attention.head_dim` — extend the existing probe so that a `v_head_dim`
   differing from `head_dim` is stated as its own clause, rather than being
   dropped. This is a one-line change to a probe that already exists.
4. `moe.routing` — extend to report `topk_method` (`noaux_tc` here) and
   `n_group`/`topk_group`, which decide the routing entry, not just the
   scoring function.

Also worth doing, and cheaper: add the prefixes `swa_`, `local_`, and
`full_attention_` to the "already accounted for" prefix list once probe 1
lands, so the unrecognized list shrinks to things that really are inert.

## Deliberately not proposed

Making the unrecognized list a failure. It is right that it is prose. The
change is to move four specific keys out of it, not to change what the list
means.
