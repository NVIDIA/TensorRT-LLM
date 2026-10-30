# Hybrid recurrent-state targets (linear attention beside full attention)

For a checkpoint whose layers are a mix of `full_attention` and a recurrent
mixer — Gated DeltaNet, Mamba2, KDA — where the recurrent layers need a
**conv + recurrent state pool** the runtime allocates, alongside the ordinary
paged KV cache for the attention layers.

Sibling of `latent-kv-cache.md`, which is MLA's cache mechanism; this one is
the recurrent-state mechanism. Both describe what a target must do so the
runtime allocates the right pool.

## The runtime decides from the published config, not from the weights

Whether a recurrent-state pool exists at all is chosen before any weight is
read, from `model_config.pretrained_config` — **the checkpoint's own config, read
exactly as published**. If the predicate does not match, the model still builds,
the boot gate still passes, and the recurrent layers have no state.
<!-- PORT: verify — out of tree this section said the predicate keys on the
     architecture string. In tree `is_nemotron_hybrid` reads
     `hybrid_override_pattern` off the config, not `architectures`, so which
     field the hybrid predicate actually tests was not re-confirmed here. What
     is confirmed is that it reads the published config. -->

A target can neither change nor restate what that predicate sees. Routing
returns the target class itself and the loader carries it on
`ModelConfig.modeling_v2_target` — `pretrained_config.architectures` is not
mutated — so nothing a target names has any influence on what the runtime
allocates, and there is no stub `config.json` to patch. If a checkpoint the target is meant to serve does not trip the predicate,
that is a fact about the checkpoint and the runtime: **report it as a symptom**,
and do not try to route around it from modeling code. `runtime-interface.md` §2
has the resolution path and the reasoning.

## A nested `text_config` is not reliably mirrored

Composite VLM checkpoints keep the text fields under `text_config`. The runtime
has a mirroring pass that copies `text_config` attributes onto the parent, but
relying on it is a trap: it only fills in attributes that raise
`AttributeError` on the parent, so any key for which the config class already
supplies a default is silently **not** mirrored, and the text value is ignored.

Out of tree a target sidestepped the question by shipping a flattened stub.
There is no stub now, so the trap is live: a field the target's geometry
depends on may be read by the runtime from the class default rather than from
`text_config`. Read each such field from where the runtime will read it, assert
it at construction, and report any field whose live value differs from
`text_config`'s as a symptom rather than compensating for it in the forward.

Fields the hybrid path reads off the config: `architectures`,
`layer_types` (validated against `num_hidden_layers`, and every entry must be
one of `full_attention` / `linear_attention`), plus the recurrent geometry the
cache manager sizes the pool from — for the Qwen3-family naming, the key head
dim, conv kernel dim, and the value/key head counts and dims.

## The state-cache dtype field is not the one the checkpoint declares

A checkpoint may carry a field expressing the recurrent **compute** dtype
intent — Qwen3.5-family configs write `mamba_ssm_dtype: float32`. That is not
the cache dtype and the runtime deliberately ignores it. Only an explicit
`mamba_ssm_cache_dtype` (config field or `kv_cache_config`) is honored, falling
back to the model dtype.

**Do not set it under the cache name.** Doing so switches the
state cache to fp32, which disables the bf16-state decode kernel and doubles
state memory traffic — upstream's own comment puts that at roughly 20% serving
throughput. This is a performance loss with no gate that can see it: the boot
gate passes, accuracy passes, and only a sweep would show it.

## The layer's cache offset is not its attention ordinal

On a hybrid model the attention layers are a **sparse subset** of the layer index
space, and how they map onto pool offsets depends on which cache manager the
runtime picked. Under the default `CppMambaHybridCacheManager` the mask handed to
`KVCacheManager` covers recurrent *and* attention layers, so full-attention layer
3 sits at pool offset 3 — not at attention-ordinal 0, which is what a dense model
would give and what a `MixedMambaHybridCacheManager` would give here.

So **read the offset from `attn_metadata.kv_cache_manager.layer_offsets` at the
first forward**; do not compute it by counting attention layers. Getting it wrong
is cross-layer aliasing — layer A reading or writing layer B's pages — which
produces finite, plausible output and is invisible to a single-layer test.

The engine also runs **two pools** in this configuration: the attention KV pool
and the recurrent-state pool, with the recurrent one carrying a large negative
`window size` sentinel meaning "no window". The attention op never addresses the
recurrent pool — verified by decoy, pointing its pointer row at a spare
allocation leaves output bitwise identical — so that sentinel is not something a
target passes or reasons about.

## Block reuse is off, until a feature turns the same flag back on

The runtime **disables KV block reuse for a hybrid Mamba model** when no snapshot
policy is set, which is the default. Two consequences that look contradictory
until the flag is traced:

- a hybrid target's *core* runs with `use_paged_context_fmha = False` on every
  call, so any certification of the paged-context path is unexercised by it;
- but the flag is
  `chunked_prefill or cache_reuse or has_speculative_draft_tokens`, so **adding
  speculative decoding turns it on** even though reuse is still off.

That matters on a quantized KV cache, because the paged-context path sources K/V
from the pool: on an fp8 pool the append leaves `e4m3(row/s)`, so a fresh prefill
is not numerically identical across the flag. A feature increment can therefore
move the trunk's prefill numerics without touching the trunk's code.

## What the recurrent core op needs from the target

The recurrent mixer is reached through a **registered-layer** op, the same
shape as the registered attention entry point: the call carries the activations
and a `layer_idx` string, and the op resolves a module instance out of the
model's extra-attrs to find the conv weight, the per-head gating parameters and
the state pools. So the target constructs and registers that module rather than
holding those weights itself, and the op's arguments are the projections only.

Consequence for the closed-vocabulary audit: the conv, the gating arithmetic
and the recurrence are all *inside* one catalog entry, so they do not appear as
separate calls in the forward. The projections around it — qkv, the gate
projection, the two per-head scalar projections, the output projection — are
ordinary entries and do appear.

---

*Runtime behaviour above verified on sm_100. An engine change re-opens all of
it: re-check by reading the hybrid predicate the cache manager selects on, and
confirming the state-cache dtype resolution still ignores the checkpoint's
compute-dtype field. The consequences of reading the checkpoint exactly as
published — that the target neither restates nor patches what the predicate
sees — follow from routing, not from any engine version.*
