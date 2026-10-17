# What varies per model — the re-derive list

> Extracted from the former `trtllm-runtime-integration.md` §10; the section
> number is kept so existing citations still resolve.

## 10. What varies per model — the re-derive list

The exemplar target shows the pattern; every model-specific value must be
re-derived from the new checkpoint's `config.json` and the catalog
entries' cells, never copied. The known variation axes:

**attention structure** — GQA/MHA vs **MLA**, the axis that reshapes the
most: MLA replaces per-head K/V with one compressed latent, needs two
attention calls per forward (context and generation are separate call
shapes, mixed batches rejected upstream), and brings its own load-time
obligations. It is not a variant of the geometry row below; see
`.claude/skills/trtllm-modeling-v2-kernel-onboard/references/decompositions/mla-custom-op.md` for the vocabulary and
`.claude/skills/trtllm-modeling-v2-core/references/latent-kv-cache.md` for what a target had to derive.
Then: geometry (layers, heads, kv-heads, head_dim, intermediate, vocab);
qk-norm presence and its eps; rope kind, theta, scaling, `is_neox`,
partial-rotary factor; `tie_word_embeddings` (changes the lm_head/embed
feeding and the shell's tying path); attention bias (needs cublas_mm's
fused-bias argument); sliding window; activation function; checkpoint
key names and fusion grouping; dtype and quantization scheme — including
*which modules it excludes*, since a checkpoint may quantize only its MLP
path and leave attention bf16; parallel topology (the routing decision and
the boot gate's topology flags + per-rank shapes + manifest src transforms +
collective-communication catalog entries — §12 has the launch substrate, the
rest is the first multi-rank target's to derive); KV-cache dtype (turns the
kv-scale constants into post-load tensors — certification extension first).

A construction-time assert exists for each axis the assembly depends on:
if the new config violates one, the right response is to re-derive that
part of the assembly, not to delete the assert.

### Read the axes off the engine's config object, not off AutoConfig

`model_config.pretrained_config` — the object the engine hands
`__init__` — is the **un-migrated** config: fields sit where the
checkpoint's own `config.json` put them. A standalone
`AutoConfig.from_pretrained(model_dir)` probe can return a *different*
shape of the same information, because transformers migrates fields
across versions.

Observed at transformers 5.5.4 on a checkpoint written by 4.51: through
AutoConfig, rope had been migrated into
`rope_parameters = {'rope_theta': ..., 'rope_type': 'default'}` with
`rope_scaling` an alias of that same dict, and `cfg.rope_theta` raising
`AttributeError`. Through the engine, `rope_parameters is None`,
`rope_scaling is None`, and `rope_theta` is a plain instance attribute.
Deriving the axis from the AutoConfig surface produced a target that
failed at construction with `TypeError: 'NoneType' object is not
subscriptable`.

So: an AutoConfig probe is **not** a valid oracle for what
`pretrained_config` will look like. Read the checkpoint's `config.json`
directly to learn what the model *is*, and use the flat
`cfg.<field>` idiom inside the target. If a probe is needed, instrument
the target's own `__init__`.

### A missing `dtype` is not inherited

A checkpoint may declare **no** `dtype` and no `torch_dtype` at all
(observed on gpt-oss-120b, whose weights are all BF16 outside the
quantized expert blocks). The two config surfaces then disagree:
`model_config.pretrained_config.torch_dtype` is `None` while the engine's
own `ModelConfig.torch_dtype` resolves `torch.bfloat16`. The shell reads
the **pretrained** one, so `DecoderModelForCausalLM` materializes its
shell-owned parameters — `lm_head.weight` — as **fp32**, and a target with
a dtype-checking weight manifest fails there with a cause two layers
upstream of the symptom.

There is no longer a target-owned stub to fix it in: **the checkpoint is
read exactly as published**, so the divergence cannot be patched at the
config and the target has to resolve the dtype itself. Assert both surfaces
in `__init__` so a future drift is loud.
<!-- PORT: verify — out of tree the fix was a `"dtype": "bfloat16"` line in
     the target-owned `model_dir/config.json` stub, which no longer exists
     (modeling_v2's README: "The checkpoint is read exactly as published, with
     no target-owned config.json"). What the in-tree remedy is — resolve from
     `ModelConfig.torch_dtype`, or cast in the weight manifest — is not
     established anywhere I could read. The observation is kept verbatim; the
     remedy needs an owner. -->

### Where the chat template lives is not a `tokenizer*` glob

An accuracy protocol that applies a chat template needs the template
files, and a recent checkpoint may keep **no** `chat_template` inside
`tokenizer_config.json` — it lives in `chat_template.jinja` (plus
`chat_template.json`). Those names, and `special_tokens_map.json`, match
no `tokenizer*` pattern.

The old failure mode was a link step: a target-owned `model_dir/` populated
by a fixed `tokenizer*` glob dropped the template, and the gate failed at
template application after a full engine build. That step is gone — the
checkpoint is read exactly as published, so every file it ships is already
there. What survives is the reason the glob was wrong: **inspect what the
checkpoint actually ships** when naming the protocol's template, rather than
assuming where a template lives.
