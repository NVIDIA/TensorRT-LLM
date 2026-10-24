# One-engine speculative decoding — the runtime/target boundary

> Extracted from the former `trtllm-runtime-integration.md` §13; the section
> number is kept so existing citations still resolve.

## 13. One-engine speculative decoding — what changes when the target drafts

Everything above describes a target that emits **one** token per generation
sequence per step. A speculative target emits `1 + draft_len`, and the draft
tokens come from a second forward that the target itself owns. This section
is the axis that brings: what the engine looks for on the model object, what
the drafting loop calls, where the ownership line falls, and what a rank has
to do differently inside its forward.

Read like §12, in two layers. The **binding surface** applies to every
one-engine speculative mode. The **per-step consequences** below it were
established for **MTP-Eagle one-model** (`MTP_EAGLE_ONE_MODEL`) on an MLA +
EP-MoE checkpoint at `dep4`, and are marked where another mode differs.
`.claude/skills/trtllm-modeling-v2-speculator/references/multi-token-prediction.md`
carries the other half — what an MTP layer computes, which is a checkpoint
fact rather than a runtime one.

### The checkpoint picks the mode; the target does not

`update_spec_config_from_model_config` runs **before the model is built**
and reads the MTP layer count out of the pretrained config
(`num_nextn_predict_layers`, or `mtp_num_hidden_layers` on Qwen3Next-style
configs; 1 if neither is present). `MTPDecodingConfig`'s defaults are
`use_mtp_vanilla=False` and `mtp_eagle_one_model=True`, so:

| checkpoint layer count | resulting mode |
|---|---|
| `n == 1` | **`MTP_EAGLE_ONE_MODEL`** — one layer of MTP weights, replayed |
| `n > 1` | `MTP` (vanilla) — one distinct layer per draft position |

**Under MTP-Eagle, `max_draft_len` is not bounded by the checkpoint's layer
count.** That bound belongs to vanilla MTP. The single MTP layer is replayed
autoregressively `max_draft_len` times, so "how many MTP layers do we turn
on" is the wrong question and "what is `max_draft_len`" is the right one.

**Spell `max_draft_len` out in every variant.** Left unset on the
MTP-Eagle path it resolves to **1**, not to anything derived from the
workload. `max_total_draft_tokens` is then mirrored from it (linear tree),
and `tokens_per_gen_step = 1 + max_total_draft_tokens`.

### The target owns its spec worker

The causal-LM shell builds the worker and calls it from its own `forward`;
the engine does not look for one. The dep4 target is the working example:
`tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/modeling.py::get_spec_worker`
is called at construction with the run's `spec_config`, and the shell's
`forward` returns `self.spec_worker(...)` with its own `draft_model`
(`modeling.py::DraftModel` in the same target). The factory is
`tensorrt_llm/_torch/speculative/utils.py::get_spec_worker`. Everything else
the runtime touches on the model side, it reaches through the worker's own
arguments:

| attribute / callable | type | what reads it |
|---|---|---|
| `model.spec_worker` | `SpecWorkerBase` | the shell's own `forward` |
| `model.config` | pretrained config | `update_spec_config_from_loaded_model` (the base shell already provides it) |
| `model.draft_config` | — | read with `getattr(..., None)`; **absent is correct** for a single-checkpoint MTP target |
| `draft_model.mtp_layers` | `nn.ModuleList` | only `[0]` is ever indexed — MTP-Eagle replays one layer |
| `draft_model.embed_tokens` | module | passed to the layer as a kwarg |
| `draft_model.lm_head` | module | passed to `shared_head` |
| `draft_model.model.d2t` | — | read with nested `getattr(..., None)`; **absent is correct** (draft and target share a vocabulary) |
| `mtp_layers[0](...)` | callable | the draft loop, once per draft step |
| `mtp_layers[0].shared_head(h, lm_head, attn_metadata, True)` | method | returns draft logits |

`draft_model` is a container the target defines and hands to the worker; the
runtime never constructs it and never inspects it beyond the four names
above.

The layer is called by keyword, with `inputs` splatted in:

```python
hidden_states = draft_model.mtp_layers[0](
    embed_tokens=draft_model.embed_tokens,
    all_rank_num_tokens=<per-step, see below>,
    input_ids=..., position_ids=..., hidden_states=...,
    attn_metadata=..., spec_metadata=...,
)
```

It returns **one tensor**, `[rows_this_step, hidden]`, unpadded — the same
row count its `input_ids` carried. The loop slices it with its own
`gather_ids` afterwards. (Eagle3 one-model returns a second tensor here;
MTP-Eagle does not.)

### Where the ownership line falls

| responsibility | owner |
|---|---|
| accept/reject, rejection sampling, the golden token | runtime |
| KV rewind, `attn_metadata` rewrite between draft steps and its restore | runtime |
| the draft loop, `gather_ids`, position shifting, sampling draft tokens | runtime |
| `runtime_draft_len` scheduling and padding, `(bs, draft_len)` graph capture | runtime |
| the KV pool's extra layer and extra tokens | runtime |
| **the MTP layer's forward** | target |
| **`shared_head`** | target |
| **the `draft_model` container** | target |
| **the shell's speculative branch** | target |
| **loading the MTP layer's weights** | target |
| **attention-DP padding inside the MTP layer** | target |

### Inheriting the in-tree one-engine shell is not the shortcut it looks like

`SpecDecOneEngineForCausalLM.__init__` builds its drafter by calling
`get_draft_model(...)`, which is a **module-level function, not a method** —
a subclass has no override point. It dispatches on the config's
`model_type`, and the checkpoint is read exactly as published — there is no
target-owned `config.json` to rewrite — so `model_type` names the upstream
family and the call returns **trtllm's own MTP layer**. Inheriting therefore
hands the one computation this project exists to write to the engine instead.

This is a statement about what gets constructed, not a rule against
inheriting: a shell already inherits `DecoderModelForCausalLM` from the same
package. Read an in-tree implementation to understand an interface; never to
copy a structure. Writing the branch out (about 40 lines) keeps the forward
readable end to end, which is what the Vocabulary table and the
closed-vocabulary audit both rest on.

### The shell's shape, and the four things it has to get right

```python
class ModelingV2<Checkpoint><Arch><Par>(DecoderModelForCausalLM[<Family><Par>ModelingV2, ...]):
    def __init__(self, model_config):
        ...                                    # unchanged
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            assert self.spec_config.spec_dec_mode.is_mtp_eagle_one_model()
            self.draft_model = <the target's own container>(...)
            self.spec_worker = get_spec_worker(self.spec_config, model_config,
                                               model_config.mapping)

    def forward(self, attn_metadata, **kw):
        if self.spec_worker is None:
            assert kw.get("spec_metadata") is None
            return super().forward(attn_metadata, **kw)      # bit-identical
        spec_metadata = kw["spec_metadata"]
        hidden = self.model(attn_metadata=attn_metadata, **kw)
        logits = self.logits_processor.forward(
            hidden[spec_metadata.gather_ids], self.lm_head, attn_metadata, True)
        return self.spec_worker(
            input_ids=kw["input_ids"], position_ids=kw["position_ids"],
            hidden_states=hidden, logits=logits,
            attn_metadata=attn_metadata, spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kw.get("resource_manager"))
```

The class is the one `routing.py`'s `route` returns for this (checkpoint, GPU
architecture, parallel topology) triple; its name carries the triple, and
`test_modeling_v2_claims.py` asserts it agrees with the target's path.

`get_spec_worker` is imported from `tensorrt_llm._torch.speculative` — the
**runtime**, the part of the stack this project reuses, not modeling.

Four things this shape is load-bearing about:

**The non-speculative path must be a delegation, not a reimplementation.**
A target's release criterion was measured on the inherited base forward; the
only way to keep it bit-identical is to call it. Declaring
`resource_manager` as a named parameter would silently drop it from `**kw`,
so leave it in the dict and pull it with `.get` in the speculative branch
only — then the base receives exactly what it receives today.

**The shell gathers the logits; the engine does not.** For every one-model
mode `without_logits` is True, so `_forward_step` returns the model's dict
verbatim and applies no second gather. `spec_metadata.gather_ids` holds one
row per context request (its last token) and `runtime_draft_len + 1` rows
per generation request — pass `hidden` **ungathered** to the worker and the
gathered logits alongside it.

**`position_ids` reaches the worker in the engine's `[1, T]` shape.** The
worker does `position_ids.squeeze(0)` itself. A shell that flattens before
handing it over produces a silently wrong draft position sequence.

**Nothing needs adding to `epilogue`, and nothing needs a `layer_idx`.**
`epilogue` is only consulted by `__pp_init__`'s `skip_forward`, so at
`pp_size == 1` there is nothing to register. And on this mode
`Eagle3OneModelSpecMetadata` sets `layers_to_capture = ()`, which makes
`is_layer_capture()` False at every layer and leaves `hidden_states`
unallocated — the trunk owes the runtime **no hidden-state capture hook**
(that is Eagle3's requirement, not MTP-Eagle's). Measured: nothing in
`_torch/pyexecutor/` or `_torch/speculative/` reads `model.layer_idx`.

### Draft length is per iteration, not per request

`_handle_dynamic_draft_len` runs **before** `prepare_resources`, so KV
allocation already knows the answer:

1. `draft_len_schedule` maps a batch-size threshold to a draft length.
2. The current `scheduled_batch.batch_size` selects `runtime_draft_len`.
3. Every generation request's `py_draft_tokens` is padded or truncated to
   **exactly** that length — the source comment names CUDA-graph replay and
   the attention kernel as the reasons.
4. It lands on `spec_metadata.runtime_draft_len`;
   `runtime_tokens_per_gen_step = 1 + runtime_draft_len`.
5. `runtime_draft_len == 0` takes `skip_drafting`, i.e. speculation is off
   for that iteration only.

**For modeling this means there is no ragged draft tree to handle.** The
draft length is a host int, constant across the batch, constant within a
capture. `cuda_graph_runner.get_graph_key` asserts it directly: *"All draft
lengths must be the same"*.

Without a schedule, `runtime_draft_len` is simply `max_draft_len` every
step.

**`draft_len_schedule` deadlocks under attention data parallelism, and
nothing rejects the combination.** Step 2 above reads
`scheduled_batch.batch_size` — **each rank's own local batch** — with no
cross-rank reduction, and attention DP does not equalize batch sizes:
`_pad_attention_dp_dummy_request` only tops a rank up from zero to one, and
`attention_dp_config.enable_balance` is off unless asked for. Two ranks
either side of a schedule threshold therefore run **different numbers of
draft replays**, hence different numbers of MoE collectives — and the
collectives pair by position, so the job hangs.

Measured at world size 4: boot, CUDA-graph capture and warmup all pass, and
the hang needs real traffic to make the rank batch sizes diverge. trtllm's
own `HangDetector` fired at 300 s and hard-killed all four ranks, whose
stacks sat at three different points of one forward. **On an attention-DP
target, skip this knob.** With `tp`/`ep` alone the ranks share one batch and
the mechanism is sound.

Setting it also **silently turns on `cuda_graph_config.enable_padding`** —
logged at INFO only — so it is never a single-variable experiment.

### What changes in the trunk's own forward

**A generation request arrives with `runtime_draft_len + 1` query tokens
instead of 1**, and the total-token arithmetic is
`num_ctx_tokens + (num_seqs - num_contexts) * tokens_per_gen_seq`.

**How that reaches the attention op depends on the configuration, and the two
cases do not share a mechanism.** On an **MLA** layer it is
`predicted_tokens_per_seq`. In the **standard (non-MLA) configuration it is not**:
the op derives rows per generation sequence from `num_gen_tokens / num_gen_seqs`
and `predicted_tokens_per_seq` is inert — it reaches the op only through
`MLAParams`, which `create_attention` does not build for a non-MLA layer, so the
engine passes 1 while making the query block taller. Measured by sweeping it at
1, 2, 3, 4 and 8 against generation row counts 1–4: `output` and the pool came
back **bitwise identical** to the `P = 1` run at the same row count, including
where the two disagreed outright.

Two consequences. The op **checks the division** — a ragged generation batch
raises `seq_len should be same for all generation requests`. And setting
`predicted_tokens_per_seq` is harmless but does nothing on this path, so a target
that sets it and sees a taller block has not learned that the two are connected.

*Verified on sm_100. There is no version pin to re-check against: each
target's `step_args` is the probe its opt-in step-contract check calls, so a
renamed metadata field fails loudly on the first forward under
`TRTLLM_MODELING_V2_VALIDATE=1`, and a renamed op fails at its call. Re-check
this particular behaviour by sweeping the argument against a fixed row count
and comparing output bitwise; the attention entry's `reference` and `CELLS`
are the authority for what the resulting block attends to.*

**Those extra query tokens stay on the generation call.** A one-engine mode
returns False from the runtime's `extend_ctx` predicate — "1-model has
separate logic for handling draft tokens" — so a generation request carrying
drafts is *not* re-shaped into a chunked context request the way two-model
speculation does it. The batch keeps its `[context | generation]` split and
the generation call simply gets a taller query block. What that block is
allowed to attend to — each draft position seeing the cache plus the earlier
positions of its own block, and no later one — is a **kernel** fact, so the
attention entry's executable contract (`reference`, `CELLS`) is its
authority, not this file.
Do not assume it from `mask_type` alone.

**The spec-dec mask machinery is forced off at sm_100 and stays inert.**

`tensorrt_llm/_torch/attention/backends/trtllm.py::is_spec_decoding_enabled` is
set in the attention metadata's spec-dec setup:

```python
# Blackwell trtllm-gen spec-dec is enabled only for dynamic-tree masks.
self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
    not self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()) or is_spec_dec_dynamic_tree
)
```

`is_sm_version_trtllm_gen_kernel(sm)` is `not (sm < 100 or sm in [120, 121])`,
so it is True on sm_100; a linear-tree MTP has `is_spec_dec_dynamic_tree`
False; the conjunction is **False**. `is_spec_decoding_enabled`,
`use_spec_decoding` and `is_spec_dec_tree` are all False and every
`spec_decoding_*` tensor is None — the same inert values the MLA cells of
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/thop_attention.py`
already certify. **On a pre-Blackwell arch this is not true** and the mask
surface would need certifying first.

The context path is unchanged: a context request still contributes its
prompt, one row of logits, and one attention call of the same shape.

### The draft loop rewrites the batch between step 0 and step 1+

The single most important fact for writing the layer. After the first draft
step the worker mutates `attn_metadata` in place:

| | step 0 | step 1+ |
|---|---|---|
| `_seq_lens` / `_seq_lens_cuda` | real | **filled with 1** |
| `num_contexts` | real | **0** (when a KV cache manager is present) |
| `num_ctx_tokens` | real | **0** (recomputed) |
| `host_request_types` | mixed | context entries overwritten to generation |
| tokens per generation request | `runtime_draft_len + 1` | **1** |
| a context request | the whole prompt | **1 token** |
| `use_spec_decoding` | as the engine set it | False |
| `kv_lens_cuda` | as the engine set it | rewound, then `+1` per step |

**So the layer must read its phase from `attn_metadata` on every call.** It
is invoked N times inside one forward, and a value computed on the first
call is wrong on the rest. This is not the trunk's situation — the trunk
routes and projects `step_args` once per forward because there is only one
step in it. The layer therefore routes per call: its `forward` is
`core._select_target(md).mtp_forward(...)`, and each target carries its own
`mtp_forward`, making only the calls its phase needs (on the deepseek
exemplar, decode's never makes the context attention call, the kv_b
projection or the two paged-KV helper calls — they serve a path its routing
guarantees never runs). One engine step's draft calls can cross from
prefill to decode, and each call asks again.

The read is cheap and sync-free. `on_update()` recomputes `_num_ctx_tokens`,
`_num_generations` and `_num_tokens` from `_seq_lens`, which is a **pinned
host** tensor, and the loop calls it (and triggers it again through the
`num_contexts` setter) at exactly the step-0/step-1+ boundary. Therefore

```
tokens_per_gen_seq = (rows - md.num_ctx_tokens) // (md.num_seqs - md.num_contexts)
```

evaluates to `runtime_draft_len + 1` on step 0 and to exactly `1` on step
1+, with no step counter threaded through and no device read. That is the
**generation row count** the layer's own call needs — on a standard layer that
is what the taller query block *is*, not something you pass as
`predicted_tokens_per_seq` (see above). Guard the all-context case
(`num_seqs == num_contexts`) rather than dividing by zero.

**What a context request is fed on step 0** is `prompt[1:]` with the
request's first accepted (golden) token written at its last position —
`_prepare_context_input_ids`, shared by both MTP flavours. Its hidden states
are the trunk's, in full, at full prompt length. So step 0 runs a real
context-phase attention for those rows and the layer needs both phases,
exactly like the trunk — step 0 of a batch holding a context request runs
`PrefillTarget.mtp_forward`, and steps 1+ (`num_contexts` rewritten to 0) run
`DecodeTarget.mtp_forward`.

### Attention DP: the padding basis changes source, and reading the wrong one is silent

Under `enable_attention_dp`, §12 established that every rank pads its token
block to `max(attn_metadata.all_rank_num_tokens)` so both collectives run in
their uniform form. **Inside the draft loop that list is the wrong one from
step 1 onwards.**

The worker passes the right basis in **as a keyword argument** and leaves
`attn_metadata.all_rank_num_tokens` holding the trunk's value for the whole
loop (it saves and restores it around the loop, but does not maintain it
during it):

| draft step | `all_rank_num_tokens` kwarg |
|---|---|
| 0 | `spec_metadata.all_rank_num_tokens` — the trunk's token counts |
| 1+ | `spec_metadata.subseq_all_rank_num_tokens` — the per-rank **sequence** counts |

`subseq_all_rank_num_tokens` is set to `all_rank_num_seqs` by the engine for
the one-model modes, which is semantically right: from step 1 every sequence
contributes exactly one token.

**Pad from `md.all_rank_num_tokens` inside the MTP layer and the collectives
mispair silently.**
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/comm/allgather.py` and
`.../catalog/comm/reducescatter.py` both certify that calls pair by *position*
on the communicator and that at equal byte counts a divergence produces no
hang — every rank wrong in 98–99% of elements, bitwise reproducibly. The
defence is structural: give the layer's padding helper a signature that
**only accepts a passed-in list**, so it has no way to reach the metadata.

(A `_dp_rows` that asserts `all_rank_num_tokens[rank] == rows` would in fact
fire here rather than corrupt, since the two counts differ from step 1. Do
not rely on it: the assert is a property of one target's helper, not of the
rule, and it is exactly the kind of guard a later edit removes.)

### KV cache: the engine adds the layer, and the target declares nothing

Under a one-model MTP mode `ModelConfig` raises the pool's layer count
itself:

```python
num_layers += spec_config.num_nextn_predict_layers
num_attention_layers += spec_config.num_nextn_predict_layers
```

so the pool is sized for `num_hidden_layers + n` layers and the MTP layer's
own attention addresses layer index `num_hidden_layers`. **Nothing in the
target's config or manifest declares any of this** — but the layer's
attention calls must pass the right `layer_idx`, and the pool-addressing
surface they use has to be certified at the new layer count.

**A `speculative_config` also raises `max_seq_len`, and not by the amount the
extra-KV-token helper suggests.** Three separate terms are added to the model
engine's `max_seq_len`, in `py_executor_creator`:

```python
if not disable_overlap_scheduler and spec_config is not None:
    max_seq_len += spec_config.tokens_per_gen_step - 1
if spec_config is not None:
    max_seq_len += get_num_extra_kv_tokens(spec_config)  # max_draft_len - 1
    max_seq_len += spec_config.tokens_per_gen_step - 1
```

With a linear tree (`tokens_per_gen_step = 1 + max_draft_len`) and the overlap
scheduler at its default (**on**), that is `3 * max_draft_len - 1` — 2, 5 and
8 at `max_draft_len` 1, 2 and 3. Measured: a `163840` model cap becomes
**`163848`** at `max_draft_len: 3`.

**So read `max_seq_len` off the metadata, never compute it.** A target that
derives anything from the config's own `max_position_embeddings` — a rope
table sized to it, a bound asserted against it — is off by that amount the
moment speculation is switched on, and by a different amount per
`max_draft_len`. The failure is a silent out-of-bounds read on a rope table,
or an assert that fires on a legal config.

### One more construction-time trap, from §4's weight lifecycle

**A reference to another module's parameter, captured in `__init__`, stays
bound to the meta-device shadow.** The engine materializes at t1 by
*replacing* tensor objects, not by filling them in place, so a drafter
container that does `self.embed_tokens = core.w["embed"]` at construction
holds a meta tensor forever and fails at the first draft step. §4 already says
an unregistered parameter "stays a dead shadow"; this is the adjacent case —
the parameter *is* registered, on the trunk, and the copy of the reference is
what goes stale. Resolve it lazily (a `property` that reads through to the
trunk on each access) rather than caching it.

### CUDA graphs: the whole draft loop is inside the capture

Capture wraps `_forward_step`, which calls the shell's forward, which calls
the worker — so every MTP-layer invocation is captured. The key is
`(batch_size, draft_len, is_first_draft, short_seq_len_mode,
is_all_greedy_sample)`, and the capture set becomes `(bs, draft_len)` pairs
rather than bare batch sizes; with a `draft_len_schedule` the runner also
captures one extra `(max_bs, original_max_draft_len)` graph, whose stated
purpose is to keep a later graph from resizing the shared attention
workspace and invalidating pointers baked into earlier ones.

Consequences for the layer, all of them §7's discipline applied one level
down:

* The step-0 / step-1+ divergence is **re-traced per draft step at capture
  time** and frozen at each step's position in the loop. Reading the phase
  from `attn_metadata` every call is what makes that correct — a cached
  first-call decision would be baked in at all N positions.
* Host ints read from `attn_metadata` (`num_contexts`, `num_ctx_tokens`) and
  the `all_rank_num_tokens` kwarg are per-capture constants, which is the
  host-derived class §7 permits.
* `attn_metadata.padded_num_tokens` is **`None`** unless torch-compile
  piecewise CUDA graphs are configured; without a `torch_compile_config` the
  padding path is not reachable and the base shell's slice never applies.
* The attention-DP replay gate of §12 is unchanged: all ranks
  generation-only and equal batch sizes, or the step runs eager — and eager
  is where the layer meets a real mixed batch.

### The acceptance rate is the only correctness signal

`stats.specdec_stats.acceptance_length` is computed per iteration over the
generation requests that carried draft tokens:

```
acceptance_length = (accepted_draft_tokens + requests_with_draft) / requests_with_draft
```

i.e. the mean number of tokens a drafting request produces per step, `1.0`
meaning every draft was rejected.

**This matters more than it looks.** Rejection sampling guarantees the
output distribution is unchanged, so a *miscomputed* MTP layer does not
produce wrong text — it produces correct text more slowly, with every draft
rejected. An accuracy gate cannot see it. A subtly wrong layer (a
transposed concatenation, a norm applied to the wrong operand) lands
somewhere above 1.0 and well below the reference, which no other measurement
distinguishes from "this model is just hard to draft for". The reference is
the same checkpoint under stock in-tree modeling at the same load and the
same `speculative_config` — the same job with `modeling_v2=off`
instead of `=require`; acceptance is a property of the model, so a
reference forced onto a different `max_num_tokens` / `max_seq_len` to boot
is still a valid comparison.

### Still not established here

Vanilla MTP (`n > 1`, one layer per draft position) and its per-layer
sequential loop; Eagle3 in either form, including the hidden-state capture
hook and `apply_eagle3_fc`; tree drafting of any kind (`use_dynamic_tree`,
`dynamic_tree_max_topK`), which is also the only way the Blackwell spec-dec mask
surface becomes reachable; two-model speculation; and speculative decoding
combined with pipeline parallelism, where `skip_forward` and the `epilogue`
path start to matter.
