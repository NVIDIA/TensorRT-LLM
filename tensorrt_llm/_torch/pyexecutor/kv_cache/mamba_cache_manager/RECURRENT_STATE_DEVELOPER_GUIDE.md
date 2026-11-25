<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recurrent State Developer Guide

Some model layers carry state from one step to the next instead of a KV cache: Mamba2
(Nemotron-H), GDN (Qwen3-Next, Qwen3.5, Qwen4Exp) and KDA (Kimi K3, GLM5Next). This guide
describes the contract between shared runtime code and such a model family: what the model class
declares, how the declaration is found, and what the family provides once a cache manager exists.

Code: `tensorrt_llm/_torch/recurrent_state_spec.py` (the declaration),
`tensorrt_llm/_torch/models/modeling_utils.py` (the model-class hooks and the `resolve_*` entries
that find them), `common.py` in this directory (`MambaState`, `MambaStateLayout`),
`mamba_cache_manager_v2.py` (the shared hybrid manager), `pyexecutor/_util.py`
(`get_kv_cache_manager_cls`, `_create_kv_cache_manager`), and the family modules under
`tensorrt_llm/_torch/modules/`.

## The contract

| Face | What it is | Owner |
|---|---|---|
| Declaration | The kind of every layer and, for a recurrent layer, its planes of base state: role, shape, dtype and how ranks split it. Plain data that depends on the configuration alone | The model class |
| Cache manager | The buffers of base state it allocates, the geometry resolved for one rank (`MambaStateLayout`) and its public methods | The V2 manager of the family |
| State | The family's algorithm over state slots (`MambaState`) and the memory it needs beyond the base state | The family module |

The declaration and the cache manager are built from the same configuration, so the buffers a
cache allocates agree with the declaration;
`tests/unittest/_torch/pyexecutor/kv_cache/test_recurrent_state_spec.py` holds the two together.

Not unified, and not part of the contract: mixers and kernels, how a family recovers after a
rejected draft, the layout of replay buffers, when a speculative worker reports acceptance, and
the contents of per-step metadata.

## Where shared code gets its answers

The declaration says, before a cache manager exists, which layers carry recurrent state and what
the geometry of their base state is: shapes, dtypes and the split across ranks. Shared code reads
it for these questions:

| Question | Reader |
|---|---|
| Geometry and dtypes of the base state | `extract_mamba_kv_cache_params` (`pyexecutor/config_utils.py`); see "The cache parameters of a declaration" below |
| Number of attention layers and of recurrent layers | `ModelConfig.get_num_attention_layers` and `ModelConfig.get_num_mamba_layers` |
| Whether the model is a recurrent architecture | `declares_recurrent_state` (`models/modeling_utils.py`), called in `pyexecutor/_util.py`, `model_engine.py` and `py_executor_creator.py` |
| Which V2 manager class holds the state | `get_kv_cache_manager_cls`, which asks the model class (`resolve_recurrent_kv_cache_manager_cls`) |
| How a recurrent manager is constructed | `_create_kv_cache_manager`; see "Construction" below |

The predicate `is_hybrid_linear` (`pyexecutor/config_utils.py`) is still read in three places:

- The entry of `get_kv_cache_manager_cls` and `_validate_and_adjust_mamba_snapshot_config` in
  `pyexecutor/model_loader.py`. Either can be given the composite config of Nemotron-H multimodal,
  Qwen3.5 VL or MiniCPM-V 4.6. The predicate does not recognise these configs while their wrappers
  declare state, so reading the declaration there changes the answer for these models.
- `tools/layer_wise_benchmarks`, which constructs its cache managers on its own.

The per-family predicates remain elsewhere in shared code, for example where the arguments of a
V1 hybrid manager are chosen ("Construction" below), and so do `isinstance` checks on the
recurrent manager classes. New shared code uses the declaration instead of adding to them. A
decision that depends on a built cache or on the loaded model is asked of that object instead; see
[What shared runtime code asks](#what-shared-runtime-code-asks).

## The declaration

| Type | Meaning |
|---|---|
| `RecurrentStateSpec` | One `LayerStateSpec` per target-model layer; the position is the layer index |
| `LayerStateSpec` | The layer's `LayerKind` and, for a recurrent layer, its planes |
| `LayerKind` | `ATTENTION` keeps a KV cache, `RECURRENT` carries state planes, `NONE` keeps neither (an MLP or MoE layer of Nemotron-H) |
| `StatePlaneSpec` | One plane: buffer role, shape of one slot before any split across ranks, dtype, `PlaneShard` |
| `PlaneShard` | The axis that the ranks sharing a request split evenly, and the sections along it that are each split on their own; no axis when every rank holds the plane whole |

The declared planes, with `inner = head_dim * num_heads` and `grouped = n_groups * state_size`:

| Plane | Shape of one slot | Split across ranks |
|---|---|---|
| SSM (`MambaRole.SSM_STATE`) | `[num_heads, head_dim, state_size]` | Axis 0 |
| Convolution (`MambaRole.CONV_STATE`) | `[inner + 2 * grouped, conv_kernel - 1]` | Axis 0, in three sections: `(inner, grouped, grouped)` for Mamba2, `(grouped, grouped, inner)` for GDN and KDA |
| Qwen4Exp PLE (`PLE_CONV_STATE`, `PLE_NGRAM_CONTEXT`), on the layer that runs PLE | `[hc_count * hidden_size, (ple_conv_kernel_size - 1) * ngram_size]` and `[ngram_size - 1]` | None |

Rules:

- **Base state only.** Replay history and other speculative buffers depend on the built model and
  the runtime environment, so they are declared as regions (`../REGIONS_DEVELOPER_GUIDE.md`), not
  here.
- **Storage, not algorithm.** A plane says what is stored: a role, a shape, a dtype and how ranks
  split it. What the numbers mean to the family is not declared, for example the group count
  behind a convolution section or which section holds which signal. The family's state code
  takes those from the model's configuration.
- **Target-model layers only.** Shared code appends the draft layers of one-model speculative
  decoding, which are attention layers.
- **A model, not a rank.** Shapes are whole. `StatePlaneSpec.local_shape(num_shards)` and
  `local_sections(num_shards)` give one rank's part, where `num_shards` is
  `mamba_effective_tp_size(mapping)`. A sharded axis that the ranks cannot split evenly is an
  error.
- **Dtypes follow the quantization config.** The SSM dtype comes from
  `quant_config.mamba_ssm_cache_dtype`, which `validate_and_set_mamba_ssm_cache_dtype` sets from
  the KV cache configuration while the model config is loaded. A reader that runs before that
  gets the family's default dtype, so it uses the layer kinds only.
- **`None` is not "no recurrent layers".** The hook returns `None` for an architecture that has no
  recurrent layers at all. A recurrent architecture whose configuration selects none returns a
  declaration without `RECURRENT` layers; it has no plane, so it carries no geometry.

### Who declares, and how it is found

`get_recurrent_state_spec(pretrained_config, *, quant_config=None)` is a classmethod of the model
class; `DecoderModelForCausalLM` returns `None`. The five family classes implement it by converting
the result of their geometry function (`recurrent_state_spec_from_mamba_params`).

A VLM wrapper forwards to the class of its language model and passes on the config that its
`__init__` builds that language model from. The declaration read from the composite config before
construction then equals the one read from the built model's config, which the wrapper accepts as
well.

| Wrapper | Forwards to | Config passed on |
|---|---|---|
| `NemotronHMultimodalModel` | `NemotronHForCausalLM` | `llm_config` |
| `_Qwen3_5VLModel` (Qwen3.5 VL, dense and MoE) | `Qwen3NextForCausalLM` | `text_config` |
| `MiniCPMV4_6Model` | The class registered for `text_config.architectures[0]` | `text_config` |
| `Qwen4ExpForConditionalGeneration` | `Qwen4ExpForCausalLM` | `text_config` |
| `KimiK3ForConditionalGeneration` | `KimiLinearForCausalLM` | `text_config` |
| `Glm5NextVLM` | `Glm5NextForCausalLM` | The composite config, unchanged |

`resolve_model_class(pretrained_config)` returns the class registered for
`pretrained_config.architectures[0]`, or `None`. `resolve_recurrent_state_spec(pretrained_config,
...)` asks that class for its declaration. A reader asks through this entry, before and after the
model is built. It does not ask the class of the built model: a `modeling_v2` target can be a
class of its own without the hook, which then inherits the `None` of `DecoderModelForCausalLM`,
while the architecture name still leads to the class that declares.

The entry returns `None` for a recurrent model in three situations:

1. **The config names no architecture.** `load_pretrained_config` names the composite config and
   the text config of a Kimi K3 or Qwen4Exp checkpoint, so the text config that the built wrapper
   leaves on `model_config` leads to its class. Configs that `tools/layer_wise_benchmarks` and
   `bench/tuning` load on their own take the name from the checkpoint. A config built by hand, as
   in a unit-test fixture, has to carry the name.
2. **The name is not registered**, for example a name the loader rewrites before it resolves the
   class.
3. **The registered class is a wrapper without the hook.** None in the tree; a new wrapper around a
   recurrent language model adds the forwarding.

What a reader does then:

| Reader | Without a class that declares |
|---|---|
| `extract_mamba_kv_cache_params` | Raises `ValueError` |
| `get_kv_cache_manager_cls` | Raises `ValueError` for a config that `is_hybrid_linear` recognises |
| `_create_kv_cache_manager` | Builds the manager class through the branches of a model without recurrent state |
| The layer counts and the "is it a recurrent architecture" readers | Answer as for a model without recurrent state |

### The cache parameters of a declaration

The cache managers take one geometry for all recurrent layers, `MambaKVCacheParams`.
`mamba_params_from_recurrent_state_spec` (`recurrent_state_spec.py`) derives it from a declaration,
and `extract_mamba_kv_cache_params` applies it to the declaration of the config's model class. It
is the inverse of `recurrent_state_spec_from_mamba_params`, with these limits:

- Every recurrent layer declares the same convolution plane and the same SSM plane. Other planes,
  such as the PLE planes of Qwen4Exp, are not part of the cache parameters.
- The two planes are the ones `recurrent_state_spec_from_mamba_params` declares, in either
  section order above. The conversion raises `ValueError` for any other shape, section split or
  split across ranks, because the cache parameters cannot express it.
- A declaration without a recurrent layer has no geometry, and the conversion raises `ValueError`.

## What a family provides

Everything lives in the family's model file and family module.

| Provided | Answers |
|---|---|
| `get_recurrent_state_spec` | Layer kinds and the geometry of the base state |
| `get_recurrent_kv_cache_manager_cls` | The V2 manager class of the family. A classmethod of the model class; `DecoderModelForCausalLM` returns `None`, and a VLM wrapper forwards it as it forwards the declaration. `get_kv_cache_manager_cls` returns this class on the V2 route. A model can replace the resolved class with a subclass through `RuntimeHooks.kv_cache_manager_cls` |
| `requires_kv_cache_manager_v2` | Whether V2 is the only route the model supports, which GLM5Next declares. Model loading and `get_kv_cache_manager_cls` then reject a setting that leads to a V1 manager |
| The V2 manager subclass of the family | Which manager implements the cache, and what its construction needs; see "Construction" below |
| `MambaState` and the region providers | The family's algorithm over state slots and the memory beyond the base state |
| `RuntimeHooks`, `get_preferred_kv_cache_manager_version`, `get_preferred_transceiver_runtime` | Unsupported combinations and model-level runtime facts |

### Construction

`_create_kv_cache_manager` builds the recurrent manager of every family in one branch, which it
enters when the model class declares recurrent state. The branch writes the state geometry, the
layer masks and the arguments common to the families once. A V2 manager class answers the rest:

| Member of the manager class | Answers | Default in `MambaHybridCacheManagerV2` |
|---|---|---|
| `attention_cache_geometry(pretrained_config, *, num_kv_heads, head_dim)` | Cache type, KV head counts and head dimension of the attention layers. Kimi K3 and GLM5Next store one key-only head of the latent and RoPE widths | Plain attention KV with the head counts and head dimension of the config |
| `family_init_kwargs(...)` | The keyword arguments the family adds to the constructor call, for example the order of the convolution sections | No arguments |

The two classmethods mirror the constructor's signature and change with it. They are not a stable
interface.

Some constructor options depend on the built model: `kda_draft_records` (KDA), `gdn_fused_verify`
and `gdn_mtp_replay` (GDN), and `inforward_snapshot_min_main` (Qwen4Exp). The family reads them in
`family_init_kwargs` from its `model` argument, which is `None` when a manager is built without a
model, as the manager of a separate draft cache is.

The cost estimates of a manager (`get_cache_size_per_token` and the per-request estimates) run
before the manager exists and take the configurations, not the model. Where such an option selects
state that an estimate sizes, the manager class returns it from `family_estimate_kwargs(model)`,
and `KvCacheCreator._per_manager_cache_cost` adds the result to the estimate calls of a target
manager that keeps recurrent state. The GDN managers return `gdn_mtp_replay`, which selects their
replay state; the default on `MambaHybridCacheManager` adds nothing. A draft manager is costed
without them, as it is built without the model.

A manager that lays out the attention layers of an NVFP4 MLA model together with the recurrent
state lists `Fp4MlaLayoutHooks` in its bases (see `../REGIONS_DEVELOPER_GUIDE.md`), as Kimi K3's
does. `get_kv_cache_manager_cls` rejects such a model for a family whose manager does not.

The manager class and the family of the config can differ: with QSA, `get_kv_cache_manager_cls`
selects `Qwen4ExpHybridCacheManager` for a model that is not Qwen4Exp, so that class decides by
the config which arguments it adds.

The branch raises `ValueError` when the model declares no recurrent layer, and when the manager
class does not declare that it keeps recurrent state (`keeps_recurrent_state`, below).

A V1 hybrid manager (`CppMambaHybridCacheManager`, `MixedMambaHybridCacheManager`) takes the same
shared arguments; its family arguments come from `_legacy_recurrent_manager_arguments` in
`pyexecutor/_util.py`, which decides by family predicate.

`tools/layer_wise_benchmarks/runner.py` constructs cache managers in branches of its own, chosen
by family predicates.

## After construction

`MambaHybridCacheManagerV2` allocates the base state. A family manager subclasses it and can hand
it a state object, a `MambaState`, which the manager then calls at these points:

| Hook | When | Use |
|---|---|---|
| `_initialize_spec_state()` | `MambaHybridCacheManagerV2.__init__`, before the pool is built | The family manager returns its `MambaState`. The shared manager returns `None`, and raises when speculative decoding is configured |
| `bind()` | `_setup_model_state()`, once the layout, the pools and the slot capacity are final | Take the views of the SSM and convolution state. The pool keeps owning them |
| `reset_slots()` | `prepare_resources()` and `add_dummy_requests()`, for the context requests of a batch | Reset the state those requests start from |
| `make_layer_cache()` | `mamba_layer_cache(layer_idx)`, when speculative decoding is configured | Build the `MambaLayerCache` a mixer reads for one layer. Without a state the manager returns the convolution and SSM views alone |
| `update(MambaAcceptanceBatch)` | `update_mamba_states()`, which a speculative worker calls after it accepts tokens | Apply the accepted tokens to the state |
| `intermediate_indices`, `intermediate_ssm`, `intermediate_conv` | Read through the manager's `intermediate_state_indices`, `intermediate_ssm_states` and `intermediate_conv_states` | The intermediate views that mixers pass to their kernels; `None` when the state keeps none |
| `shutdown()` | The manager's `shutdown()` | Release what the state holds |

A state that keeps replay buffers has further hooks. `../REGIONS_DEVELOPER_GUIDE.md` describes how
those buffers are laid out, bound and reset.

The manager itself answers the rest:

| Hook | When | Use |
|---|---|---|
| `get_disagg_role_mapper_kinds()`, `get_disagg_role_layouts()` | Disaggregated transfer | How each state role is resharded across ranks: convolution sections shard independently, SSM state shards by head |
| `get_disagg_ignored_roles()` | Disaggregated transfer | The buffers the receiving worker rebuilds, replay buffers among them |
| `get_disagg_global_layer_ids()`, `get_disagg_pool_view_partition()` | Disaggregated transfer | The transfer IDs of a layer group's layers, and which layers of one pool form separate transfer views |
| `on_state_transfer_complete(request_ids)` | `PyExecutor`, after a disaggregated transfer completes with speculative decoding enabled | A family manager rebuilds what was not transferred; Kimi K3 and GLM5Next seed their replay slots |

### What shared runtime code asks

The executor, the decoder runner, the KV connector setup and the KV cache creator ask the owner of
a fact. For the facts below they do not check for a recurrent manager class:

| Hook | When | Use |
|---|---|---|
| `has_state_layers(manager)` in `pyexecutor/connectors/kv_cache_layout.py` | `PyExecutor`, when a KV connector is configured on a V2 cache | Whether the layout has an `SsmLayerConfig` layer. A connector has nothing to transfer for such a layer, so the setup is rejected |
| `prepare_expect_snapshot_points(requests)` | `PyExecutor`, before it schedules the active requests | Fill `LlmRequest.expect_snapshot_points`. The hybrid managers set the snapshot points; `MambaHybridCacheManagerV2` also aligns them to the attention layout and, with block reuse, adds branch-snapshot and waiter points |
| `keeps_recurrent_state`, read through `cache_keeps_recurrent_state()` in `pyexecutor/resource_manager.py` | Step metadata, decoder warmup, the MLA latent-cache append of the TRTLLM backend, the cache managers' constructors, the KV cache creator and the executor's resource factory | `True` on the `MambaHybridCacheManager` mixin, and so on every hybrid manager, and on the V1 `MambaCacheManager`: the cache is that of a recurrent model. The manager answers for the model and not for its rank, so the manager of a rank that holds no recurrent layer still answers `True`. The next two sections say what step metadata and warmup decide with it. A cache manager's constructor turns a positive speculative recompute tail into a full re-prefill for such a cache, since the state already summarizes the matched prefix; the TRTLLM backend appends the MLA latent cache of a generation step on the graph-safe path for it. `create_py_executor_resources` passes such a cache to `create_kv_cache_transceiver` a second time, as `mamba_cache_manager`: the creation rejects the hybrid managers it cannot serve by that argument, and the C++ transceiver takes the recurrent layers of the rank from it |
| `is_context_only_cache` | Decoder warmup, dummy requests and the executor | `True` when the cache omits generation-only state, so its worker runs context requests only: the KDA draft-verification buffers, or the FP4 MLA V scales and packed V. The KDA and FP4 MLA managers report their own flags. A CSA2 cache pruned for a context worker reports `context_swa_layer_limit` instead |
| `resets_recurrent_state_in_context` on the causal LM | The CUDA-graph runner, before it promotes a one-token context request into a decode graph | A class flag, `True` when only the context forward resets the state of a fresh slot. The runner also reads `has_ple` from the decoder, which is how Qwen4-exp reports active PLE |
| `describes_own_target_layers()`, `appended_draft_layers_use_target_kv_factor()` | The KV cache creator, when it places the layers of a DFlash or standalone DSpark drafter in the target's cache | Classmethods of the V2 manager, `False` on `KVCacheManagerV2` and `True` on `MambaHybridCacheManagerV2`. A manager that describes its own target layers is given the drafter's layer specs alone, and the token estimate does not take its group count from the combined layout. Where appended draft layers take the target's `kv_factor`, a target and a drafter of which only one stores K alone keep separate caches |

`prepare_expect_snapshot_points` and `on_state_transfer_complete` are steps of
`BaseResourceManager`, which does nothing in either, and `is_context_only_cache` and
`keeps_recurrent_state` are `False` there.
Such a default goes there and not on a KV cache manager base: `MambaHybridCacheManagerV2` derives
from `KVCacheManagerV2` before the `MambaHybridCacheManager` mixin, so a default on the KV base
would shadow the mixin's implementation. The executor calls both steps on the primary KV cache
manager only. The creator reads its two declarations from the manager class, before a manager
exists, so they are classmethods beside `accepts_appended_draft_layers` on `KVCacheManagerV2`.

`PyExecutor` checks the manager class in two places, both on V1 caches: the connector setup,
which has no layout to ask, and `_schedule`, which filters context requests by capacity for
`MixedMambaHybridCacheManager` when a transceiver is present.

### Per-step metadata

The step metadata of the recurrent layers is a `Mamba2Metadata` or a subclass of it, kept on
`AttentionMetadata.mamba_metadata`.

| Step | Where | What happens |
|---|---|---|
| Declare the class | `mamba_metadata_cls`, on the model or on the cache manager | The V1 managers serve several families and declare none, so every recurrent model declares its class, as a class attribute or a property. A VLM wrapper forwards the declaration of its language model |
| Resolve it | `resolve_mamba_metadata_cls(model, kv_cache_manager)` in `pyexecutor/engine/runners/decoder/metadata.py` | The model's class, else the manager's, else `Mamba2Metadata`. Warmup resolves through the same function |
| Hand it to attention | `DecoderMetadata.setup_attention` | Writes the resolved class to `AttentionMetadata.mamba_metadata_cls` for every model. For a cache that keeps recurrent state it also copies `chunk_size` from the configuration to `mamba_chunk_size` |
| Build and prepare | `AttentionMetadata._prepare_mamba_metadata`, from `prepare()` | With a cache that keeps recurrent state, the first call builds the object and every call runs its `prepare()`. With any other manager, or none, `mamba_metadata` is `False` |

Both read `keeps_recurrent_state`, which the manager class declares, and not the layout of the
rank. Under pipeline parallelism a rank can hold none of the recurrent layers; its hybrid manager
then has no state layer, and the metadata is still read there. A speculative worker reads `mamba_metadata.state_indices` on every
rank before it calls `update_mamba_states`, and GLM5Next's kpool attention keeps its block tables
on the metadata object.

### Warmup and CUDA graphs

Decoder warmup and the CUDA-graph runner read these declarations:

| Declared on | Name | Read by | Use |
|---|---|---|---|
| The metadata class | `prefill_chunk_alignment` | `_run_mamba_hybrid_warmup`, on the class that `resolve_mamba_metadata_cls` returns | Set by a class whose prefill kernel compiles one variant for context lengths that are all multiples of this value and another for the rest; warmup then covers both. `None` on `Mamba2Metadata`, 16 on `KimiK3MambaMetadata` |
| `Mamba2Metadata` | `force_initial_states_for_warmup()` | `_run_mamba_hybrid_warmup` | A context manager that switches on the warmup-only branches of `prepare()`: the kernel variants with initial states for Mamba2, the state-reset kernel for GDN. The readers read the flag on `Mamba2Metadata`, so it is called on that class and not on a subclass |
| A module of the model | `prewarm_decode_kernels(max_num_tokens, verify_batches)` | `_prewarm_model_decode_kernels` | Optional. Compiles the kernels that no warmup forward reaches with every variant, and returns how many it compiled |
| The V2 manager | `inforward_snapshot_rows` | The CUDA-graph runner, the V2 scheduler and `PyExecutor` | Staging rows per batch for snapshots taken inside the forward; 0 on `KVCacheManagerV2`. When it is positive, the runner warms the kernels of the split |
| The causal LM and its decoder | `resets_recurrent_state_in_context`, `has_ple` | The CUDA-graph runner | See [What shared runtime code asks](#what-shared-runtime-code-asks) |

Warmup reads `keeps_recurrent_state` in three places. Two are in `_warmup_scheduled`: a cache
that keeps recurrent state skips the general warmup, the context and mixed shapes of the attention
warmup, and the pre-population of the memory pool. The third is in `_run_mamba_hybrid_warmup`,
which runs for such a cache only and compiles the multi-sequence and initial-state kernel variants.
The source does not record why such a cache skips the general warmup.

## Adding a recurrent family

1. Implement `get_recurrent_state_spec` on the model class. Declare one entry per target layer and
   only base state; give each plane its whole shape and how ranks split it.
2. Subclass `MambaHybridCacheManagerV2` in the family module and return the subclass from
   `get_recurrent_kv_cache_manager_cls` on the model class. Override `attention_cache_geometry`
   and `family_init_kwargs` for what the constructor needs beyond the shared arguments, and
   `family_estimate_kwargs` when one of those arguments comes from the built model and selects
   state that the cost estimates size.
3. In every VLM wrapper around the model, forward both classmethods with the config the wrapper
   builds the language model from, and accept that config as well as the composite one.
4. Add the family's predicate to `is_hybrid_linear` in `pyexecutor/config_utils.py`, which the
   entry of `get_kv_cache_manager_cls` and the snapshot validation of the loader read.
5. Return `True` from `requires_kv_cache_manager_v2` when the V1 hybrid managers do not hold the
   family's state.
6. Declare memory beyond the base state as regions (`../REGIONS_DEVELOPER_GUIDE.md`).
7. Add the family's configuration to `test_recurrent_state_spec.py`. The test compares the
   declaration with the buffers the manager allocates, and with the family's geometry function
   when the declaration is converted from one.
8. `tools/layer_wise_benchmarks` supports a family only through a branch of its own.
