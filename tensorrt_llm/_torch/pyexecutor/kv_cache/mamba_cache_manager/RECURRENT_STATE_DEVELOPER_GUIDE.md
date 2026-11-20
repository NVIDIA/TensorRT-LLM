<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recurrent State Developer Guide

Some model layers carry state from one step to the next instead of a KV cache: Mamba2
(Nemotron-H), GDN (Qwen3-Next, Qwen3.5, Qwen4Exp) and KDA (Kimi K3, GLM5Next). This guide
describes the contract between shared runtime code and such a model family: what the model class
declares, how the declaration is found, and what the family provides once a cache manager exists.

Code: `tensorrt_llm/_torch/recurrent_state_spec.py` (the declaration),
`tensorrt_llm/_torch/models/modeling_utils.py` (`get_recurrent_state_spec`,
`resolve_recurrent_state_spec`), `common.py` in this directory (`MambaState`, `MambaStateLayout`),
`mamba_cache_manager_v2.py` (the shared hybrid manager), and the family modules under
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
the geometry of their base state is: shapes, dtypes and the split across ranks.

Shared code does not read the declaration. The model-family predicates in
`pyexecutor/config_utils.py` (`is_hybrid_linear` and the per-family ones), `MambaKVCacheParams` and
`isinstance` checks on the recurrent manager classes answer these questions. New shared code uses
the declaration instead of adding to them.

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

`resolve_recurrent_state_spec(pretrained_config, ...)` looks the class up by
`pretrained_config.architectures[0]` and asks it. A reader asks through this entry, before and
after the model is built. It does not ask the class of the built model: a `modeling_v2` target
can be a class of its own without the hook, which then inherits the `None` of
`DecoderModelForCausalLM`, while the architecture name still leads to the class that declares.

The entry returns `None` for a recurrent model in three situations. A site that can meet one of
them keeps its family predicate:

1. **The config names no architecture.** A built `KimiK3ForConditionalGeneration` leaves its text
   config on `model_config`, and only the composite config is given a name when it is loaded
   (`load_pretrained_config` names both for Qwen4Exp).
   Configs that `tools/layer_wise_benchmarks` and `bench/tuning` load on their own take the name
   from the checkpoint. Several unit-test fixtures omit it.
2. **The name is not registered**, for example a name the loader rewrites before it resolves the
   class.
3. **The registered class is a wrapper without the hook.** None in the tree; a new wrapper around a
   recurrent language model adds the forwarding.

The family predicates do not recognise the composite configs of Nemotron-H multimodal, Qwen3.5 VL
and MiniCPM-V 4.6, while their wrappers declare state. A site that sees such a config and moves
from a predicate to the declaration changes its answer for these models, for example the snapshot
validation in `pyexecutor/model_loader.py`.

## What a family provides

Everything lives in the family's model file and family module.

| Provided | Answers |
|---|---|
| `get_recurrent_state_spec` | Layer kinds and the geometry of the base state |
| The V2 manager subclass of the family | Which manager implements the cache. It is chosen by family predicates in `get_kv_cache_manager_cls`; a model can replace the resolved class with a subclass through `RuntimeHooks.kv_cache_manager_cls` |
| `MambaState` and the region providers | The family's algorithm over state slots and the memory beyond the base state |
| `RuntimeHooks`, `get_preferred_kv_cache_manager_version`, `get_preferred_transceiver_runtime` | Unsupported combinations and model-level runtime facts |

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

## Adding a recurrent family

1. Implement `get_recurrent_state_spec` on the model class. Declare one entry per target layer and
   only base state; give each plane its whole shape and how ranks split it.
2. In every VLM wrapper around the model, forward the classmethod with the config the wrapper
   builds the language model from, and accept that config as well as the composite one.
3. Add the family to what shared code reads, in `pyexecutor/config_utils.py`: its predicate to
   `is_hybrid_linear`, and its geometry function, built on `build_mamba_kv_cache_params`, to
   `extract_mamba_kv_cache_params`. The geometry function gives the same layers, shapes and dtypes
   as the declaration.
4. Subclass `MambaHybridCacheManagerV2` in the family module when the family keeps a state object
   or memory of its own, and select the subclass in `get_kv_cache_manager_cls`
   (`pyexecutor/_util.py`).
5. Declare memory beyond the base state as regions (`../REGIONS_DEVELOPER_GUIDE.md`).
6. Add the family's configuration to `test_recurrent_state_spec.py`. The test compares the
   declaration with the geometry function and with the buffers the manager allocates.
