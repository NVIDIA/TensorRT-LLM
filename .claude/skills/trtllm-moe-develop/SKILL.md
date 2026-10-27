---
name: trtllm-moe-develop
description: >-
  Review, design, and refactor TensorRT-LLM PyTorch MoE code for architecture fit,
  clean code, maintainability, and testability. Always use for any modification,
  review, refactor, or design planning that touches MoE modules, including
  tensorrt_llm/_torch/moe/fused_moe, ConfigurableMoE, MoE implementations
  (MoEImplBase leaves and families), backend selection (moe_resolution,
  can_implement, implementation identities), MoEScheduler/moe_scheduler.py,
  forward execution/chunking, communication strategies, EPLB,
  quantization/weight handling, activation support, routing, factories, MoE
  docs, or MoE tests. Also use when the user asks whether a MoE design follows
  the current architecture or whether a MoE refactor is reasonable.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM MoE Code Quality

`MOE_DEVELOPER_GUIDE.md` states the MoE architecture and its rules. This skill
is the workflow on top of it: what to read, the owner boundaries to check a
change against, which rules only a reviewer enforces, the knowledge the guide
does not carry (EPLB staging, imported kernel ABIs, communication
checkpointing), and the templates for a design gate and a review. Favor module
roles, API boundaries, and testability over local style cleanup.

## Required Context

Before proposing or editing MoE code, read:

1. `CODING_GUIDELINES.md`
2. `tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md`: Architecture,
   File Map, Backend Selection, Backend Capability Matrix, Canonical Examples
   (including its notes on inheritance, one class per identity, class names,
   and aliases), and Anti-Patterns. Its File Map names the file that owns each
   area.
3. The target files and their tests under `tests/unittest/_torch/moe/`.

Two places the File Map does not point to:

- Model-side construction: the model's `create_moe(...)` call, in particular
  how it computes `override_quant_config`.
- Shared test helpers: `tests/unittest/_torch/moe/moe_test_utils.py` and
  `quantize_utils.py`, whenever a change adds backend, quantization, skip, or
  parameter coverage.

For module-specific work, open the matching section of
`references/moe-canonical-code-examples.md`. Each design gate or review should
cite at least one concrete code example with file:line evidence.

## Keeping MOE_DEVELOPER_GUIDE.md Current

The guide is the in-repo source of truth; this skill and its reference point to
it rather than restating it. If guide and code disagree, inspect code and
tests, mention the mismatch, and either update the guide as part of the change
or report it as follow-up.

Guide-update checklist:

- File map changed: update `File Map`.
- New, renamed, or re-parented implementation: update `File Map` (and the
  family's leaf table), `Backend Capability Matrix`, and every Canonical
  Examples note that enumerates classes.
- Backend or quant support changed: update `Backend Capability Matrix`.
- Selection rule changed (priority, family membership, fallback, reject reason,
  environment probe): update `Backend Selection`.
- Activation declaration changed: update `Activation Support`.
- New backend/communication/forward-execution pattern: update `Canonical Examples`.
- New forbidden pattern or ownership rule: update `Anti-Patterns`.
- Test convention changed: update `Tests` and the Identity Tests row of
  `Canonical Examples`.

## Owner Boundaries

A change is good only if it leaves these roles at least as clear as before.

| Owner | Owns | Must not |
|---|---|---|
| `ConfigurableMoE` (`configurable_moe.py`) | Backend construction and attribute sync, communication strategy lifetime, `validate_backend`, DWDP wait/bind/record, advancing `repeat_idx` once per `forward_impl`, constructing the scheduler last (it reads that state) | Choose the implementation; grow chunking, routing, communication, EPLB, or backend-specific branches in `forward_impl` |
| `MoEScheduler` (`moe_scheduler.py`) | Per-forward policy: padding/truncation, chunking, quantize/dispatch order, EPLB hook order, zero-token chunks, external vs fused communication, calling `backend.quantize_input` / `backend.run_moe` | Advance `repeat_idx`, make DWDP calls, create or destroy communication strategies, call weight lifecycle hooks; change `moe.comm` other than through the `determine_communication_method` fallback |
| Implementation (`MoEImplBase` leaf) | `can_implement` and the declarations it reads, `quantize_input` / `run_moe`, the weight lifecycle hooks (defaults from `MoEWeightOwnerMixin`) | Define `forward`; run host-side communication unless it is a `FUSED_COMM` kernel that owns the exchange; carry forward-time policy |
| Quantization method (`quantization.py`) | Format-specific tensor layout, loading details, post-load transforms, scale setup, format-specific EPLB registrations | Replace the backend's lifecycle hooks as the public owner of weight handling |
| Communication strategy (`communication/`) | External dispatch/combine and the state the pair shares | Be forced onto a backend whose kernel owns the exchange |
| Selection (`moe_resolution.resolve_moe_impl`) | Ordering candidates, asking each `can_implement(p, d)`, the `MoEResolutionReport` | Hold capability knowledge; gain a second entry point |
| Test helpers (`moe_test_utils.py`, `quantize_utils.py`) | Backend/quantization matrices and skip logic | Drift from a test file that adds a case |

Cross-row checks the guide does not spell out:

- `run_moe` is the one compute entrypoint the scheduler reaches, for every
  backend `ConfigurableMoE` can wrap. A backend-specific helper is private and
  called from `run_moe`; a wrapper or scheduler calling it directly, while
  `run_moe` is `NotImplementedError` or unused, is a contract violation.
- Backend-specific scheduler kwargs stay centralized and narrow in scheduler
  helper code, with a comment saying why the common `run_moe` contract is
  insufficient for that backend.
- Backend-specific wrapper constraints go in `validate_configurable_moe`, not
  in forward branches.

## What Machines Check, And What Only Review Catches

The guide's Backend Selection rules hold everywhere, but only some are
enforced. Spend review attention on the rows nothing checks.

| Rule | Enforced by | Reviewer action |
|---|---|---|
| Every `BACKEND_FAMILY` member is ranked in `IMPL_PRIORITY` | Import-time `RuntimeError` in `moe_resolution.py` (`_UNRANKED`) | None |
| A family member is what the registry returns for the identity it publishes | `test_family_members_are_reachable_by_the_identity_they_publish` | None |
| Identity and implementation sit on one concrete class | `test_the_identity_and_the_implementation_sit_on_one_class`, for the classes in `_SINGLE_FORMAT_IMPLS`; `test_cutlass_leaf_declarations` and `test_cutedsl_leaf_declarations` for the CUTLASS and CuteDSL grouped-GEMM leaves, each kept complete by its `*_declarations_cover_every_leaf` | Add a new single-format implementation to `_SINGLE_FORMAT_IMPLS` |
| `create_moe_backend` builds the class it is handed | `test_create_moe_backend_dispatches_the_impl`, over every class in `IMPL_PRIORITY` | None |
| A model whose `quant_config_dict` / `exclude_modules` can move its experts passes `override_quant_config` | `require_layer_quant_format` in the CUTLASS, TRTLLM-Gen, B12x and CuteDSL grouped-GEMM `create_weights`, and only when that checkpoint is loaded; other implementations are not checked | Check the model's `create_moe` call |
| `can_implement` is pure: no `get_sm_version()`, no `import` as a presence test, no `os.environ` | Nothing | Read every new or changed gate |
| A routing, activation, EPLB, or LoRA constraint is declined in `can_implement` (reading `p.routing`, `activation_support`, `d.*`), not in a factory, scheduler, wrapper, or test-local branch | Nothing | Read the diff outside `can_implement` |
| A family alias is matched with `issubclass` / `isinstance`, never `==`, `is`, or `in (...)` | Nothing | Grep the diff |
| A `MoeConfig.backend` literal change follows the AGENTS.md LLM-args manifest and serve-schema rules | The manifest and schema checks | Regenerate both |

## Implementation Checklist

The guide's Canonical Examples notes cover inheritance, family roots, one class
per identity, and naming. In addition:

- Weight lifecycle hooks are implemented or rejected with a narrow error:
  `create_weights()` is safe under `ConfigurableMoE` deferred weight creation,
  `load_weights()` honors or rejects `allow_partial_loading`, and
  `post_load_weights()` / `process_weights_after_loading()` /
  `pre_reload_weights()` keep transformed weights and reload metadata coherent.
- The backend selects and stores its quantization method before delegating
  layout-specific registration, loading, or transforms; callers do not reach
  into `quantization.py` directly.
- Cross-rank exchange ownership is declared through `scheduler_kind`, not
  hidden behind wrapper `isinstance` checks.

### Imported Kernel ABI Checklist

When importing or wrapping an upstream kernel, derive the TRT-LLM adapter
contract from the lowest-level kernel consumer. Comments, docs, design notes,
and parameter names are useful hints, but they are not proof of the runtime ABI.

- Derive weight shape and layout from the kernel entrypoint, `make_layout`, TMA,
  MMA/GEMM transforms, and stride usage. Record required tensor shape, stride,
  physical storage layout, and boundary view layout.
- Derive alpha and scale semantics from kernel consumption points. Trace where
  alpha, norm constants, block scales, activation scales, and weight scales are
  loaded and multiplied before deciding how upper layers compute or pack them.
  Treat weight bytes, block scales/SF, and global alpha/norm constants as
  separate contracts.
- Design the upper-layer adapter from the kernel ABI upward. Map each kernel
  input/output to an adapter responsibility: storage tensor, view/transposition,
  dtype reinterpretation, padding, scale packing, workspace ownership,
  synchronization, and output reduction. Validate parity with upstream
  invocation dumps, not just final output.

## Quantization And Weights

- A leaf is one quantization format, so its `_get_quant_method` returns one
  method and does not branch on `quant_config` to choose among formats.
- Look for a reusable quant method or base class before creating one; reuse it
  when the tensor layout and scale semantics match.
- `create_weights()` registers module parameters with the correct slot, expert,
  hidden, intermediate, and scale layout.
- `load_weights()` handles supported loading modes and rejects unsupported ones
  clearly. Preserve the EPLB split: common MoE FC weights/biases
  (`w3_w1_weight`, `w2_weight`, and bias tensors when present) use the shared
  `FusedMoEMethodBase.load_weights()` / `post_load_weights()` path, where
  `need_load_shared_weights(module)` gates CPU shared staging and registration.
- Quantization methods add only their quantization-specific EPLB registrations
  for scales, alphas, transformed weights, or layout-specific views that are not
  covered by the base FC weight path. Those extra tensors must also be gated by
  `need_load_shared_weights(module)` before loading, transforming, or registering
  shared copies. If a specialized method cannot reuse the base FC path because
  its raw parameter layout is incompatible, the design must call out that
  exception and preserve equivalent base semantics explicitly.
- `setup_quant_scales()` is updated when a quant mode exposes scales consumed by
  backend, communication, or forward-execution paths.
- EPLB support status is explicit: `SUPPORTED`, `NOT_SUPPORTED`, or
  `NOT_VERIFIED` (`eplb_support_status`).

## EPLB

EPLB is cross-cutting: a correct change may need updates in selection, the
implementation, quantization, forward execution, communication, and tests.

- Selection-time support is `capabilities.supports_eplb` read with
  `d.eplb_enabled` in `can_implement`. The instance-level
  `_supports_load_balancer()` answers a different question (TRTLLM-Gen
  overrides it to mean "routing is separated from the kernel"), so neither
  stands in for the other, and a leaf that inherits `True` from a parent but
  cannot run EPLB declines `d.eplb_enabled` in its own gate.
- Forward execution collects routing statistics, chooses `ignore_allreduce`
  for the communication path, and keeps the hook order around routing,
  `run_moe`, and CPU weight migration.
- `num_slots`, `num_experts`, `ep_size`, and slot-vs-expert IDs are not mixed.
- Add or update concrete EPLB tests in `test_moe_module.py` for the
  backend/comm/quant combination that changed.

### CPU shared-staging buffer family (EPLB migration)

Dynamic EPLB needs host-resident copies of per-expert tensors so that
`MoeLoadBalancer` can migrate experts between ranks via host shared memory.
Each per-expert `nn.Parameter` on the module has a parallel CPU staging buffer;
all of them are passed to `register_all_parameter_slot_and_to_fix_weight_fns`
once loading finishes. Any new per-expert Parameter MUST add its own staging
buffer and migration hook, or the shared-load path will either write out of
bounds or silently corrupt routed slots (NVBug 6130334 / PR #13856).

Full family in the NVFP4 path (`quantization.py`):

| GPU `nn.Parameter` on module | CPU shared staging buffer | Sized by |
|---|---|---|
| `w3_w1_weight` (packed FP4) | `module.local_shared_w3_w1_tensors` | `len(local_shared_load_expert_ids)` |
| `w2_weight` (packed FP4) | `module.local_shared_w2_tensors` | same |
| `w3_w1_bias` / `w2_bias` (if `bias=True`) | `module.local_shared_w3_w1_bias_tensors` / `module.local_shared_w2_bias_tensors` | same |
| `w3_w1_weight_scale` / `w2_weight_scale` (block scales) | `module.local_shared_w3_w1_scale_tensors` / `module.local_shared_w2_scale_tensors` | same |
| `fc31_alpha` / `fc2_alpha` (per-expert fp32 scalar) | `shared_fc31_alpha` / `shared_fc2_alpha` (local variables in `process_weights_after_loading`) | `num_shared = len(tmp_shared_weight_scale_2)` |
| `fc31_weight_scale_2` / `fc2_weight_scale_2` (per-expert fp32 scalar, gated by `force_dynamic_quantization`) | `shared_fc31_weight_scale_2` / `shared_fc2_weight_scale_2` (local variables) | same |

Key index-space distinction:

- `expert_size_per_partition = num_slots / ep_size` is the routed-slot count on
  this rank; sizes the on-GPU module Parameters.
- `num_shared = len(local_shared_load_expert_ids) = num_experts / shared_size`,
  where `shared_size = shared_mpi_comm.Get_size()` is the same-node MPI rank
  count (from `MPI_COMM_TYPE_SHARED` split); sizes the CPU staging buffers.
- On multi-node setups `shared_size < ep_size` is legal and makes
  `num_shared > expert_size_per_partition`. Any code that writes into a
  routed-sized Parameter using a staging-space index will go out of bounds.
- On single-node setups `shared_size == ep_size` is enforced by the
  `assert shared_size == local_size` in `MoeLoadBalancer._setup_mpi_comm`, so
  single-node unit tests cannot exercise the
  `num_shared > expert_size_per_partition` failure mode through parameter
  tuning alone. A regression test for staging-index correctness must either
  (a) invoke the reconcile/migration function directly with a crafted staging
  dict, or (b) run on a real multi-node Slurm environment.

Naming convention quirk: bulk weights and block-scales use
`module.local_shared_*_tensors` (attribute on module, deleted after register);
per-expert scalars (alphas, `weight_scale_2`) use `shared_*` (function-local).
Both are equally valid migration sources  --  the distinction is historical.

Checklist for adding a new per-expert Parameter to an EPLB-supporting
quantization method:

1. Register the on-module `nn.Parameter` sized `expert_size_per_partition` in
   `create_weights()`.
2. In whichever loader fills it, also fill a `tmp_shared_*_weight_scale_X` dict
   keyed by `enumerate(local_shared_load_expert_ids)` during the
   `need_load_shared_weights(module)` branch.
3. In `process_weights_after_loading()` (or the equivalent finalize step),
   allocate a CPU `shared_*` buffer sized `num_shared` and fill it from the
   temp dict. Pass it as an explicit destination to reconcile/compute
   helpers  --  do NOT write into the on-module `.data[expert_idx]` from the
   shared path, since `expert_idx` is in staging space and the on-module
   Parameter is in routed space.
4. Add the staging buffer to the `weight_fns` dict handed to
   `register_all_parameter_slot_and_to_fix_weight_fns({...})` so migration can
   find it.
5. If the reconcile/compute helper is shared between routed and staging paths,
   its signature must take the destination tensor as a parameter (not read
   `module.<param>.data` directly), so the same body serves both index spaces.

Red flags:

- A new per-expert Parameter registered in `create_weights()` but never added
  to any `weight_fns` migration dict  --  it will be stale after the first EPLB
  migration.
- A reconcile/compute function that both reads `tmp_shared_*` and writes
  `module.<per_expert_param>.data[expert_idx]`  --  the staging-space index can
  exceed the routed-space bound (multi-node) or silently overwrite routed
  slots (single-node).
- Asymmetric gating: one of `fc31_*` / `fc2_*` pair registered but its twin
  not (or one added to `weight_fns` but not the other)  --  migration will leave
  half the state stale.

## Communication

The guide covers the communication timeout guard. In addition:

- Strategy selection and forced-method behavior go through
  `communication_factory.py`.
- `supports_post_quant_dispatch()` matches the payload layout, and
  `prepare_dispatch()` carries only metadata/statistics that must exist before
  dispatch.
- `dispatch()` and `combine()` keep enough internal state for the pair to be
  correct.
- `checkpoint_blocked_reason()` returns `None` by default, meaning the strategy
  can be checkpointed. Override it when a configuration blocks checkpointing;
  `PyExecutor.__init__` uses it to reject `sleep_config` before the engine serves.
- A strategy whose native workspace must follow executor sleep/wakeup implements
  `CheckpointableCommunication`; discovery goes through that protocol, never a
  concrete strategy check or a private workspace attribute.
- EPLB statistics gathered by the strategy are fed back to the load balancer
  through the forward-execution path.

## Test Matrix And Helpers

The guide's Tests table and Identity Tests row say which file a test belongs in
and what an identity test should cover. In addition:

- A new family gets a `MoeBackendType` value (one per family) and an entry in
  `find_backend_class`, which returns `None` for an absent leaf instead of
  raising, plus the backend/module matrices and skip logic.
- A new quantization method is added to the test quant parameters and EPLB
  support checks when applicable.
- A skip reason maps to a production capability check (`can_implement`) or an
  environment constraint, not a broad local skip.
- The CI subset and the local exhaustive matrix stay intentionally different,
  and the helpers document how.
- Scheduler changes get module-level tests for chunking, zero-token chunks, DP
  padding/truncation, EPLB hook order, and fused-communication lockstep.

## Design Gate

Before editing, write a short gate:

```markdown
## MoE Design Gate
- Change area: <ConfigurableMoE / MoEScheduler-forward-execution / selection-identity / backend / quantization-weights / activation / EPLB / communication / routing / test-matrix / tests>
- Owner boundary: <where the behavior belongs and why>
- Main API touched: <method/class names>
- Selection impact: <can_implement gates, descriptor/identity, IMPL_PRIORITY / BACKEND_FAMILY, create_moe_backend branch, or "none">
- Reference pattern: <existing file/class/function from references/moe-canonical-code-examples.md, with file:line evidence>
- Guide sections used: <MOE_DEVELOPER_GUIDE.md sections>
- Guide update needed: <yes/no; which section if yes>
- Refactor needed: <yes/no; one reason tied to an owner boundary, not style>
- Test plan: <backend/impl-identity/module/comm/routing/EPLB/multi-GPU tests>
```

If the owner boundary is unclear, inspect more code before editing.

## Review Output

For reviews, lead with findings and concrete references:

```markdown
## Findings
- [High] <file:line> <architecture, correctness, or testability issue>
- [Medium] <file:line> <maintainability or boundary issue>
- [Low] <file:line> <local cleanup>

## Architecture Fit
- Owner boundaries respected: <yes/no; which row of Owner Boundaries if no>
- Unenforced rules checked: <the rows of "What Machines Check" this change touches>
- Refactor recommended: <yes/no + the owner boundary it clarifies>

## Guide Alignment
- Sections checked: <MOE_DEVELOPER_GUIDE.md sections>
- Guide update needed: <yes/no + section>

## Coverage
- <selection-identity / weights-quantization / activation / EPLB / communication / scheduler / backend / test helpers>: <covered/gap>
```

If there are no findings, say so and list remaining test or performance risk.

## Test Selection

Prefer the unified MoE tests:

- Shared test matrix/helper changes: inspect `tests/unittest/_torch/moe/moe_test_utils.py` and `quantize_utils.py`, then run the affected backend/module tests below.
- Selection/identity changes: `pytest tests/unittest/_torch/moe/test_moe_impl.py -k '<impl or family>'`.
- Backend interface changes: `pytest tests/unittest/_torch/moe/test_moe_backend.py -k '<backend or quant>'`.
- Module/create/forward changes: `pytest tests/unittest/_torch/moe/test_moe_module.py -k '<backend or feature>'`.
- Communication changes: `pytest tests/unittest/_torch/moe/test_moe_comm.py -k '<strategy>'`.
- Routing changes: `pytest tests/unittest/_torch/moe/test_moe_routing.py -k '<routing>'`.
- Load balancer changes: `pytest tests/unittest/_torch/moe/test_moe_load_balancer.py -k '<case>'`.
- Multi-GPU EP/all-to-all behavior: `pytest tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py -k '<case>'`.

When GPU resources are required, use the TRT-LLM GPU allocation/test-runner
skills first and record skipped tests with reasons.
