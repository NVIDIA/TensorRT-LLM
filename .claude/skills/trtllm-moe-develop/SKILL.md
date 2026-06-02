---
name: trtllm-moe-develop
description: >-
  Review, design, and refactor TensorRT-LLM PyTorch MoE code for architecture fit,
  clean code, maintainability, and testability. Always use for any modification,
  review, refactor, or design planning that touches MoE modules, including
  tensorrt_llm/_torch/modules/fused_moe, ConfigurableMoE, MoE backends,
  MoEScheduler/moe_scheduler.py, forward execution/chunking, communication
  strategies, EPLB, quantization/weight
  handling, routing, factories, MoE docs, or MoE tests. Also use when the user
  asks whether a MoE design follows the current architecture or whether a MoE
  refactor is reasonable.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM MoE Code Quality

Use this skill to keep MoE changes aligned with the current TensorRT-LLM MoE
architecture. Favor module roles, API boundaries, and testability over local
style cleanup.

## Required Context

Before proposing or editing MoE code, read:

1. `CODING_GUIDELINES.md`
2. `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md`
3. The target files being changed
4. The relevant tests under `tests/unittest/_torch/modules/moe/`

Also inspect these files when the area is relevant:

- Forward execution/chunking: inspect `moe_scheduler.py`, `configurable_moe.py`,
  `interface.py`, backend `run_moe`/`quantize_input` paths, and communication code.
- MegaMoE/fused communication: inspect `moe_scheduler.py`, `mega_moe/`,
  `configurable_moe.py`, `quantization.py`, and communication code.
- Communication: `tensorrt_llm/_torch/modules/fused_moe/communication/base.py`
  and `communication_factory.py`.
- Quantization and weights: `tensorrt_llm/_torch/modules/fused_moe/quantization.py`.
- EPLB/load balancing: `interface.py`, `moe_load_balancer.py`, `quantization.py`,
  `moe_scheduler.py`, current forward-execution/chunking code, and
  `test_moe_module.py`.
- Test matrix/helpers: `tests/unittest/_torch/modules/moe/moe_test_utils.py` and
  `quantize_utils.py` when adding backend, quantization, skip, or parameter
  coverage.


For module-specific work, read `references/moe-canonical-code-examples.md`
after the guide and load only the relevant section. Each design gate or review
should cite at least one concrete code example with file:line evidence.

## Working With MOE_DEVELOPER_GUIDE.md

Treat `MOE_DEVELOPER_GUIDE.md` as the in-repo source of truth for MoE
architecture. Treat this skill as the agent workflow
layer that tells Codex how to apply that source of truth while designing,
editing, or reviewing code.

Use the guide this way:

- Start from the guide sections that match the requested change: Architecture,
  File Map, Backend Capability Matrix, execution-flow/EPLB constraints,
  Canonical Examples, and Anti-Patterns.
- Use guide content to fill the design gate: owner boundary, main API, reference
  pattern, and test plan.
- Do not duplicate fast-changing matrices or backend support tables in this
  skill; prefer the guide as the current reference.
- If a code change adds a backend, quantization method, communication strategy,
  fused-communication behavior, EPLB behavior, or test convention, check whether
  the guide also needs an update.
- If guide and code disagree, inspect code and tests, mention the mismatch, and
  either update the guide as part of the change or report it as follow-up.

Guide-update checklist:

- File map changed: update `File Map`.
- Backend or quant support changed: update `Backend Capability Matrix`.
- New backend/communication/forward-execution pattern: update `Canonical Examples`.
- New forbidden pattern or ownership rule: update `Anti-Patterns`.
- Test convention changed: update `Tests`.

## Core Principle

Preserve these owner boundaries:

- `ConfigurableMoE` is the assembler/orchestrator. It wires backend,
  communication, EPLB, weight lifecycle delegation, and shared wrapper
  bookkeeping.
- Backends declare capabilities, run MoE computation, and own the MoE module's
  weight lifecycle boundary. They expose and implement `create_weights`,
  `load_weights`, `post_load_weights`, `process_weights_after_loading`, and
  `pre_reload_weights` as needed, select any `FusedMoEMethod`, and make those
  hooks compatible with ConfigurableMoE deferred weight creation and reload
  flows. Backends may delegate quantization-specific tensor layout, loading,
  post-load transforms, and scale setup to a quantization method, but backend
  lifecycle hooks remain the public owner of weight handling. New
  ConfigurableMoE-compatible backends should expose `quantize_input` and
  `run_moe`, not `forward` or `forward_impl`, unless the user explicitly asks
  for legacy standalone behavior. For an active ConfigurableMoE-compatible
  backend, `run_moe` must be a concrete implementation, not an empty stub, and
  ConfigurableMoE or its scheduler should call `backend.run_moe(...)` as the
  compute entrypoint. Backend-specific alternatives such as `run_with_prequant`
  are acceptable only as private helpers called from `run_moe`, not as public
  wrapper/scheduler targets that bypass the common contract. The current `MoE`
  interface still covers both legacy standalone MoE modules and newer backends;
  treat legacy `forward` methods as transitional until a dedicated `MoEBackend`
  interface exists. Backends should not become orchestration or
  external-communication state machines.
- Quantization methods are backend-selected implementation helpers for
  quantization-specific weight tensor layout, loading details, post-load
  transforms, scale setup, and EPLB fix-up registration. They do not replace
  backend ownership of the weight lifecycle API.
- Communication strategies own external cross-rank dispatch/combine.
- `MoEScheduler` owns forward-time policy: padding/truncation, chunking,
  dispatch/quantize ordering, EPLB hook ordering, zero-token chunk behavior,
  external-vs-fused communication workflow, and backend `run_moe` invocation.
  `ConfigurableMoE` constructs the scheduler from `backend.scheduler_kind` and
  delegates to it; schedulers may read wrapper state and call wrapper helpers but
  must not own lifecycle, weight loading, DWDP record, `repeat_idx` advancement,
  or communication lifetime. The only sanctioned scheduler mutation of
  `moe.comm` is through `determine_communication_method` fallback.
- Shared test helpers own backend/quantization matrices and skip logic. Updating
  one test file while leaving `moe_test_utils.py` or `quantize_utils.py` stale is
  usually incomplete.
- Tests should exercise the boundary that changed: backend, module,
  communication, routing, EPLB, or multi-GPU behavior.

A refactor is good only if it keeps these roles clearer than before.

## Module Blocks

### ConfigurableMoE: Assembler

Role:

- Compose backend, communication strategy, EPLB, and wrapper-level lifecycle.
- Keep `forward_impl` focused on wrapper-level work: resolve output dtype,
  delegate execution, record DWDP, advance `repeat_idx` once.
- Own backend construction/sync and validation, not backend-specific forward
  policy.

Main APIs / references:

- `configurable_moe.py`: `ConfigurableMoE.__init__`, backend construction,
  communication strategy creation/bypass, scheduler construction, `forward_impl`,
  `validate_backend`.
- `MOE_DEVELOPER_GUIDE.md`: ConfigurableMoE orchestrator and file map.

Checklist:

- New behavior still leaves `ConfigurableMoE` as an assembler.
- No new backend-specific fast path in `forward_impl` unless it is a temporary
  compatibility bridge with a clear follow-up.
- `forward_impl` or an extracted scheduler should invoke backend computation
  through `backend.run_moe(...)`; direct calls to backend-specific compute
  entrypoints such as `run_with_prequant` are red flags unless the change is
  explicitly a short-lived adapter and `run_moe` remains the real implementation.
- Shared wrapper state such as `repeat_idx`, DWDP record, backend attr sync, and
  communication lifetime stays in one place.
- Scheduler creation happens after backend, communication, chunking streams,
  validation, and optional DWDP setup are initialized, because schedulers read
  that wrapper state.
- `forward_impl` should not accumulate chunking, routing, communication, EPLB,
  or fused-kernel branches; that policy belongs in `MoEScheduler`.

### MoE Scheduler: Forward Execution Strategy

Role:

- Own per-forward execution policy for ConfigurableMoE: padding/truncation,
  chunking, dispatch ordering, adaptive pre/post quant dispatch, EPLB wait/stat
  update/route/CPU-stage hook ordering, zero-token chunk behavior, and backend
  `run_moe` invocation.
- Select external-vs-fused communication behavior through `MoESchedulerKind`, not
  through wrapper `isinstance` checks.
- Read wrapper state and call wrapper helpers, but do not own module lifecycle,
  backend construction, weight lifecycle, communication object lifetime, DWDP
  record, or `repeat_idx` advancement.

Main APIs / references:

- `moe_scheduler.py`: `MoEScheduler`, `ExternalCommMoEScheduler`,
  `FusedCommMoEScheduler`, `create_moe_scheduler`.
- `interface.py`: `MoESchedulerKind` and backend `scheduler_kind`.
- `configurable_moe.py`: scheduler construction and thin `forward_impl`
  delegation.
- `communication/base.py`: `supports_post_quant_dispatch`, `prepare_dispatch`,
  `dispatch`, and `combine` contracts used by `ExternalCommMoEScheduler`.

Checklist:

- New forward policy goes in `moe_scheduler.py`, not in ConfigurableMoE or a
  backend, unless it is truly backend-local compute inside `run_moe`.
- `ExternalCommMoEScheduler` owns host-side dispatch/combine, communication
  fallback, optional multi-stream chunk overlap, padding/truncation, and external
  communication EPLB statistic paths.
- `FusedCommMoEScheduler` owns fused-kernel lockstep: ADP stripping,
  per-rank-consistent chunk count, zero-token launches, no external
  dispatch/combine, and `ignore_allreduce=False` EPLB statistic update.
- Schedulers call `backend.quantize_input(...)` and `backend.run_moe(...)`; they
  must not call backend-specific alternate compute helpers that bypass `run_moe`.
- Schedulers must not advance `repeat_idx`, run DWDP record/prefetch, create or
  destroy communication strategies, or call weight lifecycle hooks.
- If backend-specific kwargs are needed, keep them centralized and narrow inside
  scheduler helper code, with comments explaining why the common `run_moe`
  contract is insufficient for that backend.
- Add/update module-level tests for changed scheduler behavior, especially
  chunking, zero-token chunks, DP padding/truncation, EPLB hook order, and
  fused-communication lockstep.

### MoE Backend

Role:

- Pure MoE computation and backend-specific capability/config validation.
- Own module-level weight handling and lifecycle delegation through
  `create_weights`, `load_weights`, `post_load_weights`,
  `process_weights_after_loading`, and `pre_reload_weights`.
- Own `quantize_input` and `run_moe` shape/kernel contracts. `run_moe` must
  launch the backend compute path for every active ConfigurableMoE-compatible
  backend. Do not leave it as `NotImplementedError` while the wrapper calls an
  alternate method such as `run_with_prequant`.
- Do not implement `forward` or `forward_impl` for new ConfigurableMoE-compatible
  backends unless the user explicitly requests legacy standalone behavior; if
  required, document why the normal backend contract is insufficient.
- Declare whether the backend's cross-rank exchange is external to the kernel or
  fused inside the kernel.

Main APIs / references:

- `interface.py`: `MoE`, `scheduler_kind`, `can_implement`,
  `_supports_load_balancer`, `validate_configurable_moe` when present, and
  weight lifecycle hooks
  (`create_weights`, `load_weights`, `post_load_weights`,
  `process_weights_after_loading`, `pre_reload_weights`).
- `fused_moe_cutlass.py`: reference backend using external communication.
- `mega_moe/`: reference area for a fused-communication backend.
- `create_moe.py`: backend selection and fallback path.

Checklist:

- `can_implement()` returns clear `(False, reason)` for unsupported quant,
  dtype, shape, or hardware.
- Backend weight lifecycle hooks are implemented or explicitly rejected with a
  narrow error; `create_weights()` is safe under ConfigurableMoE deferred weight
  creation, `load_weights()` honors or rejects `allow_partial_loading`, and
  `post_load_weights()` / `process_weights_after_loading()` /
  `pre_reload_weights()` keep transformed weights and reload metadata coherent.
- The backend selects and stores the quantization method before delegating
  layout-specific weight registration/loading/transforms; callers should not
  need to reach into `quantization.py` directly.
- `run_moe` is implemented and is the method reached by ConfigurableMoE or the
  scheduler. If a helper like `run_with_prequant` exists for performance or
  naming compatibility, it is called from `run_moe`, not directly from wrapper
  policy code.
- Cross-rank exchange ownership is explicit via `scheduler_kind` and not hidden
  behind wrapper `isinstance` checks. Backends with kernel-fused exchange declare
  `MoESchedulerKind.FUSED_COMM`; normal backends use `EXTERNAL_COMM`.
- Backend-specific wrapper constraints go in a validation hook or an equivalent
  narrow contract, not in scattered forward branches.
- Weight handling remains backend API scope even when the actual tensor layout is
  implemented by a `FusedMoEMethod`.
- Do not add external host communication logic to a backend, except for a true
  fused-communication backend whose kernel owns the exchange.
- New backend tests belong in `test_moe_backend.py`.
- Existing legacy `forward` methods can be read for compatibility context, but
  they are not the default pattern for new backend work.

### Quantization And Weights

Role:

- Weight handling is backend scope at the module/API boundary: the backend
  exposes the lifecycle hooks, owns when they are called, and is accountable for
  reload/EPLB consistency.
- Quantization-specific tensor creation, loading details, post-load transforms,
  quant scales, and EPLB weight fix-ups should live in `quantization.py` as a
  backend-selected `FusedMoEMethod` implementation when they are specific to a
  quantization layout.
- When adding new weight handling, first look for a reusable existing quant
  method or base class before creating a new one, then make the backend select
  and invoke it through the lifecycle hooks.

Main APIs / references:

- `quantization.py`: `FusedMoEMethodBase`, `create_weights`, `load_weights`,
  `post_load_weights`, `setup_quant_scales`, `eplb_support_status`,
  `supports_online_eplb`, `need_load_shared_weights`.
- Existing quant methods in `quantization.py` are the reference patterns.

Checklist:

- New backend weight handling is surfaced through backend lifecycle hooks; new
  quantization-specific tensor layouts are represented by a backend-selected
  quantization method, not ad hoc caller or wrapper code.
- Existing quant method/layout is reused when the tensor layout and scale
  semantics match.
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
- `post_load_weights()` performs transforms, shared-weight setup, and scale
  setup in the quantization method only for tensors outside the base FC path;
  base FC weight registration should still flow through the base class whenever
  possible.
- `setup_quant_scales()` is updated when a quant mode exposes scales consumed by
  backend, communication, or forward-execution paths.
- EPLB support status is explicit: `SUPPORTED`, `NOT_SUPPORTED`, or
  `NOT_VERIFIED`.

### EPLB

Role:

- EPLB is cross-cutting. A correct change may need updates in interface,
  quantization, forward execution, communication, and tests.
- Do not treat EPLB as only a backend flag.

Main APIs / references:

- `interface.py`: `_supports_load_balancer`, `_add_raw_shared_weights_for_unmap`,
  `_using_load_balancer`, `_using_dynamic_load_balancer`, validation hooks.
- `quantization.py`: `eplb_support_status`, `need_load_shared_weights`,
  `register_all_parameter_slot_and_to_fix_weight_fns`, `setup_quant_scales`,
  `post_load_weights`.
- Current forward-execution code: statistic update, route, `ignore_allreduce`,
  per-chunk first/last hook ordering.
- `test_moe_module.py`: EPLB params and `generate_*_eplb_test_params`.

Checklist:

- Backend reports whether load balancing is supported.
- Quantization method declares online EPLB status.
- EPLB weight registration is split into two layers:
  1. Common MoE FC weights/biases are handled by `FusedMoEMethodBase` using
     `need_load_shared_weights(module)` in its shared-load/register flow.
  2. Quantization-specific scales, alphas, transformed weights, or layout views
     are handled by the concrete quantization method and must add their own
     `need_load_shared_weights(module)` gated shared-load/register logic.
- Shared quant-specific tensors needed by EPLB are registered in the
  quantization method, including any fix-up functions for transformed weights.
- Forward execution collects routing statistics and chooses `ignore_allreduce`
  correctly for the communication path.
- EPLB hook order is preserved around routing, `run_moe`, and CPU weight
  migration.
- `num_slots`, `num_experts`, `ep_size`, and slot-vs-expert IDs are not mixed.
- Add or update concrete EPLB tests in `test_moe_module.py`, including the
  backend/comm/quant combination that changed.

#### CPU shared-staging buffer family (EPLB migration)

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

### Communication

Role:

- External communication strategies implement dispatch/combine and expose what
  ordering they support relative to quantization.
- Backends whose kernel owns cross-rank exchange should bypass external
  communication strategies rather than being forced through the factory.

Main APIs / references:

- `communication/base.py`: `Communication`, `is_platform_supported`,
  `is_workload_feasible`, `supports_post_quant_dispatch`, `prepare_dispatch`,
  `dispatch`, `combine`.
- `communication/communication_factory.py`: strategy selection.
- Existing strategies: `nvlink_one_sided.py`, `nvlink_two_sided.py`, `deep_ep.py`,
  `allgather_reducescatter.py`.

Checklist:

- Strategy selection and forced method behavior are handled through the factory.
- `supports_post_quant_dispatch()` is correct for the payload layout.
- `prepare_dispatch()` is used only for metadata/statistics that must happen
  before dispatch.
- `dispatch()` and `combine()` maintain enough internal state for the pair to be
  correct.
- EPLB statistics gathered by the communication strategy are fed back to the
  load balancer through the forward-execution path.
- Add/update `test_moe_comm.py` or module-level tests when changing strategy
  behavior.

### Forward Execution And Chunking

Role:

- Treat `moe_scheduler.py` as the current owner of forward-time policy. Use this
  section as the detailed checklist for scheduler changes and for reviews that
  suspect policy has leaked back into the wrapper or backend.
- Keep lifecycle outside this policy: backend construction, weight loading,
  communication strategy lifetime, DWDP record, and `repeat_idx` advancement
  remain wrapper-level concerns.

Main APIs / references:

- `moe_scheduler.py`: scheduler ABC, external/fused scheduler implementations,
  chunk helpers, EPLB hook order, and backend kwargs construction.
- `configurable_moe.py`: scheduler construction and wrapper lifecycle after
  scheduler return.
- Current communication interfaces and backend `run_moe`/`quantize_input`
  contracts.
- Existing tests that exercise module forward, multi-GPU EP, EPLB, and
  communication behavior.

Checklist:

- The wrapper advances `repeat_idx` once per `forward_impl`; schedulers must not
  mutate it independently.
- External-communication scheduler respects padding, chunking, communication
  fallback, quantize/dispatch order, EPLB hooks, and output truncation.
- Fused-communication path does not call external `Communication.dispatch` or
  `combine`.
- Per-chunk EPLB first/last-call behavior is preserved.
- Multi-stream overlap is used only on paths that support it.
- Add module or focused forward-path tests for new policy, especially chunking
  and zero-token behavior.

### Routing And Factory

Role:

- Routing methods map router logits to expert or slot selections.
- Factory/config code selects a backend based on requested backend, quantization,
  hardware capability, and model config.

Main APIs / references:

- `routing.py`: routing method implementations.
- `create_moe.py`: `get_moe_cls`, `create_moe_backend`, `create_moe`.
- `moe_test_utils.py`: backend enum, backend class map, skip logic.

Checklist:

- Routing output dtype/shape matches backend and forward-execution expectations.
- Unsupported backend/quant/model combinations fall back or skip with clear
  reasons.
- Test skip logic mirrors backend `can_implement()` instead of hiding bugs with
  broad skips.

### Test Matrix And Helpers

Role:

- Keep backend, quantization, model-shape, routing, communication, and CI/local
  test matrices centralized and consistent across backend-level and module-level
  tests.
- Keep skip reasons aligned with production capability checks such as
  `can_implement()` instead of hiding failures with broad local skips.

Main APIs / references:

- `tests/unittest/_torch/modules/moe/moe_test_utils.py`: `MoeBackendType`,
  `get_backend_class`, `get_quick_skip_reason`, backend-specific
  `should_skip_*`, `iter_base_test_configs`, CI acceleration logic.
- `tests/unittest/_torch/modules/moe/quantize_utils.py`: quantized test weight
  generation and quant-parameter setup.
- `test_moe_backend.py`: backend interface tests for `quantize_input` and
  `run_moe`.
- `test_moe_module.py`: ConfigurableMoE integration matrix, multi-GPU, and EPLB
  coverage.
- `test_moe_comm.py`: communication dispatch/combine coverage.

Checklist:

- New backend is added to `MoeBackendType`, `get_backend_class`, backend/module
  matrices, and skip logic.
- New quantization method is added to test quant parameters and EPLB support
  checks when applicable.
- New unsupported combination returns a precise skip reason tied to production
  capability checks.
- CI subset and local exhaustive matrix stay intentionally different and are
  documented in the test helpers.
- Legacy tests such as `test_fused_moe.py` are used only for compatibility; new
  ConfigurableMoE behavior belongs in `test_moe_backend.py`, `test_moe_module.py`,
  or focused comm/routing/load-balancer tests.

## Design Gate

Before editing, write a short gate:

```markdown
## MoE Design Gate
- Change area: <ConfigurableMoE / MoEScheduler-forward-execution / backend / quantization-weights / EPLB / communication / routing-factory / test-matrix / tests>
- Owner boundary: <where the behavior belongs and why>
- Main API touched: <method/class names>
- Reference pattern: <existing file/class/function from references/moe-canonical-code-examples.md, with file:line evidence>
- Guide sections used: <MOE_DEVELOPER_GUIDE.md sections>
- Guide update needed: <yes/no; which section if yes>
- Refactor needed: <yes/no; one reason tied to architecture, not style>
- Test plan: <backend/module/comm/routing/EPLB/multi-GPU tests>
```

If the owner boundary is unclear, inspect more code before editing.

## Refactor Rubric

Recommend a refactor when it:

- Moves behavior to the correct owner boundary.
- Simplifies `ConfigurableMoE` while preserving its assembler role.
- Clarifies backend ownership of the weight lifecycle and quantization-method
  delegation for weights/scales.
- Makes backend capabilities and unsupported combinations explicit.
- Separates external-communication and fused-communication policies cleanly in
  `MoEScheduler` rather than wrapper/backend branches.
- Makes EPLB support testable across interface, quantization, forward execution,
  and module tests.
- Updates shared test matrices/helpers when backend, quantization, or skip
  semantics change.
- Reduces duplicate dispatch/chunking/EPLB ordering logic by centralizing
  forward-time policy in `moe_scheduler.py` without changing
  performance-critical semantics.

Reject or question a refactor when it:

- Adds backend-specific forward branches to `ConfigurableMoE` instead of
  selecting behavior through `MoESchedulerKind` / `MoEScheduler`.
- Moves weight layout logic out of quantization methods without a strong reason.
- Hides hardware or quantization constraints behind vague abstractions.
- Changes communication/EPLB ordering without tests.
- Adds one-off skips in individual tests instead of shared capability/skip helpers.
- Touches legacy MoE paths for new features when the ConfigurableMoE path should
  be used.

## Review Output

For reviews, lead with findings and concrete references:

```markdown
## Findings
- [High] <file:line> <architecture, correctness, or testability issue>
- [Medium] <file:line> <maintainability or boundary issue>
- [Low] <file:line> <local cleanup>

## Architecture Fit
- ConfigurableMoE remains assembler: <yes/no>
- Owner boundaries respected: <yes/no>
- Scheduler boundary respected: <yes/no; forward policy in `moe_scheduler.py`, lifecycle in wrapper, compute in backend>
- Refactor recommended: <yes/no + reason>

## Guide Alignment
- Sections checked: <MOE_DEVELOPER_GUIDE.md sections>
- Guide update needed: <yes/no + section>

## Checklist Coverage
- Weights/quantization: <covered/gap>
- EPLB: <covered/gap>
- Communication: <covered/gap>
- MoEScheduler/forward execution: <covered/gap>
- Backend: <covered/gap>
- Forward execution/chunking details: <covered/gap>
- Test matrix/helpers: <covered/gap>
- Tests: <covered/gap>
```

If there are no findings, say so and list remaining test or performance risk.

## Test Selection

Prefer the unified MoE tests:

- Shared test matrix/helper changes: inspect `tests/unittest/_torch/modules/moe/moe_test_utils.py` and `quantize_utils.py`, then run the affected backend/module tests below.
- Backend interface changes: `pytest tests/unittest/_torch/modules/moe/test_moe_backend.py -k '<backend or quant>'`.
- Module/create/forward changes: `pytest tests/unittest/_torch/modules/moe/test_moe_module.py -k '<backend or feature>'`.
- Communication changes: `pytest tests/unittest/_torch/modules/moe/test_moe_comm.py -k '<strategy>'`.
- Routing changes: `pytest tests/unittest/_torch/modules/test_moe_routing.py -k '<routing>'`.
- Load balancer changes: `pytest tests/unittest/_torch/modules/test_moe_load_balancer.py -k '<case>'`.
- Multi-GPU EP/all-to-all behavior: `pytest tests/unittest/_torch/multi_gpu/test_moe_a2a.py -k '<case>'`.

When GPU resources are required, use the TRT-LLM GPU allocation/test-runner
skills first and record skipped tests with reasons.
