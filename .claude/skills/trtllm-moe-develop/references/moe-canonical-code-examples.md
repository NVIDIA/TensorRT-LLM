# MoE Canonical Code Examples

Use this reference to find concrete implementation patterns before designing,
editing, or reviewing MoE code. Open only the section relevant to the change and
then inspect the actual repository files.

## ConfigurableMoE Assembler

Use these examples to understand wrapper-level lifecycle and composition:

- `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py`
  - `ConfigurableMoE.__init__`: backend construction, communication creation,
    chunking-related wrapper state, validation, DWDP setup, and weight-removal
    lifecycle.
  - `_create_and_sync_backend`: backend construction and wrapper/backend attr
    synchronization.
  - `_create_comm_strategy_auto`: communication strategy creation or bypass when
    the backend owns cross-rank exchange.
  - `forward_impl`: wrapper-level forward entry; should stay thin and delegate
    to `self.scheduler.forward(...)`.
  - `validate_backend` / `validate_config`: capability and wrapper/backend
    compatibility checks.

Good uses:

- Add wrapper lifecycle once and delegate policy-heavy behavior out.
- Keep backend-specific constraints in validation hooks or construction paths.

Red flags:

- New direct backend type checks in `forward_impl`.
- Chunking, communication ordering, or EPLB hook sequencing added directly to
  the wrapper instead of `moe_scheduler.py`.
- Weight layout or communication protocol logic added directly to the wrapper.

## MoE Scheduler / Forward Execution

Use these examples when changing forward policy, chunking, communication order,
EPLB hook order, or fused-communication behavior:

- `tensorrt_llm/_torch/modules/fused_moe/moe_scheduler.py`
  - `MoEScheduler`: abstract forward strategy.
  - `ExternalCommMoEScheduler`: host-side communication path, chunking, adaptive
    quantize/dispatch ordering, communication fallback, multi-stream overlap,
    output truncation, and external-comm EPLB statistic paths.
  - `FusedCommMoEScheduler`: MegaMoE-style fused-communication path, ADP
    stripping, consistent per-rank chunk count, zero-token kernel launches,
    no external `Communication`, `ignore_allreduce=False`, and backend
    `run_moe` invocation.
  - `create_moe_scheduler`: factory keyed by `backend.scheduler_kind`.
- `tensorrt_llm/_torch/modules/fused_moe/interface.py`
  - `MoESchedulerKind`: backend-declared selection axis for external vs fused
    cross-rank exchange.
- `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py`
  - Scheduler is constructed after backend/comm/chunking/DWDP state is ready;
    wrapper `forward_impl` delegates and then runs shared lifecycle bookkeeping.

Good uses:

- Put forward-time policy in `moe_scheduler.py`, leaving ConfigurableMoE as the
  assembler and backends as compute/weight lifecycle owners.
- Keep `repeat_idx` advancement and DWDP record in ConfigurableMoE, after the
  scheduler returns.
- Use `backend.scheduler_kind` to choose external-vs-fused communication path.
- Call `backend.quantize_input(...)` and `backend.run_moe(...)`; keep
  backend-specific kwargs centralized and narrow.
- Preserve zero-token and per-chunk first/last semantics when changing chunking.

Red flags:

- Wrapper or backend reimplements chunk splitting, EPLB hook sequencing,
  dispatch/combine ordering, or fused-communication lockstep.
- Scheduler mutates `repeat_idx`, performs DWDP lifecycle work, calls weight
  lifecycle hooks, or creates/destroys communication objects.
- Fused-communication backend is forced through external `Communication` or a
  normal backend bypasses communication factory selection.
- Scheduler calls backend-specific compute helpers instead of `run_moe`.

## MoE Backend

Use these examples to decide what belongs in a backend:

- `tensorrt_llm/_torch/modules/fused_moe/interface.py`
  - `MoE`: current transitional base contract shared by legacy standalone MoE
    modules and newer ConfigurableMoE-compatible backends. A dedicated backend
    interface is expected in the future; do not treat legacy `forward` methods
    as the default pattern for new backend work.
  - `can_implement`: capability gate returning `(bool, reason)`.
  - `_supports_load_balancer`: backend EPLB capability.
  - `validate_configurable_moe` when present: backend-specific wrapper
    constraints.
  - `create_weights`, `load_weights`, `post_load_weights`,
    `process_weights_after_loading`, and `pre_reload_weights`:
    backend-owned lifecycle entrypoints for module-level weight handling.
  - `quant_method` and `_get_quant_method` in concrete backends: quantization
    method selection and delegation.
  - `quantize_input` and `run_moe`: backend-facing compute contract. For active
    ConfigurableMoE-compatible backends, `run_moe` must be a concrete compute
    entrypoint reached by the wrapper/scheduler. Alternate helpers such as
    `run_with_prequant` should sit behind `run_moe`, not replace it.
  - `forward` and `forward_impl`: legacy standalone-module entrypoints only;
    avoid adding them to new ConfigurableMoE-compatible backends unless the user
    explicitly requests that behavior.
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`
  - Reference backend for normal external communication and broad quant support.
  - Inspect `_get_quant_method`, `create_weights`, `load_weights`, and
    `post_load_weights` for the standard backend-to-quantization delegation
    pattern.
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py`
  - Reference backend for TRTLLMGen-specific input/scale contracts.
  - Inspect `_get_quant_method`, alignment validation, weight lifecycle
    delegation, and quant scale usage.
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py`
  - Reference backend for DeepGEMM workspace and Blackwell FP8 block-scale path.
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_densegemm.py` when present
  - Reference for a direct `MoE` backend with dense GEMM constraints.
  - Inspect backend-owned `create_weights`, `load_weights`, and
    `post_load_weights` delegation.
- `tensorrt_llm/_torch/modules/fused_moe/mega_moe/` when present
  - Reference area for a backend whose kernel owns cross-rank exchange. Inspect
    capability checks, load-balancer support, validation, weight lifecycle
    delegation, `quantize_input`, and `run_moe`.

Good uses:

- Put kernel shape/dtype/quant constraints in `can_implement`.
- Let the backend own the module-level weight lifecycle (`create_weights`,
  `load_weights`, `post_load_weights`, `process_weights_after_loading`,
  `pre_reload_weights`) and quantization-method selection, then delegate
  quantization-specific tensor layout, loading details, post-load transforms,
  and scale setup to `quant_method`.
- Confirm `ConfigurableMoE` delegates wrapper weight lifecycle to the backend,
  and the backend delegates tensor-layout details to the quantization method.
- Implement backend compute through `quantize_input` and `run_moe`; read legacy
  `forward` methods only as compatibility context unless the user explicitly
  requests standalone module behavior. Treat an active backend with
  `run_moe = NotImplementedError` plus wrapper calls to `run_with_prequant` as
  a contract violation to fix or call out in the design.
- Put backend-specific wrapper constraints in a narrow validation hook.
- Keep host-side external communication out of backends unless the kernel truly
  owns the exchange.

Red flags:

- New ConfigurableMoE-compatible backend implements `forward` or `forward_impl`
  without explicit user request for legacy standalone behavior.
- Wrapper or scheduler calls a backend-specific compute helper such as
  `run_with_prequant` directly while the backend's `run_moe` is empty or unused.
- Backend adds a new compute path but leaves `create_weights`, `load_weights`,
  `post_load_weights`, `process_weights_after_loading`, `pre_reload_weights`,
  or reload behavior inconsistent.
- Backend bypasses an appropriate quantization method for layout-specific
  weights, scales, or transformed-weight fixups.
- Backend starts orchestrating external dispatch/combine or wrapper lifecycle.
- Unsupported cases assert late in forward instead of returning clear skip/fall
  back reasons through capability checks.

## Quantization And Weights

Use these examples before adding any new weight handling. Pair this section with
the backend section: backend owns the module-level lifecycle entrypoints and
method selection; quantization owns the delegated tensor layout, loading
details, transforms, and scale registration for a quantization layout:

- `tensorrt_llm/_torch/modules/fused_moe/quantization.py`
  - `FusedMoEMethodBase`: common weight create/load/post-load flow, shared weight
    loading, EPLB hooks, and quant scale setup.
  - `UnquantizedFusedMoEMethod`: simple supported-EPLB baseline.
  - `FP8QDQFusedMoEMethod`: example of explicit unsupported-EPLB status and
    scale setup.
  - `DeepSeekFP8BlockScalesFusedMoEMethod`: FP8 block-scale loading and scale
    handling.
  - `NVFP4FusedMoEMethod`: NVFP4 weight layout, scale setup, and shared-weight
    registration patterns.
  - `MXFP4WeightFusedMoEMethod`: MXFP4 weight layout and online EPLB pattern.
  - Backend-specific methods when present: reference for specialized transformed
    weights.

Good uses:

- Backend lifecycle hooks surface the weight behavior; new quantization-specific
  tensor layouts are represented as backend-selected quantization methods.
- Existing method/base class is reused when tensor and scale semantics match.
- `create_weights`, `load_weights`, `post_load_weights`, and
  `setup_quant_scales` stay together for a quant layout.
- Transformed weights needed by EPLB are registered through quantization-owned
  fix-up hooks.

Red flags:

- Backend lifecycle hooks bypass an existing matching quantization abstraction
  and hand-roll quantization-specific create/load/transform details inline.
- Scales consumed by backend or communication are not registered in
  `setup_quant_scales`.

## EPLB / Load Balancing

Use these examples when touching dynamic expert migration or slot routing:

- `tensorrt_llm/_torch/modules/fused_moe/interface.py`
  - `_using_load_balancer`, `_using_dynamic_load_balancer`.
  - `_load_balancer_update_statistic` and
    `_load_balancer_update_statistic_with_gathered_statistic`.
  - `_load_balancer_route` and slot/expert-ID conversion.
  - `register_all_parameter_slot_and_to_fix_weight_fns`.
- `tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py`
  - Slot registration, expert migration, per-layer update flow, and iteration
    context.
- `tensorrt_llm/_torch/modules/fused_moe/quantization.py`
  - `eplb_support_status`, `supports_online_eplb`,
    `need_load_shared_weights`, shared-weight registration, and transformed
    weight fix-up hooks.
- Current forward-execution code
  - EPLB wait/update/route hook order, per-chunk first/last behavior,
    statistics gathering, and allreduce bypass semantics.
- `tests/unittest/_torch/modules/moe/test_moe_module.py`
  - `_create_moe_load_balancer`, `_run_eplb_test`, `_should_skip_EPLB`,
    `generate_eplb_test_params`, and backend-specific EPLB param generators
    when present.

Good uses:

- Keep expert IDs and slot IDs explicit.
- Preserve the two-layer EPLB registration split: common MoE FC weights and
  biases are loaded/staged/registered by `FusedMoEMethodBase.load_weights()` and
  `FusedMoEMethodBase.post_load_weights()` using `need_load_shared_weights(module)`;
  concrete quantization methods add the same gate only for their own scales,
  alphas, transformed weights, or layout-specific views.
- Register all quantization-specific raw/transformed tensors required for slot
  migration, without duplicating the base FC weight registration path.
- Add module-level tests that force expert movement and validate output.

Red flags:

- EPLB support is declared only in backend without quantization/test updates.
- Concrete quantization method duplicates common FC shared-weight registration
  already handled by `FusedMoEMethodBase`, or forgets to add
  `need_load_shared_weights(module)` gated registration for its own scale/layout
  tensors.
- Statistics are gathered through communication but never fed back to the load
  balancer.

### CPU shared-staging buffer family  --  concrete code pointers

Use these exact locations when adding or reviewing a per-expert Parameter that
must survive online EPLB migration.

- Bulk FC weights (base-class path, every EPLB-supporting quant method):
  - `tensorrt_llm/_torch/modules/fused_moe/quantization.py::FusedMoEMethodBase.load_weights`
     --  `need_load_shared_weights(module)` branch allocates and fills
    `module.local_shared_w3_w1_tensors` / `module.local_shared_w2_tensors`
    (and bias twins when `module.bias`) sized
    `(len(local_shared_load_expert_ids),) + weight.shape[1:]` on CPU.
  - `FusedMoEMethodBase.post_load_weights`  --  builds `weight_fns` with the
    bulk-weight staging buffers and calls
    `module.register_all_parameter_slot_and_to_fix_weight_fns(weight_fns)`,
    then `delattr`s the staging attributes and finalizes host sharing via
    `module.layer_load_balancer.host_tensor_sharer.finalize_layer_weights()`.

- Block scales (NVFP4 concrete path):
  - `NVFP4FusedMoEMethod.load_quant_scales`  --  `need_load_shared_weights` branch
    allocates `module.local_shared_w3_w1_scale_tensors` /
    `module.local_shared_w2_scale_tensors`, fills `tmp_shared_weight_scale_2`
    dict.
  - `NVFP4FusedMoEMethod.process_weights_after_loading` step 4  --  allocates
    `shared_fc31_alpha`, `shared_fc2_alpha`, `shared_fc31_weight_scale_2`,
    `shared_fc2_weight_scale_2` (function-local CPU tensors sized
    `(num_shared,) + ...`), calls `_reconcile_and_compute_alphas` with them as
    explicit destinations, then composes `weight_fns` and registers.

- Per-expert scalar Parameters (alpha / weight_scale_2):
  - `NVFP4FusedMoEMethod.create_weights`  --  registers `fc31_alpha` / `fc2_alpha`
    unconditionally, and registers `fc31_weight_scale_2` / `fc2_weight_scale_2`
    only when `getattr(module, 'force_dynamic_quantization', False)` (see
    `TorchLlmArgs.force_dynamic_quantization`,
    `_torch/modules/linear.py` for the runtime-amax activation-quant path that
    reads `weight_scale_2` at forward time, and PR #12320 for the
    runtime-weight-update path that reads it when reloading weights).
  - `_reconcile_and_compute_alphas(module, tmp, dst_fc31_alpha, dst_fc2_alpha,
    dst_fc31_weight_scale_2=None, dst_fc2_weight_scale_2=None)`  --  single body
    serving both routed and shared paths; callers pass distinct `dst_` tensors
    per index space so the function never closes over
    `module.fc31_weight_scale_2.data` directly.

- Migration hook wiring:
  - `interface.py::FusedMoE.register_all_parameter_slot_and_to_fix_weight_fns`
    is what actually connects CPU staging buffers to the
    `MoeLoadBalancer.single_layer_load_balancers[...]` slot infrastructure.
  - `moe_load_balancer.py::SingleLayerMoeLoadBalancer.__init__` computes
    `load_expert_ids = range(shared_rank * expert_count // shared_size, ...)`,
    which is the authoritative source of `local_shared_load_expert_ids` and
    `num_shared`.
  - `moe_load_balancer.py::MoeLoadBalancer._setup_mpi_comm` performs the
    `MPI.COMM_TYPE_SHARED` split and asserts
    `shared_size == local_mpi_size()`; this is the single-node vs multi-node
    boundary that determines whether
    `num_shared > expert_size_per_partition` is reachable.

Good uses:

- Treat the staging-buffer family as a symmetric contract: every per-expert
  Parameter (whether weight, scale, alpha, or scale_2) should have a matching
  CPU staging buffer and appear in `weight_fns`.
- Keep reconcile/compute/fix-up helpers parameterized on destination tensors
  rather than reading `module.<param>.data` directly  --  that way routed and
  shared paths share the body without index-space leakage.

Red flags (NVBug 6130334 / PR #13856 pattern):

- A new per-expert `nn.Parameter` added to `create_weights()` but never
  mirrored into a staging buffer and never added to `weight_fns`  --  first EPLB
  migration will leave it stale, silently corrupting quantization state.
- Reconcile/compute logic writes into `module.<per_expert_param>.data[expert_idx]`
  while `expert_idx` comes from `enumerate(local_shared_load_expert_ids)`  -- 
  multi-node runs IndexError, single-node runs silently overwrite routed
  slots.
- Adding one half of a pair (`fc31_weight_scale_2`) to `weight_fns` but
  forgetting the twin (`fc2_weight_scale_2`), or gating them on different
  conditions  --  asymmetric migration leaves half the state behind.

Regression pattern: a CPU-only single-GPU pytest that directly invokes the
reconcile/compute helper with a crafted `tmp_shared_*` dict and oversized
index range is the only pre-merge check that catches this class of bug  -- 
the existing `test_configurable_moe_multi_gpu_eplb` cannot, because its
single-node topology forces `num_shared <= expert_size_per_partition` and
routed/shared index spaces overlap on the same global expert IDs.

## Communication

Use these examples when changing dispatch/combine behavior:

- `tensorrt_llm/_torch/modules/fused_moe/communication/base.py`
  - `Communication` ABC, `supports_post_quant_dispatch`, `prepare_dispatch`,
    `dispatch`, and `combine`.
- `communication/communication_factory.py`
  - Strategy selection, forced-method behavior, feasibility checks, and fallback.
- `communication/allgather_reducescatter.py`
  - Baseline external communication strategy.
- `communication/nvlink_two_sided.py`
  - Pre-dispatch metadata/statistics pattern.
- `communication/nvlink_one_sided.py`
  - Dispatch-time statistics/workspace pattern.
- `communication/deep_ep.py` and `communication/deep_ep_low_latency.py`
  - DeepEP pre/post quant dispatch constraints.
- `tests/unittest/_torch/modules/moe/test_moe_comm.py`
  - Focused communication behavior tests.

Good uses:

- External dispatch/combine state stays inside the communication strategy.
- `supports_post_quant_dispatch` matches actual payload layout.
- EPLB statistics collected by communication are returned to the forward path.

Red flags:

- Wrapper or backend duplicates communication state that belongs in a strategy.
- New strategy lacks direct tests or module-level coverage.

## Forward Execution And Chunking

Use these examples when wrapper forward policy grows complicated:

- `tensorrt_llm/_torch/modules/fused_moe/moe_scheduler.py`
  - Scheduler forward entry, external/fused scheduler implementations, routing
    order, dispatch order, EPLB hook order, zero-token behavior, output
    truncation, and backend kwargs.
- `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py`
  - Wrapper entry and lifecycle boundary around scheduler execution.
- Backend forward paths
  - Useful only for legacy compatibility context or for identifying policy that
    should move into the scheduler.
- `tests/unittest/_torch/modules/moe/test_moe_module.py`
  - Module-level multi-GPU, chunking, routing, and EPLB cases.
- `tests/unittest/_torch/multi_gpu/test_moe_a2a.py`
  - Multi-GPU all-to-all behavior when relevant.

Good uses:

- Keep policy-heavy forward flow in `moe_scheduler.py` rather than adding wrapper
  or backend branches.
- Keep wrapper lifecycle state outside scheduler forward policy.
- Preserve per-chunk first/last semantics and zero-token behavior.

Red flags:

- New backend-specific branch in wrapper forward flow.
- Chunking, EPLB, and communication ordering spread across several owners.

## Routing And Factory

Use these examples when changing model-to-backend selection or routing output:

- `tensorrt_llm/_torch/modules/fused_moe/routing.py`
  - Routing method output shape/dtype and top-k behavior.
- `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`
  - `get_moe_cls`, `create_moe_backend`, `create_moe`, backend fallback, and
    quantization-specific selection.
- `tests/unittest/_torch/modules/moe/moe_test_utils.py`
  - Backend enum, class map, quick skip reason, CI/local matrix generation.

Good uses:

- Keep unsupported combinations tied to production capability checks.
- Make fallback and skip reasons precise.

Red flags:

- Test helper skips diverge from `can_implement` behavior.
- Factory code silently falls back without a useful reason.

## Test Matrix And Helpers

Use these examples before adding backend, quantization, routing, or EPLB tests:

- `tests/unittest/_torch/modules/moe/moe_test_utils.py`
  - `MoeBackendType`, `get_backend_class`, backend-specific `should_skip_*`,
    `get_quick_skip_reason`, `supports_autotuner_capture`,
    `iter_base_test_configs`, and CI acceleration logic.
- `tests/unittest/_torch/modules/moe/quantize_utils.py`
  - Quantized test weight generation, reference module selection, backend-aware
    weight preparation.
- `tests/unittest/_torch/modules/moe/test_moe_backend.py`
  - Backend-level `quantize_input` and `run_moe` contracts.
- `tests/unittest/_torch/modules/moe/test_moe_module.py`
  - ConfigurableMoE integration matrix, multi-GPU, and EPLB coverage.
- `tests/unittest/_torch/modules/moe/test_moe_comm.py`
  - Communication strategy tests.

Good uses:

- Add new backend/quant support through shared helpers first.
- Keep local CI subset and broader local matrix intentionally different.
- Every skip reason should map to a real production capability or environment
  constraint.

Red flags:

- One test file has a backend case but shared helper enum/skip matrix is stale.
- Broad skips hide unsupported combinations instead of documenting them.
