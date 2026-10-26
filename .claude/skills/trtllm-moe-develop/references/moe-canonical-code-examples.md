# MoE Canonical Code Examples

Where to look for each pattern before designing, editing, or reviewing MoE
code. Open only the section relevant to the change, then read the actual files.
The rules themselves live in `MOE_DEVELOPER_GUIDE.md` and the skill; this file
only names the code that demonstrates them. Paths are relative to
`tensorrt_llm/_torch/moe/fused_moe/` unless shown in full.

## ConfigurableMoE Assembler

- `configurable_moe.py`
  - `ConfigurableMoE.__init__`: backend construction, communication creation,
    chunking-related wrapper state, validation, DWDP setup, weight-removal
    lifecycle, and the scheduler constructed last.
  - `_create_and_sync_backend`: backend construction and wrapper/backend attr
    synchronization.
  - `_create_comm_strategy_auto`: communication strategy creation, or `None`
    when the backend owns cross-rank exchange.
  - `forward_impl`: thin delegation to `self.scheduler.forward(...)`, then the
    shared bookkeeping.
  - `validate_backend` / `validate_config`: capability and wrapper/backend
    compatibility checks.
  - `determine_communication_method`: the one sanctioned way a scheduler
    changes `moe.comm`.

## MoE Scheduler / Forward Execution

- `moe_scheduler.py`
  - `MoEScheduler`: abstract forward strategy.
  - `ExternalCommMoEScheduler`: host-side communication, chunking, adaptive
    quantize/dispatch ordering, communication fallback, multi-stream overlap,
    output truncation, and external-comm EPLB statistic paths.
  - `FusedCommMoEScheduler`: fused-communication lockstep, ADP stripping,
    consistent per-rank chunk count, zero-token kernel launches,
    `ignore_allreduce=False`.
  - `create_moe_scheduler`: factory keyed by `backend.scheduler_kind`.
- `interface.py`: `MoESchedulerKind`.
- `tests/unittest/_torch/moe/test_moe_module.py`: module-level multi-GPU,
  chunking, routing, and EPLB cases;
  `tests/unittest/_torch/moe/multi_gpu/test_nvlink_one_sided.py` for multi-GPU
  all-to-all.

## Backend Selection And Identity

- `moe_resolution.py`: `resolve_moe_impl` builds `MoEProblem` and
  `MoEDeployment`, walks `IMPL_PRIORITY` / `BACKEND_FAMILY` through
  `_candidates_for`, and returns the report; `_UNRANKED` is the import-time
  table check; `_reject_unsupported_activation` reads each candidate's
  `activation_support`.
- `impl_contract.py`: `MoEProblem` / `MoEDeployment` (everything a gate may
  read, including `MoEProblem.routing`), `MoEEligibility`, `MoERejectReason`,
  `MoEResolutionReport.degraded_from`, and the two halves of "a leaf is one
  format": `check_quant_matches_identity` at selection,
  `require_layer_quant_format` at `create_weights`.
- `impl_environment.py`: `collect_moe_environment`, the one place that probes
  SM, imports, and environment variables, and `override_moe_environment` for
  tests.
- `impl_identity.py`: `MoEImplId`, `MoEImplQuery`, `MOE_IMPL_REGISTRY`,
  `register_moe_impl`.
- `create_moe.py`: `create_moe_backend` matches with `issubclass`, so one
  branch admits every leaf of a family (the CUTLASS branch also takes the B12x
  leaves).
- `tensorrt_llm/_torch/models/modeling_utils.py`:
  `moe_experts_quant_config`, which every MoE model without a lookup of its own
  (for example `Qwen3MoE` in `modeling_qwen3_moe.py`, `NemotronHMOE` in
  `modeling_nemotron_h.py`) passes to `create_moe` as `override_quant_config`.
- `tests/unittest/_torch/moe/test_moe_impl.py`: for example
  `test_pinned_identity_fails_hard_where_the_backend_literal_degrades`,
  `test_fp8_block_scales_leaves_admit_only_what_their_kernel_runs`,
  `test_a_leaf_refuses_a_layer_whose_format_moved_after_it_was_picked`, and
  the hand-listed `_SINGLE_FORMAT_IMPLS`.

## MoE Backend

- `impl_base.py`: `MoEImplBase` (four abstract methods, no `forward`) and
  `apply_moe_impl_construction_state()`.
- `impl_blocks.py`: `MoEExecutionContractMixin` (scheduler-facing
  declarations, `validate_configurable_moe`, `forward_fake`),
  `MoEWeightOwnerMixin` (the default lifecycle hooks, delegating to
  `self.quant_method`), `MoEEplbWeightLayoutMixin`.
- One class per identity: `fused_moe_deepgemm.py`
  (`DeepgemmCudaFp8BlockScalesImpl`) and `fused_moe_densegemm.py`
  (`TrtllmCutedslDenseGemmNvfp4Impl`).
- Family root plus leaves:
  - `cutlass/`: `CutlassFusedMoEBase` in `base.py` (construction, the
    layer-format check in `create_weights`, the default `quantize_input` /
    `run_moe`, `_run_fused_moe_op` configured per leaf by `FusedMoeOpFlags`,
    and the `quantize_input` implementations several formats bind); one leaf
    module per format, listed in `CUTLASS_LEAVES` in `__init__.py` next to
    `find_cutlass_leaf` / `cutlass_leaf`; gates composed in `eligibility.py`.
  - `trtllm_gen/`: per-kernel-ABI layers (`fp4_block_scale.py`,
    `fp8_block_scale.py`) between `TrtllmGenFusedMoEBase` and the
    `trtllm_<quant>.py` / `flashinfer_<quant>.py` leaves.
  - `marlin/`: one kernel ABI, so `MarlinFusedMoEBase` implements three of the
    four abstract methods and a leaf is an identity plus `can_implement`.
- Inheriting another implementation: `fused_moe_cute_dsl_b12x.py`
  (`CuteDslB12xNvfp4FusedMoE`, `CuteDslB12xFusedMoEBase`); the guide's B12x
  note says what that inheritance costs.
- `mega_moe/`: `FUSED_COMM` implementations; read capability checks,
  validation, symmetric-memory allocation in `create_weights`,
  `quantize_input`, and `run_moe`.

## Quantization And Weights

- `quantization.py`
  - `FusedMoEMethodBase`: common weight create/load/post-load flow, shared
    weight loading, EPLB hooks, and quant scale setup.
  - `UnquantizedFusedMoEMethod`: simple supported-EPLB baseline.
  - `FP8QDQFusedMoEMethod`: explicit unsupported-EPLB status and scale setup.
  - `DeepSeekFP8BlockScalesFusedMoEMethod`: FP8 block-scale loading and scale
    handling.
  - `NVFP4FusedMoEMethod`: NVFP4 weight layout, scale setup, and shared-weight
    registration patterns.
  - `MXFP4WeightFusedMoEMethod`: MXFP4 weight layout and online EPLB pattern.
- Leaf `_get_quant_method`, for example `cutlass/nvfp4.py` and the three
  overrides in `trtllm_gen/fp4_block_scale.py`: one leaf, one format, one
  method.

## EPLB / Load Balancing

- `impl_contract.py`: `MoEStaticCapability.supports_eplb`, the selection-time
  answer; its comment explains why it is not `_supports_load_balancer()`.
- `impl_blocks.py` (`MoEEplbWeightLayoutMixin`): `_supports_load_balancer`,
  `_using_load_balancer`, `_using_dynamic_load_balancer`,
  `_add_raw_shared_weights_for_unmap`,
  `register_all_parameter_slot_and_to_fix_weight_fns`.
- `trtllm_gen/base.py`: the `_supports_load_balancer` override that means
  "routing is separated from the kernel".
- `interface.py` (`MoE`): `_load_balancer_*` stage hooks,
  `_load_balancer_update_statistic`,
  `_load_balancer_update_statistic_with_gathered_statistic`, and
  `_load_balancer_route` with slot/expert-ID conversion.
- `fused_moe_cute_dsl_b12x.py`: a gate that declines `d.eplb_enabled` although
  the inherited `_supports_load_balancer()` answers True.
- `moe_load_balancer.py`: slot registration, expert migration, per-layer
  update flow, and iteration context.
- `quantization.py`: `eplb_support_status`, `supports_online_eplb`,
  `need_load_shared_weights`, shared-weight registration, and transformed
  weight fix-up hooks.
- `tests/unittest/_torch/moe/test_moe_module.py`: `_create_moe_load_balancer`,
  `_run_eplb_test`, `_should_skip_EPLB`, `generate_eplb_test_params`, and
  backend-specific EPLB param generators when present.

### CPU shared-staging buffer family  --  concrete code pointers

Use these exact locations when adding or reviewing a per-expert Parameter that
must survive online EPLB migration.

- Bulk FC weights (base-class path, every EPLB-supporting quant method):
  - `tensorrt_llm/_torch/moe/fused_moe/quantization.py::FusedMoEMethodBase.load_weights`
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
  - `impl_blocks.py::MoEEplbWeightLayoutMixin.register_all_parameter_slot_and_to_fix_weight_fns`
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

- `communication/base.py`: `Communication` ABC, `supports_post_quant_dispatch`,
  `prepare_dispatch`, `dispatch`, `combine`, `checkpoint_blocked_reason`
  (default `None`), and the `CheckpointableCommunication` protocol.
- `moe_comm_timeout_guard.py`: `MoECommTimeoutGuard` and the
  `MoECommTimeoutProxy` a kernel registers at construction.
- `communication/communication_factory.py`: strategy selection, forced-method
  behavior, feasibility checks, and fallback.
- `communication/allgather_reducescatter.py`: baseline external strategy.
- `communication/nvlink_two_sided.py`: pre-dispatch metadata/statistics.
- `communication/nvlink_one_sided.py`: dispatch-time statistics/workspace, and
  a `checkpoint_blocked_reason` override.
- `communication/deep_ep.py` and `communication/deep_ep_low_latency.py`:
  DeepEP pre/post quant dispatch constraints.
- `tests/unittest/_torch/moe/test_moe_comm.py`: focused communication tests.

## Test Matrix And Helpers

- `tests/unittest/_torch/moe/moe_test_utils.py`: `MoeBackendType` (one value
  per family), `get_backend_class`, `find_backend_class` /
  `find_backend_classes` (a family resolves to the leaf for `quant_algo`;
  absence returns `None` and `()` respectively), backend-specific `should_skip_*`,
  `get_quick_skip_reason`, `supports_autotuner_capture`,
  `iter_base_test_configs`, and CI acceleration logic.
- `tests/unittest/_torch/moe/quantize_utils.py`: quantized test weight
  generation, reference module selection, backend-aware weight preparation.
- `tests/unittest/_torch/moe/test_moe_backend.py`: backend-level
  `quantize_input` and `run_moe` contracts.
- `tests/unittest/_torch/moe/test_moe_routing.py`: routing method tests.
