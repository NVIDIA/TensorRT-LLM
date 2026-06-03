# Rebase PR 14751 onto github/main - 2026-06-10

Branch: `user/fanrongl/add-dsv4-pro-disagg-tests`

Base after rebase: `github/main` at `90cb7ffcc3`

Backup branch before rebase: `backup/add-dsv4-pro-disagg-tests-prerebase-20260610_090824`

Final relation to `github/main`: `0 behind / 155 ahead`

## Summary

Rebased the DSV4 PR branch onto latest `github/main`. The rebase had conflicts mostly from long-lived DSV4 changes overlapping with recent mainline CI, KV cache, disaggregated serving, MoE, and CUDA/CuTe updates.

General policy used:

- Keep mainline infrastructure updates unless the DSV4 branch had a newer, specific replacement.
- Preserve DSV4 model, disaggregated serving, NVFP4, KV cache, and CI coverage changes.
- For CI stage conflicts, keep current main stage matrix and add or move only the DSV4 stages required by each commit.
- For generated FMHA binary artifacts, take the DSV4 PR side for the artifact-update commit, because that commit's purpose was to update V4 FMHA cubins/libs.

## Conflict Resolutions

### DeepSeek-V4 model support

Resolved model-support conflicts by keeping mainline registry/defaults and layering DSV4 entries on top:

- `_torch/configs/__init__.py`: kept main registrations for `deepseek_v32`, `kimi_k2`, and `laguna`; added DeepSeekV4.
- `llmapi/__init__.py`: kept main exports such as CUDA graph and ThinkingBudget; added `DeepSeekV4SparseAttentionConfig`.
- `model_config.py`: kept main composite `text_config.dtype` fallback/imports; added DSV4 `is_hybrid_linear`.
- `config_utils.py` and docs: kept main model additions and added DSV4 mappings/rows.
- `modeling_speculative.py`: kept Step3 MTP support and DSV4 MTP support.

### KV cache manager and router changes

Resolved repeated conflicts by keeping the newer mainline `ReuseScope`/`TreeTaskId` style and applying DSV4-specific probes/stats on top:

- `_block_radix_tree.py`, `_kv_cache.py`, `_kv_cache_manager.py`, `resource_manager.py`, router tests: kept `ReuseScope` APIs, salt/lora handling, dirty stats, and expected prompt length accounting.
- `_util.py`: retained main `cache_transceiver_config` plumbing and DSV4 manager options; removed duplicate `is_disagg` assignments introduced by overlapping cherry-picks.
- `router.py`/`test_router.py`: kept main async tokenization and `BlockHashMixin` tests; added tool-calling/chat-template forwarding coverage.

### Disaggregated serving and ADP dummy handling

Merged DSV4 disaggregated fixes with current PyExecutor changes:

- `py_executor.py`: kept current `BatchState` timing/event-release path, added pending iter stats flushing, preserved ADP dummy request ID and CTX/GEN dummy role handling.
- `request_utils.py`: kept both helper families.
- `proxy.py`: kept dummy result discard in `GenerationExecutorProxy`.
- `transceiver.py` and cache reuse tests: preserved SWA clamp fixes and ctx/gen behavior split.

### MoE, DeepGEMM, CuteDSL, and FP8/NVFP4 paths

Resolved MoE/CUDA/CuTe conflicts by preserving current mainline scheduling and CUDA 13 compatibility while applying DSV4-specific fast paths:

- `configurable_moe.py`: kept main `resolve_moe_cls` and deepcopy `backend_model_config`; added `swiglu_limit_scalar` propagation.
- `fused_moe_cutlass.py`: merged LoRA parameters with DSV4 `input_ids` routing and fixed stale local typo.
- CuteDSL MoE files: kept main `fence_proxy("async.shared", space="cta")` CUDA 13 style and preserved DSV4 `o_a_proj`/swiglu limit fast paths.
- `attentionOp.cpp`: kept main `AttentionWorkspaceManager` layout; redirected `workspaceViews.fp8QBuf` to `params.mla_param->quant_q_buf` only when MLA fused Q FP8 quantization is enabled.
- `deep_ep_low_latency.py`: kept main SM120/121 platform rejection via `get_sm_version` and added multinode feasibility checks via `local_mpi_size`.

### Indexer and DSA

Resolved indexer conflicts by keeping the DSV4 implementation where it was more complete and preserving main compatibility updates:

- C++ indexer top-K files: kept dynamic SM-count/device-aware launch policy, caller-owned radix auxiliary scratch, compress-ratio support, and CUDA graph safety.
- `dsa.py`: kept main FP4 `data_bytes_per_token` handling while preserving compressed lengths for DSV4 indexer.
- Tests: kept radix/multi-pass tests and added launch-policy/top-K saturation coverage.

### FMHA artifacts

The commit `chore: update v4 fmha cubins and libs` conflicted across many binary cubin/lib artifacts. Selected the DSV4 PR side for all unmerged binary artifacts because the commit purpose was to update V4 FMHA kernels. Later DSV4 FMHA OOB fix commits remained on top after the rebase.

### CI stage and waiver files

Resolved CI conflicts by preserving main's latest stage matrix and adding only intended DSV4 deltas:

- `jenkins/L0_Test.groovy`
  - Kept main `SBSASlurmTestConfigs`, GB10/GB300 entries, load-balanced GB200 splits, and multi-node perf sanity `buildStageConfigs`.
  - Kept 4-GPU GB200 DS as a single-node Slurm stage.
  - Moved 8/16/24 GPU DS stages into `multiNodesSBSAConfigs` when applying the registered-platform commit.
  - Changed those DS multi-node stages from `auto:gb200-x4` to `auto:gb200-flex` when applying the flex-platform commit.
  - Later moved DSV4 Pro multi-node DS stages to GB300 with `auto:gb300-flex` and `l0_gb300_multi_nodes_ds`, preserving all main GB200/GB300 perf sanity stages.
- `tests/integration/test_lists/test-db/l0_gb200_multi_gpus_ds.yml`: kept intended GB200 DS content after moving large/pro stages out.
- `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_ds.yml`: added DSV4 Pro multi-node DS coverage.
- `waives.txt`: retained latest main waives and appended only PR-specific DSV4/disagg waivers from the branch.
- During final pre-commit validation, removed stale EXAONE/GLM waive-only entries that were no longer present in active test lists and were already absent from `github/main`.
- `.github/workflows/blossom-ci.yml`: kept corrected `"Mingyang"` allowlist entry.

### Package/version/dependency files

Resolved dependency/version conflicts by keeping current mainline versions:

- `README.md`, `examples/constraints.txt`, `tensorrt_llm/version.py`: kept main `1.3.0rc18`; old `1.3.0rc15.post1` bump became obsolete.
- `security_scanning/pyproject.toml`: kept CUDA 13 CUTLASS DSL package and main's newer dependency bounds.
- `poetry.lock`: kept main `flashinfer-python==0.6.12`, `nvep` extra, and HEAD content hash.

## Post-Rebase Checks

Ran:

- `git status --short`
- `git rev-list --left-right --count github/main...HEAD`
- `git log --oneline --decorate -n 16`
- `rg -n '^(<<<<<<<|=======$|>>>>>>> )' --glob '!3rdparty/**' --glob '!tmp/**'`
- `python -m py_compile` on the manually resolved Python files from the last conflict

The conflict-marker scan only found literal `=======` lines in attribution/generated cache files, not Git conflict markers.

## Post-Rebase Validation Fixes

After the rebase, build/install and the `l0_b200_ds.yml` test list exposed several stale API and test-list assumptions from the older PR base. These were fixed in the final rebase-resolution commit:

- `PoolConfiguration` constructor/field names changed on main from `head_dim` to `size_per_head`; updated AutoDeploy shim code accordingly and forwarded `pool_configurations` into the C++ KV cache manager wrapper.
- KV cache manager V2 warmup now passes `kv_reserve_draft_tokens`; added the argument to both dummy-request paths and included it in the V2 dummy capacity.
- Main's KV pool rebalance hook uses `PyExecutor._is_kv_manager_v2`; restored that attribute and kept the DSV4 `_scheduler_manages_kv_suspend` gate as the same V2 check.
- Current MTP/Eagle3 helpers expect position-id normalization helpers; added the missing helpers.
- Current AutoDeploy executor imports moved; added the missing DSV4-related imports.
- `AttentionForwardArgs` callers still pass legacy `is_generation`; mapped that to `attention_input_type` with validation for compatibility.
- Speculative mode helpers now need to tolerate simple test stubs that do not implement all predicate methods.
- Current THOP and DSA op surfaces differ from the older PR base; updated tests for unavailable legacy scratch/FP4 ops and adjusted the DSA Python wrapper call signature.
- Disaggregated SWA gen-only cache skip can be negative when block reuse is disabled; clamped it to zero instead of asserting.
- Pure-Python NIXL is not always installed locally; skipped only that subtest when the package is unavailable.
- Updated the stale accuracy node in `l0_b200_ds.yml` from the removed method name to the current parameterized DeepSeek-V3-Lite MTP test id.

## Build And Test Results

Build/install:

- Built wheel with `python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures "90-real;100-real" --configure_cmake`.
- Installed `build/tensorrt_llm-1.3.0rc18-cp312-cp312-linux_x86_64.whl` into `.venv-3.12`.
- Verified test imports used the worktree Python package path under `TensorRT-LLM_dsv4_pr`; native libraries came from the freshly installed wheel.

Validation ran on this shared machine's available B300 GPU 0 with:

- `LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models`
- `MODEL_CACHE_DIR=/home/scratch.trt_llm_data/llm-models`
- `CUDA_VISIBLE_DEVICES=0`

Results:

- Non-accuracy entries expanded from `tests/integration/test_lists/test-db/l0_b200_ds.yml`: `4826 passed, 63 skipped, 102 warnings, 3 subtests passed in 1579.70s`.
- Accuracy entry `tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_bfloat16[mtp_nextn=2-attention_dp=False-cuda_graph=False-overlap_scheduler=False-torch_compile=False-enable_chunked_prefill=False-v2_kv_cache=True]`: `1 passed, 3 warnings in 266.09s`.
