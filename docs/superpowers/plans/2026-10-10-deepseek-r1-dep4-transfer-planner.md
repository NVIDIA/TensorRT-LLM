# DeepSeek-R1 dep4 target-owned TransferPlanner — Phase 2 plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** The first modeling_v2 target that plans its own disaggregated KV transfer: the
`r1_0528_nvfp4__sm_103__dep4` target ships a `TransferPlanner` in its own directory, overrides
`build_transfer_planner`, and a ctx dep4 -> gen dep4 deployment across two 4-GPU nodes runs on it.

**Design (agreed in chat, 2026-10-10):**
- Under ADP dep4 every ctx rank pairs with exactly one gen rank (same dp_rank; flat rank == dp_rank
  since pp=cp=1), MLA is one K buffer per layer (`kv_factor=1`), FP8 KV, one PAGED layer group over
  one physical pool. The target planner therefore needs no pool matching, no policy, no mapper:
  `targets` = `[peer_dp_rank]`, `plan` = ordinal pairing then `base + block*slot_stride`, merging
  runs of blocks that are consecutive on both sides into one fragment (the default planner emits
  one fragment per block).
- `check_peer` is the target declaring the only topology it supports: peer `dp_size == 4`,
  `pp_size == cp_size == 1`, `enable_attention_dp`, one PAGED group with one pool, equal
  `slot_bytes`, `tokens_per_block` and layer count. Anything else -> `PeerIncompatibleError`.
- `planner_id = "deepseek_r1_0528_nvfp4_sm103_dep4"`; `receiver_payload_bytes` = 0; `describe` = b"".
- Correctness proof is a CPU parity test against `DefaultTransferPlanner` on a synthesized R1
  page table: split the target planner's merged runs back to slot granularity and the fragment
  multiset must equal the default planner's.
- GPU gate: two nodes (ctx on A, gen on B), NIXL over UCX tcp/eth0, `TRTLLM_MODELING_V2=require`,
  GSM8K against the `deepseek-ai/DeepSeek-R1-0528` NVFP4/FP8 reference (94.24). The in-repo
  accuracy test is the standard localhost 8-GPU form for CI; the two-node launcher is a dev script
  outside the repo because the harness only knows localhost.

**Spec:** `docs/superpowers/specs/2026-10-10-disagg-transfer-planner-design.md` (Phase 2 section)
and the agreed design above.

## Global Constraints
- NVIDIA header on new files; `git commit -s`; no AI attribution.
- Target isolation (`test_targets_do_not_share_files`): the planner file imports the catalog or
  framework packages absolutely, never another target.
- Zero change to `native/` and to the default planner.

## Review Focus
1. Non-owning / empty pairing: a chunk whose ordinals the peer does not hold must yield
   `TransferPlan.empty()` (pinned in the parity test with an all-holes dst table).
2. Merged runs must stop at any discontinuity on *either* side (pinned: src consecutive, dst not).
3. `dst_start_token` masking must match the default planner (pinned in parity).
4. A peer with a second pool (e.g. MTP layer in its own pool) must be refused, not silently
   half-transferred (pinned in `test_check_peer_refuses_*`).
5. The target planner must never be selected for the gen side when the ctx side runs the default
   planner: framework `planner_id` gate (already pinned in Phase 1).

---

### Task 1: planner + parity test
Files: Create `tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/transfer.py`;
Modify `.../modeling.py` (override `build_transfer_planner`);
Test: `tests/unittest/_torch/modeling_v2/models/test_modeling_v2_deepseek_transfer_planner.py`.
Steps: write the parity + refusal tests, see them fail on import, implement, see them pass, commit.

### Task 2: in-repo accuracy gate (8 GPUs, localhost form)
Files: `tests/integration/defs/accuracy/test_modeling_v2_deepseek_v3.py` += `test_gsm8k_disagg`
(ctx dep4 + gen dep4, NIXL PYTHON, require mode via extra_env, DuckLLM args patched with
`kv_cache_quant_algo=FP8` so the R1 reference entry matches). Not runnable on 4-GPU nodes; it is
the CI form of what Task 3 runs by hand.

### Task 3: two-node dev launcher + GSM8K
Files (outside repo, `/scratch/.../users/fredw/mv2-disagg/twonode/`): `alloc2.sbatch`
(2 nodes x 4 GPUs, keepalive on both), `serve_ctx.sh`, `serve_gen.sh`, `disagg_server.sh`,
`client_gsm8k.py` (copy of the harness client closure around `run_accuracy_test`).
Verify: both workers log `planner_id=deepseek_r1_0528_nvfp4_sm103_dep4`... (grep "Resolved" +
transceiver logs), GSM8K within tolerance of 94.24, no `PeerIncompatibleError`.

## Results (2026-10-10)

- Task 1: planner + parity test committed (9a9b56e485); 33 tests green including the modeling_v2
  claims and target-contract suites.
- Task 2: in-repo 8-GPU gate committed (048bd64c25); collected, not runnable on 4-GPU nodes.
- Task 3: two nodes (ctx dep4 on nvl72d202-T12, gen dep4 on nvl72d202-T16, NIXL over UCX tcp/eth0,
  static worker URLs, `TRTLLM_MODELING_V2=require`). Both workers log
  `KV transfer planned by DeepseekR1Dep4TransferPlanner`, no default-planner fallback, no
  `PeerIncompatibleError`. GSM8K flexible-extract **95.22** against the 94.24 reference, PASSED;
  1319 requests in 6.5 min. Launcher findings (loopback-only HTTP discovery, `--server_role`
  crash in static mode, `UCX_TLS=^ib` requirement) are recorded in the dev scripts and memory,
  not in the repo. Logs: `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/mv2-disagg/twonode/logs/run_20261010T182505Z`.

## Redesign of the binding (after review)

The target-overrides-`build_transfer_planner` binding was rejected in review: one planner
must be able to serve several targets, including pairs such as ctx dep4 -> gen dep8 and
targets from different cores. A `COVERS` matrix of target-id pairs, then of
architecture-plus-topology pairs, followed and was dropped in turn: it encoded a topology
nothing consumed and made coverage an enumeration. Implemented instead:

- Planner at `disaggregation/planners/deepseek_mla_latent.py` as
  `DeepseekMlaLatentTransferPlanner`, `planner_id = "trtllm.deepseek_mla_latent"`, with
  `supports_deployment(architecture, mapping)`: `DeepseekV3ForCausalLM`, attention DP,
  pp = cp = 1, any width. `check_peer` keeps the layout checks (one pool, layer count,
  `tokens_per_block`, `slot_bytes`) because the arithmetic depends on them; `describe()` is
  the base class's empty blob.
- `disaggregation/planners/__init__.py`: `PLANNER_ENV_VAR = "TRTLLM_KV_TRANSFER_PLANNER"`,
  `PLANNERS` (planner_id -> class), `bind_transfer_planner`. The user names the planner; the
  framework never picks one.
- `AutoModelForCausalLM.from_config` binds after construction for every model; the target's
  `modeling.py` has zero transfer-related lines.
- Tests: `tests/unittest/_torch/disaggregation/test_planner_binding.py` (supports_deployment,
  PLANNERS keys, binding direction, unset/empty variable, named planner installed, unknown name,
  unsupported deployment) and `test_deepseek_mla_latent_planner.py` (parity, merge, targets,
  geometry refusal).

## Interface hardening (after review)

Reviewing DeepSeek-V4.1 (PR #19727, merged 2026-10-11) against this branch showed
that `plan()` took a copy of `Chunk`'s field list as positional arguments. main
added `Chunk.excluded_pool_views` (remote-tail / bounded-replay SWA views the
generation side rebuilds locally) and filters on it inside the sender's pool
loop, which this branch moved into `DefaultTransferPlanner.plan`. With the old
signature that field could not reach the planner, and a textual rebase would
have merged cleanly while dropping the filter.

`plan(self_ri, peer_ri, chunk, req_info, prompt_len)` and
`receiver_payload_bytes(self_ri, peer_ri, req_info)` now receive the two request
descriptions whole. `RecvReqInfo` moved from `native/transfer.py` to
`base/backend.py` next to `Chunk`, since it is the wire protocol type the planner
contract depends on. The sender's `KVSendTask` is deliberately not passed: it
carries transfer progress, counters and timers.

Done at the rebase onto main: `DefaultTransferPlanner.plan` skips the
`chunk.excluded_pool_views` main added (the DeepSeek planner returns an empty plan
when its sole pool is excluded), and the `bounded_replay_on_generation` peer check
main added to `PeerRegistrar` lives in `DefaultTransferPlanner.check_peer`; both
sit on the planner, not on `transfer.py`. The disaggregated gates pass
`modeling_v2: require` through each server's config, since main made the switch
an `LLM` argument.

Verified: 292 unit tests green (planner, chunked transfer, ownership regressions,
unsubmitted session, peer, modeling_v2 transfers/claims/routing/target
contract/op bind).
