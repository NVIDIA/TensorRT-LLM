# Kimi K3.1 (`KimiK31ForConditionalGeneration`)

`routing.py` decides which target a deployment builds; read it first.

## Rubin targets (sm_107)

Moonshot's MXFP4 checkpoint runs on the general target on sm_107 as on sm_103. The ModelOpt
NVFP4 export runs on sm_107 only, on four specialized targets, one default target and the ULL
generation target. Each specialized target is one layout of one recipe, and routing admits exactly
the rows below; the default target takes the same layouts with the other recipes, and the ULL
generation target its own row (both below). Anything else raises `NoTarget` with the reason
(`modeling_v2="require"` turns that into an error).

| Target | Layout | Server role: MoE backend, `max_draft_len` |
|---|---|---|
| `kimi_k31_pro_nvfp4__sm_107__cep` | `cep{N}`, N 8 or 16: TP 1, Ulysses CP N (`context_parallel_size: N`, `cp_config: {cp_type: ULYSSES}`) without attention DP, MoE EP N (`moe_expert_parallel_size: N`) | context, disaggregated in front of tep8: `CUTEDSL` or `MEGAMOE_CUTEDSL`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__dep16` | TP 16 with attention DP, MoE EP 16 | context: `CUTEDSL`, 3 to 7; generation: `MEGAMOE_CUTEDSL`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__dep16` (dep12) | TP 12 with attention DP, MoE EP 12: ranks 0-7 hold 75 experts, ranks 8-11 hold 74 (the all-to-all's split); the default target does not take it | context: `CUTEDSL`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__dep16` (dep8) | TP 8 with attention DP, MoE EP 8: 112 experts per rank; dense DWDP (`dwdp_config.enable_dense`) shards the MoE layers' dense weights over the 8 ranks | context: `CUTEDSL`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__helix8` | Helix CP 8, TP 1, MoE TP 2 x EP 4 (`moe_tensor_parallel_size: 2` and `moe_expert_parallel_size: 4` must be set) | generation, disaggregated only: `CUTEDSL_FC12`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__tep8` | TP 8, MoE EP 8, no attention DP | context: `CUTEDSL`, 3 to 7; generation: `CUTEDSL_FC12`, 3 to 7 |
| `kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1` (ULL) | Helix CP 16, TP 1, MoE TP 16 x EP 1 (both must be set) | generation: `CUTEDSL`, 7 (DFlash2 with the Kimi-K3.1-DFlash2 drafter) |

A target of one MoE backend per role builds it whatever `moe_config.backend` asks, so routing
requires the arguments to name that backend; the cep context server builds the one of its two that
the arguments name. A server runs one draft length and verifies a block of
`max_draft_len + 1` rows per request on every step. A generation server's first block holds the
draft the context server in front of it hands over, so the two run the same draft length. Longer
drafts are refused: a draft of k fills 2k + 1 rows of the KDA verify kernels' 16-row chunk (the
replayed drafts and the new rows) and k + 1 rows of one indexer-logits atom (32 index heads a row,
at most 256 columns). The helix8 constants are those of a generation server at batch 8 to 24.

The all-to-all around the CuteDSL experts (the dep16, dep12 and dep8 context servers, and cep
under `CUTEDSL`) combines in FP8 unless `moe_config.use_low_precision_moe_combine` is `false`:
those targets' model defaults set it `true` where the arguments leave it out. With `false` the
combine is BF16, and the all-to-all's workspace is sized for BF16. The flag changes no other
server: tep8 (context and generation), helix8 and the helix16 targets reduce the routed experts
with a latent all-reduce, not this all-to-all; MegaMoE (dep16's generation servers, cep under
`MEGAMOE_CUTEDSL`) combines in BF16; routing refuses `true` on the helix16 generation targets.

The recipe of the three specialized targets:

- the ModelOpt NVFP4 export as stored (`hf_quant_config.json`: `MIXED_PRECISION`; routed experts
  NVFP4 with a scale per 16 values, shared experts and KDA projections MXFP8, everything else BF16;
  no KV-cache quantization). Nothing is quantized at load time. The vision tower and projector
  (BF16 in the export) serve image prompts on the dep16 target's context servers (below, "Images");
- an FP8 latent KV cache on every DSA layer (`kv_cache_config.dtype: fp8`), with the TRTLLM
  attention backend and DSA on every MLA layer;
- FP8 index keys (`indexer_k_dtype: fp8`, the loader's default);
- an FP32 KDA state (`mamba_ssm_cache_dtype: float32`, or `auto`, which a Kimi model resolves to
  float32);
- DFlash2 speculative decoding (one-engine DFlash with the DFlash2 MLA drafter), its context
  history in FP8 in the unified draft cache of KV cache manager V2. The FP8 history is a constant
  of the targets: `speculative_config.context_kv_dtype` may stay `auto` on both servers or say
  `fp8`. A context server may load only what builds the history (`context_kv_only: true`); a
  generation server loads the whole drafter. The cache keeps only the drafter's sliding window of
  the history (its checkpoint's 1024 positions; a context chunk stores the rows of that window
  before its end), on every target, the ULL target included;
  `model_kwargs: {text_config: {kimi_k31_draft_window: false}}` keeps all of it, and then the
  servers on both sides of a hand-off set it alike. The drafter attends to the same positions
  either way. DSpark and no speculative decoding are refused on these three and on the ULL
  target.

Also refused, each with its reason: torch.compile (`torch_compile_config`, and
`prefill_cuda_graph_backend` other than `disabled`), since the targets' kernels are Python
launchers rather than torch ops and no prefill graph masks the MoE padding rows; a
`draft_len_schedule` or `max_concurrency` (DFlash2 verifies a fixed block on every step and
request) and the speculation gate (`acceptance_rate_threshold`,
`acceptance_rate_window_size`); an MoE load balancer; KV cache manager V1; a beam width above 1;
the KV pool rebalance (`enable_kv_pool_rebalance`); `tokens_per_block` other than 64; a
transceiver other than Python NIXL; `enable_lm_head_tp_in_adp`. A helix8 server must be a
disaggregated generation server without block reuse; it runs with or without CUDA graphs and the
overlap scheduler.

The server role is the one `trtllm-serve --server_role` passes as `TRTLLM_DISAGG_ROLE`, in the
environment and in `llm_args.env_overrides`; `routing.disagg_role()` reads the overrides first,
since they also reach MPI ranks spawned before the variable was set. A server without a role
(aggregated) takes the generation role. Routing records the role in
`model_config.extra_attrs["kimi_k31_disagg_role"]`; the model loader carries `extra_attrs` to
the config the target is built from, and the target fixes its role at construction from that
record; it refuses a config that has none (one not built through routing).

The targets' own code reads no environment variable; the family kernels they launch follow the
engine-wide PDL switch `TRTLLM_ENABLE_PDL`, which changes launch scheduling only. Their kernel
levers are launch constants, so a variable that tunes an engine path a target replaces would be
ignored. Routing records those variables (`TRTLLM_K3_FC12_*`, `TRTLLM_CUTEDSL_FC2_*`,
`TRTLLM_NVLINK_ONE_SIDED_A2A_*`, `MEGAMOE_*`, `TLLM_MOE_ASYNC_INPUT_QUANT`) and the two the
engine reads outside the target (`TLLM_ROUTING_DISABLE_COOP_LAUNCH` in C++,
`TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET` in the KV cache estimate), as the workers see them, in
`model_config.extra_attrs["kimi_k31_lever_env"]`. At construction each server refuses, with a
`ValueError`, a recorded value that contradicts its role's launch constants, and a logits
budget other than its role's.

### Images

The specialized targets take image prompts as the general target does: the family's input
processor (`input_processor.py`, registered for model_type `kimi_k31` whatever target serves)
tiles each image and repeats the in-vocabulary placeholder `<|media_pad|>` once per output row.
The engine keys an image's first token with the image's content digest and its other tokens with
their offsets in the image, and chains every block key through the blocks before it, so block
reuse never shares an image's blocks with another image: a prompt reuses another prompt's blocks
up to the first token of the first image they do not share. Image tokens take ordinary 1-D
positions.

- **dep16 and tep8** build the MoonViT tower and its projector (the export's BF16 tensors, whole
  on every rank, as `core.w` rows: `w4_rows`) on both server roles, unless the server sets
  `disable_mm_encoder` (a disaggregated generation server never encodes; with it the target is
  exactly the text-only one). A context step whose requests carry images encodes each request
  once (the general target's `K31VisionTower.encode`, call for call on the same entries, so its
  rows are that tower's bits; a request whose images exceed one largest image's patches is
  encoded in groups within that budget) and splices the rows the step covers over the
  placeholders, ahead of the prefill target; a step without images runs as before. The KV
  capacity estimate's context steps hold one largest image's rows, so the pool counts the
  tower's activations. Admission (`validate_request`) refuses a grid the tower cannot encode and
  media on a server without a tower, unless the request is generation-only.
  Routing sends a server without `disable_mm_encoder` to these targets only as a dep16, dep12 or
  dep8 context server (`routing._VISION_CELLS`). A dep16 generation server or a tep8 server
  without it goes to the configurable default target, so no routed tep8 server builds its
  tower. A disaggregated image deployment puts a dep16 (or dep12, dep8) context server, the one
  that encodes, in front of a text-only generation server.
- **helix8** encodes nothing: as a disaggregated generation server it receives an image prompt's
  KV and ids from its context server, and its requests carry no media (the generation-only
  frontend leaves the relayed media unresolved). Its admission refuses media on any other
  request.

A disaggregated generation server (dep16 or helix8) gets the context server's token ids but not
the images' content digests, so its reuse keys cannot tell two images of one size apart: it keeps
block reuse off (`kv_cache_config.enable_block_reuse: false`).

### The configurable default target

`kimi_k31_pro_nvfp4__sm_10x__default` serves SM100, SM103, and SM107 with
parallelism derived from `Mapping`: attention TP or DP, Helix CP, and valid
MoE TP/EP splits. It supports TRTLLM, CUTEDSL, and MEGAMOE_CUTEDSL MoE,
image inputs, and DFlash2 or non-speculative execution.

Main-cache FP8/NVFP4, indexer FP8/FP4, and KDA-state FP32/BF16 are independent.
DFlash history uses BF16 with `context_kv_dtype: auto` and FP8 with `fp8`,
in both aggregated and disaggregated serving. CTX and GEN must use matching
cache and history formats. NVFP4 storage is gathered into FP8 scratch for MLA;
`handles_nvfp4_kv_cache` preserves this storage on SM107.

Eligible SM107 fixed-layout deployments route to the specialized targets in
the table above. These targets require `disable_mm_encoder: true`, except on
the dep16, dep12 and dep8 context servers, which encode images (above,
"Images"), and explicit `context_kv_dtype: fp8`, in addition to their layout,
backend, role, and recipe constraints. Other supported deployments use the
configurable default, which contains no fixed-layout optimized branches.
Backend selection is explicit: a requested specialized-only backend does not
silently become a different backend when its layout or features are
incompatible.

See `examples/kimi_k31/` for configurable-target serving configurations.

### ULL generation target

`kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1` serves the ULL generation server: the layout of
its row above, the ModelOpt NVFP4 export as stored (no load-time conversion), an FP8 latent KV
cache, FP8 index keys, an FP32 KDA state, KV cache manager V2, beam width 1, static experts, no
low-precision MoE combine, and DFlash2 speculative decoding (one-engine DFlash) with the
Kimi-K3.1-DFlash2 MLA drafter at `max_draft_len` 7: the target verifies a block of 8 rows per
request on every step. The drafter's history is FP8 in the unified draft cache of KV cache manager
V2, paged over the 16 Helix ranks as the target's KV is (`context_kv_dtype` `auto` or `fp8`); the
whole drafter is loaded (`context_kv_only` is refused). The drafter is the helix8 target's DFlash2
drafter on the CP16 layout: every rank attends with all 96 heads over its own history pages, the
Helix exchange leaves it 6 heads, and `o_proj` and the drafter MLP are row-parallel over the 16
ranks; the engine's `DFlashWorker` drives it, and the target and drafter logits are the target
head's gathered full-vocabulary logits.
`_route_ull_generation` checks the recipe (`ops/common/rubin_ull_features.py`, `decide`) and
selects its one profile, `ull16/generation`, before it imports the target. It runs the engine's
TRTLLM DSA backend with its own metadata class (`ops/attention/rubin_ull_dsa_metadata.py`,
returned by the target's `attention_backend_for`). Like the other targets, its own code reads no
environment variable: the family kernels it launches follow the engine-wide PDL switch
`TRTLLM_ENABLE_PDL`, and the vendored `kernels/vendor/vllm_main/specialized_decode/symm.py` sets
torch's `TORCH_SYMM_MEM_DISABLE_MULTICAST` around its own rendezvous (not a user switch). Its
kernel levers are the `ull16` profile's launch constants; routing records no lever variables for
it. Each rank logs at INFO:

```
modeling_v2 kimi_k31: kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 profile ull16/generation rank=<r> ...; ULL flat generation
```

The target serves generation only; routing refuses a context server on its layout. Its context
server is one of the DFlash2 context targets above (dep16, tep8, or cep under DKV, whose group
hands each page to the Helix rank that keeps it), at the same draft length (7): the drafter
history they hand over has the same layout (5 layers, one 576-value FP8 latent per token, the
drafter's 1024-position attention window). Its cache keeps that window of the history, as the
context targets' cache does (`kimi_k31_draft_window`, default on), so the context server hands
over the window and its pages land at their positions. The drafter numbers its history by position
and reads only the window, so it reads the same bytes as from a whole history. Each side's page
table states its draft window: the generation server refuses a context server that keeps another
one when the two pair, before any KV moves (`Standalone draft peers keep different drafter history
windows`), and the history each request carries is restored only into a cache of its window
(`Standalone draft transfer layout does not match the receiving worker`).

### The cep context server

`kimi_k31_pro_nvfp4__sm_107__cep` is the context parallelism of the generic Kimi K3.1 model's
CEP16 work as a specialized target over N ranks,
N bound at construction from the `Mapping`. The deployment is TP 1 with Ulysses context
parallelism over N ranks (`context_parallel_size: N`, `cp_config: {cp_type: ULYSSES}`), without
attention DP, MoE EP N; the engine treats the CP ranks as one replica (every rank takes every
request, the KV cache keeps the TP layout over the N ranks, the MoE group spans them). So every
rank holds the whole batch; CP rank `r` runs rows `[r * rows, (r + 1) * rows)` of the step,
`rows = ceil(num_tokens / N)`, through every layer:

- MLA + DSA: projections on the rank's rows with whole weights; one all-gather of the new latent
  rows and index keys, which every rank inserts and appends (the caches keep the TP layout, whole
  on every rank); the indexer scores the rank's query rows; the sparse MLA of those rows is the
  attention op's context call over the rank's row slices, each a chunked-context request (its past
  KV the request's cached tokens and its new tokens before the slice), all 96 heads;
- KDA, by default the sequence split with a serial FP32 hand-off (`ops/ssm/rubin_kda_handoff.py`):
  every rank runs the convolution and the recurrence of all 96 heads over its own rows; a request
  that spans ranks hands its FP32 end state and its 3-row convolution halo to the next rank, its
  cached state comes from the owners of its heads and its end state and snapshots go back to them
  (the KDA state stays head-sharded over the N ranks, as under TP N); gated norm and `o_proj` on its
  rows. A rank computes the chunk factors of the rows it continues (the prefill's first kernel,
  which reads no state) before the hand-off rounds, so a round waits only for the state recursions
  of the ranks before it (`kda_handoff_prep_ahead`; the same bits either way). A step whose longest
  request spans more than 8 ranks, or with fewer than 3 rows per rank,
  runs Ulysses instead: one push hands each rank its 96 / N heads for all rows, the convolution and
  chunked prefill of those heads over the batch, one push brings each rank its rows of all heads;
- MoE: the rank's rows through its 896 / N experts and the others', by `moe_config.backend`:
  `CUTEDSL`, the all-to-all around the CuteDSL experts (FP8 combine; BF16 with
  `moe_config.use_low_precision_moe_combine: false`); `MEGAMOE_CUTEDSL`, MegaMoE
  (dispatch, FC1, SiTU, FC2 and a BF16 combine in one persistent kernel over the EP group, the
  bucket ladder up to `ceil(min(max_num_tokens, moe_config.max_num_tokens) / N)` rows a rank, one
  launch per layer and step up to that). Each bucket keeps a local workspace for the process
  (7.3 GiB at 8192 rows on cep16; 21.4 GiB at 16384 rows on cep8, which does not fit beside
  cep8's weights: a cep8 server under MegaMoE sets `moe_config.max_num_tokens: 65536`, 8192 rows a
  launch). The shared experts run whole on the side stream beside
  the routed chain (`model_kwargs: {text_config: {kimi_k31_moe_shared_beside: false}}` runs them
  after it). MegaMoE waits inside the kernel for every EP rank's launch, so nothing that waits on
  another rank runs beside it: it launches on the current stream between the mixers' exchanges,
  and the side streams hold only the shared experts and copy-engine reads during the block.
  Per-layer dynamic EPLB runs under `CUTEDSL` only;
- the drafter taps and the final hidden states are all-gathered; the head, the drafter history and
  the first drafts run whole on every rank (the shell builds its LM head and its worker on the
  attention-DP view of the mapping).

Routing admits N = 8 and 16 (`_CEP_SIZES`). N must split the 96 KDA heads and the 896 experts;
below 8 a rank's 896 / N experts and whole attention weights exceed a Rubin GPU's memory (a cep16
rank loads 158 GiB of weights, each further 56 experts about 89 GiB more); cep32 has no sm_107 run
yet. The Mapping alone tells cep8 from tep8 (TP 8). A cep server serves context steps only; its
generation server is tep8: the transfer sees the Ulysses group as TP N (the Python transceiver
only) and fans N / 8 cep ranks into each tep8 rank, two for cep16, one for cep8. It runs eagerly.
It serves text only (`disable_mm_encoder: true`): it builds no vision tower, and the default
target does not take Ulysses CP, so an image deployment needs a dep16 context server instead.
The CP group's row exchanges (the KDA hand-off and Ulysses, the indexer's row moves) are one-sided
MNNVL pushes (`ops/comm/rubin_cp_push.py`, a CuTe DSL kernel): one fabric segment per rank mapped on
every rank, built at construction, two alternating receive regions, no NCCL point-to-point buffers.
`model_kwargs: {kimi_k31_kda_split: ulysses}` always runs the pushed Ulysses (BF16);
`ulysses_fp8` sends the Ulysses all-to-alls as FP8 rows and `relay` runs the sequence split with
the owners composing the start states, both over NCCL.

A static routed-expert placement, `model_kwargs: {kimi_k31_expert_placement: <file>}`, relabels
the routed experts at load: slot `s` of a placed layer holds checkpoint expert
`expert_of_slot[s]`, the router's rows and score-correction bias are permuted into slot order, and
routing returns slots, so both MoE backends run unchanged (only a tie of two router scores breaks
by slot instead of by expert). The file is the generic Kimi K3 work's
(`kimi_k3_expert_placement/v1`: `num_experts`, the `ep_size` it was balanced for and
`expert_of_slot` by global layer; a v2 file loads as its v1 part;
`ops/moe/_rubin_expert_placement.py`). Any N loads it; an N that divides its `ep_size` keeps its
balance (a cep8 rank holds two whole cep16 ranks' slots), another N warns.

With `dkv_config: {attention_mode: sp, kv_layout: page_split}` in the context server's llm-args
YAML, the cep group stores each request's MLA latent once under distributed KV (DKV,
`KVCacheManagerV2(dkv=True)`, `ops/attention/rubin_cp_dkv.py`), in the `page_split` layout: global
page g of `tokens_per_block` tokens lives on rank g % N, a rank appends the new tokens it owns, and
the sparse MLA of a rank's rows reads a local buffer: the step's new tokens, then the history rows
its Top-K selects, copied once each from their owners through the peer mapping
(`ops/attention/rubin_dkv_topk_read.py`, one CuTe DSL gather; the selected count stays on the
device), with the Top-K remapped to rows of that buffer: one CuTe DSL launch
(`rubin_dkv_topk_read_fused`) writes the indices (the new tokens' rows, the copied history rows,
the -1 padding pointed at row 0) and zeroes the buffer's page 0, where that padding points.
The DFlash2 drafter's history (one FP8 latent per token and drafter layer) is split the same way:
every rank computes every row, global page g lives on rank g % N alone
(`KVCacheManagerV2.draft_page_split`), and that rank sends it to the generation server
(`model_kwargs: {text_config: {kimi_k31_dkv_draft: false}}` keeps every page on every rank).
Without `dkv_config` every rank holds the whole latent cache and history. The index keys are stored
once per group too by default (`dkv_config.index_layout: sharded`): a rank inserts the keys it owns, a
one-element all-gather fences the group, and the indexer's per-chunk key gather reads each key from
its owner through the mapping; `index_layout: whole` keeps every key on every rank. Block reuse is token-granular
(`enable_partial_reuse`): the group resumes at the shortest match of its ranks, each rank copying
its own page of the last DKV block. The context call takes its query in FP8 (`rubin_mla_ctx_q_fp8`
for the latent columns, the op's RoPE kernel for the RoPE columns), as dep16's
`mla_ctx_fused_q_fp8`. The indexer scores rows of equal cost per rank (`balanced_query_bounds`: a row's cost grows
with its history) and the Top-K returns to the rows' ranks; the sparse MLA keeps the equal rows (a
fixed Top-K). `model_kwargs: {trtllm_cep_indexer_split: equal}` scores each rank's own rows.

### Code layout

- A specialized target directory holds `__init__.py`, `modeling.py` and `weights.py`, nothing
  else but dep16's `kernels/moe/megamoe_token_path.py`: the engine's MegaMoE generation kernel
  and its token exchange subclassed with dep16's three token-path levers, which its runtime build
  binds for the process before the MegaMoE workspace. `modeling.py` is straight-line code: module
  constants for the layout, the recipe and the launch constants of each server role, then the
  step and runtime interfaces, the KDA, MLA + DSA and MoE blocks, the DFlash2 drafter and worker,
  the vision tower (dep16, tep8), the core with its prefill and decode targets, and the shell
  with the engine hooks. `weights.py` holds the weight table: each checkpoint tensor the target
  reads, the parameter it fills and its transform. The default target's directory holds the same
  three files.
- The Rubin kernels are in `kernels/<kind>/rubin_*.py`; their OpWrappers and the runtime helpers
  the engine needs (the DSA attention metadata and its backend class, the KDA + DSA cache manager
  and the KDA metadata, the builders of the collective workspaces) are in `ops/<kind>/rubin_*.py`.
  `kernels/activation/situ_and_mul.py` and `kernels/ssm/reset_recurrent_state_rows.py` are the
  family's copies of the engine kernels that its SiTU and state-reset entries
  (`ops/activation/situ_and_mul.py`, `ops/ssm/reset_recurrent_state_rows.py`) launch. The targets
  share kernels, entries and runtime helpers only through these directories and `config.py`
  (`modelopt_nvfp4_mismatch`); each target keeps the rest, its weight-table machinery included,
  in its own `modeling.py` and `weights.py`. The DFlash2 drafter and its worker are part of each
  target's `modeling.py`.
- These kernels register no torch op: each OpWrapper calls the kernel's Python launcher directly.

### Log lines

Each rank logs, at INFO, its target, role and launch constants once the weights are loaded,
then the recorded values of the two variables the engine reads outside the target (`unset`
when not set):

```
Kimi K3.1 Rubin constants: kimi_k31_pro_nvfp4__sm_107__<layout>/<role> (<n>): <key>=<value>, ...
Kimi K3.1 Rubin lever env: kimi_k31_pro_nvfp4__sm_107__<layout>/<role>: TLLM_ROUTING_DISABLE_COOP_LAUNCH=<value>, TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET=<value>
```

A cep rank logs its N first:

```
Kimi K3.1 Rubin layout: kimi_k31_pro_nvfp4__sm_107__cep: cep<N> (Ulysses CP <N>, MoE EP <N>, <96 / N> KDA heads and <896 / N> experts per rank)
```

A helix8 generation rank logs one more line between those two: its DFlash2 drafter's switches,
every `draft_*` generation constant as the constants line states it, how many of the on / off
ones are on, and `fp`, the first 12 hex digits of the SHA-256 of the listed items (two runs with
the same `fp` ran the same drafter switches):

```
Kimi K3.1 Rubin DFlash2 drafter switches: kimi_k31_pro_nvfp4__sm_107__helix8/generation fp=<12 hex> on=<n>/<m>: draft_agree_check=<value>, ...
```

Each helix8 generation rank logs, at INFO and once per process, the first time a CUDA graph's
decode step runs a DSA layer (warm-up or capture):

```
Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream beside [q_a | kv_a]
```

A context server whose `TLLM_ROUTING_DISABLE_COOP_LAUNCH` is not `1` also logs a warning: its
MoE routing kernels then launch cooperatively.

Each helix8 generation rank logs, at INFO and once per process, the first time a CUDA graph's
decode step runs its MoE (warm-up or capture):

```
Kimi K3.1 Rubin MoE front: FC12 input quantized on the latent stream beside the routing tail
```

With `router_first_wide` (a helix8 generation constant, on), each helix8 generation rank also
logs, at INFO and once per process, the first time a CUDA graph's decode step
of more than 96 rows issues its router GEMM before `latent_down` (warm-up or capture):

```
Kimi K3.1 Rubin MoE front: router GEMM issued before latent_down above 96 rows (router_first_wide)
```

With a `gemm_pins_*` group on (helix8 generation constants, on by default: `gemm_pins_moe`,
`gemm_pins_mla`, `gemm_pins_dense`, `gemm_pins_draft`; the binding needs cuBLAS 13), the first call of each pinned GEMM site and row count outside capture
(warm-up) runs the site's GEMM and its pinned cuBLASLt algorithm on the same operands and compares
the words, and each helix8 generation rank logs the outcome once per process, site and row count:
at INFO when they are equal (the pinned algorithm then runs at that row count),

```
Kimi K3.1 Rubin GEMM pins: <site> at <rows> rows runs cuBLASLt's <pin>, today's GEMM's bits
```

and otherwise at WARNING, today's GEMM then running there:

```
Kimi K3.1 Rubin GEMM pins: <site> at <rows> rows: <why>; today's GEMM runs
```

With `gemm_pins_a_class` on as well (off by default), the A-class pins of `_GEMM_PIN_TABLE_A` (another
FP32 summation order of the same products, no dtype change) are checked against a band of today's largest
magnitude instead of the words, and log at INFO:

```
Kimi K3.1 Rubin GEMM pins: <site> at <rows> rows runs cuBLASLt's <pin> (A-class: another summation order, within 0.03125 of today's largest magnitude)
```

With `moe_latent_up_resident` (a helix8 generation constant), each helix8 generation rank logs, at
INFO and once per process, the first time a CUDA graph's decode step whose rows fall in
`moe_latent_up_resident_bands` runs latent_up as the resident split-K GEMM behind the latent
all-reduce row kernel (warm-up or capture):

```
Kimi K3.1 Rubin MoE back: latent_up streams its weight while the latent all-reduce waits (resident, split)
```

With `moe_cp_tail` (a helix8 generation constant, on), each helix8 generation rank logs, at INFO
and once per process, the first time a decode step whose rows fall in `moe_cp_tail_bands` runs the
column-parallel MoE tail: latent_up by output columns over the MoE TP8 group, pushed into the
all-gather rows of the Lamport call that carries the shared experts' partials, which the next mix
finishes.

```
Kimi K3.1 Rubin MoE tail: column-parallel latent_up pushed into the shared experts' Lamport call (moe_cp_tail)
```

With `moe_shared_late` (a helix8 generation constant, on), a CUDA graph's decode step whose rows fall
in `moe_shared_late_bands` (row bands within `moe_cp_tail_bands`) runs its layers on streams of the
highest priority and forks the shared experts after the routing tail onto the one default-priority
stream, so FC12 takes the SMs first; each helix8 generation rank logs, at INFO and once per process,
the first time such a step's column-parallel tail forks them (warm-up or capture):

```
Kimi K3.1 Rubin MoE shared experts: forked after the routing tail, below the layers' streams (moe_shared_late)
```

A DEP16 (DEP12, DEP8) context rank logs, at INFO once the weights are loaded, the wire of its
all-to-all's combine: `FP8`, or `BF16` with `moe_config.use_low_precision_moe_combine: false`. A
cep rank under `CUTEDSL` states it in its MoE line (`all-to-all (FP8 combine)` or
`(BF16 combine)`).

```
Kimi K3.1 Rubin MoE: kimi_k31_pro_nvfp4__sm_107__dep16/context: all-to-all combine in <FP8|BF16>
```

A DEP16 context rank also logs, once, when its MoE first zeroes the rows the FC2 finalize adds
into (`rubin_moe_zero_local_rows`): only the received rows of the all-to-all's combine payload.

```
Kimi K3.1 Rubin MoE zero: received rows only (rubin_moe_zero_local_rows)
```

Each helix8 generation rank logs, at INFO and once per process, the first time its DSA indexer
projects a step's rows:

```
Kimi K3.1 Rubin DSA indexer: x widened with full-width accesses, (w * qs) * scale in one kernel
```

Each helix8 generation rank logs, at INFO and once per process, the first time a mix finishes
the all-reduce of the pushed o_proj partials before its grid wait:

```
Kimi K3.1 Rubin o_proj all-reduce: the mix finishes the pushed partials before its grid wait
```

Each helix8 generation rank (its `dsa_key_shard_failing_fallback` is on) logs, at INFO and once
per process, the first time its key-sharded DSA Top-K merges: a row whose certificate
fails is then re-selected from the failing ranks' local logits and the other ranks' candidates
(the same rows as from every rank's logits):

```
Kimi K3.1 Rubin DSA key shard: a re-selected row reads only the failing ranks' local logits
```

With `dsa_ks_select_pack` (a helix8 generation constant, on), each helix8 generation rank logs,
at INFO and once per process, the first time one launch of helix16's select / pack / merge writes a
step's key-sharded Top-K as KV pool rows, its counts and mask (no local Top-K, pack, merge or remap
launch; its select on min(`dsa_ks_select_cluster_max`, SMs / rows) CTAs per row, sized per step; 1 by
default: the one-kernel exchange). It runs at a row count once the process compared it,
on every rank and on that row count's first eager step, word for word with today's chain on the
step's logits and on tie-heavy, forced-fallback and full-width forms of them; where it differs on
any rank, every rank warns once for that row count and keeps today's chain.

```
Kimi K3.1 Rubin DSA key shard: helix16's select / pack / merge writes the KV-pool rows (cluster per step; checked word for word against the Triton chain per row count)
```

Each helix8 generation rank logs, at INFO and once per process, the first time a CUDA graph's
generation step drafts (warm-up or capture): the drafter's fc projection of the step's captured
target rows ran on the aux stream before the acceptance.

```
Kimi K3.1 Rubin DFlash2: drafter fc projection issued on the aux stream before the acceptance
```

With `draft_lowm_gemm` (a helix8 generation constant, on), each helix8 generation rank logs, at INFO and
once per process and row count, after its first drafter step at that row count, which of the
drafter's projections ran a low-M kernel of the plan there (`cublas` where the plan names none),
and the plan's class: `B` for `K31_LOW_M_DRAFT_PLAN_HELIX8` (the drafter's bits unchanged), `D`
under `draft_lowm_gemm_d_class` (on: another summation order of the drafter's products), for example:

```
Kimi K3.1 Rubin DFlash2: drafter projections on the low-M plan (B) at 54 rows: attn_conv_proj=cublas, q_a=cublas, q_b=cublas, kv_a=cublas, o=cublas, gate_up=tgvchunk, down=cublas
```

Under `draft_q_a_kv_a_fused` (a helix8 generation constant, on: each drafter layer's q_a and kv_a as
one GEMM over the stacked `d_l{j}_q_a_kv_a` rows, at the row counts of its detail
`draft_q_a_kv_a_bands`, 1-78) the line names that GEMM `q_a_kv_a` in place of `q_a`
and `kv_a`. The startup constants line states it, `draft_conv_pdl` (the drafter's convolutions with
PDL where they are their own kernels; off) and `draft_norm_ctas` (inside `draft_conv_norm`'s value,
on: the CTAs of each row of its conv + add + RMSNorm, 7; 0 for one). The low-M plans, the
`[q_a ; kv_a]` bands, the norm's cluster and the CuTe DSL window below serve only a drafter of
draft length 5 (block 6), the one they were measured and checked at; at any other draft length the
drafter projects on cuBLAS in separate GEMMs, normalizes a row a CTA and runs the Triton window.

With `draft_window_mla` (a helix8 generation constant, on), each helix8 generation rank logs, at INFO and
once per process, after its first drafter step that runs the window on the CuTe DSL MLA decode
(the BF16-packed window, in place of the Triton body and tail; only at the row counts of its detail
`draft_window_mla_bands`, 1-30 and 70-120, outside which the Triton window runs; a row count captured without its
buffers warns once and keeps the Triton window):

```
Kimi K3.1 Rubin DFlash2 window: the CuTe DSL MLA decode over the BF16-packed window (draft_window_mla)
```

The drafter tail of a helix8 generation step has seven switches, helix8 generation constants, on
but for `draft_block_rows` and `draft_history_aux`; each runs the same values through fewer launches or on another stream (the drafter's and the
target's bits are unchanged), and the constants line states each:

- `draft_select_accept`: on an all-greedy step whose target tokens come from the vocab shard, one
  `select_accept` launch reduces the exchanged argmax pairs, accepts strictly, forces the count
  (the engine's forced acceptance and its totals), writes the KDA replay record and the drafter's
  `kv_lens` prep;
- `draft_greedy_tokens`: the greedy draft tokens straight from the selector's walk
  (`k31_dflash2_selector_greedy_tokens` with the FP32 block dtype), without the full-vocabulary
  block and its argmax; the shard pairs keep their id order;
- `draft_index_pairs`: the shard's top-32 pairs written in id order by the selection kernel
  (`k31_dflash2_shard_index_pairs`), in place of `_pairs_in_id_order`'s sort;
- `draft_block_rows`: the block rows the draft logits read as a strided view and one copy, in place
  of the slot ids and the index gather;
- `draft_tail_rewind`: the drafter's `kv_lens` rewind inside the next-new-tokens launch;
- `draft_noise_block`: the generation preparation's drafter input block as one copy kernel over the
  target's embedding table (`k31_dflash2_noise_block`), in place of the five launches of
  `dflash_noise_block_embedding`;
- `draft_history_aux`: on a graph step whose drafter fc projection is issued early on the aux
  stream, the history chain of the same rows (the stacked `kv_a` projection, the context norm and
  RoPE) runs there too, after the fc; only the latent store stays behind the acceptance. Each
  helix8 generation rank logs, at INFO and once per process, after its first preparation that
  stores those latents:

```
Kimi K3.1 Rubin DFlash2: drafter history (kv_a, context norm / RoPE) on the aux stream after the early fc (draft_history_aux)
```

With `draft_select_accept`, each helix8 generation rank logs, at INFO, once per path, which
acceptance its first steps ran: `fused=1` with what the launch folded, or `fused=0` with the
reason it did not apply (`not a greedy shard step`, `batch <contexts>/<generations>`, `layout`,
`kda <why>`), for example:

```
Kimi K3.1 Rubin DFlash2 select_accept: fused=1 kv_prep=1 kda_replay=1
```

With `draft_pairs_zero_padded` or `draft_agree_zero_padded` (helix8 generation constants, on), each
helix8 generation rank logs, at INFO and once per process, when its drafter runtime is bound,
each of the drafter's TP8 copies that runs as a zero-padded FP32 MNNVL all-reduce on the target
head's pair exchange (the selector's top-32 pairs gather, the draft-token agreement), with the
rows and columns of its buffer; a copy without the line keeps the NCCL all-gather (for example at
16 requests, draft length 5):

```
Kimi K3.1 Rubin DFlash2: drafter top-32 pairs gather as a zero-padded MNNVL all-reduce (FP32, 80 rows x 512 columns)
Kimi K3.1 Rubin DFlash2: drafter draft-token agreement as a zero-padded MNNVL all-reduce (FP32, 16 rows x 2048 columns)
```

With `draft_agree_check` (a helix8 generation constant, off), the agreement still runs and each helix8
generation rank counts, on the device, the rows whose local drafts differed from rank 0's (real
rows and CUDA-graph padding rows apart), and logs the totals at INFO every 200 steps (a WARNING
the first time a real row differed):

```
K31_H8_AGREE rank=0 mismatched_real=0 real_rows=8640 mismatched_padding=0 padding_rows=960 steps=1001
```

With `draft_window_vec_io` (a helix8 generation constant, on), each helix8 generation rank logs,
at INFO and once per process, after its first drafter step, on how many drafter layers both
window kernels moved their rows as 16-byte vectors (`ALIGNED`: every row stride, column offset
and base address a multiple of 16). A launch whose strides refuse it warns once and runs the
scalar loads and stores; the output bytes are the same either way.

```
Kimi K3.1 Rubin DFlash2 window: ALIGNED=True on 5 of 5 drafter layers (16-byte vector loads and stores)
```

With `kda_verify_pdl` (a helix8 generation constant, on), each helix8 generation rank logs, at
INFO and once per process, the first time a KDA verify launches as the f_b GEMM's PDL dependent:
where the verify's v2 kernel serves the step (12 heads times the requests within the SM count);
above that the two-CTA kernel runs as a plain launch. The output bytes are the same either way.

```
Kimi K3.1 Rubin KDA: the verify launches as f_b's PDL dependent (state, replay entries and conv windows load before the grid wait)
```

With `kda_verify_fused_fb` (a helix8 generation constant, on; it acts with `kda_verify_pdl` on the
same rows), the first eager KDA verify at each row count checks the forget gate the v2 verify
kernel computes from the f_a columns against the f_b GEMM path on the device: both run on the same
pools, and the output and pool words must be equal. Each helix8 generation rank logs the outcome
once per row count, at INFO when it passed (the fused form runs at those rows from then on), or at
WARNING with the reason when it did not (the f_b GEMM and the verify run as before). A CUDA graph
captured before its row count was checked keeps the f_b GEMM. The rank also logs, at INFO and once
per process, the first time a verify runs the fused form; the PDL line above then stands only for
the verifies that still run the f_b GEMM.

```
Kimi K3.1 Rubin KDA: the verify's forget gate from f_a at 54 rows writes the f_b GEMM path's output and pool words
Kimi K3.1 Rubin KDA: the verify computes the forget gate from f_a (no f_b GEMM) and launches as the QKVG GEMM's PDL dependent
```

With `kda_oproj_push` (a helix8 generation constant, on), each helix8 generation rank logs, at
INFO and once per process, the first time a KDA verify block's o_proj epilogue stores its partial
into the Lamport slots of the all-reduce (the push copy kernel does not run): on the push-path
rows (up to `attn_res_pdl_max_m` on a non-boundary layer with snapshot rows), where o_proj runs
the shared low-M / mid-M MXFP8 GEMM. A call that falls back to the GEMM and the push kernel warns
once instead. The pushed bytes are the same either way.

```
Kimi K3.1 Rubin KDA: o_proj's epilogue pushes its partial into the all-reduce (no push kernel)
```

The verify's early release of o_proj (`kda_decode_early_trigger` with `kda_verify_pdl`: right after
the verify's grid wait) only moves o_proj's launch: o_proj then streams its weights during the
recurrence. The constants line states whether it is on.

With `dsa_oproj_push` (a helix8 generation constant, on), each helix8 generation rank checks, once
per process and row count, a DSA layer's o_proj on reduce_norm's push-path rows that fall in
`dsa_oproj_push_bands` (up to 72 rows): the push GEMM's output (FlashInfer's BF16 split-K GEMM, no
K split, one wave of CTAs) against today's o_proj, word for word. Equal, it logs at
INFO and that row count's o_proj stores its partial into the Lamport slots of the all-reduce from
its epilogue (the push copy kernel does not run); not equal, it warns and today's GEMM and push
kernel keep running. The pushed bytes are the same either way.

```
Kimi K3.1 Rubin DSA: o_proj's epilogue pushes its partial into the all-reduce at {rows} rows (today's GEMM's bits)
```

The mix-node levers (helix8 generation constants; `mix_bank_in_mix` and `boundary_ar_push` on,
`draft_tap_stream` and `dense_ar_push` off) move nodes of the step's graph and leave the
bytes as they are; each logs, at INFO and once per process, the first time it acts:

- `mix_bank_in_mix`: a block boundary's bank snapshot is stored by its producer instead of a copy
  (the boundary's MoE mix stores its updated prefix into the bank row where a PDL mix serves the
  rows; layer 0's embedding gathers into bank row 0); other kernels keep the copy.
- `draft_tap_stream`: the drafter's capture copies run on a stream of their own, forked after the
  mix that makes the tapped stream and joined where the drafter first reads the captures.
- `boundary_ar_push`: on a KDA verify step's block boundary, o_proj's epilogue pushes its partial
  and the mix after the attention finishes the all-reduce as the new prefix sum. The first eager
  call at each row count and bank size compares both paths on the device, the group agrees on the
  verdict, and the result is logged (INFO when the push takes over, WARNING when a word differs
  and the C++ all-reduce stays). It finishes the partial the KDA verify's o_proj epilogue pushes,
  so it needs `kda_oproj_push` and `kda_oproj_quant_k32`; without them the import fails.
- `dense_ar_push`: layer 0's dense MLP output goes to the next mix unsummed, which pushes it on its
  push path (the one the DSA o_proj partials take).

```
Kimi K3.1 Rubin attn_res: a block boundary's bank snapshot is stored by its producer (no copy)
Kimi K3.1 Rubin DFlash2: the drafter's capture copies run on the tap stream (draft_tap_stream)
Kimi K3.1 Rubin attn_res: a block boundary's o_proj all-reduce is pushed and finished by the mix as the prefix (boundary_ar_push)
Kimi K3.1 Rubin dense MLP: the output goes to the next mix unsummed, which pushes it (dense_ar_push)
```

### Gates

- **Boot.** Every layout spans hosts (a Rubin host has 4 GPUs). A specialized target boots as
  `trtllm-serve --config <file>` over an MPI world of the layout's size, with its accuracy gate's
  LLM arguments in the file (`modeling_v2: require` among them); each rank's constants line above
  names the target and the role it built.
- **Accuracy.** `tests/integration/defs/accuracy/modeling_v2/test_kimi_k31.py`,
  `TestModelingV2KimiK31ProNvfp4Sm107{Cep,Dep16,Helix8,Tep8}`: GSM8K with the recipe. DEP16 and
  TEP8 run as aggregated servers (the generation role); helix8 runs disaggregated behind a DEP16
  context server; cep8 and cep16 run disaggregated in front of a TEP8 generation server. The reference and the threshold are the `quant_algo: MIXED_PRECISION`,
  `kv_cache_quant_algo: FP8`, `spec_dec_algo: DFlash` entry of `moonshotai/Kimi-K3.1-Pro` in
  `tests/integration/defs/accuracy/references/gsm8k.yaml`. The export has no measurement of its own
  there yet: the gate is the fixed floor of 95.0 on a single pass. helix8's test needs 24 GPUs on
  one host, which no Rubin system has, so it skips there. The configurable default's integration class,
  `TestModelingV2KimiK31ProNvfp4Sm10xDefault`, covers its serving configurations
  independently of these specialized-target gates.
- **Acceptance.** Kimi K3.1 has no built-in implementation to run under `modeling_v2=off`, so
  the drafter's acceptance length is read off greedy serving runs of the recipe without forced
  acceptance (a GSM8K subset and prompts over 8k tokens): each target's natural acceptance
  length must stay within 4 sigma of that of the exact FP32 window decode. dep16 and tep8 run that
  decode (`rubin_dflash2_window_fp32` on the dep16 target, its operands staged by two Triton
  launches; `rubin_dflash2_window_exact` on tep8 and in the default target's dep16 and tep8
  layouts); helix8 runs `rubin_dflash2_window_triton` and
  `rubin_dflash2_helix_combine`. With `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` set, each
  rank's worker logs, every 200 calls,
  `Kimi K3.1 Rubin forced acceptance: <target> value=... calls=... rows=... accepted=... mean=...`.
