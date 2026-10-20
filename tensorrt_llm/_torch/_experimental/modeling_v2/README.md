# ModelingV2

One self-contained modeling codebase per deployment target, beside the
built-in model zoo rather than inside it.

Where `_torch/models/modeling_deepseekv3.py` is one class serving V3, V3-Lite,
R1 and V3.2 across every GPU generation and parallel topology,
`_torch/_experimental/modeling_v2/models/deepseek_v3/` is one flat forward per (checkpoint,
GPU architecture, parallel topology) triple — assembled only from `catalog/`
entries, sharing nothing with its siblings, and trusted through accuracy gates
instead of shared abstractions. The one-to-one correspondence between
`models/<x>/` here and `modeling_<x>.py` there is the point of the exercise.

## Using it

```python
LLM(model, modeling_v2="require")        # off | auto | require
```

```yaml
# trtllm-serve --config serve.yaml
modeling_v2: require
```

| `modeling_v2` | Behaviour |
|---|---|
| `off` (default) | The resolver returns immediately. Nothing beyond it is imported and behaviour is byte-for-byte what it is today. |
| `auto` | Uses a target when one claims this exact configuration; falls back to the built-in implementation otherwise, logging why. |
| `require` | Raises instead of falling back, saying which criterion did not hold. |

It is an LLM API argument rather than an environment variable so that it
reaches every rank the way the rest of the arguments do; a variable set after
MPI initialized reached the driver and not the workers, and a driver building
a target while its workers build the built-in is exactly the split `require`
exists to catch.

The decision is one function, `route`, in the family's `routing.py`. It reads
the facts that fix which code runs -- the checkpoint's config shape, the
parallel mapping, the GPU's compute capability and the speculative-decoding
config -- and either returns a target class or raises `NoTarget` with the
reason in words. Tuning knobs such as `max_batch_size`, `max_num_tokens` and CUDA graph
settings are not routing inputs: they do not change which code runs, so a
target claims a configuration regardless of them. `require` therefore promises
that the target ran, not that a particular knob setting was measured. The model
loader decides when it is about to build the model, after the built-in model's
defaults have been applied to the arguments, and carries the answer on
`ModelConfig.modeling_v2_target`.

**Use `require` for anything you will attribute to modeling_v2.** Under `auto`,
a configuration that misses a target's criteria silently gets the built-in
implementation — and a performance curve measured that way reads as
modeling_v2's. That is the single most expensive mistake available here.

The predecessor `MODELING_V2_TARGET` is gone. It named a *target*; this names
only a mode, and routing picks the target from the configuration.

The checkpoint is read exactly as published, with no target-owned
`config.json`.

## How a config finds its target

```
LLM(model=..., modeling_v2=...)  ->  ModelLoader
                                        |  model defaults applied to llm_args
                                        |
                                        +-- modeling_v2_resolve(config, llm_args)
                                              |
                    _router_index.py          +-- architectures[0] -> routing module
                    models/<x>/routing.py     +-- route(config, mapping, compute_capability,
                                              |         spec_config) -> target class, or NoTarget
                                        |
                                     config.modeling_v2_target = class
                                        |
                    AutoModelForCausalLM._resolve_class(config)
                                        |
                                     returns that class; the registry is not consulted
```

`route` imports the target class on the branch that returns it, so a routing
module that declines never loads modeling code, and the class a reader sees
returned is the class that runs -- no name to look up elsewhere.

## What may decide a target

A quantity may be a routing criterion only if it is known when
`_resolve_class` runs **and** constant for the engine's life — and only if it
changes the *structure of the forward*. Config shape, SM version, parallel
topology qualify. `max_num_tokens` and `cuda_graph_config.max_batch_size` pass
the first test and fail the second: they are tuning knobs — LLM API arguments,
not identity. Batch composition and `num_contexts` fail the
first outright — those move every step, and a target chosen from them would
be chosen once and then be wrong. Per-step specialization is a separate
mechanism (dispatch inside a target's `forward`), not a routing dimension.

### Checkpoint identity is a shape fingerprint

Routing recognizes a checkpoint by `(num_hidden_layers, hidden_size, ...)`,
the same sniffing idiom as `is_mla` / `is_nemotron_hybrid` upstream. It cannot
tell a fine-tune from the original. That is a deliberate trade, and it changes
what a gate record means: not "this target passed" but "this modeling code
passed **on the checkpoint the accuracy gate names**". Running it on any other
checkpoint of the same shape is ungated.

## Layout

```
_router_index.py    architectures[0] -> routing module. Small, stable, test-guarded.
models/<family>/
  routing.py        one forward-reading decision tree per architecture family
  <checkpoint>__<gpu arch>__<parallel>/
                      modeling.py  weights.py
                      ops/<category>/   entries only this target calls
catalog/<category>/ entries more than one target calls
```

An entry, shared or owned, is one `OpWrapper` subclass: `__call__` is the op
call a target makes, `reference` is what it is supposed to compute in plain
torch, `is_valid` names the inputs it accepts and answers wrongly, and `CELLS`
/ `ARCHS` are the configurations and architectures it is proven at. The
wrapper is stateless and `__call__` reads nothing off `self`, so a call site
shows exactly what the kernel receives; `catalog/_op.py` states the rest.

Accuracy anchors are not kept here. They live where every other model's do,
`tests/integration/defs/accuracy/references/`, read by the gates CI runs.

An entry's GPU test lives in the tests tree, mirroring the entry's path:

```
tests/unittest/_torch/modeling_v2/
  test_modeling_v2_claims.py     source-level gates over the whole package (no GPU)
  test_modeling_v2_routing.py    what modeling_v2_resolve does (no GPU)
  test_modeling_v2_*weight_table.py   what a target's weight table reads and slices (no GPU)
  _validating.py                 the fixtures that interpose is_valid and arm reference
  <category>/test_modeling_v2_<entry>.py                      shared entries
  models/<family>/<target>/<category>/test_modeling_v2_<entry>.py   owned entries
  comm/_<entry>_op_matrix.py   the two collectives' 4-rank rank bodies
  comm/_rank_job.py            starts one of those and asserts on its exit code
```

That split is not a preference; it is where this repo's CI collects from, and
an in-package test is on no list. Each test is parametrized off its entry's
`CELLS` and gates `__call__` against `reference` within a band the test states
and derives, so the certification is re-proved on every commit rather than
recorded once.

The two collectives' rank bodies sit in the tests tree with everything else
that only tests run. Both halves are started by file path -- the launcher must
not import `tensorrt_llm`, because that calls `MPI_Init` and an
MPI-initialized process cannot start `mpirun`, and the ranks reach the catalog
by absolute import -- so neither needs a package to live in. Neither those
file names nor their `check_*` bodies match pytest's collection patterns:
each is one fixed 4-rank sequence that cannot run as independent cases.

Identity is the directory name, and it carries all three segments:
`gpt_oss_120b__sm_103__tp1`. They were three nested directories once, which
read as a hierarchy that was never one -- every level had exactly one child,
and each needed an `__init__.py` whose only job was to exist. The class name
carries the same triple; `test_modeling_v2_claims.py` asserts they agree.

### Why beside `_torch/models/`, not inside it

Targets are not registered in the model zoo: `route` returns the class and the
loader builds it. Inside `_torch/models/` they would still be swept up by the
zoo's own guards -- `test_lazy_model_zoo.py` scans that directory and expects
every modeling module there to appear in the built-in static index.

Conceptually `models/` means "one architecture, one class, shared across
checkpoints", which is the opposite of what this package is for.

## Gates

1. **Boot** — minutes, binary. `examples/llm-api/quickstart_advanced.py` with
   the target's topology flags and `modeling_v2=require` passed. Engine
   cold start, weight-manifest coverage, a handful of greedy continuations.
   Catches catastrophes, not accuracy. A variant that changes the forward gets
   its own minutes-scale gate on the same footing.
2. **Accuracy** — the release criterion. `trtllm-eval` with the protocol from
   `tests/integration/defs/accuracy/references/` and `modeling_v2=require`
   passed. One-sided: measured >= reference − tol. This is the gate CI runs,
   in the accuracy suite's modeling_v2 files.
3. **Acceptance** — required whenever a variant is distribution-preserving by
   construction, speculative decoding above all. Rejection sampling holds the
   emitted distribution to the target model's, so a *miscomputed* draft path
   produces correct text more slowly: boot passes, accuracy passes, only
   speed moves. The detector is `acceptance_length` against the same
   checkpoint under stock in-tree modeling at the same workload and the same
   `speculative_config` — i.e. `modeling_v2=off` versus `=require`,
   which is now one variable rather than two harnesses.

Compare a variant against an identity run **in the same session**: one target
measured 94.7688 and 95.0720 on a bit-identical forward a day apart, so a
cross-session delta carries session variance into the judgement.

Perf is measured, never gated.

## Status of every record in this tree

**Every entry is certified on sm_103, and both targets are gated there.**

The 19 entries -- 8 shared, 8 owned by deepseek, 3 by gpt-oss -- each drive
their cells through `__call__` and `reference` in pre-merge CI, plus the two
4-rank collective matrices. Certification is per architecture: `ARCHS` says
where the cells were driven, and a missing architecture reads as unknown,
which is the honest state.

Getting the catalog there surfaced four real differences. None was resolved by
widening a tolerance, and each is written up in its own entry:

* **Op schema drift** (`thop_attention`, `mla_rope_generation`,
  `mla_rope_append_paged_kv_assign_q`). Parameters were renamed and added. The
  wrappers now mirror their schemas argument for argument, so the next drift
  fails loudly rather than shifting a positional list silently.
* **The MoE FC1 epilogue changed block-scale recipe**, bit-exactly:
  `floor(log2(amax))-8` on sm_100, `ceil(log2(amax/448))` on sm_103. The
  reference is architecture-keyed and each arch refutes the other's recipe.
* **torch 2.12 made fp32 matmul default to TF32**, so `cublas_mm`'s *reference*
  was the imprecise side; the op is bit-identical to a TF32-disabled product.
* **The MLA append op now accepts NVFP4 latent pools** as well as fp8 (accepted
  by the op, not certified here).

Both targets are gated by the accuracy suite's modeling_v2 files,
`tests/integration/defs/accuracy/modeling_v2/test_{gpt_oss,deepseek_v3}.py`,
with `modeling_v2="require"` in force, which is what rules out the built-in
implementation having been measured instead. gpt-oss-120b / sm_103 / tp1 runs
gsm8k; deepseek-r1-0528-nvfp4 / sm_103 / dep4 runs gsm8k for the identity
forward and the `mtp3` variant **paired in one session**, and the acceptance
gate against stock modeling under the same `speculative_config`. The numbers
live with those runs, not here: a number written into this file is one that
drifts with nothing to notice.

The acceptance row is the one that matters for a speculative variant:
rejection sampling makes a miscomputed draft layer *slower*, not wrong, so
boot and accuracy are blind to it and only the acceptance rate against a
reference can see it. Draft lengths 1 and 2 remain ungated -- they are the
dominated end of the measured draft-length axis.

### sm_107 and the nemotron_h target

`Arch.SM_107` is the second architecture. The two Nemotron-3 Ultra NVFP4
targets (`models/nemotron_h/nemotron3_ultra_nvfp4__sm_107__{tep4,dep4}`) are
built from entries driven on sm_107 at both targets' call sites:

* shared, certified on sm_107 beside sm_103: `thop_attention`, `cublas_mm`,
  `flashinfer_rmsnorm`, `flashinfer_fused_add_rmsnorm`, `fp4_quantize`,
  `fp4_block_scale_moe_runner`, `allgather`, `reducescatter`, and the two
  promoted from other targets -- `noaux_tc_op` (from deepseek) and
  `causal_conv1d_fwd` / `causal_conv1d_update` (from qwen4_exp);
* shared, sm_107 only (promoted from tep4 when dep4 arrived):
  `static_quantize_e4m3_per_tensor`, `cublas_scaled_mm`,
  `mamba_chunk_scan_combined`, `selective_state_update`, `rms_norm_gated_grouped`;
* shared with the sm_100 BF16 tep8 target below, certified on sm_107 and sm_100
  (promoted from tep4 when tep8 arrived): `allreduce`.

What the sm_107 cells surfaced, each written up in its entry:

* **The relu2 FC1 epilogue takes its own scalar convention.** For act type
  Relu2 the kernel computes `out1 * relu(out1_gate * acc)^2`: `out1_gate`
  dequantizes FC1 and `out1` is FC2's input quantization alone. The SwiGlu
  convention (`out1 = out1_gate / fc2_input_scale`) returns exactly zero on this
  checkpoint's scales.
* **A NemotronH hybrid pool puts attention in pool 1**, mamba state in pool 0 --
  the reverse of qwen4_exp's -- so attention addresses rows by
  `kv_cache_manager.layer_offsets`, never by counting attention layers.
* **Static FP8 activation scales saturate** on the mamba `out_proj` input at the
  first token of a chat-templated prompt; the quantize entry certifies the
  saturating conversion.

Both targets are gated by
`tests/integration/defs/accuracy/modeling_v2/test_nemotron_h.py` (gsm8k, the
protocol of `TestNemotronV3Ultra`, `modeling_v2="require"`).

### sm_100 and the BF16 nemotron_h target

`Arch.SM_100` is the third architecture. `models/nemotron_h/nemotron3_ultra_bf16__sm_100__tep8`
runs the BF16 release of Nemotron-3 Ultra -- no quantization at all, 1.1 TB of
weights -- on 8 sm_100 ranks, with tensor-parallel attention, mamba and shared
expert and expert-parallel routed experts. It has run on two GB200 nodes; an
8-GPU HGX B200 node, the single-node shape of the same routing, has not been
tried. It is the sm_107 NVFP4 tep4 target with the quantization taken out and
the rank count doubled. It has no dep8 sibling: attention data parallelism
replicates every non-expert weight on every rank (about 63 GiB: the mamba
projections, the shared experts, attention, the latent projections, the
embeddings) beside 120 GiB of experts, which is 183 of a 185 GiB GPU before the
unsharded fp32 recurrent state (384 MiB per sequence), the KV cache or any
activation. Routing claims it for the BF16 checkpoint (no quantization
declared), compute capability (10, 0), exactly 8 ranks tensor-parallel with
expert-parallel MoE, attention DP off and no speculative decoding.

What differs from the NVFP4 target:

* Every GEMM is `cublas_mm` over bf16 weights. The FP8 `static_quantize_e4m3_per_tensor`
  / `cublas_scaled_mm` pairs, the NVFP4 `fp4_quantize` / `fp4_block_scale_moe_runner`
  pair and every `input_scale`, `weight_scale` and `weight_scale_2` the other
  checkpoint carries are gone, and with them the expert weight layout recipe
  (32-row block shuffle, 128x4 scale swizzle): the experts are stacked as the
  checkpoint stores them.
* The routed experts run through `fused_moe`, the CUTLASS grouped GEMM, with
  `activation_type` Relu2: FC1 is the up projection alone (`[E, 5120, 2048]`, no
  gate half), FC2 is `[E, 2048, 5120]`, the weights are this rank's 64 experts and
  the ids are global, so an expert-parallel split is 8 calls whose outputs sum.
  The entry was deepseek's own (`fused_moe`, SwiGLU, bf16); it is promoted to the
  catalog with its test and gains Relu2 and the Ultra geometry as cells.
* The KV cache is bf16 (`quant_mode` 0, no scales to pass) and the SSM state is fp32,
  the checkpoint's `mamba_ssm_cache_dtype`; the NVFP4 gate runs fp16. The target
  asserts both, and a bf16 conv state, when it takes the pools off the cache manager:
  `kv_cache_config.dtype` or `mamba_ssm_cache_dtype` set to anything else would
  otherwise be read and written in the wrong width without a shape error.
* TP8 geometry per rank: 8 query heads over one kv head (each of the 2 kv heads is
  replicated on four ranks), 32 mamba heads in the rank's single group, a 1280-wide
  slice of the shared expert, 64 of the 512 experts.
* The serving envelope is `max_batch_size` 256: decode graphs up to 256 rows, and an
  engine recurrent-state pool of 271 slots (35 at `max_batch_size` 32; the target logs
  the depth when it binds the pools). One slot holds the 48 Mamba layers side by side,
  so a layer's fp32 SSM view has slot stride 48 * 32 * 64 * 128 = 12.6M elements and
  the slots from 171 on lie past 2^31 elements. Every batch-dependent entry the target
  calls carries cells at 256 rows, the SSM entries at that pool's last slot, and the
  accuracy gate runs at `max_batch_size` 256.
* The all-reduce is `AUTO` inside one node (the IPC one-shot / two-shot kernels for
  decode-sized payloads, NCCL for large ones). Across nodes the IPC workspace has no
  peers, so on an NVL72-class system (aarch64, fabric handles) the target uses the MNNVL
  kernels, which fold the residual add and the next block's RMSNorm into the all-reduce:
  one kernel up to 1 MiB of payload (8 rows at 8 ranks), an all-reduce kernel and an
  RMSNorm kernel above it. Anywhere else a multi-node group runs NCCL followed by the
  fused add + RMSNorm. The user's `allreduce_strategy` decides, as it does for the
  engine's own `AllReduce`: `AUTO` and `MNNVL` take MNNVL where it applies, `NCCL` opts
  out. The log names the path a run took ("all-reduce over 8 ranks via ..."). Measured
  at 8 ranks over two GB200 nodes (trtllm-bench, c=32, 1k/1k, 256 requests, one
  allocation): 2473 tok/s with MNNVL, 2087 with `allreduce_strategy: NCCL`, 2398 for the
  built-in implementation's default.

Entries: `Arch.SM_100` is declared on every shared entry the target calls, each driven
whole on sm_100: `thop_attention`, `cublas_mm`, `flashinfer_rmsnorm`,
`flashinfer_fused_add_rmsnorm`, `rms_norm_gated_grouped`, `noaux_tc_op`,
`causal_conv1d_fwd`, `causal_conv1d_update`, `mamba_chunk_scan_combined`,
`selective_state_update`, `flashinfer_selective_state_update` and
`flashinfer_direct_bf16_gemm` (new, sm_100 only) and the promoted ones, `fused_moe`,
`allreduce` and `mnnvl_fusion_allreduce`. The
cells added for tep8 -- the rank's in_proj, qkv, o_proj and shared-expert GEMM shapes;
8 q / 1 kv head over a bf16 pool, including rows 4 and 55 of a real
`MambaHybridCacheManagerV2`'s 60-row mapping; conv, scan and decode-step shapes at 32
heads in one group over an fp32 state pool; one group of 2048 for the gated norm; Relu2
over a window of 64 experts -- were driven on sm_100 only, so on an entry that also lists
sm_103 or sm_107 they are not a claim about those.

The target calls two all-reduce entries, each with a multi-rank matrix. `allreduce` (the
IPC / NCCL sum) is the catalog's, shared with tep4; its matrix sits with the catalog's
other collectives, runs only on the architectures its `ARCHS` names (sm_100, sm_107) and
skips elsewhere, and runs at 8 ranks where the node has 8 GPUs and at 4 where it has 4. It
was driven at 4 ranks (one GB200 node), so the one-shot / two-shot kernels at 8 ranks
inside one node -- an HGX B200 -- are not covered by any run yet. The other,
`mnnvl_fusion_allreduce` (the sum, the residual add and the next RMSNorm in one op over
the multicast workspace; `ARCHS` sm_100, NVL72 only), was this target's own under its
`ops/comm/` and moved to `catalog/comm/` when the tp8ep2 sibling below arrived calling
it too; its matrix moved beside the other collectives' and skips off sm_100. It was
driven at 4 ranks on one GB200 node and at 8 ranks over two: both kernels, told apart by
the stage count the workspace records rather than by the payload size; exact payloads
bitwise on `residual_out` and inside a derived band on `norm_out`; a 108-call decode graph
replayed three times at one row (one-shot) and at 32 rows (two-shot); and a graph captured
on the engine's default workspace that keeps replaying after the workspace grew to the
8192-row footprint. The accuracy gate ran on two GB200 nodes with the MNNVL path (94.807
against the reference's 94.769) and, before that path existed, with NCCL across nodes
(94.693 and 94.655).

No CI stage runs the sm_100 pieces yet. `l0_b300.yml` lists the modeling_v2 directory for
sm_103 (waived today), nothing lists it for sm_100, and the accuracy gate and the BF16
`noaux_tc_op` cells read the 1.1 TB BF16 release from `LLM_MODELS_ROOT`, which has to be
staged on the CI model share first.

What the sm_100 runs surfaced, each written up in its entry or test:

* **The BF16 release's router biases are not on the NVFP4 checkpoint's 0.125 grid.**
  Layer 1's span 0.0022 over 23 levels and layer 107's 0.35 over 394, so selection
  leans on the fp32 bias add more than the NVFP4 cells did; `noaux_tc_op` has a cell
  for each layer read from the BF16 release.
* **Relu2 squares FC1's output**, which doubles the accumulation-order noise reaching
  FC2: over the whole tactic space `fused_moe` stays inside the same gates at the Ultra
  geometry as at R1's, with a thinner element-wise margin (the test's `_compare` has the
  measured worst cases).
* **The IPC all-reduce has no peers across nodes**: `get_allreduce_workspace` allocates
  nothing to open there, so the target passes NCCL explicitly when the mapping spans
  nodes (and MNNVL is not taken) rather than leaving `AUTO` to discover it.
* **The MNNVL kernels round the all-reduce sum to bf16 before the residual is added**,
  one rounding more than NCCL followed by the fused add + RMSNorm, which adds in fp32.
  The one-shot and two-shot kernels also associate the last fp32 product differently
  (`(x * rcp) * gamma` against `(gamma * x) * rcp`). `residual_out` is gated bit for bit
  on the exact payloads (and measured at zero difference on random ones); `norm_out`
  differs from the entry's `reference` by a rare one-flip bf16 rounding of an element,
  which is where the matrix's band comes from. Worst measured, element-wise / RMS in the
  band's units: 0.93 / 0.007 at 4 ranks and 0.92 / 0.003 at 8, against gates of 2.0 / 0.02.

The target is gated by
`tests/integration/defs/accuracy/modeling_v2/test_nemotron_h.py::TestModelingV2Nemotron3UltraBf16Sm100Tep8`
(gsm8k, the protocol of `TestNemotronV3Ultra`, 8 ranks, `modeling_v2="require"`). It
reads the BF16 release as `NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` under
`LLM_MODELS_ROOT`, and its reference entry is the one `references/gsm8k.yaml` carries for
`nvidia/Nemotron-Ultra-V3` with neither a `quant_algo` nor a `kv_cache_quant_algo`.

#### The tp8ep2 sibling

`models/nemotron_h/nemotron3_ultra_bf16__sm_100__tp8ep2` is the tep8 target with the routed
experts laid out the way the engine's mapping lays them out for `moe_expert_parallel_size` 2
under `tensor_parallel_size` 8: two windows of 256 experts, each expert's 5120-wide
intermediate split over the four ranks of its window -- rank r runs window r % 2 and slice
r // 2 (FC1 rows and FC2 columns [1280 t, 1280 t + 1280)). The weight bytes per rank are
tep8's, and so is every call, shape for shape, except `fused_moe`, which is handed the rank's
slices of its 256 experts and its tp coordinates; the all-reduce that completes a MoE block
sums the windows and the slices at once.

Why a second layout: a decode step's tokens do not spread evenly over tep8's eight windows,
and the all-reduce after every MoE block waits for the rank that drew the most -- in a rank-0
profile it runs 3-5x longer than the one after a mamba block, about 1 / 4.4 / 5.2 ms per step
at 1 / 32 / 128 concurrent requests. On the built-in implementation, two GB200 nodes, one
allocation (`max_batch_size` 256, ISL/OSL 1k/1k), moe_ep 2 x moe_tp 4 delivers 3.5 / 6.3 /
5.3% more output throughput than moe_ep 8 at 1 / 32 / 128 concurrent requests for 1.4% more
time to first token. Splitting every expert over all eight ranks instead (moe_ep 1) gains as
much at 32 but less at 1 and 128, and costs 12-13% time to first token: its 640-wide slices
leave the prefill's grouped GEMMs short.

Routing claims it for the BF16 checkpoint on sm_100 at tp 8 with moe_ep 2 / moe_tp 4 and
attention DP off. `fused_moe` carries six cells for the layout (decode at 1 / 32 / 256 rows, a
300-row chunk and an 8192-row prefill, over both windows and all four slices); its reference
needs no tp coordinates -- the partial sum over the slices it is given -- which those cells
confirm. Every other entry is called exactly as tep8 calls it, and `mnnvl_fusion_allreduce`,
now called by both, moved to the catalog. The target is gated by
`tests/integration/defs/accuracy/modeling_v2/test_nemotron_h.py::TestModelingV2Nemotron3UltraBf16Sm100Tp8ep2`
(gsm8k at `max_batch_size` 256, the tep8 gate's protocol and reference): 94.579 against the
reference's 94.769.

#### FlashInfer's decode step at high concurrency

From 32 decode rows on, both BF16 targets run the Mamba2 decode step through
`flashinfer_selective_state_update` (FlashInfer's CUDA kernel) instead of
`selective_state_update` (Triton). At the rank's shapes -- 32 heads of 64, one group of 128
states, an fp32 state -- FlashInfer's kernel replays 1.17x as fast at 32 rows, 1.31x at 128 and
1.36x at 256, and is the slower one below 32 (0.87x at 8 rows, 0.73x at one), hence the
threshold. What the switch takes:

* The kernel accepts D and dt_bias only in dt's dtype, so the targets keep bf16 copies -- the
  checkpoint's own values -- beside the fp32 ones the Triton kernel reads.
* It reads x, B and C through TMA and needs each 128-byte aligned. A decode step's in_proj
  output is; a mixed step's generation rows are not after an odd number of context rows (8768
  bytes each), so those stay on the Triton kernel. The kernel refuses a misaligned start
  loudly, which is how a first version that also took the mixed rows failed its gate.
* Its softplus and decay use fast intrinsics, so dt carries more error than the Triton
  kernel's; the entry's test bands it from the intrinsics' documented bounds.

End to end on two GB200 nodes (one allocation, `max_batch_size` 256, ISL/OSL 1k/1k), output
throughput at 1 / 32 / 128 / 256 concurrent requests moves 0 / +1.5 / +3.4 / +4.9% on tep8 and
0 / -0.2 / +2.9 / +4.5% on tp8ep2 (at 32 the decode batch straddles the threshold; within the
noise there), time to first token unchanged; gsm8k 94.617 on tep8 and 94.920 on tp8ep2. With
both changes the tp8ep2 target serves 145.8 / 2616.9 / 5831.8 / 8309.4 output tok/s where
tep8 without FlashInfer's step served 138.9 / 2402.1 / 5356.7 / 7671.0 (mean of two runs):
+5.0 / +8.9 / +8.9 / +8.3%.

#### Decode projections at one and two rows

A decode step of one or two rows is latency-bound: every dense projection reads its whole
weight to produce a row or two. Both BF16 targets run the four K = 8192 projections of such a
step -- the mamba in_proj, qkv, the shared expert's up projection and fc1_latent -- through
`flashinfer_direct_bf16_gemm`, FlashInfer's register-prefetch kernel, instead of `cublas_mm`:
at one row it is 1.3-2.3x faster on those shapes in CUDA-graph replay, at two still ahead, and
from four rows on not, hence the two-row threshold. The router keeps `cublas_mm`: each shape
has one fixed tactic, so ranks running a replicated projection get the same bits, but the
router's fp32 logits are out of the kernel's reach. On two GB200 nodes, alternating off / on /
off / on on one allocation, tp8ep2 serves 154.4 output tok/s at one concurrent request where it
served 146.0 (+5.8%, time per output token 6.78 -> 6.39 ms) and is unchanged at eight (884 vs
890, within the noise of the two off runs); gsm8k 94.845 on tp8ep2 and 94.617 on tep8. An
nsys trace of the one-request step puts in_proj, fc1_latent and qkv / shared up at 13.3, 8.6
and 6.6 us per call in the model against 11.2, 6.5 and 4.5 us in isolation: the kernel waits
for its predecessor before it loads weights and releases its dependents only after its stores,
which an earlier PDL trigger would overlap.

### What replaced the version pin

Out of tree each target hard-asserted a pinned `tensorrt_llm` version at
import, because a target reads private engine surface and a drifting engine
silently voids its records. In tree that assert is meaningless — the target
moves with the trunk — so it is gone, replaced by an SM assert, which is the
part of the identity that does *not* move.

The same reasoning retires the version from the records themselves. An
entry's `ARCHS` is keyed by GPU architecture and nothing else: there is no
external pin left to drift against, and a version written into an entry is a
number that is wrong the next day with nothing to notice. What the claim is
fresh against is the entry's own test, and that CI re-proves on every commit.

The pin was earning its keep out of tree, though, and the migration paid the
bill immediately: the attention backends moved from
`_torch/attention_backend/{interface,trtllm}.py` to
`_torch/attention/backends/`. The compatibility shim left behind re-exports
the names but not the submodule paths, so all five files importing them
failed at import. They now use the canonical path.

The lesson generalizes: with no pin, a target's contact with private engine
surface is checked only by running it. The step contract check does that by
calling the target's own `step_args` on the metadata it was handed, which reads
every field the target consumes -- so a rename upstream becomes a loud failure rather
than a wrong answer. Declaring the same surface a second time, as a list of
names beside the code that reads it, only adds something to forget to update.

## Two facts this migration surfaced about upstream

* **`pip` is a runtime dependency of TensorRT-LLM itself.** `libtensorrt_llm.so`
  locates its bundled kernel headers on the NVRTC JIT path by running
  `pip show tensorrt_llm`. ModelingV2 was merely the first thing to write that
  down (a `uv`-managed venv does not install `pip` by default), and it is not
  a modeling_v2 dependency.
* **The catalog entries are now documentation of upstream ops.** For
  example `_torch/modules/linear.py` calls `cublas_mm(input, module.weight.t(), ...)`
  correctly, but nothing there says why the `.t()` is mandatory or that a
  non-contiguous weight computes a silently wrong answer — `cublas_mm`'s
  `is_valid` does. The reverse duty comes with it: if one of these ops changes
  and its entry does not follow, the misleading is no longer confined to
  modeling_v2.

## Op ownership

An entry only one target calls lives under that target's `ops/`; one that two
or more call lives in `catalog/`. A later target that needs an owned entry
promotes it by moving the file and its test to the shared tree -- the test is
the certification record, so nothing else has to travel with it. The five
other out-of-tree targets and the agent definitions are not migrated in this
batch.
