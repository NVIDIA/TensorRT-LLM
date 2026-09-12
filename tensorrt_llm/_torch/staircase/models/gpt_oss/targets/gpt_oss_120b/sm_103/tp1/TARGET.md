# Target: gpt-oss-120b / sm_103 / tp1

## Identity

| | |
|---|---|
| Checkpoint | gpt-oss-120b (HF safetensors; bf16 attention/router/embedding/lm_head, **MXFP4 experts** — E2M1 blocks + per-32 E8M0 scales; untied embeddings; sparse MoE on every layer: 128 experts, top-4, renormalized; **attention sinks**; alternating sliding-window / full attention) |
| GPU arch | sm_103 (GB300) |
| Parallel | tp1 |
| Registered class | `StaircaseGptOss120bSm103Tp1` — a synthetic architecture name no checkpoint declares. `models/gpt_oss/routing.py` rewrites `GptOssForCausalLM` into it when the config, SM and topology all match; the checkpoint is read unpatched. Per-target names mean one process can hold every target at once |

> **NO GATE RECORD HOLDS FOR THIS TARGET.** Every result below was
> measured on **sm_100 (B200)** through the pre-move standalone harness. This
> target is **sm_103 (GB300)**, and certification is per architecture. The
> numbers are kept as provenance — they are true records of what the same
> modeling code did on another device — but this target is **ungated** until
> the boot and gsm8k runs in *Verification* are repeated on GB300 and their
> results replace those rows. Read every "passed" below as "passed, on
> sm_100, before the move".

Checkpoint sha256 — the checkpoint directory passed to `--model` should
resolve to files with these digests. **Routing does not check them**: it
fingerprints the config's shape, so a fine-tune of this checkpoint routes
here silently. That is the deliberate trade in `models/gpt_oss/routing.py`,
and it changes what a gate record means — not "this target passed" but
"this modeling code passed *on the checkpoint with these digests*". Run it
on another one and the result is ungated (recorded at:
`umbriel-b200-027:/home/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-120b`):

```
695218884684c611fe08a74751ee443f971e9bd9bc062edba822da3fe45969b7  model-00000-of-00014.safetensors
a881aa5f561b26a22b14a8262aa61849ace349ffd73d74769e030ac90a1fcf8a  model-00001-of-00014.safetensors
022478dd04398c5bdb545a5be0a6437ecc2eb53d1dbd29edafcfff4b3ddf0a41  model-00002-of-00014.safetensors
47aee9e7b9d5bedb215042c01ccededd9bd9c30b0dddea862dc2506b9d6c74de  model-00003-of-00014.safetensors
f6c2752acda607b1d5ca52df9e75c1b9b2761e6875ff10c9bd6ddac473c0262e  model-00004-of-00014.safetensors
0c8dd401544c31cb93b8459eee7da20ea2a07626a59455d7d92b85257df9b46c  model-00005-of-00014.safetensors
28d839f2e027985a8b14e45f2323798862eddb7770ee9800ea6b7c803abee489  model-00006-of-00014.safetensors
c8958c5f183c04f6ea959cfd90562b5128124154b2bbf979b8a22b9405b30ed8  model-00007-of-00014.safetensors
bf1f2a88868ffc37d520dcf77d26f0e823710b5e682d473ff10f6974fa3b7517  model-00008-of-00014.safetensors
f72d34a4004241b45c332b61f8ffa124e9a913bc1ab442b66e717d3e94e741ce  model-00009-of-00014.safetensors
f48c867c2cb0a44bfc2f8768cb98e4aec9a350946fceacfebdcad5d32ad4a471  model-00010-of-00014.safetensors
a06851b2cfd35f48722f823bc1ab8f7bcb4a878a5b8e975f4d3544f230454eeb  model-00011-of-00014.safetensors
3af33667c307e20ae2a7648ea52653de46dd0171601ec5c696e47a2f5d5bf1e4  model-00012-of-00014.safetensors
bcbcb74b043e071d1e05471d500d74dcf661175e00878ed302ccdf1801a75aef  model-00013-of-00014.safetensors
54b1be1609696c307cc5ca117b1fa54feaddebffa04e9c2db117652a01964230  model-00014-of-00014.safetensors
ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e  model.safetensors.index.json
0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3  tokenizer.json
9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070  tokenizer_config.json
dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293  special_tokens_map.json
f8d9255777615591a7cc1a7c932f5a69e181128902295e1b81221d20d983cac7  chat_template.jinja
7bfd294f3e29b53db1e126d5cec050a12dc27adad4445fb5eab540e1cad74ea1  chat_template.json
199566674b96510c3b9a1141b494223a86a3ff83097e2c2f259c4d94fafc5847  generation_config.json
```

Two link-set notes specific to this checkpoint. Its chat template lives in
`chat_template.jinja` (`tokenizer_config.json` carries none), and the
accuracy gate's protocol applies that template — so the template files and
`special_tokens_map.json` are linked alongside `tokenizer*`. And its
`config.json` declares **no dtype at all**, which used to be patched around
with a target-owned `model_dir/config.json` declaring `dtype: bfloat16`.

That stub is gone — the checkpoint is now read exactly as published — so the
divergence it papered over is live code. `DecoderModelForCausalLM` sizes
`lm_head` from `pretrained_config.torch_dtype`, which is `None` here, and
would materialize it in the torch default fp32 while every other tensor is
bf16; the failure surfaces two layers from its cause. The shell fills that
gap explicitly before `super().__init__`, adopting the dtype the engine
already resolved (`ModelConfig.torch_dtype`, bf16) and only when the
checkpoint declares none. Every safetensors tensor outside the MXFP4 expert
blocks is bf16, so that is what the checkpoint is.

## Version

| | |
|---|---|
| tensorrt_llm | in-tree — the target moves with the trunk, so there is no version to pin and none is asserted. What *is* asserted at construction is the SM version (`_SM = (10, 3)`), which the pin used to stand in for. The gate records below name the commit they were taken at |
| torch | 2.11.0+cu130 |
| transformers | 5.5.4 (the config surface the engine hands the target; see the rope note in `modeling.py`) |
| Attention metadata fact source | `TrtllmAttentionMetadata` (TRTLLM backend) |

## Vocabulary

Forward: `flashinfer_rmsnorm`, `flashinfer_fused_add_rmsnorm`,
`cublas_mm` (fused bias — qkv, o, router all carry one),
`fused_qk_norm_rope` (`is_qk_norm=False`: YaRN RoPE only),
`thop_attention` (per-layer `attention_sinks` and `attention_window_size`),
`mxfp8_quantize`, `mxe4m3_mxe2m1_block_scale_moe_runner`,
`torch/embedding`, `torch/empty`, `torch/reshape`.

The MoE call is the whole expert block — routing, both grouped GEMMs, the
clamped GLU, the MXFP8 requantization between the GEMMs and the combine —
so no `activation/*` and no `moe/*routing*` entry appears: this checkpoint
has no dense MLP, and the router bias rides the router GEMM's fused-bias
epilogue because the MoE op silently ignores `routing_bias` at
`routing_method_type=1`.

**Changed by the perf campaign (`iter1-mxfp8-moe`, see *Performance*).**
The assembly shipped the W4A16 member of this kernel family
(`bf16_mxe2m1_block_scale_moe_runner`) with `torch/pad` widening hidden
2880 → 3072 in front of it; the target now runs the W4A8 member over
MXFP8 activations, and `mxfp8_quantize` does that widening inside itself,
so the `torch/pad` call is gone. Weight preparation is unchanged — the two
ops consume the identical prepared expert stack.

Weight loading fills parameters via the manifest loop (`torch/copy_`
semantics); `.t()` views are cublas_mm's column-major consumption form,
derived once post-load. The expert operands are the exception: the
manifest's source transforms rebuild the checkpoint's MXFP4 blocks and
E8M0 scales into the MoE op's kernel-ready layout (pad → `[up ; gate]`
concat → row interleave → 32-row block shuffle → 128x4 scale swizzle),
promote both expert biases and the attention sinks from bf16 to fp32, and
run on device one layer at a time.

Audit is mechanical: grep the forward's calls against `catalog/index.yaml`.

## Verification

### Required on sm_103 — not yet run

| Gate | Command | Result |
|---|---|---|
| boot | `TRTLLM_STAIRCASE=require python examples/llm-api/quickstart_advanced.py --model_dir <ckpt> --max_tokens 16 --prompt "The capital of France is" "The chemical symbol for gold is" "1, 2, 3, 4, 5,"` | **passed, 10/10** greedy keyword asserts, 2026-09-09, GPU 0 of nvl72d001-T18 (NVIDIA GB300, 284208 MiB, sm_103), trtllm 1.3.0rc26. 65.70 GiB of weights loaded; decode CUDA graphs captured at batch sizes 1-32, 64, 128; engine boot to last token 3m29s. The run had the switch set to `require`, so the built-in GptOss implementation could not have been substituted. Measured with a per-target `smoke.py` that asserted a keyword in each of ten greedy continuations; that file was removed in favour of the generic script above, which boots the same way and prints the same continuations for a human to read |

**Execution is verified on sm_103; the accuracy gate is not yet.** Everything up to the
weight load is driven by the checkpoint's config alone, and that part
was exercised on a GB300 against the config *as published* (no
target-owned stub): routing resolved this target, the module imported,
`StaircaseCore.__init__` passed every geometry, topology and dtype
assert, and 543 parameters declared, 36 layers, hidden 2880. `lm_head.weight`
came out **bfloat16**, which is the specific thing removing the stub
put at risk -- the shell sizes it from the pretrained dtype, and a
regression there materializes fp32 two layers from its cause.

The weight path is verified too, by the boot run above: the manifest
load fills every declared parameter (its coverage asserts are part of
the gate), the post-load derivations run, and the forward produces
coherent greedy continuations through both the prefill and the
CUDA-graph decode path.

The checkpoint it ran on is the one this file records. Every digest
above was re-verified after download -- the six small files by hand and
the fifteen safetensors shards by git-lfs, whose object id *is* the
sha256 -- so this gate record and the sm_100 records below were measured
on byte-identical weights.

| gsm8k full | `TRTLLM_STAIRCASE=require trtllm-eval --model <ckpt> gsm8k --output_path <dir> --apply_chat_template --fewshot_as_multiturn --max_output_length 8192`, or in CI as `accuracy/test_staircase.py::TestStaircaseGptOss120bSm103Tp1::test_gsm8k` | **passed, 90.6748** (`exact_match,flexible-extract`, +-0.8010, full 1319 questions) against threshold **85.5989** (anchor `openai/gpt-oss-120b` = 90.5989, tol 5.0) -- pass by 5.08 points. 2026-09-09, GPU 0 of nvl72d173-T18 (GB300), trtllm 1.3.0rc26, 2m54s wall. `strict-match` on the same run: 27.0660 |

`TRTLLM_STAIRCASE=require` is what makes the second command a gate at all: under
`auto` a configuration that missed this target would measure the built-in
GptOss implementation and report it as this target's score.


#### Two notes on reading the gsm8k number

**`--check_accuracy` is not used, and the filter is read by hand.** The gate
filter is `exact_match,flexible-extract`, and `trtllm-eval` exposes no CLI flag
for `scores_filter` -- it is a keyword of the evaluator's `evaluate()`. Left
unset the evaluator *averages* the filters, which for this checkpoint mixes
90.6748 with a `strict-match` of 27.0660 and reports 58.87. That average is
meaningless here: this is a reasoning model, its answer never arrives in the
strict `#### N` form, so the strict filter scores it near the floor. Read the
flexible-extract row of the logged table, which is also saved to
`--output_path`.

**The anchor in `references/accuracy.yaml` was deliberately not written back.**
The rule in that file is to write a target's first passing score back -- but it
also says to skip the write-back when doing so would confuse what the anchor
means. It would here: the anchor is keyed by *checkpoint* (`openai/gpt-oss-120b`)
and currently holds an sm_100 measurement of the W4A16 assembly, deliberately
left high so the gate stays the stricter of the two. This measurement is a third
thing -- the W4A8 forward on sm_103 -- and folding it into a checkpoint-keyed
anchor would make that anchor architecture-dependent without saying so. The
number lives here, where gate records are per target and therefore per
architecture.

For the record, the three measurements of this checkpoint through this harness:
sm_100 W4A16 **90.5989** (the anchor), sm_100 W4A8 **89.16**, sm_103 W4A8
**90.6748**. The last is +1.51 over the sm_100 W4A8 record, which is ~1.9 sigma
on this run's +-0.80 stderr and is **not** claimed as an improvement -- the MoE
FC1 epilogue genuinely uses a different block-scale recipe on sm_103 (see
`catalog/moe/mxe4m3_mxe2m1_block_scale_moe_runner.md`), so a difference of this
size has a plausible mechanism, but separating it from session variance would
need repeated runs on both architectures.

### Prior record — sm_100 (B200), pre-move harness, does not gate this target

Both gates passed under trtllm defaults (block reuse and CUDA graphs on,
`llm_args.yaml` empty, single-process worker via `scripts/env.sh`),
2026-07-26/27, GPU 0 of umbriel-b200-027 (NVIDIA B200, driver 595.58.03):

| Gate | Result |
|---|---|
| smoke — `uv run targets/gpt-oss-120b/sm_100/tp1/smoke.py` | **passed**, 10/10 greedy keyword asserts (keywords frozen against continuations observed on this model; the arithmetic case is Q/A-framed because a bare `2 + 2 =` is genuinely ambiguous here) |
| gsm8k full — `uv run bench/accuracy.py --target targets/gpt-oss-120b/sm_100/tp1` | **passed**, `measured 89.16 >= 85.6`, exit 0 (1319 questions, 5-shot, chat template + few-shot-as-multiturn + 8192 output tokens, filter `exact_match,flexible-extract`) |

**The row above is the shipped forward** — the W4A8 MoE the perf campaign
left in place. The gate it clears is the written-back anchor
`openai/gpt-oss-120b` = 90.5989 (`source: trtllm-eval`, tol 5.0 ⇒
threshold 85.6): **pass by 3.56 points**.

The anchor itself was measured on the *assembly's* W4A16 forward, and is
deliberately not re-written down to the W4A8 number — leaving it high
keeps the gate the stricter of the two, and it remains a real measurement
of this checkpoint through this harness. The W4A16 record it came from,
kept because the anchor derives from it: `measured 90.60 >= 85.3` against
`reference: openai/gpt-oss-120b = 90.3 (trtllm)`, exit 0, same protocol —
reproduced bit-identically by two independent invocations, the
assembler's and the orchestrator's verification re-run.

Measured GSM8K on that W4A16 forward, full 1319 questions (2026-07-27):

| filter | score |
|---|---|
| **flexible-extract (gated)** | **90.5989 ± 0.8039** |
| strict-match (not gated) | 25.4738 ± 1.2002 |

Reference `openai/gpt-oss-120b` = 90.3 (`source: trtllm`), tol 5.0 ⇒
threshold 85.3: **pass by 5.30 points**, and 0.30 *above* the anchor
itself. Stock trtllm on this checkpoint under the identical protocol
measured 90.2199 here during the onboard's anchor phase. The strict-match
filter is not the gated metric and is not comparable: it scores the
`#### N` surface form a harmony-format model never emits.

**Rerun noise measured on this target, protocol-specific.** The same build
under the identical protocol scored 90.2199 in a first run and 90.5989 in
the gating run — 1190 vs 1195 of 1319, a 0.38-point spread with no code
change between them (strict-match swung further, 22.59 → 25.47). GSM8K
with 8192-token reasoning generations is therefore noisier than the
~0.1-0.3 the repo's guidance cites for completion MMLU: treat anything
under ~0.5 points on this gate as unresolved.

Engine facts observed on those runs: model init 14.1-14.5 s (63 GB of
checkpoint, including the on-device expert relayout); one KV pool sized
for full attention (99.27 GiB, 1,445,664 tokens, `tokens_per_block` 32,
`window size=131072`), `host_kv_cache_pool_mapping` `[36, 2]` with
identity rows `[0, l]` — the runtime is not told about the 128-token
sliding layers, which is exactly the certified single-pool route, and the
alternation therefore saves no memory. Both FMHA kernel families appear at
warmup (`...H64PagedKvDense...` for the full layers,
`...H64PagedKvSlidingOrChunkedCausal...` for the sliding ones), which is
the per-layer window taking effect. Under trtllm defaults
`cache_reuse=True`, so the engine prepares `use_paged_context_fmha=True`
and the target passes it through as documented; both gates above ran with it
True. That value was **not certified** when this target merged — the
contract's certified column said `False`, and this line was the only
record of the mismatch anywhere. It was closed on 2026-07-28 by a
certification extension, which also established that the target's
behaviour was the correct one: at `False` a context call with a cached
prefix returns normally and lands 51x outside the tolerance band, while
with nothing cached the flag is bitwise inert. The paged read does add a
caller obligation — the pages in range must be valid and distinct — which
this target satisfies by passing the engine's own offsets through.

Pre-gate check of the highest-risk axis, run before the first engine boot
(scratch script, not a repo product): layer 0's real MXFP4 expert tensors
prepared by `weights.py` and fed to the catalog MoE call land **1.12 bf16
ulp** (of the token row's largest magnitude) from a pure-torch HF
reference — dequantized with the transformers mxfp4 semantics, gate/up
split as `[..., ::2]` / `[..., 1::2]`, clamped GLU, top-4 renormalized
routing — while the gate/up-swapped reference lands **146.36 ulp**. That
is the interleave-parity trap, pinned by measurement rather than by
reading.

## Performance

![Serving Pareto](perf/figures/pareto.png)

Environment — every curve below: GPU 0 of `umbriel-b200-027` (NVIDIA B200,
driver 595.58.03, CUDA 13.0), one device for the whole campaign,
2026-07-27; `tensorrt_llm 1.3.0rc21`, `torch 2.11.0+cu130`,
`transformers 5.5.4`, single-process worker via `scripts/env.sh`. Sweeps:
`bench/perf.py`, ISL=OSL=1024, concurrency 1→256, requests per point =
concurrency × rounds (20 for con≤8, else 5). `baseline` 07:30-08:10 UTC,
`trtllm` 08:10-08:38 UTC (back to back), `iter1-mxfp8-moe` 13:29-13:55 UTC.
Perf is recorded, never gated.

### Curves

`peak tok/s/gpu` is the maximum over the sweep; the concurrency that
reaches it is in parentheses.

| label | config | commit | accuracy | con=1 tok/s/user | peak tok/s/gpu | change |
|---|---|---|---|---|---|---|
| `trtllm` | `llm_args.yaml` only (`{}`) | `c5e6d53` | ungated reference | 394.66 | 7962.9 (con=128) | stock trtllm modeling, stock defaults |
| `baseline` | `llm_args.yaml` only (`{}`) | `c5e6d53` | 90.5989 | 228.74 | 4476.7 (con=256) | staircase at trtllm defaults, W4A16 MoE |
| `iter1-mxfp8-moe` | `llm_args.yaml` only (`{}`) | `41f955a` | **89.1585** | 380.35 | **8421.1 (con=128)** | MoE swapped to the W4A8 (MXFP8-activation) member of the same kernel family |

No config variant was kept, so every curve runs the identity config and
`configs/` does not exist. Both accuracy cells are full 1319-question
gsm8k under the target's own protocol; each reproduced bit-identically
across two independent `bench/accuracy.py` invocations.

Full point-by-point (tok/s/gpu, ISL=OSL=1024):

| con | trtllm | baseline | iter1 | iter1 vs baseline | iter1 vs trtllm | iter1 tpot ms | iter1 ttft ms |
|---|---|---|---|---|---|---|---|
| 1 | 394.6 | 228.7 | 380.3 | +66.3% | −3.6% | 2.61 | 22.3 |
| 2 | 690.0 | 324.0 | 669.7 | +106.7% | −3.0% | 2.96 | 35.0 |
| 4 | 1181.2 | 527.8 | 1146.2 | +117.2% | −3.0% | 3.45 | 47.6 |
| 8 | 1856.8 | 749.9 | 1866.2 | +148.9% | +0.5% | 4.22 | 76.6 |
| 16 | 2799.8 | 1099.6 | 2872.8 | +161.3% | +2.6% | 5.47 | 109.2 |
| 32 | 4028.0 | 1434.6 | 4186.2 | +191.8% | +3.9% | 7.50 | 155.9 |
| 64 | 5851.3 | 2015.0 | 6066.8 | +201.1% | +3.7% | 10.34 | 218.0 |
| 128 | 7962.9 | 2760.4 | 8421.1 | +205.1% | +5.8% | 14.89 | 306.6 |
| 256 | 5509.8 | 4476.7 | 7267.3 | +62.3% | +31.9% | 34.74 | 455.2 |

### Kept iterations

**iter1 — the MoE call from W4A16 to W4A8 (MXFP8 activations).**

*Evidence.* Four `nsys` steady-state windows (100 executor iterations,
every one pure decode) said the W4A16 forward was GPU-saturated at every
concurrency — idle −10.7% / −1.1% / −0.8% at con=1 / 128 / 256 (negative =
slight cross-stream overlap) — and that 60.8% / 91.9% / 89.8% of that GPU
time was the MoE call. Per layer per step at con=128 its two grouped GEMMs
cost 711.8 µs and 352.9 µs against the stock reference's 126.7 µs and
65.6 µs — **5.62× and 5.38× on byte-identical expert weights**. The
reference resolves this checkpoint's `quant_method: mxfp4` to
`W4A8_MXFP4_MXFP8`; our kernel's name carries `castBfloat16` (MXFP4
expanded to bf16 for a `m128x8x16` bf16 MMA at 3 pipeline stages, 1-CTA
clusters) while the reference's feeds MXFP4 straight into a block-scaled
`m256x16x32` MMA at 6 stages, 2-CTA clusters. Non-MoE GPU work already
matched (3.41 vs 3.10 ms/step), so the MoE was the entire gap.

*Change.* One catalog call swapped for one: `mxfp8_quantize(o, False, 512)`
produces e4m3 data plus per-32 UE8M0 **linear** block scales, and
`mxe4m3_mxe2m1_block_scale_moe_runner` consumes the pair. The quantizer
performs the hidden widening 2880 → 3072 itself, so the `torch/pad` that
fed the W4A16 call is gone — the forward issues 530 kernels per decode step
instead of 566. `weights.py` is untouched: both ops read the identical
prepared expert stack. The scale layout is spelled out rather than
defaulted, because the 128×4 swizzled buffer has the same byte count
whenever `num_tokens % 128 == 0` — every decode CUDA graph of 128 or 256 —
and is then accepted as a silently wrong answer.

*Effect.* Every point improves, from +62.3% (con=256) to +205.1%
(con=128); peak throughput +88.1% (4476.7 → 8421.1 tok/s/gpu) and con=1
+66.3% (228.74 → 380.35 tok/s/user). TTFT at con=1 falls 54.1 → 22.3 ms.
The curve also changes shape: it now peaks at con=128 and turns over at
256, exactly as the stock reference does.

*Accuracy cost — a result, not noise.* gsm8k **89.1585**, gate passed by
3.56 points (threshold 85.6), but **1.44 points below** the W4A16
measurement of 90.5989 recorded above. Two independent invocations of
`bench/accuracy.py` returned 89.1585 bit-identically (stderr 0.8564 both
times), as the W4A16 path reproduces 90.5989 bit-identically — so this is
the recipe's price, well outside the ~0.5-point band this gate leaves
unresolved. The mechanism is documented in the op's contract: FC1's
epilogue requantizes the activation to MXFP8 on the OCP scale (`floor` of
the block exponent, block max saturating) before FC2 reads it. For
reference, stock trtllm running this same recipe measured 90.2199 here
during the onboard.

### Gap decomposition

`trtllm-tuned` is **absent by construction**: no config variant was kept,
so it would be byte-identical to `trtllm` and the portable-config share of
the gap is 0. The decomposition below is therefore kernel-level, from
`nsys` windows of 100 pure-decode iterations at con=128:

| | baseline (W4A16) | iter1 (W4A8) | trtllm |
|---|---|---|---|
| step (wall) | 41.84 ms | **16.10 ms** | 16.18 ms |
| GPU active | 42.30 ms | 11.60 ms | 10.60 ms |
| GPU idle | −1.1% | **+27.9%** | +34.5% |
| MoE total | 38.89 ms | 8.34 ms | 7.51 ms |
| non-MoE | 3.41 ms | 3.26 ms | 3.10 ms |
| kernels / step | 566 | 530 | 502 |

The target's decode step is now **16.10 ms against the reference's
16.18 ms**, and the bottleneck has moved: this forward was 100% GPU-bound
and is now 27.9% idle, i.e. host-bound in the same regime as the stock
reference. What remains at the kernel level is an 11% MoE-GEMM difference
from tactic selection alone — the autotuner picks `t128x8x512_s3` /
`m128x8x32` / 1-CTA for us (139.3 and 74.1 µs per layer) against the
reference's `t128x16x256u2_s6` / `m256x16x32` / 2-CTA (126.7 and 65.6 µs)
on the same op family. The new quantize call costs 143.12 µs per step
(36 calls, 3.98 µs each) — 1.2% of GPU-active time.

### Reference lines

- `trtllm` = the original HF checkpoint under **stock in-tree trtllm
  modeling**, sharing this target's `llm_args.yaml` (identity config, `{}`
  at tp1) and otherwise trtllm defaults — `bench/perf.py --trtllm`. Since
  `iter1`, both systems run the same W4A8 numerical recipe, so this is now
  a like-for-like comparison; against `baseline` it was not (that curve is
  W4A16).
- `trtllm-tuned` is absent by construction (no kept config variant).
- The reference is **ungated** — no accuracy gate is run against in-tree
  modeling. Pinned to `tensorrt_llm 1.3.0rc21`.

### Measurement caveats

- The host CPU of `umbriel-b200-027` is shared and other tenants ran GPU
  work on neighbouring devices during the campaign; GPU 0 was reserved
  throughout, but host and chassis contention is an error bar on every
  point.
- **Session variance at mid-curve concurrencies exceeds the 3% rule of
  thumb.** Under the W4A16 MoE, con=128 at rounds=2 measured 3097.2,
  3101.0 and 2725.2 tok/s across three sessions of configurations later
  shown equivalent (12.1% spread) against 0.35% at con=1 and 0.77% at
  con=256. Under the W4A8 MoE the same point was far steadier: two control
  probes 26 minutes apart measured 8615.3 and 8603.4 tok/s (0.14%). Probe
  A/Bs in this campaign therefore always carry a same-session control.
- Wrap-up spot-check of the reference (`--trtllm`, con=1/32/256, default
  rounds, 5 h after the label): 394.0 (−0.15%), 4267.2 (+5.94%), 5607.6
  (+1.77%). The `trtllm` label was **not** re-swept: the +5.94% sits inside
  the session variance above, and re-measuring one curve alone would break
  the back-to-back pairing with `baseline`.
- Profiled runs are never comparable to clean ones: under `nsys` the
  W4A16 con=128 point measured 2468.4 tok/s against 2760.4 clean.

Figure regeneration:

```
uv run utils/plot_pareto.py targets/gpt-oss-120b/sm_100/tp1/perf/data \
    -o targets/gpt-oss-120b/sm_100/tp1/perf/figures/pareto.png
```

### Tried and rejected (with the number that rejected it)

The first four were measured **against the W4A16 forward**, when the GPU
was saturated; that scope matters, because the first one changed verdict
after iter1 and had to be re-tested.

| hypothesis | axis | measured | verdict |
|---|---|---|---|
| decode CUDA-graph coverage 256 (`cuda_graph_config.max_batch_size: 256`, `enable_padding: true`) — **under W4A16** | config | con=1 +0.04%, con=128 +0.12%, con=256 +0.77% | inert: no idle existed to recover |
| `max_seq_len: 12288` (blocks/seq 4096 → 384, confirmed in `server.log`) | config | con=1 −0.3%, con=256 +0.07% | inert: the per-step H2D block-offset staging measures 4.19 MB / 87 µs = 0.2% of a con=128 step |
| `cute_dsl_bf16_gemm_blackwell` for the qkv / o / router GEMMs | modeling | graph-replay kernel time at M=1: 7.12 vs cublas+bias 5.95 µs (qkv), 7.30 vs 6.28 (o), 5.61 vs 5.18 (router) | rejected: slower at every shape before the bias cublas fuses |
| FC1 K-padding 3072 → 2944 (−4.17% of FC1 weight bytes) | modeling | MoE µs/layer: T=128 1176.3 vs 1151.4 (worse), T=256 1177.0 vs 1191.9 | rejected: no consistent gain |
| decode CUDA-graph coverage 256 — **re-tested under W4A8** | config | see below | **trade-off, not shipped** |

**The graph-coverage trade-off, re-measured after iter1.** Once the MoE
stopped saturating the GPU, the con=256 point began to collapse the way the
stock reference's does (8421.1 at con=128 → 7267.3 at con=256). Capturing a
batch-256 decode graph recovers it, but costs the con=128 point. Same
session, control probed twice before and after (rounds=2, con=128/256):

| config | con=128 | con=256 |
|---|---|---|
| identity (control, 14:04) | 8615.3 | 7221.6 |
| identity (control, 14:30) | 8603.4 | 7516.3 |
| `max_batch_size: 256`, padding on | 7897.7 (−8.3%) | 9805.3 (+33.1%) |
| `max_batch_size: 256`, padding off | 7753.2 (−10.0%) | 9792.2 (+32.9%) |

Padding is not the mechanism — exact-hit coverage regresses con=128 just as
much — and neither is memory: the KV pool moves 99.31 → 99.15 GiB (0.16%)
and graph memory 10.83 → 11.28 GiB. Decode TPOT at con=128 rises 14.35 →
15.97 ms with the extra graph present. The keep rule is regress-nowhere, so
this is **not shipped**; a deployment that only ever serves con≥256 should
adopt it deliberately, since its con=256 point (9805.3 tok/s at 38.6
tok/s/user) is outside anything the shipped config reaches.

### Remaining headroom

- **The target is now host-bound at the throughput end** (27.9% GPU idle at
  con=128), in the same regime as stock trtllm (34.5%) and for the same
  reason: stock executor Python between steps. No knob in the tuner's table
  reaches it.
- **~11% of the MoE GEMM time is tactic selection**, not recipe: same op,
  same weights, different autotuner choice than the in-tree path makes
  (`t128x8x512_s3`/1-CTA vs `t128x16x256u2_s6`/2-CTA). Worth a look at what
  drives the autotuner's bucket set.
- **The con=256 graph-coverage trade-off above** is a real +33% at the
  throughput end blocked by a −9% at con=128 whose mechanism is not
  established. Establishing it would unlock the point.
- **Decode cost still saturates with batch**, measured on the W4A16 op per
  layer: T=1 67.3 µs, T=64 986.9, T=128 1151.4, T=256 1191.9, T=1024
  1307.3 — +13.5% for 8× the tokens above T=128. The expert-weight read is
  a fixed per-step cost once the batch touches all 128 experts.
- **Per-window KV pools are not the win** and that thread is retired: the
  pool is sized for full attention (99.27 GiB, 1,445,664 tokens = 705
  requests at ISL+OSL 2048) but capacity never binds at con≤256 (36.3%
  used), the per-layer window already limits what the FMHA kernel reads,
  and stock trtllm sizes the same single pool on this checkpoint.
- **The accuracy cost of W4A8 (−1.44 points) is the price of the frontier
  above.** A deployment that needs the last point of gsm8k accuracy should
  run the W4A16 forward (`baseline`, commit `c5e6d53`) and accept a third
  of the throughput.
