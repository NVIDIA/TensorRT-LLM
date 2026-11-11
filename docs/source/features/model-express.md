<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress (MX) Checkpoint Loading

The MX checkpoint-loading integration is intended to reduce repeated disk
reads when multiple TensorRT LLM workers load the same model. A worker that
loads from disk can publish its weights as an MX source, and later workers can
receive those weights directly through MX.

TensorRT LLM can use ModelExpress (MX) as a checkpoint-loading path for
PyTorch backend deployments. `checkpoint_format="MX"` selects this loading
path; it does not identify an MX-specific on-disk checkpoint format, and no
checkpoint conversion is required. TensorRT LLM attempts to fetch compatible
weights from another running TensorRT LLM instance through the MX server. If
no compatible source is available, or if MX transfer fails, loading falls back
to the provided Hugging Face checkpoint.

## Current Support Scope

The post-transform MX receive path currently supports these exact qualification
profiles:

| Profile | Root class | Config identity | Scope | Protocol | Transform-layout ABI | Constraints |
|---------|------------|-----------------|-------|----------|----------------------|-------------|
| `llama-for-causal-lm-target-v1` | `LlamaForCausalLM` | `LlamaForCausalLM` / `llama` | Target model | 1 | `trtllm-llama-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen2-for-causal-lm-bf16-target-v1` | `Qwen2ForCausalLM` | `Qwen2ForCausalLM` / `qwen2` | Target model | 1 | `trtllm-qwen2-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen3-for-causal-lm-fp8-block-sm90-target-v1` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` / `qwen3` | Target model | 1 | `trtllm-qwen3-dense-fp8-block128-fp32scale-target-layout-v1` | Single-node dense BF16, FP8 block-scale (128x128) weights and unquantized KV cache, SM90 only, TRTLLM attention, default fused QK-norm/RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen3-for-causal-lm-bf16-target-v1` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` / `qwen3` | Target model | 1 | `trtllm-qwen3-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused QK-norm/RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen3-for-causal-lm-bf16-tied-target-v1` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` / `qwen3` | Target model | 1 | `trtllm-qwen3-dense-tied-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused QK-norm/RoPE, tied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `mistral-for-causal-lm-bf16-target-v1` | `MistralForCausalLM` | `MistralForCausalLM` / `mistral` | Target model | 1 | `trtllm-mistral-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused RoPE, no sliding window, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `phi3-for-causal-lm-bf16-target-v1` | `Phi3ForCausalLM` | `Phi3ForCausalLM` / `phi3` | Target model | 1 | `trtllm-phi3-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen3_5-for-conditional-generation-bf16-complete-v1` | `Qwen3_5VLModel` | `Qwen3_5ForConditionalGeneration` / `qwen3_5` | Complete model | 1 | `trtllm-qwen3_5-dense-vl-complete-layout-v1` | Single-node dense BF16 vision-language model with its vision encoder, unquantized weights and KV cache, TRTLLM attention, fused QK-norm/M-RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `qwen3-moe-for-causal-lm-bf16-cutlass-target-v1` | `Qwen3MoeForCausalLM` | `Qwen3MoeForCausalLM` / `qwen3_moe` | Target model | 1 | `trtllm-qwen3-moe-cutlass-target-layout-v1` | Single-node MoE BF16, unquantized weights and KV cache, TRTLLM attention, CUTLASS MoE backend, default fused QK-norm/RoPE, untied embeddings, TP=1 or 2, MoE TP=1 or 2 and MoE EP=1 or 2 with MoE TP x MoE EP = TP, MoE cluster=1, PP/CP=1, no LoRA, sparse attention, attention DP, MoE load balancer, DWDP, speculative mode, or separately loaded draft model |
| `qwen3_5-moe-for-conditional-generation-bf16-cutlass-complete-v1` | `Qwen3_5MoeVLModel` | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` | Complete model | 1 | `trtllm-qwen3_5-moe-vl-cutlass-complete-layout-v1` | Single-node MoE BF16 vision-language model with its vision encoder, unquantized weights and KV cache, TRTLLM attention, CUTLASS MoE backend, fused QK-norm/M-RoPE, untied embeddings, TP=1 or 2, MoE TP=1 or 2 and MoE EP=1 or 2 with MoE TP x MoE EP = TP, MoE cluster=1, PP/CP=1, no LoRA, sparse attention, attention DP, MoE load balancer, DWDP, speculative mode, or separately loaded draft model |

The registry matches the exact root class, the architecture/model type captured
from the resolved config before model construction, the transfer scope that
`ModelLoader` derives from the constructed root, and any runtime constraints
declared by the profile. Profiles whose post-transform layout depends on the GPU
architecture also constrain the SM version of the device that runs the load. An
unregistered subclass, config alias, or undeclared runtime variant does not
inherit support. It falls back to the standard Hugging Face checkpoint path
before any P2P transfer starts, and a donor that does not qualify does not
publish its weights.

The Scope column names the part of the model that one transfer carries. A
text-only root transfers the target model. A multimodal root transfers the
complete model when it owns its multimodal encoder, which is the aggregated
serving case. With `disable_mm_encoder`, or in multimodal encoder/prefill
disaggregation (`TLLM_MULTIMODAL_DISAGGREGATED=1`), the root holds only its
language model; that scope is not qualified, so those workers fall back.
Encoder-only workers (`mm_encoder_only`) build the vision-encoder root, which is
not registered, and fall back too.

The Qwen2 identity includes Qwen2 and Qwen2.5 dense checkpoints that resolve to
the exact `Qwen2ForCausalLM` / `qwen2` pair and satisfy the constraints above.
Other Qwen roots and variants, including checkpoints with tied embeddings, do
not match this profile.

The BF16 Qwen3 identity has two sibling profiles that differ only in
`tie_word_embeddings`. Untied checkpoints match
`qwen3-for-causal-lm-bf16-target-v1`; Qwen3-8B is the untied canary. Tied
checkpoints match `qwen3-for-causal-lm-bf16-tied-target-v1`; Qwen3-4B is the
tied canary, and Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B-Instruct-2507 are also
tied. A tied model makes `lm_head.weight` an alias of
`model.embed_tokens.weight` at construction on both sides, so MX registers and
transfers the shared table once, as `model.embed_tokens.weight`, and the
`lm_head.weight` tensor that some tied checkpoints also store is never loaded.
Because the two profiles transfer different tensor sets, the tied profile has
its own transform-layout ABI, and a tied source never pairs with an untied
receiver, even under the `ENFORCE` identity policy.
The constructed model must realize the tie that the checkpoint declares: when
the `tie_word_embeddings` flag and the constructed `lm_head` and `embed_tokens`
weights disagree, no profile matches.

The Mistral identity covers dense Hugging Face-format checkpoints that resolve
to the exact `MistralForCausalLM` / `mistral` pair and whose realized attention
runs without a sliding window on every layer; Mistral-7B-Instruct-v0.3 is the
qualified canary. Checkpoints that enable `sliding_window`, including
Ministral-style `layer_types` mixes, do not match this profile. YaRN scaling
and tied embeddings are separate constraints that also exclude a checkpoint.
The native `mistral` checkpoint format (`params.json` with
`consolidated.safetensors`) is a separate `checkpoint_format` that rewrites the
model type to `mistral_common`; it cannot be combined with the MX loading path.
The unregistered Llama-based `MistralForCausalLM` class in `modeling_llama.py`
shares components with the qualified root but is not qualified.

The Phi3 identity covers dense Hugging Face-format checkpoints that resolve to
the exact `Phi3ForCausalLM` / `phi3` pair; Phi-4 is the qualified canary.
Phi-4-reasoning-plus, Phi-3-mini-4k-instruct, and Phi-3-medium-4k-instruct
declare the same identity with BF16 weights, default RoPE, and untied
embeddings, so they match the profile under the same runtime constraints. The
Phi-3 4k configurations set `sliding_window`, but the Phi3 root does not apply
it: every Phi3 checkpoint runs full attention in TensorRT LLM, so this profile
does not constrain the sliding window. The 128k-context Phi-3 checkpoints and
Phi-3.5-mini-instruct use LongRoPE scaling, and Phi-4-mini-instruct also ties
its embeddings, so these checkpoints do not match the profile. Phi-3-small
(`Phi3SmallForCausalLM`), the Phi-3 and Phi-3.5 vision models
(`Phi3VForCausalLM`), Phi-3.5-MoE (`PhiMoEForCausalLM`), and
Phi-4-multimodal (`Phi4MMForCausalLM`, including its quantized variants) use
other roots. Phi4MM builds a Phi3 language model internally, but qualification
matches the outer root, so these checkpoints fall back as well.

Phi3 has one known limitation. `Phi3Attention.__init__` in
`tensorrt_llm/_torch/models/modeling_phi3.py` writes the runtime `max_seq_len`
into the shared pretrained configuration for LongRoPE, and `SourceIdentity`
hashes that configuration. A donor and a receiver with different `max_seq_len`
values therefore get different model fingerprints: the receiver logs a
`SourceIdentity mismatch` warning and falls back to non-shared weight loading,
even though no transferred weight of a default-RoPE checkpoint depends on the
value. Until the model code stops recording it, run both sides with the same
`max_seq_len`.

The Qwen3.5 dense vision-language identity covers Hugging Face-format
checkpoints that resolve to the exact `Qwen3_5ForConditionalGeneration` /
`qwen3_5` pair, served with the vision encoder in the same instance.
Qwen3.5-9B is the qualified canary and the only checkpoint that the
donor/receiver rows cover. Checkpoints with tied embeddings, such as
Qwen3.5-0.8B, 2B, and 4B, quantized checkpoints such as the FP8 and NVFP4
variants, the Qwen3.5 MoE vision-language root (which has its own profile
below), the text-only `Qwen3_5ForCausalLM` root, and the Qwen3-VL roots do not
match this profile.
The root sizes its M-RoPE buffers by `max_num_tokens` (and the PP size), and
`SourceIdentity` hashes every realized buffer shape, so a donor and a receiver
must use the same `max_num_tokens` to share weights. The vision configuration
is bound by the checkpoint artifact identity and the realized vision parameter
shapes rather than by the model fingerprint, which hashes the text
configuration.

The Qwen3.5 MoE vision-language identity covers Hugging Face-format checkpoints
that resolve to the exact `Qwen3_5MoeForConditionalGeneration` /
`qwen3_5_moe` pair, served with the vision encoder in the same instance, and
combines the dense Qwen3.5 vision-language constraints with the Qwen3 MoE
backend and partition pins: every decoder layer holds a replicated router,
CUTLASS routed experts, a fused shared expert, and a replicated shared-expert
gate. Qwen3.5-35B-A3B is the qualified canary; Qwen3.6-35B-A3B declares the same
identity and layout. FP8 and GPTQ variants fall back. Qwen3.5-122B-A10B shares
the identity and the per-layer tensor layout, so it qualifies wherever TP=2 holds
its roughly 234 GiB of BF16 weights, which takes GPUs with more than about 120 GiB
of memory each, such as B200 or B300; the Qwen3.5-35B-A3B rows exercise that
layout and no row loads the 122B. Qwen3.5-397B-A17B does not fit at TP=2 or
below on any current GPU, so it falls back at the TP constraint. The text-only
`Qwen3_5MoeForCausalLM` root is a different identity and falls back. The profile
pins the CUTLASS MoE backend, which `moe_config.backend: AUTO` selects for BF16
weights, and admits MoE TP=2 and MoE EP=2 at TP=2 like the Qwen3 MoE profile.
Qwen3.5-35B-A3B holds about 66 GiB of BF16 weights after loading, so a TP=1
donor or receiver needs an 80 GB GPU. The stacked expert tensors of the checkpoint are
unfused into the CUTLASS layout while loading; the receiver never loads, and the
loader-only `weight_loading_mode` of the expert backend that this changes does
not affect forward.

The Qwen3 MoE identity covers Hugging Face-format checkpoints that resolve to
the exact `Qwen3MoeForCausalLM` / `qwen3_moe` pair; Qwen3-30B-A3B is the
qualified canary. Its Instruct-2507 and Thinking-2507 releases declare the same
identity with BF16 weights, default RoPE, and untied embeddings. The
Qwen3-235B-A22B checkpoints share the identity and the per-layer tensor layout,
so they qualify too wherever TP=2 holds their roughly 470 GB of BF16 weights,
which takes GPUs with more than about 240 GB of memory each, such as B300 or
GB300. The Qwen3-30B-A3B rows exercise that layout; no row loads Qwen3-235B-A22B
itself. Quantized checkpoints, such as the FP8 and NVFP4 variants, fail the
quantization constraints. Qwen3-VL-MoE checkpoints use another root:
`Qwen3VLMoeForConditionalGeneration` builds a `Qwen3MoeForCausalLM` internally,
but qualification matches the outer root, so these checkpoints fall back. The
profile pins the CUTLASS MoE backend, which `moe_config.backend: AUTO` selects
for these checkpoints; an explicitly requested other MoE backend falls back. At
TP=2 the profile admits MoE TP=2 (the default) and MoE EP=2
(`moe_expert_parallel_size: 2`), so MoE TP x MoE EP always equals TP. MoE
cluster parallelism, the MoE load balancer, and DWDP fall back.

The Qwen3 FP8 block-scale identity covers `Qwen3ForCausalLM` / `qwen3`
checkpoints whose quantization resolves to `FP8_BLOCK_SCALES`, whose Hugging
Face quantization config declares `weight_block_size: [128, 128]`, and that run
with an unquantized KV cache; Qwen3-8B-FP8 (`quant_method: fp8`, dynamic
activations) is the qualified canary. Sources that resolve to the same algorithm
without declaring that block size, such as static-activation FP8 checkpoints or
ModelOpt and compressed-tensors block FP8 exports, do not match this profile,
and neither do per-tensor FP8 checkpoints (ModelOpt `FP8`), an FP8 KV cache, or
any GPU other than SM90. On SM100-family GPUs (by default) and SM120/121 GPUs,
`transform_weights()` resmooths the weights to power-of-two scales and repacks
the scales for DeepGEMM, and SM107 can use a CuTe DSL scale layout, so the SM90
tensors are not valid there. Such loads fall back to the Hugging Face path.
`SourceIdentity` does not record the SM version, so this profile constraint is
what keeps the SM90 layout on SM90.

TensorRT LLM applies two independent compatibility gates:

- The qualification profile records that a model/config/lifecycle combination
  has passed full-load versus staged-load equivalence testing.
- `SourceIdentity` format version 4 binds two concrete runs to the same
  checkpoint artifact, runtime layout choices, local shard layout, and
  transform-layout ABI. Quantization exclusion patterns compare as a set, so
  their order in the checkpoint's config does not matter.

The transfer protocol version identifies the staged receiver protocol. The
transform-layout ABI identifies the meaning of the transferred tensor names,
layouts, aliases, and receiver finalization. An identity of another format
version, a missing ABI, or a different ABI is rejected rather than treated as
compatible.

Loads that require a separately loaded draft model also fall back to the
standard checkpoint path. Target-plus-draft post-transform transfer remains
disabled until layout state is tracked and qualified independently for each
submodel.

The Llama, Qwen2, Qwen3, Mistral, Phi3, and Qwen3 MoE profiles are text-only and
do not enable reward-model, embedding, or vision-language roots; Qwen3 MoE and
the Qwen3.5 MoE vision-language root are the only qualified MoE roots. FP16,
quantized weight formats other than SM90 FP8 block scales on Qwen3, a quantized KV cache, alternate attention backends, YaRN
or LongRoPE scaling, tied embeddings, TP greater than 2, PP greater than 1, CP
greater than 1, LoRA, sparse attention, attention DP, multi-node transfer, and
speculative decoding require separate qualification rows. The Qwen3 FP8
block-scale profile is pinned to SM90 because its realized weight and scale
layout differs on other GPU architectures. Each profile also pins its qualified
RoPE realization: Llama, Qwen2, Mistral, and Phi3 require the default fused RoPE
path, so unfused RoPE requires separate qualification for them, while Qwen3 and
Qwen3 MoE fuse RoPE into the QK-norm kernel and therefore require
`rope_fusion=False` in the realized configuration. The Mistral profile
additionally pins the realized attention window: every attention layer must run
without a sliding window, so sliding-window checkpoints such as Ministral fall
back to the Hugging Face path until they are qualified separately. The dense
profiles do not constrain MoE-only backend and mapping settings because dense
roots do not consume them; the Qwen3 MoE and Qwen3.5 MoE profiles pin them as
the table above lists. `SourceIdentity` still requires donor and receiver
configurations to match.

The Qwen3.5 dense vision-language profile carries the vision encoder together
with the language model and realizes interleaved M-RoPE (`rope_type=mrope`),
applied by the fused QK-norm/RoPE kernel (`rope_fusion=False`). A YaRN
`rope_parameters` entry also realizes `mrope`, so the RoPE constraint alone does
not exclude it; the transferred tensor layout is the same, and `SourceIdentity`
still requires donor and receiver configurations to match.

### Adding a Model Family

Support for another model family requires a focused qualification change:

1. Audit every post-load hook in the family and its nested modules. Move
   structural wiring to `setup_aliases()`, one-time tensor-layout changes to
   `transform_weights()`, and process-local derived state to
   `cache_derived_state()`.
2. Verify that every one-time transform is guarded by `_weights_transformed`
   and that the staged receiver can skip `transform_weights()` without
   changing aliases, derived state, tensor layout, or outputs. Declare the guard
   before any weight load: staged receivers mark only modules that already carry
   it, and a later transform walk would run any module left unmarked on already
   transformed bytes. For quantized rows, audit each quantization method's
   `transform_weights()` per GPU architecture and constrain `sm_versions` to the
   architectures whose realized layout the row's ABI describes.
3. Add an exact qualification profile only after the reusable harness in
   `tests/unittest/utils/post_transform_qualification.py` proves tensor,
   alias, transform-guard, derived-state, and deterministic output
   equivalence. Include an unregistered-root negative control. Set the
   case's `prepare_producer` so the producer runs the family's real
   `load_weights()`; without it the harness cannot see state that only the
   load path or the weight mapper sets, which a staged receiver never
   computes.
4. Cover compatible transfer, source-identity mismatch, unsupported layout or
   protocol/ABI, no-disk staged reception, and unqualified-profile fallback.
   Keep target-plus-draft loading disabled unless that combination has its own
   mixed-layout tests.
5. Run a real ModelExpress donor/receiver test with the model configurations
   being claimed, including the supported quantization and TP/PP/EP layouts.
   Compare deterministic output token IDs with the standard Hugging Face load
   path, and require the per-rank weight manifests to match (see the
   qualification test below), before documenting the family as supported.
6. For a multimodal root, audit the encoder's hooks and load path as well, and
   register the scope that was qualified. `ModelLoader` derives the scope from
   parameter ownership: the complete model when the root owns parameters
   outside its language model, and the language model otherwise.
7. For a MoE root, pin the MoE backend in the profile, because routed-expert
   layouts differ per MoE implementation and each needs its own
   transform-layout ABI. The pinned backend and quantization must name exactly
   one implementation. Also pin every MoE partition dimension (`moe_tp_sizes`,
   `moe_ep_sizes`, and `moe_cluster_sizes`); the qualification harness rejects
   a profile that qualifies a model with routed experts but leaves any of them
   unconstrained. The size pins cannot exclude the MoE load balancer (EPLB) or
   DWDP: EPLB changes which experts a rank stores without changing the MoE TP
   or EP size, and DWDP forces the MoE TP and EP sizes to 1, which a TP=1 or
   TP=2 profile admits, while each rank stores its own expert range.
   `ModelLoader` reports both as features instead. Keep them unsupported
   unless they are qualified separately, and add one donor/receiver row per
   admitted MoE partition.

### Qualification Test

The reusable GPU harness is
`tests/integration/defs/model_express/test_model_express.py`. It launches an HF
baseline, a live MX donor, and an MX receiver on disjoint GPU sets. The receiver
uses a metadata-only view of the donor's canonical snapshot and contains no
weight shards. A positive result therefore requires direct transfer; disk
fallback cannot accidentally satisfy the test.

Each role generates eight fixed prompts for exactly 32 greedy tokens
(`end_id=-1`, so no sequence stops early), and the test requires exact
token-ID equality of the donor and the receiver against the HF baseline.
Vision-language rows use an image probe instead of token-ID prompts: each of
`seashore.png` and `inpaint.png` from `LLM_MODELS_ROOT/multimodals/test_data`
is asked the same four questions, so every role also runs the vision encoder
and M-RoPE positions, with the same 32-token greedy equality and the same
engine limits for all roles.

**Weight manifests.** Every rank also writes a SHA-256 manifest of all
registered parameters and buffers
(`tensorrt_llm/_torch/weight_sharing/weight_manifest.py`) when
`MX_WEIGHT_MANIFEST_DIR` is set; the harness sets it for all three roles and
production loads never write one. Two manifest families are compared byte for
byte:

- `manifest.final.<role>.rank<N>.json` is written at the end of
  `ModelLoader.load`, after every post-load hook and MoE load-balancer
  finalization. Baseline, donor, and receiver must be pairwise identical:
  same tensor names, dtypes, shapes, strides, storage offsets, digests,
  skipped-tensor sets, and storage-alias partitions.
- `manifest.transfer.<role>.rank<N>.json` is written inside the MX checkpoint
  loader at the donor's publish point and at the receiver's P2P success point
  (MX roles only). Donor and receiver parameters must be identical at this
  boundary; derived buffers are enforced at the final tier because the
  receiver's `cache_derived_state()` runs after the transfer.

A row may list `final_manifest_exempt_patterns` on its `MxE2ECase` to exempt
named tensors from the final-tier digest comparison. That is never a numeric
tolerance and never applies to the transfer tier, and every pattern needs a
code comment explaining why; the current rows exempt nothing.
Manifests carry a `manifest_format_version`, and manifests of different
versions never compare.

Run the TP=1 smoke test against an isolated ModelExpress 0.5.1 service with
NIXL enabled:

```bash
TRTLLM_MX_E2E_REQUIRED=1 \
MODEL_EXPRESS_URL=http://127.0.0.1:8001 \
LLM_MODELS_ROOT=/path/to/llm-models \
pytest -v tests/integration/defs/model_express/test_model_express.py \
  -k llama-bf16-tp1 --output-dir /path/to/artifacts
```

With `--output-dir` (always set in CI), the worker payloads, worker logs,
transfer logs, weight manifests, and `timing.json` are copied to
`model_express/<case-id>/` under that directory, so they are part of the stage
results archive even when the test fails. The test also prints one
`MX E2E timing` line per role and rank with the load, generation, and manifest
durations.

Run the TP=2 rank-mapping qualification on four GPUs by selecting
`llama-bf16-tp2`. `TRTLLM_MX_LLAMA_MODEL` can override the default TinyLlama
checkpoint path. The Qwen2, Qwen3, Mistral, Phi3, and Qwen3.5 profile rows use
`qwen2-bf16-tp1` / `qwen2-bf16-tp2`, `qwen3-bf16-tp1` / `qwen3-bf16-tp2`,
`mistral-bf16-tp1` / `mistral-bf16-tp2`, `phi3-bf16-tp1` /
`phi3-bf16-tp2`, and `qwen3_5-bf16-tp1` / `qwen3_5-bf16-tp2`, with optional
model path overrides in `TRTLLM_MX_QWEN2_MODEL`, `TRTLLM_MX_QWEN3_MODEL`,
`TRTLLM_MX_MISTRAL_MODEL`, `TRTLLM_MX_PHI3_MODEL`, and
`TRTLLM_MX_QWEN3_5_MODEL`.
The Qwen3 MoE rows use `qwen3-moe-bf16-tp1`, `qwen3-moe-bf16-tp2` (MoE TP=2),
and `qwen3-moe-bf16-tp2-ep2` (MoE EP=2), with `TRTLLM_MX_QWEN3_MOE_MODEL`
overriding the default `Qwen3/Qwen3-30B-A3B` checkpoint. All three roles of
these rows set `moe_config.disable_finalize_fusion=True`: the CUTLASS
FC2+finalize fusion is not deterministic for top-k greater than 2, and the test
requires exact token equality. MoE rows also require every rank's routed-expert
tensor shapes in the final manifests to match the row's MoE EP x MoE TP
partition, so the comparisons cannot pass on a partition that all roles share
but the row did not request.
The Qwen3.5 MoE rows use `qwen3_5-moe-bf16-tp1`, `qwen3_5-moe-bf16-tp2`, and
`qwen3_5-moe-bf16-tp2-ep2`, with `TRTLLM_MX_QWEN3_5_MOE_MODEL` overriding the
default `Qwen3.5-35B-A3B` checkpoint. They probe with images and, like the Qwen3
MoE rows, disable the finalize fusion. The TP=1 row holds the roughly 66 GiB of
weights after loading (about 71 GiB at the memory-profiling peak) on one GPU and
skips below 80 GB of device memory.
The Qwen3 FP8 block-scale profile rows use `qwen3-fp8-block-tp1` /
`qwen3-fp8-block-tp2` with Qwen3-8B-FP8, overridable through
`TRTLLM_MX_QWEN3_FP8_MODEL`; they run only on SM90 GPUs, which the H100 CI
stages provide, and skip elsewhere.
`TRTLLM_MX_E2E_REQUIRED=1` converts missing service, model, or NIXL
prerequisites from skips into failures and must be set by a CI qualification
stage. That stage must also allocate the GPUs declared by the selected test
row. `TRTLLM_MX_E2E_TIMEOUT_S` controls the 1200-second timeout used for the
baseline worker, receiver worker, and donor-readiness wait; increase it for
slow model storage or startup. The tied Qwen3 profile rows
`qwen3-tied-bf16-tp1` / `qwen3-tied-bf16-tp2` load the tied Qwen3-4B
checkpoint, with an optional model path override in
`TRTLLM_MX_QWEN3_TIED_MODEL`.

The dedicated H100 CI stages own isolated Redis and ModelExpress 0.5.1
sidecars. Coverage is tiered so that pull requests pay for one lightweight
end-to-end canary while every qualified family still runs on each main commit:

| Stage | When it runs | Rows |
| --- | --- | --- |
| `DGX_H100-2_GPUs-PyTorch-ModelExpress-1` | Pre-merge and post-merge | `llama-bf16-tp1` and `test_mx_source_identity_gate.py` |
| `DGX_H100-2_GPUs-PyTorch-ModelExpress-Post-Merge-1` | Post-merge only | The other TP=1 rows and the accuracy canaries below |
| `DGX_H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1` | On demand only | Every TP=2 row |

The two-GPU stages are classified as multi-GPU, so running either of them on a
pull request requires the `ci: full pre-merge approved` label; the pre-merge
stage is also selected automatically when a multi-GPU file changes. Trigger
them directly with:

```text
/bot run --stage-list "DGX_H100-2_GPUs-PyTorch-ModelExpress-1"
/bot run --stage-list "DGX_H100-2_GPUs-PyTorch-ModelExpress-Post-Merge-1"
```

TP=2 is the minimum evidence for adding or changing a parallel profile. Its
four-GPU stage is intentionally on demand and does not join ordinary
multi-GPU runs:

```text
/bot run --stage-list "DGX_H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1"
```

All three stages set `TRTLLM_MX_E2E_REQUIRED=1`, so missing service, model,
client, or NIXL prerequisites fail instead of skipping. Do not grow pre-merge
coverage with each new family: its TP=1 row belongs in the `post_merge` block of
`tests/integration/test_lists/test-db/l0_model_express.yml` and its TP=2 row
in the on-demand stage, while the pre-merge block keeps a single
representative row.

### Accuracy Canaries (Post-Merge)

`tests/integration/defs/model_express/test_model_express_accuracy.py` runs
reference-backed accuracy canaries on an MX receiver. The donor publishes exactly as in the smoke test and never evaluates; the receiver starts
from the metadata-only snapshot and, right after load, confirms that every rank
wrote its transfer manifest at the MX P2P success boundary. If any rank did
not, the receiver exits with status 3 before spending any evaluation time.
Otherwise it evaluates the task with `tensorrt_llm.evaluate` inside its own
subprocess and writes the score to JSON. The pytest process never constructs an `LLM`: it loads the
accuracy reference YAMLs, asserts the same hypothesis-testing threshold as
`tests/integration/defs/accuracy/`, and additionally requires the transfer
evidence and the donor/receiver weight manifests to match. There is no paired
HF baseline evaluation because the reference value is that baseline.

Current rows (TP=1, BF16; `references/*.yaml` holds the expected values):

| Test ID | Model | Task | Model path override |
| --- | --- | --- | --- |
| `qwen3-8b-gsm8k-tp1` | `Qwen3/Qwen3-8B` | GSM8K | `TRTLLM_MX_QWEN3_MODEL` |

A canary needs a bare BF16 reference for its task that was measured on the
PyTorch backend. The Qwen3 GSM8K entry is the one gated by
`tests/integration/defs/accuracy/test_disaggregated_serving.py::TestQwen3_8B`,
so re-run that test to re-measure it if the value drifts. No in-envelope Llama,
Qwen2, Mistral, Phi3, Qwen3.5, or tied Qwen3 checkpoint has such a reference
today, so those profiles are covered by their smoke rows until one is added. The
bare Qwen3-30B-A3B GSM8K and MMLU entries are shared with its quantized variants
and have no recorded PyTorch-backend measurement, so Qwen3 MoE is covered by its
smoke rows too, and the Qwen3 FP8 block-scale profile has no accuracy canary
either. A Qwen3.5 canary must also size its donor exactly like its receiver,
because `SourceIdentity` binds `max_num_tokens` for that root.

The rows are registered as `stage: post_merge` entries in
`tests/integration/test_lists/test-db/l0_model_express.yml`, so they run in
`DGX_H100-2_GPUs-PyTorch-ModelExpress-Post-Merge-1` on every main commit and
are never selected automatically in pre-merge pipelines. Trigger the stage on a
pull request, which requires the `ci: full pre-merge approved` label, with:

```text
/bot run --stage-list "DGX_H100-2_GPUs-PyTorch-ModelExpress-Post-Merge-1"
```

GSM8K needs the `lm_eval` package from `requirements-dev.txt`; the test checks
for it and, under `TRTLLM_MX_E2E_REQUIRED=1`, fails instead of skipping when it
is absent. The receiver gets its own `TRTLLM_MX_ACCURACY_TIMEOUT_S` budget
(default 3600 seconds) for load, transfer, and the whole evaluation, while
`TRTLLM_MX_E2E_TIMEOUT_S` still bounds the donor-readiness wait. The reference
guard `test_mx_accuracy_cases_have_references` needs no GPU and also runs in the
pre-merge CPU stage, so a pruned reference fails on the pull request that prunes
it. Each run records the donor and receiver load times, the evaluation
time, the score, and the threshold as junit properties and as
`model_express_accuracy/<test-id>.json` under `--output-dir`; load times are
observed only, not gated. To run one canary locally:

```bash
TRTLLM_MX_E2E_REQUIRED=1 \
MODEL_EXPRESS_URL=http://127.0.0.1:8001 \
LLM_MODELS_ROOT=/path/to/llm-models \
pytest -v tests/integration/defs/model_express/test_model_express_accuracy.py \
  -k qwen3-8b-gsm8k-tp1 --output-dir /path/to/artifacts
```

Adding a canary is one `MxAccuracyCase` row (the model must be inside the
family's qualified runtime envelope and have a bare BF16 reference entry for
the task, measured on the PyTorch backend with the task's default evaluator
settings), one line in the `post_merge` block of `l0_model_express.yml`, and a
row in the table above.

### Transform-Layout ABI Rules

An existing transform-layout ABI ID is immutable. Introduce a new ID when a
change affects any transferred tensor name, shape, dtype, packing, sharding,
alias relationship, one-shot transform result, or receiver-side
`setup_aliases()`/`cache_derived_state()` interpretation. Keep the existing ID
for implementation-only changes that preserve all of those observable
semantics.

When adding an ABI ID:

1. Give the qualified profile the new ID and propagate it through
   `SourceIdentity` and MX source metadata.
2. Add matching, missing, and mismatched producer/receiver compatibility
   tests. ABI mismatches remain incompatible even under the `ENFORCE` identity
   policy.
3. Re-run the qualification harness and the real donor/receiver GPU test for
   every profile that adopts the ID.
4. Never reinterpret an already published ID. Supporting two ABIs requires an
   explicit compatibility decision and tests for each producer/receiver pair.
5. When a quantization method realizes a different tensor layout per GPU
   architecture, give each realized layout its own ID and pin the profile's
   `sm_versions` to the architectures it was qualified on. `SourceIdentity`
   does not record the SM version, so the pin is what keeps a receiver from
   accepting bytes laid out for another architecture.

Routed-expert layouts differ per MoE implementation: the CUTLASS implementation
for unquantized weights keeps the loaded layout, while the TRTLLM-Gen BF16
method shuffles and block-converts the expert weights at load. A MoE profile
therefore pins the MoE backend together with the quantization, which must name
exactly one implementation, and each implementation gets its own ABI ID. A
backend that gains a second implementation for the pinned quantization needs a
realized-implementation constraint before its profile stays exact.

## Installation

TensorRT LLM release containers that include this feature already install a
compatible MX Python client; no additional client installation is needed for
P2P transfer. For earlier TensorRT LLM releases, use a release or container
built with this feature; upgrading the MX package alone does not add the
missing TensorRT LLM integration. MX remains opt-in at runtime: TensorRT LLM
uses the client only when the MX checkpoint-loading path and a server URL are
configured. Installing the client does not expand the model support scope
described above.

For pip installations outside the official release container, install the MX
Python client through the optional `mx` extra:

```bash
pip install "tensorrt-llm[mx]"
```

The extra accepts ModelExpress client versions `>=0.5.1,<0.6.0`. Version
`0.5.1` is the minimum client release that provides the TensorRT LLM adapter,
while the upper bound prevents resolving unqualified `0.6.0` or newer client
APIs. Deploy a compatible MX server version.
The extra can be added to an existing TensorRT LLM installation. If the MX
loading path is configured but the client cannot be imported, TensorRT LLM
logs a warning and uses the Hugging Face fallback described above. Source
discovery and transfer failures use the same fallback.

## Deploy the MX Service

Deploy the MX server and its Redis metadata backend independently of
TensorRT LLM. One MX service can be shared by multiple TensorRT LLM launches,
provided every instance can reach the MX endpoint. TensorRT LLM does not start,
stop, or otherwise manage either service.

The following commands illustrate a standalone Docker deployment. Production
deployments should manage service lifecycle, persistence, networking, and
security according to their environment.

```bash
docker network create modelexpress
docker run -d --name modelexpress-redis \
  --network modelexpress \
  redis:8-alpine
docker run -d --name modelexpress-server \
  --network modelexpress \
  -p 8001:8001 \
  -e MODEL_EXPRESS_SERVER_PORT=8001 \
  -e MODEL_EXPRESS_LOG_LEVEL=info \
  -e MX_METADATA_BACKEND=redis \
  -e REDIS_URL=redis://modelexpress-redis:6379 \
  nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.5.1
```

## Configure TensorRT LLM

Select the MX checkpoint-loading path and provide the MX server URL in a
`trtllm-serve` config. The model argument remains a standard Hugging Face model
ID or checkpoint path:

```yaml
checkpoint_format: MX
mx_config:
  server_url: http://mx-server.example.com:8001
```

```bash
trtllm-serve /path/to/model --config config.yaml
```

The `MODEL_EXPRESS_URL` environment variable can also provide the server URL
when `mx_config.server_url` is not set.

Multiple TensorRT LLM launches can use the same configuration. A worker that
does not find a compatible source loads from Hugging Face storage and publishes
its weights through MX. Later compatible workers can receive those weights by
P2P transfer.

If neither `mx_config.server_url` nor `MODEL_EXPRESS_URL` is set, MX transfer is
not attempted and checkpoint loading falls back to the standard Hugging Face
path.

## Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `mx_config.server_url` | `null` | URL of the separately managed MX server. |
| `mx_config.server_query_timeout_s` | `null` | Deprecated and ignored. MX checks once for a compatible source, then falls back to native checkpoint loading. |

## Notes and Limitations

- Post-transform MX reception is currently limited to the exact Llama,
  Qwen2/Qwen2.5 dense, Qwen3 dense (BF16 or SM90 FP8 block-scale), Mistral
  dense, Phi3 dense, Qwen3.5 dense vision-language, Qwen3 MoE, and Qwen3.5 MoE
  profiles above. Other roots and variants that do not match the documented identity and
  runtime envelope safely fall back to Hugging Face loading until explicitly
  qualified.
- Mistral checkpoints served through the native `mistral` checkpoint format
  (`mistral_common` model type), sliding-window or Ministral `layer_types`
  variants, YaRN variants, Mistral3 vision-language roots, and Mistral Large 3
  are not qualified and use the Hugging Face fallback.
- Phi LongRoPE variants (128k-context Phi-3, Phi-3.5-mini, and Phi-4-mini),
  tied-embedding variants, and the Phi-3-small, vision, MoE, and Phi4MM
  multimodal roots are not qualified and use the Hugging Face fallback.
- Qwen3.5 dense and MoE vision-language checkpoints fall back when they run
  with speculative decoding (MTP or DFlash), without a local vision encoder (`disable_mm_encoder`
  or multimodal encoder/prefill disaggregation), as encoder-only workers, with
  tied embeddings, or with quantized weights.
- The MX server and Redis lifecycle is external to TensorRT LLM. Every
  TensorRT LLM instance must be able to reach the configured MX server URL.
- The MX server coordinates source discovery but does not store model weights.
  A source TensorRT LLM process must remain running and network-reachable until
  receiver transfers finish.
- The first worker may still load weights from disk if no compatible MX source
  is already registered.
- This page describes the MX checkpoint-loading path only. GPU Memory Service
  (GMS) integration is configured separately.
