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
| `qwen3-for-causal-lm-bf16-target-v1` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` / `qwen3` | Target model | 1 | `trtllm-qwen3-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused QK-norm/RoPE, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |
| `mistral-for-causal-lm-bf16-target-v1` | `MistralForCausalLM` | `MistralForCausalLM` / `mistral` | Target model | 1 | `trtllm-mistral-dense-target-layout-v1` | Single-node dense BF16, unquantized weights and KV cache, TRTLLM attention, default fused RoPE, no sliding window, untied embeddings, TP=1 or 2, PP/CP=1, no LoRA, sparse attention, attention DP, speculative mode, or separately loaded draft model |

The registry matches the exact root class, the architecture/model type captured
from the resolved config before model construction, and any runtime constraints
declared by the profile. An unregistered subclass, config alias, or undeclared
runtime variant does not inherit support. It falls back to the standard Hugging
Face checkpoint path before any P2P transfer starts.

The Qwen2 identity includes Qwen2 and Qwen2.5 dense checkpoints that resolve to
the exact `Qwen2ForCausalLM` / `qwen2` pair and satisfy the constraints above.
Other Qwen roots and variants, including checkpoints with tied embeddings, do
not match this profile.

The Mistral identity covers dense Hugging Face-format checkpoints that resolve
to the exact `MistralForCausalLM` / `mistral` pair and whose realized attention
runs without a sliding window on every layer; Mistral-7B-Instruct-v0.3 is the
qualified canary. Checkpoints that enable `sliding_window`, including
Ministral-style `layer_types` mixes, YaRN scaling, or tied embeddings do not
match this profile. The native `mistral` checkpoint format (`params.json` with
`consolidated.safetensors`) is a separate `checkpoint_format` that rewrites the
model type to `mistral_common`; it cannot be combined with the MX loading path.
The unregistered Llama-based `MistralForCausalLM` class in `modeling_llama.py`
shares components with the qualified root but is not qualified.

TensorRT LLM applies two independent compatibility gates:

- The qualification profile records that a model/config/lifecycle combination
  has passed full-load versus staged-load equivalence testing.
- `SourceIdentity` format version 3 binds two concrete runs to the same
  checkpoint artifact, runtime layout choices, local shard layout, and
  transform-layout ABI.

The transfer protocol version identifies the staged receiver protocol. The
transform-layout ABI identifies the meaning of the transferred tensor names,
layouts, aliases, and receiver finalization. A pre-version-3 identity, a
missing ABI, or a different ABI is rejected rather than treated as compatible.

Loads that require a separately loaded draft model also fall back to the
standard checkpoint path. Target-plus-draft post-transform transfer remains
disabled until layout state is tracked and qualified independently for each
submodel.

The Llama, Qwen2, Qwen3, and Mistral profiles are text-only and do not enable
reward-model, embedding, MoE, or vision-language roots. FP16, quantized weights
or KV cache, alternate attention backends, YaRN, tied embeddings, TP greater
than 2, PP greater than 1, CP greater than 1, LoRA, sparse attention, attention
DP, multi-node transfer, and speculative decoding require separate
qualification rows. Each profile also pins its qualified RoPE realization:
Llama, Qwen2, and Mistral require the default fused RoPE path, so unfused RoPE
requires separate qualification for them, while Qwen3 fuses RoPE into the
QK-norm kernel and therefore requires `rope_fusion=False` in the realized
configuration. The Mistral profile additionally pins the realized attention
window: every attention layer must run without a sliding window, so
sliding-window checkpoints such as Ministral fall back to the Hugging Face path
until they are qualified separately. The profiles do not constrain MoE-only
backend and mapping settings because these dense roots do not consume them.
`SourceIdentity` still requires donor and receiver configurations to match.

### Adding a Model Family

Support for another model family requires a focused qualification change:

1. Audit every post-load hook in the family and its nested modules. Move
   structural wiring to `setup_aliases()`, one-time tensor-layout changes to
   `transform_weights()`, and process-local derived state to
   `cache_derived_state()`.
2. Verify that every one-time transform is guarded by `_weights_transformed`
   and that the staged receiver can skip `transform_weights()` without
   changing aliases, derived state, tensor layout, or outputs.
3. Add an exact qualification profile only after the reusable harness in
   `tests/unittest/utils/post_transform_qualification.py` proves tensor,
   alias, transform-guard, derived-state, and deterministic output
   equivalence. Include an unregistered-root negative control.
4. Cover compatible transfer, source-identity mismatch, unsupported layout or
   protocol/ABI, no-disk staged reception, and unqualified-profile fallback.
   Keep target-plus-draft loading disabled unless that combination has its own
   mixed-layout tests.
5. Run a real ModelExpress donor/receiver test with the model configurations
   being claimed, including the supported quantization and TP/PP/EP layouts.
   Compare deterministic output token IDs with the standard Hugging Face load
   path before documenting the family as supported.

### Qualification Test

The reusable GPU harness is
`tests/integration/defs/model_express/test_model_express.py`. It launches an HF
baseline, a live MX donor, and an MX receiver on disjoint GPU sets. The receiver
uses a metadata-only view of the donor's canonical snapshot and contains no
weight shards. A positive result therefore requires direct transfer; disk
fallback cannot accidentally satisfy the test.

Run the TP=1 smoke test against an isolated ModelExpress 0.4.1 service with
NIXL enabled:

```bash
TRTLLM_MX_E2E_REQUIRED=1 \
MODEL_EXPRESS_URL=http://127.0.0.1:8001 \
LLM_MODELS_ROOT=/path/to/llm-models \
pytest -v tests/integration/defs/model_express/test_model_express.py \
  -k llama-bf16-tp1
```

Run the TP=2 rank-mapping qualification on four GPUs by selecting
`llama-bf16-tp2`. `TRTLLM_MX_LLAMA_MODEL` can override the default TinyLlama
checkpoint path. The Qwen2, Qwen3, and Mistral profile rows use
`qwen2-bf16-tp1` / `qwen2-bf16-tp2`, `qwen3-bf16-tp1` / `qwen3-bf16-tp2`, and
`mistral-bf16-tp1` / `mistral-bf16-tp2`, with optional model path overrides in
`TRTLLM_MX_QWEN2_MODEL`, `TRTLLM_MX_QWEN3_MODEL`, and `TRTLLM_MX_MISTRAL_MODEL`.
`TRTLLM_MX_E2E_REQUIRED=1` converts missing service, model, or NIXL
prerequisites from skips into failures and must be set by a CI qualification
stage. That stage must also allocate the GPUs declared by the selected test
row. `TRTLLM_MX_E2E_TIMEOUT_S` controls the 1200-second timeout used for the
baseline worker, receiver worker, and donor-readiness wait; increase it for
slow model storage or startup.

The dedicated H100 CI stages own isolated Redis and ModelExpress 0.4.1
sidecars. The two-GPU TP=1 stage is classified as multi-GPU: it runs
automatically in post-merge pipelines or when a multi-GPU file changes, while
direct pre-merge dispatch requires the `ci: full pre-merge approved` label.
Trigger it directly with:

```text
/bot run --stage-list "DGX_H100-2_GPUs-PyTorch-ModelExpress-1"
```

TP=2 is the minimum evidence for adding or changing a parallel profile. Its
four-GPU stage is intentionally on demand and does not join ordinary
multi-GPU runs:

```text
/bot run --stage-list "DGX_H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1"
```

Both stages set `TRTLLM_MX_E2E_REQUIRED=1`, so missing service, model, client,
or NIXL prerequisites fail instead of skipping. Do not add every model profile
to recurring coverage: use the harness for representative rows claimed by the
support table and keep wider matrices in scheduled qualification.

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

## Installation

The official TensorRT LLM release container includes the MX Python client. No
additional Python package installation is required in that container. MX
remains opt-in at runtime: TensorRT LLM uses the client only when the MX
checkpoint-loading path and a server URL are configured. Installing the client
does not expand the model support scope described above.

For pip installations outside the official release container, install the MX
Python client through the optional `mx` extra:

```bash
pip install "tensorrt-llm[mx]"
```

The extra accepts ModelExpress client versions `>=0.4.1,<0.6.0`. Version
`0.4.1` is the minimum client API qualified by this integration, while the
upper bound prevents resolving unqualified `0.6.0` or newer client APIs.
Deploy a compatible MX server version.
The extra can be added to an existing TensorRT LLM installation. If the MX
loading path is configured but the client cannot be imported, TensorRT LLM
fails with an actionable installation message instead of silently loading from
the Hugging Face checkpoint. Source discovery and transfer failures continue to
use the Hugging Face fallback described above.

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
  nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.4.1
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
| `mx_config.server_query_timeout_s` | `null` | Timeout for MX source discovery. When unset, TensorRT LLM uses a short fallback cap when no source exists and otherwise lets MX wait for long donor loads. |

## Notes and Limitations

- Post-transform MX reception is currently limited to the exact Llama,
  Qwen2/Qwen2.5 dense, Qwen3 dense, and Mistral dense profiles above. Other
  roots and variants that do not match the documented identity and runtime
  envelope safely fall back to Hugging Face loading until explicitly qualified.
- Mistral checkpoints served through the native `mistral` checkpoint format
  (`mistral_common` model type), sliding-window or Ministral `layer_types`
  variants, YaRN variants, Mistral3 vision-language roots, and Mistral Large 3
  are not qualified and use the Hugging Face fallback.
- The MX server and Redis lifecycle is external to TensorRT LLM. Every
  TensorRT LLM instance must be able to reach the configured MX server URL.
- The MX server coordinates source discovery but does not store model weights.
  A source TensorRT LLM process must remain running and network-reachable until
  receiver transfers finish.
- The first worker may still load weights from disk if no compatible MX source
  is already registered.
- This page describes the MX checkpoint-loading path only. GPU Memory Service
  (GMS) integration is configured separately.
