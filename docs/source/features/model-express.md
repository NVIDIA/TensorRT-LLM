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

The post-transform MX receive path currently supports one exact qualification
profile:

| Profile | Root class | Config identity | Scope | Protocol | Transform-layout ABI | Constraints |
|---------|------------|-----------------|-------|----------|----------------------|-------------|
| `llama-for-causal-lm-target-v1` | `LlamaForCausalLM` | `LlamaForCausalLM` / `llama` | Target model | 1 | `trtllm-llama-target-layout-v1` | No speculative mode or separately loaded draft model |

The registry matches the exact root class and the architecture/model type
captured from the resolved config before model construction. An unregistered
subclass or config alias does not inherit support. It falls back to the
standard Hugging Face checkpoint path before any P2P transfer starts.

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

The extra pins the ModelExpress client to version `0.4.1`, matching the client
API qualified by this integration. Deploy a compatible MX server version.
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

- Post-transform MX reception is currently limited to the Llama model family.
  Other model families safely fall back to Hugging Face loading until they are
  explicitly qualified and added as exact capability profiles.
- The MX server and Redis lifecycle is external to TensorRT LLM. Every
  TensorRT LLM instance must be able to reach the configured MX server URL.
- The MX server coordinates source discovery but does not store model weights.
  A source TensorRT LLM process must remain running and network-reachable until
  receiver transfers finish.
- The first worker may still load weights from disk if no compatible MX source
  is already registered.
- This page describes the MX checkpoint-loading path only. GPU Memory Service
  (GMS) integration is configured separately.
