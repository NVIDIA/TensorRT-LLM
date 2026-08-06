<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- Copyright (c) 2026 by FlashInfer team. -->

# Experimental Task-Scheduled Attention

`tensorrt_llm._torch.attention_backend.prims_ts` contains experimental CuTe DSL attention
kernels for NVIDIA Blackwell GPUs. Scheduling, tile selection, and split-KV
reduction are implementation details; the public interfaces expose attention
and cache semantics without tuning knobs.

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but is not yet signoff-qualified.

## Guides and public APIs

Import all entries below from `tensorrt_llm._torch.attention_backend.prims_ts`.

| Kernel | Guide | Public APIs |
| --- | --- | --- |
| FMHA context/prefill | [Task-Scheduled FMHA Context](kernels/fmha_context/README.md) | `BatchPrefillTSWrapper`, `batch_prefill`, `BatchPrefillPagedTSWrapper`, `batch_prefill_with_paged_kv_cache` |
| FMHA decode | [Task-Scheduled FMHA Decode](kernels/fmha_decode/README.md) | `BatchDecodePagedTSWrapper`, `batch_decode_with_paged_kv_cache`, `get_prims_ts_batch_decode_workspace_size`, `prims_ts_batch_decode_with_kv_cache` |
| MLA decode | [Task-Scheduled MLA Decode](kernels/mla_decode/README.md) | `BatchMLADecodePagedTSWrapper`, `batch_decode_mla_with_paged_kv_cache`, `get_prims_ts_batch_decode_mla_workspace_size`, `prims_ts_batch_decode_with_kv_cache_mla` |

The component guides define supported shapes, layouts, metadata lifetime,
output/workspace ownership, examples, limitations, and validation commands.

## Provenance

This directory is vendored from `flashinfer/attention/prims_ts` in
[FlashInfer PR #4357](https://github.com/flashinfer-ai/flashinfer/pull/4357) at commit
[`74790b32b55f6c45a4fee78007b4d2b2109497e3`](https://github.com/flashinfer-ai/flashinfer/commit/74790b32b55f6c45a4fee78007b4d2b2109497e3).
The kernel implementation is preserved. TensorRT-LLM adds its copyright,
uses its local package namespace, and replaces FlashInfer's optional API
logging and trace-template decorators with a zero-overhead local compatibility
decorator.

## Validation

### TensorRT-LLM checkout

Run the local registry, adapter, and GPU backend unit suites from the
TensorRT-LLM repository root:

```bash
pytest -q \
  tests/unittest/_torch/attention/test_fmha_registry.py \
  tests/unittest/_torch/attention/test_prims_ts_fmha.py \
  tests/unittest/_torch/attention/test_prims_ts_attention_backend.py
```

The model-level end-to-end coverage requires a Blackwell GPU and the model
weights under `LLM_MODELS_ROOT`:

```bash
LLM_MODELS_ROOT=/path/to/models pytest -q \
  tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestQwen2_7BInstruct::test_prims_ts_bfloat16 \
  tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_prims_ts_bfloat16
```

### Upstream FlashInfer checkout

The following provenance checks cover the upstream numerical, graph,
scheduler/resource, alias-safety, and public-surface contracts for the pinned
snapshot. These paths do not exist in TensorRT-LLM; run them only from the
FlashInfer checkout at the commit linked above:

```bash
pytest -q \
  tests/attention/test_attention_ts_context.py \
  tests/attention/test_attention_ts_decode.py \
  tests/attention/test_attention_ts_mask.py \
  tests/attention/test_attention_ts_mla_decode.py
```
