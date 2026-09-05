---
name: trtllm-model-onboard-multimodal
description: >
  Onboard a HuggingFace multimodal model (vision/audio/video + text) to the
  TensorRT-LLM PyTorch backend. Use when writing a new
  `tensorrt_llm/_torch/models/modeling_{vlm}.py` plus its input processor and
  weight mapper, or extending an existing VLM. Not for AutoDeploy — use
  `ad-model-onboard` for that path.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Multimodal Model Onboarding (PyTorch backend)

> **Scope.** PyTorch backend only (`tensorrt_llm/_torch/`) — the default for `LLM(..., backend="pytorch")`, `trtllm-serve`, `trtllm-bench`. **Not** for AutoDeploy (`tensorrt_llm/_torch/auto_deploy/`); use `ad-model-onboard` for that.

**Output:**
- `tensorrt_llm/_torch/models/modeling_{name}.py` — wrapper class (multimodal encoder + LLM) decorated with `@register_auto_model`, `@register_vision_encoder`, `@register_input_processor` (and `@support_multimodal_disaggregated` if EPD is supported), plus a `BaseMultimodalInputProcessor` (+ `BaseMultimodalDummyInputsBuilder`) subclass.
- `_torch/models/checkpoints/hf/{name}_weight_mapper.py` if HF prefixes need surgery.
- Per-model unit test under `tests/unittest/_torch/modeling/test_modeling_<name>.py` (subclass of `TestModelingMultimodal`); supplemental utility tests under `tests/unittest/_torch/multimodal/` if needed; an accuracy test under `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`; support-matrix entry; verified `trtllm-serve` flow.

## System map

### Aggregated path (default)

```text
[1] API event loop  (server-side, async)
    The chat handler wraps each image/video/audio URL part as an async_load_*
    coroutine (not yet awaited). apply_chat_template builds the text prompt.
    asyncio.gather then decodes all media for one request in parallel.

[2] Input pipeline  (asyncio.to_thread, off the event loop)
    BaseMultimodalInputProcessor.__call__ dispatches by input shape: a text
    prompt goes to the per-model call_with_text_prompt; prompt_token_ids +
    mm_data goes to the base-class call_with_token_ids fast path (or is
    detokenized back to call_with_text_prompt when the model opts out). The
    per-model HF processing lives in call_with_text_prompt:
       HF AutoProcessor → pixel_values + token_ids
       mm-token layout (positions / lengths / special_token_offsets)
       (mRoPE) mrope_position_ids + deltas computed on CPU
       _postprocess: HF mm token ids → tllm_multimodal_token_id (OOV sentinel)
    The framework wrapper around your processor computes blake3 content hashes
    for KV-cache reuse.
    MultimodalParams.to_handle("multimodal_data") at the end → each tensor in
    multimodal_data is replaced by a small dict pointing at its CUDA-IPC / shm
    handle, so the broadcast in [3] carries pointers, not megabytes of pixels.

[3] Worker fan-out  (TP / PP / CP)
    Each worker rebuilds local tensor views via to_tensor("multimodal_data").
    multimodal_input (hashes / positions / lengths) is forwarded to the C++
    executor to drive KV-cache hash matching.

[4] Per-iteration staging  (model engine)
    Context:    build MultimodalRuntimeData (positions / lengths / chunk bounds).
                For item-scheduled models, reserve space in the model's one
                TensorLRUCache. Reuse ready outputs and encode each missing item only
                once, within the encoder item / token / output-byte limits.
                Legacy models stage the request payload here as before. H2D obeys
                multimodal_data_device_paths and uses pinned, non-blocking copies.
    Generation (mRoPE only): strip everything except mrope_position_deltas.
    Post-prefill / termination: release request-held cache references, then strip
                raw MM fields so they don't ride along in decode.

[5] Model.forward(attn_metadata, input_ids, position_ids, multimodal_params=…)
    MultimodalModelMixin.prepare_multimodal_inputs: concatenates prompt-ordered
       cache-owned item outputs once into the existing embedding-tensor contract;
       the legacy path still uses get_multimodal_embeddings and its
       full-request/partial-hit behavior.
    find_input_mm_embeds: slices active chunk rows from that tensor.
    prepare_mrope_config (mRoPE models): one-shot mrope_rotary_cos_sin per
       request from the staged mrope_position_ids buffer.
    fuse_input_embeds: merges text and MM rows through the existing precomputed
       index path (optional extra_embeds remain available for multi-feature encoders).
    self.llm.forward(inputs_embeds=..., mrope_config=...) → logits.
```

**Key invariants:**
- [1] and [2] both run off the API event loop. [1] fans media decode out across one request's items with `asyncio.gather`; [2] is single-threaded per request because the HF processor is request-scoped.
- The producer hands off as **handles** at the end of [2], so the broadcast in [3] stays small (Contract 3).
- [4] is the only per-iteration GPU staging; H2D is `non_blocking=True` from pinned host memory.
- [5] runs on the compute stream and must be sync-free (Contract 1).
- Item-scheduled encoder outputs live only in the model-owned `TensorLRUCache`.
  Each request remembers the cache entry for every item, in prompt order, but
  does not keep a second output tensor or combined output buffer. The cache is
  large enough for one legal encoder iteration; `encoder_cache_max_bytes` may
  make it larger when reuse is enabled. Stable item keys let requests share
  one encoded output. Items without a stable key use a request-local key. The
  single tensor assembled for an LLM forward is temporary and is not retained
  as a second request-owned cache.

### EPD-disaggregated path

When `@support_multimodal_disaggregated` is set and the deployment uses `TLLM_MULTIMODAL_DISAGGREGATED=1`:

- **Encoder worker:** runs as a standalone `MultimodalEncoder` (`mm_encoder_only=True`). It executes only the multimodal encoder and ships `mm_embeddings` (+ mRoPE position ids/deltas) to prefill+decode workers as shared-tensor handles.
- **Prefill+decode worker:** the model's `__init__` skips constructing `self.mm_encoder` when `_is_mm_disagg()` is true; the input processor's `attach_multimodal_embeddings()` override binds the encoder handles into the request. For context-only requests, the engine re-clones mrope tensors so IPC handles outlive the encoder worker's freed memory — replicate that pattern for any new GPU-resident mm tensors.

This item-scheduled cache path currently applies only when the encoder and LLM
run in the same process. `mm_encoder_only` / EPD keeps its existing
shared-handle transfer path. Item scheduling also rejects side-stream encoder
prefetch today, and LLM prefill waits until every item in a request is ready.
Side-stream support and item/LLM prefill overlap are separate follow-ups.

### Templates to study

`modeling_qwen3vl.py` and `modeling_mistral.py` are the canonical references for `MultimodalModelMixin`, item scheduling, and the unified encoder-output cache. Use `modeling_llava_next.py` and `modeling_gemma3vl.py` for modality/family details while recognizing that their runtime path is not migrated yet. Other examples: `modeling_phi4mm.py` (audio), `modeling_mllama.py`, `modeling_hyperclovax.py`. `modeling_qwen2vl.py` retains an HF-passthrough vision tower for the outdated Qwen2-VL family; read it for context but don't copy that pattern.

---

## Reuse before you write (the most important rule)

Every common block has a TRT-LLM implementation; compose, don't reimplement. Hand-rolled `nn.Linear` / `nn.LayerNorm` / `nn.MultiheadAttention` silently work in fp16/bf16 single-GPU eager and silently break under quantization, TP, attention-backend selection, KV cache, and CUDA graphs. Browse `tensorrt_llm/_torch/modules/` before writing a layer; the reference VLMs (`modeling_qwen3vl.py`, `modeling_llava_next.py`, `modeling_gemma3vl.py`) show canonical wiring. Reuse is also where every future perf improvement lands automatically.

### Compute modules

Mappings most often missed by adapters:

| Concern | Module | Non-obvious wiring |
|---------|--------|--------------------|
| Linear | `_torch.modules.linear.Linear` | Pass `mapping=model_config.mapping`, `tensor_parallel_mode=TensorParallelMode.{COLUMN,ROW,NONE}`, `allreduce_strategy=model_config.allreduce_strategy`. Every quant scheme (FP8 / NVFP4 / W4A8 / AWQ / weight-only) is automatic — never substitute `nn.Linear`. |
| Attention (text **and** vision) | `_torch.attention.attention.Attention` (variants: `qk_norm_attention.QKNormRoPEAttention` for QK-norm + YARN; `mla.MLA` for DeepSeek-style) | Same module runs the LLM and the multimodal encoder. For the encoder side, build an ad-hoc `attn_metadata` per forward and pass `predefined_attention_mask=PredefinedAttentionMask.FULL` (or windowed). Reference: `Qwen2_5_VisionModel.prepare_attn_metadata`. |
| MLP / Gated MLP | `_torch.modules.mlp.MLP`, `_torch.modules.gated_mlp.GatedMLP`, `_torch.modules.swiglu.swiglu` | `GatedMLP` covers the SwiGLU pattern (gate + up + silu + down) with fused gate/up weights — don't roll it from two `Linear`s and `F.silu`. Plain `MLP` for non-gated cases. Both inherit the same TP / quant story as `Linear`. Reference: `Qwen2_5_VLMLP`. |
| RoPE | `_torch.attention.rotary_embedding.{RotaryEmbedding, MRotaryEmbedding}` | `MRotaryEmbedding` (mRoPE) is for the **LLM** side of mRoPE-using VLMs (Qwen-VL family), with `mrope_section`-aware cos/sin slicing and 3D `position_ids`. The encoder's internal 2D RoPE uses plain `RotaryEmbedding`. |

### LLM backbone — reuse via AutoModel

The inner LLM is loaded via TRT-LLM's own `AutoModelForCausalLM` (`tensorrt_llm._torch.models.modeling_auto`), **not** `transformers.AutoModelForCausalLM`. It dispatches on `pretrained_config.architectures[0]` to whichever class is registered via `@register_auto_model`. The canonical wiring (using `text_config` to surface the inner LLM) lives in the Phase 2 template; if the inner LLM doesn't yet have a TRT-LLM modeling file, finish that text-only onboarding first.

### Multimodal encoder — port to TRT-LLM modules

**This is required, not a preference.** Re-implement encoder blocks from `_torch.modules.*`; the encoder builds its own `attn_metadata` via `prepare_attn_metadata`. Reference: `Qwen2_5_VisionModel`, `Qwen2_5_VLVisionAttention`, `Qwen2_5_VLPatchMerger`. Two reasons:

1. **Performance.** HF-eager runs on PyTorch's stock kernels — vanilla SDPA, `nn.LayerNorm`, plain `nn.Linear` — losing TRT-LLM-attention / FlashInfer, fused RMSNorm, FP8/NVFP4/AWQ Linear, TP, and CUDA-graph capture for static-shape paths. For a 0.5–7 B encoder running every prefill, the regression compounds each iteration.
2. **Version coupling.** Every `from transformers.models.<family> import <X>` ties the modeling file to a specific `transformers` release. Upstream HF refactors (renamed classes, signature changes, internal helper migrations) silently break TRT-LLM imports months later, often surfacing only when users upgrade their environment. Porting cuts the dependency. The same applies to importing HF *computations* / helper functions, not just modules — keep both out of new modeling files.

**The lone existing exception** is `Qwen2VLModel`, which keeps `Qwen2VisionTransformerPretrainedModel` from `transformers` because Qwen2-VL is an outdated family on life support — not because passthrough is acceptable for new onboarding. Don't copy that pattern. If you genuinely cannot port (e.g. patching an existing legacy model), the HF import must carry a code comment explicitly justifying it; otherwise PR review will bounce.

### Weight loading — reuse mappers

If HF prefixes don't match (`model.vision_tower.* → mm_encoder.*`, fused/un-fused QKV, etc.), inherit from a related mapper rather than ad-hoc translation. Reference: `_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py`, `qwen3vl_weight_mapper.py`.

**Host memory during init / weight loading.** Large VLMs can blow past host RAM if every rank materializes the full state_dict before sharding. Two patterns from `modeling_nemotron_nano.py:NemotronH_Nano_VL_V2` (PR [#13283](https://github.com/NVIDIA/TensorRT-LLM/pull/13283)):

1. **Defer multimodal-encoder construction out of `__init__` and into `load_weights()`** when the encoder contains HF submodules whose deterministic init ops (`ones_`, `zeros_`, `fill_`, `.detach()`, `.to(dtype=...)`) clash with the LLM's `MetaInitMode` fast path. Snapshot the multimodal `ModelConfig` in `__init__` (since `post_config()` overwrites `self.model_config.pretrained_config` to the LLM-only config), construct the encoder + `.to("cuda")` inside `load_weights()`. Otherwise `MetaInitMode` raises and the entire model falls back to slow CPU init.
2. **Call `weights.mark_consumed(<prefix>)` after each sub-module's `load_weights(...)`** so the mmap-backed shards behind those weights can be released. Without it, peak host memory holds the *entire* checkpoint; with it, peak holds only the shard you're currently loading. Tag every prefix you've finished — encoder, sound, projector, LLM.

### Don't touch

PyExecutor + the C++ core own `AttentionMetadata`, KV cache, scheduler, sampler, decoder. Your model receives `attn_metadata` and `multimodal_params` as inputs and returns logits — never builds request-level metadata. The only `attn_metadata` you build yourself is the **multimodal encoder's own**, on the synthetic per-item batch (concatenated patches with per-image seqlens, mel frames, etc.).

---

## Performance contracts

Three rules. Multimodal prefill is long (image/audio tokens balloon sequence length) and media tensors are big (MBs–GBs); the overlap scheduler hides host work behind GPU work only if all three hold.

### Contract 1 — Zero CPU-GPU syncs inside `forward`

A single sync inside `forward` collapses overlap, and per-iteration GPU work is long for VLMs, so stalls compound.

**Banned in `forward` and anything it calls:**
- `.item()`, `.tolist()`, `int(t)`, `bool(t)`, `float(t)` on GPU tensors
- `t.cpu()`, `t.to("cpu")`, any device-crossing read
- Python `if`/`while` on tensor *values* (shape is fine; values are not)
- `torch.nonzero`, single-arg **`torch.where(condition)`** (index form; documented sync hazard in `filter_mm_token_from_input_ids` when run on GPU `input_ids`), `torch.unique`, `masked_select`
- `torch.tensor([...], device="cuda")` from a Python list (hidden H2D)
- HF runtime branches (`if pixel_values is None: ...`) that change tensor shapes

Three-arg **`torch.where(cond, x, y)`** is fine when **`cond`** is built only on-device (no scalar readback). **`fuse_input_embeds`:** kwargs **`text_token_indices` + `mm_token_indices`** together ⇒ skip internal `filter_*`. **`trtllm-serve`** usually supplies both via **`model_engine.py`** (CPU-side index build → `inputs` → `fuse_input_embeds(..., **kwargs)`). Pure-text batches have no MM `inputs`; bare unit tests / direct calls may omit indices ⇒ in-model `filter_*` runs.

**Patterns:**

- **Static graph for mixed batches.** Don't add `if has_mm:` branches. `find_input_mm_embeds` returns input unchanged when runtime is None; `fuse_input_embeds` returns `(input_ids, None)` when `mm_embeds == []` — preserve that contract.
- **mRoPE: compute once per request, never per layer.** The pipeline (input processor → engine → `prepare_mrope_config`) is laid out in the system map; the constraint here is that per-layer attention must read pre-sliced `(cos, sin)` — never recompute mrope inside the decoder loop.

**Audit:** grep for the banned constructs; run one prefill iteration with `torch.cuda.set_sync_debug_mode("warn")` and confirm zero warnings from your model.

### Contract 2 — Preprocessing on CPU, async, server-side

CPU-bound work (decode / resize / normalize / mel-spectrogram / frame extraction) must not compete with GPU work, block the request loop, or serialize across requests.

- HF AutoProcessor + image_processor + tokenizer run inside the input processor's `call_with_text_prompt` (dispatched from `__call__`) — *not* in the model worker.
- URL/bytes media goes through `async_load_image` / `async_load_video` / `async_load_audio` (all wrap blocking decode in `asyncio.to_thread`). Never call `PIL.Image.open(...).load()` / `cv2.VideoCapture` / `soundfile.read` synchronously on the request hot path.
- Pin host tensors before H2D with `prefer_pinned()` (False under Confidential Compute (CC), True otherwise). The engine pins `multimodal_data` automatically via `to_device(..., pin_memory=prefer_pinned())`.
- **Declare `multimodal_data_device_paths`** on the model — list of dotted paths (e.g. `["image.pixel_values", "image.image_grid_thw", "video.pixel_values_videos", "video.video_grid_thw", "multimodal_embedding"]`) telling the engine which fields go to CUDA. Anything not listed stays on CPU.
- Optional tokenized+MM fast path: set `supports_token_id_mm_expansion = True` (a `ClassVar`, default `False`) and implement `get_text_with_mm_placeholders` + `expand_prompt_token_ids_for_mm`. The base-class `__call__` then routes `prompt_token_ids + multi_modal_data` (no `prompt`) requests through `call_with_token_ids`, skipping redundant detokenization. When the flag is `False` (most VLMs), the base class detokenizes `prompt_token_ids → prompt` and re-runs `call_with_text_prompt`, so token-ID inputs still work — just less efficiently. Only LlavaNext + NanoV2VL opt in today.
- Forward `mm_processor_kwargs` from `inputs.get("mm_processor_kwargs", {})` to the HF processor (callers tune things like video sample rate via this).

### Contract 3 — Large media via shared tensors, never raw pickle

A 1024×1024 fp32 patch tensor is ~12 MB; a video clip can be hundreds of MB. Naive pickle through MPI broadcast turns the leader into the IPC bottleneck.

- **Always use `MultimodalParams.to_handle`/`to_tensor`.** `to_handle` swaps each tensor inside `multimodal_data` for a small dict — `{method_key, tensor_size, storage_handle, ...}` — that points at the same memory: a CUDA-IPC handle for GPU tensors (`REBUILD_CUDA`) or a POSIX-shm handle for CPU tensors (`REBUILD_CPU`). The dict is a few hundred bytes regardless of the original tensor size. Consumers call `to_tensor` to rebuild local tensor views from the handle. See `_torch/shared_tensor/`.
- **Where it crosses ranks:** the executor broadcasts `py_multimodal_data` via `dist.broadcast` / `tp_cp_broadcast` / PP send-recv. Payload size = the literal byte size of whatever's in `py_multimodal_data` — confirm every tensor inside has been swapped for its handle dict (i.e. `to_handle` ran) before this point.
- **Release and strip after prefill.** `PyExecutor._release_multimodal_resources`
  releases the request's cache entries and then calls
  `strip_mm_data_for_generation`. Repeated cleanup is safe. If your model needs
  data during decode, update `strip_mm_data_for_generation`; do not add another
  post-prefill strip helper.
- **EPD disagg.** Embeddings still cross workers as shared tensors, not bytes — see the EPD-disaggregated path section above for the encoder/prefill-worker split.
- **Hashes are small; broadcast eagerly.** `MultimodalInput.multimodal_hashes` (blake3) drives KV-cache reuse — never substitute raw pixels for them.

**Audit:** payload size in NVTX `broadcast_requests` / `tp_broadcast_requests` ranges should be < 1 MB per rank per request. More means a broadcast leaked raw tensors.

### Contract 4 — Batch the multimodal encoder across requests

Implement `MultimodalModelMixin.encode_multimodal_inputs` as one batched forward over its
`MultimodalParams` list. The item-scheduled path calls `prepare_multimodal_encoder_inputs` for
only the selected producer items, then `forward_multimodal_encoder_items` groups compatible
modality runs and calls that same encoder hook. The legacy path also calls it for uncached
requests through `get_multimodal_embeddings`. Concatenate requests' `pixel_values` /
`image_grid_thw` / mel frames, build one ad-hoc `attn_metadata` whose `seq_lens` preserves item
boundaries, and run the encoder blocks once. Never loop over requests and call the encoder once
per request.

**Pattern (Qwen3-VL).** `encode_multimodal_by_groups` concatenates items for compatible
modalities, runs one encoder call, and restores request/prompt order from `mm_item_order`.
`forward_multimodal_encoder_items` then splits the result into one tensor per atomic item using
the processor-declared output lengths. Override `build_multimodal_encoder_input` only when the
default packed-grid, stacked-image, or stacked-audio slicers cannot represent the model's input.

**Audit.** Under load with several multimodal requests in one batch, the encoder kernels in nsys should appear as **one wide block per iteration**, not N narrow blocks. A fan of N narrow blocks means the encoder is being looped per request instead of batched — one of the easiest VLM perf regressions to introduce while refactoring.

---

## Phases

### Phase 0 — Gather resources

```bash
huggingface-cli download {org}/{model} --exclude "*.safetensors" "*.bin" "*.pt" "*.gguf"
```

Confirm `preprocessor_config.json` and `chat_template.json` are pulled. Verify `AutoProcessor.from_pretrained(model_path)` loads. Estimate **LLM + multimodal encoder** params for VRAM sanity (multimodal encoders are often 0.5–7 B on top of the LLM).

### Phase 1 — Survey existing coverage

Read `config.json`'s `architectures` and `model_type`. If a `_torch/models/modeling_*.py` already claims that architecture via `@register_auto_model`, extend rather than create new. Identify the closest existing multimodal model and note which TRT-LLM modules it reuses.

### Phase 2 — Model wrapper

Create `tensorrt_llm/_torch/models/modeling_{name}.py`. The default pattern below mirrors `modeling_llava_next.py` and `modeling_gemma3vl.py` — a single wrapper class that composes a multimodal encoder + an LLM resolved through `AutoModelForCausalLM.from_config(text_config)`. The `*ModelBase + *Model` Base/non-Base split in `modeling_qwen2vl.py` and `modeling_qwen3vl.py` is an implementation detail for sharing one wrapper between two variants of the same family (Qwen2-VL ↔ Qwen2.5-VL; Qwen3-VL ↔ Qwen3-VL-MoE) — keep the wrapper a single class unless you have the same multi-variant need.

```python
class {Name}VisionModel(nn.Module):
    """Multimodal encoder. Composes _torch.attention.{attention.Attention, rotary_embedding.RotaryEmbedding} and _torch.modules.{linear.Linear, rms_norm.RMSNorm, gated_mlp.GatedMLP}."""

    def forward(self, multimodal_params: List[MultimodalParams]) -> torch.Tensor:
        # Concat pixel_values across all requests, build per-image attn_metadata
        # via prepare_attn_metadata, then run encoder blocks once (Contract 4).
        ...


class {Name}Model(MultimodalModelMixin, PreTrainedModel):
    config_class = {Name}Config
    supports_encoder_cache = True
    # Enable only after implementing the Phase 3 atomic-item contracts.
    supports_mm_encoder_item_scheduling = True

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return  # idempotency guard — re-entry from `post_config` etc.

        if not _is_mm_disagg():
            self.mm_encoder = {Name}VisionModel(model_config)
        else:
            self.mm_encoder = None

        # Inner LLM is resolved from text_config; no architectures rewrite needed.
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = model_config.pretrained_config.text_config
        # TRT-LLM's AutoModel (tensorrt_llm._torch.models.modeling_auto), not transformers'.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.model_config = model_config
        self.post_config()

    def post_config(self):
        # After llm is constructed, downstream code expects self.config to be the LLM-shaped config.
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def encode_multimodal_inputs(
        self, multimodal_params: List[MultimodalParams]
    ) -> torch.Tensor:
        if self.mm_encoder is None:
            raise ValueError("Raw multimodal inputs require a local encoder")
        return self.mm_encoder.forward(multimodal_params)

    @property
    def language_model(self) -> torch.nn.Module:
        return self.llm

    @property
    def text_embedding_layer(self):
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.text_embedding_layer.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "image.image_grid_thw", "multimodal_embedding"]
```

**Required (every multimodal model):**

- Inherit `MultimodalModelMixin` and use its `forward` unless the family needs explicit
  language-model delegation. If overriding, call `prepare_multimodal_inputs`; never add
  `pixel_values` / `image_grid_thw` / `attention_mask` as direct args.
- Implement `encode_multimodal_inputs`, `language_model`, `text_embedding_layer`,
  `embedding_dim`, and `embedding_dtype`. The encoder must return one tensor whose first
  dimension equals the processor-declared aggregate output length. The item path validates and
  splits it per item; the legacy path retains its existing full-request behavior.
- Set `supports_mm_encoder_item_scheduling=True` only when the input processor emits valid atomic
  item metadata and the model can slice and batch selected items. Set `supports_encoder_cache=True`
  only when the production forward consumes the mixin cache path. Do not construct or mutate a
  request-local encoder-output cache in the model.

**Family-specific extras (apply only when relevant):**

- **mRoPE (Qwen-VL family):** add `init_mrope_embedding(model_config)` in `__init__`, then
  build and forward the mRoPE config from `get_language_model_extra_forward_kwargs`. Reference:
  `Qwen3VLModelBase.prepare_mrope_config`.
- **Deepstack features (Qwen3-VL):** keep each item's primary and deepstack rows in the same cache
  entry, then use the existing `after_active_multimodal_embeddings` and
  `_fuse_multimodal_embeddings` hooks after the prompt-order tensor is assembled.
- **HF wrapper without a clean `text_config`:** Qwen2-VL's `Qwen2VLModelBase` rewrites `architectures` to surface the inner LLM. Fall back to that pattern only when the multimodal HF config does not expose a `text_config` sub-config.
- **Inner LLM that doesn't match HF's `text_config` schema (Qwen3.5-MoE-VL → Qwen3Next).** When the VLM's HF `text_config` schema differs from the TRT-LLM runtime model you want to reuse, write a config normalizer (e.g. `_normalize_qwen35_moe_vl_config`) that maps HF aliases to the runtime's expected names (mRoPE keys, `intermediate_size` aliases, quantization-exclude module paths). Wire it via **lazy import** from `pyexecutor.config_utils.load_pretrained_config` — the `Mistral` and `Qwen3_5` branches are templates. Two gotchas: transformers 5.x's `rope_scaling` is a **property aliasing `rope_parameters`** — setting either silently overwrites the other, so the normalizer should mutate `rope_parameters` directly if the HF code still reads from it. And for VLMs, the normalizer must run on the **composite** config (with `text_config` / `vision_config`), not flattened away.
- **Thin wrapper for runtime reuse.** Even when the LM class body is identical to the runtime's existing class, still create a `@register_auto_model("YourArch")`-decorated thin subclass — that's how weight-mapper dispatch picks the family-specific mapper. You can't stack two `@register_auto_model` decorators on a single shared class.

### Phase 3 — Input processor + dummy builder

Subclass **both** `BaseMultimodalInputProcessor` (drives every real request) and `BaseMultimodalDummyInputsBuilder` (drives engine warmup / KV-cache profiling). Colocate in the modeling file. References: `Qwen3VLInputProcessorBase` (image+video), `Mistral3InputProcessor` (Pixtral).

**Encoder KV-cache memory profiling (deterministic dummy sizing).** The KV-cache profiler sizes the encoder's memory contribution by running the encoder **once** on a worst-case dummy (held resident through the peak measurement), **decoupled** from the text-only LLM dummy. To opt in, the model exposes `encode_multimodal_inputs` (via `MultimodalModelMixin`) and the input processor implements the existing `BaseMultimodalDummyInputsBuilder` contract:

- `get_mm_max_tokens_per_item(max_num_encoder_tokens=None) -> {modality: tokens}` — report bounded startup maxima when the argument is `None`, or the largest legal item per modality under a concrete runtime token budget.
- `get_dummy_mm_data(*, max_num_encoder_tokens, mm_counts, dtype) -> multimodal_data` — build the profiler-selected per-modality item counts **directly** as processed encoder tensors (no PIL image + HF-processor round-trip).

Following vLLM's separation of common budgeting from model processing geometry, the profiler selects the largest legal runtime workload and computes its repetition count under `encoder_max_num_tokens` and `encoder_max_batch_size`. It passes the resulting plan (for example, `{"image": 8}`) to the concrete processor's `get_dummy_mm_data`; the processor owns only geometry and tensor layout. Qwen builds `pixel_values`/`*_grid_thw`; Mistral builds `pixel_values`/`image_sizes` and uses its private `_vit_tokens` helper so the LLM-side Pixtral hashing count is unchanged. A model without the contract falls back to a text-only dummy (encoder memory unaccounted). Don't hardcode the encoder attention workspace (`max_num_*=8192`): inherit `MultimodalEncoderMixin` and let the engine size it via `setup_attn_metadata` at load.

The workspace dimension comes from `encoder_max_num_tokens`, falling back to the LLM-side `max_num_tokens` when unset. `encoder_max_batch_size` independently bounds the number of atomic items in a multimodal encoder iteration and falls back to `max_batch_size`. The profiler resolves both limits into `mm_counts` before calling `get_dummy_mm_data`. Model-internal attention sequence counts are not the item batch size; they are derived separately from the token budget and model geometry.

> Mixed image+video+audio models profile the supported modality with the largest legal per-item workload under the configured limits. Runtime mixed-modality requests still share the same aggregate token limit.

**Atomic-item scheduling.** Override `get_mm_encoder_item_metadata` to return
prompt-ordered `item_refs`, the encoder-token cost of each item, and each
item's output length. Also implement `get_max_mm_encoder_output_embeddings` so
the engine can size the cache for one legal encoder iteration. The declared
output lengths must match both the MM placeholder spans and the tensor splits
from `forward_multimodal_encoder_items`.

Implement `call_with_text_prompt(inputs, sampling_params)` — the per-model text-prompt path. **Don't override `__call__`**: the base class's concrete `__call__` dispatches here for text prompts, and also detokenizes `prompt_token_ids → prompt` and falls through to here for non-fast-path VLMs. `call_with_text_prompt` does:

1. Pull `text_prompt`, `mm_data`, `mm_processor_kwargs` from `inputs`.
2. `_preprocess(...)` — HF processor produces `pixel_values` / `pixel_values_videos` / `*_grid_thw` / `input_ids`.
3. Build `multimodal_data` keyed by modality: `{"image": {"pixel_values": ..., "image_grid_thw": ...}, "video": {...}}`.
4. Compute `mrope_config` on **CPU** (`.to("cpu").clone()`) into `multimodal_data["mrope_config"]`. Required even on text-only Qwen-VL prompts — no branch.
5. `_postprocess(input_ids)` rewrites HF's `image_token_id` / `video_token_id` to `tllm_multimodal_token_id = vocab_size + 1` (the OOV sentinel). Skip when `mm_data` is empty.
6. Return `(prompt_token_ids_list, {"multimodal_data": multimodal_data})`.

**Optional tokenized+MM fast path (skip unless needed):** set `supports_token_id_mm_expansion = True` (`ClassVar`) and implement `get_text_with_mm_placeholders(mm_counts)` + `expand_prompt_token_ids_for_mm(prompt_token_ids, num_mm_tokens, ...)`. The base-class `call_with_token_ids` then builds dummy placeholder text, runs `call_with_text_prompt` on it, expands the real token IDs, and merges any returned `mm_data_updates` (e.g. video `evs_ids`) into `multimodal_data`. Leave the flag `False` and the base class just detokenizes token-ID inputs and re-runs `call_with_text_prompt`. Only LlavaNext + NanoV2VL opt in today.

**EPD override (if `@support_multimodal_disaggregated`):** override `_attach_multimodal_embeddings_impl(inputs, multimodal_embedding, sampling_params)` — **not** the `attach_multimodal_embeddings` wrapper — to consume encoder outputs in the prefill+decode worker. The base wrapper detokenizes tokenized inputs for non-fast-path VLMs before delegating to your impl.

**Decorator stack** — bottom-up application; `register_vision_encoder` requires `register_auto_model` to have run first:

```python
@support_multimodal_disaggregated                              # outermost (after validation)
@register_vision_encoder({Name}VisionModel, vlm_base_model=HFVisionTransformerClass)
@register_auto_model("{ArchName}ForConditionalGeneration")
@register_input_processor(
    {Name}InputProcessor, model_type="{model_type}",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|vision_start|><|image_pad|><|vision_end|>", ...},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
        content_format=ContentFormat.STRING,
    ),
)
class {Name}Model(PreTrainedModel): ...
```

### Phase 4 — Weight loading

```python
def load_weights(self, weights, weight_mapper):
    if not _is_mm_disagg():
        self.mm_encoder.load_weights(weights)
        # Release mmap pages backing the encoder weights as soon as we're done.
        if hasattr(weights, "mark_consumed"):
            weights.mark_consumed("vision_model")  # adjust prefix per your checkpoint
    weight_mapper = {Name}HfWeightMapper()
    weight_mapper.init_model_and_config(self.llm, self.model_config)
    filtered = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
    self.llm.load_weights(filtered, weight_mapper)
    if hasattr(weights, "mark_consumed"):
        weights.mark_consumed("language_model")
```

Inherit from a related mapper for prefix surgery — don't write a one-off translator.

### Phase 5 — Tests

**Per-model unit test (the main one)** at `tests/unittest/_torch/modeling/test_modeling_<name>.py`. Subclass `TestModelingMultimodal` from `tests/unittest/_torch/modeling/test_modeling_multimodal.py` (an abstract `unittest.TestCase`) and implement six abstract methods: `get_model_config`, `get_trtllm_model_class`, `get_hf_model_class`, `get_weight_mapper_class`, `get_model_type`, `get_model_config_class`. The base class drives a `MultimodalScenario`-parameterized run (modality ∈ `image` / `multiple_image` / `video` / `text` / `mixture_text_image` / `audio`, with optional `use_cuda_graph` / `chunked_prefill` / `kv_cache_reuse`) — comparing TRT-LLM logits to HF reference, exercising the KV cache manager, attn metadata, mrope, fusion path, and CUDA graph capture in one harness. Override `get_scenarios` to declare which combinations apply to your model. Reference: `test_modeling_qwen3vl.py`, `test_modeling_qwen2_5vl.py`, `test_modeling_nemotron_nano_v2_vl.py`. Test data lives under `${LLM_MODELS_ROOT}/multimodals/test_data/`.

**Hybrid linear-attention models.** Override `_dummy_request_kwargs` to return `{"use_mrope": True}` if the model uses mRoPE (allocates the 3-D position-id buffer at dummy-request time). The base class's `init_kv_cache_manager` already dispatches on `is_qwen3_hybrid` / `is_nemotron_hybrid` to build `CppMambaHybridCacheManager` — don't override unless you need a different concrete manager. Use `PyKvCacheConfig` from `llmapi.llm_args` (Pydantic), not the C++ bindings `KvCacheConfig` — `CppMambaHybridCacheManager.__init__` reads `mamba_state_cache_interval` which only exists on the Pydantic side. **CUDA-graph capture in the harness doesn't currently address the Mamba SSM state buffer** — keep `use_cuda_graph=False` in `get_scenarios` for hybrid models until that's wired through; production CUDA-graph support is independent and unaffected.

**Synthetic-config shape couplings.** `head_dim × partial_rotary_factor / 2 == sum(mrope_section)` — `head_dim` can't be shrunk independently. If the test loads the real tokenizer via `_name_or_path`, `vocab_size` must equal the real tokenizer's vocab — otherwise chat-template specials at ids `>=` your synthetic `vocab_size` get misclassified as mm tokens by `fuse_input_embeds`'s OOV filter (manifests as "found N image tokens but received M image embeddings", off by exactly the number of chat-template specials). Vision deepstack indices `[i, j, k]` require `depth > k` — the HF processor reserves placeholder tokens for deepstack outputs regardless of whether the encoder is configured to emit them.

**Two-config Approach B for tests.** If you've added a config normalizer (Phase 2), keep `self.hf_config` raw and route a `deepcopy + normalize` only through `create_trtllm_model`. Reusing one normalized config for both HF and TRT-LLM construction trips the transformers 5.x property aliasing and silently corrupts HF-side schema (`rope_scaling ↔ rope_parameters`).

**Tolerance band.** Default `get_tolerance` returns `0.4 / 0.4`, calibrated to pass for the existing VLM tests but wide enough to mask argmax-changing bugs. After your test passes, dial it tighter — keep `atol = 0.4` to absorb single-logit tail outliers seen on `multiple_image` / `video` scenarios; tighten `rtol` toward `0.1` to gate bulk-of-logits relative agreement. Don't drop `rtol` below `0.05` without cross-SKU validation.

**Supplemental utility tests (only if your model exercises new logic in shared utilities)** under `tests/unittest/_torch/multimodal/`: `test_fuse_input_embeds.py`, `test_multimodal_runtime.py`, `test_find_num_image_tokens.py`, `test_external_embedding.py`, `test_share_multiparams.py`, `test_mm_encoder_standalone.py`. Extend the existing tests rather than creating new files when the coverage is generic.

**Accuracy test** at `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py` — subclass `LlmapiAccuracyTestHarness`, set `MODEL_NAME` / `MODEL_PATH` / `MAX_NUM_TOKENS=16384`, run `MMMU` (or `ChartQA`/`ScienceQA`). Reference: `TestQwen2_5_VL_7B`. Wire into `tests/integration/test_lists/test-db/l0_<gpu>.yml`.

- **First-run reference capture.** Set `TRTLLM_ACCURACY_NO_REFERENCE=1` for the first local run; the harness synthesizes a baseline reference (`0` for higher-is-better metrics like MMMU), runs end-to-end, and prints the achieved accuracy. Paste the printed value verbatim into `tests/integration/defs/accuracy/references/mmmu.yaml` — that's the measured reference; the threshold derives from it via `sigma` / `alpha` / `beta`.
- **`quant_algo` assertion in `test_fp8_prequantized` must match what the checkpoint actually advertises.** Flat per-tensor FP8 is `QuantAlgo.FP8`; block-scaled FP8 (DeepSeek-V3 / Qwen3.5 style) is `QuantAlgo.FP8_BLOCK_SCALES`. Same applies to NVFP4 variants. Easy to copy from a peer model and assert the wrong one.

**Be parsimonious.** The cartesian product `modality × use_cuda_graph × chunked_prefill × kv_cache_reuse` explodes fast. In `get_scenarios()`, pick the smallest set covering this model's distinctive paths — e.g. one image, one `mixture_text_image`, plus one chunked-prefill / one cuda-graph entry only if the model claims those features. One accuracy benchmark per model (MMMU for image VLMs); add another only for capabilities the first doesn't exercise (audio, video, very long context).

### Phase 6 — Docs + serve verification

`docs/source/models/supported-models.md`:
- **Supported Models** table: row alphabetical by architecture class.
- **Multimodal Feature Support Matrix (PyTorch Backend)**: row with columns *Overlap Scheduler / CUDA Graph / Chunked Prefill / Torch Sampler / TLLM C++ Sampler / KV Cache Reuse / Logits Post Processor / EPD Disaggregated Serving / Modality (L+I+V+A)*. Mark `Yes` only what you've verified.

**First line of defense — quickstart smoke test.** Before bringing up a server, run the bundled quickstart against your model:

```bash
python examples/llm-api/quickstart_multimodal.py \
    --model_dir <hf_model_id> --modality image \
    --media <url-or-path>
```

It exercises `setup_llm` + `default_multimodal_input_loader` + the chat template + `LLM.generate` end-to-end with a couple of bundled prompts. Cheaper than spinning up `trtllm-serve` and fails fast on input-processor / encoder / fusion bugs. Run for every modality your model supports (`--modality image|video|audio|image_audio|...`).

**Then aggregated serving:**

```bash
trtllm-serve <hf_model_id> --backend pytorch --max_num_tokens 16384 --port 8000
```

Send a chat completion with a real image; confirm coherent output. (TODO: 2ez4bz to provide ready-to-use curl examples.)

**Chunked-prefill cache verification (mandatory).** Re-run with a deliberately small `--max_num_tokens` to force the prefill of one image to span multiple chunks, then grep the server log for these two lines:

- `Multimodal hashing failed:` → the input processor's hash path fell back; KV-cache reuse across requests with the same image is broken (Contract 3 hash invariant).
- `Multimodal runtime data missing or incomplete, will not cache embeddings.` → the encoder-output cache is being skipped, so the encoder is being re-run on every chunked-prefill iteration of the same request (Phase 2 Required: encoder output length must match MM placeholder count).

A clean serving log shows neither line. If either appears, fix it before declaring the model done — these are silent perf cliffs, not crashes.

For EPD: run `MultimodalEncoder` and `LLM` as separate process groups; verify embeddings cross via `disaggregated_params.multimodal_embedding_handles`.

### Phase 7 — Pull request

Follow `CONTRIBUTING.md`. Title `[JIRA/NVBUG/None][type] description`, `git commit -s`. Body: one full multimodal prompt → output verbatim, reproduction commands, pytest output verbatim. Trigger CI via `/bot run`.

---

## Pre-PR checklist

**Architecture & registration**
- [ ] Decorator stack in correct order: `@support_multimodal_disaggregated` (outermost, optional) → `@register_vision_encoder` → `@register_auto_model` → `@register_input_processor` (innermost).
- [ ] Model inherits `MultimodalModelMixin`; an overridden `forward` calls `prepare_multimodal_inputs` and takes no raw MM tensor args.
- [ ] `multimodal_data_device_paths` lists every GPU-resident mm field.
- [ ] If runtime-reusing (e.g. Qwen3.5 → Qwen3Next): thin `@register_auto_model` wrapper class present; config normalizer lazy-imported from `pyexecutor.config_utils.load_pretrained_config`.

**Module reuse**
- [ ] No raw `nn.Linear` / `nn.LayerNorm` / `nn.MultiheadAttention` / hand-rolled attention — use `_torch/modules/*`.
- [ ] LLM backbone surfaced via `text_config` + `AutoModelForCausalLM.from_config` (TRT-LLM's, not transformers'); `architectures` rewrite only as the documented Family-specific fallback.
- [ ] Multimodal encoder is **ported** to TRT-LLM modules. No imports of HF modules *or* computations from `transformers.models.<family>` — they couple the file to a specific `transformers` release and break silently on upstream refactors. The only existing exception is the outdated Qwen2-VL family (`Qwen2VLModel`); new models must not follow that pattern.
- [ ] HF→TRT-LLM weight surgery via a `BaseWeightMapper` subclass under `_torch/models/checkpoints/hf/`.
- [ ] `weights.mark_consumed(<prefix>)` called after each sub-module load (mm encoder / projector / LLM) so mmap shards release incrementally; multimodal-encoder construction deferred to `load_weights()` if its HF submodules clash with `MetaInitMode`.

**Input processor**
- [ ] Subclasses both `BaseMultimodalInputProcessor` and `BaseMultimodalDummyInputsBuilder`.
- [ ] Encoder profiling implements `get_mm_max_tokens_per_item` + `get_dummy_mm_data`; item scheduling additionally implements `get_mm_encoder_item_metadata` + `get_max_mm_encoder_output_embeddings`. The model exposes batched `encode_multimodal_inputs`; encoder metadata capacity comes from `setup_attn_metadata`, not hardcoded `max_num_*` values.
- [ ] `call_with_text_prompt` (not `__call__` — that's the base-class dispatcher) runs HF AutoProcessor + tokenizer, builds `multimodal_data` by modality, computes `mrope_config` on CPU, `_postprocess`-rewrites mm token ids to the OOV sentinel.
- [ ] `mm_processor_kwargs` flow-through preserved. (Tokenized fast path is optional: set `supports_token_id_mm_expansion = True` + implement `get_text_with_mm_placeholders` / `expand_prompt_token_ids_for_mm`; otherwise the base class detokenizes token-ID inputs automatically.)
- [ ] `_attach_multimodal_embeddings_impl` implemented (not the `attach_multimodal_embeddings` wrapper) if `@support_multimodal_disaggregated`.

**Performance contracts**
- [ ] Grep clean for Contract 1 bans (`.item()` / `.cpu()` / `.tolist()` / `torch.nonzero` / single-arg `torch.where` / value-dependent `if`, etc.) in modeling `forward` paths — elementwise `torch.where(cond, x, y)` with GPU-only `cond` is fine.
- [ ] `set_sync_debug_mode("warn")` audit on prefill: zero warnings from your model.
- [ ] Async loaders used for URL/bytes inputs.
- [ ] Broadcast payload < 1 MB per rank per request (NVTX `broadcast_requests` / `tp_broadcast_requests`); media crosses ranks only via `to_handle` / `to_tensor`.
- [ ] Post-prefill/termination cleanup drains item-cache refs exactly once and leaves only fields retained by `strip_mm_data_for_generation`.
- [ ] Encoder output is one tensor whose first dim equals the processor-declared MM output rows; cache item splits preserve prompt order and are reassembled into the existing single-tensor fusion contract.
- [ ] Encoder is batched across requests: a multi-request batch produces a single wide encoder block in nsys, not N narrow blocks (Contract 4).

**Tests & docs**
- [ ] Per-model unit test at `tests/unittest/_torch/modeling/test_modeling_<name>.py` subclassing `TestModelingMultimodal`; six abstract methods implemented; `get_scenarios()` declares the *minimum* modality × cuda_graph × chunked_prefill × kv_cache_reuse combinations that cover this model's distinctive paths (no full cartesian product) and they all pass.
- [ ] Mixed-batch scenario (`mixture_text_image`) included and passes against HF reference logits.
- [ ] If hybrid linear-attention: `_dummy_request_kwargs` overridden to set `use_mrope=True`; CUDA-graph scenarios skipped (harness limitation, not a model limitation).
- [ ] Accuracy test under `test_llm_api_pytorch_multimodal.py`; entry in `test-db/l0_<gpu>.yml`.
- [ ] Reference accuracy captured via `TRTLLM_ACCURACY_NO_REFERENCE=1` and entered in `references/mmmu.yaml`; threshold derived value sanity-checked.
- [ ] Prequantized assertions use the correct `QuantAlgo` variant for the checkpoint (flat FP8 vs `FP8_BLOCK_SCALES`; FP4 vs `NVFP4`).
- [ ] Rows added to **Supported Models** + **Multimodal Feature Support Matrix**.
- [ ] `examples/llm-api/quickstart_multimodal.py` round-trip passes for every modality the model supports (`--modality image|video|audio|...`); then `trtllm-serve` round-trip verified with a real image prompt; EPD round-trip verified if applicable.
- [ ] Chunked-prefill verification: ran `trtllm-serve` with low `--max_num_tokens` and confirmed the log contains neither `Multimodal hashing failed:` nor `Multimodal runtime data missing or incomplete, will not cache embeddings.`
- [ ] `/bot run` triggered and multimodal stages pass.
