---
name: trtllm-model-onboard-multimodal
description: >
  Onboard a HuggingFace multimodal (vision/audio/video + text) model to the
  TensorRT-LLM PyTorch backend (`tensorrt_llm/_torch/models/`) — NOT
  AutoDeploy. Wires a vision/audio encoder + LLM into a single
  `@register_auto_model` class plus a `BaseMultimodalInputProcessor`. Keeps
  the overlap scheduler healthy: zero CPU-GPU syncs in `forward`, async CPU
  preprocessing on the server, shared-tensor (not pickle) broadcast of media
  across TP/PP/CP, and (optionally) EPD disaggregated serving.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Multimodal Model Onboarding (PyTorch backend)

> **Scope.** PyTorch backend only (`tensorrt_llm/_torch/`) — the default for `LLM(..., backend="pytorch")`, `trtllm-serve`, `trtllm-bench`. **Not** for AutoDeploy (`tensorrt_llm/_torch/auto_deploy/`); use `ad-model-onboard` for that.

**Output:**
- `tensorrt_llm/_torch/models/modeling_{name}.py` — wrapper class (vision tower + LLM) decorated with `@register_auto_model`, `@register_vision_encoder`, `@register_input_processor` (and `@support_multimodal_disaggregated` if EPD is supported), plus a `BaseMultimodalInputProcessor` (+ `BaseMultimodalDummyInputsBuilder`) subclass.
- `_torch/models/checkpoints/hf/{name}_weight_mapper.py` if HF prefixes need surgery.
- Per-model unit test under `tests/unittest/_torch/modeling/test_modeling_<name>.py` (subclass of `TestModelingMultimodal`); supplemental utility tests under `tests/unittest/_torch/multimodal/` if needed; an accuracy test under `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`; support-matrix entry; verified `trtllm-serve` flow.

## System map

### Aggregated path (default)

Two separate CPU stages run on the API host, then the request hits the worker as a handle and gets staged for GPU per iteration.

```
[1] API event loop (async)                         tensorrt_llm/serve/openai_server.py
    │  parse_chat_messages_coroutines(...)         → tensorrt_llm/serve/chat_utils.py
    │     for each image_url / video_url / audio_url part:
    │        wrap async_load_image / _video / _audio (asyncio.to_thread) — coroutine only
    │  apply_chat_template(...)                    → text prompt
    │  await mm_data_tracker.retrieve_all_async()  → asyncio.gather decodes all media concurrently
    │  prompt["multi_modal_data"] = mm_data  (or multi_modal_embeddings if pre-encoded)
    │  generate_inputs = await asyncio.to_thread(LLM.preprocess, ...)
    ▼
[2] LLM._preprocess  (CPU thread, off the event loop)   tensorrt_llm/llmapi/llm.py
    │  Branch on inputs:
    │   (a) is_mm_disagg (prefill+decode worker, EPD): consume
    │       disaggregated_params.multimodal_embedding_handles directly →
    │       input_processor.get_prompt_token_ids(inputs, mm_handles); no encoder
    │   (b) prompt_token_ids only, no mm: passthrough
    │   (c) standard path: input_processor_with_hash =
    │       create_input_processor_with_hash(self.input_processor)
    │       → blake3 hash mm_data (+ optional multi_modal_uuids)
    │       → BaseMultimodalInputProcessor.__call__():
    │           HF AutoProcessor → pixel_values / *_grid_thw / input_ids
    │           mrope_position_ids/deltas on CPU into multimodal_data["mrope_config"]
    │           _postprocess: HF image/video token ids → tllm_multimodal_token_id (OOV)
    │       → find_mm_token_lengths / find_mm_token_positions / special_token_offsets
    │   (d) multi_modal_embeddings: input_processor.attach_multimodal_embeddings(...)
    │  build MultimodalParams(multimodal_input=..., multimodal_data=...)
    │  multimodal_params.to_handle("multimodal_data")    ← producer-side: handles, not bytes
    ▼
[3] GenerationExecutor.generate_async                tensorrt_llm/executor/base_worker.py
    │  enqueue → consumer worker:
    │  - pull multimodal_input.{hashes,positions,lengths,uuids} into the C++
    │    executor_request (drives KV-cache hash matching)
    │  - request.multimodal_params.to_tensor("multimodal_data") rebuilds local views
    │  - attach as executor_request.py_multimodal_data
    ▼
[4] RequestBroadcaster.broadcast                     _torch/pyexecutor/request_utils.py
    │  tp_cp_broadcast / PP send-recv — payload = py_multimodal_data (handle-shaped)
    ▼
[5] ModelEngine._prepare_inputs  (per iteration)     _torch/pyexecutor/model_engine.py
    │  Context (prefill) requests:
    │     MultimodalRuntimeData(mm_token_lengths, mm_token_positions,
    │                           past_seen_token_num, chunk_end_pos, special_token_offsets)
    │     MultimodalParams(multimodal_data=py_multimodal_data, multimodal_runtime=...)
    │     to_device("multimodal_data", "cuda",
    │               pin_memory=prefer_pinned(), non_blocking=True,
    │               target_keywords=model.multimodal_data_device_paths)
    │     pad mrope_position_ids into preallocated mrope_position_ids_padding_cuda
    │  Generation requests (mRoPE models only):
    │     MultimodalParams(...).strip_for_generation()  → keep only mrope_position_deltas
    │     to_device("multimodal_data", "cuda",
    │               target_keywords=["mrope_config.mrope_position_deltas"])
    │  Post-prefill: _strip_py_multimodal_data_post_prefill releases pinned encoder outputs
    ▼
[6] Model.forward(attn_metadata, input_ids, position_ids, multimodal_params=…)
    │  _get_requests_with_mm_data: keep params with pixels OR cached embedding
    │  get_multimodal_embeddings(encoder.forward, mm_params)
    │     ↳ _get_uncached_multimodal_params skips encoder when
    │       multimodal_data["multimodal_embedding"] already populated (chunked-prefill cache)
    │     ↳ _cache_multimodal_embeddings writes results back per-request
    │  find_input_mm_embeds(...)        slice for current chunk / KV-reuse
    │  prepare_mrope_config(...)        one-shot mrope_rotary_cos_sin per request
    │  fuse_input_embeds(embed_tokens, input_ids, mm_embeds [, extra_embeds=deepstack])
    │  self.llm.forward(inputs_embeds=..., mrope_config=...)
    ▼
logits → sampling (PyExecutor) → tokens
```

**Key invariants:**
- Step [1] is the only place blocking media decode happens, and it happens *concurrently* across one request's media items via `asyncio.gather`.
- Step [2] runs in `asyncio.to_thread`, so HF-processor CPU work never blocks the API event loop.
- Step [2]→[3]→[4] crosses a process / rank boundary; the payload is always handle-shaped because `to_handle` ran in [2].
- Step [5] is the only place per-iteration GPU staging happens; H2D is `non_blocking=True` from pinned host memory.
- Step [6] runs on the compute stream and must be sync-free (Contract 1).

### EPD-disaggregated path

When `@support_multimodal_disaggregated` is set and `TLLM_MULTIMODAL_DISAGGREGATED=1`:

- **Encoder worker:** standalone `tensorrt_llm.llmapi.MultimodalEncoder` (`mm_encoder_only=True`). `_forward_step_mm_encoder_only` returns `{"mm_embeddings", "mrope_position_ids", "mrope_position_deltas"}`. Outputs cross workers via shared-tensor handles in `disaggregated_params.multimodal_embedding_handles`.
- **Prefill+decode worker:** `_is_disagg()` skips constructing `self.mm_encoder`; the input processor's `attach_multimodal_embeddings()` consumes the handle list; for `is_context_only_request` the engine re-clones mrope tensors so the prefill worker owns the IPC handles.

### Templates to study

`modeling_qwen2vl.py` and `modeling_qwen3vl.py` are the canonical references — the former shows HF-passthrough vision (legacy), the latter the fully-ported pattern with deepstack and EPD support. Other examples by modality: `modeling_pixtral.py`, `modeling_llava_next.py`, `modeling_phi4mm.py` (audio), `modeling_mllama.py`, `modeling_gemma3vl.py`, `modeling_hyperclovax.py`, `modeling_mistral_large3.py`. Pick the closest one (modality + LLM family + RoPE variant).

---

## Reuse before you write (the most important rule)

**Default: every common block has a TRT-LLM implementation; compose, don't reimplement.** Hand-rolled `nn.Linear` / `nn.LayerNorm` / `nn.MultiheadAttention` will silently work in fp16/bf16 single-GPU eager and silently break under quantization, TP, attention-backend selection, KV cache, and CUDA graphs. The reuse path is also where every future perf improvement lands.

### Compute modules

| Concern | Module | Notes |
|---------|--------|-------|
| Linear | `_torch.modules.linear.Linear` (`TensorParallelMode.{COLUMN,ROW,NONE}`, `mapping=model_config.mapping`, `allreduce_strategy=...`) | Every quant scheme (FP8 / NVFP4 / W4A8 / AWQ / weight-only) and TP path. |
| Embedding / LM head | `_torch.modules.embedding.{Embedding, LMHead}` | Vocab parallelism, masked input. |
| RMSNorm / LayerNorm | `_torch.modules.rms_norm.RMSNorm`, `layer_norm.LayerNorm` | Optimized kernels. |
| Attention (text + vision) | `_torch.modules.attention.Attention` (or `qk_norm_attention.QKNormRoPEAttention` for QK-norm + YARN); vision builds its own `attn_metadata` per encoder forward | All backends (TRTLLM / FlashInfer / FlashAttention / vanilla / sparse) via `AttentionMetadata`. |
| MLA | `_torch.modules.attention.MLA` | DeepSeek-style + DSA. |
| MLP | `_torch.modules.gated_mlp.GatedMLP`, `swiglu.swiglu`, `mlp.MLP` | |
| RoPE | `_torch.modules.rotary_embedding.RotaryEmbedding`; `MRotaryEmbedding` (mRoPE — Qwen-VL `mrope_section`-aware cos/sin slicing, 3D `position_ids`) | mRoPE is for the LLM side; vision-internal 2D RoPE uses `RotaryEmbedding`. |
| MoE | `_torch.modules.fused_moe.create_moe` (see `MOE_DEVELOPER_GUIDE.md`) | TP/EP, multiple kernel backends. |
| Multi-stream overlap | `_torch.modules.multi_stream_utils.maybe_execute_in_parallel` | Overlap vision tower with LLM prefill of other requests. |

### LLM backbone — reuse via AutoModel

```python
import copy
llm_model_config = copy.deepcopy(model_config)
llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]  # inner LLM arch
self.llm = AutoModelForCausalLM.from_config(llm_model_config)
```

If the inner LLM doesn't yet have a TRT-LLM modeling file, finish that text-only onboarding first.

### Vision tower — port preferred, HF passthrough only for legacy

- **Port (preferred):** re-implement vision blocks from `_torch.modules.*` so quantization + TP + attention-backend selection apply. Reference: `Qwen2_5_VisionModel`, `Qwen2_5_VLVisionAttention`, `Qwen2_5_VLPatchMerger`. Vision attention builds its own `attn_metadata` via `prepare_attn_metadata`.
- **HF passthrough (legacy only):** import `transformers.models.<family>.modeling_<family>.<HFVisionTransformer>` directly. Reference: `Qwen2VLModel` keeps `Qwen2VisionTransformerPretrainedModel` because Qwen2-VL is outdated; **always include a code comment justifying it**.

### Weight loading — reuse mappers

If HF prefixes don't match (`model.vision_tower.* → mm_encoder.*`, fused/un-fused QKV, etc.), inherit from a related mapper rather than ad-hoc translation. Reference: `_torch/models/checkpoints/hf/qwen2vl_weight_mapper.py`, `qwen3vl_weight_mapper.py`.

### Don't touch

PyExecutor + the C++ core own `AttentionMetadata`, KV cache, scheduler, sampler, decoder. Your model receives `attn_metadata` and `multimodal_params` as inputs and returns logits — never builds request-level metadata. The only `attn_metadata` you build yourself is the **vision encoder's own**, on the synthetic per-image batch.

---

## Performance contracts

Three rules. Multimodal prefill is long (image/audio tokens balloon sequence length) and media tensors are big (MBs–GBs); the overlap scheduler hides host work behind GPU work only if all three hold.

### Contract 1 — Zero CPU-GPU syncs inside `forward`

A single sync inside `forward` collapses overlap, and per-iteration GPU work is long for VLMs, so stalls compound.

**Banned in `forward` and anything it calls:**
- `.item()`, `.tolist()`, `int(t)`, `bool(t)`, `float(t)` on GPU tensors
- `t.cpu()`, `t.to("cpu")`, any device-crossing read
- Python `if`/`while` on tensor *values* (shape is fine; values are not)
- `torch.nonzero`, `torch.where(mask)`, `torch.unique`, `masked_select` (host syncs)
- `torch.tensor([...], device="cuda")` from a Python list (hidden H2D)
- HF runtime branches (`if pixel_values is None: ...`) that change tensor shapes

**Patterns:**

- **Three-tier mm-token strategy** for `fuse_input_embeds`, in precedence order:
  1. **Precomputed `text_token_indices` + `mm_token_indices`** — fully sync-free; host produces them before the request enters the executor.
  2. **`mm_token_ids` + `torch.isin`** — for bounded mm-token sets.
  3. **OOV sentinel + `input_ids >= vocab_size`** — what `_postprocess` enables (rewrites HF's `image_token_id` to `tllm_multimodal_token_id = vocab_size + 1`); still has one `torch.where`.
- **Static graph for mixed batches.** Don't add `if has_mm:` branches. `find_input_mm_embeds` returns input unchanged when runtime is None; `fuse_input_embeds` returns `(input_ids, None)` when `mm_embeds == []` — preserve that contract.
- **mRoPE: precompute once at request boundary.** Input processor produces `mrope_position_ids/deltas` on CPU (`.to("cpu").clone()`). Engine pads into a preallocated CUDA buffer per iteration. Model's `prepare_mrope_config` concatenates `mrope_rotary_cos_sin` once at model entry. Per-layer attention reads pre-sliced `(cos, sin)`.

**Audit:** grep for the banned constructs; run one prefill iteration with `torch.cuda.set_sync_debug_mode("warn")` and confirm zero warnings from your model.

### Contract 2 — Preprocessing on CPU, async, server-side

CPU-bound work (decode / resize / normalize / mel-spectrogram / frame extraction) must not compete with GPU work, block the request loop, or serialize across requests.

- HF AutoProcessor + image_processor + tokenizer run inside `BaseMultimodalInputProcessor.__call__` — *not* in the model worker.
- URL/bytes media goes through `async_load_image` / `async_load_video` / `async_load_audio` (all wrap blocking decode in `asyncio.to_thread`). Never call `PIL.Image.open(...).load()` / `cv2.VideoCapture` / `soundfile.read` synchronously on the request hot path.
- Pin host tensors before H2D with `prefer_pinned()` (False under CC, True otherwise). The engine pins `multimodal_data` automatically via `to_device(..., pin_memory=prefer_pinned())`.
- **Declare `multimodal_data_device_paths`** on the model — list of dotted paths (e.g. `["image.pixel_values", "image.image_grid_thw", "video.pixel_values_videos", "video.video_grid_thw", "multimodal_embedding"]`) telling the engine which fields go to CUDA. Anything not listed stays on CPU.
- Implement `get_text_with_mm_placeholders` + `expand_prompt_token_ids_for_mm` to enable the tokenized+MM fast path (`tokenized_multimodal_process`) — skips redundant detokenization.
- Forward `mm_processor_kwargs` from `inputs.get("mm_processor_kwargs", {})` to the HF processor (callers tune things like video sample rate via this).

### Contract 3 — Large media via shared tensors, never raw pickle

A 1024×1024 fp32 patch tensor is ~12 MB; a video clip can be hundreds of MB. Naive pickle through MPI broadcast turns the leader into the IPC bottleneck.

- **Always use `MultimodalParams.to_handle`/`to_tensor`.** Producer ships handles (CUDA-IPC `REBUILD_CUDA` for GPU, POSIX-shm `REBUILD_CPU` for CPU); consumers rebuild local views via `to_tensor`. See `_torch/shared_tensor/`.
- **Where it crosses ranks:** `RequestBroadcaster._collect_py_objects` collects `py_multimodal_data` and ships it via `dist.broadcast` / `tp_cp_broadcast` / PP send-recv. Payload size = whatever's in `py_multimodal_data` — keep it handle-shaped.
- **Strip after prefill.** `_strip_py_multimodal_data_post_prefill` clears everything except `mrope_config.mrope_position_deltas`. If your model needs to retain something across decode, update `strip_mm_data_for_generation` explicitly.
- **EPD disagg.** `@support_multimodal_disaggregated` + `TLLM_MULTIMODAL_DISAGGREGATED=1`. Encoder runs as a standalone `MultimodalEncoder`; embeddings cross workers as shared tensors. The prefill worker re-clones mrope tensors for `is_context_only_request` so IPC handles outlive the encoder's freed memory (see `model_engine.py`).
- **Hashes are small; broadcast eagerly.** `MultimodalInput.multimodal_hashes` (blake3) drives KV-cache reuse — never substitute raw pixels for them.

**Audit:** payload size in NVTX `broadcast_requests` / `tp_broadcast_requests` ranges should be < 1 MB per rank per request. More means a broadcast leaked raw tensors.

---

## Phases

### Phase 0 — Gather resources

```bash
huggingface-cli download {org}/{model} --exclude "*.safetensors" "*.bin" "*.pt" "*.gguf"
```

Confirm `preprocessor_config.json` and `chat_template.json` are pulled. Verify `AutoProcessor.from_pretrained(model_path)` loads. Estimate **LLM + vision tower** params for VRAM sanity (vision towers are often 0.5–7 B on top of the LLM).

### Phase 1 — Survey existing coverage

Read `config.json`'s `architectures` and `model_type`. If a `_torch/models/modeling_*.py` already claims that architecture via `@register_auto_model`, extend rather than create new. Identify the closest existing multimodal model and note which TRT-LLM modules it reuses.

### Phase 2 — Model wrapper

Create `tensorrt_llm/_torch/models/modeling_{name}.py`:

```python
class {Name}VisionModelBase(nn.Module):
    """Vision tower. Composes _torch.modules.{Attention,Linear,RMSNorm,GatedMLP,RotaryEmbedding}."""
    def forward(self, pixel_values, grid_thw, **kwargs) -> torch.Tensor: ...

class {Name}ModelBase(PreTrainedModel):
    def __init__(self, model_config):
        ...
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config.architectures = ["{InnerLLM}ForCausalLM"]
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)
        if not _is_disagg():
            self.mm_encoder = {Name}VisionModelBase(...)
        if self.use_mrope:
            self.init_mrope_embedding(model_config)  # preallocates mrope_position_ids_padding_cuda

    @torch.inference_mode()
    def forward(self, attn_metadata, input_ids=None, position_ids=None,
                input_embeds=None, return_context_logits=False, **kwargs):
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_params = self._get_requests_with_mm_data(multimodal_params)
        mm_embeds = []
        if len(mm_params) > 0 and not _is_disagg():
            mm_embeds = get_multimodal_embeddings(self.mm_encoder.forward, mm_params)
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_params)

        mrope_config = (self.prepare_mrope_config(multimodal_params, attn_metadata.num_contexts)
                        if self.use_mrope else {})
        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens, input_ids, mm_embeds, **kwargs)
        return self.llm.forward(attn_metadata=attn_metadata, input_ids=input_ids,
                                position_ids=position_ids, inputs_embeds=input_embeds,
                                return_context_logits=return_context_logits,
                                mrope_config=mrope_config)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "image.image_grid_thw", "multimodal_embedding"]
```

**Required:**
- Compose from `_torch.modules.*`. No raw `nn.Linear` / `nn.LayerNorm` / `nn.MultiheadAttention`.
- `forward` takes `multimodal_params` via `**kwargs`. **Never** add `pixel_values` / `image_grid_thw` / `attention_mask` as direct args — they live in `multimodal_params.multimodal_data`.
- Deepstack-style features (Qwen3VL): split encoder output into `mm_embed` + `deepstack_embeds`, pass `extra_embeds=deepstack_embeds` to `fuse_input_embeds`.

### Phase 3 — Input processor + dummy builder

Subclass **both** `BaseMultimodalInputProcessor` and `BaseMultimodalDummyInputsBuilder` (colocate in the modeling file). Reference: `Qwen3VLInputProcessorBase`.

`__call__(inputs, sampling_params)` does:

1. Pull `text_prompt`, `mm_data`, `mm_processor_kwargs` from `inputs`.
2. `_preprocess(...)` — HF processor produces `pixel_values` / `pixel_values_videos` / `*_grid_thw` / `input_ids`.
3. Build `multimodal_data` keyed by modality: `{"image": {"pixel_values": ..., "image_grid_thw": ...}, "video": {...}}`.
4. Compute `mrope_config` on **CPU** (`.to("cpu").clone()`) into `multimodal_data["mrope_config"]`. Required even on text-only Qwen-VL prompts — no branch.
5. `_postprocess(input_ids)` rewrites HF's `image_token_id` / `video_token_id` to `tllm_multimodal_token_id = vocab_size + 1` (the OOV sentinel). Skip when `mm_data` is empty.
6. Return `(prompt_token_ids_list, {"multimodal_data": multimodal_data})`.

**Encouraged overrides:** `get_text_with_mm_placeholders(mm_counts)` + `expand_prompt_token_ids_for_mm(prompt_token_ids, num_mm_tokens, ...)` for the tokenized fast path.

**EPD override (if `@support_multimodal_disaggregated`):** `attach_multimodal_embeddings(inputs, multimodal_embedding, sampling_params)` consumes encoder outputs in the prefill+decode worker.

**Decorator stack** — bottom-up application; `register_vision_encoder` requires `register_auto_model` to have run first:

```python
@support_multimodal_disaggregated                              # outermost (after validation)
@register_vision_encoder({Name}VisionModelBase, vlm_base_model=HFVisionTransformerClass)
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
class {Name}Model({Name}ModelBase): ...
```

### Phase 4 — Weight loading

```python
def load_weights(self, weights, weight_mapper):
    if not _is_disagg():
        self.mm_encoder.load_weights(weights)
    weight_mapper = {Name}HfWeightMapper()
    weight_mapper.init_model_and_config(self.llm, self.model_config)
    filtered = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
    self.llm.load_weights(filtered, weight_mapper)
```

Inherit from a related mapper for prefix surgery — don't write a one-off translator.

### Phase 5 — Tests

**Per-model unit test (the main one)** at `tests/unittest/_torch/modeling/test_modeling_<name>.py`. Subclass `TestModelingMultimodal` from `tests/unittest/_torch/modeling/test_modeling_multimodal.py` (an abstract `unittest.TestCase`) and implement six abstract methods: `get_model_config`, `get_trtllm_model_class`, `get_hf_model_class`, `get_weight_mapper_class`, `get_model_type`, `get_model_config_class`. The base class drives a `MultimodalScenario`-parameterized run (modality ∈ `image` / `multiple_image` / `video` / `text` / `mixture_text_image` / `audio`, with optional `use_cuda_graph` / `chunked_prefill` / `kv_cache_reuse`) — comparing TRT-LLM logits to HF reference, exercising the KV cache manager, attn metadata, mrope, fusion path, and CUDA graph capture in one harness. Override `get_scenarios` to declare which combinations apply to your model. Reference: `test_modeling_qwen3vl.py`, `test_modeling_qwen2_5vl.py`, `test_modeling_nemotron_nano_v2_vl.py`. Test data lives under `${LLM_MODELS_ROOT}/multimodals/test_data/`.

**Supplemental utility tests (only if your model exercises new logic in shared utilities)** under `tests/unittest/_torch/multimodal/`: `test_fuse_input_embeds.py`, `test_multimodal_runtime.py`, `test_find_num_image_tokens.py`, `test_external_embedding.py`, `test_share_multiparams.py`, `test_mm_encoder_standalone.py`. Extend the existing tests rather than creating new files when the coverage is generic.

**Accuracy test** at `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py` — subclass `LlmapiAccuracyTestHarness`, set `MODEL_NAME` / `MODEL_PATH` / `MAX_NUM_TOKENS=16384`, run `MMMU` (or `ChartQA`/`ScienceQA`). Reference: `TestQwen2_5_VL_7B`. Wire into `tests/integration/test_lists/test-db/l0_<gpu>.yml`.

### Phase 6 — Docs + serve verification

`docs/source/models/supported-models.md`:
- **Supported Models** table: row alphabetical by architecture class.
- **Multimodal Feature Support Matrix (PyTorch Backend)**: row with columns *Overlap Scheduler / CUDA Graph / Chunked Prefill / Torch Sampler / TLLM C++ Sampler / KV Cache Reuse / Logits Post Processor / EPD Disaggregated Serving / Modality (L+I+V+A)*. Mark `Yes` only what you've verified.

Verify aggregated:

```bash
trtllm-serve <hf_model_id> --backend pytorch --max_num_tokens 16384 --port 8000
```

Send a chat completion with a real image; confirm coherent output. For EPD: run `MultimodalEncoder` and `LLM` as separate process groups; verify embeddings cross via `disaggregated_params.multimodal_embedding_handles`.

### Phase 7 — Pull request

Follow `CONTRIBUTING.md`. Title `[JIRA/NVBUG/None][type] description`, `git commit -s`. Body: one full multimodal prompt → output verbatim, reproduction commands, pytest output verbatim. Trigger CI via `/bot run`.

---

## Pre-PR checklist

**Architecture & registration**
- [ ] Decorator stack in correct order: `@support_multimodal_disaggregated` (outermost, optional) → `@register_vision_encoder` → `@register_auto_model` → `@register_input_processor` (innermost).
- [ ] `forward` takes `multimodal_params` via `**kwargs`; no `pixel_values` / `image_grid_thw` / `attention_mask` direct args.
- [ ] `multimodal_data_device_paths` lists every GPU-resident mm field.

**Module reuse**
- [ ] No raw `nn.Linear` / `nn.LayerNorm` / `nn.MultiheadAttention` / hand-rolled attention or MoE — use `_torch/modules/*`.
- [ ] LLM backbone via `architectures` rewrite + `AutoModelForCausalLM.from_config`.
- [ ] Vision encoder uses TRT-LLM modules — HF passthrough only with a code-comment justification.
- [ ] HF→TRT-LLM weight surgery via a `BaseWeightMapper` subclass under `_torch/models/checkpoints/hf/`.

**Input processor**
- [ ] Subclasses both `BaseMultimodalInputProcessor` and `BaseMultimodalDummyInputsBuilder`.
- [ ] `__call__` runs HF AutoProcessor + tokenizer, builds `multimodal_data` by modality, computes `mrope_config` on CPU, `_postprocess`-rewrites mm token ids to the OOV sentinel.
- [ ] `mm_processor_kwargs` flow-through preserved; tokenized fast-path overrides implemented when feasible.
- [ ] `attach_multimodal_embeddings` implemented if `@support_multimodal_disaggregated`.

**Performance contracts**
- [ ] Grep clean for `.item()` / `.cpu()` / `.tolist()` / `torch.where` / `torch.nonzero` / value-dependent `if` in modeling code (outside fallback paths).
- [ ] `set_sync_debug_mode("warn")` audit on prefill: zero warnings from your model.
- [ ] Async loaders used for URL/bytes inputs.
- [ ] Broadcast payload < 1 MB per rank per request (NVTX `broadcast_requests` / `tp_broadcast_requests`); media crosses ranks only via `to_handle` / `to_tensor`.
- [ ] Decode-iteration `mm_data` is empty (post-prefill strip exercised in e2e test).

**Tests & docs**
- [ ] Per-model unit test at `tests/unittest/_torch/modeling/test_modeling_<name>.py` subclassing `TestModelingMultimodal`; six abstract methods implemented; `get_scenarios()` declares the modality × cuda_graph × chunked_prefill × kv_cache_reuse combinations relevant to your model and they all pass.
- [ ] Mixed-batch scenario (`mixture_text_image`) included and passes against HF reference logits.
- [ ] Accuracy test under `test_llm_api_pytorch_multimodal.py`; entry in `test-db/l0_<gpu>.yml`.
- [ ] Rows added to **Supported Models** + **Multimodal Feature Support Matrix**.
- [ ] `trtllm-serve` round-trip verified with a real image prompt; EPD round-trip verified if applicable.
- [ ] `/bot run` triggered and multimodal stages pass.
