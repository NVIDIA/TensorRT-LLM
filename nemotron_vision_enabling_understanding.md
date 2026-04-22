# Nemotron Vision Enabling Understanding

## Scope of this note

This note captures my current understanding of how to enable vision support for
`tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_nano_omni.py`
using the recent AutoDeploy multimodal patterns.

I based this on:

- The current local Nemotron AutoDeploy onboarding file:
  `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_nano_omni.py`
- The current local Qwen3.5 AutoDeploy vision implementation:
  `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py`
- The shared AutoDeploy factory/runtime plumbing:
  `tensorrt_llm/_torch/auto_deploy/models/hf.py`
  `tensorrt_llm/_torch/auto_deploy/llm.py`
  `tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py`
  `tensorrt_llm/inputs/registry.py`
  `tensorrt_llm/inputs/multimodal.py`
- The non-AutoDeploy Nemotron multimodal implementation:
  `tensorrt_llm/_torch/models/modeling_nemotron_nano.py`
- The finalized Gemma4 multimodal implementation from local git history:
  `cabf42091f`, `4c811f8329`, `75f91fb1f9`

Important context: the current checked-out `modeling_gemma4.py` in this branch
is text-only, but local history contains the later Gemma4 vision implementation
and follow-up fixes. I used those commits as the Gemma4 reference.

## 1. What Nemotron AutoDeploy does today

The current Nemotron AD onboarding is explicitly text-only.

### Current structure

- `NemotronNanoOmniForConditionalGeneration` owns:
  - `self.language_model = NemotronHForCausalLM(config.llm_config)`
  - `self.vision_model = nn.Module()`
  - `self.mlp1 = nn.Module()`
  - `self.sound_encoder = nn.Module()`
  - `self.sound_projection = nn.Module()`
- It registers `_drop_multimodal_weights`, which drops all checkpoint keys under:
  - `vision_model.`
  - `mlp1.`
  - `sound_encoder.`
  - `sound_projection.`
- It aliases `config.text_config = config.llm_config` only so
  `TextModelExportInfo.from_autoinferred()` can find the text submodule.
- Its `forward()` just asserts `position_ids is not None` and delegates straight
  to `self.language_model(...)`.

### Consequences

- Vision weights are not loaded.
- Audio weights are not loaded.
- No multimodal input processor is registered.
- No multimodal placeholder metadata is registered.
- The wrapper is currently registered via `AutoModelForCausalLMFactory`, not
  `AutoModelForImageTextToTextFactory`.
- The exported object is effectively the full text-only wrapper, not a proper
  eager VLM wrapper plus exported inner text model split.

### Current tests match that text-only design

`tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py`
only checks:

- text-only forwarding
- dropping multimodal weights
- `config.text_config` aliasing
- input embedding delegation
- export of the text backbone behavior

There is no current AD vision-path test coverage for Nemotron.

## 2. The shared AutoDeploy multimodal pattern

The important shared pattern is:

- The top-level VLM wrapper stays eager.
- Only the inner text model is exported.

This is implemented by `AutoModelForImageTextToTextFactory` in
`tensorrt_llm/_torch/auto_deploy/models/hf.py`.

### Export behavior

- `AutoModelForImageTextToTextFactory.get_export_infos()` returns
  `TextModelExportInfo.from_autoinferred(model)`.
- `TextModelExportInfo.from_autoinferred()` finds the first submodule whose
  `config` matches `type(model.config.text_config)`.
- So the VLM wrapper must expose a real inner text submodule whose `config` is
  the text config, and the top-level config must expose `text_config`.

### Why the outer wrapper still matters

The eager wrapper is where model-specific multimodal work happens:

- run vision encoder
- turn vision output into text-hidden-size embeddings
- merge embeddings into the token stream
- compute any multimodal position or mask metadata
- call the exported text model / GraphModule with the exact arguments it needs

### Input processor hook

`tensorrt_llm/_torch/auto_deploy/llm.py` now does:

- build a base `ADInputProcessor`
- if the factory defines `init_input_processor(base)`, use that

This is important because Qwen and Gemma both need model-specific multimodal
input preprocessing that the generic AD processor does not provide.

### Runtime multimodal span plumbing

The executor now forwards request-level multimodal layout metadata to models.

`ad_executor._store_prefill_multimodal_metadata()` materializes tensors such as:

- `mm_item_cu_seqlen`
- `mm_item_types`
- `mm_token_positions`
- `mm_token_lengths`
- `mm_special_offsets_cu_seqlen`
- `mm_special_offsets`
- and chunked-prefill slice helpers like `mm_chunk_flat_start` / `mm_chunk_count`

These come from:

- `multimodal_input` hash/span info
- `py_multimodal_data["layout_metadata"]`

So for a multimodal model to use chunked prefill correctly, its input processor
must populate either the generic fields, or custom layout metadata, or both.

## 3. Qwen3.5 pattern: eager vision wrapper, exported text model, shared lm_head

Qwen3.5 is the clearest reference for a wrapper-driven multimodal AD model.

### Core shape

- `Qwen3_5MoeModel` is the eager multimodal wrapper.
- `Qwen3_5MoeModel.language_model` is the exported text submodule.
- `Qwen3_5MoeForConditionalGeneration` owns top-level `lm_head`, matching HF
  checkpoint layout.
- The top-level wrapper shares that `lm_head` into the inner text model with
  `self.model.language_model.set_lm_head(self.lm_head)`.

This is the key fix for the “wrapper not exported, text model exported” world:

- the exported graph must still contain the LM head
- but the checkpoint layout may keep `lm_head` at the outer wrapper

So Qwen pushes the outer `lm_head` reference into the inner text model before
export.

### What the eager wrapper does

`Qwen3_5MoeModel.forward()` does the multimodal orchestration:

- run vision tower
- split image/video embeddings by item
- select only the chunk-relevant embeddings during chunked prefill
- scatter embeddings into `inputs_embeds`
- compute 3D mRoPE positions for multimodal requests
- call the inner text model with `inputs_embeds` and 3D `position_ids`

### Export implications

Qwen needs a custom `TextExportInfo` because exported `position_ids` are 3D
`(3, B, S)`, not standard 2D.

### Input processor implications

Qwen uses a custom AD input processor because it needs:

- exact multimodal span discovery
- `item_types`
- `special_token_offsets`
- custom `multimodal_input`

It sets `multimodal_hashing_supported = False` because it already builds
`multimodal_input` directly and does not want the generic hashing path to
reconstruct it.

## 4. Gemma4 pattern: eager vision wrapper plus exported text-side multimodal mask op

The final Gemma4 design in local history is the other important reference.

### High-level shape

The finalized Gemma4 pattern is:

- eager outer wrapper remains
- exported inner text model remains
- vision encoder runs in eager wrapper
- some multimodal semantics are pushed into the exported text model via a
  custom op

### lm_head / tying / load-hook pattern

Gemma4 had to solve the exact “outer wrapper not exported” problem.

The final solution was:

- `Gemma4Model.language_model = Gemma4ForCausalLM(config.text_config)`
- `Gemma4ForConditionalGeneration._tied_weights_keys` points at
  `model.language_model.lm_head.weight`
- top-level wrapper remaps checkpoint `lm_head.weight` into
  `model.language_model.lm_head.weight` in a load pre-hook
- the inner text model owns the exported `lm_head`
- the text model re-ties `lm_head.weight` to embeddings after load

This is exactly the kind of handling the user was warning about.

### Vision path in eager wrapper

The eager `Gemma4Model` does:

- instantiate a real `vision_tower`
- instantiate `embed_vision`
- run vision encoder on `pixel_values`
- project vision output into text hidden size
- identify placeholder positions in the token stream
- scatter image features into `inputs_embeds`

For chunked prefill it also:

- splits image features by original image item
- uses `mm_token_positions`, `mm_token_lengths`, and
  `mm_special_offsets{,_cu_seqlen}` to select only the current chunk’s image
  feature slice

### Exported text-side multimodal mask

Gemma4 also needed bidirectional attention within image spans, not pure causal
text attention. Its solution was:

- custom op `auto_deploy::gemma4_multimodal_mask`
- backend lowering op `auto_deploy::gemma4_prepare_multimodal_mask`
- semantic mask registration at model import time

The exported text model consumes multimodal span tensors and emits that mask op.
This is Gemma-specific because Gemma needs semantic attention behavior, not just
embedding injection.

### Input processor and export info

The final Gemma4 implementation also added:

- `Gemma4TextExportInfo` with dynamic shapes for all multimodal span tensors
- `Gemma4ForConditionalGenerationFactory(AutoModelForImageTextToTextFactory)`
- custom processor / tokenizer
- custom AD input processor
- placeholder metadata registration

Its AD input processor:

- builds `multimodal_input` directly from image spans
- emits `layout_metadata` with `special_token_offsets` and `item_types`
- disables the generic hashing path

## 5. What the production Nemotron VLM does outside AutoDeploy

The non-AD Nemotron implementation in
`tensorrt_llm/_torch/models/modeling_nemotron_nano.py` is very helpful because
it shows the real multimodal model structure.

### Model shape

`NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3` own:

- `llm`
- lazily-created `vision_encoder`
- lazily-created `sound_encoder`

The vision path loads real weights from:

- `vision_model.*`
- `mlp1.*`

The audio path loads real weights from:

- `sound_encoder.*`
- `sound_projection.*`

So the current AD file is structurally aligned with the checkpoint prefixes, but
it intentionally stubs all of those modules out today.

### Placeholder/input processing

The production Nemotron input processor registers placeholder metadata for:

- `image`
- `video`
- `audio`

and uses placeholders placed before text.

Its processor expands prompts into:

- image blobs: `img_start + img_context * N + img_end`
- video blobs: per-frame/per-tubelet image-context expansions
- optional audio placeholders when audio is attached to video

It also emits modality-specific `multimodal_data` such as:

- image `pixel_values`
- image `image_sizes` for dynamic resolution
- video `pixel_values`
- video `video_size`
- EVS-related metadata
- audio features

### Merge behavior in the production model

The production VLM does not use a Gemma-style semantic attention mask.
Instead it:

- encodes multimodal inputs into embeddings
- adjusts token IDs for EVS/video pruning if needed
- fuses those multimodal embeddings at specific context-token IDs using
  `fuse_input_embeds`
- then runs the text model causally

This is an important insight for Nemotron AD:

- vision support may not require a Gemma-style custom attention mask at all
- the likely minimum viable vision path is “produce the right multimodal
  embeddings and inject them into `inputs_embeds` at the right placeholder
  tokens”

## 6. What seems required for Nemotron AD vision enablement

My current understanding is that Nemotron vision enablement should follow the
Qwen/Gemma outer-wrapper pattern, but likely with simpler text-model changes
than Gemma.

### A. Move Nemotron to the ImageTextToText factory path

The Nemotron wrapper likely needs to move from:

- `AutoModelForCausalLMFactory`

to:

- `AutoModelForImageTextToTextFactory`

because we now want:

- eager top-level VLM wrapper
- exported inner text model only
- model-specific multimodal input processor hook

### B. Keep a real eager multimodal wrapper

The top-level Nemotron wrapper should remain the orchestrator that:

- accepts multimodal kwargs
- runs vision encoding in eager mode
- projects vision features into the LLM hidden size
- merges them into `inputs_embeds`
- calls the exported text model

This matches Qwen and Gemma, and also matches the existing production Nemotron
pattern conceptually.

### C. Stop dropping vision weights

For vision support, the current `_drop_multimodal_weights()` behavior is wrong
for the vision path.

At minimum, Nemotron will need to stop dropping:

- `vision_model.*`
- `mlp1.*`

Audio may still stay stubbed if we only enable vision first, but that should be
an explicit choice.

### D. Make the exported text model contain the effective lm_head

This is the main “wrapper not exported” constraint.

Current Nemotron structure:

- `NemotronNanoOmniForConditionalGeneration.language_model = NemotronHForCausalLM`
- `NemotronHForCausalLM` already owns `lm_head`

This is actually a nice starting point.

Because the inner text model already owns `lm_head`, Nemotron may not need the
same outer-to-inner `lm_head` sharing trick that Qwen and Gemma needed.

But we still need to verify checkpoint load behavior carefully:

- does the checkpoint already store `language_model.lm_head.weight`?
- if yes, that already matches the exported inner model better than Qwen/Gemma
- if not, a remap hook like Gemma’s may still be needed

From the current Nemotron wrapper comments, the checkpoint key layout is:

- `language_model.lm_head.weight -> self.language_model.lm_head`

So my current expectation is:

- Nemotron likely does not need a Gemma-style top-level `lm_head` remap
- but we still need to preserve the exported-inner-model ownership model and
  confirm tied-weight behavior explicitly

### E. Add a Nemotron-specific AD processor or equivalent processor hookup

The current generic `ADInputProcessor` only has a simple image path and does not
look sufficient for full Nemotron Omni-style multimodal behavior.

At minimum, Nemotron AD likely needs either:

- a custom `init_input_processor(base)` wrapper, or
- a custom processor returned by `init_processor()`, or both

The processor needs to provide enough information for the wrapper/runtime to
know:

- where multimodal spans start/end
- which tokens are placeholder tokens
- which special tokens surround the blobs
- what raw multimodal tensors to pass (`pixel_values`, maybe image sizes, maybe
  video sizes)

If Nemotron image/video prompts use start/end tokens plus repeated context
tokens like the production processor does, then special-token offsets may
matter for chunked feature slicing exactly like they do in Qwen/Gemma.

### F. Add chunked-prefill-safe multimodal feature selection

This is one of the most important lessons from Qwen/Gemma.

It is not enough to compute full-image embeddings and blindly scatter them for
the entire original request, because during chunked prefill the current model
invocation may only cover a subset of the multimodal placeholder span.

Nemotron likely needs logic analogous to:

- Qwen’s `_build_chunked_multimodal_embeds(...)`
- Gemma’s `_build_chunked_image_features(...)`

Specifically:

- split full vision features by multimodal item
- reconstruct which feature tokens belong to the current chunk from
  `input_pos`, `cu_seqlen`, `mm_token_positions`, `mm_token_lengths`, and
  `mm_special_offsets`
- scatter only the currently-visible feature slice into the current chunk’s
  `inputs_embeds`

### G. Nemotron may not need Gemma’s text-side semantic mask machinery

This is the biggest architectural difference I see.

Gemma needed custom bidirectional mask semantics inside image spans.
Nemotron’s production multimodal path appears to work by embedding injection at
context-token positions with normal causal text processing afterward.

If that remains true in AD, then Nemotron may not need:

- custom semantic mask ops
- custom `TextExportInfo` multimodal tensor dynamic shapes
- backend mask lowering registry

Instead, the likely minimum path is:

- eager wrapper handles all vision encoding and embedding injection
- exported text model only needs standard `inputs_embeds` and `position_ids`

One extra simplifying observation: the current Nemotron AD text backbone does
not appear to consume `position_ids` internally at all. So unlike Qwen, there
is no visible need for 3D mRoPE export handling.

That makes Nemotron look closer to:

- “Qwen/Gemma wrapper structure”
- but with a simpler exported text-model contract

## 7. My current best guess for the implementation shape

If I had to sketch the likely target design now, it would be:

1. Keep `NemotronNanoOmniForConditionalGeneration` as the eager top-level VLM
   wrapper.
2. Convert Nemotron registration to `AutoModelForImageTextToTextFactory`.
3. Keep `config.text_config = config.llm_config` so text submodule discovery
   still works.
4. Replace empty `vision_model` / `mlp1` stubs with real AD-side modules that
   can load the checkpoint and produce LLM-sized image embeddings.
5. Keep audio stubbed initially if the goal is vision-only, but do not drop
   vision weights anymore.
6. Add a Nemotron-specific AD input processor so multimodal span metadata and
   raw multimodal tensors are produced in the way the eager wrapper expects.
7. In the eager wrapper:
   - identify image/video placeholder token positions
   - compute full vision embeddings
   - split them per multimodal item
   - slice them per chunk during chunked prefill
   - scatter into `inputs_embeds`
   - call `self.language_model(...)`
8. Keep the exported inner text model as the object containing the effective
   `lm_head`.

## 8. Open questions I would want to resolve before coding

### Open question 1: image only or image + video?

The production Nemotron multimodal processor supports:

- image
- video
- audio

“Enable vision support” could mean:

- image only
- image + video

The production path strongly suggests both image and video are part of “vision”.
If we only do image first, that should be an explicit scoped decision.

### Open question 2: can we reuse the production Nemotron prompt/placeholder scheme directly?

The production model uses:

- image/video/audio placeholders
- start/end special tokens
- repeated context-token placeholders

I expect AD should follow the same token contract so checkpoint behavior and
chat-template behavior stay aligned, but this should be confirmed while wiring
the processor/factory.

### Open question 3: dynamic resolution expectations

The production image path has a dynamic-resolution branch with `image_sizes`.
If the HF Omni processor for Nemotron emits that path in practice, the AD
vision encoder may need to preserve it rather than assume a fixed image shape.

### Open question 4: whether audio should remain dropped for now

The current AD onboarding file is “vision + audio + text” in structure, but the
user only asked for vision enablement. I currently assume:

- vision should be enabled
- audio can stay stubbed for now

but the wrapper/load hooks should make that separation explicit.

## 9. Bottom line

My current understanding is:

- Nemotron AD needs to adopt the modern `ImageTextToText` architecture so the
  eager outer VLM wrapper remains responsible for multimodal orchestration while
  only the inner text model is exported.
- The most important things to get right are:
  - exported-inner-text-model discovery
  - multimodal input processing
  - chunked-prefill-safe feature slicing
  - keeping the effective `lm_head` inside the exported text model
- Nemotron probably does **not** need the Gemma4 semantic-mask machinery,
  because its production multimodal path looks like embedding injection plus
  ordinary causal text processing.
- So the likely Nemotron solution is structurally similar to Qwen/Gemma, but
  simpler on the exported text-model side.
