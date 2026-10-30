# The vision path

> **Verified** by the qwen3.6-35b-a3b-nvfp4 / sm_100 / tp1 target's
> multimodal increment (Qwen3.6-35B-A3B-NVFP4, 27-block ViT, sm_100), and,
> where a paragraph says so, by a second vision-language target on sm_103 (the
> processor hand-off, the splice under block reuse). Everything below carries
> the evidence that established it. A claim here is a measurement on *that*
> checkpoint's shape, not a law.

## Where the runtime's half lives

`tensorrt_llm/inputs/` — `multimodal.py`, `multimodal_data.py`, `registry.py`,
`media_io.py`. Media arrive there already decoded and batched; the input
processor `registry.py` builds is the hand-off point between the runtime's
input handling and a model's own vision path.

Note what is *not* there: the towers themselves live in the engine's per-model
definitions. Read them to understand an interface; never to copy a structure —
the tower's **semantics** come from the checkpoint's HF reference code, like
every other computation in a target.

## The five answers

**1. The runtime hands you decoded media; you own the HF processor call.**
`LLM.generate(TextPrompt(prompt=..., multi_modal_data={"image": [PIL.Image]}))`
reaches your registered input processor's `call_with_text_prompt` with PIL
images already loaded. Turning them into `pixel_values` is yours:
`AutoProcessor.from_pretrained(<the checkpoint path>)` does it, which means the
checkpoint has to carry the processor's own files — `preprocessor_config.json`,
and for the Qwen-VL family `video_preprocessor_config.json` beside it, or the
processor class will not construct. The checkpoint is read exactly as
published, so a checkpoint missing one is a checkpoint to report, not a
directory to patch.
<!-- PORT: verify — out of tree this was solved by symlinking those files into
a target-owned model directory. That directory no longer exists, and no
in-tree target has exercised the multimodal path, so what a target does when a
published checkpoint lacks them is unestablished. -->

**2. How the processor reaches the launcher, and what it must be.**

- **Through the family's routing module, never `@register_input_processor`.**
  `create_input_processor` (`tensorrt_llm/inputs/registry.py`) looks a
  decorated processor up by the model class the built-in zoo maps
  `architectures[0]` to, and a modeling_v2 target is never in the zoo. The
  decorator on a target's class registers a processor nothing asks for; the
  launcher builds `DefaultInputProcessor`, with no placeholder metadata, and
  the engine never learns it serves a multimodal model. Measured on that
  target (sm_103), where the decorator recipe never reached the target.

  The way in is a hook in the routing module the architecture selects
  (`models/<family>/routing.py`): beside `route`, an
  `input_processor(pretrained_config)` returning a `RoutedInputProcessor`, or
  `None` for a checkpoint the target does not serve:

  ```python
  def input_processor(pretrained_config: Any) -> Optional[RoutedInputProcessor]:
      if _shape_misses(pretrained_config):  # the fingerprints `route` holds deployments to
          return None
      # Imported here: a launcher resolving another architecture's processor
      # then loads routing modules only.
      from .<target>.input_processor import MODEL_TYPE, PLACEHOLDER_METADATA, ProcessorCls

      return RoutedInputProcessor(
          processor_cls=ProcessorCls,  # a BaseMultimodalInputProcessor subclass
          model_type=MODEL_TYPE,  # the model_type the checkpoint's config.json publishes
          placeholder_metadata=PLACEHOLDER_METADATA,
      )
  ```

  `create_input_processor` asks the hook (`modeling_v2_input_processor` in the
  package's `_router_index.py`) before the zoo, registers the metadata under
  `model_type`, constructs the class as it constructs a built-in model's, and
  pins content hashing on. It asks in the launcher process, from the checkpoint
  alone and whatever `modeling_v2` is set to, before any LLM argument exists:
  answer from the checkpoint's shape, and supply a processor only for an
  architecture the zoo does not serve, so that `modeling_v2="off"` still
  serves what it served.

  A tree whose `_router_index.py` defines no `modeling_v2_input_processor` has
  no hook yet, and adding it is engine work, a task of its own ahead of the
  target's: `RoutedInputProcessor(processor_cls, model_type,
  placeholder_metadata)` and `modeling_v2_input_processor(pretrained_config)`
  in `_router_index.py`, exported from the package, and
  `create_input_processor` asking it before the zoo with the three effects
  above. That target's onboarding added it as one commit ("modeling_v2: let a
  routing module supply its target's input processor"), with a stub-target
  test and a check that every built-in architecture still resolves as before.
- The processor is keyed by nothing that is `architectures[0]`: the hook
  finds it through the routing module, and its placeholder metadata is
  registered under the `model_type` string the checkpoint publishes, which is
  what prompt builders and evaluators look up. A target that must keep a
  particular architecture name for an unrelated reason (a recurrent-state
  pool, say) keeps it.
- `ProcessorCls` subclasses `BaseMultimodalInputProcessor` and implements
  `call_with_text_prompt` plus the `processor` / `tokenizer` / `config` /
  `dtype` properties. `get_num_tokens_per_image` comes free from the base if
  the HF processor has `_get_num_multimodal_tokens`; `get_mm_token_ids` needs
  an override when the processor exposes no `mm_token_ids` (return a 1-D
  tensor holding the placeholder token id).
- It is constructed **in the launcher process at `LLM(...)` time**, by
  `create_input_processor` → `ModelConfig.from_pretrained(<checkpoint>)` —
  **without the LLM-API's `model_kwargs`**. So any geometry a configuration
  passes through `model_kwargs` reaches the *model* and not the *processor*.
  The only source both halves read is the checkpoint's own `config.json`, and
  there is no target-owned copy to put a number in: keep `model_kwargs` to
  switches, or the two halves will derive the tower's tables from different
  numbers.
  <!-- PORT: verify — out of tree the remedy was to write the geometry into a
  target-owned `config.json`. That product is retired (the checkpoint is read
  exactly as published), so a target whose two halves need a number the
  checkpoint does not publish has an unsolved problem here, not a recipe. -->
- The processor makes `model_engine.is_multimodal` true for every
  configuration of that target, including text-only ones. That is what hands
  the KV-cache manager the vocabulary size block reuse keys media content
  with (`tensorrt_llm/_torch/pyexecutor/_util.py`); measured on that
  target, the text path stays bit-identical. Read `is_multimodal`'s uses again
  when the input path changes.
- A text-only prompt must still work: implement the no-media branch as exactly
  `DefaultInputProcessor`'s `tokenizer.encode(prompt, add_special_tokens=...)`
  so the identity configuration's token ids do not move.

**3. The splice happens in the shell, as `inputs_embeds` — not inside the
model's forward.** This is the trap that costs the most time:

> **`DecoderModelForCausalLM.forward` forwards only the keywords it names.**
> `multimodal_params` sits in `**kwargs` and is dropped. Measured: the core
> forward saw `['spec_metadata']` and nothing else, the splice never ran, and
> every image produced the *same* answer — a boot gate that boots, generates
> fluent text and is completely blind.

So the shell's `forward` reads `kwargs["multimodal_params"]`, runs the tower,
writes its rows over the image placeholder embeddings, and passes the result
down as `inputs_embeds` (with `input_ids=None`). That is upstream's own shape
for this, and it keeps the inherited forward — and with it the text gate
record — untouched.

Make the failure impossible to miss next time: a boot-gate image set whose
images differ from one another (four colours, two shapes, light against dark,
two rendered digits, and one request carrying two images) failed **8 of 10**
with the splice dead and passes with it live. A single image case would have
passed either way. Vary the image *sizes* too — that is what puts more than one
sequence length through the tower's attention call and exercises the ragged
multi-image path.

**4. Which shapes vary, and what that means for capture.** The whole media
path is context-only: generation steps carry no payload (the executor strips
`multimodal_data` down to `mrope_config.mrope_position_deltas` after prefill),
and context batches are not CUDA-graph captured. So the data-dependent shapes
never enter a captured region and the sync-free rules are satisfied by
construction. What *does* need care is the host side:

- **Everything derived from the image grid should be computed in the input
  processor, not the forward.** The rope cos/sin table, the position-grid
  gather indices and weights, the per-image lengths that act as `cu_seqlens`,
  the activation op's tile map, the language model's 3-D mRoPE positions — all
  of it is integer and float arithmetic over metadata. Doing it host-side once
  per request is what leaves the forward with nothing but catalog calls, and
  it is the difference between a forward that satisfies the closed-vocabulary
  rule and one that needs `arange`, `linspace`, `cos` and `sin` in it.
- `MultimodalParams.to_device` moves **every** tensor under `multimodal_data`
  to CUDA unless the model class declares
  `multimodal_data_device_paths = ["image.pixel_values", ...]`. A cache-free
  attention call needs host-resident length vectors, so without that list they
  arrive in device memory. The tower's attention is served by
  `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/attention/thop_attention`
  in its no-KV-cache mode — all four pool arguments `None` — which is the mode
  trtllm's own diffusion stack runs, and which a feasibility pass once missed,
  descoping the feature for a ceiling that was not there.
  <!-- PORT: verify — the in-tree contract's certified surface does not state
  the no-KV-cache mode with all four pool arguments `None`; the claim comes
  from the feasibility skill's record of that incident, not from this
  contract's certification table. -->
- `MultimodalParams.input_ids_start_offset` is the request's first row in the
  flattened batch; each image's placeholder tokens are one contiguous run, so
  a whole image is a slice copy rather than a scatter — no index tensor and no
  host sync. Assert the run is contiguous in the processor.
- **A context chunk may hold part of an image.** Chunked prefill splits a
  prompt into context chunks (at the token budget, and on a hybrid model at
  its state-snapshot points), and with block reuse on a request resumes from
  its reused prefix; any of these can fall inside an image's run. Measured on
  that target with reuse on: an image prefilled in chunks ending
  at 256, 512 and 768, its snapshot points, and a repeated image resumed at
  512, inside its span. So the shell places, per
  chunk, exactly the feature rows of the placeholder positions that chunk
  covers: the positions from the request's `embed_mask_cumsum` over
  `[past_seen_token_num, chunk_end_pos)` (`MultimodalRuntimeData` in
  `tensorrt_llm/inputs/multimodal.py`) plus `input_ids_start_offset`, the rows
  from `get_multimodal_embeddings` and `find_input_mm_embeds`
  (`tensorrt_llm/_torch/models/modeling_multimodal_utils.py`), which hand back
  rows `[num_cached_mm_tokens, + num_mm_tokens_in_chunk)`. The tower runs once
  per request and the engine keeps its rows for the later chunks. An assert
  that a whole image lies inside the chunk fails the first long image; check
  instead that the placed rows, chunk by chunk, are bitwise the whole image's.

**5. The MMMU gate does carry images end to end.** `trtllm-eval mmmu_pro`
builds the prompt through `MultimodalLmEvalWrapper.apply_chat_template` and
submits `multi_modal_data={"image": [...]}` to the serving endpoint; lm-eval's
own VLM adapters are not on the path. Verified by a passing gate on a real
target — 65.3179 ± 1.1446 on `mmmu_pro_standard_10`, n = 1730 — and cheaply
pre-verifiable by the boot-gate image set above.

Two details of that path are protocol, not modelling, and they move the score:

- the wrapper reads **`model_type` from the checkpoint's `config.json`** and
  looks the placeholder metadata up by it. Whatever you register under that
  string decides whether placeholders are bulk-prepended or interleaved at the
  original `<image N>` positions, and whether the chat template sees a string
  or an OpenAI content list.
- so if the anchor was measured on stock trtllm, **register the same
  metadata values the stock model type carries**, or the two runs are not
  comparable. They are readable without touching model code:
  `MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder_metadata("<stock model_type>")`.

## What the target adds

**A second module tree.** The vision tower plus the projector that maps its
output into the language model's hidden size. Declared beside the language
model under the same weight-table rule: `W` rows in the same table's
`WEIGHTS` (`layers=None` or a `layers` callable over the tower's own blocks,
with its own key prefix), `torch.empty`, HF storage layout, every axis read
from the config or pinned by routing.

**A second weight family.** Usually a different layout convention from the
language model's, and commonly **unquantized** while the language model is
quantized — so the manifest's dtype expectations cannot be assumed uniform
across the target for the first time. (Verified on the qwen3.6 checkpoint: 291
`quantized_layers` entries, none of them a vision layer, and no `*_scale`
tensor under `model.visual.` — so every vision projection is a plain bf16 GEMM
with a bias while the language model is FP8 + NVFP4.)

**A splice.** See answer 3.

**And possibly a change to the language model's RoPE.** A VLM whose text model
uses mRoPE is running a *degenerate* mRoPE on text — all three position rows
carry the same value — and a text-only target may legitimately take row 0 and
run plain RoPE. Image spans break that: the three axes genuinely differ. Branch
on the feature flag (a per-engine constant, so the text path stays
bit-identical) rather than on the batch.

## How to check the tower before booting an engine

Do this first. It is an afternoon's work and it converts every later failure
from "somewhere in 27 blocks plus an engine" into "the plumbing".

Load the checkpoint's vision weights, build the HF reference module from the
checkpoint's own `vision_config`, and run both on the same random
`pixel_values` at several grid shapes. **Compare against the reference's own
bf16 rounding, not against an absolute threshold**: a 27-block residual tower
has a noise floor, and on the qwen3.6 tower the target's max deviation from an
fp32 reference (0.035-0.091) was *smaller* than the reference's own bf16
deviation (0.036-0.173) at three of four shapes. A bare `rel < 2%` gate called
that broken three times out of four.

Two construction notes for the reference: build it on `meta`, then
`to_empty(device=...)` and `load_state_dict(..., assign=True)` — and re-derive
any **non-persistent buffer** by hand afterwards (the rotary embedding's
`inv_freq` is absent from the state dict, and `.to()` on a meta buffer raises).
