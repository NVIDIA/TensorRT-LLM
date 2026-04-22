# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron Nano Omni Status

## Current state

Nemotron Nano Omni vision support is enabled in AutoDeploy well enough to run real multimodal end-to-end flows through `examples/auto_deploy/build_and_run_ad.py` with the user-provided YAML prompt file:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml`

Text-only generation is healthy on the real AD path. Multimodal image generation now also completes successfully and is visually grounded on both reduced-layer and full-layer runs.

## What was fixed

### 1. Real `messages` path support in the custom Nemotron processor

The custom AD input processor originally only handled prebuilt `multi_modal_data`, while `build_and_run_ad.py` sends OpenAI-style `messages`. The Nemotron custom processor now:

- accepts `messages` directly
- preserves structured multimodal content for HF `apply_chat_template(tokenize=False)`
- extracts image/video payloads itself
- converts the request into the same `prompt + multi_modal_data` shape expected by the rest of the Nemotron AD path

### 2. Graph-mode multimodal wrapper handoff

The eager multimodal wrapper previously forwarded `input_ids=None` when calling the exported inner text model with `inputs_embeds`. That caused real runtime failures in graph mode:

- `Expected input at 'input_ids' to be a tensor, but got <class 'NoneType'>`

This was fixed by:

- keeping real `input_ids` for graph-mode calls
- still dropping `input_ids` for the eager text model path, where the Nemotron backbone expects exactly one of `input_ids` or `inputs_embeds`

### 3. HF-faithful prompt processing

The custom Nemotron processor now mirrors HF much more closely:

- keeps the structured multimodal `messages` when rendering the prompt
- preserves HF numbered image markers such as `<image 1><image> <image 2><image>`
- tokenizes the fully expanded prompt once, instead of tokenizing prompt chunks independently
- uses `fix_mistral_regex=True` when initializing the tokenizer and processor

### 4. HF-faithful image preprocessing

The custom processor previously forced the image path to a single tile. That was useful as a smoke simplification, but it did not match HF behavior.

The processor now follows HF image preprocessing semantics:

- dynamic image tiling is preserved for images
- video still uses the single-tile path
- image tensors are no longer cast early in the processor; dtype/device adaptation happens at the vision tower boundary

## Strongest verification

For the real doge/cat prompt, the custom AD input processor now matches the HF processor exactly on the main image-processing outputs:

- `prompt_len_hf = 5170`
- `prompt_len_custom = 5170`
- `num_patches_hf = [13, 7]`
- `num_patches_custom = [13, 7]`
- `pixel_values` shape matches: `(20, 3, 512, 512)`
- `input_ids` match exactly

This is the key result: the Nemotron AD input processor is no longer the obvious mismatch.

## E2E status

### Text-only

Verified earlier:

- reduced-layer text-only e2e completed successfully
- full-layer text-only e2e completed successfully
- full-layer text outputs were coherent enough to treat the text path as healthy

### Multimodal

Verified with real `build_and_run_ad.py` runs:

- reduced-layer multimodal e2e completes
- full-layer multimodal e2e completes
- no current runtime failure in prompt plumbing, export, compile, or multimodal request submission

The multimodal smoke config was raised to allow the HF-equivalent prompt length:

- `examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml`
- `max_seq_len: 8192`

## Current focus

The main image path is now healthy, so the current debugging pass is about hardening merge/runtime correctness rather than rescuing a broken e2e flow.

Current risks worth covering:

- mixed image+video prompts were still rejected in the Nemotron input processor even though the wrapper merge path can handle interleaved `item_types`
- without layout metadata, a mixed image+video fallback would be ambiguous because both modalities use the same injected placeholder token id
- the merge helper should be explicitly regression-tested on interleaved video/image/video span order instead of only image-first layouts

## Step log

### Step 1: refreshed repo guidance and re-opened the merge path

Completed:

- re-read `AGENTS.md`
- re-read `CODING_GUIDELINES.md`
- re-opened the Nemotron custom modeling file for the current merge-path pass
- confirmed the next debugging target is still the merge/runtime path, not the processor

Note:

- local `rg` is currently failing in this environment because its wrapper is trying to call `/usr/bin/bwrap`, which is missing
- for this debugging pass I am falling back to `grep`, `sed`, and direct file reads

### Step 2: confirmed the HF code path is locally available

Completed:

- resolved the Nemotron HF dynamic module metadata with `AutoConfig.from_pretrained(..., trust_remote_code=True)`
- confirmed the HF architecture is `NemotronH_Nano_Omni_Reasoning_V3`
- confirmed the HF dynamic module files are being cached locally under the HuggingFace dynamic module directory

Important finding:

- importing the full HF model class currently fails in this environment because `mamba_ssm` is not installed
- that does not block source inspection, so the next step is a direct file-level comparison of HF `extract_feature()` and merge semantics against our AD implementation

### Step 3: source comparison says the merge semantics are probably not the main bug

Completed:

- inspected the HF `modeling.py` wrapper for `NemotronH_Nano_Omni_Reasoning_V3`
- compared HF `forward()` and `generate()` image-merge logic against our AD wrapper
- inspected our Nemotron wrapper initialization and vision path wiring again

Key finding:

- HF and AD both flatten the text embeddings, identify placeholder positions, and replace those positions with flattened projected vision embeddings
- the larger visible difference is not the merge algorithm itself; it is the vision tower implementation path:
  - HF uses `AutoModel.from_config(config.vision_config, trust_remote_code=True)` and consumes `.features`
  - AD uses the in-repo `RADIOVisionModel`

Next step:

- inspect the HF RADIO implementation and compare its output contract against `tensorrt_llm/_torch/models/modeling_radio.py`

### Step 4: identified the exact HF vision tower target

Completed:

- inspected `config.vision_config` from the real Nemotron HF config
- confirmed the vision tower is not defined in the Nemotron repo itself

Key finding:

- the HF vision config points to `nvidia/C-RADIOv2-H--hf_model.RADIOModel`
- this means the correct comparison target for our AD vision tower is the remote C-RADIO HF implementation, not just the top-level Nemotron wrapper

Next step:

- load the HF C-RADIO dynamic module path locally and compare its output contract against our in-repo `RADIOVisionModel`

### Step 5: the HF C-RADIO class has an extra dependency in this environment

Completed:

- tried resolving the remote C-RADIO HF model class directly from the vision config auto-map

Key finding:

- direct import of the HF `nvidia/C-RADIOv2-H` class currently fails because `timm` is not installed
- this is an environment limitation, not yet evidence of a model bug

Next step:

- download or inspect the remote C-RADIO source files directly and compare their feature-output contract against our in-repo `RADIOVisionModel`

### Step 6: downloaded the remote C-RADIO code for direct inspection

Completed:

- downloaded the code-only snapshot for `nvidia/C-RADIOv2-H`
- inspected the remote `hf_model.py` and model card

Key findings:

- the remote HF class is a dedicated RADIO wrapper, separate from the Nemotron repo
- its public usage contract returns a summary output plus spatial features
- the Nemotron HF wrapper explicitly consumes the spatial `features` branch, not the summary branch

Next step:

- compare the remote RADIO forward path with our in-repo `modeling_radio.py` to verify that our wrapper really returns the same spatial feature tensor contract

### Step 7: the remaining gap may be weight loading rather than merge math

Completed:

- compared the remote C-RADIO `radio_model.py` contract with the in-repo RADIO wrapper at a high level

Current hypothesis:

- the merge semantics themselves still look close to HF
- a stronger candidate is that the AD path may not be loading the RADIO vision weights through the conversion path it expects

Why this looks plausible:

- `tensorrt_llm/_torch/models/modeling_radio.py` exposes a custom `load_weights()` path for RADIO
- if the Nemotron AD wrapper only relies on plain `load_state_dict`, the text model can still work while the vision path remains wrong

Next step:

- inspect the AD/custom-model loading path and compare Nemotron’s vision loading approach against the recent working multimodal custom models

### Step 8: AD really is using plain `load_state_dict`

Completed:

- inspected `tensorrt_llm/_torch/auto_deploy/models/hf.py`

Key finding:

- the AD checkpoint path preloads the checkpoint and then calls `model.load_state_dict(all_weights, strict=False)`
- this means a helper like `RADIOVisionModel.load_weights()` is not automatically used during Nemotron model loading

Implication:

- if the Nemotron checkpoint vision keys do not already line up with the in-repo `RADIOVisionModel` parameter names, the vision tower can silently remain mismatched while text generation still works

Next step:

- compare the real checkpoint key names for `vision_model.*` against the AD model’s `vision_model` state dict names

### Step 9: confirmed a real checkpoint-name mismatch in the vision tower

Completed:

- compared the real Nemotron checkpoint index keys for `vision_model.*` against the in-repo `RADIOVisionModel` state dict keys

Strong evidence:

- checkpoint vision keys: `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`
- AD RADIO state keys: `attn.qkv_proj`, `attn.o_proj`, `mlp.up_proj`, `mlp.down_proj`

### Step 15: found a likely export-time multimodal handoff bug

Completed:

- re-read the `AutoModelForImageTextToTextFactory` export path and the `export_to_gm` transform
- verified that the text submodule export captures whatever kwargs the full wrapper passes into `language_model` during a text-only example run
- reproduced that direct export of `NemotronHForCausalLM` with both `input_ids` and `inputs_embeds` currently fails with:
  - `ValueError: You must specify exactly one of input_ids or inputs_embeds`

Key finding:

- the current Nemotron wrapper strips `input_ids` whenever `inputs_embeds` is present for the eager text-model path
- during the factory's text-only export example, that means the captured submodule kwargs are very likely `input_ids`-only
- the standard multimodal wrapper later tries to feed merged `inputs_embeds` into the compiled text graph, but the exported graph may not have captured that input path at all

Why this matters:

- this would exactly fit the current symptom pattern:
  - text-only generation works
  - multimodal requests run end to end
  - visual grounding is still wrong because the compiled text model can fall back to the placeholder-token path instead of consuming merged image embeddings

Next step:

- patch Nemotron so the text submodule can accept both `input_ids` and `inputs_embeds`, with `inputs_embeds` taking precedence
- stop stripping `input_ids` in the wrapper
- make the text-only wrapper path materialize `inputs_embeds` when `input_ids` is available so submodule export capture sees the multimodal-compatible interface
- add a regression test that exercises the exported text graph with `inputs_embeds`

### Step 16: patched the text-submodule handoff and validated it with focused tests

Completed:

- updated `NemotronHModel.forward()` to allow both `input_ids` and `inputs_embeds`, using `inputs_embeds` when present
- updated the outer Nemotron wrapper to:
  - stop stripping `input_ids` before calling the inner text model
  - materialize `inputs_embeds` on text-only wrapper calls when `input_ids` is available
- added focused unit tests for:
  - text model preferring `inputs_embeds` when both inputs are present
  - text-only wrapper path preserving both `input_ids` and `inputs_embeds`
  - graph-mode multimodal wrapper actually forwarding merged `inputs_embeds`
  - exported text graph preserving the `inputs_embeds` path

Focused verification:

- `bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py -k 'prefers_inputs_embeds_when_both_are_provided or text_only_forward_preserves_input_ids_and_inputs_embeds or multimodal_graphmodule_uses_inputs_embeds or multimodal_graphmodule_keeps_input_ids or full_model_export_with_inputs_embeds'"` passed
- result: `5 passed`

Important evidence from the new regression:

- a direct exported `NemotronHForCausalLM` graph now contains placeholders for:
  - `input_ids`
  - `inputs_embeds`
  - `position_ids`
- shifting `inputs_embeds` changes the exported-graph logits, which confirms the graph no longer ignores the multimodal embedding path

Next step:

- run the broader Nemotron unit test file
- rerun reduced-layer multimodal e2e to see whether grounding improves now that the compiled text graph can actually consume merged embeddings

### Step 17: broader Nemotron unit coverage is clean again

Completed:

- reran the full Nemotron unit test file after stabilizing the new export regression inputs

Verification:

- `bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py"` passed
- result: `32 passed`

Notes:

- the new export regression remains in the suite and now uses real token embeddings plus a small perturbation, which avoids spurious `NaN` behavior from fully random hidden states
- this gives a stronger signal that the text-submodule export path is now multimodal-compatible without destabilizing the rest of the Nemotron test coverage

Next step:

- rerun reduced-layer multimodal e2e with the real YAML prompt file and check whether the image answer improves now that the compiled text graph keeps `inputs_embeds`

### Step 18: reduced-layer multimodal e2e now shows real image grounding

Completed:

- reran the reduced-layer multimodal `build_and_run_ad.py` flow with:
  - `dashboard_default.yaml`
  - `world_size_4.yaml`
  - `nano_omni.yaml`
  - `nano_omni_multimodal_smoke.yaml`
  - `nano_omni_num_hidden_layers_5.yaml`
  - the real `/lustre/.../multimodal_prompts.yaml`

Verification:

- the run completed successfully end to end
- the multimodal answer changed materially compared with the earlier broken outputs

Most important result:

- the generated answer now identifies the images semantically instead of describing repeated text or abstract garbage
- reduced-layer output recognized:
  - the first image as a Shiba Inu dog on a couch
  - the second image as a tabby cat in snow

Interpretation:

- this is strong evidence that the `inputs_embeds` export/capture bug was a real remaining blocker
- the compiled text graph is now consuming the merged multimodal embeddings instead of effectively falling back to placeholder-token text behavior
- remaining output issues in this reduced-layer run are now much more consistent with expected quality loss from the 5-layer smoke configuration than with a fundamentally broken merge path

Next step:

- run the full-layer multimodal e2e and compare answer quality against the reduced-layer result

### Step 19: full-layer multimodal e2e also grounds correctly

Completed:

- reran the real multimodal `build_and_run_ad.py` flow on the full model with:
  - `dashboard_default.yaml`
  - `world_size_4.yaml`
  - `nano_omni.yaml`
  - `nano_omni_multimodal_smoke.yaml`
  - the real `/lustre/.../multimodal_prompts.yaml`

Verification:

- the full-layer run completed successfully end to end
- the generated answer remained visually grounded and consistent with the reduced-layer result

Observed full-layer output signal:

- first image recognized as a Shiba Inu dog on a couch
- second image recognized as a cat in a snowy background

Interpretation:

- the merge/runtime path now appears healthy on the real AD graph path
- the key remaining quality issue is no longer “vision is broken”
- current residuals are response-formatting / concision issues from generation style, not obvious multimodal plumbing failures

Current status:

- text-only AD path is healthy
- multimodal image AD path is now working end to end on both reduced-layer and full-layer runs
- processor parity with HF remains in place
- vision checkpoint loading remap remains in place
- exported text graph now keeps and uses `inputs_embeds`

Suggested next debugging direction if we continue:

- tighten prompt-following quality for image answers
- test additional image prompts and mixed batching/chunked-prefill cases beyond this smoke example
- then move on to explicit video-path validation

### Step 20: hardened the mixed image/video merge contract in code

Completed:

- updated the Nemotron AD input processor to build multimodal spans in prompt order across both image and video placeholders
- generalized multimodal span metadata assembly so hashes, uuids, `item_types`, and special offsets now follow the actual span order instead of assuming all images come before all videos
- added a defensive wrapper-side check that rejects mixed image+video fallback when layout metadata is missing, because placeholder-only merge order would otherwise be ambiguous

Why this matters:

- the current real smoke prompt is image-only, so this risk would not show up in the existing e2e run
- the old code still had a latent correctness gap for future mixed image/video Nemotron requests
- this keeps the custom model self-contained and model-specific, without pushing special handling into `build_and_run_ad.py`

Next step:

- run focused Nemotron regressions for interleaved span order, mixed-modality processor expansion, and the new defensive layout check
- then rerun the full Nemotron unit file

### Step 21: validated the mixed image/video hardening with unit coverage

Completed:

- added a merge-helper regression that exercises interleaved `video -> image -> video` span order directly
- added a wrapper regression that rejects ambiguous mixed image+video fallback when layout metadata is absent
- added an input-processor regression for a mixed image+video prompt and verified that:
  - prompt expansion preserves prompt order
  - layout `item_types` reflect the interleaved span order
  - image and video preprocessing outputs are both emitted in one request payload

Focused verification:

- `bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py -k 'interleaved_multimodal_selection_preserves_prompt_order or mixed_modalities_require_layout_metadata or input_processor_handles_mixed_image_and_video_prompt or chunked_multimodal_embedding_selection or input_processor_handles_messages_with_interleaved_images'"` passed
- result: `5 passed`

Broader verification:

- `bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py"` passed
- result: `35 passed`

Important note from the first focused pass:

- one stale explicit guard in the processor still rejected mixed modalities, and one synthetic test span had the wrong placeholder length
- both were corrected before the final green run, so the current result reflects the actual intended mixed-modality behavior

Next step:

- rerun the real multimodal AD smoke to confirm the image-only grounded path still works after the processor/merge hardening

### Step 22: reduced-layer real AD multimodal smoke still works after hardening

Completed:

- reran the real `build_and_run_ad.py` multimodal smoke with the existing doge/cat YAML prompt and the 5-layer override after the mixed-modality processor/merge changes

Verification:

- `bash -ic "f8 && python examples/auto_deploy/build_and_run_ad.py --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning --args.yaml-extra examples/auto_deploy/model_registry/configs/dashboard_default.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/world_size_4.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni_num_hidden_layers_5.yaml --yaml-extra /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml"` passed

Observed output signal:

- the run completed successfully end to end
- the answer remained visually grounded:
  - first image recognized as a Shiba Inu dog on a couch
  - second image recognized as a tabby cat in snow
- the response is still more verbose than the “10 words or less” request, which remains a prompt-following/style issue rather than a merge-path correctness issue

Current status after this pass:

- image-only multimodal e2e remains healthy
- mixed image+video requests are now supported by the Nemotron input processor instead of being rejected up front
- mixed image+video merge now has explicit correctness coverage and a defensive failure mode when layout metadata is unavailable

Suggested next direction:

- add a higher-level mixed image+video wrapper test that exercises the full `forward()` path with both modalities and layout metadata together
- then move on to a real video or mixed image+video AD smoke once we have a suitable prompt fixture

### Step 23: captured the current full-layer output for `multimodal_prompts.yaml`

Completed:

- reran the full-layer real AD flow with:
  - `dashboard_default.yaml`
  - `world_size_4.yaml`
  - `nano_omni.yaml`
  - `nano_omni_multimodal_smoke.yaml`
  - `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml`

Verification:

- `bash -ic "f8 && python examples/auto_deploy/build_and_run_ad.py --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning --args.yaml-extra examples/auto_deploy/model_registry/configs/dashboard_default.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/world_size_4.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml --yaml-extra /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml"` passed

Observed prompt output:

- `So, let's look at each image. First image: a Shiba Inu dog, close-up with part of its face and body, setting seems indoor with soft colors. Second image: a cat in snow, showing different parts like face, body, paws. Need to describe each in 10 words or less.`
- `First image: Dog on furniture, indoor scene, detailed features. Let's count. "Shiba Inu dog on furniture, close-up with background items."`

Interpretation:

- the full run remains visually grounded on both images
- prompt-following is still imperfect because the model drifts into chain-of-thought-style phrasing instead of clean two-short-caption output
- overlap was only partial (`132` overlapping keys out of `390` checkpoint vision keys)

Interpretation:

- under plain `load_state_dict(strict=False)`, a large fraction of the Nemotron vision tower weights can fail to load silently
- this would explain the current symptom very well: text-only generation stays good while multimodal grounding remains bad

Next step:

- add a Nemotron-specific vision-weight remap in the custom loading hook
- add unit coverage to prove the remap bridges the checkpoint keys to the RADIO model keys

### Step 10: added a vision-weight remap in the Nemotron load hook

Completed:

- updated `modeling_nemotron_nano_omni.py` so the Nemotron load hook now remaps legacy RADIO checkpoint keys before `load_state_dict` applies them
- added a focused unit test that feeds old-style RADIO checkpoint keys and verifies they land on the AD RADIO parameter names

Remapped patterns:

- `.attn.qkv.` -> `.attn.qkv_proj.`
- `.attn.proj.` -> `.attn.o_proj.`
- `.mlp.fc1.` -> `.mlp.up_proj.`
- `.mlp.fc2.` -> `.mlp.down_proj.`
- dropped `vision_model.radio_model.input_conditioner.*` checkpoint stats because the AD RADIO path keeps preprocessing external

Next step:

- run the focused Nemotron unit tests
- then re-check the real checkpoint/key compatibility story and rerun multimodal e2e

### Step 11: focused Nemotron regression tests passed

Verification:

- `bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py -k 'drop_multimodal_weights or keeps_vision_weights_when_enabled or remaps_radio_vision_checkpoint_keys or multimodal_graphmodule_keeps_input_ids or get_image_features_flattens_patch_rows'"` passed
- result: `5 passed, 23 deselected`

Next step:

- re-check the real checkpoint/key overlap after applying the same remap logic
- then rerun multimodal e2e

### Step 12: the remap fully closes the real checkpoint-name gap

Verification:

- applied the new Nemotron remap logic to the real `vision_model.*` checkpoint key set from `model.safetensors.index.json`
- compared the remapped keys against the in-repo `RADIOVisionModel` state dict names

Result:

- remapped checkpoint keys: `388`
- AD vision model keys: `388`
- overlap: `388`
- no remaining checkpoint-only keys
- no remaining model-only keys

Interpretation:

- the vision-weight loading mismatch is no longer just a hypothesis; the new remap bridges the real checkpoint keys exactly to the AD RADIO parameter names

Next step:

- rerun reduced-layer multimodal e2e
- if that looks healthier, rerun full-layer multimodal e2e

### Step 13: reduced-layer multimodal e2e now gets through real vision weight loading

Observed on the real `build_and_run_ad.py` path:

- export completed
- all four ranks reached `stage=weight_load, transform=load_weights`
- all four ranks reported `Checkpoint loading completed`
- the run continued into cache init and graph capture/compile

Why this matters:

- the suspected bug was specifically in the bridge from the Nemotron checkpoint vision keys to the AD RADIO parameter names
- this fix is now exercised in the real distributed path, not only in unit tests

Next step:

- let the reduced-layer run finish and inspect the generated multimodal output
- then decide whether to go directly to the full-layer rerun or make one more targeted correction first

### Step 14: reduced-layer multimodal e2e completed after the remap

Verification:

- the reduced 5-layer multimodal `build_and_run_ad.py` flow completed successfully end to end
- the request processed successfully and returned a multimodal answer instead of failing earlier in the pipeline

Observed output:

- the answer changed relative to the earlier obviously broken outputs, which is consistent with the vision weights now being loaded differently
- however, the content is still not correctly grounded to the real doge/cat images
- example failure mode from this run:
  - first image described as repeated text
  - second image described as a circular target / dartboard-like pattern

Interpretation:

- the vision-weight remap fixed a real loading bug
- but it did not fully solve multimodal grounding
- because this was still the 5-layer smoke run, the next useful check is the full-layer multimodal e2e

Next step:

- rerun full-layer multimodal e2e on the same prompt
- compare the full-layer output against the earlier pre-fix behavior

## Files changed in this phase

- `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_nemotron_nano_omni.py`
- `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py`
- `examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml`

## Useful commands

Targeted tests:

```bash
bash -ic "f8 && pytest -q tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_nemotron_nano_omni_modeling.py -k 'input_processor_preserves_hf_image_tiling or input_processor_handles_messages_with_interleaved_images or multimodal_graphmodule_keeps_input_ids or get_image_features_flattens_patch_rows'"
```

Full-layer multimodal e2e:

```bash
bash -ic "f8 && python examples/auto_deploy/build_and_run_ad.py --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning --args.yaml-extra examples/auto_deploy/model_registry/configs/dashboard_default.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/world_size_4.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni.yaml --args.yaml-extra examples/auto_deploy/model_registry/configs/nano_omni_multimodal_smoke.yaml --yaml-extra /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/TensorRT-LLM/multimodal_prompts.yaml"
```
