---
name: ad-model-onboard
description: >
  Translates a HuggingFace model into a prefill-only AutoDeploy custom model
  using reference custom ops, validates with hierarchical equivalence tests.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy Model Onboarding

**Input:** HuggingFace model ID. **Output:** prefill-only custom model file + hierarchical tests + summary report.

## Phase 0 — Gather All Resources Upfront
Web/GitHub fetches require user approval and the user may leave. Do ALL network access now and save locally before proceeding.

### Step 0 — GPU memory sanity check

Before anything else, check whether the model can fit on the current system.

1. Run `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` to get the total VRAM (in MiB) across all GPUs on the system.
2. Estimate the model's memory footprint from the HuggingFace model card or config (number of parameters × bytes per parameter, e.g. 7B params × 2 bytes = ~14 GB for bfloat16).
3. If the estimated size exceeds total system VRAM, **stop and report this to the user** — do not proceed with onboarding until the user acknowledges and decides how to proceed. Example message: "This model requires ~Xgb but the system only has Ygb across N GPUs. Onboarding is likely to fail at the e2e run stage."

**Step 1 — Check local transformers install first:**
```bash
python -c "import transformers; print(transformers.__file__)"
```
Look for `models/{model_type}/modeling_*.py` under that path. If found, use it directly — no network needed.

**Step 2 — If not found, download the HF repo (code only, skip weights):**
```bash
huggingface-cli download {org}/{model} --exclude "*.safetensors" "*.bin" "*.pt" "*.gguf"
```
This downloads config, code, and tokenizer files into the standard HF cache (`$HF_HOME` or `~/.cache/huggingface/`) while skipping large weight files. Files cached here are automatically found by `transformers.AutoConfig.from_pretrained` and similar APIs — no extra path wiring needed. Once downloaded you can work fully offline — read `config.json` and `modeling_*.py` from the cache snapshot directory printed by the command.

## Phase 1 — Survey Existing Coverage & Analyze HF Model

### Step 1 — Check for existing AD custom modeling code

Before writing anything, check if an AD custom model already covers this architecture:

1. Read the model's `config.json` to find its `model_type` and `architectures` fields.
2. Search `tensorrt_llm/_torch/auto_deploy/models/custom/` for existing `modeling_*.py` files that register the same config class name (grep for the `architectures` value or `model_type`).
3. Also check `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py` for existing registrations.

**If existing code is found:**
- Read it carefully. It may already handle this exact model — in which case no new modeling file is needed, only registry entries and possibly tests.
- If the existing code covers a closely related model in the same family but needs adaptation (e.g., the family added MoE in a newer variant, or changed the attention type), decide whether to **extend** the existing file or create a new one. Prefer extending if the changes are minor; create a new file if the architecture diverges significantly. Report the decision and rationale to the user before proceeding.

**If no existing code is found:** proceed to write a new model file in Phase 2.

### Step 2 — Survey the model family in the registry

Check `examples/auto_deploy/model_registry/models.yaml` for **other models from the same family** (e.g., if asked to onboard `Qwen/Qwen3-8B`, look for `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-235B-A22B`, etc.). Also check HuggingFace for the full set of model sizes/variants in the family.

- **Identify which family members already have registry entries** and which are missing.
- **Identify which family members share the same architecture** (same `model_type` / `architectures` in their config) — these can all use a single modeling file.
- **Plan to onboard the entire family cohesively**: one modeling file + one test file should cover all members that share an architecture. The registry should have entries for all commonly-used sizes.
- Report the family survey findings to the user: which models exist, which are missing, and the proposed plan for covering them all.

### Step 3 — Analyze HF model architecture

Study the locally-available `config.json` and `modeling_*.py` (NOT from `tensorrt_llm/_torch/models/`). Identify attention type (MHA/GQA/MLA), MoE config, RoPE variant, normalization, activation, and any data-dependent ops that break `torch.export` (e.g. `torch.nonzero`, data-conditioned `if`).

## Phase 2 — Write a Lean Prefill-Only Model
Create `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_{name}.py`. Use `modeling_glm4_moe_lite.py` as a **structural template only** (class layout, dataclass outputs, forward signature).

**The goal is a minimal prefill-only model for `torch.export` with AD canonical IR ops.** Keep the code as lean as possible — every line should serve the export path. Do not port HF features that AD doesn't need.

Strip: KV cache, training paths, dropout, flash attention variants, `repeat_interleave`/`repeat_kv` for GQA (AD attention ops handle this natively), fallback logic for generating `position_ids` (assert instead), optional code paths gated on config flags irrelevant to prefill export.

Keep: `PreTrainedModel` hierarchy, `ModelOutput` dataclass, minimal forward `(input_ids, position_ids, inputs_embeds=None, **kwargs)`.

**Critical:** Make sure the custom modeling code nn.Module hierarchy matches what the checkpoint safetensor json expects.

**Critical rule: Do NOT import or reuse existing AD custom model code** (e.g. `from .modeling_deepseek import ...`). Every `modeling_{name}.py` must be self-contained. Use the HF source (`$CLONE_DIR/modeling_*.py`) as the source of truth for the model's logic and translate it fresh — even if a structurally similar AD model already exists. This prevents hidden coupling, makes each model auditable on its own, and ensures model-specific quirks are captured correctly.

## Phase 3 — Use AutoDeploy Canonical Ops (CRITICAL)
**Use `torch.ops.auto_deploy.torch_*` canonical ops WHENEVER POSSIBLE.** These are the IR nodes that AD transforms later replace with optimized backends (triton, flashinfer, trtllm) at deployment time. If a canonical op exists for an operation, you MUST use it — do not reimplement the logic in plain PyTorch.

Available canonical ops (see `tensorrt_llm/_torch/auto_deploy/custom_ops/README.md` for full list):
- **Attention:** `torch_attention`, `torch_attention_sdpa`, `torch_attention_repeat_kv`
- **MLA:** `torch_mla`
- **RoPE:** `torch_rope_with_explicit_cos_sin`, `torch_rope_with_complex_freqs`, `torch_rope_with_qk_interleaving`
- **MoE:** `torch_moe`, `torch_moe_fused`, `torch_moe_router`, `torch_moe_dense_mlp`
- **Normalization:** `torch_rmsnorm`, `torch_rmsnorm_gated`, `torch_l2norm`
- **Linear:** `torch_linear_simple`
- **SSM/Mamba:** `torch_ssm`, `torch_causal_conv1d`
- **FLA:** `torch_gated_delta_rule`
- **Quantization:** `torch_quant_fp8_linear`, `torch_quant_nvfp4_linear`, etc.

**Never** use `triton_*`/`flashinfer_*`/`trtllm_*` — backend selection happens later in AD transforms. Plain PyTorch is acceptable ONLY for operations where no canonical op exists (e.g., simple activation functions, embedding lookups, basic tensor arithmetic). If you find yourself writing manual attention, MoE routing, RoPE, or normalization in plain PyTorch, stop and use the canonical op instead.

**Do NOT use `repeat_interleave` or `repeat_kv` for GQA.** HF reference code often repeats K/V heads to match the Q head count before attention. The AD canonical attention ops (`torch_attention`, `torch_attention_sdpa`) handle GQA natively — they accept Q, K, V with different head counts and do the right thing internally. Manually repeating K/V heads is unnecessary bloat and prevents AD from optimizing the attention path.

## Phase 4 — Register
1. Bottom of model file: `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)`.
2. Add import + `__all__` entry in `models/custom/__init__.py`.
3. **Prefer reusing the existing config class** — if the config can be loaded via `AutoConfig.from_pretrained(model_id)` (either from the installed `transformers` or from files in the HF cache downloaded in Phase 0), import it from `transformers` and use it directly. Do NOT recreate or copy the config class into the modeling file when it is already available. Note: AD's factory already calls `AutoConfig.from_pretrained(model_id, trust_remote_code=True)` and passes the result to your model, so you rarely need to import the config at all — if you find yourself doing so, sanity-check that it's genuinely needed.
4. Only if the config is truly not available (not in `transformers` and not bundled with the checkpoint), define a minimal config class in the modeling file and `AutoConfig.register(model_type, ConfigCls, exist_ok=True)`. A good sanity check: if the E2E test passes without a custom config class, you don't need one — `AutoConfig.from_pretrained` already picked up the right class.

## Phase 5 — Model Input Contract
The custom model's forward signature must follow these rules:

1. **Always `input_ids`** — The top-level model always receives `input_ids`. A submodule graph may internally receive `inputs_embeds` (e.g., after the embedding layer), but the exported entry point takes token IDs.
2. **Always `position_ids`** — Vanilla sequential `position_ids` are always provided. **Assert `position_ids is not None`** at the top of the forward method — it is a required input, never optional. Do not include fallback logic to generate `position_ids` from `input_ids` (HF models often do this; strip it). If the model uses a non-standard RoPE variant or custom position encoding, the model must compute it internally on top of the provided vanilla `position_ids`.
3. **Multi-modal inputs** — If the model supports vision/audio/etc., those additional inputs are passed during prefill alongside `input_ids`.
4. **No attention mask, no cache inputs, no HF-runtime features** — Do not accept `attention_mask`, `past_key_values`, `use_cache`, or similar HF-runtime arguments. AD manages masking and caching via its own transforms and runtime.

## Phase 6 — Hierarchical Tests
Create `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_{name}_modeling.py`. Use `test_glm4_moe_lite_modeling.py` as template. **No smoke tests.** Small config (hidden=64, layers=2-3, vocab=1000). Use `pytest.skip` if HF class unavailable.

**HF Reference Strategy:** Equivalence tests compare our custom implementation against the HF reference with identical weights and inputs. **Use actual HF classes if they exist — prefer importing directly over standalone HF-like implementations for unit tests.** Standalone "reference" implementations are effectively alternative AD IR models and defeat the purpose of the reference test; they also tend to silently agree with whatever bugs exist in the custom model.
- **If HF modules exist in the installed `transformers`**: import them directly (e.g., `from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM`). Wrap imports in `_get_hf_*_class()` try/except helpers that return `None` on `ImportError`, and use `pytest.skip` when `None`.
- **If HF modules are NOT in the installed `transformers`**: copy the minimal module definitions from the HF `modeling_*.py` source into the test file as standalone reference classes. This keeps tests self-contained without requiring a specific `transformers` version or HF cache at test time. **Important**: make sure the copy is minimal and strictly faithful to the HF implementation only. Do NOT tweak the functionality of the reference. The same applies to **config classes** that use `trust_remote_code` (i.e., not available in `transformers`): copy a minimal faithful version into the test file. The modeling file should NOT import the config class — AD loads it at runtime via `AutoConfig.from_pretrained(..., trust_remote_code=True)`. The test-only config copy lets you verify config-wrapping behavior (e.g., structure of state_dict).
- **Weight conversion helpers**: Write test-only helpers for any weight format differences between HF and custom (e.g., RoPE de-interleaving, stacked-to-per-expert MoE weights, gate weight key remapping). For full-model tests, prefer using `load_state_dict` pre-hooks already registered on the custom model.

**Numerical comparison:** For equivalence tests comparing custom ops against HF reference, use the shared `assert_rmse_close` utility from `_model_test_utils`:
```python
from _model_test_utils import assert_rmse_close
```
This computes `rmse(actual - expected) / rmse(expected)` — more robust than per-element `torch.testing.assert_close` since a few outlier elements won't fail the test. Use `torch.testing.assert_close` only for blocks with identical math (e.g., plain MLP with no custom ops).

Recommended `rmse_ratio_tol` values for bfloat16:
- **Identical math** (MLP, Norm): use `torch.testing.assert_close` with tight rtol/atol (1e-3)
- **MoE block** (fused routing): `0.02`
- **Decoder layer / MoE layer / full model**: `0.05`
- **Attention**: `0.10`

**Bottom-up levels (each must pass before next):**
1. **Block equivalence** — Test MLP, Attention, MoE, Norm individually: same weights + same input → `assert_rmse_close` (or `torch.testing.assert_close` for identical-math blocks).
2. **Layer equivalence** — Full decoder layer. If model has heterogeneous layers (dense vs MoE, attention vs SSM), test each type separately.
3. **Full model equivalence** — End-to-end logits comparison. Use a small config with <10 layers that covers the essence of the architecture (e.g., at least one of each layer type).
4. **Export test** — `torch_export_to_gm` with `Dim.DYNAMIC` for batch+seq, verify finite output, test a second shape.

## Phase 7 — Independent Review (MANDATORY)

Invoke the `ad-onboard-reviewer` subagent with ONLY the following information:
- Model name
- Path to the model file created
- Path to the test file created

**Do NOT include your own assessment of correctness. Do NOT summarize what you did.** Let the reviewer read the files and judge independently.

If the reviewer returns **FAIL** on any item:
1. Read the reviewer's specific failure reasons and file:line references
2. Fix each failed item
3. Invoke the reviewer again with the same minimal inputs
4. Repeat until you get a full **PASS**

Do NOT proceed to Phase 8 until the reviewer returns PASS.

## Phase 8 — Create or Update Model Registry Entries (Including Family)

Before running the model end-to-end, ensure it **and all identified family members from Phase 1** have valid entries in the AutoDeploy model registry at `examples/auto_deploy/model_registry/`.

For **each model** (the requested model + any family members identified in Phase 1 Step 2):

1. **Check `examples/auto_deploy/model_registry/models.yaml`** for an existing entry matching the model's HF id.
2. **If the entry is missing**, add it with the appropriate `yaml_extra` list:
   - Always include `dashboard_default.yaml` first.
   - Pick `world_size_N.yaml` based on model size (1 for <2B, 2 for 2-15B, 4 for 20-80B, 8 for 80B+). The `world_size` determines how many GPUs are needed for the run.
   - Add model-specific YAML if the model needs custom settings (e.g., `model_kwargs`, non-default transforms).
3. **If a model-specific config YAML is needed** and doesn't exist, create it under `examples/auto_deploy/model_registry/configs/`. See existing configs for format examples.
4. **If the entry exists but needs changes** (e.g., wrong world_size, missing model-specific config), update it.

Family members that share the same architecture should all use the same modeling code. Different sizes only need different `world_size_N.yaml` entries and maybe different sharding configurations.

See `examples/auto_deploy/model_registry/README.md` for full documentation on the registry format and best practices.

## Phase 9 — AutoDeploy End-to-End Run

### ⚠️ MANDATORY: You MUST use the standalone config YAML with `--args.yaml-extra` ⚠️

**You MUST run the model using the standalone config YAML created in Phase 8. The same YAML will be referenced by the cookbook's `trtllm-serve` command in Phase 11. The command is:**

```bash
CUDA_VISIBLE_DEVICES=<SELECTED_GPUS> python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --args.yaml-extra examples/auto_deploy/model_registry/configs/<model>.yaml
```

**The standalone config YAML under `examples/auto_deploy/model_registry/configs/` is self-contained — it includes all settings needed for running the model (compile backend, batch size, seq len, transforms, world_size, etc.). This is the same YAML that `trtllm-serve --extra_llm_api_options` will use in the cookbook, so validating it here ensures the cookbook works out of the box.**

**If the run FAILS:**
1. **Fix the standalone config YAML** — update settings in `examples/auto_deploy/model_registry/configs/<model>.yaml` and re-run.
2. The standalone config YAML is the source of truth. If it is wrong, fix it. If it is missing settings, add them. The model MUST work via this YAML before you are done.

Invoke the `ad-run-agent` subagent to run the model through AutoDeploy on GPU. Pass it:

Step 1: Reduced num layers
Run with reduced num layers to test the e2e flow for issues and iterate faster.
The generation will be bad in step 1 because we are not loading all layers.

Step 2: Full layers
Run with full num layers. The generation should be coherent in step 2.

- **Model HF ID:** the HuggingFace model-id (or local checkpoint path) used throughout onboarding
- **Standalone config YAML path:** the path to the config YAML under `examples/auto_deploy/model_registry/configs/`
- **Description:** a short description of the current state, e.g.:
  - "first try after onboarding"
  - "updated yaml with reduced layers"
  - "changed attention backend to torch_mha"
  - "fixed weight loading hooks"

The model is run via:
```bash
CUDA_VISIBLE_DEVICES=<SELECTED_GPUS> python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --args.yaml-extra examples/auto_deploy/model_registry/configs/<model>.yaml
```
The `ad-run-agent` will determine the required `world_size` from the config YAML, check GPU availability via `nvidia-smi`, select free GPUs, and wait if not enough are available.

The ad-run-agent will build+run the model, check generation quality, archive logs, and update its worklog.

If the run **fails** or produces **bad generation**:
1. Read the ad-run-agent's worklog and log file to understand the error
2. Fix the issue (model code, **standalone config YAML**, weight hooks, etc.)
3. Re-invoke the ad-run-agent with an updated description reflecting the change (e.g., "retry after fixing RoPE scaling in config")
4. **Always re-run with `--args.yaml-extra`.** Fix the standalone config YAML, don't work around it.
5. Repeat until the run succeeds with meaningful generation

Do NOT proceed to Phase 10 until the step 2 with full layers reports a successful run with coherent generation.

**Important:** The successful E2E run outputs (prompts and generated text) will be needed for the cookbook notebook in Phase 11 and the summary report in Phase 12. Save them.

## Phase 10 — Update Model Support Matrix

After a successful E2E run, update the TensorRT-LLM model support matrix at `docs/source/models/supported-models.md` to include the newly onboarded model.

1. **Read the current support matrix** to understand the format and existing entries.
2. **Add a row to the "Supported Models" table** (the first table in the file) with:
   - `Architecture`: The model's architecture class name (e.g., `MiniMaxM2ForCausalLM`) — use the class name registered in Phase 4.
   - `Model`: The model family/display name (e.g., `MiniMax M2/M2.1/M2.7`).
   - `HuggingFace Example`: A representative HF model ID (e.g., `MiniMaxAI/MiniMax-M2.7`).
   - Place the new row **alphabetically** by architecture class name to keep the table sorted.
3. **If the model is AutoDeploy-only** (i.e., it does NOT have native PyTorch backend support in `tensorrt_llm/_torch/models/`), add a footnote indicating AutoDeploy support with a link to the AD config YAML, following the pattern of existing AD-only models (e.g., `[^N]: Supported via the [AutoDeploy](../features/auto_deploy/auto-deploy.md) backend. See [AD config](../../../examples/auto_deploy/model_registry/configs/<model>.yaml).`).
4. **If the model warrants an entry in the Model-Feature Support Matrix** (second table — typically for key/flagship models), add a row there too. For newly onboarded AD models, most advanced features should be marked `Untested` unless you have verified them. Use existing AD model entries (e.g., `Glm4MoeLiteForCausalLM`) as a reference for which features to mark as supported vs untested.

## Phase 11 — Create AutoDeploy Cookbook

Create an AutoDeploy cookbook notebook for the model, following the pattern of existing cookbooks.

1. **Use `examples/auto_deploy/cookbooks/glm_4.7_flash_trtllm_cookbook.ipynb` as the template.** Copy its structure exactly.
2. **Create the new notebook** at `examples/auto_deploy/cookbooks/{model_name}_trtllm_cookbook.ipynb`, using a snake_case version of the model name (e.g., `minimax_m2.7_trtllm_cookbook.ipynb`).
3. **Adapt all model-specific content:**
   - Title and description: update the model name, HF model ID, and description.
   - Model Resources: update links to the model's HuggingFace card, blog posts, technical reports, API platform, and community links. Search the web or the model's HF card for relevant URLs.
   - Model Highlights: update architecture details (e.g., MoE params, context length, special features like tool calling, interleaved thinking, etc.) from the model card.
   - Prerequisites: update VRAM requirements based on model size and precision.
   - `trtllm-serve` command: update the model ID and use `--extra_llm_api_options` pointing to the **standalone** AD config YAML under `examples/auto_deploy/model_registry/configs/` (e.g., `examples/auto_deploy/model_registry/configs/glm-4.7-flash.yaml`). This is the same standalone config YAML validated in Phase 9 via `build_and_run_ad.py --args.yaml-extra`. It is self-contained — it includes all the settings `trtllm-serve` needs (compile backend, batch size, seq len, transforms, etc.).
   - OpenAI client `MODEL_ID`: update to the correct HF model ID.
   - Evaluation Parameters: update recommended inference parameters from the model's documentation/model card.
   - Additional Resources: update all links to be model-specific.
4. **Do NOT include cell outputs** in the committed notebook — the notebook should be clean with no pre-run outputs, so users run it themselves. (Exception: if the model was already run and outputs were captured during Phase 9, you may include them for reference, but this is optional.)
5. **Verify the notebook is valid JSON** — malformed `.ipynb` files will not render on GitHub or in Jupyter.

## Phase 12 — Summary Report

### ⚠️ MANDATORY: You MUST include ALL raw prompts and generated outputs from the final `build_and_run_ad.py --args.yaml-extra` run ⚠️

Print (not file) after completion:

1. Model overview + unique features
2. Tricky parts needing human review
3. Files created/modified (including any new registry configs)
4. Test results table (name | validates | PASS/FAIL)
5. Known limitations
6. Reviewer result (PASS + how many review iterations it took)
7. AD end-to-end run result (success/fail, number of iterations, final generation quality)
8. Registry entry added/updated in `models.yaml` and any new config YAMLs created
9. **ALL raw prompts and their corresponding generated outputs from the final successful `build_and_run_ad.py --args.yaml-extra` run.** Copy-paste the COMPLETE prompt→output pairs verbatim from the run log. Do NOT summarize, truncate, or paraphrase them. The user needs to see exactly what the model generated to judge quality.
10. Model support matrix update — confirm the row was added to `docs/source/models/supported-models.md` and which footnote (if any) was used.
11. AutoDeploy cookbook created — path to the new notebook file (`examples/auto_deploy/cookbooks/<model>_trtllm_cookbook.ipynb`).

## Phase 13 — Prepare a Pull Request

**GitHub CLI config:** Before running any `gh` command, confirm which `GH_CONFIG_DIR` to use. The default is `~/.config/gh`, but a different directory may be needed when targeting a fork (e.g., `nv-auto-deploy/TensorRT-LLM` vs `NVIDIA/TensorRT-LLM`). Check if the user has specified a custom `GH_CONFIG_DIR` (e.g., in `CLAUDE.local.md` or environment). If not, **ask the user** before proceeding. Prefix all `gh` commands with: `GH_CONFIG_DIR=<path> gh ...`

Prepare a pull request against `upstream` (https://github.com/NVIDIA/TensorRT-LLM) targeting
branch `main`. Then, ask the user to provide feedback on the PR and wait for the
user to get back to you when the feedback has been posted. Then continue iterating according to the
user's feedback. For any comment or other post, please prepend your message with "[AGENT]" so that it is clear that this was a coding agent posting the comment.
When you post a PR, you **MUST** include:
1. **ALL raw prompts and their complete generated outputs** from the final successful `build_and_run_ad.py --args.yaml-extra` run. Copy-paste the COMPLETE prompt→output pairs verbatim — do NOT summarize, truncate, or paraphrase. The reviewer needs to see exactly what the model generated.
2. A reproducible command:
```bash
python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --args.yaml-extra examples/auto_deploy/model_registry/configs/<model>.yaml
```
3. A detailed pytest command for the unit tests you added so they can be run by the reviewer as well. Make sure you have run this pytest command on the latest commit that you are pushing, and include these results in the PR.

### ⚠️ MANDATORY: Re-run and re-post logs on EVERY PR update — NO EXCEPTIONS ⚠️

**Every single time you push changes to the PR — whether it is a new commit, a rebase, an amendment, a fixup, or any other update — you MUST:**

1. **Re-run `build_and_run_ad.py --args.yaml-extra`** using the `ad-run-agent` subagent, exactly as in Phase 9. The code has changed, so previous run results are stale and invalid.
2. **Re-run the full unit test suite** (`pytest <test_file> -v`) for the model's test file created in Phase 6. Previous test results are stale and invalid after any code change.
3. **Post ALL raw output from both runs** as a PR comment:
   - The COMPLETE prompt→output pairs from `build_and_run_ad.py` verbatim — do NOT summarize, truncate, or paraphrase.
   - The COMPLETE pytest output verbatim — every test name, every PASSED/FAILED line, every error traceback if any. Do NOT summarize or cherry-pick.

**This is not optional. There are no exceptions.** Even if the change seems trivial (a typo fix, a comment edit, a formatting change), both runs must be re-executed and the full raw logs must be posted. The reviewer cannot verify correctness without seeing generation output AND test results from the exact code that is currently on the branch.

**Workflow for every PR update cycle:**
1. Make the requested code changes
2. Commit the changes
3. Before pushing, always rebase onto the target branch to check for conflicts: `git fetch upstream && git rebase upstream/main`. If there are conflicts, resolve them before proceeding. Do NOT push without rebasing first — the branch must be up-to-date with the target branch.
4. Push (or force-push if rebase rewrote history)
5. Re-invoke the `ad-run-agent` to run `build_and_run_ad.py --model <MODEL-ID> --args.yaml-extra examples/auto_deploy/model_registry/configs/<model>.yaml` on the updated code
6. Re-run the unit tests: `pytest <test_file> -v`
7. Wait for both runs to complete
8. Post a reply to every PR comment containing:
   - A brief description of what changed in this update
   - The COMPLETE raw prompts and generated outputs from the `build_and_run_ad.py` run
   - The COMPLETE raw pytest output (full verbatim log)
   - The reproducible commands used for both runs
9. Resume polling for new comments (see below)

### ⚠️ MANDATORY: Poll PR for new comments every 5 minutes ⚠️

**After opening the PR and after every PR update you post, you MUST set up a polling loop that checks for new PR comments every 5 minutes.** Do not simply post and walk away — actively monitor the PR for reviewer feedback.

**How to poll:**
```bash
# Fetch all PR comments, sorted newest-first, and check for any posted after your last comment
GH_CONFIG_DIR=<path> gh api "repos/<owner>/<repo>/pulls/<PR_NUMBER>/comments?sort=created&direction=desc&per_page=10"
# Also check issue-level comments (top-level PR comments, not inline review comments)
GH_CONFIG_DIR=<path> gh api "repos/<owner>/<repo>/issues/<PR_NUMBER>/comments?sort=created&direction=desc&per_page=10"
# Also check the PR's review status
GH_CONFIG_DIR=<path> gh pr view <PR_NUMBER> --json reviews,state
```

**Polling loop behavior:**
1. After posting your PR (or posting an update comment), immediately start polling every 5 minutes.
2. On each poll, check for:
   - **New review comments** (inline or top-level) posted after your last comment's timestamp
   - **PR approval status** — check if the PR has been approved
   - **Termination signals** — any comment clearly indicating the agent's work is done (e.g., "LGTM", "looks good, we're done", "no more changes needed", "agent work complete", or similar)
3. If **new actionable comments are found**: stop polling, process the feedback, and execute the full PR update cycle (steps 1–8 above). After posting the update, resume polling.
4. If the **PR is approved** or a **termination signal** is found: stop polling, report to the user that the PR review cycle is complete, and end.
5. If **no new comments** are found: sleep 5 minutes and poll again.

**Do NOT stop polling prematurely.** The loop must continue until the PR is approved or a clear termination signal is received. If polling has been running for an extended period (e.g., >2 hours) with no new activity, inform the user that you are still monitoring and ask if they want you to continue or stop.

## Sharding-aware IR model porting (`modeling_*_ir.py`)

Use this when porting an existing AutoDeploy custom model (`tensorrt_llm/_torch/auto_deploy/models/custom/modeling_*.py`) to explicit sharding hint ops in `modeling_*_ir.py` **in the same directory** (no separate `new_sharding/` tree). The exported FX graph must fully specify how the model should be sharded: the `apply_sharding_hints` transform combines hints with a runtime `DistConfig` for deterministic, node-local sharding.

**Argument reference:** Do not duplicate operator tables here. Refer to the custom op docstrings in `tensorrt_llm/_torch/auto_deploy/custom_ops/` for the complete argument reference (including sharding hints, `tp_mode`, `layer_type`, and which ops accept hints).

### Reference examples (study before porting)

| Original | IR / sharding-aware | Layer types |
|----------|---------------------|-------------|
| `modeling_nemotron_h.py` | `modeling_nemotron_h_ir.py` | Mamba SSM, MHA, SwiGLU MLP, MoE |
| `modeling_qwen3_5_moe.py` | `modeling_qwen3_5_moe_ir.py` | GatedDeltaNet, Gated MHA, SwiGLU MLP, MoE |
| `modeling_mistral.py` | `modeling_mistral_ir.py` | MHA, SwiGLU MLP (simplest) |
| `modeling_deepseek_v2.py` | `modeling_deepseek_v2_ir.py` | MLA, SwiGLU MLP, MoE |

### Step-by-step porting procedure

#### Step 1: Copy the source file

```bash
cp tensorrt_llm/_torch/auto_deploy/models/custom/modeling_foo.py \
   tensorrt_llm/_torch/auto_deploy/models/custom/modeling_foo_ir.py
```

#### Step 2: Update the module docstring and add imports

At the top of the IR file:

```python
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
```

Do **not** add global `SHARD_*` flags. Layer-level control uses the `layer_type` hint on each op and `shard_layers` in YAML.

#### Step 3: Replace linear projections

For every `self.proj(x)` or `nn.Linear` call, use `torch.ops.auto_deploy.torch_linear_simple` with explicit `tp_mode` and `layer_type`. Always set `tp_mode` unconditionally (no `if _s else "none"`). **Rules:** opening projections (Q/K/V/gate/up/in_proj) → `"colwise"`; closing (O/down/out_proj) → `"rowwise"`; tiny outputs (e.g. `shared_expert_gate` dim 1) → `"none"`; MLA latent projections (q_a, kv_a) → `"none"`. For fused weights split later, pass `output_sizes=[...]`. For GQA, use `tp_min_local_shape=self.head_dim` on K/V colwise lines.

#### Step 4: Replace split / chunk after fused colwise projections

Use `torch.ops.auto_deploy.split_with_sizes` with `shardable` / `layer_type` where sizes scale with TP.

#### Step 5: Replace view / reshape with concrete head counts

During `torch.export`, `-1` becomes concrete; after TP, wrong values break. Any reshape whose dimension is a head count that scales with TP must use `torch.ops.auto_deploy.view` with `tp_scaled_dim` set appropriately. Safe cases: flat-to-2D, or `[B,S,-1]` when the input is already correctly sharded.

#### Step 6: Insert `all_reduce`

After every rowwise projection, add `torch.ops.auto_deploy.all_reduce(..., layer_type=...)`. **Parallel branch rule:** when branches merge by addition, use a **single** `all_reduce` after the sum (e.g. MoE routed + shared expert; parallel attention + MLP residual branches).

#### Step 7: Special ops (Conv1d, SSM, GatedDeltaNet, gated RMSNorm)

Add sharding hints on `torch_causal_conv1d`, `torch_ssm`, `torch_gated_delta_rule`, `torch_rmsnorm_gated` per docstrings—typically `shardable` / `output_sizes` / `tp_mode` as required.

#### Step 8: MoE

Pass `layer_type="moe"` into `torch_moe`; `apply_sharding_hints` handles EP/TP.

#### Step 9: Register the IR model

1. Bottom of the IR file: `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)` (same pattern as Phase 4).
2. Add a **side-effect import** in `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py` (e.g. `from . import modeling_foo_ir  # noqa: F401`) and extend `__all__` if you export symbols. Without this import, worker processes may not load your class and `apply_sharding_hints` can report **0 nodes processed**. Do **not** use a separate `register_sharded_models.py` indirection.

#### Step 10: YAML — composable registry pattern

Prefer the model registry (`examples/auto_deploy/model_registry/models.yaml`) and **compose** shared fragments under `examples/auto_deploy/model_registry/configs/`, same as other models: list `dashboard_default.yaml`, the right `world_size_N.yaml`, then a dedicated fragment (e.g. `enable_sharder_ir.yaml`) that holds IR sharding transforms. That fragment should disable legacy sharding passes and enable hint-driven sharding. Registry fragments are deep-merged in `yaml_extra` order (see `DynamicYamlMixInForSettings` in `tensorrt_llm/_torch/auto_deploy/utils/_config.py`); place transform keys under `transforms:` so they merge with `dashboard_default.yaml`. Standalone experiment YAMLs for `build_and_run_ad` may wrap the same fields under a top-level `args:` block matching `LlmArgs`.

Example transform block:

```yaml
# Typical contents for enable_sharder_ir.yaml (registry composable fragment)
transforms:
  export_to_gm:
    num_moe_experts_for_export: 2   # often required when expert count is large (>64)
  detect_sharding:
    stage: sharding
    enabled: false
  sharding_transform_executor:
    stage: sharding
    enabled: false
  apply_sharding_hints:
    stage: sharding
    enabled: true
    run_shape_prop: true
    allreduce_strategy: NCCL
    # shard_layers: ['mha', 'mlp']   # optional selective sharding
  gather_logits_before_lm_head:
    enabled: true
```

Use `world_size: 8` when validating TP head-divisibility. Optional `shard_layers` limits which `layer_type` hints are processed; unset means shard all shardable nodes.

#### Step 11: Validate

Do not report success until a run completes successfully.

1. Prefer `python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry` after adding/updating the registry entry and composable YAMLs (Phase 8–9 style).
2. `apply_sharding_hints` logs should show **`N nodes processed` with N > 0**.
3. If validation fails with infrastructure limits (e.g. head count not divisible by `world_size`), document the assert and compatible sizes; do not "fix" core `sharding.py` / custom op schemas without owner review.
4. If blocked by missing infrastructure support, rename artifacts to `broken_modeling_*_ir.py` / broken YAML and file a short error report for humans (do not silently patch core transforms).

**Layer type strings** (for `layer_type` / `shard_layers`): use `"mha"`, `"mla"`, `"mlp"`, `"moe"`, `"ssm"`, `"delta"`, or `"unknown"` (default; skipped when `shard_layers` is set). Match the conventions used in `apply_sharding_hints` and project enums.

### Layer-specific sharding patterns

**MHA (standard or gated):** `layer_type="mha"`: q/k/v colwise (GQA: `tp_min_local_shape`), `view` with `tp_scaled_dim` for head dim, o rowwise + `all_reduce`. Fused Q+gate interleaved per head: colwise without `output_sizes`; contiguous Q|K|V fused blocks need `output_sizes`.

**SwiGLU MLP:** `layer_type="mlp"`: gate/up colwise, down rowwise + `all_reduce`.

**Mamba / SSM:** `layer_type="ssm"`: in_proj colwise + `output_sizes`, splits shardable, conv1d shardable + `output_sizes`, views, `torch_ssm` shardable, norm gated colwise if weight scales, out rowwise + `all_reduce`.

**GatedDeltaNet:** `layer_type="delta"`: in_proj_qkv with `output_sizes`, other in_projs colwise, conv1d/splits/views as above, `torch_gated_delta_rule` shardable, out rowwise + `all_reduce`.

**MoE + shared expert:** `layer_type="moe"`: router replicated; one `all_reduce` after `routed + shared`, not two.

**MLA (DeepSeek):** `layer_type="mla"`: keep `torch_mla` intact with `shardable=True`—do **not** decompose into separate linears + `torch_attention` (introduces bad `expand`/`view` with concrete head counts). q_a/kv_a latent: `tp_mode="none"`; q_b colwise; `o_proj` rowwise + `all_reduce`.

### Common pitfalls (sharding IR)

1. **Missing `auto_deploy::view` for head reshapes** — concrete shapes from export break after sharding.
2. **Sharding tiny projections** — dim-1 gates: `tp_mode="none"`.
3. **Double `all_reduce` in MoE** — one merge-point reduction for routed + shared.
4. **Cross-layer parameter contamination** — in `_apply_hint_*` handlers using `get_source_nodes()`, restrict with `allowed_ops` so residual links do not pull weights from other layers.
5. **Missing `num_moe_experts_for_export`** for very large expert counts — export can hang.
6. **Decomposing ops that absorb weights** (e.g. `torch_mla`) — use `shardable` + handler instead of splitting into plain linears.
7. **Interleaved vs contiguous fused weights** — interleaved per-head groups: colwise only; contiguous Q|K|V blocks: require `output_sizes`.
8. **Omitting `layer_type` when using `shard_layers`** — `"unknown"` nodes are skipped; set hints explicitly on sharding-aware ops.
9. **`layer_type` on non-hint ops** — do **not** pass `layer_type` to ops that are not designed for sharding hints (e.g. `torch_attention`, `torch_l2norm`, `torch_rope_*`); extra positional args break calls. Confirm in `custom_ops/` docstrings which ops accept hints.
10. **Conditional hint values** — no `if _s else "none"`; use unconditional hints and rely on `shard_layers` / transform config.

### Sharding IR validation checklist (human review)

- `world_size=1`: unsharded path; hints should not break correctness.
- `world_size=2` and `8`: shape checks and coherent output.
- `apply_sharding_hints` node count vs expectation.
- Optional: `shard_layers: ['moe']` to verify selective sharding.

## Key Gotchas
- **Canonical ops first:** Always use `torch.ops.auto_deploy.torch_*` canonical ops whenever one exists for the operation. This is how AD knows what to optimize. Writing manual attention, MoE, RoPE, or normalization in plain PyTorch instead of using the canonical op will prevent AD transforms from working.
- **No `repeat_interleave`:** AD attention ops handle GQA natively. Never repeat K/V heads manually.
- **Lean code:** Every line should serve prefill export. No optional HF features, no dead code paths, no fallback logic.
- **Reuse config classes:** Import from `transformers` or load from checkpoint whenever possible. Only bundle a config class if it truly doesn't exist anywhere.
- **Assert `position_ids`:** Always assert `position_ids is not None` — it is a required input, never optional.
- **Self-contained files only**: Never import from other AD custom models. Each `modeling_{name}.py` is a standalone translation from HF source.
- **RoPE cos/sin: slice ONCE, not per layer.** `_ad_` prefix for RoPE buffers. `RotaryEmbedding.forward(x, position_ids)` MUST slice by `position_ids` once and return pre-sliced `(cos, sin)`. Pass those tensors to all layers. NEVER pass `position_ids` through to each layer/attention forward to re-index — that is redundant compute that bloats the exported graph. See Phase 2 for the full pattern.
- MoE weights: use `nn.ModuleList` per-expert for checkpoint compatibility. Write test-only state_dict converters for HF stacked format.
- `noaux_tc` routers (DeepSeek-V3 style): use vanilla PyTorch (sigmoid + bias + group topk + normalize + scale). AD transforms can replace with fused `trtllm` kernels at deployment time.
- Vision towers are typically **not** exported. Keep vision logic in eager PyTorch and export only the text path unless explicitly requested otherwise.
- Model code and tests must run on CPU. Use only `torch_*` prefixed reference ops in AutoDeploy — never `triton_*`, `flashinfer_*`, or `trtllm_*`.
