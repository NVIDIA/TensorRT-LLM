---
name: ad-model-onboard
description: Translate a Hugging Face causal LM into a prefill-only AutoDeploy custom model, add hierarchical equivalence tests, run the independent onboarding reviewer, and validate the model through an end-to-end AutoDeploy run.
---

# AutoDeploy Model Onboarding

Use this skill when onboarding a new Hugging Face causal LM into AutoDeploy custom models under `tensorrt_llm/_torch/auto_deploy/models/custom/`, along with hierarchical unit tests and end-to-end validation.

Input: Hugging Face model id.
Output: prefill-only custom model file, hierarchical tests, reviewer pass, AD run result, and summary report.

## Phase 0 - Gather Resources Upfront

Fetch all external resources early and save them locally before proceeding. Prefer local sources first.

Step 1 - Check the local `transformers` install first:

```bash
python -c "import transformers; print(transformers.__file__)"
```

Look for `models/{model_type}/modeling_*.py` under that path. If found, use it directly.

Step 2 - If the model code is not present locally, download the HF repo without weights:

```bash
huggingface-cli download {org}/{model} --exclude "*.safetensors" "*.bin" "*.pt" "*.gguf"
```

This pulls config, code, and tokenizer files into the HF cache while skipping large weights. Files cached here are automatically found by `transformers.AutoConfig.from_pretrained` and similar APIs, so no extra path wiring is needed. After that, work from the cached snapshot directory reported by the command.

## Phase 1 - Survey Existing Coverage & Analyze HF Model

### Step 1 - Check for existing AD custom modeling code

Before writing anything, check if an AD custom model already covers this architecture:

1. Read the model's `config.json` to find its `model_type` and `architectures` fields.
2. Search `tensorrt_llm/_torch/auto_deploy/models/custom/` for existing `modeling_*.py` files that register the same config class name (grep for the `architectures` value or `model_type`).
3. Also check `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py` for existing registrations.

**If existing code is found:**
- Read it carefully. It may already handle this exact model — in which case no new modeling file is needed, only registry entries and possibly tests.
- If the existing code covers a closely related model in the same family but needs adaptation (e.g., the family added MoE in a newer variant, or changed the attention type), decide whether to **extend** the existing file or create a new one. Prefer extending if the changes are minor; create a new file if the architecture diverges significantly. Report the decision and rationale to the user before proceeding.

**If no existing code is found:** proceed to write a new model file in Phase 2.

### Step 2 - Survey the model family in the registry

Check `examples/auto_deploy/model_registry/models.yaml` for **other models from the same family** (e.g., if asked to onboard `Qwen/Qwen3-8B`, look for `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-32B`, `Qwen/Qwen3-235B-A22B`, etc.). Also check HuggingFace for the full set of model sizes/variants in the family.

- **Identify which family members already have registry entries** and which are missing.
- **Identify which family members share the same architecture** (same `model_type` / `architectures` in their config) — these can all use a single modeling file.
- **Plan to onboard the entire family cohesively**: one modeling file + one test file should cover all members that share an architecture. The registry should have entries for all commonly-used sizes.
- Report the family survey findings to the user: which models exist, which are missing, and the proposed plan for covering them all.

### Step 3 - Analyze HF model architecture

Study the locally available `config.json` and `modeling_*.py`. Do not use `tensorrt_llm/_torch/models/` as the source of truth for new onboarding.

Identify:
- Attention type: MHA, GQA, MLA, or another variant
- MoE structure
- RoPE variant
- Normalization and activation
- Any data-dependent ops that may break `torch.export`, such as `torch.nonzero` or input-dependent branches

## Phase 2 - Write a Lean Prefill-Only Model

Create `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_{name}.py`.

Use `modeling_glm4_moe_lite.py` only as a structural template for class layout, dataclass outputs, and forward signature.

**The goal is a minimal prefill-only model for `torch.export` with AD canonical IR ops.** Keep the code as lean as possible — every line should serve the export path. Do not port HF features that AD doesn't need.

Strip:
- KV cache
- Training paths
- Dropout
- Flash attention variants
- `repeat_interleave` / `repeat_kv` for GQA — AD canonical attention ops handle GQA natively
- Fallback logic for generating `position_ids` — it is always provided (assert instead)
- Optional code paths gated on config flags that are irrelevant to prefill export

Keep:
- `PreTrainedModel` hierarchy
- `ModelOutput` dataclass
- Minimal forward signature of `(input_ids, position_ids, inputs_embeds=None, **kwargs)`

Critical:
- The custom modeling code must match the nn.Module hierarchy expected by the checkpoint safetensors json.
- Do not import or reuse existing AD custom model code such as `from .modeling_deepseek import ...`.
- Every `modeling_{name}.py` must be self-contained and translated fresh from the HF source.

## Phase 3 - Use AutoDeploy Canonical Ops (CRITICAL)

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

Never use non-torch backends in custom model code:
- `triton_*`
- `flashinfer_*`
- `trtllm_*`

Plain PyTorch is acceptable ONLY for operations where no canonical op exists (e.g., simple activation functions, embedding lookups, basic tensor arithmetic). If you find yourself writing manual attention, MoE routing, RoPE, or normalization in plain PyTorch, stop and use the canonical op instead.

**Do NOT use `repeat_interleave` or `repeat_kv` for GQA.** HF reference code often repeats K/V heads to match the Q head count before attention. The AD canonical attention ops (`torch_attention`, `torch_attention_sdpa`) handle GQA natively — they accept Q, K, V with different head counts and do the right thing internally. Manually repeating K/V heads is unnecessary bloat and prevents AD from optimizing the attention path.

## Phase 4 - Register the Model

1. Add `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)` at the bottom of the model file.
2. Add the import and `__all__` entry in `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py`.
3. **Prefer reusing the existing config class** — if the config can be loaded via `AutoConfig.from_pretrained(model_id)` (either from the installed `transformers` or from files in the HF cache downloaded in Phase 0), import it from `transformers` and use it directly. Do NOT recreate or copy the config class into the modeling file when it is already available.
4. Only if the config is truly not available (not in `transformers` and not bundled with the checkpoint), define a minimal config class in the modeling file and call `AutoConfig.register(model_type, ConfigCls, exist_ok=True)`.

## Phase 5 - Follow the Model Input Contract

The custom model forward signature must obey these rules:

1. Always take `input_ids` at the top-level entry point. A submodule graph may internally use `inputs_embeds`, but the exported entry point always starts from token ids.
2. Always take `position_ids`. **Assert `position_ids is not None`** at the top of the forward method — it is a required input, never optional. Do not include fallback logic to generate position ids from `input_ids` (HF models often do this; strip it). If the model uses a custom position scheme or non-standard RoPE variant, derive it internally from the provided vanilla `position_ids`.
3. For multimodal models, pass additional inputs during prefill alongside `input_ids`.
4. Do not accept HF runtime arguments such as `attention_mask`, `past_key_values`, `use_cache`, or similar cache/runtime features. AD manages masking and caching via its own transforms and runtime.

## Phase 6 - Add Hierarchical Tests

Create `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_{name}_modeling.py`.

Use `test_glm4_moe_lite_modeling.py` as the template.

Rules:
- No smoke tests
- Use a small config such as hidden size around 64, 2 to 3 layers, and vocab size around 1000
- Use `pytest.skip` if the HF reference class is unavailable

HF reference strategy:
- If HF modules exist in installed `transformers`, import them directly behind helper functions that return `None` on `ImportError`, and `pytest.skip` if the reference class is unavailable.
- If they do not exist, copy the minimal reference module definitions from HF `modeling_*.py` into the test file so the test remains self-contained. Keep the copied reference strictly faithful to the HF implementation; do not tweak its behavior.
- Add test-only weight conversion helpers where HF and custom layouts differ, such as RoPE de-interleaving, stacked-to-per-expert MoE weights, or gate-weight key remapping.
- For full-model tests, prefer `load_state_dict` pre-hooks already registered on the custom model when possible.

Numerical comparison:

```python
from _model_test_utils import assert_rmse_close
```

Use `assert_rmse_close` for custom-op-backed equivalence tests. It measures `rmse(actual - expected) / rmse(expected)`, which is more robust than per-element checks when a few outliers exist. Use `torch.testing.assert_close` only for blocks with identical math.
Important: No smoke tests that only check for inf or NaN.

Recommended `rmse_ratio_tol` values for bfloat16:
- Identical math blocks such as plain MLP or norm: prefer `torch.testing.assert_close` with tight tolerances (for example `rtol=1e-3`, `atol=1e-3`)
- MoE block: `0.02`
- Decoder layer, MoE layer, or full model: `0.05`
- Attention: `0.10`

Bottom-up levels, in order:
1. Block equivalence. Test MLP, attention, MoE, and norm individually. If the model has heterogeneous block types, cover each type separately.
2. Layer equivalence. Test the full decoder layer, or each distinct layer type if the architecture mixes dense and MoE or attention and SSM variants.
3. Full model equivalence. Use a small config with fewer than 10 layers that still covers the essential architecture, ideally including at least one of each layer type.
4. Export test with `torch_export_to_gm`, `Dim.DYNAMIC` for batch and sequence, finite output checks, and a second shape.

## Phase 7 - Run Independent Review

Use the project subagent `ad_onboard_reviewer`.

Pass only:
- Model name
- Path to the model file
- Path to the test file

Do not include your own assessment or summary of what you changed. Let the reviewer inspect the files independently.

If the reviewer returns FAIL:
1. Read each failed item and cited `file:line`
2. Fix the issue
3. Run `ad_onboard_reviewer` again with the same minimal inputs
4. Repeat until the result is PASS

Do not proceed until the reviewer returns PASS.

## Phase 8 - Create or Update Model Registry Entries (Including Family)

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

## Phase 9 - Run AutoDeploy End to End

Use the project subagent `ad_run_agent`.

Step 1: Reduced num layers
Run with reduced num layers to test the e2e flow for issues and iterate faster. 
The generation will be bad in step 1 because we are not loading all layers.

Step 2: Full layers
Run with full num layers. The generation should be coherent in step 2.

Pass:
- Model HF id
- Short run description such as `first try after onboarding` or `retry after fixing RoPE scaling`

The model is run via:
```bash
CUDA_VISIBLE_DEVICES=<SELECTED_GPUS> python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry
```
The `--use-registry` flag resolves the model's config from `models.yaml` automatically. The `ad_run_agent` will determine the required `world_size` from the registry, check GPU availability via `nvidia-smi`, select free GPUs, and wait if not enough are available.
The `ad_run_agent` should also archive logs and use them to evaluate generation quality.

If the run fails or generation quality is poor:
1. Read the subagent's worklog and archived log
2. Fix the model code, registry config yaml, or weight loading path as needed
3. Run `ad_run_agent` again with an updated description
4. Repeat until the run succeeds with coherent generation

Do not proceed until the step 2 with full layers run succeeds.

## Phase 10 - Print the Summary Report

Print, do not write a file:
1. Model overview and unique features
2. Tricky parts that still need human review
3. Files created and modified (including any new registry configs)
4. Test results table with `name | validates | PASS/FAIL`
5. Known limitations
6. Reviewer result and review iteration count
7. End-to-end AD run result, run iteration count, and final generation quality
8. Registry entry added/updated in `models.yaml` and any new config YAMLs created

## Phase 11 - Prepare a Pull Request

Before running any `gh` command, check whether a custom `GH_CONFIG_DIR` is already specified in the environment or local developer overrides. If none is specified, use the default `~/.config/gh`.

Prepare a pull request against `origin` targeting branch `feat/paperclip_maximizer`.

When creating the PR:
- Include the results from running `build_and_run_ad.py` with the model registry configs.
- Include a reproducible command such as `python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry`.
- Include the detailed `pytest` command for the unit tests added for the onboarding.

After posting the PR, ask the user to provide feedback once comments are available, then continue iterating based on that feedback.

## Key Gotchas

- **Canonical ops first:** Always use `torch.ops.auto_deploy.torch_*` canonical ops whenever one exists for the operation. This is how AD knows what to optimize. Writing manual attention, MoE, RoPE, or normalization in plain PyTorch instead of using the canonical op will prevent AD transforms from working.
- **No `repeat_interleave`:** AD attention ops handle GQA natively. Never repeat K/V heads manually.
- **Lean code:** Every line should serve prefill export. No optional HF features, no dead code paths, no fallback logic.
- **Reuse config classes:** Import from `transformers` or load from checkpoint whenever possible. Only bundle a config class if it truly doesn't exist anywhere.
- **Assert `position_ids`:** Always assert `position_ids is not None` — it is a required input, never optional.
- Every custom model file must be self-contained.
- RoPE buffers should use the `_ad_` prefix. Slice by `position_ids` once in the rotary embedding path and return pre-sliced `(cos, sin)` to downstream layers instead of re-slicing in every attention block.
- MoE weights should use `nn.ModuleList` per-expert for checkpoint compatibility. Write test-only state-dict converters when HF uses stacked expert weights.
- `noaux_tc`-style routers should stay in plain PyTorch.
- Vision towers are usually not exported unless explicitly requested; keep vision logic in eager PyTorch and export only the text path unless the task explicitly requires more.
- Model code and tests must run on CPU.
- Use only `torch_*` prefixed reference ops in the modeling path — never `triton_*`, `flashinfer_*`, or `trtllm_*`.
