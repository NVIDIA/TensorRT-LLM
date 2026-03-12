---
name: ad-model-onboard
description: Translates a HuggingFace model into a prefill-only AutoDeploy custom model using reference custom ops, validates with hierarchical equivalence tests.
---

# AutoDeploy Model Onboarding

**Input:** HuggingFace model ID. **Output:** prefill-only custom model file + hierarchical tests + summary report.

## Phase 0 — Gather All Resources Upfront
Web/GitHub fetches require user approval and the user may leave. Do ALL network access now and save locally before proceeding.

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
3. **Prefer reusing the existing config class** — if the config can be loaded via `AutoConfig.from_pretrained(model_id)` (either from the installed `transformers` or from files in the HF cache downloaded in Phase 0), import it from `transformers` and use it directly. Do NOT recreate or copy the config class into the modeling file when it is already available.
4. Only if the config is truly not available (not in `transformers` and not bundled with the checkpoint), define a minimal config class in the modeling file and `AutoConfig.register(model_type, ConfigCls, exist_ok=True)`.

## Phase 5 — Model Input Contract
The custom model's forward signature must follow these rules:

1. **Always `input_ids`** — The top-level model always receives `input_ids`. A submodule graph may internally receive `inputs_embeds` (e.g., after the embedding layer), but the exported entry point takes token IDs.
2. **Always `position_ids`** — Vanilla sequential `position_ids` are always provided. **Assert `position_ids is not None`** at the top of the forward method — it is a required input, never optional. Do not include fallback logic to generate `position_ids` from `input_ids` (HF models often do this; strip it). If the model uses a non-standard RoPE variant or custom position encoding, the model must compute it internally on top of the provided vanilla `position_ids`.
3. **Multi-modal inputs** — If the model supports vision/audio/etc., those additional inputs are passed during prefill alongside `input_ids`.
4. **No attention mask, no cache inputs, no HF-runtime features** — Do not accept `attention_mask`, `past_key_values`, `use_cache`, or similar HF-runtime arguments. AD manages masking and caching via its own transforms and runtime.

## Phase 6 — Hierarchical Tests
Create `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_{name}_modeling.py`. Use `test_glm4_moe_lite_modeling.py` as template. **No smoke tests.** Small config (hidden=64, layers=2-3, vocab=1000). Use `pytest.skip` if HF class unavailable.

**HF Reference Strategy:** Equivalence tests compare our custom implementation against the HF reference with identical weights and inputs.
- **If HF modules exist in the installed `transformers`**: import them directly (e.g., `from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM`). Wrap imports in `_get_hf_*_class()` try/except helpers that return `None` on `ImportError`, and use `pytest.skip` when `None`.
- **If HF modules are NOT in the installed `transformers`**: copy the minimal module definitions from the HF `modeling_*.py` source into the test file as standalone reference classes. This keeps tests self-contained without requiring a specific `transformers` version. **Important**: make sure the copy is minimal and strictly faithful to the HF implementation only. Do NOT tweak the functionality of the reference.
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

Invoke the `ad-run-agent` subagent to run the model through AutoDeploy on GPU. Pass it:

Step 1: Reduced num layers
Run with reduced num layers to test the e2e flow for issues and iterate faster.
The generation will be bad in step 1 because we are not loading all layers.

Step 2: Full layers
Run with full num layers. The generation should be coherent in step 2.

- **Model HF ID:** the HuggingFace model-id (or local checkpoint path) used throughout onboarding
- **Description:** a short description of the current state, e.g.:
  - "first try after onboarding"
  - "updated yaml with reduced layers"
  - "changed attention backend to torch_mha"
  - "fixed weight loading hooks"

The model is run via:
```bash
CUDA_VISIBLE_DEVICES=<SELECTED_GPUS> python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry
```
The `--use-registry` flag automatically resolves the model's config from `models.yaml`, so no manual `--args.yaml-extra` is needed. The `ad-run-agent` will determine the required `world_size` from the registry, check GPU availability via `nvidia-smi`, select free GPUs, and wait if not enough are available.

The ad-run-agent will build+run the model, check generation quality, archive logs, and update its worklog.

If the run **fails** or produces **bad generation**:
1. Read the ad-run-agent's worklog and log file to understand the error
2. Fix the issue (model code, registry config yaml, weight hooks, etc.)
3. Re-invoke the ad-run-agent with an updated description reflecting the change (e.g., "retry after fixing RoPE scaling in config")
4. Repeat until the run succeeds with meaningful generation

Do NOT proceed to Phase 10 until the step 2 with full layers reports a successful run with coherent generation.

## Phase 10 — Summary Report
Print (not file) after completion: (1) model overview + unique features, (2) tricky parts needing human review, (3) files created/modified (including any new registry configs), (4) test results table (name | validates | PASS/FAIL), (5) known limitations, (6) reviewer result (PASS + how many review iterations it took), (7) AD end-to-end run result (success/fail, number of iterations, final generation quality), (8) registry entry added/updated in `models.yaml` and any new config YAMLs created.

## Phase 11 — Prepare a Pull Request

**GitHub CLI config:** Before running any `gh` command, confirm which `GH_CONFIG_DIR` to use. The default is `~/.config/gh`, but a different directory may be needed when targeting a fork (e.g., `nv-auto-deploy/TensorRT-LLM` vs `NVIDIA/TensorRT-LLM`). Check if the user has specified a custom `GH_CONFIG_DIR` (e.g., in `CLAUDE.local.md` or environment). If not, **ask the user** before proceeding. Prefix all `gh` commands with: `GH_CONFIG_DIR=<path> gh ...`

Prepare a pull request against `origin` (https://github.com/nv-auto-deploy/TensorRT-LLM) targeting
branch `feat/paperclip_maximizer`. Then, ask the user to provide feedback on the PR and wait for the
user to get back to you when the feedback has been posted. Then continue iterating according to the
user's feedback. For any comment or other post, please prepend your message with "[AGENT]" so that it is clear that this was a coding agent posting the comment.
When you post a PR, make sure to include the results from running `build_and_run_ad.py` with the configs
in the model registry as well as a reproducible command along the lines of
```
python examples/auto_deploy/build_and_run_ad.py --model <MODEL-ID> --use-registry
```
to give a set of instructions for the user to reproduce the test. Also include a detailed pytest
command for the unit tests you added so they can be run by the reviewer as well.

## Key Gotchas
- **Canonical ops first:** Always use `torch.ops.auto_deploy.torch_*` canonical ops whenever one exists for the operation. This is how AD knows what to optimize. Writing manual attention, MoE, RoPE, or normalization in plain PyTorch instead of using the canonical op will prevent AD transforms from working.
- **No `repeat_interleave`:** AD attention ops handle GQA natively. Never repeat K/V heads manually.
- **Lean code:** Every line should serve prefill export. No optional HF features, no dead code paths, no fallback logic.
- **Reuse config classes:** Import from `transformers` or load from checkpoint whenever possible. Only bundle a config class if it truly doesn't exist anywhere.
- **Assert `position_ids`:** Always assert `position_ids is not None` — it is a required input, never optional.
- **Self-contained files only**: Never import from other AD custom models. Each `modeling_{name}.py` is a standalone translation from HF source.
- RoPE buffers: `_ad_` prefix. The `RotaryEmbedding.forward(x, position_ids)` should slice by `position_ids` once and return pre-sliced `(cos, sin)` to all layers. Do NOT pass `position_ids` through to every attention forward — that is wasteful redundant slicing.
- MoE weights: use `nn.ModuleList` per-expert for checkpoint compatibility. Write test-only state_dict converters for HF stacked format.
- `noaux_tc` routers (DeepSeek-V3 style): use vanilla PyTorch (sigmoid + bias + group topk + normalize + scale). AD transforms can replace with fused `trtllm` kernels at deployment time.
- Vision towers are typically **not** exported. Keep vision logic in eager PyTorch and export only the text path unless explicitly requested otherwise.
- Model code and tests must run on CPU. Use only `torch_*` prefixed reference ops in AutoDeploy — never `triton_*`, `flashinfer_*`, or `trtllm_*`.
