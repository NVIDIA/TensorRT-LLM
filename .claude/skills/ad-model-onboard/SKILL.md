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

**Step 2 — If not found, clone the HF repo (code only, skip weights):**
```bash
CLONE_DIR="/tmp/ad_onboard_{model_name}"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/{org}/{model} "$CLONE_DIR"
```
`GIT_LFS_SKIP_SMUDGE=1` skips LFS blobs (model weights, binaries) so only code/config files are downloaded. This is far faster and safer than fetching individual files. Once cloned you can work fully offline — read `config.json` and `modeling_*.py` directly from `$CLONE_DIR`.

## Phase 1 — Analyze HF Model
Study the locally-available `config.json` and `modeling_*.py` (NOT from `tensorrt_llm/_torch/models/`). Identify attention type (MHA/GQA/MLA), MoE config, RoPE variant, normalization, activation, and any data-dependent ops that break `torch.export` (e.g. `torch.nonzero`, data-conditioned `if`).

## Phase 2 — Write Prefill-Only Model
Create `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_{name}.py`. Use `modeling_glm4_moe_lite.py` as a **structural template only** (class layout, dataclass outputs, forward signature). Strip: KV cache, training paths, dropout, flash attention variants. Keep: `PreTrainedModel` hierarchy, `ModelOutput` dataclass, minimal forward `(input_ids, position_ids, inputs_embeds=None, **kwargs)`.

**Critical rule: Do NOT import or reuse existing AD custom model code** (e.g. `from .modeling_deepseek import ...`). Every `modeling_{name}.py` must be self-contained. Use the HF source (`$CLONE_DIR/modeling_*.py`) as the source of truth for the model's logic and translate it fresh — even if a structurally similar AD model already exists. This prevents hidden coupling, makes each model auditable on its own, and ensures model-specific quirks are captured correctly.

## Phase 3 — Use Reference Custom Ops Only
Replace HF ops with `torch_*` prefixed AD reference ops. **Never** use `triton_*`/`flashinfer_*`/`trtllm_*` — backend selection happens later in AD transforms. Browse `tensorrt_llm/_torch/auto_deploy/custom_ops/` for all available reference ops and their exact signatures. For vanilla components (RMSNorm, MLP), plain PyTorch is also fine — AD fusion passes replace them.

## Phase 4 — Register
1. Bottom of model file: `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)`.
2. Add import + `__all__` entry in `models/custom/__init__.py`.
3. If config not in installed transformers, bundle config class and `AutoConfig.register(model_type, ConfigCls, exist_ok=True)`.

## Phase 5 — Hierarchical Tests
Create `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_{name}_modeling.py`. Use `test_glm4_moe_lite_modeling.py` as template. **No smoke tests.** Small config (hidden=64, layers=2-3, vocab=1000). Use `pytest.skip` if HF class unavailable.

**Bottom-up levels (each must pass before next):**
1. **Block equivalence** — Test MLP, Attention, MoE, Norm individually: same weights + same input → `torch.testing.assert_close`.
2. **Layer equivalence** — Full decoder layer. If model has heterogeneous layers (dense vs MoE, attention vs SSM), test each type separately.
3. **Full model equivalence** — End-to-end logits comparison. Use a small config with <10 layers that covers the essence of the architecture (e.g., at least one of each layer type).
4. **Export test** — `torch_export_to_gm` with `Dim.DYNAMIC` for batch+seq, verify finite output, test a second shape.

## Phase 6 — Summary Report
Print (not file) after completion: (1) model overview + unique features, (2) tricky parts needing human review, (3) files created/modified, (4) test results table (name | validates | PASS/FAIL), (5) known limitations.

## Key Gotchas
- **Self-contained files only**: Never import from other AD custom models. Each `modeling_{name}.py` is a standalone translation from HF source.
- RoPE buffers: `_ad_` prefix, return full table (not sliced), slice by `position_ids` downstream.
- MoE weights: use `nn.ModuleList` per-expert for checkpoint compatibility. Write test-only state_dict converters for HF stacked format.
- `noaux_tc` routers (DeepSeek-V3 style) need `trtllm` ops — flag as exception in report. Simple routers: vanilla softmax+topk.
- Vision towers are typically **not** exported. Keep vision logic in eager PyTorch and export only the text path unless explicitly requested otherwise.
- Model code and tests must run on CPU. Use only torch reference ops in AutoDeploy (e.g., `torch_rmsnorm`, `torch_mla`, `torch_moe`) and avoid CUDA-only kernels in the modeling path.
