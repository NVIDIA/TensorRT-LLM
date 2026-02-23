---
name: ad-model-onboard
description: Translates a HuggingFace model into a prefill-only AutoDeploy custom model using reference custom ops, validates with hierarchical equivalence tests.
---

# AutoDeploy Model Onboarding

**Input:** HuggingFace model ID. **Output:** prefill-only custom model file + hierarchical tests + summary report.

## Phase 0 — Gather All Resources Upfront
Web/GitHub fetches require user approval and the user may leave. Do ALL network access now and save locally before proceeding. Check if the HF modeling code is already available in the local transformers install (find the path via `TRANSFORMERS_MAIN_HOME` env var or `python -c "import transformers; print(transformers.__file__)"`). If the model's code is there, use it directly — no network needed. If not, fetch `config.json` and `modeling_*.py` from HuggingFace now and save to `/tmp/ad_onboard_{model_name}/`. Once you have everything locally, you can work fully offline.

## Phase 1 — Analyze HF Model
Study the locally-available `config.json` and `modeling_*.py` (NOT from `tensorrt_llm/_torch/models/`). Identify attention type (MHA/GQA/MLA), MoE config, RoPE variant, normalization, activation, and any data-dependent ops that break `torch.export` (e.g. `torch.nonzero`, data-conditioned `if`).

## Phase 2 — Write Prefill-Only Model
Create `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_{name}.py`. Use `modeling_glm4_moe_lite.py` as template. Strip: KV cache, training paths, dropout, flash attention variants. Keep: `PreTrainedModel` hierarchy, `ModelOutput` dataclass, minimal forward `(input_ids, position_ids, inputs_embeds=None, **kwargs)`.

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
- RoPE buffers: `_ad_` prefix, return full table (not sliced), slice by `position_ids` downstream.
- MoE weights: use `nn.ModuleList` per-expert for checkpoint compatibility. Write test-only state_dict converters for HF stacked format.
- `noaux_tc` routers (DeepSeek-V3 style) need `trtllm` ops — flag as exception in report. Simple routers: vanilla softmax+topk.
