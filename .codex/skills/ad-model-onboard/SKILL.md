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

This pulls config, code, and tokenizer files into the HF cache while skipping large weights. After that, work from the cached snapshot directory reported by the command.

## Phase 1 - Analyze the HF Model

Study the locally available `config.json` and `modeling_*.py`. Do not use `tensorrt_llm/_torch/models/` as the source of truth for new onboarding.

Identify:
- Attention type: MHA, GQA, MLA, or another variant
- MoE structure
- RoPE variant
- Normalization and activation
- Any data-dependent ops that may break `torch.export`, such as `torch.nonzero` or input-dependent branches

## Phase 2 - Write a Prefill-Only Model

Create `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_{name}.py`.

Use `modeling_glm4_moe_lite.py` only as a structural template for class layout, dataclass outputs, and forward signature.

Strip:
- KV cache
- Training paths
- Dropout
- Flash attention variants

Keep:
- `PreTrainedModel` hierarchy
- `ModelOutput` dataclass
- Minimal forward signature of `(input_ids, position_ids, inputs_embeds=None, **kwargs)`

Critical:
- The custom modeling code must match the hierarchy expected by the checkpoint safetensors json.
- Do not import or reuse existing AD custom model code such as `from .modeling_deepseek import ...`.
- Every `modeling_{name}.py` must be self-contained and translated fresh from the HF source.

## Phase 3 - Use Reference Custom Ops Only

Replace HF ops with `torch_*` AD reference ops where needed.

Never use:
- `triton_*`
- `flashinfer_*`
- `trtllm_*`

Browse `tensorrt_llm/_torch/auto_deploy/custom_ops/` for available reference ops and signatures.

For vanilla components such as RMSNorm and MLP, plain PyTorch is acceptable.

## Phase 4 - Register the Model

1. Add `AutoModelForCausalLMFactory.register_custom_model_cls("ConfigClassName", ForCausalLM)` at the bottom of the model file.
2. Add the import and `__all__` entry in `tensorrt_llm/_torch/auto_deploy/models/custom/__init__.py`.
3. If the config is not available in installed `transformers`, bundle the config class and call `AutoConfig.register(model_type, ConfigCls, exist_ok=True)`.

## Phase 5 - Follow the Model Input Contract

The custom model forward signature must obey these rules:

1. Always take `input_ids` at the top-level entry point.
2. Always take `position_ids`. If the model uses a custom position scheme, derive it internally from those vanilla `position_ids`.
3. For multimodal models, pass additional inputs during prefill alongside `input_ids`.
4. Do not accept HF runtime arguments such as `attention_mask`, `past_key_values`, `use_cache`, or similar cache/runtime features.

## Phase 6 - Add Hierarchical Tests

Create `tests/unittest/_torch/auto_deploy/unit/singlegpu/models/test_{name}_modeling.py`.

Use `test_glm4_moe_lite_modeling.py` as the template.

Rules:
- No smoke tests
- Use a small config such as hidden size around 64, 2 to 3 layers, and vocab size around 1000
- Use `pytest.skip` if the HF reference class is unavailable

HF reference strategy:
- If HF modules exist in installed `transformers`, import them directly behind helper functions that return `None` on `ImportError`.
- If they do not exist, copy the minimal reference module definitions from HF `modeling_*.py` into the test file so the test remains self-contained.
- Add test-only weight conversion helpers where HF and custom layouts differ.
- For full-model tests, prefer `load_state_dict` pre-hooks already registered on the custom model when possible.

Numerical comparison:

```python
from _model_test_utils import assert_rmse_close
```

Use `assert_rmse_close` for custom-op-backed equivalence tests. Use `torch.testing.assert_close` only for blocks with identical math.
Important: No smoke tests just check for inf or Nan

Recommended `rmse_ratio_tol` values for bfloat16:
- MoE block: `0.02`
- Decoder layer, MoE layer, or full model: `0.05`
- Attention: `0.10`

Bottom-up levels, in order:
1. Block equivalence
2. Layer equivalence
3. Full model equivalence
4. Export test with `torch_export_to_gm`, `Dim.DYNAMIC` for batch and sequence, finite output checks, and a second shape

## Phase 7 - Run Independent Review

Use the project subagent `ad_onboard_reviewer`.

Pass only:
- Model name
- Path to the model file
- Path to the test file

Do not include your own assessment.

If the reviewer returns FAIL:
1. Read each failed item and cited `file:line`
2. Fix the issue
3. Run `ad_onboard_reviewer` again with the same minimal inputs
4. Repeat until the result is PASS

Do not proceed until the reviewer returns PASS.

## Phase 8 - Run AutoDeploy End to End

Use the project subagent `ad_run_agent`.

Step 1: Reduced num layers
Run with reduced num layer - to test e2e flow for issues and iterate faster. 
The generation will be bad in step 1 because we are not loading all layers.

Step 2: Full layers
Run with full num layers. The generation should be coherent in step 2.


Pass:
- Model HF id
- Config yaml under `examples/auto_deploy/model_registry/configs/`
- Short run description such as `first try after onboarding` or `retry after fixing RoPE scaling`

If the run fails or generation quality is poor:
1. Read the subagent's worklog and archived log
2. Fix the model code, yaml, or weight loading path as needed
3. Run `ad_run_agent` again with an updated description
4. Repeat until the run succeeds with coherent generation

Do not proceed until the step 2 with full layers run succeeds.

## Phase 9 - Print the Summary Report

Print, do not write a file:
1. Model overview and unique features
2. Tricky parts that still need human review
3. Files created and modified
4. Test results table with `name | validates | PASS/FAIL`
5. Known limitations
6. Reviewer result and review iteration count
7. End-to-end AD run result, run iteration count, and final generation quality

## Key Gotchas

- Every custom model file must be self-contained.
- RoPE buffers should use the `_ad_` prefix, return the full table, and be sliced downstream by `position_ids`.
- MoE weights should use `nn.ModuleList` per-expert for checkpoint compatibility.
- `noaux_tc`-style routers should stay in plain PyTorch.
- Vision towers are usually not exported unless explicitly requested.
- Model code and tests must run on CPU.
- Use only torch reference ops in the modeling path.
