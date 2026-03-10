# Model-to-Source Mapping

Use exact checked-in configs and model-specific docs. Preserve explicit latency / balanced / throughput intent when current repo sources label it. Do **not** infer architecture classes from model names alone.

## Current Source Anchors

| Model family | Primary sources to consult | Notes |
|---|---|---|
| DeepSeek-R1 | `examples/configs/database/`, `examples/configs/curated/deepseek-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md` | Current checked-in configs include model-specific MoE / ADP choices. Default to non-spec scope, but keep exact checked-in `MTP` serve configs for `deepseek-ai/DeepSeek-R1-0528` and `nvidia/DeepSeek-R1-0528-FP4-v2` in scope when the YAML explicitly sets `speculative_config.decoding_type: MTP`. |
| DeepSeek-V3 / DeepSeek-V3.2-Exp | `examples/models/core/deepseek_v3/README.md` plus nearby checked-in DeepSeek configs | Prefer the shared DeepSeek-V3 README for V3-family behavior instead of routing through the R1 deployment guide alone. |
| GPT-OSS-120B | `examples/configs/database/openai/gpt-oss-120b/`, `examples/configs/curated/gpt-oss-120b-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` | Current repo configs/docs use MoE-related fields such as `moe_config` and `moe_expert_parallel_size`. |
| Qwen3 | `examples/configs/curated/qwen3.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`, model docs under `examples/models/core/qwen/` | Read the exact model's checked-in config/docs before tuning. |
| Qwen3-Next | `examples/configs/curated/qwen3-next.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-next-on-trtllm.md`, model docs under `examples/models/core/qwen/` | Prefer the exact-model guide over the broader Qwen3 guide. |
| Llama-4 Scout | `examples/configs/curated/llama-4-scout.yaml`, `docs/source/deployment-guide/deployment-guide-for-llama4-scout-on-trtllm.md` | Current checked-in docs/configs discuss MoE / EP tuning; do not treat it as a dense-only template. |
| Llama-3.3-70B | `examples/configs/curated/llama-3.3-70b.yaml`, its deployment guide | Use current checked-in config/docs rather than family-level extrapolation. |
| Kimi-K2 | `examples/configs/curated/kimi-k2-thinking.yaml`, `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md`, `examples/configs/curated/lookup.yaml` | Treat the current Kimi config as an exact-model `cache_transceiver_config` exception, not as a generic basic-agg template. The curated lookup maps Kimi-K2 to `DeepseekV3ForCausalLM`, so nearby DeepSeek-V3 / R1 configs are only secondary comparators for unverified interpolation. |

## Conservative Rules

- Prefer an exact scenario match over any family heuristic.
- Preserve the user's explicit performance objective (`Min Latency`, `Balanced`, `Max Throughput`, or guide-labeled low-latency / max-throughput variants) when choosing between nearby configs.
- If no exact match exists, choose the nearest checked-in config from the same model family and mark any interpolation as unverified.
- If a same-model config is unlabeled for objective, treat it as a default starting point rather than claiming it satisfies a latency / throughput target.
- If the nearest config contains out-of-scope speculative sections, skip it for this skill by default.
- Exception: keep exact checked-in DeepSeek-R1 serve configs in scope when they explicitly set `speculative_config.decoding_type: MTP`.
- Never infer MTP eligibility from model-family similarity alone, and never generalize this carveout to other speculative modes.
- Treat `cache_transceiver_config` as requiring exact-model verification. It often appears in disaggregated setups, but an exact-model guide may still point to a config that contains it; do not exclude that config solely for that reason, and call out the exception explicitly instead of assuming it is a clean basic-agg default.
- If a model has no deployment guide or nearby checked-in config, stop and label the output unverified instead of inventing an architecture class.
- When docs and configs disagree, prefer the checked-in config for the exact scenario you are following and note the mismatch.
