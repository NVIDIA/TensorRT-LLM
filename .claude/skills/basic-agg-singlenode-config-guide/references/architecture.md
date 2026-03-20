# Model-to-Source Mapping

Consult this table to find the primary checked-in sources for each supported model family before selecting or adjusting configs.

| Model family | Primary sources | Notes |
|---|---|---|
| DeepSeek-R1 | `examples/configs/database/`, `examples/configs/curated/deepseek-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md` | MoE/ADP choices are model-specific. Checked-in configs include MTP per SKILL.md's **MTP carveout**. |
| DeepSeek-V3 / V3.2-Exp | `examples/models/core/deepseek_v3/README.md` plus nearby DeepSeek configs | Use the V3 README for V3-family behavior, not the R1 deployment guide. |
| GPT-OSS-120B | `examples/configs/database/openai/gpt-oss-120b/`, `examples/configs/curated/gpt-oss-120b-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` | Uses `moe_config` and `moe_expert_parallel_size`. |
| Qwen3 | `examples/configs/curated/qwen3.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`, `examples/models/core/qwen/` | Exclude `qwen3-disagg-prefill.yaml` for basic-agg scope. |
| Qwen3-Next | `examples/configs/curated/qwen3-next.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-next-on-trtllm.md`, `examples/models/core/qwen/` | Prefer exact-model guide over broader Qwen3 guide. |
| Llama-4 Scout | `examples/configs/curated/llama-4-scout.yaml`, `docs/source/deployment-guide/deployment-guide-for-llama4-scout-on-trtllm.md` | MoE/EP tuning in docs; not a dense-only template. |
| Llama-3.3-70B | `examples/configs/curated/llama-3.3-70b.yaml`, its deployment guide | Use checked-in config/docs, not family-level extrapolation. |
| Kimi-K2 | `examples/configs/curated/kimi-k2-thinking.yaml`, `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` | **`cache_transceiver_config` rule** exception. Maps to `DeepseekV3ForCausalLM`; nearby DeepSeek configs are secondary comparators only. Enables `trust_remote_code: true`. |

If a model has no deployment guide or nearby checked-in config, label any draft as unverified.
