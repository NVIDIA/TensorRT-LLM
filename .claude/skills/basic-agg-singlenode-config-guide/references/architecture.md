# Model-to-Source Mapping

Consult this table to find the primary checked-in sources for each supported model family before selecting or adjusting configs. This table is derived from the repo's lookup YAML files — update it when those files change.

| Model family | Primary sources | Notes |
|---|---|---|
| DeepSeek-R1 | `examples/configs/database/`, `examples/configs/curated/deepseek-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md` | MoE/ADP choices are model-specific. Deployment guide points to `deepseek-r1-throughput.yaml` and `deepseek-r1-deepgemm.yaml`; treat `deepseek-r1-latency.yaml` as source-backed but not deployment-guide-explicit. **MTP eligible:** exact checked-in configs for `deepseek-ai/DeepSeek-R1-0528` or `nvidia/DeepSeek-R1-0528-FP4-v2` with `speculative_config.decoding_type: MTP` are in scope — copy the full `speculative_config` block verbatim, never interpolate, and do not generalize MTP to other models. |
| DeepSeek-V3 / V3.2-Exp | `examples/models/core/deepseek_v3/README.md` plus nearby DeepSeek configs | Use the V3 README for V3-family behavior, not the R1 deployment guide. |
| GPT-OSS-120B | `examples/configs/database/openai/gpt-oss-120b/`, `examples/configs/curated/gpt-oss-120b-*.yaml`, `docs/source/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.md` | Uses `moe_config` and `moe_expert_parallel_size`. |
| Qwen3 | `examples/configs/curated/qwen3.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-on-trtllm.md`, `examples/models/core/qwen/` | Exclude `qwen3-disagg-prefill.yaml` for basic aggregate scope. |
| Qwen3-Next | `examples/configs/curated/qwen3-next.yaml`, `docs/source/deployment-guide/deployment-guide-for-qwen3-next-on-trtllm.md`, `examples/models/core/qwen/` | Prefer exact-model guide over broader Qwen3 guide. |
| Llama-4 Scout | `examples/configs/curated/llama-4-scout.yaml`, `docs/source/deployment-guide/deployment-guide-for-llama4-scout-on-trtllm.md` | MoE/EP tuning in docs; not a dense-only template. |
| Llama-3.3-70B | `examples/configs/curated/llama-3.3-70b.yaml`, `docs/source/deployment-guide/deployment-guide-for-llama3.3-70b-on-trtllm.md` | Use checked-in config/docs, not family-level extrapolation. |
| Kimi-K2 | `examples/configs/curated/kimi-k2-thinking.yaml`, `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` | Config includes `cache_transceiver_config` — keep it, as the deployment guide explicitly requires it (not a disagg indicator here). Maps to `DeepseekV3ForCausalLM`; nearby DeepSeek configs are secondary comparators only. Enables `trust_remote_code: true` — call out as a trust boundary. |

If a model has no deployment guide or nearby checked-in config, label any draft as unverified.
