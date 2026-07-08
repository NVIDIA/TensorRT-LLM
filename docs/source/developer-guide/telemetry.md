# Telemetry

This page documents TensorRT-LLM usage telemetry. It is generated during the
Sphinx docs build by rendering the committed telemetry manifest
(`tensorrt_llm/usage/llm_args_golden_manifest.json`).

Start with the
[Telemetry Data Collection section in the root README](source:README.md#telemetry-data-collection)
for the user-facing collection and opt-out overview, and the
[telemetry schema reference](source:tensorrt_llm/usage/schemas/README.md)
for the wire schema.

**No PII or free-form fields are captured.** LLM API configuration capture is
*type-driven*: fields whose type is categorical (`Literal`/`Enum`/`bool`) or
numeric (`int`/`float`), plus safe collections of those, are captured
automatically. Free-form `str`/`Any`/`Path`/`dict`/`Callable` are never captured
unless a field carries an explicit allowlist (`TelemetryField.categorical(...)`),
and any field may opt out with `telemetry=False`. Every captured field is listed
below; the runtime can capture nothing absent from this list.

If the manifest check fails, run `python3 scripts/generate_llm_args_golden_manifest.py`, then commit
`tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval.

## LLM API Configuration Fields

A field can still be absent from a specific payload when its parent config is
unset or when the safety sanitizer rejects the runtime value.

### `TorchLlmArgs`

258 captured fields.

| Captured key | Annotation | Kind | Converter | Allowed values |
|--------------|------------|------|-----------|----------------|
| `allreduce_strategy` | `Optional[Literal['AUTO', 'NCCL', 'UB', 'MINLATENCY', 'ONESHOT', 'TWOSHOT', 'LOWPRECISION', 'MNNVL', 'NCCL_SYMMETRIC']]` | `categorical` |  | `AUTO`, `NCCL`, `UB`, `MINLATENCY`, `ONESHOT`, `TWOSHOT`, `LOWPRECISION`, `MNNVL`, `NCCL_SYMMETRIC` |
| `attention_dp_config.batching_wait_iters` | `<class 'int'>` | `value` |  |  |
| `attention_dp_config.enable_balance` | `<class 'bool'>` | `value` |  |  |
| `attention_dp_config.enable_kv_cache_aware_routing` | `<class 'bool'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_account_for_in_transfer` | `<class 'bool'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_cold_start_warmup` | `<class 'bool'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_conversation_affinity` | `<class 'bool'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_fair_share_multiplier` | `<class 'float'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_load_balance_weight` | `<class 'float'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_match_rate_threshold` | `<class 'float'>` | `value` |  |  |
| `attention_dp_config.kv_cache_routing_max_sessions` | `<class 'int'>` | `value` |  |  |
| `attention_dp_config.timeout_iters` | `<class 'int'>` | `value` |  |  |
| `attn_backend` | `<class 'str'>` | `categorical` | allowlist | `VANILLA`, `TRTLLM`, `FLASHINFER`, `FLASHINFER_STAR_ATTENTION` |
| `backend` | `Literal['pytorch']` | `categorical` |  | `pytorch` |
| `batch_wait_max_tokens_ratio` | `<class 'float'>` | `value` |  |  |
| `batch_wait_timeout_iters` | `<class 'int'>` | `value` |  |  |
| `batch_wait_timeout_ms` | `<class 'float'>` | `value` |  |  |
| `cache_transceiver_config.backend` | `Optional[Literal['DEFAULT', 'UCX', 'NIXL', 'MOONCAKE', 'MPI']]` | `categorical` |  | `DEFAULT`, `UCX`, `NIXL`, `MOONCAKE`, `MPI` |
| `cache_transceiver_config.kv_transfer_poll_interval_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_sender_future_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.max_tokens_in_buffer` | `Optional[int]` | `value` |  |  |
| `cache_transceiver_config.transceiver_runtime` | `Optional[Literal['CPP', 'PYTHON']]` | `categorical` |  | `CPP`, `PYTHON` |
| `context_parallel_size` | `<class 'int'>` | `value` |  |  |
| `cp_config.block_size` | `Optional[int]` | `value` |  |  |
| `cp_config.cp_anchor_size` | `Optional[int]` | `value` |  |  |
| `cp_config.cp_type` | `<enum 'CpType'>` | `categorical` |  | `ULYSSES`, `STAR`, `RING`, `HELIX` |
| `cp_config.fifo_version` | `Optional[int]` | `value` |  |  |
| `cp_config.tokens_per_block` | `Optional[int]` | `value` |  |  |
| `cp_config.use_nccl_for_alltoall` | `Optional[bool]` | `value` |  |  |
| `cuda_graph_config.batch_sizes` | `Optional[List[int]]` | `value` |  |  |
| `cuda_graph_config.enable_padding` | `<class 'bool'>` | `value` |  |  |
| `cuda_graph_config.max_batch_size` | `<class 'int'>` | `value` |  |  |
| `cuda_graph_config.max_num_token` | `<class 'int'>` | `value` |  |  |
| `cuda_graph_config.max_seq_len` | `<class 'int'>` | `value` |  |  |
| `cuda_graph_config.mode` | `Literal['decode']` | `categorical` |  | `decode`, `encode` |
| `cuda_graph_config.num_tokens` | `Optional[List[Annotated[int, Gt(gt=0)]]]` | `value` |  |  |
| `cuda_graph_config.seq_lens` | `Optional[List[Annotated[int, Gt(gt=0)]]]` | `value` |  |  |
| `disable_flashinfer_sampling` | `<class 'bool'>` | `value` |  |  |
| `disable_overlap_scheduler` | `<class 'bool'>` | `value` |  |  |
| `dtype` | `<class 'str'>` | `categorical` | allowlist | `auto`, `float16`, `bfloat16`, `float32` |
| `dwdp_config.contention_opt` | `<class 'bool'>` | `value` |  |  |
| `dwdp_config.dwdp_size` | `<class 'int'>` | `value` |  |  |
| `dwdp_config.num_experts_per_worker` | `<class 'int'>` | `value` |  |  |
| `dwdp_config.num_groups` | `<class 'int'>` | `value` |  |  |
| `dwdp_config.num_prefetch_experts` | `<class 'int'>` | `value` |  |  |
| `enable_attention_dp` | `<class 'bool'>` | `value` |  |  |
| `enable_autotuner` | `<class 'bool'>` | `value` |  |  |
| `enable_chunked_prefill` | `<class 'bool'>` | `value` |  |  |
| `enable_early_first_token_response` | `<class 'bool'>` | `value` |  |  |
| `enable_energy_metrics` | `<class 'bool'>` | `value` |  |  |
| `enable_iter_perf_stats` | `<class 'bool'>` | `value` |  |  |
| `enable_iter_req_stats` | `<class 'bool'>` | `value` |  |  |
| `enable_layerwise_nvtx_marker` | `<class 'bool'>` | `value` |  |  |
| `enable_lm_head_tp_in_adp` | `<class 'bool'>` | `value` |  |  |
| `enable_lora` | `<class 'bool'>` | `value` |  |  |
| `enable_min_latency` | `<class 'bool'>` | `value` |  |  |
| `enable_resource_governor` | `<class 'bool'>` | `value` |  |  |
| `enable_speculative_beam_history_d2h` | `<class 'bool'>` | `value` |  |  |
| `encode_only` | `<class 'bool'>` | `value` |  |  |
| `encoder_max_batch_size` | `Optional[int]` | `value` |  |  |
| `encoder_max_num_tokens` | `Optional[int]` | `value` |  |  |
| `force_dynamic_quantization` | `<class 'bool'>` | `value` |  |  |
| `garbage_collection_gen0_threshold` | `<class 'int'>` | `value` |  |  |
| `gather_generation_logits` | `<class 'bool'>` | `value` |  |  |
| `gms_config.mode` | `Literal['auto', 'rw', 'ro']` | `categorical` |  | `auto`, `rw`, `ro` |
| `gpus_per_node` | `Optional[int]` | `value` |  |  |
| `guided_decoding_backend` | `Optional[Literal['xgrammar', 'llguidance']]` | `categorical` |  | `xgrammar`, `llguidance` |
| `iter_stats_max_iterations` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.attention_dp_events_gather_period_ms` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.avg_seq_len` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `kv_cache_config.block_reuse_policy` | `Literal['all_reusable', 'per_request']` | `categorical` |  | `all_reusable`, `per_request` |
| `kv_cache_config.copy_on_partial_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.cross_kv_cache_fraction` | `Optional[float]` | `value` |  |  |
| `kv_cache_config.disk_cache_size` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `kv_cache_config.disk_prefetch_num_reqs` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.dtype` | `<class 'str'>` | `categorical` | allowlist | `auto`, `float16`, `bfloat16`, `float32`, `fp8`, `nvfp4` |
| `kv_cache_config.enable_block_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_kv_pool_rebalance` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_partial_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_swa_scratch_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.event_buffer_max_size` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.free_gpu_memory_fraction` | `Optional[float]` | `value` |  |  |
| `kv_cache_config.host_cache_size` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.iteration_stats_interval` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.kv_cache_event_hash_algo` | `Literal['auto', 'v1_block_key', 'v2_sha256', 'v2_sha256_64']` | `categorical` |  | `auto`, `v1_block_key`, `v2_sha256`, `v2_sha256_64` |
| `kv_cache_config.mamba_ssm_cache_dtype` | `Literal['auto', 'float16', 'bfloat16', 'float32']` | `categorical` |  | `auto`, `float16`, `bfloat16`, `float32` |
| `kv_cache_config.mamba_ssm_philox_rounds` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.mamba_ssm_stochastic_rounding` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.mamba_state_cache_interval` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.max_attention_window` | `Optional[List[int]]` | `value` |  |  |
| `kv_cache_config.max_gpu_total_bytes` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.max_tokens` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.max_util_for_resume` | `<class 'float'>` | `value` |  |  |
| `kv_cache_config.pool_ratio` | `Optional[List[float]]` | `value` |  |  |
| `kv_cache_config.secondary_offload_min_priority` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.sink_token_length` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.tokens_per_block` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.use_kv_cache_manager_v2` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.use_uvm` | `<class 'bool'>` | `value` |  |  |
| `kv_connector_config.connector` | `Optional[str]` | `categorical` | allowlist | `lmcache`, `lmcache-mp`, `kvbm` |
| `layer_wise_benchmarks_config.calibration_layer_indices` | `Optional[List[int]]` | `value` |  |  |
| `layer_wise_benchmarks_config.calibration_mode` | `Literal['NONE', 'MARK', 'COLLECT']` | `categorical` |  | `NONE`, `MARK`, `COLLECT` |
| `load_format` | `Union[str, tensorrt_llm.llmapi.llm_args.LoadFormat]` | `categorical` | allowlist | `auto`, `dummy`, `vision_only`, `gms` |
| `lora_config.lora_ckpt_source` | `Literal['hf', 'nemo']` | `categorical` |  | `hf`, `nemo` |
| `lora_config.max_cpu_loras` | `Optional[int]` | `value` |  |  |
| `lora_config.max_lora_rank` | `<class 'int'>` | `value` |  |  |
| `lora_config.max_loras` | `Optional[int]` | `value` |  |  |
| `lora_config.swap_gate_up_proj_lora_b_weight` | `<class 'bool'>` | `value` |  |  |
| `max_batch_size` | `Optional[int]` | `value` |  |  |
| `max_beam_width` | `Optional[int]` | `value` |  |  |
| `max_input_len` | `Optional[int]` | `value` |  |  |
| `max_num_tokens` | `Optional[int]` | `value` |  |  |
| `max_seq_len` | `Optional[int]` | `value` |  |  |
| `max_stats_len` | `<class 'int'>` | `value` |  |  |
| `mm_encoder_only` | `<class 'bool'>` | `value` |  |  |
| `moe_cluster_parallel_size` | `Optional[int]` | `value` |  |  |
| `moe_config.backend` | `Literal['AUTO', 'CUTLASS', 'CUTEDSL', 'WIDEEP', 'TRTLLM', 'DEEPGEMM', 'DENSEGEMM', 'VANILLA', 'TRITON', 'MARLIN', 'MEGAMOE_DEEPGEMM']` | `categorical` |  | `AUTO`, `CUTLASS`, `CUTEDSL`, `WIDEEP`, `TRTLLM`, `DEEPGEMM`, `DENSEGEMM`, `VANILLA`, `TRITON`, `MARLIN`, `MEGAMOE_DEEPGEMM` |
| `moe_config.disable_finalize_fusion` | `<class 'bool'>` | `value` |  |  |
| `moe_config.max_num_tokens` | `Optional[int]` | `value` |  |  |
| `moe_config.use_low_precision_moe_combine` | `<class 'bool'>` | `value` |  |  |
| `moe_expert_parallel_size` | `Optional[int]` | `value` |  |  |
| `moe_tensor_parallel_size` | `Optional[int]` | `value` |  |  |
| `mx_config.preshard_strategy` | `<class 'str'>` | `categorical` | allowlist | `per_module` |
| `mx_config.server_query_timeout_s` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `num_postprocess_workers` | `<class 'int'>` | `value` |  |  |
| `nvfp4_gemm_config.allowed_backends` | `List[Literal['cutlass', 'cublaslt', 'cutedsl', 'cuda_core', 'marlin']]` | `value` |  | `cutlass`, `cublaslt`, `cutedsl`, `cuda_core`, `marlin` |
| `orchestrator_type` | `Optional[Literal['rpc', 'ray']]` | `categorical` |  | `rpc`, `ray` |
| `peft_cache_config.device_cache_percent` | `<class 'float'>` | `value` |  |  |
| `peft_cache_config.host_cache_size` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_adapter_size` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_pages_per_block_device` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_pages_per_block_host` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_copy_streams` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_device_module_layer` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_ensure_workers` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_host_module_layer` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_put_workers` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.optimal_adapter_size` | `<class 'int'>` | `value` |  |  |
| `perf_metrics_max_requests` | `<class 'int'>` | `value` |  |  |
| `pipeline_parallel_size` | `<class 'int'>` | `value` |  |  |
| `pp_partition` | `Optional[List[int]]` | `value` |  |  |
| `print_iter_log` | `<class 'bool'>` | `value` |  |  |
| `prometheus_metrics_config.e2e_request_latency_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_decode_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_inference_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_prefill_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_queue_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.time_per_output_token_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.time_to_first_token_buckets` | `Optional[List[float]]` | `value` |  |  |
| `ray_placement_config.defer_workers_init` | `<class 'bool'>` | `value` |  |  |
| `ray_placement_config.per_worker_gpu_share` | `Optional[float]` | `value` |  |  |
| `ray_placement_config.placement_bundle_indices` | `Optional[List[List[int]]]` | `value` |  |  |
| `reasoning_parser` | `Optional[str]` | `categorical` | allowlist | `auto`, `deepseek-r1`, `laguna`, `qwen3`, `qwen3_5`, `minimax_m2`, `minimax_m2_append_think`, `nano-v3`, `gemma4`, `kimi_k2`, `kimi_k25` |
| `reorder_policy_config.policy_args.agent_inflight_seq_num` | `<class 'int'>` | `value` |  |  |
| `reorder_policy_config.policy_args.agent_percentage` | `<class 'float'>` | `value` |  |  |
| `reorder_policy_config.policy_name` | `Optional[Literal['AgentTree']]` | `categorical` |  | `AgentTree` |
| `request_stats_max_iterations` | `Optional[int]` | `value` |  |  |
| `return_perf_metrics` | `<class 'bool'>` | `value` |  |  |
| `sampler_force_async_worker` | `<class 'bool'>` | `value` |  |  |
| `sampler_type` | `Union[str, tensorrt_llm.llmapi.llm_args.SamplerType]` | `categorical` | allowlist | `TRTLLMSampler`, `TorchSampler`, `auto` |
| `scheduler_config.capacity_scheduler_policy` | `<enum 'CapacitySchedulerPolicy'>` | `categorical` |  | `MAX_UTILIZATION`, `GUARANTEED_NO_EVICT`, `STATIC_BATCH` |
| `scheduler_config.context_chunking_policy` | `Optional[tensorrt_llm.llmapi.llm_args.ContextChunkingPolicy]` | `categorical` |  | `FIRST_COME_FIRST_SERVED`, `EQUAL_PROGRESS`, `FORCE_CHUNK` |
| `scheduler_config.dynamic_batch_config.dynamic_batch_moving_average_window` | `<class 'int'>` | `value` |  |  |
| `scheduler_config.dynamic_batch_config.enable_batch_size_tuning` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.dynamic_batch_config.enable_max_num_tokens_tuning` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.enable_prefix_aware_scheduling` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.use_python_scheduler` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.waiting_queue_policy` | `<enum 'WaitingQueuePolicy'>` | `categorical` |  | `fcfs`, `priority` |
| `skip_tokenizer_init` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.algorithm` | `Literal['dsa']` | `categorical` |  | `dsa`, `deepseek_v4`, `minimax_m3`, `rocket`, `skip_softmax` |
| `sparse_attention_config.compress_ratios` | `List[int]` | `value` |  |  |
| `sparse_attention_config.enable_heuristic_topk` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.index_head_dim` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_n_heads` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_topk` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_k_dtype` | `Literal['fp8', 'fp4']` | `categorical` |  | `fp8`, `fp4` |
| `sparse_attention_config.indexer_max_chunk_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_rope_interleave` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.kernel_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.kt_cache_dtype` | `Optional[str]` | `categorical` | allowlist | `bfloat16`, `float8_e5m2` |
| `sparse_attention_config.page_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.prompt_budget` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.q_split_threshold` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.seq_len_threshold` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.skip_indexer_for_short_seqs` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.sparse_block_size` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_disable_index_value` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.sparse_index_dim` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_init_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_local_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_num_index_heads` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_score_type` | `Literal['max']` | `categorical` |  | `max` |
| `sparse_attention_config.sparse_topk_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.topk` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.topr` | `Union[int, float, NoneType]` | `value` |  |  |
| `sparse_attention_config.use_cute_dsl_paged_mqa_logits` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.use_cute_dsl_topk` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.window_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.acceptance_length_threshold` | `Optional[Annotated[float, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.acceptance_window` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.allow_advanced_sampling` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.begin_thinking_phase_token` | `<class 'int'>` | `value` |  |  |
| `speculative_config.decoding_type` | `Literal['AUTO']` | `categorical` |  | `AUTO`, `DFlash`, `Draft_Target`, `Eagle3`, `Eagle`, `Lookahead`, `MTP`, `Medusa`, `NGram`, `PARD`, `SA`, `SaveState`, `User_Provided` |
| `speculative_config.dynamic_tree_max_topK` | `Optional[int]` | `value` |  |  |
| `speculative_config.eagle3_layers_to_capture` | `Optional[Set[int]]` | `value` |  |  |
| `speculative_config.eagle3_model_arch` | `Literal['llama3', 'mistral_large3']` | `categorical` |  | `llama3`, `mistral_large3` |
| `speculative_config.eagle3_one_model` | `Optional[bool]` | `value` |  |  |
| `speculative_config.eagle_choices` | `Optional[List[List[int]]]` | `value` |  |  |
| `speculative_config.enable_global_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.end_thinking_phase_token` | `<class 'int'>` | `value` |  |  |
| `speculative_config.global_pool_size` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `speculative_config.greedy_sampling` | `Optional[bool]` | `value` |  |  |
| `speculative_config.is_keep_all` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.is_public_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.is_use_oldest` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.mask_token_id` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_concurrency` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `speculative_config.max_draft_len` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.max_matching_ngram_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_ngram_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_non_leaves_per_layer` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_total_draft_tokens` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_verification_set_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_window_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.medusa_choices` | `Optional[List[List[int]]]` | `value` |  |  |
| `speculative_config.mtp_eagle_one_model` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.num_eagle_layers` | `Optional[int]` | `value` |  |  |
| `speculative_config.num_medusa_heads` | `Optional[int]` | `value` |  |  |
| `speculative_config.num_nextn_predict_layers` | `Optional[int]` | `value` |  |  |
| `speculative_config.posterior_threshold` | `Optional[float]` | `value` |  |  |
| `speculative_config.relaxed_delta` | `<class 'float'>` | `value` |  |  |
| `speculative_config.relaxed_topk` | `<class 'int'>` | `value` |  |  |
| `speculative_config.sa_config.enable_global_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.sa_config.threshold` | `<class 'int'>` | `value` |  |  |
| `speculative_config.target_layer_ids` | `Optional[List[int]]` | `value` |  |  |
| `speculative_config.use_dynamic_tree` | `Optional[bool]` | `value` |  |  |
| `speculative_config.use_mtp_vanilla` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.use_rejection_sampling` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.use_relaxed_acceptance_for_thinking` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.write_interval` | `<class 'int'>` | `value` |  |  |
| `stream_interval` | `<class 'int'>` | `value` |  |  |
| `telemetry_config.disabled` | `<class 'bool'>` | `value` |  |  |
| `telemetry_config.usage_context` | `<enum 'UsageContext'>` | `categorical` |  | `unknown`, `llm_class`, `cli_serve`, `cli_bench`, `cli_eval` |
| `tensor_parallel_size` | `<class 'int'>` | `value` |  |  |
| `tokenizer_mode` | `Literal['auto', 'slow']` | `categorical` |  | `auto`, `slow` |
| `torch_compile_config.capture_num_tokens` | `Optional[List[Annotated[int, Gt(gt=0)]]]` | `value` |  |  |
| `torch_compile_config.enable_fullgraph` | `<class 'bool'>` | `value` |  |  |
| `torch_compile_config.enable_inductor` | `<class 'bool'>` | `value` |  |  |
| `torch_compile_config.enable_piecewise_cuda_graph` | `<class 'bool'>` | `value` |  |  |
| `torch_compile_config.enable_userbuffers` | `<class 'bool'>` | `value` |  |  |
| `torch_compile_config.max_num_streams` | `<class 'int'>` | `value` |  |  |
| `trust_remote_code` | `<class 'bool'>` | `value` |  |  |
| `use_cute_dsl_bf16_bmm` | `<class 'bool'>` | `value` |  |  |
| `use_cute_dsl_bf16_gemm` | `<class 'bool'>` | `value` |  |  |
| `use_cute_dsl_blockscaling_bmm` | `<class 'bool'>` | `value` |  |  |
| `use_cute_dsl_blockscaling_mm` | `<class 'bool'>` | `value` |  |  |
| `video_pruning_rate` | `Optional[float]` | `value` |  |  |

### `TrtLlmArgs`

279 captured fields.

| Captured key | Annotation | Kind | Converter | Allowed values |
|--------------|------------|------|-----------|----------------|
| `backend` | `Optional[str]` | `categorical` | allowlist | `pytorch`, `tensorrt`, `_autodeploy` |
| `batching_type` | `Optional[tensorrt_llm.llmapi.llm_args.BatchingType]` | `categorical` |  | `STATIC`, `INFLIGHT` |
| `build_config.dry_run` | `<class 'bool'>` | `value` |  |  |
| `build_config.enable_debug_output` | `<class 'bool'>` | `value` |  |  |
| `build_config.force_num_profiles` | `Optional[int]` | `value` |  |  |
| `build_config.gather_context_logits` | `<class 'bool'>` | `value` |  |  |
| `build_config.gather_generation_logits` | `<class 'bool'>` | `value` |  |  |
| `build_config.kv_cache_type` | `Optional[tensorrt_llm.llmapi.kv_cache_type.KVCacheType]` | `categorical` |  | `continuous`, `paged`, `disabled` |
| `build_config.lora_config.lora_ckpt_source` | `Literal['hf', 'nemo']` | `categorical` |  | `hf`, `nemo` |
| `build_config.lora_config.max_cpu_loras` | `Optional[int]` | `value` |  |  |
| `build_config.lora_config.max_lora_rank` | `<class 'int'>` | `value` |  |  |
| `build_config.lora_config.max_loras` | `Optional[int]` | `value` |  |  |
| `build_config.lora_config.swap_gate_up_proj_lora_b_weight` | `<class 'bool'>` | `value` |  |  |
| `build_config.max_batch_size` | `<class 'int'>` | `value` |  |  |
| `build_config.max_beam_width` | `<class 'int'>` | `value` |  |  |
| `build_config.max_draft_len` | `<class 'int'>` | `value` |  |  |
| `build_config.max_encoder_input_len` | `<class 'int'>` | `value` |  |  |
| `build_config.max_input_len` | `<class 'int'>` | `value` |  |  |
| `build_config.max_num_tokens` | `<class 'int'>` | `value` |  |  |
| `build_config.max_prompt_embedding_table_size` | `<class 'int'>` | `value` |  |  |
| `build_config.max_seq_len` | `Optional[int]` | `value` |  |  |
| `build_config.monitor_memory` | `<class 'bool'>` | `value` |  |  |
| `build_config.opt_batch_size` | `<class 'int'>` | `value` |  |  |
| `build_config.opt_num_tokens` | `Optional[int]` | `value` |  |  |
| `build_config.plugin_config.bert_attention_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.bert_context_fmha_fp32_acc` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.context_fmha` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.dora_plugin` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.fp8_rowwise_gemm_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.fuse_fp4_quant` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.gemm_allreduce_plugin` | `Optional[Literal['float16', 'bfloat16', None]]` | `categorical` |  | `float16`, `bfloat16`, `None` |
| `build_config.plugin_config.gemm_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', 'fp8', 'nvfp4', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `fp8`, `nvfp4`, `None` |
| `build_config.plugin_config.gemm_swiglu_plugin` | `Optional[Literal['fp8', None]]` | `categorical` |  | `fp8`, `None` |
| `build_config.plugin_config.gpt_attention_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.identity_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.layernorm_quantization_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.lora_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.low_latency_gemm_plugin` | `Optional[Literal['fp8', None]]` | `categorical` |  | `fp8`, `None` |
| `build_config.plugin_config.low_latency_gemm_swiglu_plugin` | `Optional[Literal['fp8', None]]` | `categorical` |  | `fp8`, `None` |
| `build_config.plugin_config.mamba_conv1d_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.manage_weights` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.moe_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.multiple_profiles` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.nccl_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.norm_quant_fusion` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.paged_kv_cache` | `Optional[bool]` | `value` |  |  |
| `build_config.plugin_config.paged_state` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.pp_reduce_scatter` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.qserve_gemm_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.quantize_per_token_plugin` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.quantize_tensor_plugin` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.reduce_fusion` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.remove_input_padding` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.rmsnorm_quantization_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.smooth_quant_gemm_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.smooth_quant_plugins` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.streamingllm` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.tokens_per_block` | `<class 'int'>` | `value` |  |  |
| `build_config.plugin_config.use_fp8_context_fmha` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.use_fused_mlp` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.use_paged_context_fmha` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.user_buffer` | `<class 'bool'>` | `value` |  |  |
| `build_config.plugin_config.weight_only_groupwise_quant_matmul_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.plugin_config.weight_only_quant_matmul_plugin` | `Optional[Literal['auto', 'float16', 'float32', 'bfloat16', 'int32', None]]` | `categorical` |  | `auto`, `float16`, `float32`, `bfloat16`, `int32`, `None` |
| `build_config.speculative_decoding_mode` | `<flag 'SpeculativeDecodingMode'>` | `categorical` |  | `NONE`, `DRAFT_TOKENS_EXTERNAL`, `MEDUSA`, `LOOKAHEAD_DECODING`, `EXPLICIT_DRAFT_TOKENS`, `EAGLE`, `NGRAM`, `USER_PROVIDED`, `SAVE_HIDDEN_STATES`, `AUTO` |
| `build_config.strongly_typed` | `<class 'bool'>` | `value` |  |  |
| `build_config.use_mrope` | `<class 'bool'>` | `value` |  |  |
| `build_config.use_refit` | `<class 'bool'>` | `value` |  |  |
| `build_config.use_strip_plan` | `<class 'bool'>` | `value` |  |  |
| `build_config.weight_sparsity` | `<class 'bool'>` | `value` |  |  |
| `build_config.weight_streaming` | `<class 'bool'>` | `value` |  |  |
| `cache_transceiver_config.backend` | `Optional[Literal['DEFAULT', 'UCX', 'NIXL', 'MOONCAKE', 'MPI']]` | `categorical` |  | `DEFAULT`, `UCX`, `NIXL`, `MOONCAKE`, `MPI` |
| `cache_transceiver_config.kv_transfer_poll_interval_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_sender_future_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.max_tokens_in_buffer` | `Optional[int]` | `value` |  |  |
| `cache_transceiver_config.transceiver_runtime` | `Optional[Literal['CPP', 'PYTHON']]` | `categorical` |  | `CPP`, `PYTHON` |
| `calib_config.calib_batch_size` | `<class 'int'>` | `value` |  |  |
| `calib_config.calib_batches` | `<class 'int'>` | `value` |  |  |
| `calib_config.calib_max_seq_length` | `<class 'int'>` | `value` |  |  |
| `calib_config.device` | `Literal['cuda', 'cpu']` | `categorical` |  | `cuda`, `cpu` |
| `calib_config.random_seed` | `<class 'int'>` | `value` |  |  |
| `calib_config.tokenizer_max_seq_length` | `<class 'int'>` | `value` |  |  |
| `context_parallel_size` | `<class 'int'>` | `value` |  |  |
| `cp_config.block_size` | `Optional[int]` | `value` |  |  |
| `cp_config.cp_anchor_size` | `Optional[int]` | `value` |  |  |
| `cp_config.cp_type` | `<enum 'CpType'>` | `categorical` |  | `ULYSSES`, `STAR`, `RING`, `HELIX` |
| `cp_config.fifo_version` | `Optional[int]` | `value` |  |  |
| `cp_config.tokens_per_block` | `Optional[int]` | `value` |  |  |
| `cp_config.use_nccl_for_alltoall` | `Optional[bool]` | `value` |  |  |
| `dtype` | `<class 'str'>` | `categorical` | allowlist | `auto`, `float16`, `bfloat16`, `float32` |
| `embedding_parallel_mode` | `Literal['NONE', 'SHARDING_ALONG_VOCAB', 'SHARDING_ALONG_HIDDEN']` | `categorical` |  | `NONE`, `SHARDING_ALONG_VOCAB`, `SHARDING_ALONG_HIDDEN` |
| `enable_attention_dp` | `<class 'bool'>` | `value` |  |  |
| `enable_build_cache.max_cache_storage_gb` | `<class 'float'>` | `value` |  |  |
| `enable_build_cache.max_records` | `<class 'int'>` | `value` |  |  |
| `enable_chunked_prefill` | `<class 'bool'>` | `value` |  |  |
| `enable_energy_metrics` | `<class 'bool'>` | `value` |  |  |
| `enable_lm_head_tp_in_adp` | `<class 'bool'>` | `value` |  |  |
| `enable_lora` | `<class 'bool'>` | `value` |  |  |
| `enable_prompt_adapter` | `<class 'bool'>` | `value` |  |  |
| `enable_tqdm` | `<class 'bool'>` | `value` |  |  |
| `extended_runtime_perf_knob_config.cuda_graph_cache_size` | `<class 'int'>` | `value` |  |  |
| `extended_runtime_perf_knob_config.cuda_graph_mode` | `<class 'bool'>` | `value` |  |  |
| `extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc` | `<class 'bool'>` | `value` |  |  |
| `extended_runtime_perf_knob_config.multi_block_mode` | `<class 'bool'>` | `value` |  |  |
| `fail_fast_on_attention_window_too_large` | `<class 'bool'>` | `value` |  |  |
| `fast_build` | `<class 'bool'>` | `value` |  |  |
| `gather_generation_logits` | `<class 'bool'>` | `value` |  |  |
| `gpus_per_node` | `Optional[int]` | `value` |  |  |
| `guided_decoding_backend` | `Optional[Literal['xgrammar', 'llguidance']]` | `categorical` |  | `xgrammar`, `llguidance` |
| `iter_stats_max_iterations` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.attention_dp_events_gather_period_ms` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.avg_seq_len` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `kv_cache_config.block_reuse_policy` | `Literal['all_reusable', 'per_request']` | `categorical` |  | `all_reusable`, `per_request` |
| `kv_cache_config.copy_on_partial_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.cross_kv_cache_fraction` | `Optional[float]` | `value` |  |  |
| `kv_cache_config.disk_cache_size` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `kv_cache_config.disk_prefetch_num_reqs` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.dtype` | `<class 'str'>` | `categorical` | allowlist | `auto`, `float16`, `bfloat16`, `float32`, `fp8`, `nvfp4` |
| `kv_cache_config.enable_block_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_kv_pool_rebalance` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_partial_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.enable_swa_scratch_reuse` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.event_buffer_max_size` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.free_gpu_memory_fraction` | `Optional[float]` | `value` |  |  |
| `kv_cache_config.host_cache_size` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.iteration_stats_interval` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.kv_cache_event_hash_algo` | `Literal['auto', 'v1_block_key', 'v2_sha256', 'v2_sha256_64']` | `categorical` |  | `auto`, `v1_block_key`, `v2_sha256`, `v2_sha256_64` |
| `kv_cache_config.mamba_ssm_cache_dtype` | `Literal['auto', 'float16', 'bfloat16', 'float32']` | `categorical` |  | `auto`, `float16`, `bfloat16`, `float32` |
| `kv_cache_config.mamba_ssm_philox_rounds` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.mamba_ssm_stochastic_rounding` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.mamba_state_cache_interval` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.max_attention_window` | `Optional[List[int]]` | `value` |  |  |
| `kv_cache_config.max_gpu_total_bytes` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.max_tokens` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.max_util_for_resume` | `<class 'float'>` | `value` |  |  |
| `kv_cache_config.pool_ratio` | `Optional[List[float]]` | `value` |  |  |
| `kv_cache_config.secondary_offload_min_priority` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.sink_token_length` | `Optional[int]` | `value` |  |  |
| `kv_cache_config.tokens_per_block` | `<class 'int'>` | `value` |  |  |
| `kv_cache_config.use_kv_cache_manager_v2` | `<class 'bool'>` | `value` |  |  |
| `kv_cache_config.use_uvm` | `<class 'bool'>` | `value` |  |  |
| `load_format` | `Literal['auto', 'dummy']` | `categorical` |  | `auto`, `dummy` |
| `lora_config.lora_ckpt_source` | `Literal['hf', 'nemo']` | `categorical` |  | `hf`, `nemo` |
| `lora_config.max_cpu_loras` | `Optional[int]` | `value` |  |  |
| `lora_config.max_lora_rank` | `<class 'int'>` | `value` |  |  |
| `lora_config.max_loras` | `Optional[int]` | `value` |  |  |
| `lora_config.swap_gate_up_proj_lora_b_weight` | `<class 'bool'>` | `value` |  |  |
| `max_batch_size` | `Optional[int]` | `value` |  |  |
| `max_beam_width` | `Optional[int]` | `value` |  |  |
| `max_input_len` | `Optional[int]` | `value` |  |  |
| `max_num_tokens` | `Optional[int]` | `value` |  |  |
| `max_prompt_adapter_token` | `<class 'int'>` | `value` |  |  |
| `max_seq_len` | `Optional[int]` | `value` |  |  |
| `moe_cluster_parallel_size` | `Optional[int]` | `value` |  |  |
| `moe_expert_parallel_size` | `Optional[int]` | `value` |  |  |
| `moe_tensor_parallel_size` | `Optional[int]` | `value` |  |  |
| `normalize_log_probs` | `<class 'bool'>` | `value` |  |  |
| `num_postprocess_workers` | `<class 'int'>` | `value` |  |  |
| `orchestrator_type` | `Optional[Literal['rpc', 'ray']]` | `categorical` |  | `rpc`, `ray` |
| `peft_cache_config.device_cache_percent` | `<class 'float'>` | `value` |  |  |
| `peft_cache_config.host_cache_size` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_adapter_size` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_pages_per_block_device` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.max_pages_per_block_host` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_copy_streams` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_device_module_layer` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_ensure_workers` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_host_module_layer` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.num_put_workers` | `<class 'int'>` | `value` |  |  |
| `peft_cache_config.optimal_adapter_size` | `<class 'int'>` | `value` |  |  |
| `perf_metrics_max_requests` | `<class 'int'>` | `value` |  |  |
| `pipeline_parallel_size` | `<class 'int'>` | `value` |  |  |
| `pp_partition` | `Optional[List[int]]` | `value` |  |  |
| `prometheus_metrics_config.e2e_request_latency_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_decode_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_inference_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_prefill_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.request_queue_time_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.time_per_output_token_buckets` | `Optional[List[float]]` | `value` |  |  |
| `prometheus_metrics_config.time_to_first_token_buckets` | `Optional[List[float]]` | `value` |  |  |
| `quant_config.clamp_val` | `Optional[List[float]]` | `value` |  |  |
| `quant_config.group_size` | `Optional[int]` | `value` |  |  |
| `quant_config.has_zero_point` | `<class 'bool'>` | `value` |  |  |
| `quant_config.kv_cache_quant_algo` | `Optional[tensorrt_llm.quantization.mode.QuantAlgo]` | `categorical` |  | `W8A16`, `W4A16`, `W4A16_AWQ`, `W4A8_AWQ`, `W8A16_GPTQ`, `W4A16_GPTQ`, `W8A8_SQ_PER_CHANNEL`, `W8A8_SQ_PER_TENSOR_PLUGIN`, `W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN`, `W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN`, `W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN`, `W4A8_QSERVE_PER_GROUP`, `W4A8_QSERVE_PER_CHANNEL`, `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES`, `INT8`, `MIXED_PRECISION`, `NVFP4`, `W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`, `W4A16_MXFP4`, `MXFP8`, `NVFP4_AWQ`, `NVFP4_ARC`, `NO_QUANT` |
| `quant_config.mamba_ssm_philox_rounds` | `<class 'int'>` | `value` |  |  |
| `quant_config.mamba_ssm_stochastic_rounding` | `<class 'bool'>` | `value` |  |  |
| `quant_config.pre_quant_scale` | `<class 'bool'>` | `value` |  |  |
| `quant_config.quant_algo` | `Optional[tensorrt_llm.quantization.mode.QuantAlgo]` | `categorical` |  | `W8A16`, `W4A16`, `W4A16_AWQ`, `W4A8_AWQ`, `W8A16_GPTQ`, `W4A16_GPTQ`, `W8A8_SQ_PER_CHANNEL`, `W8A8_SQ_PER_TENSOR_PLUGIN`, `W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN`, `W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN`, `W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN`, `W4A8_QSERVE_PER_GROUP`, `W4A8_QSERVE_PER_CHANNEL`, `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_BLOCK_SCALES`, `INT8`, `MIXED_PRECISION`, `NVFP4`, `W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`, `W4A16_MXFP4`, `MXFP8`, `NVFP4_AWQ`, `NVFP4_ARC`, `NO_QUANT` |
| `quant_config.smoothquant_val` | `<class 'float'>` | `value` |  |  |
| `quant_config.use_meta_recipe` | `<class 'bool'>` | `value` |  |  |
| `reasoning_parser` | `Optional[str]` | `categorical` | allowlist | `auto`, `deepseek-r1`, `laguna`, `qwen3`, `qwen3_5`, `minimax_m2`, `minimax_m2_append_think`, `nano-v3`, `gemma4`, `kimi_k2`, `kimi_k25` |
| `request_stats_max_iterations` | `Optional[int]` | `value` |  |  |
| `return_perf_metrics` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.capacity_scheduler_policy` | `<enum 'CapacitySchedulerPolicy'>` | `categorical` |  | `MAX_UTILIZATION`, `GUARANTEED_NO_EVICT`, `STATIC_BATCH` |
| `scheduler_config.context_chunking_policy` | `Optional[tensorrt_llm.llmapi.llm_args.ContextChunkingPolicy]` | `categorical` |  | `FIRST_COME_FIRST_SERVED`, `EQUAL_PROGRESS`, `FORCE_CHUNK` |
| `scheduler_config.dynamic_batch_config.dynamic_batch_moving_average_window` | `<class 'int'>` | `value` |  |  |
| `scheduler_config.dynamic_batch_config.enable_batch_size_tuning` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.dynamic_batch_config.enable_max_num_tokens_tuning` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.enable_prefix_aware_scheduling` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.use_python_scheduler` | `<class 'bool'>` | `value` |  |  |
| `scheduler_config.waiting_queue_policy` | `<enum 'WaitingQueuePolicy'>` | `categorical` |  | `fcfs`, `priority` |
| `skip_tokenizer_init` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.algorithm` | `Literal['dsa']` | `categorical` |  | `dsa`, `deepseek_v4`, `minimax_m3`, `rocket`, `skip_softmax` |
| `sparse_attention_config.compress_ratios` | `List[int]` | `value` |  |  |
| `sparse_attention_config.enable_heuristic_topk` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.index_head_dim` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_n_heads` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_topk` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_k_dtype` | `Literal['fp8', 'fp4']` | `categorical` |  | `fp8`, `fp4` |
| `sparse_attention_config.indexer_max_chunk_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_rope_interleave` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.kernel_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.kt_cache_dtype` | `Optional[str]` | `categorical` | allowlist | `bfloat16`, `float8_e5m2` |
| `sparse_attention_config.page_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.prompt_budget` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.q_split_threshold` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.seq_len_threshold` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.skip_indexer_for_short_seqs` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.sparse_block_size` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_disable_index_value` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.sparse_index_dim` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_init_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_local_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_num_index_heads` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.sparse_score_type` | `Literal['max']` | `categorical` |  | `max` |
| `sparse_attention_config.sparse_topk_blocks` | `<class 'int'>` | `value` |  |  |
| `sparse_attention_config.topk` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.topr` | `Union[int, float, NoneType]` | `value` |  |  |
| `sparse_attention_config.use_cute_dsl_paged_mqa_logits` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.use_cute_dsl_topk` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.window_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.acceptance_length_threshold` | `Optional[Annotated[float, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.acceptance_window` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.allow_advanced_sampling` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.begin_thinking_phase_token` | `<class 'int'>` | `value` |  |  |
| `speculative_config.decoding_type` | `Literal['AUTO']` | `categorical` |  | `AUTO`, `DFlash`, `Draft_Target`, `Eagle3`, `Eagle`, `Lookahead`, `MTP`, `Medusa`, `NGram`, `PARD`, `SA`, `SaveState`, `User_Provided` |
| `speculative_config.dynamic_tree_max_topK` | `Optional[int]` | `value` |  |  |
| `speculative_config.eagle3_layers_to_capture` | `Optional[Set[int]]` | `value` |  |  |
| `speculative_config.eagle3_model_arch` | `Literal['llama3', 'mistral_large3']` | `categorical` |  | `llama3`, `mistral_large3` |
| `speculative_config.eagle3_one_model` | `Optional[bool]` | `value` |  |  |
| `speculative_config.eagle_choices` | `Optional[List[List[int]]]` | `value` |  |  |
| `speculative_config.enable_global_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.end_thinking_phase_token` | `<class 'int'>` | `value` |  |  |
| `speculative_config.global_pool_size` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `speculative_config.greedy_sampling` | `Optional[bool]` | `value` |  |  |
| `speculative_config.is_keep_all` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.is_public_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.is_use_oldest` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.mask_token_id` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_concurrency` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `speculative_config.max_draft_len` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.max_matching_ngram_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_ngram_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_non_leaves_per_layer` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_total_draft_tokens` | `Optional[int]` | `value` |  |  |
| `speculative_config.max_verification_set_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.max_window_size` | `<class 'int'>` | `value` |  |  |
| `speculative_config.medusa_choices` | `Optional[List[List[int]]]` | `value` |  |  |
| `speculative_config.mtp_eagle_one_model` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.num_eagle_layers` | `Optional[int]` | `value` |  |  |
| `speculative_config.num_medusa_heads` | `Optional[int]` | `value` |  |  |
| `speculative_config.num_nextn_predict_layers` | `Optional[int]` | `value` |  |  |
| `speculative_config.posterior_threshold` | `Optional[float]` | `value` |  |  |
| `speculative_config.relaxed_delta` | `<class 'float'>` | `value` |  |  |
| `speculative_config.relaxed_topk` | `<class 'int'>` | `value` |  |  |
| `speculative_config.sa_config.enable_global_pool` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.sa_config.threshold` | `<class 'int'>` | `value` |  |  |
| `speculative_config.target_layer_ids` | `Optional[List[int]]` | `value` |  |  |
| `speculative_config.use_dynamic_tree` | `Optional[bool]` | `value` |  |  |
| `speculative_config.use_mtp_vanilla` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.use_rejection_sampling` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.use_relaxed_acceptance_for_thinking` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.write_interval` | `<class 'int'>` | `value` |  |  |
| `telemetry_config.disabled` | `<class 'bool'>` | `value` |  |  |
| `telemetry_config.usage_context` | `<enum 'UsageContext'>` | `categorical` |  | `unknown`, `llm_class`, `cli_serve`, `cli_bench`, `cli_eval` |
| `tensor_parallel_size` | `<class 'int'>` | `value` |  |  |
| `tokenizer_mode` | `Literal['auto', 'slow']` | `categorical` |  | `auto`, `slow` |
| `trust_remote_code` | `<class 'bool'>` | `value` |  |  |
