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

269 captured fields.

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
| `cache_transceiver_config.kv_cache_bounce_size_mb` | `<class 'int'>` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_poll_interval_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_sender_future_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.kv_transfer_timeout_ms` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `cache_transceiver_config.max_tokens_in_buffer` | `Optional[int]` | `value` |  |  |
| `cache_transceiver_config.transceiver_runtime` | `Optional[Literal['CPP', 'PYTHON', 'auto']]` | `categorical` |  | `CPP`, `PYTHON`, `auto` |
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
| `disable_mm_encoder` | `<class 'bool'>` | `value` |  |  |
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
| `enable_low_latency_host_dispatch` | `<class 'bool'>` | `value` |  |  |
| `enable_min_latency` | `<class 'bool'>` | `value` |  |  |
| `enable_resource_governor` | `<class 'bool'>` | `value` |  |  |
| `enable_speculative_beam_history_d2h` | `<class 'bool'>` | `value` |  |  |
| `encode_only` | `<class 'bool'>` | `value` |  |  |
| `encoder_max_num_items` | `Optional[int]` | `value` |  |  |
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
| `kv_cache_config.block_reuse_policy` | `Literal['all_reusable', 'per_request', 'per_conversation']` | `categorical` |  | `all_reusable`, `per_request`, `per_conversation` |
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
| `kv_cache_config.use_kv_cache_manager_v2` | `Union[bool, Literal['auto']]` | `value` |  | `auto` |
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
| `multimodal_config.encoder_cache_max_bytes` | `<class 'int'>` | `value` |  |  |
| `multimodal_config.encoder_side_stream_max_ahead` | `<class 'int'>` | `value` |  |  |
| `multimodal_config.video_pruning_rate` | `Optional[float]` | `value` |  |  |
| `mx_config.preshard_strategy` | `<class 'str'>` | `categorical` | allowlist | `per_module` |
| `mx_config.server_query_timeout_s` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `num_postprocess_workers` | `<class 'int'>` | `value` |  |  |
| `num_serve_frontends` | `<class 'int'>` | `value` |  |  |
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
| `sparse_attention_config.implementation` | `Literal['triton', 'msa']` | `categorical` |  | `triton`, `msa` |
| `sparse_attention_config.index_head_dim` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_n_heads` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.index_topk` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_k_dtype` | `Literal['fp8', 'fp4']` | `categorical` |  | `fp8`, `fp4` |
| `sparse_attention_config.indexer_max_chunk_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.indexer_rope_interleave` | `<class 'bool'>` | `value` |  |  |
| `sparse_attention_config.kernel_size` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.kt_cache_dtype` | `Optional[str]` | `categorical` | allowlist | `bfloat16`, `float8_e5m2` |
| `sparse_attention_config.num_attention_heads` | `Optional[int]` | `value` |  |  |
| `sparse_attention_config.num_key_value_heads` | `Optional[int]` | `value` |  |  |
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
| `speculative_config.acceptance_rate_threshold` | `Optional[float]` | `value` |  |  |
| `speculative_config.acceptance_rate_window_size` | `Optional[Annotated[int, Ge(ge=0)]]` | `value` |  |  |
| `speculative_config.allow_advanced_sampling` | `<class 'bool'>` | `value` |  |  |
| `speculative_config.begin_thinking_phase_token` | `<class 'int'>` | `value` |  |  |
| `speculative_config.block_size` | `Optional[Annotated[int, Gt(gt=0)]]` | `value` |  |  |
| `speculative_config.decoding_type` | `Literal['AUTO']` | `categorical` |  | `AUTO`, `DFlash`, `DSpark`, `Draft_Target`, `Eagle3`, `Eagle`, `Lookahead`, `MTP`, `Medusa`, `NGram`, `PARD`, `SA`, `SaveState`, `User_Provided` |
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
| `speculative_config.markov_head_type` | `Optional[Literal['vanilla', 'gated', 'rnn']]` | `categorical` |  | `vanilla`, `gated`, `rnn` |
| `speculative_config.markov_rank` | `Optional[int]` | `value` |  |  |
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
