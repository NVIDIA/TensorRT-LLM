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

Union branches are sanitized independently. Explicit allowed values opt in only
otherwise unsafe scalar branches; they never filter a safe boolean, numeric,
`Literal`, or `Enum` branch in the same union.

If the manifest check fails, run `python3 scripts/generate_llm_args_golden_manifest.py`, then commit
`tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval.

## LLM API Configuration Fields

A field can still be absent from a specific payload when its parent config is
unset or when the safety sanitizer rejects the runtime value.

### `TorchLlmArgs`

301 captured fields.

| Captured key | Capture policy | Kind | Allowed values |
|--------------|----------------|------|----------------|
| `allreduce_strategy` | `literal\|none` | `categorical` | `AUTO`, `NCCL`, `UB`, `MINLATENCY`, `ONESHOT`, `TWOSHOT`, `LOWPRECISION`, `MNNVL`, `NCCL_SYMMETRIC` |
| `attention_dp_config.batching_wait_iters` | `int` | `value` |  |
| `attention_dp_config.enable_balance` | `bool` | `value` |  |
| `attention_dp_config.enable_kv_cache_aware_routing` | `bool` | `value` |  |
| `attention_dp_config.kv_cache_routing_account_for_in_transfer` | `bool` | `value` |  |
| `attention_dp_config.kv_cache_routing_cold_start_warmup` | `bool` | `value` |  |
| `attention_dp_config.kv_cache_routing_conversation_affinity` | `bool` | `value` |  |
| `attention_dp_config.kv_cache_routing_fair_share_multiplier` | `float` | `value` |  |
| `attention_dp_config.kv_cache_routing_load_balance_weight` | `float` | `value` |  |
| `attention_dp_config.kv_cache_routing_match_rate_threshold` | `float` | `value` |  |
| `attention_dp_config.kv_cache_routing_max_sessions` | `int` | `value` |  |
| `attention_dp_config.kv_cache_routing_new_conv_placement` | `literal` | `categorical` | `round_robin`, `least_queued` |
| `attention_dp_config.timeout_iters` | `int` | `value` |  |
| `attn_backend` | `allowlist` | `categorical` | `VANILLA`, `TRTLLM`, `FLASHINFER` |
| `backend` | `literal` | `categorical` | `pytorch` |
| `batch_wait_max_tokens_ratio` | `float` | `value` |  |
| `batch_wait_timeout_iters` | `int` | `value` |  |
| `batch_wait_timeout_ms` | `float` | `value` |  |
| `cache_transceiver_config.backend` | `literal\|none` | `categorical` | `DEFAULT`, `UCX`, `NIXL`, `MOONCAKE`, `MPI` |
| `cache_transceiver_config.enable_pipelined_transfer` | `bool` | `value` |  |
| `cache_transceiver_config.kv_cache_bounce_size_mb` | `int` | `value` |  |
| `cache_transceiver_config.kv_transfer_poll_interval_ms` | `int\|none` | `value` |  |
| `cache_transceiver_config.kv_transfer_sender_future_timeout_ms` | `int\|none` | `value` |  |
| `cache_transceiver_config.kv_transfer_timeout_ms` | `int\|none` | `value` |  |
| `cache_transceiver_config.max_tokens_in_buffer` | `int\|none` | `value` |  |
| `cache_transceiver_config.transceiver_runtime` | `literal\|none` | `categorical` | `CPP`, `PYTHON`, `auto` |
| `checkpoint_io_policy` | `literal` | `categorical` | `auto`, `native`, `rank_striped_read_ahead` |
| `context_parallel_size` | `int` | `value` |  |
| `cp_config.cp_type` | `enum` | `categorical` | `ULYSSES`, `RING`, `HELIX` |
| `cp_config.fifo_version` | `int\|none` | `value` |  |
| `cp_config.tokens_per_block` | `int\|none` | `value` |  |
| `cp_config.use_nccl_for_alltoall` | `bool\|none` | `value` |  |
| `cuda_graph_config.batch_sizes` | `list[int]\|none` | `value` |  |
| `cuda_graph_config.enable_padding` | `bool` | `value` |  |
| `cuda_graph_config.max_batch_size` | `int` | `value` |  |
| `cuda_graph_config.max_num_token` | `int` | `value` |  |
| `cuda_graph_config.max_seq_len` | `int` | `value` |  |
| `cuda_graph_config.mode` | `literal` | `categorical` | `decode`, `encode` |
| `cuda_graph_config.num_tokens` | `list[int]\|none` | `value` |  |
| `cuda_graph_config.seq_lens` | `list[int]\|none` | `value` |  |
| `disable_mm_encoder` | `bool` | `value` |  |
| `disable_overlap_scheduler` | `bool` | `value` |  |
| `dtype` | `allowlist` | `categorical` | `auto`, `float16`, `bfloat16`, `float32` |
| `dwdp_config.contention_opt` | `bool` | `value` |  |
| `dwdp_config.dwdp_size` | `int` | `value` |  |
| `dwdp_config.num_experts_per_worker` | `int` | `value` |  |
| `dwdp_config.num_groups` | `int` | `value` |  |
| `dwdp_config.num_prefetch_experts` | `int` | `value` |  |
| `enable_attention_dp` | `bool` | `value` |  |
| `enable_autotuner` | `bool` | `value` |  |
| `enable_chunked_prefill` | `bool` | `value` |  |
| `enable_early_first_token_response` | `bool` | `value` |  |
| `enable_encoder_decoder_mixed_cuda_graph` | `bool` | `value` |  |
| `enable_energy_metrics` | `bool` | `value` |  |
| `enable_in_graph_sampling` | `bool` | `value` |  |
| `enable_iter_perf_stats` | `bool` | `value` |  |
| `enable_iter_req_stats` | `bool` | `value` |  |
| `enable_layerwise_nvtx_marker` | `bool` | `value` |  |
| `enable_lm_head_tp_in_adp` | `bool` | `value` |  |
| `enable_lora` | `bool` | `value` |  |
| `enable_low_latency_host_dispatch` | `bool` | `value` |  |
| `enable_min_latency` | `bool` | `value` |  |
| `enable_mla_skip_correction` | `bool` | `value` |  |
| `enable_resource_governor` | `bool` | `value` |  |
| `enable_speculative_beam_history_d2h` | `bool` | `value` |  |
| `encode_only` | `bool` | `value` |  |
| `encoder_cuda_graph_config.batch_sizes` | `list[int]\|none` | `value` |  |
| `encoder_cuda_graph_config.enable_padding` | `bool` | `value` |  |
| `encoder_cuda_graph_config.max_batch_size` | `int` | `value` |  |
| `encoder_cuda_graph_config.max_num_token` | `int` | `value` |  |
| `encoder_cuda_graph_config.max_seq_len` | `int` | `value` |  |
| `encoder_cuda_graph_config.mode` | `literal` | `categorical` | `encode` |
| `encoder_cuda_graph_config.num_tokens` | `list[int]\|none` | `value` |  |
| `encoder_cuda_graph_config.seq_lens` | `list[int]\|none` | `value` |  |
| `encoder_max_batch_size` | `int\|none` | `value` |  |
| `encoder_max_num_tokens` | `int\|none` | `value` |  |
| `force_dynamic_quantization` | `bool` | `value` |  |
| `garbage_collection_gen0_threshold` | `int` | `value` |  |
| `gather_generation_logits` | `bool` | `value` |  |
| `generation_config` | `literal` | `categorical` | `auto`, `trtllm` |
| `gms_config.mode` | `literal` | `categorical` | `auto`, `rw`, `ro` |
| `gpus_per_node` | `int\|none` | `value` |  |
| `guided_decoding_backend` | `literal\|none` | `categorical` | `xgrammar`, `llguidance` |
| `iter_stats_max_iterations` | `int\|none` | `value` |  |
| `kv_cache_compression_config.algorithm` | `literal` | `categorical` | `quantization_for_cold_page`, `triattention` |
| `kv_cache_compression_config.beta` | `int` | `value` |  |
| `kv_cache_compression_config.budget` | `int` | `value` |  |
| `kv_cache_compression_config.eviction_mode` | `literal` | `categorical` | `union`, `per_head`, `per_layer_perhead` |
| `kv_cache_compression_config.normalize_scores` | `bool` | `value` |  |
| `kv_cache_compression_config.quant` | `literal` | `categorical` | `nvfp4` |
| `kv_cache_config.attention_dp_events_gather_period_ms` | `int` | `value` |  |
| `kv_cache_config.avg_seq_len` | `int\|none` | `value` |  |
| `kv_cache_config.block_reuse_config.max_num_turns` | `int` | `value` |  |
| `kv_cache_config.block_reuse_config.policy` | `literal` | `categorical` | `all_reusable`, `per_request`, `per_conversation` |
| `kv_cache_config.copy_on_partial_reuse` | `bool` | `value` |  |
| `kv_cache_config.cross_kv_cache_fraction` | `float\|none` | `value` |  |
| `kv_cache_config.disk_cache_size` | `int\|none` | `value` |  |
| `kv_cache_config.disk_prefetch_num_reqs` | `int` | `value` |  |
| `kv_cache_config.dtype` | `allowlist` | `categorical` | `auto`, `float16`, `bfloat16`, `float32`, `fp8`, `fp8_ds_mla`, `nvfp4` |
| `kv_cache_config.enable_block_reuse` | `bool` | `value` |  |
| `kv_cache_config.enable_kv_pool_rebalance` | `bool` | `value` |  |
| `kv_cache_config.enable_partial_reuse` | `bool` | `value` |  |
| `kv_cache_config.enable_swa_scratch_reuse` | `bool` | `value` |  |
| `kv_cache_config.event_buffer_max_size` | `int` | `value` |  |
| `kv_cache_config.fp8_context_mla_kv_len_cap` | `int\|none` | `value` |  |
| `kv_cache_config.free_gpu_memory_fraction` | `float\|none` | `value` |  |
| `kv_cache_config.host_cache_size` | `int\|none` | `value` |  |
| `kv_cache_config.iteration_stats_interval` | `int` | `value` |  |
| `kv_cache_config.kv_cache_event_hash_algo` | `literal` | `categorical` | `auto`, `v1_block_key`, `v2_sha256`, `v2_sha256_64` |
| `kv_cache_config.mamba_ssm_cache_dtype` | `literal` | `categorical` | `auto`, `float16`, `bfloat16`, `float32` |
| `kv_cache_config.mamba_ssm_philox_rounds` | `int` | `value` |  |
| `kv_cache_config.mamba_ssm_stochastic_rounding` | `bool` | `value` |  |
| `kv_cache_config.mamba_state_config.enable_branch_snapshot` | `bool` | `value` |  |
| `kv_cache_config.mamba_state_config.periodic_snapshot_interval` | `int` | `value` |  |
| `kv_cache_config.max_attention_window` | `list[int]\|none` | `value` |  |
| `kv_cache_config.max_gpu_total_bytes` | `int` | `value` |  |
| `kv_cache_config.max_tokens` | `int\|none` | `value` |  |
| `kv_cache_config.max_util_for_resume` | `float` | `value` |  |
| `kv_cache_config.pool_ratio` | `list[float]\|none` | `value` |  |
| `kv_cache_config.secondary_offload_min_priority` | `int\|none` | `value` |  |
| `kv_cache_config.sink_token_length` | `int\|none` | `value` |  |
| `kv_cache_config.tokens_per_block` | `int` | `value` |  |
| `kv_cache_config.use_kv_cache_manager_v2` | `bool\|literal` | `categorical` | `auto` |
| `kv_cache_config.use_uvm` | `bool` | `value` |  |
| `kv_connector_config.connector` | `allowlist\|none` | `categorical` | `lmcache`, `lmcache-mp`, `kvbm` |
| `layer_wise_benchmarks_config.calibration_layer_indices` | `list[int]\|none` | `value` |  |
| `layer_wise_benchmarks_config.calibration_mode` | `literal` | `categorical` | `NONE`, `MARK`, `COLLECT` |
| `load_format` | `allowlist\|enum` | `categorical` | `auto`, `dummy`, `vision_only`, `gms`, `AUTO`, `DUMMY`, `VISION_ONLY`, `GMS` |
| `lora_config.cuda_graph_specialize_lora` | `bool` | `value` |  |
| `lora_config.lora_ckpt_source` | `literal` | `categorical` | `hf`, `nemo` |
| `lora_config.max_cpu_loras` | `int\|none` | `value` |  |
| `lora_config.max_lora_rank` | `int` | `value` |  |
| `lora_config.max_loras` | `int\|none` | `value` |  |
| `lora_config.overlap_lora_and_base` | `bool` | `value` |  |
| `lora_config.swap_gate_up_proj_lora_b_weight` | `bool` | `value` |  |
| `max_batch_size` | `int\|none` | `value` |  |
| `max_beam_width` | `int\|none` | `value` |  |
| `max_input_len` | `int\|none` | `value` |  |
| `max_num_tokens` | `int\|none` | `value` |  |
| `max_seq_len` | `int\|none` | `value` |  |
| `max_stats_len` | `int` | `value` |  |
| `mla_skip_correction_threshold` | `float` | `value` |  |
| `mm_encoder_only` | `bool` | `value` |  |
| `moe_cluster_parallel_size` | `int\|none` | `value` |  |
| `moe_config.backend` | `literal` | `categorical` | `AUTO`, `CUTLASS`, `CUTEDSL`, `TRTLLM`, `DEEPGEMM`, `DENSEGEMM`, `VANILLA`, `TRITON`, `MARLIN`, `MEGAMOE_DEEPGEMM`, `MEGAMOE_CUTEDSL` |
| `moe_config.disable_finalize_fusion` | `bool` | `value` |  |
| `moe_config.max_num_tokens` | `int\|none` | `value` |  |
| `moe_config.use_low_precision_moe_combine` | `bool` | `value` |  |
| `moe_expert_parallel_size` | `int\|none` | `value` |  |
| `moe_tensor_parallel_size` | `int\|none` | `value` |  |
| `multimodal_config.encoder_cache_max_bytes` | `int` | `value` |  |
| `multimodal_config.encoder_scheduling_policy` | `enum` | `categorical` | `DISABLED`, `DEFAULT`, `EAGER` |
| `multimodal_config.encoder_side_stream_max_ahead` | `int` | `value` |  |
| `multimodal_config.video_pruning_rate` | `float\|none` | `value` |  |
| `mx_config.preshard_strategy` | `allowlist` | `categorical` | `per_module` |
| `mx_config.server_query_timeout_s` | `int\|none` | `value` |  |
| `num_postprocess_workers` | `int` | `value` |  |
| `num_serve_frontends` | `int` | `value` |  |
| `nvfp4_gemm_config.allowed_backends` | `list[literal]` | `categorical` | `cutlass`, `cublaslt`, `cutedsl`, `cuda_core`, `marlin` |
| `orchestrator_type` | `literal\|none` | `categorical` | `rpc`, `ray` |
| `peft_cache_config.device_cache_percent` | `float` | `value` |  |
| `peft_cache_config.host_cache_size` | `int` | `value` |  |
| `peft_cache_config.max_adapter_size` | `int` | `value` |  |
| `peft_cache_config.max_pages_per_block_device` | `int` | `value` |  |
| `peft_cache_config.max_pages_per_block_host` | `int` | `value` |  |
| `peft_cache_config.num_copy_streams` | `int` | `value` |  |
| `peft_cache_config.num_device_module_layer` | `int` | `value` |  |
| `peft_cache_config.num_ensure_workers` | `int` | `value` |  |
| `peft_cache_config.num_host_module_layer` | `int` | `value` |  |
| `peft_cache_config.num_put_workers` | `int` | `value` |  |
| `peft_cache_config.optimal_adapter_size` | `int` | `value` |  |
| `perf_metrics_max_requests` | `int` | `value` |  |
| `pipeline_parallel_size` | `int` | `value` |  |
| `pp_partition` | `list[int]\|none` | `value` |  |
| `prefill_capture_num_tokens` | `list[int]\|none` | `value` |  |
| `prefill_cuda_graph_backend` | `enum` | `categorical` | `disabled`, `piecewise`, `breakable` |
| `print_iter_log` | `bool` | `value` |  |
| `prometheus_metrics_config.e2e_request_latency_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.request_decode_time_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.request_inference_time_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.request_prefill_time_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.request_queue_time_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.time_per_output_token_buckets` | `list[float]\|none` | `value` |  |
| `prometheus_metrics_config.time_to_first_token_buckets` | `list[float]\|none` | `value` |  |
| `ray_placement_config.defer_workers_init` | `bool` | `value` |  |
| `ray_placement_config.per_worker_gpu_share` | `float\|none` | `value` |  |
| `ray_placement_config.placement_bundle_indices` | `list[list[int]]\|none` | `value` |  |
| `reasoning_parser` | `allowlist\|none` | `categorical` | `auto`, `deepseek-r1`, `poolside_v1`, `laguna`, `qwen3`, `qwen3_5`, `minimax_m2`, `minimax_m2_append_think`, `nano-v3`, `gemma4`, `kimi_k2`, `kimi_k25` |
| `reorder_policy_config.policy_args.agent_inflight_seq_num` | `int` | `value` |  |
| `reorder_policy_config.policy_args.agent_percentage` | `float` | `value` |  |
| `reorder_policy_config.policy_name` | `literal\|none` | `categorical` | `AgentTree` |
| `request_stats_max_iterations` | `int\|none` | `value` |  |
| `return_perf_metrics` | `bool` | `value` |  |
| `sampler_force_async_worker` | `bool` | `value` |  |
| `scheduler_config.capacity_scheduler_policy` | `enum` | `categorical` | `MAX_UTILIZATION`, `GUARANTEED_NO_EVICT`, `STATIC_BATCH` |
| `scheduler_config.context_chunking_policy` | `enum\|none` | `categorical` | `FIRST_COME_FIRST_SERVED`, `EQUAL_PROGRESS`, `FORCE_CHUNK` |
| `scheduler_config.dynamic_batch_config.dynamic_batch_moving_average_window` | `int` | `value` |  |
| `scheduler_config.dynamic_batch_config.enable_batch_size_tuning` | `bool` | `value` |  |
| `scheduler_config.dynamic_batch_config.enable_max_num_tokens_tuning` | `bool` | `value` |  |
| `scheduler_config.enable_prefix_aware_scheduling` | `bool` | `value` |  |
| `scheduler_config.use_python_scheduler` | `bool` | `value` |  |
| `scheduler_config.waiting_queue_policy` | `enum` | `categorical` | `fcfs`, `priority` |
| `skip_tokenizer_init` | `bool` | `value` |  |
| `sparse_attention_config.algorithm` | `literal` | `categorical` | `dsa`, `deepseek_v4`, `minimax_m3`, `qsa`, `rocket`, `skip_softmax` |
| `sparse_attention_config.compress_ratios` | `list[int]` | `value` |  |
| `sparse_attention_config.enable_heuristic_topk` | `bool` | `value` |  |
| `sparse_attention_config.implementation` | `literal` | `categorical` | `triton`, `msa` |
| `sparse_attention_config.index_head_dim` | `int\|none` | `value` |  |
| `sparse_attention_config.index_n_heads` | `int\|none` | `value` |  |
| `sparse_attention_config.index_share_for_mtp_iteration` | `bool\|none` | `value` |  |
| `sparse_attention_config.index_topk` | `int\|none` | `value` |  |
| `sparse_attention_config.indexer_k_dtype` | `literal` | `categorical` | `fp8`, `fp4` |
| `sparse_attention_config.indexer_kv_dtype` | `literal` | `categorical` | `bf16`, `fp8` |
| `sparse_attention_config.indexer_max_chunk_size` | `int\|none` | `value` |  |
| `sparse_attention_config.indexer_rope_interleave` | `bool` | `value` |  |
| `sparse_attention_config.kernel_size` | `int\|none` | `value` |  |
| `sparse_attention_config.kt_cache_dtype` | `allowlist\|none` | `categorical` | `bfloat16`, `float8_e5m2` |
| `sparse_attention_config.num_attention_heads` | `int\|none` | `value` |  |
| `sparse_attention_config.num_key_value_heads` | `int\|none` | `value` |  |
| `sparse_attention_config.page_size` | `int\|none` | `value` |  |
| `sparse_attention_config.prompt_budget` | `int\|none` | `value` |  |
| `sparse_attention_config.q_split_threshold` | `int` | `value` |  |
| `sparse_attention_config.seq_len_threshold` | `int\|none` | `value` |  |
| `sparse_attention_config.skip_indexer_for_short_seqs` | `bool` | `value` |  |
| `sparse_attention_config.sparse_block_size` | `int` | `value` |  |
| `sparse_attention_config.sparse_disable_index_value` | `bool` | `value` |  |
| `sparse_attention_config.sparse_index_dim` | `int` | `value` |  |
| `sparse_attention_config.sparse_init_blocks` | `int` | `value` |  |
| `sparse_attention_config.sparse_local_blocks` | `int` | `value` |  |
| `sparse_attention_config.sparse_num_index_heads` | `int` | `value` |  |
| `sparse_attention_config.sparse_score_type` | `literal` | `categorical` | `max` |
| `sparse_attention_config.sparse_topk_blocks` | `int` | `value` |  |
| `sparse_attention_config.target_sparsity` | `float\|none` | `value` |  |
| `sparse_attention_config.threshold_scale_factor` | `float\|none` | `value` |  |
| `sparse_attention_config.topk` | `int\|none` | `value` |  |
| `sparse_attention_config.topr` | `float\|int\|none` | `value` |  |
| `sparse_attention_config.use_cute_dsl_paged_mqa_logits` | `bool` | `value` |  |
| `sparse_attention_config.use_cute_dsl_topk` | `bool` | `value` |  |
| `sparse_attention_config.use_gvr_emission` | `bool` | `value` |  |
| `sparse_attention_config.use_self_sampling_topk` | `bool` | `value` |  |
| `sparse_attention_config.window_size` | `int\|none` | `value` |  |
| `speculative_config.acceptance_rate_threshold` | `float\|none` | `value` |  |
| `speculative_config.acceptance_rate_window_size` | `int\|none` | `value` |  |
| `speculative_config.advanced_sampling_mode` | `enum` | `categorical` | `full`, `no_topk`, `no_topp`, `no_topk_no_topp` |
| `speculative_config.allow_advanced_sampling` | `bool` | `value` |  |
| `speculative_config.attention_backend` | `literal` | `categorical` | `VANILLA`, `TRTLLM` |
| `speculative_config.begin_thinking_phase_token` | `int` | `value` |  |
| `speculative_config.block_size` | `int\|none` | `value` |  |
| `speculative_config.decoding_type` | `literal` | `categorical` | `AUTO`, `DFlash`, `DSpark`, `Draft_Target`, `Eagle3`, `Eagle`, `MTP`, `NGram`, `PARD`, `SA`, `SaveState`, `User_Provided` |
| `speculative_config.dynamic_tree_max_topK` | `int\|none` | `value` |  |
| `speculative_config.eagle3_layers_to_capture` | `none\|set[int]` | `value` |  |
| `speculative_config.eagle3_model_arch` | `literal` | `categorical` | `llama3`, `mistral_large3` |
| `speculative_config.eagle3_one_model` | `bool\|none` | `value` |  |
| `speculative_config.eagle_choices` | `list[list[int]]\|none` | `value` |  |
| `speculative_config.enable_global_pool` | `bool` | `value` |  |
| `speculative_config.enable_penalty` | `bool` | `value` |  |
| `speculative_config.end_thinking_phase_token` | `int` | `value` |  |
| `speculative_config.global_pool_size` | `int\|none` | `value` |  |
| `speculative_config.greedy_sampling` | `bool\|none` | `value` |  |
| `speculative_config.is_keep_all` | `bool` | `value` |  |
| `speculative_config.is_public_pool` | `bool` | `value` |  |
| `speculative_config.is_use_oldest` | `bool` | `value` |  |
| `speculative_config.markov_head_type` | `literal\|none` | `categorical` | `vanilla`, `gated`, `rnn` |
| `speculative_config.markov_rank` | `int\|none` | `value` |  |
| `speculative_config.mask_token_id` | `int\|none` | `value` |  |
| `speculative_config.max_concurrency` | `int\|none` | `value` |  |
| `speculative_config.max_draft_len` | `int\|none` | `value` |  |
| `speculative_config.max_matching_ngram_size` | `int` | `value` |  |
| `speculative_config.max_non_leaves_per_layer` | `int\|none` | `value` |  |
| `speculative_config.max_total_draft_tokens` | `int\|none` | `value` |  |
| `speculative_config.mtp_eagle_one_model` | `bool` | `value` |  |
| `speculative_config.num_eagle_layers` | `int\|none` | `value` |  |
| `speculative_config.num_nextn_predict_layers` | `int\|none` | `value` |  |
| `speculative_config.posterior_threshold` | `float\|none` | `value` |  |
| `speculative_config.relaxed_delta` | `float` | `value` |  |
| `speculative_config.relaxed_topk` | `int` | `value` |  |
| `speculative_config.sa_config.enable_global_pool` | `bool` | `value` |  |
| `speculative_config.sa_config.threshold` | `int` | `value` |  |
| `speculative_config.target_layer_ids` | `list[int]\|none` | `value` |  |
| `speculative_config.use_dynamic_tree` | `bool\|none` | `value` |  |
| `speculative_config.use_mtp_vanilla` | `bool` | `value` |  |
| `speculative_config.use_rejection_sampling` | `bool` | `value` |  |
| `speculative_config.use_relaxed_acceptance_for_thinking` | `bool` | `value` |  |
| `speculative_config.write_interval` | `int` | `value` |  |
| `stream_interval` | `int` | `value` |  |
| `telemetry_config.disabled` | `bool` | `value` |  |
| `telemetry_config.usage_context` | `enum` | `categorical` | `unknown`, `llm_class`, `cli_serve`, `cli_bench`, `cli_eval`, `disaggregated` |
| `tensor_parallel_size` | `int` | `value` |  |
| `tokenizer_mode` | `literal` | `categorical` | `auto`, `slow` |
| `torch_compile_config.capture_num_tokens` | `list[int]\|none` | `value` |  |
| `torch_compile_config.enable_fullgraph` | `bool` | `value` |  |
| `torch_compile_config.enable_inductor` | `bool` | `value` |  |
| `torch_compile_config.enable_piecewise_cuda_graph` | `bool` | `value` |  |
| `torch_compile_config.enable_userbuffers` | `bool` | `value` |  |
| `torch_compile_config.max_num_streams` | `int` | `value` |  |
| `trust_remote_code` | `bool` | `value` |  |
| `use_cute_dsl_bf16_bmm` | `bool` | `value` |  |
| `use_cute_dsl_bf16_gemm` | `bool` | `value` |  |
| `use_cute_dsl_blockscaling_bmm` | `bool` | `value` |  |
| `use_cute_dsl_blockscaling_mm` | `bool` | `value` |  |
| `use_fine_grained_sync` | `bool` | `value` |  |
