# Simulation Mode: Fidelity & Limitations

**Last updated**: 2026-04-02 (after Phase 5)

## What's Faithfully Modeled

| Behavior | How | Fidelity |
|----------|-----|----------|
| **Chunked prefill** | Real scheduler splits long contexts into chunks | High — token counts flow correctly to predictor |
| **Capacity policies** | Real KV cache manager tracks block allocation | High — GUARANTEED_NO_EVICT and MAX_UTILIZATION both work |
| **Block reuse / prefix caching** | Real radix tree detects shared prefixes | High — reduces context tokens passed to predictor |
| **Request queuing** | Real WaitingQueue (FCFS/Priority) | High — ordering is faithful |
| **Token budget** | Real PyMicroBatchScheduler enforces max_num_tokens | High — limits batch size correctly |
| **TP sharding** | SimDistributed + mapping config | High — AIC predictor receives correct tp_size |
| **Batch time prediction** | AIC or constant predictor | Medium — AIC is analytical, not profiled per-workload |

## What's NOT Modeled (Known Gaps)

| Behavior | Gap | Severity | When it matters |
|----------|-----|----------|----------------|
| **Piggybacking (mixed prefill+decode)** | Scheduler CAN produce mixed batches, but predictor treats as monolithic. Real GPU interleaves matmuls. | Medium | High-concurrency serving with continuous batching |
| **Overlap scheduling** | Single-process sim can't overlap CPU scheduling with GPU work. `disable_overlap_scheduler` is always True in sim. | Low | Latency-sensitive workloads where scheduling overhead matters |
| **KV cache eviction cost** | Scheduler correctly identifies evictions, but timing cost of save/restore is not predicted. | Medium | MAX_UTILIZATION policy under memory pressure |
| **KV cache transfer (disagg)** | Transceiver not created. HiSim models this via hicache_l2 durations. | Critical | Disaggregated serving (prefill/decode on different GPUs) |
| **Pipeline parallelism** | SimDistributed raises NotImplementedError for PP send/recv. | Critical | PP>1 configurations |
| **CUDA graph capture** | First-iteration graph capture cost not modeled. | Negligible | First few iterations only |
| **Sampling behavior** | Dummy token 0 always. Real sampling (top-k, top-p) may affect output length. | Low | When stop tokens affect output length distribution |
| **TTFT definition** | Sim TTFT = prefill + first decode (PyExecutor skips sampler during prefill). Real TTFT may differ slightly. | Low | Absolute TTFT comparison with real runs |

## Practical Limitations

### GPU Memory
- Sim mode still requires a GPU for KV cache allocation
- Default `--kv_cache_free_gpu_mem_fraction 0.90` may OOM on small GPUs or shared machines
- Use `0.40` or lower on RTX 3090 Ti / shared GPU servers
- Future: mock KV cache could eliminate GPU requirement entirely

### Process Model
- Sim forces single-process (`GenerationExecutorWorker`) regardless of TP
- This means no real NCCL communication — TP is a config parameter only
- PP>1 is not supported (would need PP send/recv)

### AIC Predictor Accuracy
- AIC uses analytical silicon performance tables, not profiled data
- Accuracy depends on hardware database quality (h100_sxm, b200_sxm, etc.)
- Constant predictor is useful for structural validation but not for absolute timing
- Calibrated constant predictor (from real run) gives ~30% accuracy

### Dataset Format
- trtllm-bench expects tokenized JSONL (`input_ids`, `output_tokens`)
- Use `trtllm-bench prepare-dataset token-norm-dist` to generate synthetic datasets
- Or pre-tokenize ShareGPT with `trtllm-bench prepare-dataset real-dataset`

### Iteration Log Format
- TRT-LLM's `--iteration_log` writes Python repr format (single quotes, `None`)
- Not valid JSON — `calibrate_sim.py` normalizes before parsing
