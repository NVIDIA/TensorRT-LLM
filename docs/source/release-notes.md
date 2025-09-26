(release-notes)=

# Release Notes

All published functionality in the Release Notes has been fully tested and verified with known limitations documented. To share feedback about this release, access our [NVIDIA Developer Forum](https://forums.developer.nvidia.com/).

## TensorRT-LLM Release 1.0

TensorRT LLM 1.0 brings 2 major changes: the PyTorch-based architecture is now stable and the default experience, and the LLM API is now stable. For more details on new developments in 1.0, please see below.

### Key Features and Enhancements
- **Model Support**
  - Add Mistral3.1 VLM model support
  - Add TensorRT-Engine Qwen3 (dense) model support
  - Add phi-4-multimodal model support
  - Add EXAONE 4.0 model support
  - Add Qwen3 MoE support to TensorRT backend

- **Features**
  - Add support for sm121
  - Add LoRA support for Gemma3
  - Support PyTorch LoRA adapter eviction
  - Add LoRA support for PyTorch backend in trtllm-serve 
  - Add support of scheduling attention dp request
  - Remove padding of FusedMoE in attention DP
  - Support torch compile for attention dp
  - Add KV events support for sliding window attention
  - Add TRTLLM MoE nvfp4 cubins for mid-high concurrency; attention_dp for TRTLLM MoE
  - Add Piecewise CUDA Graph support for MLA
  - Support mutliCtasKvMode for high-throughput MLA kernels
  - Enable kvcache to be reused during request generation
  - Add ADP schedule balance optimization
  - Add chunked prefill support for MLA (Blackwell)
  - Enable Multi-block mode for Hopper spec dec XQA kernel
  - Add vLLM KV Pool support for XQA kernel
  - Allow sending more than 2GiB through MPI by using mpi4py.util.pkl5
  - Add support for fused gate_up_proj scales for FP8 blockwise
  - Support FP8 row-wise dense GEMM in torch flow
  - Enable fp8 SwiGLU to minimize host overhead
  - Add Deepseek R1 FP8 Support on Blackwell
  - Add support for MXFP8xMXFP4 in pytorch
  - Support nvfp4 model and fp8 kv cache for MLA chunked prefill (Blackwell)
  - Opensource MOE MXFP8-MXFP4 implementation
  - Add support for Modelopt fp8_pb_wo quantization scheme
  - Support deepEP fp4 post quant all2all dispatch
  - Fuse w4a8 moe pre-quant scale on Hopper
  - Support Weight-Only-Quantization in PyTorch Workflow
  - Add support for per expert activation scaling factors
  - Add ReDrafter support for Qwen
  - Enable CUDA Graph for Nemotron-H
  - Add support for YARN in NemotronNAS models
  - Switch to internal version of MMProjector in Gemma3
  - Disable add special tokens for Llama3.3 70B
  - Auto-enable ngram with concurrency <= 32
  - Support turning on/off spec decoding dynamically
  - Support structural tag in C++ runtime and upgrade xgrammar to 0.1.21
  - Add support for external multimodal embeddings
  - Add support for disaggregation with pp with pytorch backend
  - Add status tags to LLM API reference
  - Support JSON Schema in OpenAI-Compatible API
  - Support chunked prefill on spec decode 2 model
  - Add KV cache reuse support for multimodal models 
  - Support nanobind bindings
  - Add support for two-model engine KV cache reuse
  - Add Eagle-3 support for qwen3 dense model
  - Migrate Eagle-3 and draft/target speculation to Drafter
  - Enable guided decoding with overlap scheduler
  - Support n-gram speculative decoding with disagg
  - Add beam search support to the PyTorch Workflow
  - Add LLGuidance Support for PyTorch Backend
  - Add NGrams V2 support
  - Add MTP support for Online EPLB
  - Support disaggregated serving in TRTLLM Sampler
  - Add core infrastructure to enable loading of custom checkpoint formats
  - Support TRTLLM_DEEP_EP_TOKEN_LIMIT to allow run deep-ep on memory-constrained GPUs
  - Use huge page mapping for host accessible memory on GB200
  - Add user-provided speculative decoding support
  - Add streaming scaffolding_llm.generate_async support
  - Detokenize option in /v1/completions request
  - Integrate TRT-LLM Gen FP4 block scale MoE with Pytorch workflow kernel autotuner
  - Remove support for llmapi + TRT backend in Triton
  - Add request_perf_metrics to triton LLMAPI backend 
  - Add support for Triton request cancellation

- Benchmark:
  - Add support for benchmarking individual gemms in MOE benchmark (#6080)
  - Add speculative metrics for trtllm-bench
  - Add the ability to write a request timeline for trtllm-bench
  - Add no_kv_cache_reuse option and streaming support for trtllm-serve bench
  - Add latency support for trtllm-bench
  - Add Acceptance Rate calculation to benchmark_serving 
  - Add wide-ep benchmarking scripts
  - Update trtllm-bench to support new Pytorch default
  - Add support for TRTLLM CustomDataset
  - Make benchmark_serving part of the library

- Documentation:
  - Refactored the doc structure to focus on the PyTorch workflow.
  - Improved the LLMAPI and API reference documentation. Stable APIs are now protected and will remain consistent in subsequent versions following v1.0.
  - Removed legacy documentation related to the TensorRT workflow.

### Infrastructure Changes
- The base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:25.06-py3`.
- The base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:25.06-py3`.
- The dependent NVIDIA ModelOpt version is updated to 0.33.
- The dependent xgrammar version is updated to 0.1.21.
- The dependent transformers version is updated to 4.53.1.

### API Changes
- **BREAKING CHANGE** Promote PyTorch to be the default LLM backend
- **BREAKING CHANGE** Change default backend to PyTorch in trtllm-serve
- **BREAKING CHANGE** Unify KvCacheConfig in LLM class for pytorch backend
- **BREAKING CHANGE** Rename cuda_graph_config padding_enabled field
- **BREAKING CHANGE** Rename mixed_sampler to enable_mixed_sampler
- **BREAKING CHANGE** Rename LLM.autotuner_enabled to enable_autotuner
- Add back allreduce_strategy parameter into TorchLlmArgs
- Add LLmArgs option to force using dynamic quantization
- Change default LoRA cache sizes and change peft_cache_config cache size fields to take effect when not explicitly set in lora_config
- Remove deprecated LoRA LLM args, that are already specified in lora_config
- Add request_perf_metrics to LLMAPI
- Remove batch_manager::KvCacheConfig and use executor::KvCacheConfig instead 
- Remove TrtGptModelOptionalParams 
- Remove ptuning knobs from TorchLlmArgs


### Fixed Issues
- Fix illegal memory access in MLA (#6437)
- Fix nemotronNAS loading for TP>1 (#6447)
- Fix wide EP when using DeepEP with online EPLB (#6429)
- Fix bugs caused by None attention_bias during Qwen3 model convert engine (#6344)
- Fix PD + MTP + overlap scheduler accuracy issue (#6136)
- Fix bug of Qwen3 when using fp4 on sm120 (#6065)
- Fix TMA error with GEMM+AR on TP=2 (#6075)
- Fix scaffolding aime test in test_e2e (#6140)
- Fix KV Cache overrides in trtllm-bench (#6103)
- Fix MOE benchmark to rotate buffers to prevent L2 cache reuse (#4135)
- Fix eagle3 two model disaggregated serving test (#6014)
- Fix chunked prefill + overlap scheduling (#5761)
- Fix mgmn postprocess error (#5835)
- Fallback to cubins for fp8 fmha kernels on Ada (#5779)
- Fix disagg + speculative decoding (#5558)
- Fix test_generate_with_seed CI failure. (#5772)
- Fix prompt adapter TP2 case (#5782)
- Fix disaggregate serving with attention DP (#4993)
- Fix a quote error introduced in #5534 (#5816)
- Fix the accuracy issue when reduce_fusion is enabled for GEMMA model. (#5801)
- Fix lost requests for disaggregated serving (#5815)
- Update unit tests: skip all_close assert for dropout in attention, increase tolerance for rope op test (#5855)
- Fix GEMM+AR fusion on blackwell (#5563)
- Fix llama4 multimodal support (#5809)
- Fix Llama4 Scout FP4 crash issue (#5925)
- Fix max batch size and max tokens in kv cache estimations for Nemotron-H (#5371)
- Fix moe regression for sm120 (#5823)
- Fix Qwen2.5VL FP8 support (#5029)
- Fix the illegal memory access issue in moe gemm on SM120 (#5636)
- Fix tileN cannot % 16==0 & support sm89 deepgemm bmm (#5531)
- Fix incremental detokenization (#5825)
- Fix MoE workspace info by storing Torch tensor itself instead of data_ptr (#5900)
- Fix mistral unit tests due to transformers upgrade (#5904)
- Fix the Llama3.1 405B hanging issue. (#5698) (#5925)
- Fix Gemma3 unit tests due to transformers upgrade (#5921)
- Fix alltoall for llama4 (apply_router_weight_on_input=True) (#5902)
- Remove SpecConfig and fix thread leak issues (#5931)
- Fast redux detection in trtllm gen routing kernel (#5941)
- Fix cancel request logic (#5800)
- Fix errors in wide-ep scripts (#5992)
- Fix error in post-merge-tests (#5949)
- Fix missing arg to alltoall_prepare_maybe_dispatch (#5669)
- Fix attention DP doesn't work with embedding TP (#5642)
- Fix broken cyclic reference detect (#5417) 
- Fix permission for local user issues in NGC docker container. (#5373)
- Fix mtp vanilla draft inputs (#5568) 
- Fix mPtrExpertCounts allocation in MoE TRT-LLM backend (nvfp4) (#5519) 
- Fix block scale fp8 support for deepseek v3 on Blackwell. (#5514)
- Fix the issue MoE autotune fallback failed to query default heuristic (#5520) 
- Fix the unexpected keyword argument 'streaming' (#5436)

### Known Issues
- When using disaggregated serving with pipeline parallelism and KV cache reuse, a hang can occur. This will be fixed in a future release. In the meantime, disabling KV cache reuse will fix this issue.
- Running multi-node cases where each node has just a single GPU is known to fail. This will be addressed in a future release. 
- For the Llama 3.x and Llama 4 models, there is an issue with pipeline parallelism when using FP8 and NVFP4 weights. As a workaround, you can set the environment variable `export TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1`.

## TensorRT-LLM Release 0.21.0

### Key Features and Enhancements
- **Model Support**
  - Added Gemma3 VLM support
- **Features**
  - Added large-scale EP support
  - Integrated NIXL into the communication layer of the disaggregated service
  - Added fabric Memory support for KV Cache Transfer
  - Added MCP in ScaffoldingLLM
  - Added support for w4a8_mxfp4_fp8 quantization
  - Added support for fp8 rowwise quantization
  - Added generation logits support in TRTLLM Sampler
  - Added log probs support in TRTLLM Sampler
  - Optimized TRTLLM Sampler perf single beam single step
  - Enabled Disaggregated serving for Qwen-3
  - Added EAGLE3 support for Qwen-3
  - Fused finalize and allreduce for Qwen-MoE model
  - Refactored Fused MoE module
  - Added support for chunked attention on Blackwell and Hopper
  - Introduced sliding-window attention kernels for the generation phase on Blackwell
  - Updated DeepSeek FP8 TRT-LLM Gen cubins to improve performance in large batch size scenarios
  - Added FP8 block-scale GEMM support on SM89
  - Enabled overlap scheduler between draft forwards
  - Added Piecewise cuda graph support for MLA
  - Added model-agnostic one-engine eagle3
  - Enabled Finalize + Allreduce + add + rmsnorm fusion
  - Integrated TRT-LLM Gen FP8 block scale MoE with Pytorch workflow kernel autotuner
  - Added support for Eagle3 + disaggregated serving in two model speculative decoding flow
  - Validated Llama 3.1 models on H200 NVL
- Benchmark:
  - Added all_reduce.py benchmark script for testing
  - Added beam width to trtllm-bench latency command
  - Fixed trtllm-bench iter_stats and cuda_graph_batch_sizes errors
  - Enabled trtllm-bench to run LoRA and add basic e2e perf testing capability for LoRA
  - Supported post_proc for bench
  - Added no_kv_cache_reuse option and streaming support for trtllm serve bench

### Infrastructure Changes
- The base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:25.05-py3`.
- The base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:25.05-py3`.
- The dependent public PyTorch version is updated to 2.7.1.
- The dependent TensorRT version is updated to 10.11.
- The dependent NVIDIA ModelOpt version is updated to 0.31.
- The dependent NCCL version is updated to 2.27.5.

### API Changes
- Set _AutoDeployLlmArgs as primary config object
- Removed decoder request from decoder interface
- Enhanced the torch_compile_config in llm args
- Removed the redundant use_kv_cache field from PytorchConfig
- Moved allreduce_strategy from committed api to reference

### Fixed Issues
- Fixed disaggregated service hang when MNNVL two-shot AllReduce is enabled (#4678)
- Fixed EP load balancer with MTP layer and route offset by EP rank (#4767)
- Fixed cuda graph padding for spec decoding (#4853)
- Fixed llama 4 long context issue (#4809)
- Fixed max_num_sequences calculation with overlap scheduling (#4532)
- Fixed chunked prefill + overlap scheduling (#5761)
- Fixed trtllm-bench hang issue due to LLM API IPC (#4798)
- Fixed index out of bounds error in spec decoding (#5954)
- Fixed MTP illegal memory access in cuda graph warmup (#5947)
- Fixed no free slots error with spec decode + disagg (#5975)
- Fixed one-off attention window size for Gemma3 1B (#5564)

### Known Issues
- accuracy/test_cli_flow::TestGpt2::test_beam_search_large is broken.
- Enabling disaggregated serving, MTP, and the overlap scheduler at the same time can lead to accuracy problems.
- In 0.21, full chunked attention support has been added to make sure LLaMA4 model can functionally run with > 8K seq length, while there is a known performance regression(only affect LLaMA4 model) on Hopper due to this functional enhancement. The root cause of the regression has been identified already and the fix will be part of the future release.

## TensorRT-LLM Release 0.20.0

### Key Features and Enhancements
- **Model Support**
  - Added Qwen3 support.Refer to “Qwen3” section in `examples/models/core/qwen/README.md`.
  - Added HyperCLOVAX-SEED-Vision support in PyTorch flow. Refer to `examples/models/contrib/hyperclovax/README.md`
  - Added Dynasor-CoT in scaffolding examples. Refer to `examples/scaffolding/contrib/Dynasor/README.md`
  - Added Mistral Small 3.1 24B VLM support in TRT workflow
  - Added Gemma3-1b-it support in PyTorch workflow
  - Added Nemotron-H model support
  - Added Eagle-3 support for LLAMA4
- **PyTorch workflow**
  - Added lora support
  - Added return logits support
  - Adopt new logprob definition in PyTorch flow
  - Enabled per-request stats with PyTorch backend
  - Enabled LogitsProcessor in PyTorch backend
- Benchmark:
  - Add beam width to low latency.
  - Fix trtllm-bench iter_stats and cuda_graph_batch_sizes errors.
  - Remove deprecated Python runtime benchmark
  - Add benchmark support for scaffolding
- Multimodal models
  - Added support in trtllm-serve
  - Added support in trtllm-bench, the support is limited to image only for now
- Supported DeepSeek-R1 W4A8 on Hopper
- Add the RTX Pro 6000 support on single GPU
- Integrated Llama4 input processor
- Added CGA reduction FHMA kernels on Blackwell
- Enabled chunked context for FlashInfer
- Supported KV cache reuse for MLA
- Added Piecewise CUDA Graph support
- Supported multiple LoRA adapters and TP
- Added KV cache-aware router for disaggregated serving
- Unfused attention for native support
- Added group_rms_norm kernel to normalize multiple inputs in a single operator
- Added smart router for the MoE module
- Added head size 72 support for QKV preprocessing kernel
- Added MNNVL MoE A2A support
- Optimized Large Embedding Tables in Multimodal Models
- Supported Top-K logprobs and prompt_logprobs in LLMAPI
- Enabled overlap scheduler in TRT workflow via executor API

### Infrastructure Changes
- **TRT-LLM team formally releases docker image on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)**.
- The pre-built TensorRT-LLM wheel on PyPI is linked against PyTorch 2.7.0 now, which uses the CXX11 ABI
- The dependent TensorRT version is updated to 10.10.0
- The dependent CUDA version is updated to 12.9.0
- The dependent public PyTorch version is updated to 2.7.0
- The dependent NVIDIA ModelOpt version is updated to 0.29.0
- The dependent NCCL version is maintained at 2.25.1
- Open-sourced XQA kernels
- Dependent datasets version was upgraded to 3.1.0
- Migrate Triton Backend to TensorRT LLM repo to TensorRT LLM submodule
- Downgrade gcc toolset version from 13 to 11

### API Changes
- [Breaking Change]:Enable scheduling overlap by default
- Remove deprecated GptSession/V1 from TRT workflow
- Set _AutoDeployLlmArgs as primary config object
- Allow overriding CLI arguments with YAML file in trtllm-serve
- Introduced multimodal embedding field in LlmRequest


### Fixed Issues
- Fix hang bug when context server doesn't have enough capacity for KV Cache (#3095)
- Fix C++ decoder synchronization in PyTorch (#3106)
- Fix bug of create cuda stream as default parameter which will be initialized during importing (#3764)
- Fix bug related to creating CUDA stream as default parameter, which will be initialized during importing (#3764)
- Fix attention DP bug on Qwen3 MoE model (#4141)
- Fix illegal memory access when running LLaMA 4 with CUDA Graph enabled (#4101)
- Reset planned states to avoid memory leak in TrtllmAttentionWrapper (#4227)

### Known Issues
- multi-GPU model support on RTX Pro 6000


## TensorRT-LLM Release 0.19.0

### Key Features and Enhancements
  - **The C++ runtime is now open sourced.**
  - **PyTorch workflow**
    - Added DeepSeek V3/R1 support. Refer to `examples/deepseek_v3/README.md`, also to the blog `docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md`.
    - Added Llava-Next support.
    - Added BERT support.
    - Added a C++ based decoder, which added support for:
      - TopK / TopP.
      - Bad words.
      - Stop words.
      - Embedding bias.
    - Added Autotuner for custom-op-compatible tuning process.
      - Added a Python-based Autotuner core framework for kernel tuning.
      - Applied the Autotuner to fused MoE and NVFP4 linear operators for concept and performance evaluations.
    - Added guided decoding support (XGrammar integration).
    - Added pipeline parallelism support for the overlap scheduler in `PyExecutor`.
    - Added Qwen2VL model support.
    - Added mixed precision quantization support.
    - Added pipeline parallelism with attention DP support.
    - Added no-cache attention support.
    - Added `PeftCacheManager` support.
    - Added Qwen2.5‑VL support and refactored Qwen2‑VL.
    - Added trtllm‑gen FP4 GEMM support.
    - Added Qwen2 MoE support.
    - Applied `AutoTuner` to both Fused MoE and NVFP4 Linear operators.
    - Introduced a `UserBuffers` allocator.
    - Added Deepseek eager mode AllReduce fusion support.
    - Added Multi-Token Prediction (MTP) support. Refer to the “Multi-Token Prediction (MTP)” section of `examples/deepseek_v3/README.md`.
    - Added FlashMLA support for SM90.
    - Added support for enabling MTP with CUDA graph padding.
    - Added initial EAGLE-3 implementation.
    - Added support for FP8 MLA on NVIDIA Hopper and Blackwell GPUs.
  - **AutoDeploy for PyTorch workflow**.
    - The AutoDeploy for PyTorch workflow is an **experimental** feature in `tensorrt_llm._torch.auto_deploy`.
    - AutoDeploy provides an automated path from off-the-shelf models to optimized deployment in the TensorRT-LLM runtime.
    - Check out `examples/auto_deploy/README.md` for more details.
  - LLM API
    - [BREAKING CHANGE] Added dynamic logits processor support, and deprecated static logits processor.
    - Added batched logits processor support.
    - Added EAGLE support.
    - Added abort request support.
    - Added `get_stats` support.
    - Added multi-node support for Slurm-based clusters, refer to `examples/llm-api/llm_mgmn_*.sh`.
  - Added InternLM-XComposer2 support. Refer to “InternLM-XComposer2” section in `examples/multimodal/README.md`.
  - Added INT4-AWQ support for MoE models. Refer to the “AWQ Quantization” section in `examples/mixtral/README.md`.
  - Added Qwen2-Audio support. Refer to `examples/qwen2audio/README.md`.
  - Added Language-Adapter support. Refer to `examples/language_adapter/README.md`.
  - Added STDiT for OpenSoRA text-to-video support. Refer to `examples/stdit/README.md`.
  - Added vision encoders with tensor parallelism and context parallelism support. Refer to `examples/vit/README.md`.
  - Added EXAONE-Deep support. Refer to `examples/exaone/README.md`.
  - Added support for Phi-4-mini and Phi‑4‑MM.
  - Added Gemma3 text‑only model support. Refer to "Run Gemma 3" section at `examples/gemma/README.md`.
  - Added FP8 quantization support for Qwen2-VL.
  - Added batched inference support for the LLM API MMLU example `examples/mmlu_llmapi.py`.
  - Added FP4 quantization-layernorm fusion plugin support. (Llama models only)
  - Added Mamba-Hybrid support.
  - Added NVILA video support. The support includes 1 prompt - N media and N prompt - N media batching modes.
  - Added a `--quantize_lm_head` option `examples/quantization/quantize.py` to support `lm_head` quantization.
  - Added batched tensor FP4 quantization support.
  - Added a `/metrics` endpoint for `trtllm-serve` to log iteration statistics.
  - Added LoRA support for Phi-2 model.
  - Added returning context logits support for `trtllm-serve`.
  - Added one-shot version for UserBuffer AllReduce-Normalization on FP16/BF16.
  - Added request BW metric measurement for `disaggServerBenchmark`.
  - Updated logits bitmask kernel to v3.
  - Enabled CUDA graphs when attention DP was used and active requests on different GPUs were uneven.
  - Added iteration log support for `trtllm-bench`.
  - `fp8_blockscale_gemm` is now open-sourced.
  - Added AWQ support for ModelOpt checkpoints.
  - Added Linear block scale layout support in FP4 quantization.
  - Added pre-quantized FP8 checkpoint support for Nemotron-mini-4b-instruct.
  - Added Variable-Beam-Width-Search (VBWS) support (part2).
  - Added LoRA support for Gemma.
  - Refactored scaffolding worker, added OpenAI API worker support.
  - Optionally split MoE inputs into chunks to reduce GPU memory usage.
  - Added UCX IP interface support.
  - [BREAKING CHANGE] Added output of first token to additional generation outputs.
  - Added FP8 support for SM120 architecture.
  - Registered `ENABLE_MULTI_DEVICE` and `ENABLE_UCX` as CMake options.
  - Made the scaffolding Controller more generic.
  - Breaking change: Added individual gatherContext support for each additional output.
  - Enabled `PyExecutor` inference flow to estimate `max_num_tokens` for `kv_cache_manager`.
  - Added `TLLM_OVERRIDE_LAYER_NUM` and `TLLM_TRACE_MODEL_FORWARD` environment variables for debugging.
  - Supported aborting disconnected requests.
  - Added an option to run disaggregated serving without context servers.
  - Fixed and improved allreduce and fusion kernels.
  - Enhanced the integrated robustness of scaffolding via `init.py`.

### API Changes
  - Exposed `kc_cache_retention_config` from C++ `executor` API to the LLM API.
  - Moved `BuildConfig` arguments to `LlmArgs`.
  - Removed speculative decoding parameters from stateful decoders.
  - Exposed `DecoderState` via bindings and integrated it in decoder.
  - Refactored the `LlmArgs` with `Pydantic` and migrated remaining pybinding configurations to Python.
  - Refactored disaggregated serving scripts.
  - Added `numNodes` to `ParallelConfig`.
  - Redesigned the multi‑stream API for DeepSeek.

### Fixed Issues
  - Fixed misused length argument of PluginField. Thanks to the contribution from @jl749 in #2712. This also fixes #2685.
  - Fixed a Llama-3.2 SmoothQuant convert checkpoint issue. (#2677)
  - Fixed a bug when loading an engine using LoRA through the LLM API. (#2782)
  - Fixed incorrect batch slot usage in `addCumLogProbs` kernel. Thanks to the contribution from @aotman in #2787.
  - Fixed incorrect output for Llama-3.2-11B-Vision-Instruct. (#2796)
  - Removed the necessary of `--extra-index-url https://pypi.nvidia.com` when running `pip install tensorrt-llm`.

### Infrastructure Changes
  - The dependent NVIDIA ModelOpt version is updated to 0.27.

### Known Issues
  - The PyTorch workflow on SBSA is incompatible with bare metal environments like Ubuntu 24.04. Please use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal support on SBSA platforms.


## TensorRT-LLM Release 0.18.2

### Key Features and Enhancements
  - This update addresses known security issues. For the latest NVIDIA Vulnerability Disclosure Information visit https://www.nvidia.com/en-us/security/.


## TensorRT-LLM Release 0.18.1

### Key Features and Enhancements
  - **The 0.18.x series of releases builds upon the 0.17.0 release, focusing exclusively on dependency updates without incorporating features from the previous 0.18.0.dev pre-releases. These features will be included in future stable releases**.

### Infrastructure Changes
  - The dependent `transformers` package version is updated to 4.48.3.


## TensorRT-LLM Release 0.18.0

### Key Features and Enhancements
  - **Features that were previously available in the 0.18.0.dev pre-releases are not included in this release**.
  - [BREAKING CHANGE] Windows platform support is deprecated as of v0.18.0. All Windows-related code and functionality will be completely removed in future releases.

### Known Issues
  - The PyTorch workflow on SBSA is incompatible with bare metal environments like Ubuntu 24.04. Please use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal support on SBSA platforms.

### Infrastructure Changes
  - The base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:25.03-py3`.
  - The base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:25.03-py3`.
  - The dependent TensorRT version is updated to 10.9.
  - The dependent CUDA version is updated to 12.8.1.
  - The dependent NVIDIA ModelOpt version is updated to 0.25 for Linux platform.


## TensorRT-LLM Release 0.17.0

### Key Features and Enhancements
  - **Blackwell support**
    - **NOTE: pip installation is not supported for TRT-LLM 0.17 on Blackwell platforms only. Instead, it is recommended that the user build from source using NVIDIA NGC 25.01 PyTorch container.**
    - Added support for B200.
    - Added support for GeForce RTX 50 series using Windows Subsystem for Linux (WSL) for limited models.
    - Added NVFP4 Gemm support for Llama and Mixtral models.
    - Added NVFP4 support for the `LLM` API and `trtllm-bench` command.
    - GB200 NVL is not fully supported.
    - Added benchmark script to measure perf benefits of KV cache host offload with expected runtime improvements from GH200.
  - **PyTorch workflow**
    - The PyTorch workflow is an **experimental** feature in `tensorrt_llm._torch`. The following is a list of supported infrastructure, models, and features that can be used with the PyTorch workflow.
    - Added support for H100/H200/B200.
    - Added support for Llama models, Mixtral, QWen, Vila.
    - Added support for FP16/BF16/FP8/NVFP4 Gemm and fused Mixture-Of-Experts (MOE), FP16/BF16/FP8 KVCache.
    - Added custom context and decoding attention kernels support via PyTorch custom op.
    - Added support for chunked context (default off).
    - Added CudaGraph support for decoding only.
    - Added overlap scheduler support to overlap prepare inputs and model forward by decoding 1 extra token.
  - Added FP8 context FMHA support for the W4A8 quantization workflow.
  - Added ModelOpt quantized checkpoint support for the `LLM` API.
  - Added FP8 support for the Llama-3.2 VLM model. Refer to the “MLLaMA” section in `examples/multimodal/README.md`.
  - Added PDL support for `userbuffer` based AllReduce-Norm fusion kernel.
  - Added runtime support for seamless lookahead decoding.
  - Added token-aligned arbitrary output tensors support for the C++ `executor` API.

### API Changes
  - [BREAKING CHANGE] KV cache reuse is enabled automatically when `paged_context_fmha` is enabled.
  - Added `--concurrency` support for the `throughput` subcommand of `trtllm-bench`.

### Known Issues
  - Need `--extra-index-url https://pypi.nvidia.com` when running `pip install tensorrt-llm` due to new third-party dependencies.
  - The PYPI SBSA wheel is incompatible with PyTorch 2.5.1 due to a break in the PyTorch ABI/API, as detailed in the related [GitHub issue](https://github.com/pytorch/pytorch/issues/144966).
  - The PyTorch workflow on SBSA is incompatible with bare metal environments like Ubuntu 24.04. Please use the [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for optimal support on SBSA platforms.

### Fixed Issues
  - Fixed incorrect LoRA output dimension. Thanks for the contribution from @akhoroshev in #2484.
  - Added NVIDIA H200 GPU into the `cluster_key` for auto parallelism feature. (#2552)
  - Fixed a typo in the `__post_init__` function of `LLmArgs` Class. Thanks for the contribution from @topenkoff in #2691.
  - Fixed workspace size issue in the GPT attention plugin. Thanks for the contribution from @AIDC-AI.
  - Fixed Deepseek-V2 model accuracy.

### Infrastructure Changes
  - The base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:25.01-py3`.
  - The base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:25.01-py3`.
  - The dependent TensorRT version is updated to 10.8.0.
  - The dependent CUDA version is updated to 12.8.0.
  - The dependent ModelOpt version is updated to 0.23 for Linux platform, while 0.17 is still used on Windows platform.


## TensorRT-LLM Release 0.16.0

### Key Features and Enhancements
  - Added guided decoding support with XGrammar backend.
  - Added quantization support for RecurrentGemma. Refer to `examples/recurrentgemma/README.md`.
  - Added ulysses context parallel support. Refer to an example on building LLaMA 7B using 2-way tensor parallelism and 2-way context parallelism at `examples/llama/README.md`.
  - Added W4A8 quantization support to BF16 models on Ada (SM89).
  - Added PDL support for the FP8 GEMM plugins.
  - Added a runtime `max_num_tokens` dynamic tuning feature, which can be enabled by setting `--enable_max_num_tokens_tuning` to `gptManagerBenchmark`.
  - Added typical acceptance support for EAGLE.
  - Supported chunked context and sliding window attention to be enabled together.
  - Added head size 64 support for the XQA kernel.
  - Added the following features to the LLM API:
    - Lookahead decoding.
    - DeepSeek V1 support.
    - Medusa support.
    - `max_num_tokens` and `max_batch_size` arguments to control the runtime parameters.
    - `extended_runtime_perf_knob_config` to enable various performance configurations.
  - Added LogN scaling support for Qwen models.
  - Added `AutoAWQ` checkpoints support for Qwen. Refer to the “INT4-AWQ” section in `examples/qwen/README.md`.
  - Added `AutoAWQ` and `AutoGPTQ` Hugging Face checkpoints support for LLaMA. (#2458)
  - Added `allottedTimeMs` to the C++ `Request` class to support per-request timeout.
  - [BREAKING CHANGE] Removed NVIDIA V100 GPU support.

### API Changes
  - [BREAKING CHANGE] Removed `enable_xqa` argument from `trtllm-build`.
  - [BREAKING CHANGE] Chunked context is enabled by default when KV cache and paged context FMHA is enabled on non-RNN based models.
  - [BREAKING CHANGE] Enabled embedding sharing automatically when possible and remove the flag `--use_embedding_sharing` from convert checkpoints scripts.
  - [BREAKING CHANGE] The `if __name__ == "__main__"` entry point is required for both single-GPU and multi-GPU cases when using the `LLM` API.
  - [BREAKING CHANGE] Cancelled requests now return empty results.
  - Added the `enable_chunked_prefill` flag to the `LlmArgs` of the `LLM` API.
  - Integrated BERT and RoBERTa models to the `trtllm-build` command.

### Model Updates
  - Added Qwen2-VL support. Refer to the “Qwen2-VL” section of `examples/multimodal/README.md`.
  - Added multimodal evaluation examples. Refer to `examples/multimodal`.
  - Added Stable Diffusion XL support. Refer to `examples/sdxl/README.md`. Thanks for the contribution from @Zars19 in #1514.

### Fixed Issues
  - Fixed unnecessary batch logits post processor calls. (#2439)
  - Fixed a typo in the error message. (#2473)
  - Fixed the in-place clamp operation usage in smooth quant. Thanks for the contribution from @StarrickLiu in #2485.
  - Fixed `sampling_params` to only be setup if `end_id` is None and `tokenizer` is not None in the `LLM` API. Thanks to the contribution from @mfuntowicz in #2573.

### Infrastructure Changes
  - Updated the base Docker image for TensorRT-LLM to `nvcr.io/nvidia/pytorch:24.11-py3`.
  - Updated the base Docker image for TensorRT-LLM Backend to `nvcr.io/nvidia/tritonserver:24.11-py3`.
  - Updated to TensorRT v10.7.
  - Updated to CUDA v12.6.3.
  - Added support for Python 3.10 and 3.12 to TensorRT-LLM Python wheels on PyPI.
  - Updated to ModelOpt v0.21 for Linux platform, while v0.17 is still used on Windows platform.

### Known Issues
  - There is a known AllReduce performance issue on AMD-based CPU platforms on NCCL 2.23.4, which can be workarounded by `export NCCL_P2P_LEVEL=SYS`.

## TensorRT-LLM Release 0.15.0

### Key Features and Enhancements
  - Added support for EAGLE. Refer to `examples/eagle/README.md`.
  - Added functional support for GH200 systems.
  - Added AutoQ (mixed precision) support.
  - Added a `trtllm-serve` command to start a FastAPI based server.
  - Added FP8 support for Nemotron NAS 51B. Refer to `examples/nemotron_nas/README.md`.
  - Added INT8 support for GPTQ quantization.
  - Added TensorRT native support for INT8 Smooth Quantization.
  - Added quantization support for Exaone model. Refer to `examples/exaone/README.md`.
  - Enabled Medusa for Qwen2 models. Refer to “Medusa with Qwen2” section in `examples/medusa/README.md`.
  - Optimized pipeline parallelism with ReduceScatter and AllGather for Mixtral models.
  - Added support for `Qwen2ForSequenceClassification` model architecture.
  - Added Python plugin support to simplify plugin development efforts. Refer to `examples/python_plugin/README.md`.
  - Added different rank dimensions support for LoRA modules when using the Hugging Face format. Thanks for the contribution from @AlessioNetti in #2366.
  - Enabled embedding sharing by default. Refer to "Embedding Parallelism, Embedding Sharing, and Look-Up Plugin" section in `docs/source/performance/perf-best-practices.md` for information about the required conditions for embedding sharing.
  - Added support for per-token per-channel FP8 (namely row-wise FP8) on Ada.
  - Extended the maximum supported `beam_width` to `256`.
  - Added FP8 and INT8 SmoothQuant quantization support for the InternVL2-4B variant (LLM model only). Refer to `examples/multimodal/README.md`.
  - Added support for prompt-lookup speculative decoding. Refer to `examples/prompt_lookup/README.md`.
  - Integrated the QServe w4a8 per-group/per-channel quantization. Refer to “w4aINT8 quantization (QServe)” section in `examples/llama/README.md`.
  - Added a C++ example for fast logits using the `executor` API. Refer to “executorExampleFastLogits” section in `examples/cpp/executor/README.md`.
  - [BREAKING CHANGE] NVIDIA Volta GPU support is removed in this and future releases.
  - Added the following enhancements to the [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html):
    - [BREAKING CHANGE] Moved the runtime initialization from the first invocation of `LLM.generate` to `LLM.__init__` for better generation performance without warmup.
    - Added `n` and `best_of` arguments to the `SamplingParams` class. These arguments enable returning multiple generations for a single request.
    - Added `ignore_eos`, `detokenize`, `skip_special_tokens`, `spaces_between_special_tokens`, and `truncate_prompt_tokens` arguments to the `SamplingParams` class. These arguments enable more control over the tokenizer behavior.
    - Added support for incremental detokenization to improve the detokenization performance for streaming generation.
    - Added the `enable_prompt_adapter` argument to the `LLM` class and the `prompt_adapter_request` argument for the `LLM.generate` method. These arguments enable prompt tuning.
  - Added support for a `gpt_variant` argument to the `examples/gpt/convert_checkpoint.py` file. This enhancement enables checkpoint conversion with more GPT model variants. Thanks to the contribution from @tonylek in #2352.

### API Changes
  - [BREAKING CHANGE] Moved the flag `builder_force_num_profiles` in `trtllm-build` command to the `BUILDER_FORCE_NUM_PROFILES` environment variable.
  - [BREAKING CHANGE] Modified defaults for `BuildConfig` class so that they are aligned with the `trtllm-build` command.
  - [BREAKING CHANGE] Removed Python bindings of `GptManager`.
  - [BREAKING CHANGE] `auto` is used as the default value for `--dtype` option in quantize and checkpoints conversion scripts.
  - [BREAKING CHANGE] Deprecated `gptManager` API path in `gptManagerBenchmark`.
  - [BREAKING CHANGE] Deprecated the `beam_width` and `num_return_sequences` arguments to the `SamplingParams` class in the LLM API. Use the `n`, `best_of` and `use_beam_search` arguments instead.
  - Exposed `--trust_remote_code` argument to the OpenAI API server. (#2357)

### Model Updates
  - Added support for Llama 3.2 and llama 3.2-Vision model. Refer to `examples/mllama/README.md` for more details on the llama 3.2-Vision model.
  - Added support for Deepseek-v2. Refer to `examples/deepseek_v2/README.md`.
  - Added support for Cohere Command R models. Refer to `examples/commandr/README.md`.
  - Added support for Falcon 2,  refer to `examples/falcon/README.md`, thanks to the contribution from @puneeshkhanna in #1926.
  - Added support for InternVL2. Refer to `examples/multimodal/README.md`.
  - Added support for Qwen2-0.5B and Qwen2.5-1.5B model. (#2388)
  - Added support for Minitron. Refer to `examples/nemotron`.
  - Added a GPT Variant - Granite(20B and 34B). Refer to “GPT Variant - Granite” section in `examples/gpt/README.md`.
  - Added support for LLaVA-OneVision model. Refer to “LLaVA, LLaVa-NeXT, LLaVA-OneVision and VILA” section in `examples/multimodal/README.md`.

### Fixed Issues
  - Fixed a slice error in forward function. (#1480)
  - Fixed an issue that appears when building BERT. (#2373)
  - Fixed an issue that model is not loaded when building BERT. (2379)
  - Fixed the broken executor examples. (#2294)
  - Fixed the issue that the kernel `moeTopK()` cannot find the correct expert when the number of experts is not a power of two. Thanks @dongjiyingdjy for reporting this bug.
  - Fixed an assertion failure on `crossKvCacheFraction`. (#2419)
  - Fixed an issue when using smoothquant to quantize Qwen2 model. (#2370)
  - Fixed a PDL typo in `docs/source/performance/perf-benchmarking.md`, thanks @MARD1NO for pointing it out in #2425.

### Infrastructure Changes
  - The base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.10-py3`.
  - The base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:24.10-py3`.
  - The dependent TensorRT version is updated to 10.6.
  - The dependent CUDA version is updated to 12.6.2.
  - The dependent PyTorch version is updated to 2.5.1.
  - The dependent ModelOpt version is updated to 0.19 for Linux platform, while 0.17 is still used on Windows platform.

### Documentation
  - Added a copy button for code snippets in the documentation. (#2288)


## TensorRT-LLM Release 0.14.0

### Key Features and Enhancements
  - Enhanced the `LLM` class in the [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html).
    - Added support for calibration with offline dataset.
    - Added support for Mamba2.
    - Added support for `finish_reason` and `stop_reason`.
  - Added FP8 support for CodeLlama.
  - Added `__repr__` methods for class `Module`, thanks to the contribution from @1ytic in #2191.
  - Added BFloat16 support for fused gated MLP.
  - Updated ReDrafter beam search logic to match Apple ReDrafter v1.1.
  - Improved `customAllReduce` performance.
  - Draft model now can copy logits directly over MPI to the target model's process in `orchestrator` mode. This fast logits copy reduces the delay between draft token generation and the beginning of target model inference.
  - NVIDIA Volta GPU support is deprecated and will be removed in a future release.

### API Changes
  - [BREAKING CHANGE] The default `max_batch_size` of the `trtllm-build` command is set to `2048`.
  - [BREAKING CHANGE] Remove `builder_opt` from the `BuildConfig` class and the `trtllm-build` command.
  - Add logits post-processor support to the `ModelRunnerCpp` class.
  - Added `isParticipant` method to the C++ `Executor` API to check if the current process is a participant in the executor instance.

### Model Updates
  - Added support for NemotronNas, see `examples/nemotron_nas/README.md`.
  - Added support for Deepseek-v1, see `examples/deepseek_v1/README.md`.
  - Added support for Phi-3.5 models, see `examples/phi/README.md`.

### Fixed Issues
  - Fixed a typo in `tensorrt_llm/models/model_weights_loader.py`, thanks to the contribution from @wangkuiyi in #2152.
  - Fixed duplicated import module in `tensorrt_llm/runtime/generation.py`, thanks to the contribution from @lkm2835 in #2182.
  - Enabled `share_embedding` for the models that have no `lm_head` in legacy  checkpoint conversion path, thanks to the contribution from @lkm2835 in #2232.
  - Fixed `kv_cache_type` issue in the Python benchmark, thanks to the contribution from @qingquansong in #2219.
  - Fixed an issue with SmoothQuant calibration with custom datasets. Thanks to the contribution by @Bhuvanesh09 in #2243.
  - Fixed an issue surrounding `trtllm-build --fast-build` with fake or random weights. Thanks to @ZJLi2013 for flagging it in #2135.
  - Fixed missing `use_fused_mlp` when constructing `BuildConfig` from dict, thanks for the fix from @ethnzhng in #2081.
  - Fixed lookahead batch layout for `numNewTokensCumSum`. (#2263)

### Infrastructure Changes
  - The dependent ModelOpt version is updated to v0.17.

### Documentation
  - @Sherlock113 added a [tech blog](https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml) to the latest news in #2169, thanks for the contribution.

### Known Issues
  - Replit Code is not supported with the transformers 4.45+


## TensorRT-LLM Release 0.13.0

### Key Features and Enhancements
  - Supported lookahead decoding (experimental), see `docs/source/speculative_decoding.md`.
  - Added some enhancements to the `ModelWeightsLoader` (a unified checkpoint converter, see `docs/source/architecture/model-weights-loader.md`).
    -  Supported Qwen models.
    -  Supported auto-padding for indivisible TP shape in INT4-wo/INT8-wo/INT4-GPTQ.
    -  Improved performance on `*.bin` and `*.pth`.
  - Supported OpenAI Whisper in C++ runtime.
  - Added some enhancements to the `LLM` class.
    - Supported LoRA.
    - Supported engine building using dummy weights.
    - Supported `trust_remote_code` for customized models and tokenizers downloaded from Hugging Face Hub.
  - Supported beam search for streaming mode.
  - Supported tensor parallelism for Mamba2.
  - Supported returning generation logits for streaming mode.
  - Added `curand` and `bfloat16` support for `ReDrafter`.
  - Added sparse mixer normalization mode for MoE models.
  - Added support for QKV scaling in FP8 FMHA.
  - Supported FP8 for MoE LoRA.
  - Supported KV cache reuse for P-Tuning and LoRA.
  - Supported in-flight batching for CogVLM models.
  - Supported LoRA for the `ModelRunnerCpp` class.
  - Supported `head_size=48` cases for FMHA kernels.
  - Added FP8 examples for DiT models, see `examples/dit/README.md`.
  - Supported decoder with encoder input features for the C++ `executor` API.

### API Changes
  - [BREAKING CHANGE] Set `use_fused_mlp` to `True` by default.
  - [BREAKING CHANGE] Enabled `multi_block_mode` by default.
  - [BREAKING CHANGE] Enabled `strongly_typed` by default in `builder` API.
  - [BREAKING CHANGE] Renamed `maxNewTokens`, `randomSeed` and `minLength` to `maxTokens`, `seed` and `minTokens` following OpenAI style.
  - The `LLM` class
    - [BREAKING CHANGE] Updated `LLM.generate` arguments to include `PromptInputs` and `tqdm`.
  - The C++ `executor` API
    - [BREAKING CHANGE] Added `LogitsPostProcessorConfig`.
    - Added `FinishReason` to `Result`.

### Model Updates
  - Supported Gemma 2, see "Run Gemma 2" section in `examples/gemma/README.md`.

### Fixed Issues
  - Fixed an accuracy issue when enabling remove padding issue for cross attention. (#1999)
  - Fixed the failure in converting qwen2-0.5b-instruct when using `smoothquant`. (#2087)
  - Matched the `exclude_modules` pattern in `convert_utils.py` to the changes in `quantize.py`. (#2113)
  - Fixed build engine error when `FORCE_NCCL_ALL_REDUCE_STRATEGY` is set.
  - Fixed unexpected truncation in the quant mode of `gpt_attention`.
  - Fixed the hang caused by race condition when canceling requests.
  - Fixed the default factory for `LoraConfig`. (#1323)

### Infrastructure Changes
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.07-py3`.
  - Base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:24.07-py3`.
  - The dependent TensorRT version is updated to 10.4.0.
  - The dependent CUDA version is updated to 12.5.1.
  - The dependent PyTorch version is updated to 2.4.0.
  - The dependent ModelOpt version is updated to v0.15.


## TensorRT-LLM Release 0.12.0

### Key Features and Enhancements
  - Supported LoRA for MoE models.
  - The `ModelWeightsLoader` is enabled for LLaMA family models (experimental), see `docs/source/architecture/model-weights-loader.md`.
  - Supported FP8 FMHA for NVIDIA Ada Lovelace Architecture.
  - Supported GPT-J, Phi, Phi-3, Qwen, GPT, GLM, Baichuan, Falcon and Gemma models for the `LLM` class.
  - Supported FP8 OOTB MoE.
  - Supported Starcoder2 SmoothQuant. (#1886)
  - Supported ReDrafter Speculative Decoding, see “ReDrafter” section in `docs/source/speculative_decoding.md`.
  - Supported padding removal for BERT, thanks to the contribution from @Altair-Alpha in #1834.
  - Added in-flight batching support for GLM 10B model.
  - Supported `gelu_pytorch_tanh` activation function, thanks to the contribution from @ttim in #1897.
  - Added `chunk_length` parameter to Whisper, thanks to the contribution from @MahmoudAshraf97 in #1909.
  - Added `concurrency` argument for `gptManagerBenchmark`.
  - Executor API supports requests with different beam widths, see `docs/source/executor.md#sending-requests-with-different-beam-widths`.
  - Added the flag `--fast_build` to `trtllm-build` command (experimental).

### API Changes
  - [BREAKING CHANGE] `max_output_len` is removed from `trtllm-build` command, if you want to limit sequence length on engine build stage, specify `max_seq_len`.
  - [BREAKING CHANGE] The `use_custom_all_reduce` argument is removed from `trtllm-build`.
  - [BREAKING CHANGE] The `multi_block_mode` argument is moved from build stage (`trtllm-build` and builder API) to the runtime.
  - [BREAKING CHANGE] The build time argument `context_fmha_fp32_acc` is moved to runtime for decoder models.
  - [BREAKING CHANGE] The arguments `tp_size`, `pp_size` and `cp_size` is removed from `trtllm-build` command.
  - The C++ batch manager API is deprecated in favor of the C++ `executor` API, and it will be removed in a future release of TensorRT-LLM.
  - Added a version API to the C++ library, a `cpp/include/tensorrt_llm/executor/version.h` file is going to be generated.

### Model Updates
  - Supported LLaMA 3.1 model.
  - Supported Mamba-2 model.
  - Supported EXAONE model, see `examples/exaone/README.md`.
  - Supported Qwen 2 model.
  - Supported GLM4 models, see `examples/chatglm/README.md`.
  - Added LLaVa-1.6 (LLaVa-NeXT) multimodal support, see “LLaVA, LLaVa-NeXT and VILA” section in `examples/multimodal/README.md`.

### Fixed Issues
  - Fixed wrong pad token for the CodeQwen models. (#1953)
  - Fixed typo in `cluster_infos` defined in `tensorrt_llm/auto_parallel/cluster_info.py`, thanks to the contribution from @saeyoonoh in #1987.
  - Removed duplicated flags in the command at `docs/source/reference/troubleshooting.md`, thanks for the contribution from @hattizai in #1937.
  - Fixed segmentation fault in TopP sampling layer, thanks to the contribution from @akhoroshev in #2039. (#2040)
  - Fixed the failure when converting the checkpoint for Mistral Nemo model. (#1985)
  - Propagated `exclude_modules` to weight-only quantization, thanks to the contribution from @fjosw in #2056.
  - Fixed wrong links in README, thanks to the contribution from @Tayef-Shah in #2028.
  - Fixed some typos in the documentation, thanks to the contribution from @lfz941 in #1939.
  - Fixed the engine build failure when deduced `max_seq_len` is not an integer. (#2018)

### Infrastructure Changes
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.07-py3`.
  - Base Docker image for TensorRT-LLM Backend is updated to `nvcr.io/nvidia/tritonserver:24.07-py3`.
  - The dependent TensorRT version is updated to 10.3.0.
  - The dependent CUDA version is updated to 12.5.1.
  - The dependent PyTorch version is updated to 2.4.0.
  - The dependent ModelOpt version is updated to v0.15.0.

### Known Issues

- On Windows, installation of TensorRT-LLM may succeed, but you might hit `OSError: exception: access violation reading 0x0000000000000000` when importing the library in Python.


## TensorRT-LLM Release 0.11.0

### Key Features and Enhancements
- Supported very long context for LLaMA (see “Long context evaluation” section in `examples/llama/README.md`).
- Low latency optimization
  - Added a reduce-norm feature which aims to fuse the ResidualAdd and LayerNorm kernels after AllReduce into a single kernel, which is recommended to be enabled when the batch size is small and the generation phase time is dominant.
  - Added FP8 support to the GEMM plugin, which benefits the cases when batch size is smaller than 4.
  - Added a fused GEMM-SwiGLU plugin for FP8 on SM90.
- LoRA enhancements
  - Supported running FP8 LLaMA with FP16 LoRA checkpoints.
  - Added support for quantized base model and FP16/BF16 LoRA.
    - SQ OOTB (- INT8 A/W) + FP16/BF16/FP32 LoRA​
    - INT8/ INT4 Weight-Only (INT8 /W) + FP16/BF16/FP32 LoRA​
    - Weight-Only Group-wise + FP16/BF16/FP32 LoRA
  - Added LoRA support to Qwen2, see “Run models with LoRA” section in `examples/qwen/README.md`.
  - Added support for Phi-3-mini/small FP8 base + FP16/BF16 LoRA, see “Run Phi-3 with LoRA” section in `examples/phi/README.md`.
  - Added support for starcoder-v2 FP8 base + FP16/BF16 LoRA, see “Run StarCoder2 with LoRA” section in `examples/gpt/README.md`.
- Encoder-decoder models C++ runtime enhancements
  - Supported paged KV cache and inflight batching. (#800)
  - Supported tensor parallelism.
- Supported INT8 quantization with embedding layer excluded.
- Updated default model for Whisper to `distil-whisper/distil-large-v3`, thanks to the contribution from @IbrahimAmin1 in #1337.
- Supported HuggingFace model automatically download for the Python high level API.
- Supported explicit draft tokens for in-flight batching.
- Supported local custom calibration datasets, thanks to the contribution from @DreamGenX in #1762.
- Added batched logits post processor.
- Added Hopper qgmma kernel to XQA JIT codepath.
- Supported tensor parallelism and expert parallelism enabled together for MoE.
- Supported the pipeline parallelism cases when the number of layers cannot be divided by PP size.
- Added `numQueuedRequests` to the iteration stats log of the executor API.
- Added `iterLatencyMilliSec` to the iteration stats log of the executor API.
- Add HuggingFace model zoo from the community, thanks to the contribution from @matichon-vultureprime in #1674.

### API Changes
- [BREAKING CHANGE] `trtllm-build` command
  - Migrated Whisper to unified workflow (`trtllm-build` command), see documents: examples/whisper/README.md.
  - `max_batch_size` in `trtllm-build` command is switched to 256 by default.
  - `max_num_tokens` in `trtllm-build` command is switched to 8192 by default.
  - Deprecated `max_output_len` and added `max_seq_len`.
  - Removed unnecessary `--weight_only_precision` argument from `trtllm-build` command.
  - Removed `attention_qk_half_accumulation` argument from `trtllm-build` command.
  - Removed `use_context_fmha_for_generation` argument from `trtllm-build` command.
  - Removed `strongly_typed` argument from `trtllm-build` command.
  - The default value of `max_seq_len` reads from the HuggingFace mode config now.
- C++ runtime
  - [BREAKING CHANGE] Renamed `free_gpu_memory_fraction` in `ModelRunnerCpp` to `kv_cache_free_gpu_memory_fraction`.
  - [BREAKING CHANGE] Refactored `GptManager` API
    - Moved `maxBeamWidth` into `TrtGptModelOptionalParams`.
    - Moved `schedulerConfig` into `TrtGptModelOptionalParams`.
  - Added some more options to `ModelRunnerCpp`, including `max_tokens_in_paged_kv_cache`, `kv_cache_enable_block_reuse` and `enable_chunked_context`.
- [BREAKING CHANGE] Python high-level API
  - Removed the `ModelConfig` class, and all the options are moved to `LLM` class.
  - Refactored the `LLM` class, please refer to `examples/high-level-api/README.md`
    - Moved the most commonly used options in the explicit arg-list, and hidden the expert options in the kwargs.
    - Exposed `model` to accept either HuggingFace model name or local HuggingFace model/TensorRT-LLM checkpoint/TensorRT-LLM engine.
    - Support downloading model from HuggingFace model hub, currently only Llama variants are supported.
    - Support build cache to reuse the built TensorRT-LLM engines by setting environment variable `TLLM_LLMAPI_BUILD_CACHE=1` or passing `enable_build_cache=True` to `LLM` class.
    - Exposed low-level options including `BuildConfig`, `SchedulerConfig` and so on in the kwargs, ideally you should be able to configure details about the build and runtime phase.
  - Refactored `LLM.generate()` and `LLM.generate_async()` API.
    - Removed `SamplingConfig`.
    - Added `SamplingParams` with more extensive parameters, see `tensorrt_llm/llmapi/utils.py`.
      - The new `SamplingParams` contains and manages fields from Python bindings of `SamplingConfig`, `OutputConfig`, and so on.
    - Refactored `LLM.generate()` output as `RequestOutput`, see `tensorrt_llm/llmapi/llm.py`.
  - Updated the `apps` examples, specially by rewriting both `chat.py` and `fastapi_server.py` using the `LLM` APIs, please refer to the `examples/apps/README.md` for details.
    - Updated the `chat.py` to support multi-turn conversation, allowing users to chat with a model in the terminal.
    - Fixed the `fastapi_server.py` and eliminate the need for `mpirun` in multi-GPU scenarios.
- [BREAKING CHANGE] Speculative decoding configurations unification
  - Introduction of `SpeculativeDecodingMode.h` to choose between different speculative decoding techniques.
  - Introduction of `SpeculativeDecodingModule.h` base class for speculative decoding techniques.
  - Removed `decodingMode.h`.
- `gptManagerBenchmark`
  - [BREAKING CHANGE] `api` in `gptManagerBenchmark` command is `executor` by default now.
  - Added a runtime `max_batch_size`.
  - Added a runtime `max_num_tokens`.
- [BREAKING CHANGE] Added a `bias` argument to the `LayerNorm` module, and supports non-bias layer normalization.
- [BREAKING CHANGE] Removed `GptSession` Python bindings.

### Model Updates
- Supported Jais, see `examples/jais/README.md`.
- Supported DiT, see `examples/dit/README.md`.
- Supported VILA 1.5.
- Supported Video NeVA, see `Video NeVA`section in `examples/multimodal/README.md`.
- Supported Grok-1, see `examples/grok/README.md`.
- Supported Qwen1.5-110B with FP8 PTQ.
- Supported Phi-3 small model with block sparse attention.
- Supported InternLM2 7B/20B, thanks to the contribution from @RunningLeon in #1392.
- Supported Phi-3-medium models, see `examples/phi/README.md`.
- Supported Qwen1.5 MoE A2.7B.
- Supported phi 3 vision multimodal.

### Fixed Issues
- Fixed brokens outputs for the cases when batch size is larger than 1. (#1539)
- Fixed `top_k` type in `executor.py`, thanks to the contribution from @vonjackustc in #1329.
- Fixed stop and bad word list pointer offset in Python runtime, thanks to the contribution from @fjosw in #1486.
- Fixed some typos for Whisper model, thanks to the contribution from @Pzzzzz5142 in #1328.
- Fixed export failure with CUDA driver < 526 and pynvml >= 11.5.0, thanks to the contribution from @CoderHam in #1537.
- Fixed an issue in NMT weight conversion, thanks to the contribution from @Pzzzzz5142 in #1660.
- Fixed LLaMA Smooth Quant conversion, thanks to the contribution from @lopuhin in #1650.
- Fixed `qkv_bias` shape issue for Qwen1.5-32B (#1589), thanks to the contribution from @Tlntin in #1637.
- Fixed the error of Ada traits for `fpA_intB`, thanks to the contribution from @JamesTheZ  in #1583.
- Update `examples/qwenvl/requirements.txt`, thanks to the contribution from @ngoanpv in #1248.
- Fixed rsLoRA scaling in `lora_manager`, thanks to the contribution from @TheCodeWrangler in #1669.
- Fixed Qwen1.5 checkpoint convert failure #1675.
- Fixed Medusa safetensors and AWQ conversion, thanks to the contribution from @Tushar-ml in #1535.
- Fixed `convert_hf_mpt_legacy` call failure when the function is called in other than global scope, thanks to the contribution from @bloodeagle40234 in #1534.
- Fixed `use_fp8_context_fmha` broken outputs (#1539).
- Fixed pre-norm weight conversion for NMT models, thanks to the contribution from @Pzzzzz5142 in #1723.
- Fixed random seed initialization issue, thanks to the contribution from @pathorn in #1742.
- Fixed stop words and bad words in python bindings. (#1642)
- Fixed the issue that when converting checkpoint for Mistral 7B v0.3, thanks to the contribution from @Ace-RR: #1732.
- Fixed broken inflight batching for fp8 Llama and Mixtral, thanks to the contribution from @bprus: #1738
- Fixed the failure when `quantize.py` is export data to config.json, thanks to the contribution from @janpetrov: #1676
- Raise error when autopp detects unsupported quant plugin #1626.
- Fixed the issue that `shared_embedding_table` is not being set when loading Gemma #1799, thanks to the contribution from @mfuntowicz.
- Fixed stop and bad words list contiguous for `ModelRunner` #1815, thanks to the contribution from @Marks101.
- Fixed missing comment for `FAST_BUILD`, thanks to the support from @lkm2835 in #1851.
- Fixed the issues that Top-P sampling occasionally produces invalid tokens. #1590
- Fixed #1424.
- Fixed #1529.
- Fixed `benchmarks/cpp/README.md` for #1562 and #1552.
- Fixed dead link, thanks to the help from @DefTruth, @buvnswrn and @sunjiabin17 in: https://github.com/triton-inference-server/tensorrtllm_backend/pull/478, https://github.com/triton-inference-server/tensorrtllm_backend/pull/482 and https://github.com/triton-inference-server/tensorrtllm_backend/pull/449.

### Infrastructure Changes
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.05-py3`.
  - Base Docker image for TensorRT-LLM backend is updated to `nvcr.io/nvidia/tritonserver:24.05-py3`.
  - The dependent TensorRT version is updated to 10.2.0.
  - The dependent CUDA version is updated to 12.4.1.
  - The dependent PyTorch version is updated to 2.3.1.
  - The dependent ModelOpt version is updated to v0.13.0.

### Known Issues

- In a conda environment on Windows, installation of TensorRT-LLM may succeed. However, when importing the library in Python, you may receive an error message of `OSError: exception: access violation reading 0x0000000000000000`. This issue is under investigation.


## TensorRT-LLM Release 0.10.0

### Announcements
- TensorRT-LLM supports TensorRT 10.0.1 and NVIDIA NGC 24.03 containers.

### Key Features and Enhancements
- The Python high level API
  - Added embedding parallel, embedding sharing, and fused MLP support.
  - Enabled the usage of the `executor` API.
- Added a weight-stripping feature with a new `trtllm-refit` command. For more information, refer to `examples/sample_weight_stripping/README.md`.
- Added a weight-streaming feature. For more information, refer to `docs/source/advanced/weight-streaming.md`.
- Enhanced the multiple profiles feature; `--multiple_profiles` argument in `trtllm-build` command builds more optimization profiles now for better performance.
- Added FP8 quantization support for Mixtral.
- Added support for pipeline parallelism for GPT.
- Optimized `applyBiasRopeUpdateKVCache` kernel by avoiding re-computation.
- Reduced overheads between `enqueue` calls of TensorRT engines.
- Added support for paged KV cache for enc-dec models. The support is limited to beam width 1.
- Added W4A(fp)8 CUTLASS kernels for the NVIDIA Ada Lovelace architecture.
- Added debug options (`--visualize_network` and `--dry_run`) to the `trtllm-build` command to visualize the TensorRT network before engine build.
- Integrated the new NVIDIA Hopper XQA kernels for LLaMA 2 70B model.
- Improved the performance of pipeline parallelism when enabling in-flight batching.
- Supported quantization for Nemotron models.
- Added LoRA support for Mixtral and Qwen.
- Added in-flight batching support for ChatGLM models.
- Added support to `ModelRunnerCpp` so that it runs with the `executor` API for IFB-compatible models.
- Enhanced the custom `AllReduce` by adding a heuristic; fall back to use native NCCL kernel when hardware requirements are not satisfied to get the best performance.
- Optimized the performance of checkpoint conversion process for LLaMA.
- Benchmark
  - [BREAKING CHANGE] Moved the request rate generation arguments and logic from prepare dataset script to `gptManagerBenchmark`.
  - Enabled streaming and support `Time To the First Token (TTFT)` latency and `Inter-Token Latency (ITL)` metrics for `gptManagerBenchmark`.
  - Added the `--max_attention_window` option to `gptManagerBenchmark`.

### API Changes
- [BREAKING CHANGE] Set the default `tokens_per_block` argument of the `trtllm-build` command to 64 for better performance.
- [BREAKING CHANGE] Migrated enc-dec models to the unified workflow.
- [BREAKING CHANGE] Renamed `GptModelConfig` to `ModelConfig`.
- [BREAKING CHANGE] Added speculative decoding mode to the builder API.
- [BREAKING CHANGE] Refactor scheduling configurations
  - Unified the `SchedulerPolicy` with the same name in `batch_scheduler` and `executor`, and renamed it to `CapacitySchedulerPolicy`.
  - Expanded the existing configuration scheduling strategy from `SchedulerPolicy` to `SchedulerConfig` to enhance extensibility. The latter also introduces a chunk-based configuration called `ContextChunkingPolicy`.
- [BREAKING CHANGE] The input prompt was removed from the generation output in the `generate()` and `generate_async()` APIs. For example, when given a prompt as `A B`, the original generation result could be `<s>A B C D E` where only `C D E` is the actual output, and now the result is `C D E`.
- [BREAKING CHANGE] Switched default `add_special_token` in the TensorRT-LLM backend to `True`.
- Deprecated `GptSession` and `TrtGptModelV1`.

### Model Updates
- Support DBRX
- Support Qwen2
- Support CogVLM
- Support ByT5
- Support LLaMA 3
- Support Arctic (w/ FP8)
- Support Fuyu
- Support Persimmon
- Support Deplot
- Support Phi-3-Mini with long Rope
- Support Neva
- Support Kosmos-2
- Support RecurrentGemma

### Fixed Issues
- - Fixed some unexpected behaviors in beam search and early stopping, so that the outputs are more accurate.
- Fixed segmentation fault with pipeline parallelism and `gather_all_token_logits`. (#1284)
- Removed the unnecessary check in XQA to fix code Llama 70b Triton crashes. (#1256)
- Fixed an unsupported ScalarType issue for BF16 LoRA. (https://github.com/triton-inference-server/tensorrtllm_backend/issues/403)
- Eliminated the load and save of prompt table in multimodal. (https://github.com/NVIDIA/TensorRT-LLM/discussions/1436)
- Fixed an error when converting the models weights of Qwen 72B INT4-GPTQ. (#1344)
- Fixed early stopping and failures on in-flight batching cases of Medusa. (#1449)
- Added support for more NVLink versions for auto parallelism. (#1467)
- Fixed the assert failure caused by default values of sampling config. (#1447)
- Fixed a requirement specification on Windows for nvidia-cudnn-cu12. (#1446)
- Fixed MMHA relative position calculation error in `gpt_attention_plugin` for enc-dec models. (#1343)


### Infrastructure changes
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.03-py3`.
  - Base Docker image for TensorRT-LLM backend is updated to `nvcr.io/nvidia/tritonserver:24.03-py3`.
  - The dependent TensorRT version is updated to 10.0.1.
  - The dependent CUDA version is updated to 12.4.0.
  - The dependent PyTorch version is updated to 2.2.2.


## TensorRT-LLM Release 0.9.0
### Announcements
- TensorRT-LLM requires TensorRT 9.3 and 24.02 containers.
### Key Features and Enhancements
- **[BREAKING CHANGES]** TopP sampling optimization with deterministic AIR TopP algorithm is enabled by default
- **[BREAKING CHANGES]** Added support for embedding sharing for Gemma
- Added support for context chunking to work with KV cache reuse
- Enabled different rewind tokens per sequence for Medusa
- Added BART LoRA support (limited to the Python runtime)
- Enabled multi-LoRA for BART LoRA
- Added support for `early_stopping=False` in beam search for C++ Runtime
- Added support for logits post processor to the batch manager
- Added support for import and convert HuggingFace Gemma checkpoints
- Added support for loading Gemma from HuggingFace
- Added support for auto parallelism planner for high-level API and unified builder workflow
- Added support for running `GptSession` without OpenMPI
- Added support for Medusa IFB
- **[Experimental]** Added support for FP8 FMHA, note that the performance is not optimal, and we will keep optimizing it
- Added support for more head sizes for LLaMA-like models
  - NVIDIA Ampere (SM80, SM86), NVIDIA Ada Lovelace (SM89), NVIDIA Hopper (SM90) all support head sizes [32, 40, 64, 80, 96, 104, 128, 160, 256]
- Added support for OOTB functionality
  - T5
  - Mixtral 8x7B
- Benchmark features
  - Added emulated static batching in `gptManagerBenchmark`
  - Added support for arbitrary dataset from HuggingFace for C++ benchmarks
  - Added percentile latency report to `gptManagerBenchmark`
- Performance features
  - Optimized `gptDecoderBatch` to support batched sampling
  - Enabled FMHA for models in BART, Whisper, and NMT family
  - Removed router tensor parallelism to improve performance for MoE models
  - Improved custom all-reduce kernel
- Infrastructure features
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.02-py3`
  - The dependent PyTorch version is updated to 2.2
  - Base Docker image for TensorRT-LLM backend is updated to `nvcr.io/nvidia/tritonserver:24.02-py3`
  - The dependent CUDA version is updated to 12.3.2 (12.3 Update 2)

### API Changes

- Added C++ `executor` API
- Added Python bindings
- Added advanced and multi-GPU examples for Python binding of `executor` C++ API
- Added documents for C++ `executor` API
- Migrated Mixtral to high-level API and unified builder workflow
- **[BREAKING CHANGES]** Moved LLaMA convert checkpoint script from examples directory into the core library
- Added support for `LLM()` API to accept engines built by `trtllm-build` command
- **[BREAKING CHANGES]** Removed the `model` parameter from `gptManagerBenchmark` and `gptSessionBenchmark`
- **[BREAKING CHANGES]** Refactored GPT with unified building workflow
- **[BREAKING CHANGES]** Refactored the Qwen model to the unified build workflow
- **[BREAKING CHANGES]** Removed all the LoRA related flags from ``convert_checkpoint.py`` script and the checkpoint content to `trtllm-build` command to generalize the feature better to more models
- **[BREAKING CHANGES]** Removed the ``use_prompt_tuning`` flag, options from the ``convert_checkpoint.py`` script, and the checkpoint content to generalize the feature better to more models. Use `trtllm-build --max_prompt_embedding_table_size` instead.
- **[BREAKING CHANGES]** Changed the `trtllm-build --world_size` flag to the `--auto_parallel` flag. The option is used for auto parallel planner only.
- **[BREAKING CHANGES]** `AsyncLLMEngine` is removed. The `tensorrt_llm.GenerationExecutor` class is refactored to work with both explicitly launching with `mpirun` in the application level and accept an MPI communicator created by `mpi4py`.
- **[BREAKING CHANGES]** `examples/server` are removed.
- **[BREAKING CHANGES]** Removed LoRA related parameters from the convert checkpoint scripts.
- **[BREAKING CHANGES]** Simplified Qwen convert checkpoint script.
- **[BREAKING CHANGES]** Reused the `QuantConfig` used in `trtllm-build` tool to support broader quantization features.
- Added support for TensorRT-LLM checkpoint as model input.
- Refined `SamplingConfig` used in `LLM.generate` or `LLM.generate_async` APIs, with the support of beam search, a variety of penalties, and more features.
- Added support for the ``StreamingLLM`` feature. Enable it by setting `LLM(streaming_llm=...)`.

### Model Updates

- Added support for distil-whisper
- Added support for HuggingFace StarCoder2
- Added support for VILA
- Added support for Smaug-72B-v0.1
- Migrate BLIP-2 examples to `examples/multimodal`

### Limitations

- `openai-triton` examples are not supported on Windows.

### Fixed Issues

- Fixed a weight-only quant bug for Whisper to make sure that the `encoder_input_len_range` is not ``0``. (#992)
- Fixed an issue that log probabilities in Python runtime are not returned. (#983)
- Multi-GPU fixes for multimodal examples. (#1003)
- Fixed a wrong `end_id` issue for Qwen. (#987)
- Fixed a non-stopping generation issue. (#1118, #1123)
- Fixed a wrong link in ``examples/mixtral/README.md``. (#1181)
- Fixed LLaMA2-7B bad results when INT8 kv cache and per-channel INT8 weight only are enabled. (#967)
- Fixed a wrong `head_size` when importing a Gemma model from HuggingFace Hub. (#1148)
- Fixed ChatGLM2-6B building failure on INT8. (#1239)
- Fixed a wrong relative path in Baichuan documentation. (#1242)
- Fixed a wrong `SamplingConfig` tensor in `ModelRunnerCpp`. (#1183)
- Fixed an error when converting SmoothQuant LLaMA. (#1267)
- Fixed an issue that `examples/run.py` only load one line from `--input_file`.
- Fixed an issue that `ModelRunnerCpp` does not transfer `SamplingConfig` tensor fields correctly. (#1183)



## TensorRT-LLM Release 0.8.0

### Key Features and Enhancements

- Chunked context support (see docs/source/advanced/gpt-attention.md#chunked-context)
- LoRA support for C++ runtime (see docs/source/lora.md)
- Medusa decoding support (see examples/medusa/README.md)
  - The support is limited to Python runtime for Ampere or newer GPUs with fp16 and bf16 accuracy, and the `temperature` parameter of sampling configuration should be 0
- StreamingLLM support for LLaMA (see docs/source/advanced/gpt-attention.md#streamingllm)
- Support for batch manager to return logits from context and/or generation phases
  - Include support in the Triton backend
- Support AWQ and GPTQ for QWEN
- Support ReduceScatter plugin
- Support for combining `repetition_penalty` and `presence_penalty` #274
- Support for `frequency_penalty` #275
- OOTB functionality support:
  - Baichuan
  - InternLM
  - Qwen
  - BART
- LLaMA
  - Support enabling INT4-AWQ along with FP8 KV Cache
  - Support BF16 for weight-only plugin
- Baichuan
  - P-tuning support
  - INT4-AWQ and INT4-GPTQ support
- Decoder iteration-level profiling improvements
- Add `masked_select` and `cumsum` function for modeling
- Smooth Quantization support for ChatGLM2-6B / ChatGLM3-6B / ChatGLM2-6B-32K
- Add Weight-Only Support To Whisper #794, thanks to the contribution from @Eddie-Wang1120
- Support FP16 fMHA on NVIDIA V100 GPU
    ```{note}
    Some features are not enabled for all models listed in the [examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples) folder.
    ```

### Model Updates

- Phi-1.5/2.0
- Mamba support (see examples/mamba/README.md)
  - The support is limited to beam width = 1 and single-node single-GPU
- Nougat support (see examples/multimodal/README.md#nougat)
- Qwen-VL support (see examples/qwenvl/README.md)
- RoBERTa support, thanks to the contribution from @erenup
- Skywork model support
- Add example for multimodal models (BLIP with OPT or T5, LlaVA)

Refer to the {ref}`support-matrix-software` section for a list of supported models.

* API
  - Add a set of LLM APIs for end-to-end generation tasks (see examples/llm-api/README.md)
  - **[BREAKING CHANGES]** Migrate models to the new build workflow, including LLaMA, Mistral, Mixtral, InternLM, ChatGLM, Falcon, GPT-J, GPT-NeoX, Medusa, MPT, Baichuan and Phi (see docs/source/new_workflow.md)
  - **[BREAKING CHANGES]** Deprecate `LayerNorm` and `RMSNorm` plugins and removed corresponding build parameters
  - **[BREAKING CHANGES]** Remove optional parameter `maxNumSequences` for GPT manager
* Fixed Issues
  - Fix the first token being abnormal issue when `--gather_all_token_logits` is enabled #639
  - Fix LLaMA with LoRA enabled build failure #673
  - Fix InternLM SmoothQuant build failure #705
  - Fix Bloom int8_kv_cache functionality  #741
  - Fix crash in `gptManagerBenchmark` #649
  - Fix Blip2 build error #695
  - Add pickle support for `InferenceRequest` #701
  - Fix Mixtral-8x7b build failure with custom_all_reduce #825
  - Fix INT8 GEMM shape #935
  - Minor bug fixes
* Performance
  - **[BREAKING CHANGES]** Increase default `freeGpuMemoryFraction` parameter from 0.85 to 0.9 for higher throughput
  - **[BREAKING CHANGES]** Disable `enable_trt_overlap` argument for GPT manager by default
  - Performance optimization of beam search kernel
  - Add bfloat16 and paged kv cache support for optimized generation MQA/GQA kernels
  - Custom AllReduce plugins performance optimization
  - Top-P sampling performance optimization
  - LoRA performance optimization
  - Custom allreduce performance optimization by introducing a ping-pong buffer to avoid an extra synchronization cost
  - Integrate XQA kernels for GPT-J (beamWidth=4)
* Documentation
  - Batch manager arguments documentation updates
  - Add documentation for best practices for tuning the performance of TensorRT-LLM (See docs/source/perf_best_practices.md)
  - Add documentation for Falcon AWQ support (See examples/falcon/README.md)
  - Update to the `docs/source/new_workflow.md` documentation
  - Update AWQ INT4 weight only quantization documentation for GPT-J
  - Add blog: Speed up inference with SOTA quantization techniques in TRT-LLM
  - Refine TensorRT-LLM backend README structure #133
  - Typo fix #739

## TensorRT-LLM Release 0.7.1

### Key Features and Enhancements

- Speculative decoding (preview)
- Added a Python binding for `GptManager`
- Added a Python class `ModelRunnerCpp` that wraps C++ `gptSession`
- System prompt caching
- Enabled split-k for weight-only cutlass kernels
- FP8 KV cache support for XQA kernel
- Added Python builder API, `trtllm-build` command, and OPT support
- Support `StoppingCriteria` and `LogitsProcessor` in Python generate API
- FHMA support for chunked attention and paged KV cache
- Performance enhancements include:

  - MMHA optimization for MQA and GQA
  - LoRA optimization: cutlass grouped GEMM
  - Optimize Hopper warp specialized kernels
  - Optimize `AllReduce` for parallel attention on Falcon and GPT-J
  - Enable split-k for weight-only cutlass kernel when SM>=75
- Added {ref}`workflow` documentation


### Model Updates

- BART and mBART support in encoder-decoder models
- FairSeq Neural Machine Translation (NMT) family
- Mixtral-8x7B model
- Support weight loading for HuggingFace Mixtral model
- OpenAI Whisper
- Mixture of Experts support
- MPT - Int4 AWQ / SmoothQuant support
- Baichuan FP8 quantization support

### Fixed Issues

- Fixed tokenizer usage in `quantize.py` [#288](https://github.com/triton-inference-server/tensorrtllm_backend/issues/288)
- Fixed LLaMa with LoRA error
- Fixed LLaMA GPTQ failure
- Fixed Python binding for InferenceRequest issue
- Fixed CodeLlama SQ accuracy issue

### Known Issues

- The hang reported in issue [#149](https://github.com/triton-inference-server/tensorrtllm_backend/issues/149) has not been reproduced by the TensorRT-LLM team. If it is caused by a bug in TensorRT-LLM, that bug may be present in that release.
