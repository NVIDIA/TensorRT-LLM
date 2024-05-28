(release-notes)=

# Release Notes

All published functionality in the Release Notes has been fully tested and verified with known limitations documented. To share feedback about this release, access our [NVIDIA Developer Forum](https://forums.developer.nvidia.com/).


## TensorRT-LLM Release Next

### Announcements
- TBD

### Key Features and Enhancements
- TBD

### API Changes
- TBD

### Model Updates
- TBD

### Fixed Issues
- TBD

### Infrastructure changes
  - Base Docker image for TensorRT-LLM is updated to `nvcr.io/nvidia/pytorch:24.04-py3`.
  - Base Docker image for TensorRT-LLM backend is updated to `nvcr.io/nvidia/tritonserver:24.04-py3`.
  - The dependent CUDA version is updated to 12.4.1.
  - The dependent PyTorch version is updated to 2.3.0.


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

- Chunked context support (see docs/source/gpt_attention.md#chunked-context)
- LoRA support for C++ runtime (see docs/source/lora.md)
- Medusa decoding support (see examples/medusa/README.md)
  - The support is limited to Python runtime for Ampere or newer GPUs with fp16 and bf16 accuracy, and the `temperature` parameter of sampling configuration should be 0
- StreamingLLM support for LLaMA (see docs/source/gpt_attention.md#streamingllm)
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
  - Add a set of High-level APIs for end-to-end generation tasks (see examples/high-level-api/README.md)
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
- New Python builder API and `trtllm-build` command (already applied to [blip2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/blip2) and [OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/opt#3-build-tensorrt-engines))
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
