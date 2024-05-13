# Change Log

## Versions 0.8.0

* Model Support
  - Phi-1.5/2.0
  - Mamba support (see examples/mamba/README.md)
    - The support is limited to beam width = 1 and single-node single-GPU
  - Nougat support (see examples/multimodal/README.md#nougat)
  - Qwen-VL support (see examples/qwenvl/README.md)
  - RoBERTa support, thanks to the contribution from @erenup
  - Skywork model support
  - Add example for multimodal models (BLIP with OPT or T5, LlaVA)
* Features
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
* API
  - Add a set of High-level APIs for end-to-end generation tasks (see examples/high-level-api/README.md)
  - **[BREAKING CHANGES]** Migrate models to the new build workflow, including LLaMA, Mistral, Mixtral, InternLM, ChatGLM, Falcon, GPT-J, GPT-NeoX, Medusa, MPT, Baichuan and Phi (see docs/source/checkpoint.md)
  - **[BREAKING CHANGES]** Deprecate `LayerNorm` and `RMSNorm` plugins and removed corresponding build parameters
  - **[BREAKING CHANGES]** Remove optional parameter `maxNumSequences` for GPT manager
* Bug fixes
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
  - Update to the `docs/source/checkpoint.md` documentation
  - Update AWQ INT4 weight only quantization documentation for GPT-J
  - Add blog: Speed up inference with SOTA quantization techniques in TRT-LLM
  - Refine TensorRT-LLM backend README structure #133
  - Typo fix #739

## Versions 0.7.0 / 0.7.1

* Models
  - BART and mBART support in encoder-decoder models
  - FairSeq Neural Machine Translation (NMT) family
  - Mixtral-8x7B model
    - Support weight loading for HuggingFace Mixtral model
  - OpenAI Whisper
  - Mixture of Experts support
  - MPT - Int4 AWQ / SmoothQuant support
  - Baichuan FP8 quantization support
* Features
  - [Preview] Speculative decoding
  - Add Python binding for `GptManager`
  - Add a Python class `ModelRunnerCpp` that wraps C++ `gptSession`
  - System prompt caching
  - Enable split-k for weight-only cutlass kernels
  - FP8 KV cache support for XQA kernel
  - New Python builder API and `trtllm-build` command(already applied to [blip2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/blip2) and [OPT](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/opt#3-build-tensorrt-engines) )
  - Support `StoppingCriteria` and `LogitsProcessor` in Python generate API (thanks to the contribution from @zhang-ge-hao)
  - fMHA support for chunked attention and paged kv cache
* Bug fixes
  - Fix tokenizer usage in quantize.py #288, thanks to the contribution from @0xymoro
  - Fix LLaMa with LoRA error #637
  - Fix LLaMA GPTQ failure #580
  - Fix Python binding for InferenceRequest issue #528
  - Fix CodeLlama SQ accuracy issue #453
* Performance
  - MMHA optimization for MQA and GQA
  - LoRA optimization: cutlass grouped gemm
  - Optimize Hopper warp specialized kernels
  - Optimize AllReduce for parallel attention on Falcon and GPT-J
  - Enable split-k for weight-only cutlass kernel when SM>=75
* Documentation
  - Add [documentation for convert/build workflow](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/checkpoint.md)

## Versions 0.6.0 / 0.6.1

  * Models
      * ChatGLM3
      * InternLM (contributed by @wangruohui)
      * Mistral 7B (developed in collaboration with Mistral.AI)
      * MQA/GQA support to MPT (and GPT) models (contributed by @bheilbrun)
      * Qwen (contributed by @Tlntin and @zhaohb)
      * Replit Code V-1.5 3B (external contribution)
      * T5, mT5, Flan-T5 (Python runtime only)

  * Features
      * Add runtime statistics related to active requests and KV cache
        utilization from the batch manager (see
        the [batch manager](docs/source/batch_manager.md) documentation)
      * Add `sequence_length` tensor to support proper lengths in beam-search
        (when beam-width > 1 - see
        [tensorrt_llm/batch_manager/GptManager.h](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * BF16 support for encoder-decoder models (Python runtime - see
        [examples/enc_dec](examples/enc_dec/README.md))
      * Improvements to memory utilization (CPU and GPU - including memory
        leaks)
      * Improved error reporting and memory consumption
      * Improved support for stop and bad words
      * INT8 SmoothQuant and INT8 KV Cache support for the Baichuan models (see
        [examples/baichuan](examples/baichuan/README.md))
      * INT4 AWQ Tensor Parallelism support and INT8 KV cache + AWQ/weight-only
        support for the GPT-J model (see [examples/gptj](examples/gptj/README.md))
      * INT4 AWQ support for the Falcon models
        (see [examples/falcon](examples/falcon/README.md))
      * LoRA support (functional preview only - limited to the Python runtime,
        only QKV support and not optimized in terms of runtime performance) for
        the GPT model (see the
        [Run LoRA with the Nemo checkpoint](examples/gpt/README.md#Run-LoRA-with-the-Nemo-checkpoint)
        in the GPT example)
      * Multi-GPU support for encoder-decoder models (Python runtime - see
        [examples/enc_dec](examples/enc_dec/README.md))
      * New heuristic for launching the Multi-block Masked MHA kernel (similar
        to FlashDecoding - see
        [decoderMaskedMultiheadAttentionLaunch.h](cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionLaunch.h))
      * Prompt-Tuning support for GPT and LLaMA models (see the
        [Prompt-tuning](examples/gpt/README.md#Prompt-tuning) Section in the GPT example)
      * Performance optimizations in various CUDA kernels
      * Possibility to exclude input tokens from the output (see `excludeInputInOutput` in
        [`GptManager`](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * Python binding for the C++ runtime (GptSession - see [`pybind`](cpp/tensorrt_llm/pybind))
      * Support for different micro batch sizes for context and generation
        phases with pipeline parallelism (see `GptSession::Config::ctxMicroBatchSize` and
        `GptSession::Config::genMicroBatchSize` in
        [tensorrt_llm/runtime/gptSession.h](cpp/include/tensorrt_llm/runtime/gptSession.h))
      * Support for "remove input padding" for encoder-decoder models (see
        [examples/enc_dec](examples/enc_dec/README.md))
      * Support for context and generation logits (see `mComputeContextLogits` and
        `mComputeGenerationLogits` in
        [tensorrt_llm/runtime/gptModelConfig.h](cpp/include/tensorrt_llm/runtime/gptModelConfig.h))
      * Support for `logProbs` and `cumLogProbs` (see `"output_log_probs"` and
        `"cum_log_probs"` in [`GptManager`](cpp/include/tensorrt_llm/batch_manager/GptManager.h))
      * Update to CUTLASS 3.x

  * Bug fixes
      * Fix for ChatGLM2 #93 and #138
      * Fix tensor names error "RuntimeError: Tensor names
        (`host_max_kv_cache_length`) in engine are not the same as expected in
        the main branch" #369
      * Fix weights split issue in BLOOM when `world_size = 2` ("array split
        does not result in an equal division") #374
      * Fix SmoothQuant multi-GPU failure with tensor parallelism is 2 #267
      * Fix a crash in GenerationSession if stream keyword argument is not None
        #202
      * Fix a typo when calling PyNVML API [BUG] code bug #410
      * Fix bugs related to the improper management of the `end_id` for various
        models [C++ and Python]
      * Fix memory leaks [C++ code and Python models]
      * Fix the std::alloc error when running the gptManagerBenchmark -- issue
        gptManagerBenchmark std::bad_alloc error #66
      * Fix a bug in pipeline parallelism when beam-width > 1
      * Fix a bug with Llama GPTQ due to improper support of GQA
      * Fix issue #88
      * Fix an issue with the Huggingface Transformers version #16
      * Fix link jump in windows readme.md #30 - by @yuanlehome
      * Fix typo in batchScheduler.h #56 - by @eltociear
      * Fix typo #58 - by @RichardScottOZ
      * Fix Multi-block MMHA: Difference between `max_batch_size` in the engine
        builder and `max_num_sequences` in TrtGptModelOptionalParams? #65
      * Fix the log message to be more accurate on KV cache #224
      * Fix Windows release wheel installation: Failed to install the release
        wheel for Windows using pip #261
      * Fix missing torch dependencies: [BUG] The batch_manage.a choice error
        in --cpp-only when torch's cxx_abi version is different with gcc #151
      * Fix linking error during compiling google-test & benchmarks #277
      * Fix logits dtype for Baichuan and ChatGLM: segmentation fault caused by
        the lack of bfloat16 #335
      * Minor bug fixes

## Version 0.5.0

  * TensorRT-LLM v0.5.0 is the first public release.
