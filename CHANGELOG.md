# Change Log

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
