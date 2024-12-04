# MLLaMA (llama-3.2 Vision model)

- [MLLaMA](#mllama-llama-32-vision-model)
  - [Overview](#overview)
  - [Support Matrix](#support-matrix)
  - [Build and run vision model](#build-and-run-vision-model)

## Overview

This document shows how to build and run a LLaMA-3.2 Vision model in TensorRT-LLM. We use [Llama-3.2-11B-Vision/](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) as an example.

For LLaMA-3.2 text model, please refer to the [examples/llama/README.md](../llama/README.md) because it shares the model architecture of llama.

## Support Matrix
  * BF16/FP16
  * Tensor Parallel
  * INT8 & INT4 Weight-Only

## Build and run vision model

* build engine of vision encoder model

```bash
python examples/multimodal/build_visual_engine.py --model_type mllama \
                                                  --model_path Llama-3.2-11B-Vision/ \
                                                  --output_dir /tmp/mllama/trt_engines/encoder/
```

* build engine of decoder model

```bash
python examples/mllama/convert_checkpoint.py --model_dir Llama-3.2-11B-Vision/ \
                              --output_dir /tmp/mllama/trt_ckpts \
                              --dtype bfloat16

python3 -m tensorrt_llm.commands.build \
            --checkpoint_dir /tmp/mllama/trt_ckpts \
            --output_dir /tmp/mllama/trt_engines/decoder/ \
            --max_num_tokens 4096 \
            --max_seq_len 2048 \
            --workers 1 \
            --gemm_plugin auto \
            --max_batch_size 4 \
            --max_encoder_input_len 4100 \
            --input_timing_cache model.cache
```

* Run test on multimodal/run.py with C++ runtime

```bash
python3 examples/multimodal/run.py --visual_engine_dir /tmp/mllama/trt_engines/encoder/ \
                                   --visual_engine_name visual_encoder.engine \
                                   --llm_engine_dir /tmp/mllama/trt_engines/decoder/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
                                   --input_text "<|image|><|begin_of_text|>If I had to write a haiku for this one" \
                                   --max_new_tokens 50 \
                                   --batch_size 2

Use model_runner_cpp by default. To switch to model_runner, set `--use_py_session` in the command mentioned above.

python3 examples/multimodal/eval.py --visual_engine_dir /tmp/mllama/trt_engines/encoder/ \
                                   --visual_engine_name visual_encoder.engine \
                                   --llm_engine_dir /tmp/mllama/trt_engines/decoder/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --test_trtllm \
                                   --accuracy_threshold 65 \
                                   --eval_task lmms-lab/ai2d
```
