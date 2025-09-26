## ViT in MultiModal

This document uses the [LLaVA-NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf) model as an example to show how to build the vision encoder in TRTLLM.

LLaVA-NeXT is an extension of LLaVA. TRT-LLM currently supports [Mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) and [ Nous-Hermes-2-Yi-34B](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) variant of LLaVA-NeXT.

1. Download Huggingface model weights. These models have both visual and LLM components
   unlike BLIP2 example which downloads only LLM components from Huggingface.

     ```bash
        export MODEL_NAME="llava-v1.6-mistral-7b-hf" #for 34b variant "llava-v1.6-34b-hf"
        git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Generate TRT-LLM engine for visual component

    ```bash
    python ./convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu/vision \
        --dtype float16

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu/vision \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision \
        --remove_input_padding disable \
        --bert_attention_plugin disable \
        --max_batch_size 8

    # copy the image newlines tensor to engine directory
    cp tmp/trt_models/${MODEL_NAME}/fp16/1-gpu/vision/image_newlines.safetensors tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision
    ```
3. Generate TRT-LLM engine for LLaMA following example in `examples/models/core/llama/README.md`

    ```bash
    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu/llm \
        --dtype float16

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu/llm \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 8 \
        --max_input_len 4096 \
        --max_seq_len 5120 \
        --max_num_tokens 32768 \
        --max_multimodal_len 32768 # 8 (max_batch_size) * 4096 (max_input_len)
    ```

4. Run
    ```bash
    python ../multimodal/run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --input_text "Question: which city is this? Answer:"
    ```

4. (Optional) Different quantization methods supported in LLaMA can be applied to LLaVA-Next as well, such as INT4/INT8 weight-only, SmoothQuant, and INT4 Activation-Aware Quantization (AWQ). Detailed instructions can be found in LLaMA [README](../llama/README.md).

   For example,

   ```bash
   # INT4 weight only
   python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu/llm \
        --use_weight_only \
        --weight_only_precision int4

   # INT4 AWQ
   python ../../../quantization/quantize.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu/llm \
        --dtype float16 \
        --qformat int4_awq \
        --calib_size 32
   ```

   Then follow the same `trtllm-build` and `run.py` steps as before. NOTE: for `trtllm-build` command, do not use `--use_fused_mlp=enable` in these quantization modes.
