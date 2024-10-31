<!-- omit from toc -->
# InternVL2

This document shows how to run Internvl2 pipelines with TensorRT-LLM, e.g. from image+text input modalities to text output.


1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/opt/README.md` and `examples/enc_dec/README.md`.

    ```bash
    export MODEL_NAME="InternVL2-8B" # options: InternVL2-26B, etc.
    git clone https://huggingface.co/OpenGVLab/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert and build TRT-LLM engine

    ```bash
    # 0 - convert
    echo "Start convert model ..."
    # Convert the InternLM2 7B model using a single GPU and FP16.
    python convert_checkpoint.py --model_dir tmp/hf_models/${MODEL_NAME} \
                    --dtype float16 \
                    --output_dir ./internlm2-chat-7b/trt_engines/fp16/1-gpu/
    # Note: setting `--dtype bfloat16` to use bfloat16 precision.

    # 1 - build trt engine
    # BUild the InternLM2 7B model using a single GPU
    trtllm-build --checkpoint_dir ./internlm2-chat-7b/trt_engines/fp16/1-gpu/ \
                --output_dir ./engine_outputs \
                --gemm_plugin float16
    echo "End build llm engine."

    ```

3. Build TensorRT engines for vision encoders

    ```bash
    python build_visual_engine.py \
        --model_path tmp/hf_models/${MODEL_NAME}  \
        --model_type internvl2
    ```

    The built engines are located in `tmp/trt_engines/${MODEL_NAME}/vision_encoder`.


4. Run pipelines

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --visual_engine_dir ./tmp/trt_engines/${MODEL_NAME}/vision_encoder \
        --llm_engine_dir ./engine_outputs \
        --input_text "Question: which city is this? Answer:" \
        --image_path ./pics/demo.jpeg
    ```

