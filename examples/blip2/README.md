# Guide to BLIP-2 pipeline

1. ViT and Qformer
- Generate ONNX model files for ViT and Qformer
    ```bash
    python onnx_export.py
    ```
    The exported ONNX files lies in `./onnx/visual_encoder` and `./onnx/Qformer`.
    Moreover, it will save test image tensor to `image.pt` and visual query tokens to `query_tokens.pt` for later pipeline inference.

- Build TensorRT engines
    ```bash
    python build_vit_qformer.py 0 # For ViT, FP16
    python build_vit_qformer.py 1 # For Qformer, FP16
    ```
    The built engines lie in `./plan/visual_encoder` and `./plan/Qformer`.

2. BLIP2 OPT-2.7B
- Download OPT-2.7B model checkpoint (same as original OPT-2.7B)
    ```bash
    # OPT-2.7B
    cd ../opt
    git-lfs clone https://huggingface.co/facebook/opt-2.7b
    ```
- Convert original checkpoint to TRT-LLM checkpoint format (same as original OPT-2.7B)
    ```bash
    # OPT-2.7B
    python3 convert_checkpoint.py --model_dir ./opt-2.7b \
                --dtype float16 \
                --output_dir ./opt/2.7B/trt_ckpt/fp16/1-gpu/
    ```
- Build TRT-LLM engines from TRT-LLM checkpoint (only need to add --max_prompt_embedding_table_size)

    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    # OPT-2.7B
    trtllm-build --checkpoint_dir=./opt/2.7B/trt_ckpt/fp16/1-gpu/ \
                    --max_batch_size 8 \
                    --use_gpt_attention_plugin float16 \
                    --use_gemm_plugin float16 \
                    --max_input_len 924 \
                    --max_output_len 100 \
                    --max_beam_width 5 \
                    --output_dir ../blip2/trt_engine/blip-2-opt-2.7b/fp16/1-gpu \
                    --max_prompt_embedding_table_size 256 # 256 = 32 (query_token number) * 8 (max_batch_size)
    ```
    The built OPT engines lie in `./trt_engine/blip-2-opt-2.7b/fp16/1-gpu`.

    **UPDATE[2023-09-21]**: We have newly added INT8/INT4 weight-only support for OPT. So you can enable it using commands as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
    ```bash
    # OPT-2.7B
    python3 convert_checkpoint.py --model_dir ./opt-2.7b \
                --dtype float16 \
                --output_dir ./opt/2.7B/trt_ckpt/int4_weightonly/1-gpu/
                --use_weight_only \
                --weight_only_precision int4

    trtllm-build --checkpoint_dir=./opt/2.7B/trt_ckpt/int4_weightonly/1-gpu/ \
                    --max_batch_size 8 \
                    --use_gpt_attention_plugin float16 \
                    --use_gemm_plugin float16 \
                    --max_input_len 924 \
                    --max_output_len 100 \
                    --max_beam_width 5 \
                    --output_dir ../blip2/trt_engine/blip-2-opt-2.7b/int4_weightonly/1-gpu \
                    --max_prompt_embedding_table_size 256 # 256 = 32 (query_token number) * 8 (max_batch_size)
    ```
    The built OPT engines lie in `./trt_engine/blip-2-opt-2.7b/int4_weightonly/1-gpu`.

3. Assemble everything into BLIP-2 pipeline
    FP16 pipeline
    ```bash
    # BLIP OPT-2.7B
    cd ../blip2
    python run.py --num_beams 1 \
                  --max_txt_len 32 \
                  --max_output_len 30 \
                  --input_text "Question: which city is this? Answer:" \
                  --engine_dir ./plan \
                  --opt_engine_dir trt_engine/blip-2-opt-2.7b/fp16/1-gpu \
                  --input_dir image.pt \
                  --query_tokens query_tokens.pt
    ```

    INT8/INT4 weight-only quantization pipeline
    ```bash
    # BLIP OPT-2.7B
    cd ../blip2
    python run.py --num_beams 1 \
                  --max_txt_len 32 \
                  --max_output_len 30 \
                  --input_text "Question: which city is this? Answer:" \
                  --engine_dir ./plan \
                  --opt_engine_dir trt_engine/blip-2-opt-2.7b/int4_weightonly/1-gpu \
                  --input_dir image.pt \
                  --query_tokens query_tokens.pt
    ```
