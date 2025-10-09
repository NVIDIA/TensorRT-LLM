# Guide to Qwen2-Audio deployment pipeline
1. Download the Qwen2-Audio model.
    ```bash
    git lfs install
    export MODEL_PATH="tmp/Qwen2-Audio-7B-Instruct"
    git clone https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct $MODEL_PATH
    ```
2. Generate the TensorRT engine of audio encoder.
    ```bash
    export ENGINE_DIR="./trt_engines/qwen2audio/fp16"
    python3 ../multimodal/build_multimodal_engine.py --model_type qwen2_audio --model_path $MODEL_PATH --max_batch_size 32 --output_dir ${ENGINE_DIR}/audio
    ```

    The TensorRT engine will be generated under `${ENGINE_DIR}/audio`.

3. Build Qwen2 LLM TensorRT engine.
- Convert checkpoint
    1. Install packages
    ```bash
    pip install -r requirements.txt
    ```
    2. Convert
    2.1 FP16 checkpoint
    ```bash
    python3 ../qwen/convert_checkpoint.py --model_dir=$MODEL_PATH \
            --dtype=float16 \
            --output_dir=./tllm_checkpoint_1gpu_fp16
    ```
    2.2 (Optional) INT8 Weight Only checkpoint
    ```bash
    python3 ../qwen/convert_checkpoint.py --model_dir=$MODEL_PATH \
            --dtype=float16 \
            --use_weight_only \
            --weight_only_precision=int8 \
            --output_dir=./tllm_checkpoint_1gpu_fp16_wo8
    ```

- Build TensorRT LLM engine

    NOTE: `max_prompt_embedding_table_size = query_token_num * max_batch_size`, therefore, if you change `max_batch_size`, `--max_prompt_embedding_table_size` must be reset accordingly.
    ```bash
    trtllm-build --checkpoint_dir=./tllm_checkpoint_1gpu_fp16 \
                 --gemm_plugin=float16 --gpt_attention_plugin=float16 \
                 --max_batch_size=1 --max_prompt_embedding_table_size=4096 \
                 --output_dir=${ENGINE_DIR}/llm
    ```
    The built Qwen engines are located in `${ENGINE_DIR}/llm`.

    You can replace the `--checkpoint_dir` with INT8 Weight Only checkpoint to build INT8 Weight Only engine as well.
    For more information about Qwen, refer to the README.md in [`example/models/core/qwen`](../qwen).

4. Assemble everything into the Qwen2-Audio pipeline.

    4.1 Run with FP16 LLM engine
    ```bash
    python3 run.py \
        --tokenizer_dir=$MODEL_PATH \
        --engine_dir=${ENGINE_DIR}/llm \
        --audio_engine_path=${ENGINE_DIR}/audio/model.engine \
        --audio_url='./audio/glass-breaking-151256.mp3'
    ```
    4.2 (Optional) For multiple rounds of dialogue, you can run:
    ```bash
    python3 run_chat.py \
        --tokenizer_dir=$MODEL_PATH \
        --engine_dir=${ENGINE_DIR}/llm \
        --audio_engine_path=${ENGINE_DIR}/audio/model.engine \
        --max_new_tokens=256
    ```

    Note:
    - This example supports reusing the KV Cache for audio segments by assigning unique audio IDs.
    - To further optimize performance, users can also cache the audio features (encoder output) to bypass the audio encoder if the original audio data remains unchanged.
