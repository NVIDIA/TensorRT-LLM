# Guide to Qwen-VL deployment pipeline
1. Download Qwen-VL
    ```bash
    git lfs install
    git clone https://huggingface.co/Qwen/Qwen-VL-Chat
    ```
2. ViT
- Generate Vit ONNX model and TensorRT engine. If you don't have ONNX file, use:
    ```bash
    python3 vit_onnx_trt.py --pretrained_model_path ./Qwen-VL-Chat
    ```
    The ONNX and TensorRT engine will be generated under ./onnx/visual_encoder and ./plan/visual_encoder respectively.

    If you already have onnx file under ./onnx/visual_encoder and want to build TensorRT engine with it, use:
    ```bash
    python3 vit_onnx_trt.py --pretrained_model_path ./Qwen-VL-Chat --only_trt
    ```
    Moreover, it will save test image tensor to `image.pt` for later pipeline inference.

3. Qwen
- Quantize the weights to INT4 with GPTQ
    1. Install the auto-gptq module
    ```bash
    pip install auto-gptq
    ```
    2. Weight quantization
    ```bash
    python3 gptq_convert.py --hf_model_dir ./Qwen-VL-Chat --tokenizer_dir ./Qwen-VL-Chat \
            --quant_ckpt_path ./Qwen-VL-Chat-4bit
    ```

- Build TRT-LLM engines (only need to add --max_prompt_embedding_table_size)

    **NOTE:** `max_prompt_embedding_table_size = query_token_num * max_batch_size`, so if you changes the max_batch_size, prompt table size must be reset accordingly.
    ```bash
    python3 ../qwen/build.py --hf_model_dir=Qwen-VL-Chat \
            --quant_ckpt_path=./Qwen-VL-Chat-4bit/gptq_model-4bit-128g.safetensors \
            --dtype float16 \
            --max_batch_size 8 \
            --max_input_len 512 \
            --max_output_len 1024 \
            --remove_input_padding \
            --use_gpt_attention_plugin float16 \
            --use_gemm_plugin float16 \
            --use_weight_only \
            --weight_only_precision int4_gptq \
            --per_group \
            --enable_context_fmha \
            --use_rmsnorm_plugin \
            --log_level verbose \
            --use_lookup_plugin float16 \
            --max_prompt_embedding_table_size 2048 \
            --output_dir=./trt_engines/Qwen-VL-7B-Chat-int4-gptq

            # --max_prompt_embedding_table_size 2048 = 256 (query_token number) * 8 (max_batch_size)
    ```
    The built Qwen engines lie in `./trt_engines/Qwen-VL-7B-Chat-int4-gptq`.
    And for more build config about QWen part, you can refer the README.md in example/qwen.

4. Assemble everything into Qwen-VL pipeline

    4.1 INT4 GPTQ weight-only quantization pipeline
    ```bash
    python3 run.py \
        --tokenizer_dir=./Qwen-VL-Chat \
        --qwen_engine_dir=./trt_engines/Qwen-VL-7B-Chat-int4-gptq \
        --vit_engine_dir=./plan
    ```
    4.2 For multiple rounds of dialogue, you can run:
    ```bash
    python3 run_chat.py \
        --tokenizer_dir=./Qwen-VL-Chat \
        --qwen_engine_dir=./trt_engines/Qwen-VL-7B-Chat-int4-gptq \
        --vit_engine_dir=./plan
    ```
    4.3 If you want to show the bounding box result in the demo picture:
    Please install opencv, request, zmq firstly:
    ```bash
    pip install opencv-python==4.5.5.64
    pip install opencv-python-headless==4.5.5.64
    pip install zmq
    pip install request
    ```

    4.3.1 If the current program is being executed on a remote machine:

    &nbsp;&nbsp;Firstly, run the below command in the local machine:
    ```bash
    python3 show_pic.py --ip=127.0.0.1 --port=8006
    ```
    &nbsp;&nbsp;Please replace "ip" and "port" value in your case. "ip" means your remote machine IP address.

    &nbsp;&nbsp;Secondly, run the below command in the remote machine:
    ```bash
    python3 run_chat.py \
        --tokenizer_dir=./Qwen-VL-Chat \
        --qwen_engine_dir=./trt_engines/Qwen-VL-7B-Chat-int4-gptq \
        --vit_engine_dir=./plan \
        --display \
        --port=8006
    ```

    &nbsp;&nbsp;Please replace "port" value in your case.

    4.3.2 If the current program is being executed on the local machine, use the following command:

    ```bash
    python3 run_chat.py \
        --tokenizer_dir=./Qwen-VL-Chat \
        --qwen_engine_dir=./trt_engines/Qwen-VL-7B-Chat-int4-gptq \
        --vit_engine_dir=./plan \
        --display \
        --local_machine
    ```

    When the question is "Print the bounding box of the girl", you'll see as below:

    ![image](./pics/1.png)
