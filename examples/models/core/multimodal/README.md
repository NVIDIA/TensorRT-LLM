<!-- omit from toc -->
# Multi-Modal

This document shows how to run multimodal pipelines with TensorRT-LLM, e.g. from image+text input modalities to text output.

Multimodal models' LLM part has an additional parameter `--max_multimodal_len` compared to LLM-only build commands. Under the hood, `max_multimodal_len` and `max_prompt_embedding_table_size` are effectively the same concept, i.e., prepended/concatenated embeddings (either multimodal feature embeddings or prompt tuning embeddings) to the LLM input embeddings. The multimodal features from the visual encoder of shape `[batch_size, num_visual_features, visual_hidden_dim]` is flattened as `[batch_size * num_visual_features, visual_hidden_dim]` and passed like a prompt embedding table.

We first describe three runtime modes for running multimodal models and how to run each model on a single GPU. We then provide general guidelines on using tensor parallelism for the LLM part of the pipeline.

- [Runtime Mode](#runtime-modes)
- [BLIP2](#blip2)
- [CogVLM](#cogvlm)
- [Deplot](#deplot)
- [Fuyu](#fuyu)
- [Gemma3](#gemma3)
- [InternLM-XComposer2](#internlm-xcomposer2)
- [InternVL2](#internvl2)
- [Kosmos-2](#kosmos-2)
- [LLaVA, LLaVa-NeXT, LLaVA-OneVision and VILA](#llava-llava-next-llava-onevision-and-vila)
- [MLLaMA](#mllama)
- [NeVA](#neva)
- [Nougat](#nougat)
- [Phi-3-vision](#phi-3-vision)
- [Phi-4-multimodal](#phi-4-multimodal)
- [Qwen2-VL](#qwen2-vl)
- [Video NeVA](#video-neva)
- [Dataset Evaluation](#dataset-evaluation)
- [Enabling Tensor Parallelism for multi-GPU](#enabling-tensor-parallelism-for-multi-gpu)
- [Enabling Embedding Table Offloading](#enabling-embedding-table-offloading)

## Runtime Modes
TensorRT LLM supports three runtime modes for running multimodal models.
- `cpp_llm_only` (default): vision engine runs in python runtime, LLM in pybind C++ runtime
- `python`: everything runs in python runtime
- `cpp`: everything runs in C++ runtime

This can be specified by the `--session RUNTIME_MODE` argument in `run.py` (see instructions of each model below).
Not all models supports end-to-end `cpp` mode, the checked ones below are supported. See footnotes for reasons models are unsupported
- [ ] BLIP-2-T5 [^1]
- [x] BLIP-2-OPT
- [x] CogVLM
- [ ] Deplot [^1]
- [ ] Pix2Struct [^1]
- [x] Fuyu
- [x] InternVL2-2b
- [x] Kosmos-2
- [x] LLaVA
- [ ] LLaVA-NeXT / OneVision [^2]
- [x] VILA [^3]
- [ ] Mllama [^1]
- [x] NeVA
- [ ] Nougat [^1]
- [ ] Phi-3-Vision [^2]
- [ ] Phi-4-multimodal
- [ ] Qwen2-VL [^4]
- [x] Video-NeVA


[^1]: Model uses cross attention to feed visiual features to LLM decoder, which is not supported
[^2]: Model requires post processing its encoder output features, which is not supported
[^3]: Currently C++ runtime only supports single image per request (VILA mode 2)
[^4]: Vision encoder requires additional inputs not supported by the C++ runtime

## BLIP2

This BLIP section covers both BLIP2-OPT and BLIP2-T5, with minor changes needed when switching the LLM backbone.

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/models/contrib/opt/README.md` and `examples/models/core/enc_dec/README.md`.

    ```bash
    export MODEL_NAME="blip2-opt-2.7b" # options: blip2-opt-6.7b, blip2-flan-t5-xl, blip2-flan-t5-xxl
    git clone https://huggingface.co/Salesforce/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

    For BLIP2-OPT family,
    ```bash
    python ../../contrib/opt/convert_checkpoint.py --model_type blip2 \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16
    ```

    For BLIP2-T5 family,
    ```bash
    python ../enc_dec/convert_checkpoint.py --model_type blip2 \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/bfloat16 \
        --tp_size 1 \
        --pp_size 1 \
        --dtype bfloat16
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    For BLIP2-OPT family,
    ```bash
    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_seq_len 1024 \
        --max_input_len 924 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features)
    ```

    For BLIP2-T5 family,
    ```bash
    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/encoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/bfloat16/llm/encoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_input_len 924 \
        --max_multimodal_len 256 # 8 (max_batch_size) * 32 (num_visual_features)

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/bfloat16/llm/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_seq_len 1024 \
        --max_encoder_input_len 924 \
        --max_input_len 1 # Same command for decoder but don't set --max_multimodal_len
    ```

    **NOTE**: `max_multimodal_len = max_batch_size * num_visual_features`, so if you change max_batch_size, max multimodal length **MUST** be changed accordingly.

3. Build TensorRT engines for vision encoders

    ```bash
    python build_multimodal_engine.py --model_type blip2 --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/bfloat16/vision --max_batch_size 8
    ```

    The built engines are located in `tmp/trt_engines/${MODEL_NAME}/bfloat16/vision` for BLIP2-T5, similarly for BLIP-OPT.

    To run the BLIP2 pipeline with batch size > 1, change `--max_batch_size` argument to `build_multimodal_engine.py` accordingly.

4. Assemble everything into BLIP2 pipeline

    For BLIP2-OPT family,
    ```bash
    python run.py \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
    ```

    For BLIP2-T5 family,
    ```bash
    python run.py \
        --max_new_tokens 30 \
        --input_text "Question: which city is this? Answer:" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/bfloat16
    ```

5. (Optional) INT8/INT4 weight-only quantization for OPT can be enabled using commands as follows (take `INT4` as an example, while `INT8` is the default precision for weight-only quantization):
    ```bash
    python ../../contrib/opt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --use_weight_only \
        --weight_only_precision int4

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu/llm \
        --gemm_plugin float16 \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_multimodal_len 256 \
        --max_input_len 924 \
        --max_seq_len 1024
    ```

    The built OPT engines lie in `tmp/trt_engines/${MODEL_NAME}/int4_weightonly/1-gpu/llm`.
    You should use this directory without the `llm` part as `--engine_dir` argument to `run.py`

    **NOTE:** INT8/INT4 option is not supported for BLIP2-T5, because quantization support has not been
          added for encoder-decoder models yet.

## CogVLM

Currently, CogVLM only support bfloat16 precision.

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="cogvlm-chat-hf"
    git clone https://huggingface.co/THUDM/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    export TOKENIZER_NAME="vicuna-7b-v1.5"
    git clone https://huggingface.co/lmsys/${TOKENIZER_NAME} tmp/hf_models/${TOKENIZER_NAME}
    ```

    Because currently onnx doesn't support `xops.memory_efficient_attention`, we need to modify some source code of the huggingface CogVLM.
    ```
    cd tmp/hf_models/${MODEL_NAME}
    sed -i '4s/.*//;40s/.*/        out = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).contiguous()/;41s/.*//;42s/.*//' visual.py   # It will replace memory_efficient_attention with some basic ops
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/cogvlm`

   CogVLM uses a Vit encoder as LLM encoder and a modified Llama as decoder.

    ```bash
    python ../../contrib/cogvlm/convert_checkpoint.py --model_dir tmp/hf_models/${MODEL_NAME}  --output_dir tmp/trt_models/${MODEL_NAME} --dtype bfloat16 --use_prompt_tuning

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME} \
    --output_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu/llm \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --remove_input_padding enable \
    --max_batch_size 48 \
    --max_input_len 2048 \
    --max_seq_len 3076 \
    --paged_kv_cache enable \
    --bert_attention_plugin disable \
    --moe_plugin disable \
    --max_multimodal_len 61440 # 48 (max_batch_size) * 1280 (max_num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_multimodal_engine.py --model_type cogvlm --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 48 --output_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu/vision

    python run.py \
    --max_new_tokens 1000 \
    --input_text " [INST] please describe this image in detail [/INST] " \
    --hf_model_dir tmp/hf_models/${TOKENIZER_NAME} \
    --engine_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu \
    --batch_size 1 \
    --top_p 0.4 \
    --top_k 1 \
    --temperature 0.2 \
    --repetition_penalty 1.2 \
    --enable_context_fmha_fp32_acc

    CogVLM uses model_runner_cpp for its LLM decoder by default. To switch to model_runner, set `--session python` in the command mentioned above.
    ```

## Deplot

1. Download Huggingface weights and convert original checkpoint to TRT-LLM checkpoint format
   following example in `examples/models/core/enc_dec/README.md`.

    ```bash
    export MODEL_NAME="deplot"
    git clone https://huggingface.co/google/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../enc_dec/convert_checkpoint.py --model_type pix2struct \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/float16 \
        --tp_size 1 \
        --pp_size 1 \
        --dtype float16
    ```

2. Build TRT-LLM engine from TRT-LLM checkpoint

    ```bash
    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/float16/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/llm/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --gemm_plugin float16 \
        --bert_attention_plugin float16 \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --context_fmha disable \
        --max_beam_width 1 \
        --max_batch_size 8 \
        --max_seq_len 2558 \
        --max_encoder_input_len 2048 \
        --max_input_len 1
    ```

    The built deplot engines are located in `tmp/trt_engines/${MODEL_NAME}/1-gpu/float16`.

3. Build TensorRT engines for visual components

    ```bash
    python build_multimodal_engine.py --model_type pix2struct --model_path tmp/hf_models/${MODEL_NAME} --max_batch_size 8 --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/vision
    ```

    The built visual engines are located in `tmp/trt_engines/${MODEL_NAME}/1-gpu/float16/vision`.

    To run the deplot pipeline with batch size > 1, change `--max_batch_size` argument to `build_multimodal_engine.py` accordingly.

4. Assemble everything into deplot pipeline

    ```bash
    python run.py \
        --max_new_tokens 100 \
        --input_text "" \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/float16
    ```

## Fuyu

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="fuyu-8b"
    git clone https://huggingface.co/adept/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/models/core/gpt`.
    The LLM portion of Fuyu uses a Persimmon model
    ```bash
    python ../gpt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16 \
        --gpt_variant persimmon

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 2048
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_multimodal_engine.py --model_type fuyu --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
    ```

## Gemma3

**NOTE: We only support Gemma3 VLMs in Pytorch workflow.**

Gemma3VL decoder requires a custom attention mask while processing images. During the context phase:
- Text tokens attend to other tokens in a causal fashion (standard autoregressive behavior)
- Image tokens attend to other tokens in a causal fashion AND attend to other tokens from the same image in a bidirectional manner

**Reference:** [Gemma3 Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/gemma3)

We support this custom mask with FlashInfer attention backend.

### Requirements

To ensure expected behavior with Gemma3VL, the following configurations are **required**:
- **Attention Backend**: Use the FlashInfer attention backend
- **Chunked Prefill**: Must be disabled
- **KV Cache Reuse**: Must be disabled

### Quick Start

#### 1. Download Model Weights

```bash
export MODEL_NAME="gemma-3-27b-it"
git clone https://huggingface.co/google/${MODEL_NAME}
```

#### 2. Interactive Testing

Use the `quickstart_multimodal.py` script for quick testing:

```bash
python3 examples/llm-api/quickstart_multimodal.py \
    --model_dir ${MODEL_NAME}/ \
    --modality image \
    --image_format pil \
    --attention_backend FLASHINFER \
    --disable_kv_cache_reuse
```

#### 3. Model Serving

Serve the model using `trtllm-serve` with the required llmapi arguments mentioned in a yaml file:

```bash
# Create the configuration file
cat > extra-llm-api-options.yaml << 'EOF'
cuda_graph_config: null
attn_backend: "FLASHINFER"
enable_chunked_prefill: false
kv_cache_config:
  enable_block_reuse: false
EOF

# Serve the model
trtllm-serve ${MODEL_NAME}/ \
    --backend pytorch \
    --tp_size 1 \
    --port 8000 \
    --max_batch_size 4 \
    --extra_llm_api_options extra-llm-api-options.yaml
```

### Supported Model Variants

Currently supported Gemma3 variants: 4B, 12B, 27B


## InternLM-XComposer2

**NOTE: We only support InternLM-XComposer-VL-7b for now**

Firstly, please install transformers with 4.45.2
```bash
    pip install -r requirements-internlm-xcomposer2.txt
```

1. Convert Huggingface weights to TRT-LLM checkpoint format using `examples/models/contrib/internlm/README.md`.

2. Use `trtllm-build` command to build TRT-LLM engine for OPT.

3. The full list of commands is as follows:

    ```bash
    export MODEL_NAME=internlm-xcomposer2-vl-7b
    git lfs clone https://huggingface.co/internlm/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}

    python ../internlm2/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --lora_plugin float16 \
        --lora_dir . \
        --max_lora_rank 256 \
        --max_input_len 1536 \
        --max_batch_size 48 \
        --max_multimodal_len 58800 # 58800 = 1225(visual token/img) * 48 (max_batch_size), as each image corresponds to 1225 visual tokens in the ViT here

    python build_multimodal_engine.py \
        --model_type internlm-xcomposer2 \
        --model_path tmp/hf_models/${MODEL_NAME} \
        --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu/vision \
        --max_batch_size 48

    python run.py \
        --max_new_tokens 200 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --batch_size 1
    ```

## InternVL2

[InternVL Family](https://github.com/OpenGVLab/InternVL): Closing the Gap to Commercial Multimodal Models with Open-Source Suites —— A Pioneering Open-Source Alternative to GPT-4o. Here we show how to deploy InternVL2‑1B/InternVL2‑2B/InternVL2‑4B/InternVL2‑8B/InternVL2‑26B in TensorRT-LLM.

Firstly, please install transformers with 4.37.2
```bash
    pip install transformers==4.37.2
```

1. Download Huggingface weights
    - For InternVL2-1B
        ```bash
        export MODEL_NAME="InternVL2-1B"
        git clone https://huggingface.co/OpenGVLab/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
        export LLM_MODEL_NAME="qwen"
        ```

    - For InternVL2-2B/InternVL2‑8B/InternVL2‑26B
        ```bash
        export MODEL_NAME="InternVL2-2B" # or InternVL2‑8B, InternVL2‑26B
        git clone https://huggingface.co/OpenGVLab/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
        export LLM_MODEL_NAME="internlm2"
        ```

    - For InternVL2-4B
        ```bash
        export MODEL_NAME="InternVL2-4B"
        git clone https://huggingface.co/OpenGVLab/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
        export LLM_MODEL_NAME="phi"
        ```

2. Convert Huggingface weights into TRT-LLM checkpoints
    ```bash
    python ../${LLM_MODEL_NAME}/convert_checkpoint.py \
            --model_dir tmp/hf_models/${MODEL_NAME} \
            --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu  \
            --dtype float16
    ```

3. Build TRT engines
    ```bash
    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin auto \
        --max_batch_size 1 \
        --max_input_len 4096 \
        --max_seq_len 4608 \
        --max_multimodal_len 3328
    ```

4. Generate TensorRT engines for visual components and combine everything into final pipeline.
    ```bash
    python build_multimodal_engine.py --model_type internvl --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision
    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/ \
        --image_path tmp/hf_models/${MODEL_NAME}/examples/image1.jpg
    ```

5. (Optional) FP8 and INT8 SmoothQuant quantization is supported for the InternVL2-4B variant (LLM model only).

   ```bash
   # FP8 quantization
   python ../../../quantization/quantize.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp8/1-gpu \
        --dtype bfloat16 \
        --qformat fp8 \
        --kv_cache_dtype fp8

   # INT8 SmoothQuant quantization
   python ../../../quantization/quantize.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/int8/1-gpu \
        --dtype bfloat16 \
        --qformat int8_sq
   ```

   Then follow the same `trtllm-build`, `build_multimodal_engine.py` and `run.py` steps as before.


## Kosmos-2

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="kosmos-2"
    git clone https://huggingface.co/microsoft/kosmos-2-patch14-224 tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/models/core/gpt`.
    ```bash
    python ../gpt/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16 \
        --gpt_variant ${MODEL_NAME}

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 512 \
        --max_seq_len 1024 \
        --max_multimodal_len 64 # 1 (max_batch_size) * 64 (num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_multimodal_engine.py --model_type kosmos-2 --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu
    ```

## LLaVA, LLaVa-NeXT, LLaVA-OneVision and VILA

[LLaVA](https://github.com/haotian-liu/LLaVA) and [VILA](https://github.com/Efficient-Large-Model/VILA) are both visual language models (VLM) that can be deployed in TensorRT LLM with many quantization options. [LLaVA-NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf) is an extension of LLaVA. TRT-LLM currently supports [Mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) and [ Nous-Hermes-2-Yi-34B](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) variant of LLaVA-NeXT. [LLaVA-OneVision](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe) is another extension of LLaVA.

1. Download Huggingface model weights. These models have both visual and LLM components
   unlike BLIP2 example which downloads only LLM components from Huggingface.

    For LLaVA,

    ```bash
        export MODEL_NAME="llava-1.5-7b-hf" # also llava-1.5-13b-hf
        git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```
    For LLaVA-NeXT,

     ```bash
        export MODEL_NAME="llava-v1.6-mistral-7b-hf" #for 34b variant "llava-v1.6-34b-hf"
        git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

    For LLaVA-OneVision,

    ```bash
        export MODEL_NAME="llava-onevision-qwen2-7b-ov-hf" # also llava-onevision-qwen2-0.5b-ov-hf, llava-onevision-qwen2-72b-ov-hf, etc
        git clone https://huggingface.co/llava-hf/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

    For VILA, we need a few more steps until it is added to HF model zoo

    ```bash
        # install the following dependency
        pip install -r requirements-vila.txt

        # clone original VILA repo
        export VILA_PATH="tmp/hf_models/VILA"
        git clone https://github.com/Efficient-Large-Model/VILA.git ${VILA_PATH}

        # download VILA checkpoints
        export MODEL_NAME="vila1.5-3b" # NOTE: name must contain vila or VILA! it's used to identify whether we need to register the non-HF VILA codebase in HF Auto class
        git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Generate TRT-LLM engine for LLaMA following example in `examples/models/core/llama/README.md` and `examples/models/core/qwen/README.md`

    ```bash
    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16

    # for LLaVA
    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 576 # 1 (max_batch_size) * 576 (num_visual_features)

    # for LLaVA-NeXT
    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 1 \
        --max_input_len 4096 \
        --max_seq_len 5120 \
        --max_num_tokens 4096 \
        --max_multimodal_len 4096 # 1 (max_batch_size) * 4096 (max_input_len)

    # for VILA
    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 196 # 1 (max_batch_size) * 196 (num_visual_features)

    # for LLaVA-OneVision
    python ../qwen/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin float16 \
        --use_fused_mlp=enable \
        --max_batch_size 1 \
        --max_input_len  7228 \
        --max_seq_len  7328 \
        --max_multimodal_len 7128 # max_batch_size * num_visual_features(depends on the image size or the specified video num frame)
    ```

3. Build TensorRT engines for visual components

    ```bash
    python build_multimodal_engine.py --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision --model_type llava # for LLaVA

    python build_multimodal_engine.py --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision --model_type llava_next --max_batch_size 5 # 1 (max_batch_size) * 5 (because LLAVA-NeXT visual encoder can have at most 5 patches)  # for LLaVA-NeXT

    python build_multimodal_engine.py --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision --model_type llava_onevision --max_batch_size 32 # max_batch_size * patch for image or frame for video # for LLaVA-OneVision

    python build_multimodal_engine.py --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision --model_type vila --vila_path ${VILA_PATH} # for VILA
    ```

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --input_text "\n Which city is this?" # for LLaVA and for LLaVA-NeXT

    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --image_path=https://github.com/Efficient-Large-Model/VILA/raw/main/demo_images/av.png,https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png \
        --input_text "\n Please elaborate what you see in the images?","\n Which city is this?" \
        --batch_size=2 # for LLaVA
    ```

    Note that Llava can support N <one image, one prompt text> pairs inference batching, `--batch_size=N` should be used. There should be N images listed under `--image_path` and N text prompts listed under `--input_text`. Don't forget to set the `--max_batch_size` and `--max_multimodal_len` during engine building.

    For LLaVA-OneVision, you can use either image or video as inputs.
    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --input_text "What is shown in this image?" \
        --image_path image.png

    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --input_text "Why is this video funny?" \
        --video_path video.mp4
        --video_num_frames 8 # sample uniformly 8 frames from the video, up to 32 frames
    ```

    For VILA, you can use either local file or web url as input images.
    Suppose you have a local image `av.png` downloaded from `https://github.com/Efficient-Large-Model/VILA/blob/main/demo_trt_llm/av.png` and the url of `merlion.png`
    ```bash
    wget -O av.png https://raw.githubusercontent.com/Efficient-Large-Model/VILA/main/demo_images/av.png

    python run.py  \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --image_path=av.png,https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png \
        --input_text="<image>\n<image>\n Please elaborate what you see in the images?" \
        --batch_size=1 # for VILA mode 1

    python run.py  \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu \
        --image_path=av.png,https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png \
        --input_text="<image>\n Please elaborate what you see in the images?","<image>\n Which city is this?" \
        --batch_size=2 \
        --check_accuracy # for VILA mode 2
    ```

    Note that VILA can support different modes in terms of batching:
    - Mode 1: if you want to query N images as a whole using a prompt, `--batch_size=1` should be used (which is the default value). Example is given above.
    - Mode 2: if you want to query N <one image, one prompt text> pairs, `--batch_size=N` should be used. There should be N images listed under `--image_path` and N text prompts listed under `--input_text`. Don't forget to set the `--max_batch_size` and `--max_multimodal_len` during engine building.

    Note: use `--run_profiling` for performance measurement, use `--check_accuracy` for accuracy check.

4. (Optional) Different quantization methods supported in LLaMA and Qwen can be applied to LLaVA/VILA/LLaVA-OneVision as well, such as INT4/INT8 weight-only, SmoothQuant, and INT4 Activation-Aware Quantization (AWQ). Detailed instructions can be found in LLaMA [README](../llama/README.md) and Qwen [README](../qwen/README.md).

   For example,

   ```bash
   # INT4 weight only
   python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --dtype float16 \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_weightonly/1-gpu \
        --use_weight_only \
        --weight_only_precision int4

   # INT4 AWQ
   python ../../../quantization/quantize.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
        --dtype float16 \
        --qformat int4_awq \
        --calib_size 32
   ```

   Then follow the same `trtllm-build` and `run.py` steps as before. NOTE: for `trtllm-build` command, do not use `--use_fused_mlp=enable` in these quantization modes.

## MLLaMA

This section shows how to build and run a LLaMA-3.2 Vision model in TensorRT-LLM. We use [Llama-3.2-11B-Vision/](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) as an example.

For LLaMA-3.2 text model, please refer to the [examples/models/core/llama/README.md](../llama/README.md) because it shares the model architecture of llama.

### Support data types <!-- omit from toc -->
  * BF16
  * Tensor Parallel
  * INT8 & INT4 Weight-Only
  * FP8

### Build and run vision model <!-- omit from toc -->

* build engine of vision encoder model

```bash
python examples/models/core/multimodal/build_multimodal_engine.py --model_type mllama \
                                                  --model_path Llama-3.2-11B-Vision/ \
                                                  --output_dir /tmp/mllama/trt_engines/vision/
```

* build engine of decoder model

```bash
python examples/models/core/mllama/convert_checkpoint.py --model_dir Llama-3.2-11B-Vision/ \
                              --output_dir /tmp/mllama/trt_ckpts \
                              --dtype bfloat16

trtllm-build --checkpoint_dir /tmp/mllama/trt_ckpts \
            --output_dir /tmp/mllama/trt_engines/llm/ \
            --max_num_tokens 4096 \
            --max_seq_len 2048 \
            --workers 1 \
            --gemm_plugin auto \
            --max_batch_size 4 \
            --max_encoder_input_len 4100 \
            --input_timing_cache model.cache
```

Note that for instruct Vision model, please set the `max_encoder_input_len` as `6404`.

* Run test on multimodal/run.py with C++ runtime (LLM part only)

```bash
python3 examples/models/core/multimodal/run.py --engine_dir /tmp/mllama/trt_engines/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
                                   --input_text "<|image|><|begin_of_text|>If I had to write a haiku for this one" \
                                   --max_new_tokens 50 \
                                   --batch_size 2

Use model_runner_cpp by default. To switch to model_runner, set `--session python` in the command mentioned above.

python3 examples/models/core/multimodal/eval.py \
                                   --engine_dir /tmp/mllama/trt_engines/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --test_trtllm \
                                   --accuracy_threshold 65 \
                                   --eval_task lmms-lab/ai2d
```

### Run MLLaMA decoder part by FP8 <!-- omit from toc -->

```bash
# install modelopt 0.21.0
pip install nvidia-modelopt[torch]~=0.21.0

python ./examples/quantization/quantize.py --model_dir Llama-3.2-11B-Vision/ \
                                           --dtype bfloat16 \
                                           --qformat fp8 \
                                           --output_dir /tmp/llama-3.2-11B-Vision/fp8/ \
                                           --kv_cache_dtype fp8 \
                                           --calib_size 512 \
                                           --calib_dataset scienceqa

trtllm-build --checkpoint_dir /tmp/llama-3.2-11B-Vision/fp8/ \
            --output_dir /tmp/trt_engines/llama-3.2-11B-Vision/fp8/llm \
            --max_num_tokens 4096 \
            --max_seq_len 2048 \
            --workers 1 \
            --gemm_plugin auto \
            --max_batch_size 4 \
            --max_encoder_input_len 4100 \
            --input_timing_cache model.cache \
            --use_paged_context_fmha enable \
            --use_fp8_context_fmha enable

# copy visiual engine directory `/tmp/mllama/trt_engines/vision/` to fp8 engine directory `/tmp/trt_engines/llama-3.2-11B-Vision/fp8/vision`

python3 examples/models/core/multimodal/run.py --engine_dir /tmp/trt_engines/llama-3.2-11B-Vision/fp8/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --image_path https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
                                   --input_text "<|image|><|begin_of_text|>If I had to write a haiku for this one" \
                                   --max_new_tokens 50 \
                                   --batch_size 2

python3 examples/models/core/multimodal/eval.py --engine_dir /tmp/trt_engines/llama-3.2-11B-Vision/fp8/ \
                                   --hf_model_dir Llama-3.2-11B-Vision/ \
                                   --test_trtllm \
                                   --accuracy_threshold 65 \
                                   --eval_task lmms-lab/ai2d
```

Note that for instruct Vision model, please set the `max_encoder_input_len` as `6404`.

## NeVA

[NeVA](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/multimodal/mllm/neva.html) is a groundbreaking addition to the NeMo Multimodal ecosystem. This model seamlessly integrates large language-centric models with a vision encoder, that can be deployed in TensorRT-LLM.

1. Generate TRT-LLM engine for NVGPT following example in `examples/models/core/gpt/README.md`. To adhere to the NVGPT conventions of the conversion script, some layer keys have to be remapped using `--nemo_rename_key`.

    ```bash
    export MODEL_NAME="neva"
    python ../gpt/convert_checkpoint.py \
	--nemo_ckpt_path ./${MODEL_NAME}.nemo \
	--dtype bfloat16 \
	--output_dir tmp/trt_models/${MODEL_NAME} \
	--nemo_rename_key model:model.language_model \
        attention.linear_qkv.layer_norm_bias:input_layernorm.bias \
        attention.linear_qkv.layer_norm_weight:input_layernorm.weight \
        mlp.linear_fc1.layer_norm_bias:post_attention_layernorm.bias \
        mlp.linear_fc1.layer_norm_weight:post_attention_layernorm.weight \
        linear_qkv:query_key_value \
        linear_fc1:dense_h_to_4h \
        linear_fc2:dense_4h_to_h \
        linear_proj:dense \
        decoder:encoder

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME} \
        --output_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu/llm \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 729 # 1 (max_batch_size) * 729 (num_visual_features)
    ```

2. Build TensorRT engines for visual components

    ```bash
    python build_multimodal_engine.py --model_path ./${MODEL_NAME}.nemo --model_type neva --output_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu/vision
    ```

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/trt_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/bf16/1-gpu \
        --input_text "Question: which city is this? Answer:"
    ```

    Note: use `--run_profiling` for performance measurement, use `--check_accuracy` for accuracy check.

## Nougat

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="nougat-base" # also nougat-small
    git clone https://huggingface.co/facebook/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/models/core/enc_dec`

   Nougat uses mBART architecture but replaces the LLM encoder with a Swin Transformer encoder.
   To achieve this, we add an extra `--nougat` flag (over mBART example) to
   `convert_checkpoint.py` in `examples/models/core/enc_dec` and `trtllm-build`.

    ```bash
    python ../enc_dec/convert_checkpoint.py --model_type bart \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/bfloat16 \
        --tp_size 1 \
        --pp_size 1 \
        --dtype bfloat16 \
        --nougat

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/bfloat16/decoder \
        --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/llm/decoder \
        --paged_kv_cache disable \
        --moe_plugin disable \
        --gemm_plugin bfloat16 \
        --bert_attention_plugin bfloat16 \
        --gpt_attention_plugin bfloat16 \
        --remove_input_padding enable \
        --max_beam_width 1 \
        --max_batch_size 1 \
        --max_seq_len 101 \
        --max_input_len 1 \
        --max_encoder_input_len 588 # 1 (max_batch_size) * 588 (num_visual_features)
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_multimodal_engine.py --model_type nougat --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16/vision

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/1-gpu/bfloat16
    ```

    Note: Nougat models usually do not need a text prompt.


## Phi-3-vision

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="Phi-3-vision-128k-instruct" # or Phi-3.5-vision-instruct
    git clone https://huggingface.co/microsoft/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/models/core/phi`.
    ```bash
    python ../phi/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 4096 \
        --max_seq_len 4608 \
        --max_multimodal_len 4096
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.

    ```bash
    python build_multimodal_engine.py --model_type phi-3-vision --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision

    python run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --kv_cache_free_gpu_memory_fraction 0.7 \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/ \
        --image_path=https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png
    ```
## Phi-4-multimodal
Navigate to the folder `TensorRT-LLM/examples/models/core/multimodal`

1. Download Huggingface weights

    ```bash
    export MODEL_NAME="Phi-4-multimodal-instruct"
    export HF_DIR="tmp/hf_models/${MODEL_NAME}"
    export CKPT_DIR="tmp/trt_models/${MODEL_NAME}/fp16/1-gpu"
    export ENGINE_DIR="tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu"
    git clone https://huggingface.co/microsoft/${MODEL_NAME} ${HF_DIR}

    ```

2. Convert Huggingface weights into TRT-LLM checkpoints and build TRT engines using scripts in `examples/models/core/phi`.
    ```bash
    python ../phi/convert_checkpoint.py \
        --model_dir ${HF_DIR} \
        --output_dir ${CKPT_DIR} \
        --dtype float16

    trtllm-build \
        --checkpoint_dir  ${CKPT_DIR} \
        --output_dir ${ENGINE_DIR} \
        --gpt_attention_plugin float16 \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 4096 \
        --max_seq_len 4608 \
        --max_multimodal_len 4096
    ```

3. Generate TensorRT engines for visual components and combine everything into final pipeline.
*Note: the encoders are not the TRT engines but are pure Pytorch ones*

    ```bash
    python build_multimodal_engine.py --model_type phi-4-multimodal --model_path ${HF_DIR} --output_dir ${ENGINE_DIR}

    python run.py \
        --hf_model_dir ${HF_DIR} \
        --kv_cache_free_gpu_memory_fraction 0.7 \
        --engine_dir ${ENGINE_DIR} \
        --image_path=https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png
        --audio_path=${HF_DIR}/examples/what_is_shown_in_this_image.wav
    ```
## Qwen2-VL
[Qwen2-VL Family](https://github.com/QwenLM/Qwen2-VL): is the latest version of the vision language models in the Qwen model families. Here we show how to deploy Qwen2-VL 2B and 7B in TensorRT-LLM.

Firstly, please install transformers and qwen-vl-utils
```bash
pip install -r requirements-qwen2vl.txt
```
### Support data types <!-- omit from toc -->
  * FP16
  * FP8

### Build and run vision model <!-- omit from toc -->
* Download Huggingface weights
    ```bash
    export MODEL_NAME="Qwen2-VL-7B-Instruct" # or Qwen2-VL-2B-Instruct
    git clone https://huggingface.co/Qwen/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
    ```
* Build engine of decoder model

    ```bash
    python3 ../qwen/convert_checkpoint.py \
        --model_dir=tmp/hf_models/${MODEL_NAME} \
        --output_dir=tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --dtype float16

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/llm \
        --gemm_plugin=float16 \
        --gpt_attention_plugin=float16 \
        --max_batch_size=4 \
        --max_input_len=2048 \
        --max_seq_len=3072 \
        --max_multimodal_len=1296 #(max_batch_size) * 324 (num_visual_features), this's for image_shape=[504,504]
    ```

* Build engine of vision encoder model
    ```bash
    python build_multimodal_engine.py --model_type qwen2_vl --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision
    ```

* Run test on multimodal/run.py with C++ runtime (LLM part only)
    ```bash
    python3 run.py \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/
    ```
### Run Qwen2-VL decoder part by FP8 <!-- omit from toc -->
* Build engine
    ```bash
    python ./examples/quantization/quantize.py \
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --dtype float16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir tmp/trt_models/${MODEL_NAME}/fp8/1-gpu \
    --calib_size 512

    trtllm-build --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp8/1-gpu \
    --output_dir tmp/trt_engines/${MODEL_NAME}/fp8/1-gpu/llm \
    --max_input_len=2048 \
    --max_seq_len 3072 \
    --gemm_plugin auto \
    --max_batch_size 4 \
    --max_multimodal_len=1296

    # copy visiual engine directory `tmp/trt_engines/${MODEL_NAME}/fp16/1-gpu/vision/` to fp8 engine directory `tmp/trt_engines/${MODEL_NAME}/fp8/1-gpu/vision`
    ```
* Run test on multimodal/run.py with C++ runtime (LLM part only)
    ```bash
    python3 run.py \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --engine_dir tmp/trt_engines/${MODEL_NAME}/fp8/1-gpu/
    ```
## Video NeVA

[Video NeVA](https://github.com/NVIDIA/NeMo/blob/main/docs/source/multimodal/mllm/video_neva.rst) is a groundbreaking addition to the NeMo Multimodal ecosystem that could work with video modality. This model seamlessly integrates large language-centric models with a vision encoder, that can be deployed in TensorRT-LLM.

1. Generate TRT-LLM engine for Nemotron model following example in `examples/models/core/nemotron/README.md`. To adhere to the NVGPT conventions of the conversion script. This will be used as our base LM for inference.

    ```bash
    pip install decord # used for loading video

    python3 ../../../quantization/quantize.py \
        --nemo_ckpt_path /path/to/nemotron/model.nemo \
        --dtype bfloat16 \
        --batch_size 64 \
        --qformat full_prec \
        --output_dir nemotron-3/trt_ckpt/bf16/1-gpu


    trtllm-build \
        --checkpoint_dir nemotron-3/trt_ckpt/bf16/1-gpu \
        --output_dir tmp/trt_engines/nemotron-3/bf16/1-gpu/llm \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --max_batch_size 1 \
        --max_input_len 4096 \
        --max_seq_len 4352 \
        --max_multimodal_len 3072 # 1 (max_batch_size) * (12 num_frames) * (256 image_token_len)
    ```

2. Build TensorRT engines for visual components

    ```bash
    python build_multimodal_engine.py --model_path /path/to/video/neva/projector.nemo --model_type video-neva --output_dir tmp/trt_engines/nemotron-3/visual_encoder --output_dir tmp/trt_engines/nemotron-3/bf16/1-gpu/vision
    ```

    ```bash
    python run.py \
        --max_new_tokens 30 \
        --hf_model_dir nemotron-3/trt_ckpt/bf16/1-gpu \
        --engine_dir tmp/trt_engines/nemotron-3/bf16/1-gpu \
        --input_text "Question: what is in the video? Answer:" \
        --video_path /path/to/your/local/video/file
    ```

    Note: use `--run_profiling` for performance measurement, use `--check_accuracy` for accuracy check.

## Dataset Evaluation

This section explains how to evaluate datasets using our provided script, including supported models and configurations.

### Evaluation Command <!-- omit from toc -->
To run an evaluation, use the following command:

```bash
python ./examples/models/core/multimodal/eval.py \
    --model_type <model_type> \
    --engine_dir <engine_dir> \
    --hf_model_dir <hf_model_dir> \
    --dataset_dir <dataset_dir> \
    --test_trtllm (or --test_hf, or both) \
    --accuracy_threshold <threshold> \
    --eval_task <eval_task> \
    --max_ite 20 \
    --visual_engine_name <visual_engine_name>
```

### Parameters <!-- omit from toc -->
- `--model_type`: Specify the model type to evaluate.
- `--engine_dir`: Path to the model engines directory.
- `--hf_model_dir`: Path to the Hugging Face model directory.
- `--dataset_dir`: Path to the dataset directory. If not specified, will load the dataset from HF with the `--eval_task` as dataset tag.
- `--test_trtllm` or `--test_hf`: Specify which evaluation framework to use. Both can be used simultaneously.
- `--accuracy_threshold`: Set the accuracy threshold for evaluation.
- `--eval_task`: Specify the evaluation task. Supported tasks: `['lmms-lab/ai2d', 'lmms-lab/VQAv2', 'lmms-lab/MME']`. Default to `'lmms-lab/VQAv2'`.
- `--max_ite`: Maximum number of iterations, default to 20.
- `--visual_engine_name`: Name of the visual engine.

### Supported Evaluation Tasks <!-- omit from toc -->
The following evaluation tasks are supported:
- `lmms-lab/ai2d`
- `lmms-lab/VQAv2`
- `lmms-lab/MME`

### Supported Model Types <!-- omit from toc -->
The script supports the following model types:
- `blip2`
- `fuyu`
- `kosmos-2`
- `llava`
- `llava_next`
- `llava_onevision`
- `phi-3-vision`
- `qwen2_vl`
- `mllama`
- `vila`
- `cogvlm`
- `neva`
- `internvl`

**Note:** The models `vila`, `cogvlm`, `neva`, and `internvl` do not support the `--test_hf` evaluation framework.

## Enabling Tensor Parallelism for multi-GPU

The LLM part of the pipeline can be run on multiple GPUs using tensor parallelism.
The visual encoder will be replicated on each GPU and operate in a data parallel fashion.

To enable tensor parallelism, both weight conversion step (from Huggingface to FT format)
and engine building step should use additional arguments. Finally `run.py` should be prefixed
with `mpirun -n NUM_GPUS --allow-run-as-root`.

The full set of commands to enable 2-way tensor parallelism for LLaVA is:

    ```bash
    export MODEL_NAME="llava-1.5-7b-hf"

    python ../llama/convert_checkpoint.py \
        --model_dir tmp/hf_models/${MODEL_NAME} \
        --output_dir tmp/trt_models/${MODEL_NAME}/fp16/2-gpu \
        --dtype float16 --tp_size 2

    trtllm-build \
        --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/2-gpu \
        --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/2-gpu/llm \
        --gemm_plugin float16 \
        --max_batch_size 1 \
        --max_input_len 2048 \
        --max_seq_len 2560 \
        --max_multimodal_len 576

    python build_multimodal_engine.py --model_type llava --model_path tmp/hf_models/${MODEL_NAME} --output_dir tmp/trt_engines/${MODEL_NAME}/fp16/2-gpu/vision

    mpirun -n 2 --allow-run-as-root \
        python run.py \
        --max_new_tokens 30 \
        --hf_model_dir tmp/hf_models/${MODEL_NAME} \
        --engine_dir tmp/trt_engines/${MODEL_NAME}/fp16/2-gpu \
    ```
## Enabling Embedding Table Offloading

Embedding Table Offloading is a memory optimization technique that helps manage large embedding tables more efficiently. It offloads the embedding table to CPU memory and uses a chunked prefetching mechanism during processing. This approach is only available when operating in context chunk mode.

To enable this feature, use the `--mm_embedding_offloading` argument:
```bash
python run.py \
    --enable_chunked_context \
    --mm_embedding_offloading true \
    --hf_model_dir ${HF_MODEL_PATH} \
    --engine_dir ${ENGINE_PATH}
```
When not explicitly specified, this feature automatically enables if you're using a multimodal model along with context chunking enabled.
