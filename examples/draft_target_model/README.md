# Draft-Target-Model Speculative Decoding (DTM)

This document shows how to build and run a model using DTM speculative decoding (also known as `Speculative-Sampling`, [`Paper`](https://arxiv.org/abs/2302.01318)) in TensorRT-LLM on single GPU, or single node multiple GPU.

## Overview

We provide two styles of running DTM now: using TensorRT-LLM-BLS in Triton Inference Server, or using TensorRT-LLM directly. Here we introduce the detailed steps of running DTM in both workflows.

## Support Matrix
  * GPU Compute Capability >= 8.0 (Ampere or newer)
  * FP16 / BF16 / FP8 (both draft and target model)
  * Paged KV Cache
  * Tensor Parallel

## Usage

### Build draft and target engines (the same in two workflows)

+ We use open-source `llama-7B/13B` as draft and target models in this example, assuming the paths to the models' repository are `DRAFT_MODEL_PATH` and `TARGET_MODEL_PATH`.
+ `--use_paged_context_fmha=enable` must be specified since we need KV-Cache reuse in this approach.
+ `--speculative_decoding_mode=draft_tokens_external` and `--max_draft_len` must be specified for target model.
+ `--use_paged_context_fmha=enable` are optional, but recommended for the performance.
+ `--gather_generation_logits` is necessary if using generation logits for selecting tokens in target model.
+ `--tp_size` can be modified set if using TP mode for draft / target model.
+ `--max_batch_size` more than 1 is acceptable in general usage, but we use 1 in this example.

```bash
cd examples/llama
export DRAFT_CKPT_PATH=/workspace/ckpt-draft
export TARGET_CKPT_PATH=/workspace/ckpt-target
export DRAFT_ENGINE_PATH=/workspace/engine-draft
export TARGET_ENGINE_PATH=/workspace/engine-target
export MAX_BATCH_SIZE=4
export MAX_DRAFT_LEN=10
export MAX_INPUT_LEN=3200
export MAX_SEQ_LEN=4800

python3 convert_checkpoint.py \
    --model_dir=${DRAFT_MODEL_PATH} \
    --output_dir=${DRAFT_CKPT_PATH} \
    --dtype=float16

python3 convert_checkpoint.py \
    --model_dir=${TARGET_MODEL_PATH} \
    --output_dir=${TARGET_CKPT_PATH} \
    --dtype=float16

trtllm-build \
    --checkpoint_dir=${DRAFT_CKPT_PATH} \
    --output_dir=${DRAFT_ENGINE_PATH} \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --max_batch_size=${MAX_BATCH_SIZE} \
    --max_input_len=${MAX_INPUT_LEN} \
    --max_seq_len=${MAX_SEQ_LEN}

trtllm-build \
    --checkpoint_dir=${TARGET_CKPT_PATH} \
    --output_dir=${TARGET_ENGINE_PATH} \
    --gemm_plugin=float16 \
    --use_paged_context_fmha=enable \
    --speculative_decoding_mode=draft_tokens_external \
    --max_batch_size=${MAX_BATCH_SIZE} \
    --max_draft_len=${MAX_DRAFT_LEN} \
    --max_input_len=${MAX_INPUT_LEN} \
    --max_seq_len=${MAX_SEQ_LEN}
```

### TensorRT-LLM workflow

+ `--draft_engine_dir` and `--engine_dir` must be specified for the draft and target engines respectively.
+ `--draft_target_model_config` is corresponding configuration of DTM, which has 4 hyperparameters that you need to specify to control the process of generation:
  - `draft_len`: the number of tokens the draft model generated in one iteration, which the range is from 4 to 10 in common usage. Empirically, the larger the value is, the higher acceptance ratio but higher overhead is expected at the same time, so the right balance based on the models and application scenarios needs to be found.
  - `draft_model_device_list`: the index list of device(s) to run the draft model. The length of it must be the same as the TP size of the draft model engine. For instances, `draft_model_device_list=[1]` means using tp_size=1 and GPU 1 for draft model, `draft_model_device_list=[4,5,6,7]` means using tp=4 and GPU from 4 to 7 for draft model.
  - `target_model_device_list`: the index list of device(s) to run the target model. The length of it must be the same as the TP size of the target model engine. For instances, `draft_model_device_list=[0]` means using tp_size=1 and GPU 0 for target model, `draft_model_device_list=[2,3]` means using tp=2 and GPU from 2 to 3 for target model.
  - `use_logits`: there are two methods to accept tokens proposed by draft model. When `use_logits=True`, the draft tokens are accepted based on the ratio of the logits from draft and target model (modified rejection sampling method in the original paper); When `use_logits=False`, the draft tokens are accepted based on per-token comparison with target predictions regardless of the logits.
  - As an example, `[4,[0],[1],False]` means `draft_len=4`, device of draft model is `GPU0`, device of target model is `GPU1`, and use tokens rather than logits to accept.
+ `--kv_cache_enable_block_reuse` must be specified for this approach.
+ Only CPP session is supported, so `--use_py_session` must not be specified.
+ `--kv_cache_free_gpu_memory_fraction` should be specified if we want to place two models on one GPU, or one of the models would use out of the GPU memory.
+ `--num_beams` can not be specified as larger than 1 since beam search is not supported in this approach yet.
+ `--output_generation_logits` is optional. In original paper, we accept the tokens by comparing logits of draft and target models, so this parameter is needed. But for simplification, we can accept the tokens by comparing the output token directly, in this occasion, we can skip this parameter.

```bash
python3 examples/run.py \
    --tokenizer_dir=${TARGET_MODEL_PATH} \
    --draft_engine_dir=/workspace/engine-draft \
    --engine_dir=/workspace/engine-target \
    --draft_target_model_config="[4,[0],[1],False]" \
    --max_output_len=256 \
    --kv_cache_enable_block_reuse \
    --kv_cache_free_gpu_memory_fraction=0.5 \
    --output_generation_logits \
    --input_text="How does Draft-Sampling work?"
```

### Triton Inference Server workflow

+ This example is based on TensorRT-LLM-0.18.0 and TRTLLM-backend-0.18.0 with docker image `nvcr.io/nvidia/tritonserver:25.03-trtllm-python-py3`.
+ Draft model approach is supported since TensorRT-LLM-0.7.0 (using two separate Tritonserver to maintain draft and target model respectively), but has significant optimization in TensorRT-LLM-0.10.0 (using one Tritonserver with [Business Logic Scripting](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#business-logic-scripting), BLS).

1. Get related repository inside the container

```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git checkout rel
git lfs pull
git submodule update --init --recursive
pip install -r requirements.txt
pip install SentencePiece tritonclient
```

2. Set necessary variables

+ If draft and target models can be placed in one GPU (llama-7B-FP8 + llama-30B-FP8, totally 40GiB in one H100-80GiB GPU as example), `DRAFT_GPU_DEVICE_IDS` and `TARGET_GPU_DEVICE_IDS` can be the same, (`0` as example). It appears better performance than placing on two separate GPUs.
+ Elsewise, the draft and target models can be placed in different GPUs, `DRAFT_GPU_DEVICE_IDS="0"` and `TARGET_GPU_DEVICE_IDS="1"` as example.
+ Furthermore, if TP mode is used, the device ids can be a list, `DRAFT_GPU_DEVICE_IDS="0"` and `TARGET_GPU_DEVICE_IDS="1,2,3,4"` as example.

```bash
export DRAFT_MODEL_NAME="tensorrt_llm_draft"
export TARGET_MODEL_NAME="tensorrt_llm"
export DRAFT_DEVICE_IDS="0"
export TARGET_DEVICE_IDS="1"
export TRITON_MODEL_REPO=llama_dtm
```

3. Edit model configuration

```bash
rm -rf ${TRITON_MODEL_REPO}
cp -r all_models/inflight_batcher_llm/ ${TRITON_MODEL_REPO}

python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/preprocessing/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${TARGET_MODEL_PATH},preprocessing_instance_count:1
python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/postprocessing/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},tokenizer_dir:${TARGET_MODEL_PATH},postprocessing_instance_count:1
python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False,tensorrt_llm_model_name:${TARGET_MODEL_NAME},tensorrt_llm_draft_model_name:${DRAFT_MODEL_NAME},logits_datatype:TYPE_FP32

cp -r ${TRITON_MODEL_REPO}/tensorrt_llm ${TRITON_MODEL_REPO}/tensorrt_llm_draft
sed -i 's/name: "tensorrt_llm"/name: "tensorrt_llm_draft"/g' ${TRITON_MODEL_REPO}/tensorrt_llm_draft/config.pbtxt
python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${TARGET_ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,gpu_device_ids:${TARGET_DEVICE_IDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32
python3 tools/fill_template.py -i ${TRITON_MODEL_REPO}/tensorrt_llm_draft/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${DRAFT_ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:True,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,gpu_device_ids:${DRAFT_DEVICE_IDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32
```

4. Start the triton inference server

+ `--multi-model` is necessary if TP mode is used.
+ Verbose log will be written in to file `triton_log.txt`.

```bash
python3 scripts/launch_triton_server.py \
    --model_repo=${TRITON_MODEL_REPO} \
    --multi-model \
    --world_size=1 \
    --log &
```

+ You can see the output below in the file if Triton server launches successfully:

```txt
Started HTTPService at 0.0.0.0:8000
Started GRPCInferenceService at 0.0.0.0:8001
Started Metrics Service at 0.0.0.0:8002
```

5. Send a request for inference

```bash
python3 inflight_batcher_llm/client/e2e_grpc_speculative_decoding_client.py \
    --url-target=localhost:8001 \
    --draft-tensorrt-llm-model-name=${DRAFT_MODEL_NAME} \
    --target-tensorrt-llm-model-name=${TARGET_MODEL_NAME} \
    --output-len=100 \
    --num-draft-tokens=4 \
    --end-id=2 \
    --pad-id=2 \
    --prompt "What is Ubuntu operation system?"
```

+ You can receive the following results if everything goes smoothly.

```txt
Final text:
 What is Ubuntu operation system?
Ubuntu is a free and open source operating system that runs from the desktop, to the cloud, to all your internet connected things. Ubuntu is used by millions of people around the world who want to explore new ideas and discover new opportunities.
Ubuntu is a community developed operating system that is perfect for laptops, desktops, servers, and cloud. It is used by millions of people around the world who want to explore new ideas and discover new opportunities.
```

6. Test DTM with a script
+ Prepare a JSON file `input_data.json` containing input data as below (more requests are acceptable).

```json
[
  {
      "input": "What is Ubuntu operation system?",
      "instruction": "Answer the question shortly.",
      "output": "                                                                "
  }
]
```

+ Use command below to launch requests for inference.

```bash
python3 tools/inflight_batcher_llm/speculative_decoding_test.py \
    --max-input-len 2500 \
    --dataset input_data.json \
    --url-control=localhost:8001 \
    --url-target=localhost:8001 \
    --url-draft=localhost:8001 \
    --draft-tensorrt-llm-model-name="${DRAFT_MODEL_NAME}" \
    --target-tensorrt-llm-model-name="${TARGET_MODEL_NAME}" \
    --bls-speculative-tensorrt-llm-model-name="tensorrt_llm_bls" \
    --execute-bls-speculative-decoding \
    --disable-output-comparison \
    --num-draft-tokens=4 \
    --use-draft-logits \
    --verbose
```

+ You can receive the following results if everything goes smoothly.

```txt
Ubuntu is a free and open source operating system. It is a Linux based operating system. ...
```

7. Stop triton inference server after all work is done

```bash
pkill tritonserver
```

8. Advanced usage: Fast logits D2D transfer.

+ Fast logits boosts the performance (TPS) by hiding the latency of logits transfer from draft engine to target engine supported since TensorRT-LLM-0.15.0.

+ Modify `participant_ids` entry in `tensorrt_llm/config.pbtxt` and `tensorrt_llm_draft/config.pbtxt` to suitable MPI ranks. Usually in this setting, rank 0 is reserved for the orchestrator rank; rank 1 is for draft engine; the rest of the ranks are for target engine. In this example, `particpant_ids` can be set as snippet below. Same logic also applies to TP>1 target engine.

```txt
### In tensorrt_llm_draft/config.pbtxt
parameters: {
    key: "gpu_device_ids"
    value: {
        string_value: "0"
    }
}
parameters: {
    key: "participant_ids"
    value: {
        string_value: "1"
    }
}
### In tensorrt_llm/config.pbtxt
parameters: {
    key: "gpu_device_ids"
    value: {
        string_value: "1"
    }
}
parameters: {
    key: "participant_ids"
    value: {
        string_value: "2"
    }
}
```

+ Enable `speculative_decoding_fast_logits` in both `tensorrt_llm/config.pbtxt` and `tensorrt_llm_draft/config.pbtxt`.

```txt
parameters: {
    key: "speculative_decoding_fast_logits"
    value: {
        string_value: "1"
    }
}
```

+ Launched Triton Server
  + Use orchestrator mode with `--disable-spawn-process`. See [model config](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md) for more information.
  + `--world_size` has to be set as 1 (orchestrator rank 0) + 1 (draft engine ranks) + 1 (target engine ranks).

```bash
python3 scripts/launch_triton_server.py \
    --model_repo=${TRITON_MODEL_REPO} \
    --multi-model \
    --disable-spawn-processes \
    --world_size=3 \
    --log &
```

+ Send request with `use_draft_logits` to tritonserver BLS API:
```
curl -X POST "http://localhost:8000/v2/models/tensorrt_llm_bls/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "text_input": "Continue writing the following story: James Best, best known for his",
        "max_tokens": 128,
        "num_draft_tokens": 10,
        "use_draft_logits": true,
        "stream": false
        }'
```

### Additional information

+ With the fast logits enabled and following optimization tips in [model configuration](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/model_config.md#some-tips-for-model-configuration), speculative decoding with draft logits achieves 2.x throughput in BS1, 1.x throughput in BS16 comparing to auto-regressive decoding using Llama 3.2 1B draft and Llama 3.1 70B target.
