# Speculative Sampling

Speculative Sampling (also referred to as Speculative Decoding) is a set of techniques designed to allow generation of more than one token per forward pass iteration. This can lead to a reduction in the average per-token latency **in situations where the GPU
is underutilized due to small batch sizes.**

Speculative Sampling involves predicting a sequence of future tokens, referred to as draft tokens, using a method
that is substantially more efficient than repeatedly executing the target Large Language Model (LLM).
These draft tokens are then collectively validated by processing them through the target LLM in a single forward pass.
The underlying assumptions are twofold:

1. processing multiple draft tokens concurrently will be as rapid as processing a single token
2. multiple draft tokens will be validated successfully over the course of the full generation

If the first assumption holds true, the latency of speculative decoding will no worse than the standard approach. If the second holds, output token generation advances by statistically more than one token per forward pass.
The combination of both these allows speculative decoding to result in reduced latency.

TensorRT-LLM supports several approaches for generating draft tokens, including:

1. Utilizing a smaller, auxiliary model, known as the draft model approach. For more information, refer to the [Fast Inference from Transformers via Speculative Decoding paper](https://arxiv.org/pdf/2211.17192.pdf).
2. Implementing additional language model heads that predict tokens for future positions, as detailed in the [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads paper](https://arxiv.org/abs/2401.10774).

## Performance Improvements

It's important to note that the effectiveness of speculative decoding techniques is highly dependent
on the specific task at hand. For instance, forecasting subsequent tokens in a code-completion scenario
may prove simpler than generating a summary for an article.

Furthermore, when integrating Medusa with a standard PyTorch model implementation which may not be as finely
tuned as TensorRT-LLM, the potential time savings are more pronounced.

# Draft Model Approach

The Draft model approach involves the use of two distinct models trained independently
but sharing the same vocabulary: a smaller Draft model and a larger Target model.
For example, a GPT 125M model can serve as the Draft model, while a GPT 6.7B model acts as the Target model.

The management of Draft and Target models is facilitated through two separate `GptManager` instances.
It is essential that you to coordinate the interactions between the Draft and Target models effectively.
Initially, the Draft model is queried to generate up to `K` draft tokens.
These tokens are then forwarded to the Target model for verification.
Upon verification, the Target model may return up to `K+1` tokens.
Subsequently, the prompt, now updated with the accepted tokens, is sent back to the Draft model to initiate the generation of new draft tokens.
This iterative process continues until a predefined stop conditions are met.
An example of this orchestration process can be found in the [TensorRT-LLM Triton backend](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/inflight_batcher_llm/client/e2e_grpc_speculative_decoding_client.py).

Configuring and executing the Draft model within the Inflight Fused Batching (IFB) framework
follows the same procedure as for any other model within IFB.
The `maxNewTokens` parameter should be set to the number of draft tokens in the `LlmRequest` for the Draft model query.

When building the Target model, it is necessary to specify the `--max_draft_len <K> --speculative_decoding_mode draft_tokens_external` option to the `trtllm-build` command.
During the Target model's inference phase in IFB, `maxNewTokens` should be set to `1`,
and the draft tokens must be set in the `draftTokens` field of the `LlmRequest` for the Target model query.

**NOTE:** To enhance performance, especially due to the repetitive querying of Draft and Target models with requests that share a common prefix,
it is advisable to enable KV cache reuse for both models.
This can be achieved by adding the `--use_paged_context_fmha=enable` flag to the `trtllm-build` command
and setting `enableBlockReuse=true` in the `KVCacheConfig`.

## Using Draft model approach with Triton Inference Server

+ Draft model approach is supported since TensorRT-LLM-0.7.0 (using two separate Tritonserver to maintain draft and target model respectively), but has significant optimization in TensorRT-LLM-0.10.0 (using one Tritonserver with [Business Logic Scripting](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#business-logic-scripting), BLS).
+ The source file of Draft model with BLS can be found [here](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/inflight_batcher_llm/tensorrt_llm_bls/1/lib/decode.py).
+ This example is based on TensorRT-LLM-0.10.0 and TRTLLM-backend-0.10.0, using docker image `nvcr.io/nvidia/tritonserver:24.05-trtllm-py3`.
+ Llama-7B-hf and Llama-30B-hf are used as draft and target model respectively in this example, assuming the paths to the models' repository are `DRAFT_MODEL_PATH` and `TARGET_MODEL_PATH`.
+ Maximum number of draft tokens is set to 10 in this example.

1. Prepare TensorRT engine for inference
    + Here are the commands to build draft / target engines in FP16 or FP8. All combinations of the data type (Draft-FP16/FP8 + Target-FP16/FP8) are supported.
    + `--remove_input_padding=enable --paged_kv_cache=enable` are necessary for inflight-batching.
    + `--context_fmha=enable --use_paged_context_fmha=enable` are optional, but recommended for the performance.
    + `--gather_generation_logits` is necessary if using generation logits for selecting tokens in target model.
    + `--tp_size` can be modified set if using TP mode for draft / target model.
    + `--max_batch_size` more than 1 is acceptable in general usage, but we use 1 in this example.

    ```bash
    export MAX_DRAFT_LENGTH=10
    export COMMON_COMMAND="--max_batch_size=1 --max_input_len=2048 --max_output_len=1024 --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --context_fmha=enable --use_paged_context_fmha=enable --gather_generation_logits"
    export DRAFT_COMMAND_FP16="$COMMON_COMMAND"
    export TARGET_COMMAND_FP16="$DRAFT_COMMAND_FP16 --max_draft_len=$MAX_DRAFT_LENGTH --speculative_decoding_mode draft_tokens_external"
    export DRAFT_COMMAND_FP8="$COMMON_COMMAND --strongly_typed --use_fp8_context_fmha=enable"
    export TARGET_COMMAND_FP8="$DRAFT_COMMAND_FP8 --max_draft_len=$MAX_DRAFT_LENGTH --speculative_decoding_mode draft_tokens_external"

    # Build checkpoints and engines in tensorrt_llm/examples/llama/
    # FP16 mode
    export DRAFT_NAME=llama-7b-fp16-tp1
    export TARGET_NAME=llama-30b-fp16-tp1
    python3 convert_checkpoint.py --model_dir=$DRAFT_MODEL_PATH --output_dir=ckpt/$DRAFT_NAME --tp_size=1
    python3 convert_checkpoint.py --model_dir=$TARGET_MODEL_PATH --output_dir=ckpt/$TARGET_NAME --tp_size=1
    trtllm-build --checkpoint_dir=ckpt/$DRAFT_NAME --output_dir=engine/draft/$DRAFT_NAME $DRAFT_COMMAND_FP16
    trtllm-build --checkpoint_dir=ckpt/$TARGET_NAME --output_dir=engine/target/$TARGET_NAME $TARGET_COMMAND_FP16
    export DRAFT_ENGINE_PATH=$(pwd)/engine/draft/$DRAFT_NAME
    export TARGET_ENGINE_PATH=$(pwd)/engine/target/$TARGET_NAME

    # FP8 mode
    export DRAFT_NAME=llama-7b-fp8-tp1
    export TARGET_NAME=llama-30b-fp8-tp1
    python3 convert_checkpoint.py --model_dir=$DRAFT_MODEL_PATH --output_dir=ckpt/$DRAFT_NAME --tp_size=1
    python3 convert_checkpoint.py --model_dir=$TARGET_MODEL_PATH --output_dir=ckpt/$TARGET_NAME --tp_size=1
    trtllm-build --checkpoint_dir=ckpt/$DRAFT_NAME --output_dir=engine/draft/$DRAFT_NAME $DRAFT_COMMAND_FP8
    trtllm-build --checkpoint_dir=ckpt/$TARGET_NAME --output_dir=engine/target/$TARGET_NAME $TARGET_COMMAND_FP8
    export DRAFT_ENGINE_PATH=$(pwd)/engine/draft/$DRAFT_NAME
    export TARGET_ENGINE_PATH=$(pwd)/engine/target/$TARGET_NAME
    ```

2. Edit Triton configuration
    + If both draft and target model can be placed in one GPU (for example, llama-7B-FP8 + llama-30B-FP8, totally 40GiB in one H100-80GiB GPU), `DRAFT_GPU_DEVICE_IDS` and `TARGET_GPU_DEVICE_IDS` can be the same, `0` as example. It appears better performance than placing on two separate GPUs.
    + Elsewise, the draft and target models can be placed in different GPUs, `DRAFT_GPU_DEVICE_IDS="0"` and `TARGET_GPU_DEVICE_IDS="1"` as example.
    + Furthermore, if TP mode is used, the value of `GPU_DEVICE_IDS` can be a list, `DRAFT_GPU_DEVICE_IDS="0"` and `TARGET_GPU_DEVICE_IDS="1,2,3,4"` as example.
    + For more configuration of launching models with Tritonserver, please visit [TensorRT-LLM Backed repo](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md).

    ```bash
    ACCUMULATE_TOKEN="false"
    BACKEND="tensorrtllm"
    BATCH_SCHEDULER_POLICY="guaranteed_no_evict"
    BATCHING_STRATEGY="inflight_fused_batching"
    BLS_INSTANCE_COUNT="1"
    DECODING_MODE="top_k_top_p"
    DECOUPLED_MODE="False"
    DRAFT_GPU_DEVICE_IDS="0"
    E2E_MODEL_NAME="ensemble"
    ENABLE_KV_CACHE_REUSE="true"
    ENGINE_PATH=$TARGET_ENGINE_PATH
    EXCLUDE_INPUT_IN_OUTPUT="false"
    KV_CACHE_FREE_GPU_MEM_FRACTION="0.8"
    MAX_ATTENTION_WINDOW_SIZE=""
    MAX_BEAM_WIDTH="1"
    MAX_QUEUE_DELAY_MICROSECONDS="0"
    MAX_TOKENS_IN_KV_CACHE=""
    NORMALIZE_LOG_PROBS="true"
    POSTPROCESSING_INSTANCE_COUNT="1"
    PREPROCESSING_INSTANCE_COUNT="1"
    TARGET_GPU_DEVICE_IDS="1"
    TENSORRT_LLM_DRAFT_MODEL_NAME="tensorrt_llm_draft"
    TENSORRT_LLM_MODEL_NAME="tensorrt_llm"
    TOKENIZER_PATH=$DRAFT_MODEL_PATH
    TOKENIZER_TYPE=llama
    TRITON_GRPC_PORT="8001"
    TRITON_HTTP_PORT="8000"
    TRITON_MAX_BATCH_SIZE="4"
    TRITON_METRICS_PORT="8002"
    TRITON_REPO="triton_repo"
    USE_DRAFT_LOGITS="false"

    # Make a copy of triton repo and replace the fields in the configuration files
    cd /tensorrtllm_backend/
    apt-get update && apt-get install -y build-essential cmake git-lfs
    pip3 install git-lfs tritonclient grpcio
    rm -rf ${TRITON_REPO}
    cp -R all_models/inflight_batcher_llm ${TRITON_REPO}
    python3 tools/fill_template.py -i ${TRITON_REPO}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
    python3 tools/fill_template.py -i ${TRITON_REPO}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${PREPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_PATH},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${POSTPROCESSING_INSTANCE_COUNT}
    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},accumulate_tokens:${ACCUMULATE_TOKEN},bls_instance_count:${BLS_INSTANCE_COUNT},tensorrt_llm_model_name:${TENSORRT_LLM_MODEL_NAME},tensorrt_llm_draft_model_name:${TENSORRT_LLM_DRAFT_MODEL_NAME}

    # Make a copy of tensorrt_llm as configurations of draft / target models.
    cp -R ${TRITON_REPO}/tensorrt_llm ${TRITON_REPO}/tensorrt_llm_draft
    sed -i 's/name: "tensorrt_llm"/name: "tensorrt_llm_draft"/g' ${TRITON_REPO}/tensorrt_llm_draft/config.pbtxt
    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm/config.pbtxt          triton_backend:${BACKEND},engine_dir:${ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${TARGET_GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE}
    python3 tools/fill_template.py -i ${TRITON_REPO}/tensorrt_llm_draft/config.pbtxt    triton_backend:${BACKEND},engine_dir:${DRAFT_ENGINE_PATH},decoupled_mode:${DECOUPLED_MODE},max_tokens_in_paged_kv_cache:${MAX_TOKENS_IN_KV_CACHE},max_attention_window_size:${MAX_ATTENTION_WINDOW_SIZE},batch_scheduler_policy:${BATCH_SCHEDULER_POLICY},batching_strategy:${BATCHING_STRATEGY},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},max_beam_width:${MAX_BEAM_WIDTH},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE},normalize_log_probs:${NORMALIZE_LOG_PROBS},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},gpu_device_ids:${DRAFT_GPU_DEVICE_IDS},decoding_mode:${DECODING_MODE}
    ```

3. Launch Triton server
    + `--multi-model` is necessary if TP mode is used for target model.

    ```bash
    python3 scripts/launch_triton_server.py \
        --model_repo=${TRITON_REPO} \
        --tensorrt_llm_model_name "tensorrt_llm,tensorrt_llm_draft" \
        --multi-model \
        --log &
    ```

    + Verbose log will be written in to file `triton_log.txt`. Triton server launches successfully if you see the output below in the file:

    ```txt
    Started HTTPService at 0.0.0.0:8000
    Started GRPCInferenceService at 0.0.0.0:8001
    Started Metrics Service at 0.0.0.0:8002
    ```

4. Send Requests
    + Prepare a JSON file `input_data.json` containing input data as below (more requests are acceptable).

    ```json
    [
        {
            "input": "James Best, best known for his ",
            "instruction": "Continue writing the following story:",
            "output": "                                                                "
        }
    ]
    ```

    + Use command below to launch requests for inference.
    + `--num-draft-tokens` can be modified by runtime draft lengths, 4 is used in this example.

    ```bash
    python3 tools/inflight_batcher_llm/speculative_decoding_test.py \
        --max-input-len 2048 \
        --dataset=input_data.json \
        --url-target=localhost:8001 \
        --url-draft=localhost:8001 \
        --draft-tensorrt-llm-model-name="${TENSORRT_LLM_DRAFT_MODEL_NAME}" \
        --target-tensorrt-llm-model-name="${TENSORRT_LLM_MODEL_NAME}" \
        --bls-speculative-tensorrt-llm-model-name="tensorrt_llm_bls" \
        --execute-bls-speculative-decoding \
        --disable-output-comparison \
        --num-draft-tokens=4 \
        --verbose
    ```

5. Kill Tritonserver after finishing inference

    ```bash
    pkill -9 -f trtllmExecutorWorker
    pkill -9 -f tritonserver
    ```

# Medusa

This approach leverages a single model to both generate and verify draft tokens.
It enhances the existing model by adding multiple extra language model heads, known as Medusa heads.
These additional heads are trained to predict future tokens while the base model remains unchanged.
Specifically, the first Medusa head is tasked with predicting the immediate next token,
the second head predicts the token after that, and so on.
With `K` Medusa heads, the model can forecast up to `K` tokens ahead.
The draft tokens generated by the Medusa heads during iteration `i`
are then verified and potentially accepted in the subsequent iteration, `i+1`.

The true potential of the Medusa strategy is realized when more than one token per head is used,
employing a TopK approach to create multiple potential paths, essentially forming a tree, rather than
a single linear path as seen in the Draft model approach. To reduce redundant computations, many of these paths,
which often share common prefixes, are consolidated into a single path.
This is achieved by applying attention with a sparse mask that represents the various paths. Sparse mask formed by Medusa tree is described in detail later.

By validating multiple paths simultaneously, there is an increased likelihood of accepting more than one token per iteration,
albeit at the expense of additional computational effort.

It is crucial to recognize that as the number of potential paths grows exponentially with `K`,
it is not necessary to explore or validate all of them. A recommended strategy for managing this complexity is to prune the tree
by focusing only on the paths with higher-probability tokens.

You must strike a balance between the breadth and depth of the tree you want to explore and the impact of a larger tree on the overall
performance for your specific application.

In the TensorRT-LLM implementation of Medusa, the configuration of the tree is a runtime parameter.
This flexibility allows you to experiment and identify the optimal tree structure for your use case,
which can then be utilized in a production environment.

## Medusa Tree

Consider the following diagram, which illustrates how the hidden states from the last layer of the base model
are passed to the base model's language model (LM) head and to four Medusa heads (MHs).

<p align="center">
    <img src="./media/medusa_tree.svg" alt="Example Medusa Tree" width="auto" height="auto">
</p>

In this example:

1. The token <code>l<sub>0</sub></code> represents the actual token generated by the model.
All other tokens, denoted as <code>p<sub>hk</sub></code>, are predictions from the MHs,
where `h` indicates the Medusa head index (1-based) and `k` represents the TopK choice index (0-based).
1. Four MHs are used, which means the model is predicting four future tokens.
2. The first two MHs utilize Top-2 predictions, while the last two use Top-1.
For instance, <code>p<sub>10</sub></code> and <code>p<sub>11</sub></code> are the top and
second top predictions from the first Medusa Head (MH1).
1. A total of four paths are explored, which is fewer than the 16 that would be examined
if a complete binary tree were used (assuming Top-2 predictions for all MHs).
1. As some of these paths may be accepted, there are ten potential candidates, referred to as `medusa_choices`.
The number of tokens that can be accepted at each step, including the true token,
ranges from 1 (if all Medusa predictions are incorrect) to 5 (if all are correct).

During the generation phase, the model receives an input of 10 tokens,
which corresponds to the last tokens of each candidate path, rather than just one.

In TensorRT-LLM, you have the option to define such trees by providing all the Medusa choices
or by simply specifying the unique paths.

- Since each candidate/path begins with the true token (<code>l<sub>0</sub></code>),
there is no need to specify it separately. For the predicted tokens, only the TopK indices are required.
- For example, to specify the path <code>l<sub>0</sub>p<sub>10</sub>p<sub>21</sub>p<sub>30</sub></code>,
one would use `[0,1,0]`. And
to specify the path <code>l<sub>0</sub>p<sub>11</sub>p<sub>20</sub></code>,
one would use `[1,0]`.
- To specify all 4 paths in the example, use `medusa_choices=[[0,0,0,0], [0,1,0], [1,0], [1,1]]`.
- It's also possible to specify all candidates explicitly, similar to the Medusa repository.
For instance, `medusa_choices=[[0], [0,0], [0,0,0], [0,0,0,0], [0,1],
[0,1,0], [1], [1,0], [1,1]]`. Note that when specifying all the candidates explicitly, **we don't include
the empty `[]` candidate** for the case where only the true token is accepted, that is, all the predictions from MHs are wrong.
So, only `9` candidates are specified.

**Specifying paths-only instead of all choices is currently supported only in the Python runtime.**

## Using Medusa with TensorRT-LLM

For guidance on constructing and executing Medusa with the Python runtime, consult the [Medusa README](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/medusa/README.md). When utilizing the Inflight Fused Batching (IFB) with the C++ API, it is necessary to define the `medusa_choices` explicitly within the model configuration. For detailed instructions, refer to the [model configuration in TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#modify-the-model-configuration) for more details.

### Limitations

- TensorRT-LLM supports Medusa only for Vicuna (fine tuned LLaMA).
However, similar to any new model, you can follow the same approach to define your own Medusa model and deploy with TensorRT-LLM.
- We match only tokens during the validation phasem that is `medusa_temperature=0`.
- Beam search is **not** compatible with Medusa.
