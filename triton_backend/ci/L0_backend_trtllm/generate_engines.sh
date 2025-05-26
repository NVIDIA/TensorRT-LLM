#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

BACKEND_ROOT=${BACKEND_ROOT:='/opt/tritonserver/tensorrtllm_backend'}
BASE_DIR=${BACKEND_ROOT}/ci/L0_backend_trtllm
GPT_DIR=${BACKEND_ROOT}/tensorrt_llm/examples/models/core/gpt
TRTLLM_DIR=${BACKEND_ROOT}/tensorrt_llm/

function build_base_model {
    local NUM_GPUS=$1
    cd ${GPT_DIR}
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
    python3 convert_checkpoint.py --model_dir gpt2 --dtype float16 --tp_size ${NUM_GPUS} --output_dir ./c-model/gpt2/${NUM_GPUS}-gpu/
    cd ${BASE_DIR}
}

function build_tensorrt_engine_inflight_batcher {
    local NUM_GPUS=$1
    cd ${GPT_DIR}
    local GPT_MODEL_DIR=./c-model/gpt2/${NUM_GPUS}-gpu/
    local OUTPUT_DIR=inflight_${NUM_GPUS}_gpu/
    # ./c-model/gpt2/ must already exist (it will if build_base_model
    # has already been run)
    # max_batch_size set to 128 to avoid OOM errors
    # enable use_paged_context_fmha for KV cache reuse
    trtllm-build --checkpoint_dir "${GPT_MODEL_DIR}" \
            --gpt_attention_plugin float16 \
            --remove_input_padding enable \
            --kv_cache_type paged \
            --gemm_plugin float16 \
            --workers "${NUM_GPUS}" \
            --max_beam_width 2 \
            --output_dir "${OUTPUT_DIR}" \
            --max_batch_size 128 \
            --use_paged_context_fmha enable
    cd ${BASE_DIR}
}

# Generate the TRT_LLM model engines
NUM_GPUS_TO_TEST=("1" "2" "4")
for NUM_GPU in "${NUM_GPUS_TO_TEST[@]}"; do
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPU" ]; then
        continue
    fi

    build_base_model "${NUM_GPU}"
    build_tensorrt_engine_inflight_batcher "${NUM_GPU}"
done

# Move the TRT_LLM model engines to the CI directory
mkdir engines
mv ${GPT_DIR}/inflight_*_gpu/ engines/

# Move the tokenizer into the CI directory
mkdir tokenizer
mv ${GPT_DIR}/gpt2/* tokenizer/
