#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Runs mypy over tensorrt_llm/ (config: mypy_broad.toml) and fails
# only on the error codes listed in CODES. mypy has no "enable only these
# codes" switch, so the full output is filtered instead.
#
# Extra arguments (e.g. explicit paths) are forwarded to mypy.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CODES=(
    syntax
    import-not-found
    name-defined
    used-before-def
    call-arg
)
CODES_REGEX="\\[($(IFS='|'; echo "${CODES[*]}"))\\]$"

output="$(mypy --config-file mypy_broad.toml --no-install-types "$@" 2>&1)"
status=$?

# Exit codes 0 (clean) and 1 (type errors) are expected; anything else is a
# mypy crash or usage error that the filter below would otherwise hide.
if [[ $status -gt 1 ]]; then
    echo "$output" >&2
    exit "$status"
fi

errors="$(grep -E ": error: .*${CODES_REGEX}" <<< "$output" || true)"

# Drop errors in CuTe DSL kernel files: the cutlass DSL is resolved dynamically, so
# mypy cannot see attributes such as cutlass.Int32. A file counts as such if it lives
# under a cute_dsl_kernels directory or imports cutlass.
filtered=""
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    file="${line%%:*}"
    if [[ "$file" == *cute_dsl_kernels/* ]] \
        || grep -qE '^[[:space:]]*(import|from)[[:space:]]+cutlass' "$file" 2>/dev/null; then
        continue
    fi
    filtered+="$line"$'\n'
done <<< "$errors"
errors="${filtered%$'\n'}"

# Drop errors that are about third-party libraries: an unresolved import of one of
# THIRD_PARTY_LIBS, or any other message that names one of them (or one of their
# core types, e.g. torch's Tensor and dtype). Whether a message "is about" a library
# is decided by this word match, so it is a heuristic.
THIRD_PARTY_LIBS=(
    cuda
    cudnn
    cupti
    nccl
    nsight
    nvMatmulHeuristics
    cutlass
    triton
    triton_kernels
    flashinfer
    flash_attn
    fmha_sm100
    deep_gemm
    fla
    causal_conv1d
    fast_hadamard_transform
    mamba_ssm
    transformer_engine
    transformer_engine_torch
    torch
    torchaudio
    transformers
    timm
    modelopt
    modelexpress
    mistral_common
    llguidance
    cache_dit
    cosmos_guardrail
    PyNvVideoCodec
    av
    cv2
    decord
    imageio
    librosa
    lpips
    ray
    mooncake
    gpu_memory_service
    grpc
    zmq
    fastapi
    starlette
    uvloop
    openai
    mcp
    opentelemetry
    boto3
    pyetcd
    cloudpickle
    numpy
    msgpack
    oyaml
    requests
    parameterized
    pytest_asyncio
    pytest_mock
    pytest_split
    line_profiler
    viztracer
    distro
    docstring_parser
    cxxfilt
    llist
    mako
    model_explorer
    nemo_skills
    lm_eval
    sacrebleu
    rapidfuzz
    latex2sympy2
    word2number
    polygraphy
    xdsl
    diffusers
    etcd3
    smg_grpc_proto
    transformers_modules
)
# Triton's language module (imported as tl) and the Hugging Face config classes
# are third-party names that do not carry their library name.
THIRD_PARTY_TYPES=(Tensor dtype tl PreTrainedConfig WhisperConfig Gemma4TextConfig ParakeetEncoderConfig)
TP_REGEX="$(IFS='|'; echo "${THIRD_PARTY_LIBS[*]}|${THIRD_PARTY_TYPES[*]}")"
TP_IMPORT_REGEX='named "('"$(IFS='|'; echo "${THIRD_PARTY_LIBS[*]}")"')[".]'
filtered=""
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    msg="${line#*: error: }"
    # The compiled tensorrt_llm.bindings extension (and its stubs) is not available here.
    if [[ "$msg" == *'"tensorrt_llm.bindings'* ]]; then
        continue
    fi
    # Compiled/vendored extensions that only exist in a built tree.
    if [[ "$msg" =~ named\ \"tensorrt_llm\.(deep_gemm|flash_mla|flash_mla_cpp_tllm|deep_ep|tensorrt_llm_transfer_agent_binding)[\".] ]]; then
        continue
    fi
    if [[ "$msg" == *"[import-not-found]" ]]; then
        if [[ "$msg" =~ $TP_IMPORT_REGEX ]]; then
            continue
        fi
    elif [[ "$msg" =~ (^|[^A-Za-z0-9_])(${TP_REGEX})([^A-Za-z0-9_]|$) ]]; then
        continue
    fi
    filtered+="$line"$'\n'
done <<< "$errors"
errors="${filtered%$'\n'}"
if [[ -n "$errors" ]]; then
    echo "$errors"
    echo "Found $(wc -l <<< "$errors" | tr -d ' ') error(s) in the selected codes: ${CODES[*]}" >&2
    exit 1
fi
echo "mypy: no errors in the selected codes"
