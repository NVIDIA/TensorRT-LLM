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

import functools,sys,torch,cutlass,cutlass.cute as cute
from cutlass import testing
from ..api import DTYPES
from ..mqa_logits_util import KernelLaunch
@functools.cache
def _compile_gemm(qkv_dtype,mode,k,cap,packed,dtype,ctas,splits,sample_size,signature,safe_relu,tuning,heads=64,glm_short_recovery=False,glm_mid_idle=False, pdl=False, native_bf16=False, native_fp16=False):
    block_fp8 = qkv_dtype == 'mxfp8'
    if block_fp8:qkv_dtype = 'mxfp4'
    if native_bf16 and native_fp16:raise ValueError('native modes are mutually exclusive')
    if native_fp16 and (qkv_dtype not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.Float16 or safe_relu):raise ValueError('native_fp16 requires H4/H32/H64 FP8 or MXFP4 inputs and FP16 logits with strict sampler')
    if native_bf16 and (qkv_dtype not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.BFloat16 or safe_relu):raise ValueError('native_bf16 requires H4/H32/H64 FP8 or MXFP4 inputs and BF16 logits with strict sampler')
    if mode != 'recover':raise ValueError('only recovery GEMM is supported')
    if qkv_dtype=='fp8':
        if glm_mid_idle and not (native_bf16 or native_fp16):from .fp8_glmflash_queue_recovery import FP8PrefillMQALogitsKernel as GEMM
        else:from .fp8_queue_recovery import FP8PrefillMQALogitsKernel as GEMM
    else:
        if glm_short_recovery and not block_fp8 and not (native_bf16 or native_fp16):from .fp4_glmflash_queue_recovery import FP4PrefillMQALogitsKernel as GEMM
        else:from .fp4_queue_recovery import FP4PrefillMQALogitsKernel as GEMM
    attrs=dict(pdl=pdl,num_heads=heads,head_dim=128,num_sms=ctas,epi_dtype=cutlass.Float32,output_dtype=dtype,
               mode=mode,top_k=k,capacity=cap,splits=splits,packed=packed,sample_size=sample_size,use_relu_trick=safe_relu)
    if qkv_dtype=='mxfp4':
        attrs.update(use_batched_store=False)
        if mode=='filter':attrs.update(max_w_in_reg=32,num_epi_subtiles=4,math_registers=192,store_registers=104)
    elif mode=='filter':attrs.update(umma_n=256,max_w_in_reg=4,num_epi_subtiles=4,math_registers=192,store_registers=104,weights_prepared=True)
    if mode in ('filter','recover'):attrs.update(dict(tuning))
    if native_bf16:attrs.update(native_bf16=True)
    if native_fp16:attrs.update(native_fp16=True)
    if block_fp8:attrs.update(input_dtype=cutlass.Float8E4M3FN,num_kv_stages=4)
    weight_dtype = DTYPES[signature[4 if qkv_dtype=="mxfp4" else 3][0]]
    if weight_dtype == cutlass.BFloat16:
        if qkv_dtype=="mxfp4":from .fp4_queue_recovery import FP4PrefillMQALogitsKernel as GEMM
        else:from .fp8_queue_recovery import FP8PrefillMQALogitsKernel as GEMM
        attrs["weight_dtype"] = weight_dtype
    kernel=GEMM(**attrs)
    # Fix contiguous head/feature dimensions for TMA and MMA, while rows remain dynamic.
    tensors=[]
    for index,(d,shape) in enumerate(signature):
        dims=tuple(cute.sym_int() if x<0 else x for x in shape)
        order=tuple(reversed(range(len(shape))))
        if qkv_dtype=='mxfp4' and index in (2,3,4):
            order=(1,0,2) if index==2 else (0,1)
        if index in (len(signature)-10, len(signature)-5):
            # Dense recovery can exceed 2^31 elements (4K Q x 1M K).
            # Keep the inherited GEMM's INT64 row-stride ABI.
            tensors.append(cute.runtime.make_fake_tensor(DTYPES[d],dims,
                           stride=(cute.sym_int64(),1),assumed_align=16))
        else:
            tensors.append(cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,
                           stride_order=order,assumed_align=16))
    return cute.compile(kernel,*tensors,cutlass.Int32(1),cutlass.Int32(1),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi')
