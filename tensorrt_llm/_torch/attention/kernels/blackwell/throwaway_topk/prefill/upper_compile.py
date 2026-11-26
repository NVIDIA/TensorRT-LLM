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
def compile_sampler(qkv,k,cap,dtype,sms,sigma,signature,safe,qstages,heads=64,qsa_weights=0,qsa_unit_scale=False,short_domain=False,dsv4=False,pro_common_domain=False,fixed_samples=0,glm_short_quantile=False,glm_prepare_sf=False,glm_short_margin=False,rank_sigma=0.0,count_override=0, pdl=False, exhaustive=False, native_bf16=False, native_fp16=False):
 block_fp8 = qkv == 'mxfp8'
 if block_fp8:qkv = 'mxfp4'
 if count_override>0:fixed_samples=count_override
 if native_bf16 and native_fp16:raise ValueError('native modes are mutually exclusive')
 if native_fp16 and (qkv not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.Float16 or safe):raise ValueError('native_fp16 sampler requires H4/H32/H64 FP8 or MXFP4 inputs and FP16 logits without relu trick')
 if native_bf16 and (qkv not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.BFloat16 or safe):raise ValueError('native_bf16 sampler requires H4/H32/H64 FP8 or MXFP4 inputs and BF16 logits without relu trick')
 if qkv=='fp8':
  from .sampler_upper_fp8 import FP8PrefillMQALogitsKernel as GEMM
 elif glm_short_margin and not block_fp8 and not (native_bf16 or native_fp16):from .sampler_upper_glmflash_fp4 import FP4PrefillMQALogitsKernel as GEMM
 else:from .sampler_upper_fp4 import FP4PrefillMQALogitsKernel as GEMM
 attrs=dict(pdl=pdl,fixed_samples=fixed_samples,pro_common_domain=pro_common_domain,dsv4=dsv4,qsa_weights=qsa_weights,qsa_unit_scale=qsa_unit_scale,num_heads=heads,head_dim=128,num_sms=sms,epi_dtype=cutlass.Float32,output_dtype=dtype,mode='sample',top_k=k,capacity=cap,splits=1,packed=False,sample_size=512,use_relu_trick=safe,sample_sigma=sigma)
 if qkv=='fp8':attrs.update(glm_short_quantile=glm_short_quantile)
 if qkv=='fp8':attrs.update(umma_n=256,max_umma_pipeline=False,max_w_in_reg=0,num_epi_subtiles=16,math_registers=48,store_registers=48,num_sms=sms*2)
 else:attrs.update(use_batched_store=False,num_umma_stages=1,num_kv_stages=4,num_epi_subtiles=8,max_w_in_reg=64,math_registers=112,store_registers=32)
 if heads in (4,32):
  attrs.update(num_epi_subtiles=1,max_w_in_reg=heads,math_registers=48)
  if qkv=='fp8':attrs.update(umma_n=64 if heads==4 else 256,math_registers=48)
 if heads==32:attrs.update(math_registers=96 if qkv=='fp8' else 112,store_registers=32)
 if heads==32 and qkv=='fp8':attrs.update(math_registers=48,store_registers=32,num_epi_subtiles=4,num_kv_stages=2)
 if heads==32 and qkv=='fp8' and short_domain:attrs.update(num_kv_stages=2,sample_shared=True)
 if heads==32 and qkv=='mxfp4':attrs.update(num_q_stages=3,sample_shared_capacity=1024)
 if glm_prepare_sf:attrs.update(dsv4_prepare_sf=True)
 if dsv4 and qkv=='mxfp4':attrs.update(dsv4_prepare_sf=True,dsv4_shared_scores=True,dsv4_short_top16=short_domain,num_q_stages=3)
 if dsv4:
  attrs.update(dsv4_positions=True,dsv4_bulk_positions=True)
  if qkv=='fp8':attrs.update(num_kv_stages=2)
  else:attrs.update(num_epi_subtiles=4)
 if heads==32 and k==512 and qkv=='fp8' and fixed_samples>0:attrs.update(num_kv_stages=1)
 if heads==4 and qkv=='fp8' and fixed_samples==256:attrs.update(sample_shared=True)
 if heads==4 and qkv=='mxfp4' and fixed_samples==256:attrs.update(sample_shared_capacity=256)
 if rank_sigma>0:attrs.update(sample_rank_sigma=rank_sigma)
 if count_override>0:attrs.update(sample_count_override=count_override)
 if count_override>0 and qkv=='fp8':attrs.update(num_kv_stages=2,sample_shared=False)
 # Wider TMEM reads preserve proposal positions, counts and register budget.
 if native_bf16:attrs.update(native_bf16=True)
 if native_fp16:attrs.update(native_fp16=True)
 if (native_bf16 or native_fp16) and qkv=='fp8':attrs.update(num_epi_subtiles=1 if heads==4 else 4)
 if block_fp8:attrs.update(input_dtype=cutlass.Float8E4M3FN,num_q_stages=2)
 weight_dtype = DTYPES[signature[4 if qkv=="mxfp4" else 3][0]]
 if weight_dtype == cutlass.BFloat16:
  if qkv=="mxfp4":from .sampler_upper_fp4 import FP4PrefillMQALogitsKernel as GEMM
  else:from .sampler_upper_fp8 import FP8PrefillMQALogitsKernel as GEMM
  attrs["weight_dtype"] = weight_dtype
 tensors=[]
 for i,(d,shape) in enumerate(signature):
  dims=tuple(cute.sym_int() if x<0 else x for x in shape);order=tuple(reversed(range(len(shape))))
  if qkv=='mxfp4' and i in (2,3,4):order=(1,0,2) if i==2 else (0,1)
  if i == len(signature)-8:t=cute.runtime.make_fake_tensor(DTYPES[d],dims,stride=(cute.sym_int64(),1),assumed_align=16)
  else:t=cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,stride_order=order,assumed_align=16)
  tensors.append(t)
 if exhaustive:
  from .short_domain import ShortDomainFP8, ShortDomainFP4
  GEMM = ShortDomainFP8 if qkv=='fp8' else ShortDomainFP4
  attrs = dict(heads=heads,safe=safe,fp4=qkv=='mxfp4',prepare_sf=dsv4 and qkv=='mxfp4',pdl=pdl)
 return cute.compile(GEMM(**attrs),*tensors,cutlass.Int32(1),cutlass.Int32(1),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi')
