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

import functools,torch,cutlass,cutlass.cute as cute
from ..api import DTYPES
@functools.cache
def compile_filter(qkv,k,cap,dtype,sms,signature,safe,epi=8,full_q_blocks=False,heads=64,small_head_large_kv=False,dense_candidates=False,qsa_weights=0,qsa_unit_scale=False,offsets32=False,glm_kv_parts=1,glm_mid_kv=False,dsv4=False,aligned_k=False,glm_integer_period=0,dsv32_short=False,glm_short_tile=False,glm_dual_kv=False,glm_prepare_sf=False,packed_store_enabled=False,glm_mx_tiny=False,q8192_flash_epi=False,q8192_glm_dual_kv=False,q8192_pair_queries=False, pdl=False, native_bf16=False, native_fp16=False, shared_kv_queries=False, shared_fp8_queries=False):
 block_fp8 = qkv == 'mxfp8'
 if block_fp8:qkv = 'mxfp4'
 if native_bf16 and native_fp16:raise ValueError('native modes are mutually exclusive')
 if native_fp16 and (qkv not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.Float16 or safe):raise ValueError('native_fp16 requires H4/H32/H64 FP8 or MXFP4 inputs and FP16 logits with strict sampler')
 if native_bf16 and (qkv not in ('fp8','mxfp4') or heads not in (4,32,64) or dtype!=cutlass.BFloat16 or safe):raise ValueError('native_bf16 requires H4/H32/H64 FP8 or MXFP4 inputs and BF16 logits with strict sampler')
 if shared_kv_queries and not (qkv=='mxfp4' and heads==64 and (native_bf16 or native_fp16)):
  raise ValueError('shared KV query pairs require H64 native MXFP4')
 if shared_fp8_queries:
  if not (qkv=='fp8' and heads==64 and native_fp16 and dtype==cutlass.Float16 and not safe):
   raise ValueError('shared FP8 query pairs require H64 native_fp16')
  from .fp8_shared_kv_safe_filter import compile_shared_filter
  return compile_shared_filter(k,cap,sms,signature,pdl)
 if qkv=='mxfp4':
  if block_fp8 or ((native_bf16 or native_fp16) and heads in (4,32)):from .fp4_safe_filter import FP4PrefillMQALogitsKernel as GEMM
  elif shared_kv_queries:from .fp4_shared_kv_safe_filter import FP4PrefillMQALogitsKernel as GEMM
  elif glm_mx_tiny:from .fp4_glm_fulltile_filter import FP4PrefillMQALogitsKernel as GEMM
  elif heads==32 and k==512 and glm_short_tile:from .fp4_glm_fulltile_filter import FP4PrefillMQALogitsKernel as GEMM
  else:from .fp4_safe_filter import FP4PrefillMQALogitsKernel as GEMM
  tune=dict(use_batched_store=False,num_umma_stages=1,num_kv_stages=4,max_w_in_reg=64,num_epi_subtiles=1,math_registers=112,store_registers=32)
 else:
  if q8192_pair_queries or ((native_bf16 or native_fp16) and heads in (4,32)):from .fp8_safe_filter import FP8PrefillMQALogitsKernel as GEMM
  elif heads==4 and DTYPES[signature[-2][0]]==cutlass.Int32:from .fp8_qwen_packed_filter import FP8PrefillMQALogitsKernel as GEMM
  elif heads==32 and k==2048 and dense_candidates and glm_short_tile:from .fp8_glmfull_fulltile_filter import FP8PrefillMQALogitsKernel as GEMM
  elif heads==32 and k==512 and dense_candidates and not glm_short_tile and dtype in (cutlass.Float16,cutlass.BFloat16):from .fp8_glm_mid_loop_filter import FP8PrefillMQALogitsKernel as GEMM
  elif heads==32 and k==512 and dense_candidates:from .fp8_fulltile_filter import FP8PrefillMQALogitsKernel as GEMM
  else:from .fp8_safe_filter import FP8PrefillMQALogitsKernel as GEMM
  tune=dict(umma_n=256,max_umma_pipeline=True,max_w_in_reg=64,num_epi_subtiles=2,math_registers=112,store_registers=32,weights_prepared=True,full_q_blocks=full_q_blocks)
 attrs=dict(pdl=pdl,offsets32=offsets32,qsa_weights=qsa_weights,qsa_unit_scale=qsa_unit_scale,num_heads=heads,head_dim=128,num_sms=sms,epi_dtype=cutlass.Float32,output_dtype=dtype,mode='filter',top_k=k,capacity=cap,splits=1,packed=False,sample_size=512,use_relu_trick=safe,**tune)
 if heads in (4,32):
  if qkv=='mxfp4':attrs.update(full_q_blocks=full_q_blocks,weights_prepared=True,branchless_publish=dense_candidates,num_umma_stages=1,num_q_stages=1)
  attrs.update(num_epi_subtiles=1,max_w_in_reg=heads)
  if qkv=='fp8':attrs.update(umma_n=64 if heads==4 and small_head_large_kv else 128,num_sms=sms*2)
 if heads==32 and qkv=='mxfp4':attrs.update(math_registers=104,store_registers=32,glm_mid_kv=True,num_epi_subtiles=1,num_sms=sms*2,glm_pair_skip_empty=not (glm_mid_kv or dense_candidates),glm_direct_bounds=not (glm_mid_kv or dense_candidates))
 if heads==32 and qkv=='fp8':attrs.update(glm_integer_period=glm_integer_period,umma_n=256,splits=glm_kv_parts,num_epi_subtiles=2,glm_skip_empty=not dense_candidates)
 if dsv4:
  attrs.update(dsv4_cast=True,dsv4_full_tiles=True)
  if qkv=='fp8':attrs.update(dsv4_direct_sync=True,dsv4_alias_kv=True,dsv4_aligned_k=aligned_k)
  else:attrs.update(dsv4_unroll=True,dsv4_packed_store=packed_store_enabled)
 if (dsv4 or glm_prepare_sf) and qkv=='mxfp4':attrs.update(remove_online_sf_transpose=True)
 if dsv4 and qkv=='fp8':attrs.update(dsv4_integer_period=32)
 if dsv4 and qkv=='mxfp4':attrs.update(dsv4_integer_period=64,dsv4_integer_offset=32,dsv4_finite_integer=True)
 if (dsv4 and k==1024 or heads==64 and k==2048) and qkv=='fp8':attrs.update(pro_packed_store=True)
 if heads==64 and k==2048 and dtype in (cutlass.Float16,cutlass.BFloat16):attrs.update(dsv4_cast=True)
 if heads==64 and k==2048 and qkv=='mxfp4':attrs.update(dsv4_integer_period=64,dsv4_integer_offset=32,dsv4_finite_integer=True)
 if heads==64 and k==2048 and qkv=='fp8' and dsv32_short:attrs.update(dsv4_integer_period=32)
 if heads==32 and qkv=='mxfp4':attrs.update(glm_dual_kv=glm_dual_kv)
 if heads==32 and (k==2048 or glm_dual_kv) and qkv=='mxfp4':attrs.update(math_registers=112,store_registers=32)
 if heads==32 and k==2048 and qkv=='fp8':attrs.update(num_sms=sms)
 if heads==32 and k==2048 and qkv=='fp8':attrs.update(glm_single_q_stage=dense_candidates)
 if heads==32 and qkv=='fp8' and glm_short_tile and not q8192_pair_queries:attrs.update(umma_n=128,glm_two_math_wg=True,math_registers=104,num_sms=sms*2)
 if heads==32 and qkv=='fp8':attrs.update(dsv4_aligned_k=aligned_k)
 if heads==32 and k==512 and qkv=='fp8':attrs.update(glm_single_q_stage=not dense_candidates)
 if heads==32 and k==512 and qkv=='fp8' and dense_candidates:attrs.update(glm_full_tiles=True)
 if heads==32 and k==2048 and qkv=="fp8" and dense_candidates and glm_short_tile and not q8192_pair_queries:attrs.update(glm_full_tiles=True)
 if heads==32 and k==512 and qkv=='fp8' and dense_candidates and not glm_short_tile:attrs.update(num_epi_subtiles=4,math_registers=96)
 if q8192_flash_epi:attrs.update(num_epi_subtiles=4)
 if q8192_glm_dual_kv:attrs.update(num_q_stages=2)
 if q8192_pair_queries:attrs.update(glm_pair_queries=True,q_stage_override=1,num_epi_subtiles=4,dsv4_aligned_k=True)
 if (native_bf16 or native_fp16) and heads in (4,32):
  attrs.pop('glm_full_tiles',None)
  attrs.update(weights_prepared=False,glm_pair_queries=False) if qkv=='fp8' else attrs.update(weights_prepared=False)
  if heads==4:attrs.update(num_epi_subtiles=1)
 # Native BF16 accepts NaN scores conservatively; K3/K4 retain exact recovery.
 if native_bf16:attrs.update(native_bf16=True,dsv4_unordered=True)
 if native_fp16:attrs.update(native_fp16=True,dsv4_unordered=True)
 if block_fp8:attrs.update(input_dtype=cutlass.Float8E4M3FN,dsv4_integer_period=0,dsv4_integer_offset=0,dsv4_finite_integer=False)
 weight_dtype = DTYPES[signature[4 if qkv=="mxfp4" else 3][0]]
 if weight_dtype == cutlass.BFloat16:
  if qkv=="fp8":
   from .fp8_safe_filter import FP8PrefillMQALogitsKernel as GEMM
   attrs.pop("glm_full_tiles",None)
  elif not shared_kv_queries or block_fp8:from .fp4_safe_filter import FP4PrefillMQALogitsKernel as GEMM
  attrs["weight_dtype"] = weight_dtype
 tensors=[]
 for i,(d,shape) in enumerate(signature):
  dims=tuple(cute.sym_int() if x<0 else x for x in shape);order=tuple(reversed(range(len(shape))))
  if qkv=='mxfp4' and i in(2,3,4):order=(1,0,2) if i==2 else (0,1)
  if i in(len(signature)-7,len(signature)-2):t=cute.runtime.make_fake_tensor(DTYPES[d],dims,stride=((shape[1] if heads in (4,32) and i==len(signature)-2 else cute.sym_int64()),1),assumed_align=16)
  else:t=cute.runtime.make_fake_compact_tensor(DTYPES[d],dims,stride_order=order,assumed_align=16)
  tensors.append(t)
 return cute.compile(GEMM(**attrs),*tensors,cutlass.Int32(1),cutlass.Int32(1),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi'+(' --ptxas-options=--maxrregcount=96' if qkv=='fp8' and heads!=32 and not (native_bf16 or native_fp16) else ''))
