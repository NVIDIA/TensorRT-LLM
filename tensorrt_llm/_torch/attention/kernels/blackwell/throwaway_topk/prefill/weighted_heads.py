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

"""Four FP32 FMA chains, preserving the indexer head-reduction order."""
from ..packed import fma_packed_f32x2 as indexer_fma_packed_f32x2
from ..packed import add_packed_f32x2 as indexer_add_packed_f32x2
import cutlass
import cutlass.cute as cute
from .native_heads import head_zero, head_sum, native_head_fma4, native_weight_pair
from .native_lowp_math import native_pack_relu, native_fma
from .integer_relu_fast import integer_relu4, finite_integer_relu4, mixed_integer_relu4

@cute.jit
def weighted_heads(fragment, weights, slot, stage, heads:cutlass.Constexpr,
                   relu_trick:cutlass.Constexpr, prepared:cutlass.Constexpr,native_dtype:cutlass.Constexpr=None,fp8:cutlass.Constexpr=False):
    acc=head_zero(native_dtype)
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(heads//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu_trick):
            positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:
            positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            value=cutlass.Float32(weights[(slot*heads+group*4+j,stage)])
            if cutlass.const_expr(relu_trick and not prepared):value=value*0.5
            w[j]=value
        if cutlass.const_expr(native_dtype == cutlass.BFloat16 and weights.element_type == cutlass.BFloat16):
            for pair in cutlass.range_constexpr(2):
                xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                ww=native_weight_pair(weights,slot*heads+group*4+2*pair,stage,native_dtype)
                acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
        elif cutlass.const_expr(native_dtype is not None):
            acc=native_head_fma4(x,w.load(),acc,native_dtype,fp8)
        else:
            acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return head_sum(acc,native_dtype)

@cute.jit
def weighted_heads_cached(fragment, weights, cache, slot, local_q, stage,
                          heads:cutlass.Constexpr, cached:cutlass.Constexpr,
                          relu_trick:cutlass.Constexpr, prepared:cutlass.Constexpr,native_dtype:cutlass.Constexpr=None,fp8:cutlass.Constexpr=False):
    acc=head_zero(native_dtype)
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(heads//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu_trick):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        if cutlass.const_expr(cache.element_type == cutlass.Uint32):
            for pair in cutlass.range_constexpr(2):
                xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                if cutlass.const_expr(group*4+2*pair<cached):
                    ww=cache[local_q*(cached//2)+(group*4)//2+pair]
                else:
                    ww=native_weight_pair(weights,slot*heads+group*4+2*pair,stage,native_dtype)
                acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
        else:
            w=cute.make_rmem_tensor((4,),cutlass.Float32)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(group*4+j<cached):value=cutlass.Float32(cache[local_q*cached+group*4+j])
                else:value=cutlass.Float32(weights[(slot*heads+group*4+j,stage)])
                if cutlass.const_expr(relu_trick and not prepared):value=value*0.5
                w[j]=value
            if cutlass.const_expr(native_dtype == cutlass.BFloat16 and weights.element_type == cutlass.BFloat16):
                for pair in cutlass.range_constexpr(2):
                    xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                    ww=native_weight_pair(weights,slot*heads+group*4+2*pair,stage,native_dtype)
                    acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
            elif cutlass.const_expr(native_dtype is not None):
                acc=native_head_fma4(x,w.load(),acc,native_dtype,fp8)
            else:
                acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return head_sum(acc,native_dtype)

@cute.jit
def weighted_heads_partial(fragment,weights,slot,stage,offset:cutlass.Constexpr,width:cutlass.Constexpr,heads:cutlass.Constexpr,acc,relu:cutlass.Constexpr,native_dtype:cutlass.Constexpr=None,fp8:cutlass.Constexpr=False):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        w=cute.make_rmem_tensor((4,),cutlass.Float32)
        for j in cutlass.range_constexpr(4):
            value=cutlass.Float32(weights[(slot*heads+offset+group*4+j,stage)])
            if cutlass.const_expr(relu):value=value*0.5
            w[j]=value
        if cutlass.const_expr(native_dtype == cutlass.BFloat16 and weights.element_type == cutlass.BFloat16):
            for pair in cutlass.range_constexpr(2):
                xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                ww=native_weight_pair(weights,slot*heads+offset+group*4+2*pair,stage,native_dtype)
                acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
        elif cutlass.const_expr(native_dtype is not None):
            acc=native_head_fma4(x,w.load(),acc,native_dtype,fp8)
        else:
            acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc

@cute.jit
def weighted_partial_cached(fragment,weights,cache,slot,local_q,stage,offset:cutlass.Constexpr,width:cutlass.Constexpr,cached:cutlass.Constexpr,acc,relu:cutlass.Constexpr,prepared:cutlass.Constexpr,integer_period:cutlass.Constexpr=0,integer_width:cutlass.Constexpr=4,native_dtype:cutlass.Constexpr=None,fp8:cutlass.Constexpr=False):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x=cute.make_tensor(fragment.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):
            positive=integer_relu4(x)
        elif cutlass.const_expr(relu):positive=indexer_add_packed_f32x2(x,abs(x),rnd="rn")
        else:positive=cute.max(x,zero,propagate_nan=True)
        if cutlass.const_expr(cache.element_type == cutlass.Uint32):
            for pair in cutlass.range_constexpr(2):
                xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                if cutlass.const_expr(offset+group*4+2*pair<cached):
                    ww=cache[local_q*(cached//2)+(offset+group*4)//2+pair]
                else:
                    ww=native_weight_pair(weights,slot*32+offset+group*4+2*pair,stage,native_dtype)
                acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
        else:
            w=cute.make_rmem_tensor((4,),cutlass.Float32)
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(offset+group*4+j<cached):value=cutlass.Float32(cache[local_q*cached+offset+group*4+j])
                else:value=cutlass.Float32(weights[(slot*32+offset+group*4+j,stage)])
                if cutlass.const_expr(relu and not prepared):value=value*0.5
                if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):value=value*2.0
                w[j]=value
            if cutlass.const_expr(native_dtype == cutlass.BFloat16 and weights.element_type == cutlass.BFloat16):
                for pair in cutlass.range_constexpr(2):
                    xx=native_pack_relu(native_dtype,x[2*pair],x[2*pair+1])
                    ww=native_weight_pair(weights,slot*32+offset+group*4+2*pair,stage,native_dtype)
                    acc[pair]=native_fma(native_dtype,xx,ww,acc[pair])
            elif cutlass.const_expr(native_dtype is not None):
                acc=native_head_fma4(x,w.load(),acc,native_dtype,fp8)
            else:
                acc=indexer_fma_packed_f32x2(positive,w.load(),acc,rnd="rn")
    return acc

@cute.jit
def weighted_pair_partial_cached(fragment0,fragment1,cache,local_q,
        offset:cutlass.Constexpr,width:cutlass.Constexpr,acc0,acc1,
        relu:cutlass.Constexpr,prepared:cutlass.Constexpr,integer_period:cutlass.Constexpr=0,integer_width:cutlass.Constexpr=4,native_dtype:cutlass.Constexpr=None,fp8:cutlass.Constexpr=False):
    zero=cute.full((4,),0.0,dtype=cutlass.Float32)
    for group in cutlass.range_constexpr(width//4):
        x0=cute.make_tensor(fragment0.iterator+group*4,cute.make_layout(4)).load()
        x1=cute.make_tensor(fragment1.iterator+group*4,cute.make_layout(4)).load()
        if cutlass.const_expr(relu and integer_period>0 and (offset+group*4)%integer_period==0):
            positive0=mixed_integer_relu4(x0,True) if cutlass.const_expr(integer_width==2) else finite_integer_relu4(x0)
            positive1=mixed_integer_relu4(x1,True) if cutlass.const_expr(integer_width==2) else finite_integer_relu4(x1)
        elif cutlass.const_expr(relu):
            positive0=indexer_add_packed_f32x2(x0,abs(x0),rnd="rn")
            positive1=indexer_add_packed_f32x2(x1,abs(x1),rnd="rn")
        else:
            positive0=cute.max(x0,zero,propagate_nan=True)
            positive1=cute.max(x1,zero,propagate_nan=True)
        if cutlass.const_expr(cache.element_type == cutlass.Uint32):
            for pair in cutlass.range_constexpr(2):
                xx0=native_pack_relu(native_dtype,x0[2*pair],x0[2*pair+1])
                xx1=native_pack_relu(native_dtype,x1[2*pair],x1[2*pair+1])
                ww0=cache[local_q*16+offset//2+group*2+pair]
                ww1=cache[(local_q+1)*16+offset//2+group*2+pair]
                acc0[pair]=native_fma(native_dtype,xx0,ww0,acc0[pair])
                acc1[pair]=native_fma(native_dtype,xx1,ww1,acc1[pair])
        else:
            w0=cute.make_rmem_tensor((4,),cutlass.Float32)
            w1=cute.make_rmem_tensor((4,),cutlass.Float32)
            for j in cutlass.range_constexpr(4):
                v0=cache[local_q*32+offset+group*4+j]
                v1=cache[(local_q+1)*32+offset+group*4+j]
                if cutlass.const_expr(relu and not prepared):
                    v0=v0*0.5;v1=v1*0.5
                w0[j]=v0;w1[j]=v1
            if cutlass.const_expr(native_dtype is not None):
                acc0=native_head_fma4(x0,w0.load(),acc0,native_dtype,fp8)
                acc1=native_head_fma4(x1,w1.load(),acc1,native_dtype,fp8)
            else:
                acc0=indexer_fma_packed_f32x2(positive0,w0.load(),acc0,rnd="rn")
                acc1=indexer_fma_packed_f32x2(positive1,w1.load(),acc1,rnd="rn")
    return acc0,acc1
